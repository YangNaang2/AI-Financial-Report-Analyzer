"""Chronological document evaluation and real local model training.

Split dates and purge overlapping label horizons before chunking. The CPU model
is TF-IDF/logistic regression saved as JSON, never executable pickle. Optional
transformers are imported/downloaded only during an explicit training action.
"""

from collections import Counter
from datetime import date, datetime, timezone
from decimal import Decimal, InvalidOperation
import hashlib
import json
import os
from pathlib import Path
import random
import re
import shutil
from uuid import uuid4


class TrainingError(ValueError):
    """The supplied data cannot produce an honestly evaluated model."""


def _notify(progress, message):
    if progress:
        progress(message)


def _iso_date(value, label):
    try:
        if isinstance(value, datetime):
            raise ValueError
        if isinstance(value, date):
            return value.isoformat()
        normalized = str(value).replace(".", "-")
        parsed = date.fromisoformat(normalized)
        if parsed.isoformat() != normalized:
            raise ValueError
        return normalized
    except (ValueError, TypeError):
        raise TrainingError(f"{label}: 유효한 YYYY-MM-DD 날짜가 필요합니다.") from None


def _hash(value):
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _normalize_records(records):
    if not isinstance(records, (list, tuple)):
        raise TrainingError("발행일과 라벨 종료일을 포함한 문서 records 목록이 필요합니다.")
    if any(not isinstance(row, dict) for row in records):
        raise TrainingError("문서 records의 각 항목은 사전이어야 합니다.")
    normalized, seen, duplicates, unavailable = [], {}, [], []
    # Collection often arrives newest-first. Select duplicate representatives
    # deterministically from the earliest publication, regardless of input order.
    ordered = sorted(enumerate(records), key=lambda pair: (
        _iso_date(pair[1].get("report_date"), "발행일")
        if pair[1].get("status", "labeled") == "labeled" and pair[1].get("label_id") is not None else "",
        str(pair[1].get("id", ""))))
    for index, source in ordered:
        if not isinstance(source, dict):
            raise TrainingError(f"{index + 1}번째 문서 형식이 올바르지 않습니다.")
        if source.get("status", "labeled") != "labeled" or source.get("label_id") is None:
            unavailable.append(str(source.get("id", index)))
            continue
        text = source.get("text")
        if not isinstance(text, str) or len(text.strip()) < 10:
            raise TrainingError(f"{index + 1}번째 문서의 분석 가능한 텍스트가 부족합니다.")
        label = source.get("label_id")
        if type(label) is not int or label not in (0, 1):
            raise TrainingError("라벨은 0(기준 하락 미발생) 또는 1(기준 하락 발생)이어야 합니다.")
        published = _iso_date(source.get("report_date"), "발행일")
        end = _iso_date(source.get("label_end_date") or source.get("end_date"), "라벨 종료일")
        if end < published:
            raise TrainingError("라벨 종료일은 발행일보다 빠를 수 없습니다.")
        if end > date.today().isoformat() or published > date.today().isoformat():
            raise TrainingError("미래 발행일 또는 미래 종료일의 라벨은 관측 완료 자료로 학습할 수 없습니다.")
        content_hash = _hash(re.sub(r"\s+", " ", text).strip())
        identifier = str(source.get("id") or content_hash)
        keys = ["text:" + content_hash, "id:" + identifier]
        if source.get("content_hash"):
            keys.append("content:" + str(source["content_hash"]))
        if source.get("group_id"):
            keys.append("group:" + str(source["group_id"]))
        previous = next((seen[key] for key in keys if key in seen), None)
        if previous is not None:
            if any(seen[key] is not previous for key in keys if key in seen):
                raise TrainingError("서로 다른 문서 그룹의 중복 식별자가 겹칩니다. 원자료를 정리해 주세요.")
            if previous["label_id"] != label:
                raise TrainingError("같은 문서/그룹에 서로 다른 라벨이 있어 학습을 중단했습니다.")
            for key in keys:
                seen[key] = previous
            duplicates.append(identifier)
            continue
        record = dict(source, id=identifier, text=text.strip(), content_hash=content_hash,
                      label_id=int(label), report_date=published, label_end_date=end)
        for key in keys:
            seen[key] = record
        normalized.append(record)
    normalized.sort(key=lambda row: (row["report_date"], row["id"]))
    return normalized, duplicates, unavailable


def temporal_split(records):
    """Return train/validation/test document lists plus a purge/deduplication audit."""
    rows, duplicates, unavailable = _normalize_records(records)
    days = sorted({row["report_date"] for row in rows})
    if len(rows) < 6 or len(days) < 3:
        raise TrainingError("시간 순 검증에는 최소 6개 고유 문서와 3개 발행일이 필요합니다. 더 많은 관측 완료 자료를 준비하세요.")
    train_count = max(1, min(len(days) - 2, int(len(days) * 0.6)))
    val_count = max(1, min(len(days) - train_count - 1, int(len(days) * 0.2)))
    val_start, test_start = days[train_count], days[train_count + val_count]
    groups = {"train": [], "validation": [], "test": []}
    purged = {"train": [], "validation": []}
    for row in rows:
        split = "train" if row["report_date"] < val_start else "validation" if row["report_date"] < test_start else "test"
        boundary = val_start if split == "train" else test_start
        if split != "test" and row["label_end_date"] >= boundary:
            purged[split].append(row["id"])
        else:
            groups[split].append(row)
    if any(not group for group in groups.values()):
        raise TrainingError("라벨 기간 중복을 제거한 뒤 학습·검증·시험 중 빈 구간이 있습니다. 날짜 간격이 더 넓은 자료가 필요합니다.")
    if len({row["label_id"] for row in groups["train"]}) != 2:
        raise TrainingError("학습 구간에 두 종류의 라벨이 모두 있어야 합니다. 미래 구간을 섞어 보충하지 않습니다.")
    audit = {"validation_start": val_start, "test_start": test_start,
             "purged_ids": purged, "duplicate_ids": duplicates, "unavailable_ids": unavailable,
             "policy": "chronological_unique_dates_60_20_20; purge label_end >= next split start; split before chunking"}
    return groups, audit


def _label_policy(groups):
    """Preserve label definitions independently of the classifier decision cutoff.

    A dated external 0/1 label can be trained, but its meaning is unknown without
    the full declared return policy. Never silently infer a 30-day/-5% default.
    """
    fields = ("window_days", "window_kind", "window_anchor", "entry_policy", "threshold")
    definitions, unknown, provenance = {}, [], []
    sources, adjustments = set(), set()
    for split, rows in groups.items():
        for row in rows:
            supplied = row.get("label_info")
            info = supplied if isinstance(supplied, dict) else {}
            identifier = row["id"]
            if info.get("label_id") is not None and info["label_id"] != row["label_id"]:
                raise TrainingError(f"문서 {identifier}의 라벨과 관측 라벨 정보가 다릅니다.")
            if info.get("end_date") and _iso_date(info["end_date"], "관측 종료일") != row["label_end_date"]:
                raise TrainingError(f"문서 {identifier}의 라벨 종료일과 관측 종료일이 다릅니다.")
            known = all(info.get(field) is not None for field in fields)
            definition = None
            if known:
                window = info["window_days"]
                if type(window) is not int or not 1 <= window <= 3650:
                    raise TrainingError("라벨 정책의 window_days는 1~3650 정수여야 합니다.")
                for field in ("window_kind", "window_anchor", "entry_policy"):
                    if not isinstance(info[field], str) or not info[field].strip():
                        raise TrainingError(f"라벨 정책의 {field} 값이 올바르지 않습니다.")
                try:
                    if isinstance(info["threshold"], bool):
                        raise InvalidOperation
                    return_threshold = Decimal(str(info["threshold"]))
                    if not return_threshold.is_finite() or not -100 <= return_threshold <= 100:
                        raise InvalidOperation
                except (InvalidOperation, ValueError):
                    raise TrainingError("라벨 정책의 수익률 기준은 -100~100 사이의 유한한 수여야 합니다.") from None
                definition = {"window_days": window, "window_kind": info["window_kind"].strip(),
                              "window_anchor": info["window_anchor"].strip(), "entry_policy": info["entry_policy"].strip(),
                              "return_threshold_pct": float(return_threshold)}
                key = json.dumps(definition, sort_keys=True)
                definitions[key] = definition
                if len(definitions) > 1:
                    raise TrainingError("서로 다른 수익률 라벨 정책(기간·기준일·진입 정책·하락 기준)을 한 모델에 섞을 수 없습니다.")
            else:
                unknown.append(identifier)
            source = info.get("source") if isinstance(info.get("source"), str) and info["source"].strip() else "unknown"
            adjustment = (info.get("adjustment_policy") if isinstance(info.get("adjustment_policy"), str)
                          and info["adjustment_policy"].strip() else "unknown")
            sources.add(source)
            adjustments.add(adjustment)
            record = {"record_id": identifier, "split": split,
                      "policy_status": "known" if known else "unknown",
                      "source": source, "adjustment_policy": adjustment,
                      "report_date": row["report_date"], "label_end_date": row["label_end_date"],
                      "label_id": row["label_id"], "definition": definition,
                      "provided_definition": {field: info[field] for field in fields if field in info}}
            if isinstance(row.get("source"), dict):
                record["document_source"] = row["source"]
            if row.get("ticker") is not None:
                record["ticker"] = row["ticker"]
            for field in ("as_of", "start_date", "end_date", "base_price", "future_price", "return_pct",
                          "adjustment_note", "policy_note"):
                if field in info:
                    record[field] = info[field]
            provenance.append(record)
    status = "unknown" if not definitions else "incomplete" if unknown else "complete"
    result = {"status": status, "definition": next(iter(definitions.values()), None),
              "scope": "retained_train_validation_test_documents",
              "comparison": "return_pct <= return_threshold_pct gives label_id 1",
              "known_record_count": len(provenance) - len(unknown),
              "unknown_record_count": len(unknown), "unknown_record_ids": unknown,
              "sources": sorted(sources), "adjustment_policies": sorted(adjustments),
              "provenance": provenance}
    try:
        json.dumps(result, ensure_ascii=False, allow_nan=False)
    except (ValueError, TypeError) as exc:
        raise TrainingError("라벨 출처 정보는 유한한 숫자와 JSON으로 저장 가능한 값이어야 합니다.") from exc
    return result


def _metrics(labels, probabilities, threshold):
    from sklearn.metrics import accuracy_score, average_precision_score, confusion_matrix
    predicted = [int(probability >= threshold) for probability in probabilities]
    tn, fp, fn, tp = confusion_matrix(labels, predicted, labels=[0, 1]).ravel()
    precision = float(tp / (tp + fp)) if tp + fp else None
    recall = float(tp / (tp + fn)) if tp + fn else None
    f1 = float(2 * tp / (2 * tp + fp + fn)) if 2 * tp + fp + fn else None
    return {"count": len(labels), "accuracy": float(accuracy_score(labels, predicted)),
            "precision": precision, "recall": recall, "f1": f1,
            "pr_auc": float(average_precision_score(labels, probabilities)) if len(set(labels)) == 2 else None,
            "confusion_matrix": [[int(tn), int(fp)], [int(fn), int(tp)]],
            "positive_count": int(sum(labels)), "threshold": float(threshold),
            "pr_auc_note": "average precision; N/A when only one actual class is present"}


def _select_threshold(labels, probabilities):
    if len(set(labels)) < 2:
        return 0.5, "검증 구간이 단일 라벨이므로 기본값 0.5를 유지했습니다."
    candidates = sorted(set([0.5] + [float(value) for value in probabilities]))
    threshold = max(candidates, key=lambda value: ((_metrics(labels, probabilities, value)["f1"] or 0),
                                                   -abs(value - 0.5), value))
    return threshold, "검증 문서 F1을 최대화하고 동점이면 0.5에 가까운 값을 선택했습니다. 시험 구간은 사용하지 않았습니다."


def _chunk_groups(groups, max_tokens, overlap, tokenizer=None):
    from data_processor import chunk_text
    chunks = {}
    for split, documents in groups.items():
        chunks[split] = []
        for row in documents:
            result = chunk_text(row["text"], tokenizer=tokenizer, max_tokens=max_tokens,
                                overlap=overlap, max_segments=500)
            segments = result.get("segments", [])
            if not segments or result.get("status") in ("error", "empty"):
                raise TrainingError(f"문서 {row['id']}에서 학습 가능한 문장을 추출하지 못했습니다.")
            if result.get("unprocessed_pages") or result.get("analyzed_chars", 0) < result.get("total_chars", 0):
                raise TrainingError(f"문서 {row['id']}가 분할 상한으로 잘렸습니다. 전체 문서를 포함하도록 입력을 줄이거나 분리하세요.")
            for segment in segments:
                chunks[split].append({"document_id": row["id"], "text": segment["text"],
                                      "label_id": row["label_id"], "weight": 1 / len(segments)})
    return chunks


def _document_probabilities(documents, chunks, probabilities):
    accum = {row["id"]: [] for row in documents}
    for chunk, probability in zip(chunks, probabilities):
        accum[chunk["document_id"]].append(float(probability))
    return [sum(accum[row["id"]]) / len(accum[row["id"]]) for row in documents]


def _train_baseline(groups, stage, max_tokens, overlap, seed):
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    chunks = _chunk_groups(groups, max_tokens, overlap)
    train = chunks["train"]
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True,
                               max_features=50_000, token_pattern=r"(?u)\b\w\w+\b")
    try:
        matrix = vectorizer.fit_transform([row["text"] for row in train])
    except ValueError as exc:
        raise TrainingError(f"학습 어휘를 만들 수 없습니다: {exc}") from exc
    counts = Counter(row["label_id"] for row in groups["train"])
    weights = {label: len(groups["train"]) / (2 * count) for label, count in counts.items()}
    model = LogisticRegression(max_iter=1000, random_state=seed, class_weight=weights)
    model.fit(matrix, [row["label_id"] for row in train], sample_weight=[row["weight"] for row in train])
    saved = {"vocabulary": {key: int(value) for key, value in vectorizer.vocabulary_.items()},
             "idf": vectorizer.idf_.tolist(), "coef": model.coef_[0].tolist(),
             "intercept": float(model.intercept_[0]), "lowercase": True,
             "token_pattern": r"(?u)\b\w\w+\b", "ngram_range": [1, 2],
             "sublinear_tf": True, "smooth_idf": True, "norm": "l2"}
    (stage / "baseline_model.json").write_text(json.dumps(saved, ensure_ascii=False, allow_nan=False), encoding="utf-8")

    def predict(split):
        encoded = vectorizer.transform([row["text"] for row in chunks[split]])
        probabilities = model.predict_proba(encoded)[:, list(model.classes_).index(1)]
        return _document_probabilities(groups[split], chunks[split], probabilities)
    return predict, chunks, None


def _train_transformer(groups, stage, max_tokens, overlap, seed, epochs, batch_size, base_model, progress):
    try:
        import numpy as np
        import torch
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
    except ImportError as exc:
        raise TrainingError("Transformer 학습에는 requirements-model.txt의 선택 의존성이 필요합니다.") from exc
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=False)
    model = AutoModelForSequenceClassification.from_pretrained(
        base_model, num_labels=2, ignore_mismatched_sizes=True, trust_remote_code=False,
        id2label={0: "non_negative", 1: "negative"}, label2id={"non_negative": 0, "negative": 1})
    chunks = _chunk_groups(groups, max_tokens, overlap, tokenizer)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
    counts = Counter(row["label_id"] for row in groups["train"])
    weights = torch.tensor([len(groups["train"]) / (2 * counts[label]) for label in (0, 1)], device=device)
    rng = random.Random(seed)
    for epoch in range(epochs):
        model.train()
        order = list(range(len(chunks["train"])))
        rng.shuffle(order)
        for start in range(0, len(order), batch_size):
            rows = [chunks["train"][index] for index in order[start:start + batch_size]]
            batch = tokenizer([row["text"] for row in rows], padding=True, truncation=True,
                              max_length=max_tokens, return_tensors="pt").to(device)
            labels = torch.tensor([row["label_id"] for row in rows], device=device)
            sample_weights = torch.tensor([row["weight"] for row in rows], device=device)
            optimizer.zero_grad()
            losses = torch.nn.functional.cross_entropy(model(**batch).logits, labels, weight=weights, reduction="none")
            (losses * sample_weights).mean().backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        _notify(progress, f"Transformer 학습 {epoch + 1}/{epochs}회 완료")
    model.save_pretrained(stage, safe_serialization=True)
    tokenizer.save_pretrained(stage)

    def predict(split):
        model.eval()
        scores = []
        with torch.no_grad():
            for start in range(0, len(chunks[split]), batch_size):
                rows = chunks[split][start:start + batch_size]
                batch = tokenizer([row["text"] for row in rows], padding=True, truncation=True,
                                  max_length=max_tokens, return_tensors="pt").to(device)
                scores.extend(torch.softmax(model(**batch).logits, dim=-1)[:, 1].cpu().tolist())
        return _document_probabilities(groups[split], chunks[split], scores)
    return predict, chunks, base_model


def train_model(texts=None, labels=None, *, records=None, output_dir="./models", epochs=3,
                batch_size=8, max_tokens=256, overlap=32, seed=42, backend="baseline",
                progress=None, base_model="kwoncho/KoFinBERT"):
    """Train/evaluate and atomically publish a completed manifest.

    Old positional texts/labels are recognized, but have no temporal provenance.
    Dated records are required; two examples cannot honestly validate a model.
    """
    if records is None:
        raise TrainingError("texts/labels만으로는 시간 누출을 검증할 수 없습니다. report_date와 label_end_date를 가진 records를 전달하세요.")
    if backend not in ("baseline", "transformer"):
        raise TrainingError("학습 방식은 baseline 또는 transformer여야 합니다.")
    if isinstance(max_tokens, bool) or not isinstance(max_tokens, int) or not 16 <= max_tokens <= 512:
        raise TrainingError("토큰 길이는 16~512 정수여야 합니다.")
    for value, label, low, high in ((epochs, "학습 횟수", 1, 100), (batch_size, "배치 크기", 1, 256),
                                    (max_tokens, "토큰 길이", 16, 512), (overlap, "중첩 길이", 0, max_tokens - 1)):
        if isinstance(value, bool) or not isinstance(value, int) or not low <= value <= high:
            raise TrainingError(f"{label}은 {low}~{high} 정수여야 합니다.")
    if not isinstance(seed, int) or isinstance(seed, bool) or not 0 <= seed <= 2**32 - 1:
        raise TrainingError("시드는 0~4294967295 정수여야 합니다.")
    groups, audit = temporal_split(records)
    label_policy = _label_policy(groups)
    _notify(progress, "문서 단위 시간 분할 및 라벨 기간 중복 제거 완료")
    model_id = f"{backend}_{datetime.now(timezone.utc):%Y%m%dT%H%M%S}_{uuid4().hex[:8]}"
    parent = Path(output_dir).expanduser().resolve()
    parent.mkdir(parents=True, exist_ok=True)
    stage, final = parent / f".{model_id}.tmp", parent / model_id
    stage.mkdir()
    try:
        if backend == "baseline":
            predict, chunks, source_model = _train_baseline(groups, stage, max_tokens, overlap, seed)
        else:
            predict, chunks, source_model = _train_transformer(groups, stage, max_tokens, overlap, seed,
                                                              epochs, batch_size, base_model, progress)
        val_labels = [row["label_id"] for row in groups["validation"]]
        val_probabilities = predict("validation")
        threshold, threshold_note = _select_threshold(val_labels, val_probabilities)
        # Final held-out predictions are never used for threshold selection.
        test_labels = [row["label_id"] for row in groups["test"]]
        test_probabilities = predict("test")
        train_counts = Counter(row["label_id"] for row in groups["train"])
        majority = int(train_counts[1] > train_counts[0])
        metrics = {"validation": _metrics(val_labels, val_probabilities, threshold),
                   "test": _metrics(test_labels, test_probabilities, threshold),
                   "majority_baseline": {"label_id": majority,
                       "validation": _metrics(val_labels, [float(majority)] * len(val_labels), 0.5),
                       "test": _metrics(test_labels, [float(majority)] * len(test_labels), 0.5)}}
        splits = {}
        for split, rows in groups.items():
            splits[split] = {"ids": [row["id"] for row in rows], "count": len(rows),
                             "date_start": min(row["report_date"] for row in rows),
                             "date_end": max(row["report_date"] for row in rows),
                             "label_end_max": max(row["label_end_date"] for row in rows),
                             "chunk_count": len(chunks[split]),
                             "class_counts": {str(label): sum(row["label_id"] == label for row in rows) for label in (0, 1)}}
        provenance = [{"id": row["id"], "hash": row["content_hash"], "report_date": row["report_date"],
                       "label_end_date": row["label_end_date"], "label_id": row["label_id"], "split": split}
                      for split, rows in groups.items() for row in rows]
        files = {str(path.relative_to(stage)): hashlib.sha256(path.read_bytes()).hexdigest()
                 for path in sorted(stage.rglob("*")) if path.is_file()}
        evaluation_reasons = []
        if label_policy["status"] != "complete":
            evaluation_reasons.append("일부 또는 전체 자료의 수익률 라벨 정의가 확인되지 않았습니다.")
        for split in ("validation", "test"):
            if len(groups[split]) < 30:
                evaluation_reasons.append(f"{split}: 문서가 30개 미만이라 평가 표본이 부족합니다.")
            if len({row["label_id"] for row in groups[split]}) < 2:
                evaluation_reasons.append(f"{split}: 한 종류의 라벨만 있어 두 분류의 성능을 평가할 수 없습니다.")
        manifest = {
            "schema_version": 1, "status": "complete", "completed": True,
            "model_type": backend, "model_id": model_id, "created_at": datetime.now(timezone.utc).isoformat(),
            "model_dir": str(final), "base_model": source_model,
            "negative_id": 1, "id2label": {"0": "non_negative", "1": "negative"},
            "label_mapping": {"0": "non_negative", "1": "negative"},
            "label_meaning": "관측 수익률이 사용자가 정한 하락 기준 이하인지 여부; 투자 의견/고의성 판단 아님",
            "label_policy": label_policy,
            "classifier_threshold": threshold,
            "threshold": threshold, "threshold_selection": threshold_note,
            "score_aggregation": "mean_chunk_probability", "score_calibration": "not_calibrated",
            "max_tokens": max_tokens, "overlap": overlap, "preprocessing_version": 2,
            "seed": seed, "metrics": metrics, "splits": splits, "split_audit": audit,
            "evaluation_status": "insufficient" if evaluation_reasons else "evaluated",
            "evaluation_reasons": evaluation_reasons,
            "class_counts": {str(key): value for key, value in train_counts.items()},
            "training": {"backend": backend, "epochs": epochs if backend == "transformer" else None,
                         "batch_size": batch_size, "class_weight": "training_documents_balanced",
                         "document_weight": "1 / document_chunk_count", "resampling": "none"},
            "files": files, "fingerprint": _hash(json.dumps({"documents": provenance, "label_policy": label_policy},
                                                              sort_keys=True, ensure_ascii=False)),
            "limitations": ["문서 단위 보류 시험 평가이며 실제 투자 수익을 보장하지 않습니다.",
                            "수익률 라벨과 가격 조정 정책은 학습 자료에 의존합니다.",
                            "확률은 별도로 교정되지 않은 분류 점수입니다."],
        }
        (stage / "model_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2,
                                                              allow_nan=False) + "\n", encoding="utf-8")
        os.replace(stage, final)
        _notify(progress, f"학습 및 보류 시험 평가 완료: {model_id}")
        return manifest
    except Exception:
        shutil.rmtree(stage, ignore_errors=True)
        raise


if __name__ == "__main__":
    raise SystemExit("python main.py train --records DATASET.json --backend baseline 명령을 사용하세요.")
