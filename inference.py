"""Offline report analysis with explicit model provenance and a score-free preview."""
from collections import Counter, OrderedDict
from copy import deepcopy
from datetime import datetime, timezone
from functools import lru_cache
from hashlib import sha256
import json
import math
from pathlib import Path
import re
import threading

from data_processor import EXTRACTION_VERSION, DocumentError, chunk_document, document_from_text


class InferenceError(ValueError):
    """An analysis/model request cannot be completed without guessing."""


_RULES = (
    ("추정치 조정", "하향"), ("성장 둔화", "둔화"), ("실적 표현", "부진"),
    ("감소 표현", "감소"), ("악화 표현", "악화"), ("수요 표현", "위축"),
    ("적자 표현", "적자"), ("일정 표현", "지연"), ("불확실성 표현", "우려"),
    ("불확실성 표현", "리스크"),
)
_MODEL_CACHE = OrderedDict()
_MODEL_LOCK = threading.RLock()
_PREDICTION_LOCK = threading.RLock()


@lru_cache(maxsize=256)
def _file_hash_revision(path, size, modified_ns):
    digest = sha256()
    with open(path, "rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _file_hash(path):
    stat = path.stat()
    return _file_hash_revision(str(path.resolve()), stat.st_size, stat.st_mtime_ns)


def _json_file(path, limit=32 * 1024 * 1024):
    try:
        if path.stat().st_size > limit:
            raise InferenceError(f"모델 설정 파일이 너무 큽니다: {path.name}")
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise InferenceError(f"모델 파일을 읽을 수 없습니다: {path.name}") from exc


def _created_at(value):
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        return parsed.replace(tzinfo=timezone.utc) if parsed.tzinfo is None else parsed
    except (ValueError, TypeError, AttributeError) as exc:
        raise InferenceError("모델 완료 시각(created_at)이 올바르지 않습니다.") from exc


def _baseline_payload(path):
    payload = _json_file(path)
    if not isinstance(payload, dict):
        raise InferenceError("기준 모델 형식이 올바르지 않습니다.")
    vocabulary, idf, coef = (payload.get(key) for key in ("vocabulary", "idf", "coef"))
    if not isinstance(vocabulary, dict) or not vocabulary or not isinstance(idf, list) or not isinstance(coef, list):
        raise InferenceError("기준 모델의 어휘 또는 계수가 없습니다.")
    size = len(vocabulary)
    if len(idf) != size or len(coef) != size or any(not isinstance(word, str) for word in vocabulary):
        raise InferenceError("기준 모델의 어휘와 계수 길이가 다릅니다.")
    if any(isinstance(index, bool) or not isinstance(index, int) for index in vocabulary.values()) or set(vocabulary.values()) != set(range(size)):
        raise InferenceError("기준 모델 어휘 인덱스가 올바르지 않습니다.")
    values = idf + coef + [payload.get("intercept")]
    if any(isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) for value in values):
        raise InferenceError("기준 모델에 유효하지 않은 숫자가 있습니다.")
    if any(value <= 0 for value in idf):
        raise InferenceError("기준 모델의 IDF 값이 올바르지 않습니다.")
    if (payload.get("token_pattern") != r"(?u)\b\w\w+\b" or payload.get("ngram_range") != [1, 2]
            or payload.get("norm") != "l2" or payload.get("sublinear_tf") is not True
            or payload.get("lowercase") is not True):
        raise InferenceError("지원하지 않는 기준 모델 전처리 설정입니다.")
    return payload


def _inspect_model(path):
    path = Path(path).resolve()
    manifest_path = path / "model_manifest.json"
    manifest = _json_file(manifest_path, limit=2 * 1024 * 1024)
    if not isinstance(manifest, dict) or manifest.get("schema_version") != 1:
        raise InferenceError("지원하는 모델 manifest가 없습니다.")
    if manifest.get("status") != "complete" or manifest.get("completed") is not True:
        raise InferenceError("학습 완료가 확인되지 않은 모델입니다.")
    if str(manifest.get("preprocessing_version")) != EXTRACTION_VERSION:
        raise InferenceError("모델과 현재 문서 전처리 버전이 다릅니다.")
    labels = manifest.get("id2label") or manifest.get("label_mapping")
    negative_id = manifest.get("negative_id")
    if (not isinstance(labels, dict) or set(labels) != {"0", "1"}
            or isinstance(negative_id, bool) or negative_id not in (0, 1)
            or labels.get(str(negative_id)) != "negative"
            or labels.get(str(1 - negative_id)) != "non_negative"):
        raise InferenceError("명시적인 이진 라벨 매핑과 negative_id가 필요합니다.")
    threshold = manifest.get("threshold", 0.5)
    if isinstance(threshold, bool) or not isinstance(threshold, (float, int)) or not math.isfinite(threshold) or not 0 <= threshold <= 1:
        raise InferenceError("모델 분류 기준값이 올바르지 않습니다.")
    max_tokens, overlap = manifest.get("max_tokens", 256), manifest.get("overlap", 32)
    reserved = 2 if manifest.get("model_type") == "transformer" else 0
    if (isinstance(max_tokens, bool) or not isinstance(max_tokens, int) or not reserved < max_tokens <= 8192
            or isinstance(overlap, bool) or not isinstance(overlap, int) or not 0 <= overlap < max_tokens - reserved):
        raise InferenceError("모델 구간 크기와 중첩 설정이 올바르지 않습니다.")
    _created_at(manifest.get("created_at"))
    files = manifest.get("files")
    if not isinstance(files, dict) or not files:
        raise InferenceError("모델 파일 체크섬 목록이 없습니다.")
    kind = manifest.get("model_type")
    if kind == "baseline":
        required = {"baseline_model.json"}
    elif kind == "transformer":
        required = {"config.json", "tokenizer_config.json"}
        if "tokenizer.json" not in files and "vocab.txt" not in files:
            raise InferenceError("로컬 토크나이저 파일이 없습니다.")
        if "model.safetensors" in files:
            required.add("model.safetensors")
        elif "model.safetensors.index.json" in files:
            required.add("model.safetensors.index.json")
            index_path = (path / "model.safetensors.index.json").resolve()
            if not index_path.is_relative_to(path):
                raise InferenceError("모델 가중치 인덱스 경로가 올바르지 않습니다.")
            index = _json_file(index_path)
            if (not isinstance(index, dict) or not isinstance(index.get("weight_map"), dict)
                    or not index["weight_map"] or any(not isinstance(value, str) for value in index["weight_map"].values())):
                raise InferenceError("모델 가중치 인덱스 형식이 올바르지 않습니다.")
            required.update(index["weight_map"].values())
        else:
            raise InferenceError("완료된 safetensors 모델 가중치가 없습니다.")
    else:
        raise InferenceError("지원하지 않는 모델 종류입니다.")
    if not required.issubset(files):
        raise InferenceError("필수 모델 파일이 체크섬 목록에서 빠졌습니다.")
    hashes = []
    for relative, expected in sorted(files.items()):
        if not isinstance(relative, str) or not isinstance(expected, str) or not re.fullmatch(r"[0-9a-f]{64}", expected):
            raise InferenceError("모델 파일 체크섬 형식이 올바르지 않습니다.")
        target = (path / relative).resolve()
        if not target.is_relative_to(path) or not target.is_file():
            raise InferenceError(f"모델 파일이 없거나 경로가 잘못되었습니다: {relative}")
        actual = _file_hash(target)
        if actual != expected:
            raise InferenceError(f"모델 파일 체크섬이 다릅니다: {relative}")
        hashes.append((relative, actual))
    if kind == "baseline":
        _baseline_payload(path / "baseline_model.json")
    else:
        config = _json_file(path / "config.json")
        if not isinstance(config, dict):
            raise InferenceError("모델 config.json 형식이 올바르지 않습니다.")
        config_labels = config.get("id2label")
        if config_labels is not None and config_labels != labels:
            raise InferenceError("모델 설정과 manifest 라벨 매핑이 다릅니다.")
        if config.get("num_labels", 2) != 2:
            raise InferenceError("이진 분류용으로 학습된 모델만 사용할 수 있습니다.")
        position_limit = config.get("max_position_embeddings")
        if isinstance(position_limit, int) and max_tokens > position_limit:
            raise InferenceError("manifest의 구간 크기가 모델의 실제 입력 한도를 초과합니다.")
    fingerprint = sha256(json.dumps([_file_hash(manifest_path), hashes], sort_keys=True).encode()).hexdigest()
    return dict(id=str(manifest.get("model_id") or path.name), path=str(path), status="ready", reason="",
                manifest=manifest, fingerprint=fingerprint)


def list_models(base_dir="./models"):
    """List complete and invalid local artifacts; never load a base model remotely."""
    base = Path(base_dir)
    if not base.exists():
        return []
    folders = [base] if (base / "model_manifest.json").exists() else sorted(path for path in base.iterdir() if path.is_dir())
    result = []
    for path in folders:
        try:
            result.append(_inspect_model(path))
        except (InferenceError, OSError, TypeError, ValueError) as exc:
            result.append(dict(id=path.name, path=str(path.resolve()), status="invalid", reason=str(exc), manifest=None))
    result.sort(key=lambda item: (_created_at(item["manifest"]["created_at"]) if item["status"] == "ready" else datetime.min.replace(tzinfo=timezone.utc)), reverse=True)
    return result


def get_latest_model_path(base_dir="./models"):
    models = [model for model in list_models(base_dir) if model["status"] == "ready"]
    if not models:
        raise FileNotFoundError("사용 가능한 완료 모델이 없습니다. 학습과 검증을 완료하거나 규칙 미리보기를 사용하세요.")
    return models[0]["path"]


class _BaselineRuntime:
    tokenizer = None

    def __init__(self, descriptor):
        self.payload = _baseline_payload(Path(descriptor["path"]) / "baseline_model.json")
        self.negative_id = descriptor["manifest"]["negative_id"]

    def predict(self, texts):
        result = []
        vocabulary, idf, coefficients = (self.payload[key] for key in ("vocabulary", "idf", "coef"))
        for text in texts:
            words = re.findall(self.payload["token_pattern"], text.lower())
            terms = words + [" ".join(words[index:index + 2]) for index in range(len(words) - 1)]
            counts = Counter(vocabulary[term] for term in terms if term in vocabulary)
            vector = {index: (1 + math.log(count)) * idf[index] for index, count in counts.items()}
            norm = math.sqrt(sum(value * value for value in vector.values()))
            logit = self.payload["intercept"] + sum(coefficients[index] * value / norm for index, value in vector.items()) if norm else self.payload["intercept"]
            probability = 1 / (1 + math.exp(-logit)) if logit >= 0 else math.exp(logit) / (1 + math.exp(logit))
            result.append(probability if self.negative_id == 1 else 1 - probability)
        return result


class _TransformerRuntime:
    def __init__(self, descriptor):
        try:
            import torch
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
        except ImportError as exc:
            raise InferenceError("Transformer 분석에는 torch와 transformers 설치가 필요합니다.") from exc
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(descriptor["path"], local_files_only=True, use_fast=True, trust_remote_code=False)
            self.model, loading = AutoModelForSequenceClassification.from_pretrained(
                descriptor["path"], local_files_only=True, trust_remote_code=False,
                use_safetensors=True, output_loading_info=True,
            )
            if any(loading.get(key) for key in ("missing_keys", "mismatched_keys", "error_msgs")):
                raise InferenceError("일부 분류 가중치가 없거나 달라 임의 초기화가 필요합니다. 완료 모델을 다시 확인하세요.")
            self.model.to("cpu")
            self.model.eval()
        except InferenceError:
            raise
        except Exception as exc:
            raise InferenceError("완료 모델을 로드할 수 없습니다. 로컬 가중치와 토크나이저를 확인하세요.") from exc
        if self.model.config.num_labels != 2:
            raise InferenceError("모델의 출력 클래스 수가 2가 아닙니다.")
        self.negative_id = descriptor["manifest"]["negative_id"]
        self.torch = torch
        self.max_tokens = min(
            int(getattr(self.tokenizer, "model_max_length", 8192)),
            int(getattr(self.model.config, "max_position_embeddings", 8192)),
        )

    def predict(self, texts):
        scores = []
        for start in range(0, len(texts), 16):
            inputs = self.tokenizer(texts[start:start + 16], return_tensors="pt", padding=True, truncation=False)
            with self.torch.inference_mode():
                logits = self.model(**inputs).logits
                values = self.torch.softmax(logits, dim=-1)[:, self.negative_id].tolist()
            if any(not math.isfinite(value) for value in values):
                raise InferenceError("모델이 유효한 분류 점수를 반환하지 않았습니다.")
            scores.extend(values)
        return scores


def _load_runtime(descriptor):
    """Small locked cache; replacing a manifest or weights changes its key."""
    key = descriptor["fingerprint"]
    with _MODEL_LOCK:
        if key not in _MODEL_CACHE:
            runtime = _BaselineRuntime(descriptor) if descriptor["manifest"]["model_type"] == "baseline" else _TransformerRuntime(descriptor)
            _MODEL_CACHE[key] = runtime
            while len(_MODEL_CACHE) > 2:
                _MODEL_CACHE.popitem(last=False)
        _MODEL_CACHE.move_to_end(key)
        return _MODEL_CACHE[key]


def clear_model_cache():
    with _MODEL_LOCK:
        _MODEL_CACHE.clear()
        _file_hash_revision.cache_clear()


def _rule_hits(text):
    return [dict(label=label, keyword=keyword, start=match.start(), end=match.end())
            for label, keyword in _RULES for match in re.finditer(re.escape(keyword), text)]


def _summary(segments, limit=3):
    selected, seen = [], set()
    ranking = sorted(segments, key=lambda segment: (segment["negative_score"] if segment["negative_score"] is not None else len(segment["rule_hits"])), reverse=True)
    for segment in ranking:
        key = (segment["page"], segment["paragraph_id"])
        if key in seen:
            continue
        seen.add(key)
        excerpt = segment["text"].strip()[:300]
        if excerpt:
            selected.append(dict(text=excerpt, page=segment["page"], paragraph_id=segment["paragraph_id"]))
        if len(selected) == limit:
            break
    return selected


def analyze_document(document, engine="rules", model_path=None, settings=None):
    """Analyze every selected chunk; preview matches never become model scores."""
    if engine not in ("rules", "model"):
        raise InferenceError("분석 방식은 rules 또는 model이어야 합니다.")
    supplied = dict(settings or {})
    descriptor = runtime = None
    if engine == "model":
        descriptor = _inspect_model(model_path or get_latest_model_path())
        runtime = _load_runtime(descriptor)
    manifest = descriptor["manifest"] if descriptor else {}
    resolved = dict(max_tokens=supplied.get("max_tokens", manifest.get("max_tokens", 256)),
                    overlap=supplied.get("overlap", manifest.get("overlap", 32)),
                    max_segments=supplied.get("max_segments", 500),
                    threshold=supplied.get("threshold", manifest.get("threshold", 0.5)) if descriptor else None,
                    aggregation="segment_mean" if descriptor else "keyword_matches_only")
    if descriptor:
        threshold = resolved["threshold"]
        if isinstance(threshold, bool) or not isinstance(threshold, (float, int)) or not math.isfinite(threshold) or not 0 <= threshold <= 1:
            raise InferenceError("분류 기준값은 0에서 1 사이여야 합니다.")
        if not isinstance(resolved["max_tokens"], int) or resolved["max_tokens"] > manifest.get("max_tokens", 256):
            raise InferenceError("구간 크기는 모델에 기록된 입력 한도를 넘을 수 없습니다.")
        if runtime.tokenizer is not None and resolved["max_tokens"] > runtime.max_tokens:
            raise InferenceError("구간 크기가 로컬 모델과 토크나이저의 실제 입력 한도를 넘습니다.")
    chunks = chunk_document(document, tokenizer=runtime.tokenizer if runtime else None,
                            max_tokens=resolved["max_tokens"], overlap=resolved["overlap"],
                            max_segments=resolved["max_segments"])
    segments = [dict(segment, rule_hits=_rule_hits(segment["text"]), negative_score=None) for segment in chunks["segments"]]
    if runtime and segments:
        with _PREDICTION_LOCK:
            scores = runtime.predict([segment["text"] for segment in segments])
        if len(scores) != len(segments) or any(not isinstance(score, (int, float)) or not math.isfinite(score) or not 0 <= score <= 1 for score in scores):
            raise InferenceError("모델 분류 결과가 올바르지 않습니다.")
        for segment, score in zip(segments, scores):
            segment["negative_score"] = float(score)
    scores = [segment["negative_score"] for segment in segments if segment["negative_score"] is not None]
    warnings = list(document.get("warnings", [])) + chunks["warnings"]
    if descriptor and any(resolved[key] != manifest.get(key, default) for key, default in (("max_tokens", 256), ("overlap", 32))):
        warnings.append("학습 때의 구간 크기 또는 중첩 설정과 다릅니다. 저장된 검증 지표와 분류 기준의 성능을 이 설정에 그대로 적용할 수 없습니다.")
    warnings.append("규칙 미리보기는 실제 표현의 일치만 표시하며 모델 점수를 계산하지 않습니다." if engine == "rules" else
                    "분류 점수는 가격 변화로 만든 학습 라벨에 대한 모델 출력이며, 보정된 미래 하락 확률이나 작성자의 의도가 아닙니다.")
    warnings.append("발췌 요약은 원문 구간입니다. 선택된 문구가 모델 판단의 인과적 근거라는 뜻은 아닙니다.")
    if descriptor and manifest.get("evaluation_status") == "insufficient":
        warnings.append("이 모델은 평가 불충분 상태입니다. 표본 수와 라벨 정책을 데이터·모델 화면에서 확인하세요.")
    if descriptor and isinstance(manifest.get("label_policy"), dict) and manifest["label_policy"].get("status") in ("unknown", "incomplete"):
        warnings.append("학습 자료의 수익률 라벨 정의가 미확인 또는 불완전합니다. 이 점수를 특정 관측 기간의 결과로 해석하지 마세요.")
    fingerprint = descriptor["fingerprint"] if descriptor else sha256(json.dumps(_RULES, ensure_ascii=False).encode()).hexdigest()
    return dict(document_hash=document["content_hash"], engine=engine,
                model_id=descriptor["id"] if descriptor else "rules-preview-v1",
                model_fingerprint=fingerprint, preprocessing_version=EXTRACTION_VERSION,
                created_at=datetime.now(timezone.utc).isoformat(), settings=resolved,
                metadata=deepcopy(document.get("metadata", {})), status=chunks["status"],
                segments=segments, summary=_summary(segments),
                metrics=dict(negative_score=sum(scores) / len(scores) if scores else None,
                             total_chars=chunks["total_chars"], analyzed_chars=chunks["analyzed_chars"],
                             segment_count=len(segments), unprocessed_pages=chunks["unprocessed_pages"]),
                warnings=warnings)


def predict_hidden_sell_signal(text, model_path=None):
    """Legacy (label, negative-class score in percent) interface, not a forecast.

    The second value always describes the negative class, even when the selected
    label is non-negative. It is never silently inverted into winning confidence.
    """
    result = analyze_document(document_from_text(text), engine="model", model_path=model_path)
    score = result["metrics"]["negative_score"]
    negative = score >= result["settings"]["threshold"]
    return ("하락 라벨 패턴 (Negative)" if negative else "비하락 라벨 패턴 (Non-negative)",
            100 * score)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="로컬 리포트 분석")
    parser.add_argument("text", help="분석할 문장")
    parser.add_argument("--model", help="완료 모델 디렉터리; 생략하면 규칙 미리보기")
    args = parser.parse_args()
    print(json.dumps(analyze_document(document_from_text(args.text), engine="model" if args.model else "rules",
                                      model_path=args.model), ensure_ascii=False, indent=2))
