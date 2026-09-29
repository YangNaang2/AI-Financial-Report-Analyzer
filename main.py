"""Explicit collect, prepare and train commands; importing never starts work."""

import argparse
from collections import Counter
from datetime import date, datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
from uuid import uuid4


def extract_metadata_from_raw_text(raw_text):
    """Compatibility helper returning ticker and dotted date from report headers.

    Handles arbitrary four-digit years rather than a hard-coded year cutoff.
    Ambiguous dates/codes are left blank instead of selecting an unrelated price.
    """
    if not isinstance(raw_text, str):
        return None, None
    header = raw_text[:4000]
    explicit = re.findall(r"(?:종목코드|티커|Ticker)\s*[:：]?\s*A?([0-9]{6})(?![0-9])", header, re.I)
    paired = re.findall(r"\(\s*A?([0-9]{6})(?:\.(?:KS|KQ))?\s*\)", header)
    codes = set(explicit or paired or re.findall(r"(?<![0-9])A([0-9]{6})(?![0-9])", header))
    dates = set()
    for year, month, day in re.findall(r"(?<![0-9])([0-9]{4})\s*[./년-]\s*([0-9]{1,2})\s*[./월-]\s*([0-9]{1,2})(?![0-9])", header):
        try:
            dates.add(date(int(year), int(month), int(day)))
        except ValueError:
            pass
    months = {name: index for index, name in enumerate(("jan", "feb", "mar", "apr", "may", "jun",
                                                        "jul", "aug", "sep", "oct", "nov", "dec"), 1)}
    for month, day, year in re.findall(r"\b(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+([0-9]{1,2}),?\s+([0-9]{4})\b", header, re.I):
        try:
            dates.add(date(int(year), months[month.lower()], int(day)))
        except ValueError:
            pass
    ticker = next(iter(codes)) if len(codes) == 1 else None
    published = next(iter(dates)).strftime("%Y.%m.%d") if len(dates) == 1 else None
    return ticker, published


def _save_json(path, value):
    target = Path(path).expanduser()
    target.parent.mkdir(parents=True, exist_ok=True)
    staged = target.with_name(f".{target.name}.{uuid4().hex}.tmp")
    try:
        staged.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        os.replace(staged, target)
    finally:
        staged.unlink(missing_ok=True)


def prepare_dataset(pdf_dir="./pdf_data", *, window_days=30, threshold=-5,
                    entry_policy="next_session", as_of=None, price_loader=None,
                    progress=None, output_path=None, dry_run=False):
    """Extract local documents and retain explicit pending/unavailable outcomes.

    No training is started. A collection manifest supplies optional provenance;
    source text and metadata evidence remain in each document-level record.
    """
    from data_processor import extract_document
    from labeler import calculate_post_report_return
    directory = Path(pdf_dir).expanduser()
    if not directory.is_dir():
        raise ValueError("자료 폴더가 없습니다. PDF를 넣거나 수집부터 실행해 주세요.")
    paths = sorted(path for path in directory.iterdir() if path.is_file() and path.suffix.lower() in (".pdf", ".txt", ".md"))
    sources = {}
    manifest_path = directory / "collection_manifest.json"
    manifest_warning = None
    if manifest_path.exists():
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            reports = manifest.get("reports", {})
            for row in (reports.values() if isinstance(reports, dict) else reports):
                if isinstance(row, dict) and row.get("filename"):
                    sources[row["filename"]] = row
        except (ValueError, OSError, AttributeError, TypeError):
            manifest_warning = "수집 이력 파일을 읽지 못했습니다. 원문에서 확인 가능한 정보만 사용합니다."
    records, errors, seen = [], [], set()
    for index, path in enumerate(paths, 1):
        if progress:
            progress(f"문서 준비 {index}/{len(paths)} · {path.name}")
        try:
            document = extract_document(path)
            digest = document.get("content_hash") or hashlib.sha256(path.read_bytes()).hexdigest()
            if digest in seen:
                errors.append({"filename": path.name, "status": "duplicate", "reason": "동일한 파일 내용이 이미 있습니다."})
                continue
            seen.add(digest)
            metadata = dict(document.get("metadata") or {})
            source = sources.get(path.name, {})
            legacy_ticker, legacy_date = extract_metadata_from_raw_text(document.get("text", ""))
            ticker = metadata.get("ticker") or source.get("ticker") or legacy_ticker
            published = metadata.get("report_date") or source.get("published_at") or legacy_date
            if published:
                published = str(published)[:10].replace(".", "-")
            metadata.update(ticker=ticker, report_date=published)
            row = {"id": digest, "content_hash": digest, "filename": path.name,
                   "text": document.get("text", ""), "ticker": ticker, "report_date": published,
                   "label_end_date": None, "label_id": None, "metadata": metadata,
                   "metadata_evidence": document.get("evidence", {}),
                   "source": {key: source.get(key) for key in ("source_url", "title", "collected_at", "published_at")},
                   "warnings": document.get("warnings", []), "status": "unavailable"}
            if document.get("status") in ("ocr_required", "partial"):
                label = {"status": "unavailable", "reason": "전체 텍스트가 추출되지 않았습니다. OCR/원문 확인 후 다시 준비하세요.", "label_id": None}
            elif not ticker or not published:
                label = {"status": "unavailable", "reason": "종목코드 또는 발행일을 확정할 수 없습니다.", "label_id": None}
            elif dry_run:
                label = {"status": "pending", "reason": "dry-run: 가격을 조회하지 않았습니다.", "label_id": None}
            else:
                label = calculate_post_report_return(ticker, published, window_days=window_days,
                                                      threshold=threshold, entry_policy=entry_policy,
                                                      as_of=as_of, price_loader=price_loader)
            row.update(status=label["status"], label_info=label, label_id=label.get("label_id"),
                       label_end_date=label.get("end_date"))
            records.append(row)
        except Exception as exc:
            errors.append({"filename": path.name, "status": "error", "reason": f"{type(exc).__name__}: {exc}"})
    counts = Counter(row["status"] for row in records)
    result = {"schema_version": 1, "created_at": datetime.now(timezone.utc).isoformat(),
              "status": "ready" if counts["labeled"] else "no_labeled_data",
              "records": records, "errors": errors,
              "counts": {"files": len(paths), "records": len(records), "labeled": counts["labeled"],
                         "pending": counts["pending"], "unavailable": counts["unavailable"],
                         "errors": sum(error["status"] == "error" for error in errors),
                         "duplicates": sum(error["status"] == "duplicate" for error in errors)},
              "label_policy": {"window_days": window_days, "threshold": threshold, "entry_policy": entry_policy,
                               "as_of": str(as_of or date.today()), "price_adjustment": "unknown"},
              "warnings": [manifest_warning] if manifest_warning else []}
    if output_path:
        _save_json(output_path, result)
    return result


def run_pipeline(pdf_dir="./pdf_data", **kwargs):
    """Compatibility entry point; prepare only, never implicitly train."""
    return prepare_dataset(pdf_dir, **kwargs)


def main(argv=None):
    parser = argparse.ArgumentParser(description="리포트 수집·관측 라벨 준비·시간 분할 학습")
    sub = parser.add_subparsers(dest="command")
    collect = sub.add_parser("collect", help="네이버 공개 기업 리포트 수집")
    collect.add_argument("--save-dir", default="./pdf_data")
    collect.add_argument("--start-page", type=int, default=1)
    collect.add_argument("--end-page", type=int, default=1)
    collect.add_argument("--dry-run", action="store_true")
    prepare = sub.add_parser("prepare", help="로컬 문서에 관측 완료 수익률 라벨 준비")
    prepare.add_argument("--pdf-dir", default="./pdf_data")
    prepare.add_argument("--output", default="./dataset.json")
    prepare.add_argument("--window-days", type=int, default=30)
    prepare.add_argument("--threshold", type=float, default=-5)
    prepare.add_argument("--entry-policy", choices=("next_session", "report_day"), default="next_session")
    prepare.add_argument("--as-of")
    prepare.add_argument("--dry-run", action="store_true")
    training = sub.add_parser("train", help="문서 시간 분할 후 학습 및 보류 시험 평가")
    training.add_argument("--records", required=True)
    training.add_argument("--output-dir", default="./models")
    training.add_argument("--backend", choices=("baseline", "transformer"), default="baseline")
    training.add_argument("--epochs", type=int, default=3)
    training.add_argument("--batch-size", type=int, default=8)
    training.add_argument("--max-tokens", type=int, default=256)
    training.add_argument("--overlap", type=int, default=32)
    training.add_argument("--seed", type=int, default=42)
    training.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    if args.command is None:
        parser.print_help()
        return 0
    try:
        if args.command == "collect":
            from crawler import download_naver_reports
            result = ({"status": "dry_run", "start_page": args.start_page, "end_page": args.end_page,
                       "save_dir": args.save_dir} if args.dry_run else
                      download_naver_reports(args.save_dir, args.start_page, args.end_page))
        elif args.command == "prepare":
            result = prepare_dataset(args.pdf_dir, window_days=args.window_days, threshold=args.threshold,
                                     entry_policy=args.entry_policy, as_of=args.as_of, dry_run=args.dry_run,
                                     output_path=args.output)
            result = {key: value for key, value in result.items() if key != "records"}
        else:
            from train import temporal_split, train_model
            loaded = json.loads(Path(args.records).read_text(encoding="utf-8"))
            records = loaded.get("records") if isinstance(loaded, dict) else loaded
            if args.dry_run:
                groups, audit = temporal_split(records)
                result = {"status": "dry_run", "counts": {key: len(value) for key, value in groups.items()}, "audit": audit}
            else:
                result = train_model(records=records, output_dir=args.output_dir, backend=args.backend,
                                     epochs=args.epochs, batch_size=args.batch_size, max_tokens=args.max_tokens,
                                     overlap=args.overlap, seed=args.seed)
        print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
        return 0
    except (ValueError, OSError) as exc:
        parser.exit(2, f"오류: {exc}\n")


if __name__ == "__main__":
    raise SystemExit(main())
