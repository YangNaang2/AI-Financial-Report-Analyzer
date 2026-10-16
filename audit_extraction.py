"""Audit header-metadata extraction on collected reports before building consensus views.

Ticker, name, broker, date, target price and opinion are checked against the
Naver listing recorded in collection_manifest.json. Entries collected before
the listing carried target prices and opinions report those as "기준 없음";
re-running collection fills them in. The CSV keeps each source line.
"""

import argparse
from collections import Counter, defaultdict
import csv
import json
from pathlib import Path
import re

from data_processor import opinion_class

FIELDS = ("company", "ticker", "broker", "report_date", "opinion", "target_price", "current_price",
          "target_change", "previous_target_price")
REFERENCES = {"company": "company", "ticker": "ticker", "broker": "broker", "report_date": "published_at",
              "opinion": "listing_opinion", "target_price": "listing_target_price"}
MISSING_REASONS = (("서로 다른 후보", "후보 충돌"), ("복합 금액", "복합 금액"), ("형식이 불명확", "형식 불명확"))


def _date(value):
    digits = re.sub(r"[^0-9]", "", str(value or ""))
    if len(digits) == 6:
        digits = "20" + digits
    return f"{digits[:4]}-{digits[4:6]}-{digits[6:8]}" if len(digits) == 8 else None


def _normalize(field, value):
    if value in (None, ""):
        return None
    if field == "report_date":
        return _date(value)
    if field == "target_price":
        return str(value).split()[0]
    if field == "opinion":
        return opinion_class(value) or re.sub(r"\s+", "", str(value))
    return re.sub(r"\s+", "", str(value))


def _missing_reason(field, warnings):
    for warning in warnings:
        if warning.startswith(f"{field}:"):
            for needle, reason in MISSING_REASONS:
                if needle in warning:
                    return reason
    return "미발견"


def load_entries(pdf_dir):
    """Return downloaded manifest rows whose PDF still exists in the folder."""
    directory = Path(pdf_dir).expanduser()
    manifest_path = directory / "collection_manifest.json"
    if not manifest_path.is_file():
        raise ValueError(f"{manifest_path}가 없습니다. 먼저 main.py collect로 수집하세요.")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    entries = []
    for row in manifest.get("reports", {}).values():
        filename = row.get("filename") if isinstance(row, dict) else None
        if row.get("status") == "downloaded" and filename and Path(filename).name == filename and (directory / filename).is_file():
            entries.append(dict(row, path=directory / filename))
    return sorted(entries, key=lambda row: (row.get("published_at") or "", row["filename"]))


def audit_document(entry, document):
    """Compare one extracted document with its listing row."""
    metadata, evidence = document.get("metadata", {}), document.get("evidence", {})
    warnings = document.get("warnings", [])
    row = dict(filename=entry["filename"], title=entry.get("title", ""),
               listing_broker=entry.get("broker") or "", listing_ticker=entry.get("ticker") or "",
               listing_date=_date(entry.get("published_at")) or "", status=document.get("status", ""))
    for field in FIELDS:
        value = metadata.get(field)
        row[field] = value or ""
        if field in REFERENCES:
            expected = _normalize(field, entry.get(REFERENCES[field]))
            actual = _normalize(field, value)
            row[f"{field}_check"] = ("기준 없음" if expected is None else "누락" if actual is None
                                     else "일치" if actual == expected else "불일치")
            row[f"{field}_expected"] = expected or ""
            if actual is None:
                row[f"{field}_reason"] = _missing_reason(field, warnings)
        elif value:
            row[f"{field}_check"] = "추출"
        else:
            row[f"{field}_check"] = _missing_reason(field, warnings)
    for field in ("opinion", "target_price"):
        row[f"{field}_page"] = evidence.get(field, {}).get("page", "")
        row[f"{field}_source"] = evidence.get(field, {}).get("text", "")
    return row


def audit(pdf_dir, extract=None):
    """Extract every collected PDF and return per-report rows."""
    if extract is None:
        from data_processor import extract_document as extract
    rows = []
    for entry in load_entries(pdf_dir):
        try:
            rows.append(audit_document(entry, extract(entry["path"])))
        except ValueError as exc:
            rows.append(dict(filename=entry["filename"], title=entry.get("title", ""),
                             listing_broker=entry.get("broker") or "", status=f"error: {exc}"))
    return rows


def summarize(rows):
    """Group fill/match rates by the listing broker, plus an overall total."""
    groups = defaultdict(list)
    for row in rows:
        groups[row.get("listing_broker") or "미상"].append(row)
    groups["전체"] = list(rows)
    summary = {}
    for broker, members in groups.items():
        extracted = [row for row in members if not row["status"].startswith("error")]
        stats = dict(reports=len(members), errors=len(members) - len(extracted))
        for field in FIELDS:
            checks = Counter(row.get(f"{field}_check") for row in extracted)
            if field in REFERENCES:
                compared = len(extracted) - checks["기준 없음"]
                stats[field] = dict(match=checks["일치"], compared=compared,
                                    mismatch=checks["불일치"], missing=checks["누락"],
                                    reasons=Counter(row[f"{field}_reason"] for row in extracted
                                                    if row.get(f"{field}_check") == "누락"))
            else:
                stats[field] = dict(filled=checks["추출"], total=len(extracted),
                                    reasons={key: value for key, value in checks.items() if key != "추출"})
        summary[broker] = stats
    return summary


def _rate(numerator, denominator):
    return f"{numerator}/{denominator} ({numerator / denominator:.0%})" if denominator else "-"


def format_summary(summary):
    columns = ("ticker", "broker", "report_date", "opinion", "target_price")
    lines = ["증권사\t리포트\t종목코드 일치\t증권사 일치\t발간일 일치\t투자의견 일치\t목표주가 일치"]
    order = sorted((key for key in summary if key != "전체"), key=lambda key: -summary[key]["reports"]) + ["전체"]
    for broker in order:
        stats = summary[broker]
        cells = [broker, str(stats["reports"])]
        for field in columns:
            item = stats[field]
            cells.append(_rate(item["match"], item["compared"]) if field in REFERENCES else _rate(item["filled"], item["total"]))
        lines.append("\t".join(cells))
    reasons = Counter()
    for field in ("opinion", "target_price"):
        for reason, count in summary["전체"][field]["reasons"].items():
            reasons[f"{field}:{reason}"] += count
    if reasons:
        lines.append("")
        lines.append("미추출 사유: " + ", ".join(f"{key} {value}건" for key, value in reasons.most_common()))
    wrong = {field: summary["전체"][field]["mismatch"] for field in ("opinion", "target_price") if summary["전체"][field]["mismatch"]}
    if wrong:
        lines.append("목록 값과 다르게 추출: " + ", ".join(f"{field} {count}건" for field, count in wrong.items())
                     + " (CSV의 *_check가 불일치인 행 참고)")
    if summary["전체"]["errors"]:
        lines.append(f"추출 실패: {summary['전체']['errors']}건 (CSV의 status 열 참고)")
    return "\n".join(lines)


def write_csv(rows, path):
    target = Path(path).expanduser()
    target.parent.mkdir(parents=True, exist_ok=True)
    columns = list(dict.fromkeys(key for row in rows for key in row))
    with target.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def main(argv=None):
    parser = argparse.ArgumentParser(description="수집한 리포트의 메타데이터 추출 정확도 점검")
    parser.add_argument("--pdf-dir", default="data/reports", help="main.py collect의 --save-dir")
    parser.add_argument("--output", default="data/extraction_audit.csv", help="리포트별 결과 CSV")
    args = parser.parse_args(argv)
    try:
        rows = audit(args.pdf_dir)
    except (ValueError, OSError) as exc:
        parser.exit(2, f"오류: {exc}\n")
    if not rows:
        parser.exit(1, "점검할 수집 PDF가 없습니다.\n")
    write_csv(rows, args.output)
    print(format_summary(summarize(rows)))
    print(f"\n리포트별 결과: {args.output}")
    print("일치율은 네이버 목록 값 기준입니다. 목록 값이 없는 예전 수집분은 main.py collect를 다시 실행하면 채워집니다.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
