"""Portable analysis exports with escaped HTML and spreadsheet-safe labels.

CSV stores ticker codes as their original strings (including leading zeroes).
When opening in Excel, use Data > From Text/CSV and set the ticker column's type
to Text. Double-clicking a CSV may make Excel infer numbers and drop the zeroes.
"""

import csv
from html import escape
import io
import json
import math

from storage import _safe_json


_METADATA_LABELS = {"company": "기업", "ticker": "종목코드", "broker": "증권사", "report_date": "발간일",
                    "opinion": "투자의견", "target_price": "목표주가", "current_price": "현재주가"}
_ENGINE_LABELS = {"model": "로컬 모델 분석", "rules": "규칙 기반 문구 탐색", "demo": "데모 예시"}


def _csv_text(value):
    value = "" if value is None else str(value)
    if value.lstrip().startswith(("=", "+", "-", "@")) or value.startswith(("\t", "\r", "\n")):
        return "'" + value
    return value


def _records(records):
    if not isinstance(records, list) or any(not isinstance(row, dict) or not isinstance(row.get("analysis"), dict) for row in records):
        raise ValueError("분석 기록 목록 형식을 확인해 주세요.")
    _safe_json(records)
    return records


def _score(analysis, row=None):
    if analysis.get("engine") != "model":
        return None
    score = (analysis.get("metrics") or {}).get("negative_score") if row is None else row.get("negative_score")
    if type(score) not in (int, float) or not math.isfinite(score) or not 0 <= score <= 1:
        return None
    return score


def analyses_to_csv(records):
    """Return BOM UTF-8 CSV; rule/demo rows never have model score values."""
    records = _records(records)
    buffer = io.StringIO(newline="")
    fields = ["run_id", "document_id", "filename", *_METADATA_LABELS,
              "engine", "model_id", "model_fingerprint", "preprocessing_version", "status",
              "negative_class_score", "segment_count", "analyzed_chars", "total_chars", "created_at", "warnings"]
    writer = csv.DictWriter(buffer, fieldnames=fields)
    writer.writeheader()
    for record in records:
        analysis = record["analysis"]
        metadata, metrics = analysis.get("metadata") or {}, analysis.get("metrics") or {}
        row = {key: metadata.get(key) for key in _METADATA_LABELS}
        row.update(run_id=record.get("id", ""), document_id=record.get("document_id", ""),
                   filename=analysis.get("document_filename", ""), engine=analysis.get("engine", ""),
                   model_id=analysis.get("model_id", ""), model_fingerprint=analysis.get("model_fingerprint", ""),
                   preprocessing_version=analysis.get("preprocessing_version", ""), status=analysis.get("status", ""),
                   negative_class_score=_score(analysis), segment_count=metrics.get("segment_count"),
                   analyzed_chars=metrics.get("analyzed_chars"), total_chars=metrics.get("total_chars"),
                   created_at=record.get("created_at", analysis.get("created_at", "")),
                   warnings=" | ".join(str(warning) for warning in analysis.get("warnings", [])))
        writer.writerow({key: _csv_text(value) for key, value in row.items()})
    return buffer.getvalue().encode("utf-8-sig")


def analyses_to_json(records):
    """Preserve complete JSON records and their immutable metadata snapshots."""
    return json.dumps(_records(records), ensure_ascii=False, indent=2, allow_nan=False)


def _h(value):
    return escape("" if value is None else str(value), quote=True)


def analysis_to_html(document, analysis):
    """Render a standalone local report; source text is always escaped."""
    if not isinstance(document, dict) or not isinstance(analysis, dict):
        raise ValueError("리포트와 분석 결과는 객체 형식이어야 합니다.")
    _safe_json(document)
    _safe_json(analysis)
    document = analysis.get("document_snapshot") or document
    metadata = analysis["metadata"] if "metadata" in analysis else document.get("metadata") or {}
    metrics = analysis.get("metrics") or {}
    filename = analysis.get("document_filename") or document.get("filename", "리포트")
    engine = _ENGINE_LABELS.get(analysis.get("engine"), "분석 엔진 미확인")
    metadata_rows = "".join(f"<tr><th>{_h(label)}</th><td>{_h(metadata.get(key)) or '미확인'}</td></tr>"
                            for key, label in _METADATA_LABELS.items())
    model_info = f"<p>모델: {_h(analysis.get('model_id', '—'))} · 전처리 버전: {_h(analysis.get('preprocessing_version', '—'))}</p>"
    score = _score(analysis)
    if analysis.get("engine") == "model":
        score_text = "모델 분류 점수 미확인" if score is None else f"부정 클래스 점수: {score:.4f}"
        score_help = "모델의 미보정 분류 점수입니다. 주가 하락이나 투자 손실의 확률이 아닙니다."
    else:
        score_text = "모델 점수 없음"
        score_help = "규칙 및 데모 결과는 문구 탐색을 위한 자료이며 모델 예측 점수를 제공하지 않습니다."
    warnings = list(document.get("warnings", [])) + list(analysis.get("warnings", []))
    warning_html = "".join(f"<li>{_h(warning)}</li>" for warning in dict.fromkeys(warnings)) or "<li>기록된 경고 없음</li>"
    evidence_rows = []
    for segment in analysis.get("segments", []):
        hits = ", ".join(str(hit.get("label") or hit.get("keyword", "")) for hit in segment.get("rule_hits", []))
        segment_score = _score(analysis, segment)
        rendered_score = "—" if segment_score is None else f"{segment_score:.4f}"
        evidence_rows.append(f"<tr><td>{_h(segment.get('page'))}</td><td>{_h(segment.get('text'))}</td>"
                             f"<td>{_h(hits) or '—'}</td><td>{rendered_score}</td></tr>")
    evidence_html = "".join(evidence_rows) or '<tr><td colspan="4">분석된 문장이 없습니다.</td></tr>'
    summary_html = "".join(f"<li>{_h(item.get('text', ''))} <small>({_h(item.get('page', ''))}쪽)</small></li>"
                           for item in analysis.get("summary", [])) or "<li>추출 요약 없음</li>"
    metadata_evidence = "".join(f"<tr><th>{_h(_METADATA_LABELS.get(field, field))}</th><td>{_h(item.get('page'))}</td><td>{_h(item.get('text'))}</td></tr>"
                                for field, item in (document.get("evidence") or {}).items() if isinstance(item, dict))
    financial_rows = "".join(f"<tr><td>{_h(fact.get('metric'))}</td><td>{_h(fact.get('value'))}</td><td>{_h(fact.get('unit'))}</td><td>{_h(fact.get('page'))}</td><td>{_h(fact.get('text'))}</td></tr>"
                             for fact in document.get("financial_facts", []) if isinstance(fact, dict))
    financial_html = ("<h2>재무 수치와 원문 근거</h2><table><thead><tr><th>항목</th><th>수치</th><th>단위</th><th>쪽</th><th>원문</th></tr></thead><tbody>"
                      + financial_rows + "</tbody></table>") if financial_rows else ""
    return f"""<!doctype html>
<html lang="ko"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<meta http-equiv="Content-Security-Policy" content="default-src 'none'; style-src 'unsafe-inline'; base-uri 'none'; form-action 'none'">
<title>{_h(filename)} · Report Lens</title>
<style>body{{font-family:system-ui,sans-serif;color:#142334;background:#eef3f8;margin:0;line-height:1.65}}main{{max-width:1100px;margin:36px auto;padding:38px;background:white;border-radius:18px}}h1{{line-height:1.25}}h2{{margin-top:32px}}table{{border-collapse:collapse;width:100%;font-size:14px}}th,td{{border:1px solid #dce4ed;padding:10px;text-align:left;vertical-align:top;overflow-wrap:anywhere}}th{{background:#eef3f8}}pre{{white-space:pre-wrap;overflow-wrap:anywhere;background:#f5f8fb;padding:20px;border-radius:10px;font-family:inherit}}.badge{{display:inline-block;background:#e9efff;color:#233e88;padding:5px 12px;border-radius:20px}}.notice{{padding:14px 18px;background:#fff5dd;border-radius:10px}}small{{color:#596b7d}}@media print{{body{{background:white}}main{{margin:0;padding:0}}pre,table{{font-size:10px}}}}@media(max-width:600px){{main{{margin:8px;padding:18px}}}}</style></head>
<body><main><p class="badge">Report Lens · {_h(engine)}</p><h1>{_h(filename)}</h1>
<p>분석 시각: {_h(analysis.get('created_at', '—'))} · 상태: {_h(analysis.get('status', '—'))}</p>
<h2>분석 당시 리포트 정보</h2><table>{metadata_rows}</table>{model_info}
<h2>{_h(score_text)}</h2><p class="notice">{_h(score_help)}</p>
<p>분석 문장 {_h(metrics.get('segment_count', 0))}개 · 분석 문자 {_h(metrics.get('analyzed_chars', 0))} / 전체 {_h(metrics.get('total_chars', 0))}</p>
<h2>추출 요약</h2><ul>{summary_html}</ul><h2>문장별 근거</h2>
<table><thead><tr><th>쪽</th><th>원문 문장</th><th>규칙 탐색</th><th>부정 클래스 점수</th></tr></thead><tbody>{evidence_html}</tbody></table>
<h2>메타데이터 추출 근거</h2><table><thead><tr><th>항목</th><th>쪽</th><th>원문</th></tr></thead><tbody>{metadata_evidence or '<tr><td colspan="3">추출 근거 없음</td></tr>'}</tbody></table>
{financial_html}
<h2>경고와 처리 한계</h2><ul>{warning_html}</ul><h2>추출 원문</h2><pre>{_h(document.get('text', ''))}</pre>
<p><small>텍스트와 분석 결과를 함께 확인하기 위한 기록입니다. 분석 결과는 투자 권고가 아닙니다.</small></p></main></body></html>"""
