import csv
import json
from pathlib import Path
import tempfile
import unittest

import audit_extraction
from data_processor import DocumentError, document_from_text


def collected(directory, reports):
    rows = {}
    for index, report in enumerate(reports):
        filename = f"report-{index}.pdf"
        (directory / filename).write_bytes(b"%PDF-1.4")
        rows[str(index)] = dict(report, status="downloaded", filename=filename)
    rows["failed"] = dict(status="error", title="실패한 수집")
    (directory / "collection_manifest.json").write_text(json.dumps({"schema_version": 1, "reports": rows}), encoding="utf-8")


HEADER = "삼성전자(005930)\n미래에셋증권\n발간일: 2026.09.29\n투자의견: 매수\n목표주가: 90,000원\n"


class AuditExtractionTests(unittest.TestCase):
    def test_checks_listing_fields_and_reports_missing_reasons(self):
        texts = {"report-0.pdf": HEADER,
                 "report-1.pdf": "SK하이닉스(000660)\n발간일: 2026.09.28\n목표주가: 1,000원\n목표주가: 2,000원\n"}
        with tempfile.TemporaryDirectory() as folder:
            directory = Path(folder)
            collected(directory, [
                dict(title="좋은 실적", broker="미래에셋증권", ticker="005930", published_at="2026-09-29"),
                dict(title="표지가 다름", broker="삼성증권", ticker="000660", published_at="2026.09.29"),
            ])
            rows = audit_extraction.audit(directory, extract=lambda path: document_from_text(texts[path.name], path.name))
        first, second = rows
        self.assertEqual((first["ticker_check"], first["broker_check"], first["report_date_check"]), ("일치", "일치", "일치"))
        self.assertEqual((first["target_price"], first["target_price_check"], first["target_price_page"]), ("90000 원", "추출", 1))
        self.assertIn("90,000", first["target_price_source"])
        self.assertEqual((second["broker_check"], second["report_date_check"]), ("누락", "불일치"))
        self.assertEqual((second["target_price_check"], second["opinion_check"]), ("후보 충돌", "미발견"))
        summary = audit_extraction.summarize(rows)
        self.assertEqual(summary["전체"]["ticker"], dict(match=2, compared=2, mismatch=0, missing=0))
        self.assertEqual(summary["미래에셋증권"]["target_price"]["filled"], 1)
        self.assertIn("target_price:후보 충돌 1건", audit_extraction.format_summary(summary))

    def test_extraction_errors_are_counted_and_csv_is_written(self):
        def extract(path):
            raise DocumentError("암호화되었거나 텍스트 추출이 제한된 PDF입니다.")

        with tempfile.TemporaryDirectory() as folder:
            directory = Path(folder)
            collected(directory, [dict(title="잠긴 문서", broker="KB증권", ticker="035420", published_at="2026-09-29")])
            rows = audit_extraction.audit(directory, extract=extract)
            output = directory / "out" / "audit.csv"
            audit_extraction.write_csv(rows, output)
            with output.open(encoding="utf-8-sig", newline="") as handle:
                written = list(csv.DictReader(handle))
        self.assertTrue(rows[0]["status"].startswith("error"))
        self.assertEqual(audit_extraction.summarize(rows)["KB증권"]["errors"], 1)
        self.assertEqual(written[0]["listing_broker"], "KB증권")

    def test_missing_manifest_is_reported(self):
        with tempfile.TemporaryDirectory() as folder:
            with self.assertRaisesRegex(ValueError, "collect"):
                audit_extraction.load_entries(folder)


if __name__ == "__main__":
    unittest.main()
