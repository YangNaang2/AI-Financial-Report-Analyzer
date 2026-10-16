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
                dict(title="좋은 실적", broker="미래에셋증권", ticker="005930", published_at="2026-09-29",
                     company="삼성전자", listing_target_price=90000, listing_opinion="Buy"),
                dict(title="표지가 다름", broker="삼성증권", ticker="000660", published_at="2026.09.29",
                     company="SK하이닉스", listing_target_price=1500, listing_opinion="매수"),
            ])
            rows = audit_extraction.audit(directory, extract=lambda path: document_from_text(texts[path.name], path.name))
        first, second = rows
        self.assertEqual((first["ticker_check"], first["broker_check"], first["report_date_check"], first["company_check"]),
                         ("일치", "일치", "일치", "일치"))
        # "매수" on the cover and "Buy" in the listing are the same opinion.
        self.assertEqual((first["target_price_check"], first["opinion_check"], first["target_price_page"]), ("일치", "일치", 1))
        self.assertIn("90,000", first["target_price_source"])
        self.assertEqual((second["broker_check"], second["report_date_check"]), ("누락", "불일치"))
        self.assertEqual((second["target_price_check"], second["target_price_reason"], second["target_price_expected"]),
                         ("누락", "후보 충돌", "1500"))
        self.assertEqual((second["opinion_check"], second["opinion_reason"]), ("누락", "미발견"))
        summary = audit_extraction.summarize(rows)
        self.assertEqual({key: summary["전체"]["ticker"][key] for key in ("match", "compared", "mismatch", "missing")},
                         dict(match=2, compared=2, mismatch=0, missing=0))
        self.assertEqual(summary["전체"]["target_price"]["match"], 1)
        self.assertIn("target_price:후보 충돌 1건", audit_extraction.format_summary(summary))

    def test_older_entries_without_listing_values_have_no_reference(self):
        with tempfile.TemporaryDirectory() as folder:
            directory = Path(folder)
            collected(directory, [dict(title="예전 수집", broker="미래에셋증권", ticker="005930", published_at="2026-09-29")])
            (row,) = audit_extraction.audit(directory, extract=lambda path: document_from_text(HEADER, path.name))
        self.assertEqual((row["target_price_check"], row["opinion_check"]), ("기준 없음", "기준 없음"))
        self.assertEqual(audit_extraction.summarize([row])["전체"]["target_price"]["compared"], 0)

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
