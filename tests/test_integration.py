from copy import deepcopy
import csv
import io
import json
import tempfile
import unittest

from data_processor import document_from_text
from demo import seed_demo
from exports import analyses_to_csv, analyses_to_json, analysis_to_html
from inference import analyze_document
from storage import Library, LibraryError


TEXT = """가상전자 (005930)
삼성증권
발간일: 2026-01-02
투자의견: 매수
목표주가: 120,000원
현재주가: 95,000원

매출액은 1조 2,000억원을 기록했다. 수요 둔화와 비용 증가로 영업이익 감소가 예상된다.

하반기 신규 수주 회복을 기대하나 재고 부담과 실적 하향 가능성을 확인해야 한다."""


class LibraryIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.library = Library(":memory:")
        self.addCleanup(self.library.close)

    def test_real_extraction_rules_storage_exports_backup_roundtrip(self):
        document = document_from_text(TEXT, filename="가상 리포트.txt")
        result = analyze_document(document, engine="rules", settings={"max_tokens": 8, "overlap": 2})
        identifier = self.library.save_document("owner", document)
        run = self.library.save_analysis("owner", identifier, result)
        saved = self.library.get_analysis("owner", run)
        self.assertEqual(saved["analysis"]["metadata"]["ticker"], "005930")
        self.assertGreater(len(saved["analysis"]["segments"]), 1)
        self.assertIsNone(saved["analysis"]["metrics"]["negative_score"])
        for segment in saved["analysis"]["segments"]:
            page = next(page for page in document["pages"] if page["number"] == segment["page"])
            self.assertEqual(segment["text"], page["text"][segment["start"]:segment["end"]])
        self.assertEqual(json.loads(analyses_to_json([saved])), [saved])
        row = next(csv.DictReader(io.StringIO(analyses_to_csv([saved]).decode("utf-8-sig"))))
        self.assertEqual(row["ticker"], "005930")
        self.assertEqual(row["negative_class_score"], "")
        html = analysis_to_html(document, saved["analysis"])
        self.assertIn("원문 문장", html)
        self.assertIn("실적 하향", html)
        backup = self.library.export_backup("owner")
        self.assertEqual(self.library.import_backup("restored", backup)["analyses"], 1)
        restored = self.library.list_analyses("restored")[0]
        self.assertNotEqual(restored["id"], saved["id"])
        self.assertEqual(restored["analysis"], saved["analysis"])
        self.assertEqual(restored["created_at"], saved["created_at"])

    def test_historical_source_offsets_and_html_survive_same_pdf_reextraction(self):
        first = document_from_text(TEXT, filename="original.txt")
        identifier = self.library.save_document("owner", first)
        analysis = analyze_document(first, engine="rules")
        run = self.library.save_analysis("owner", identifier, analysis)
        revised = document_from_text("변경된 추출문. 다른 도구로 다시 추출한 문서.", filename="reextracted.txt")
        revised["content_hash"] = first["content_hash"]  # Same original PDF bytes, new extraction.
        revised["extraction_version"] = "3"
        revised["metadata"]["company"] = "수정된 정보"
        self.library.save_document("owner", revised)
        saved = self.library.get_analysis("owner", run)["analysis"]
        self.assertEqual(saved["document_snapshot"]["text"], first["text"])
        self.assertEqual(saved["document_snapshot"]["extraction_version"], "2")
        html = analysis_to_html(self.library.get_document("owner", identifier), saved)
        self.assertIn("가상전자", html)
        self.assertNotIn("변경된 추출문", html)
        self.library.import_backup("restored", self.library.export_backup("owner"))
        restored = self.library.list_analyses("restored")[0]["analysis"]
        self.assertEqual(restored["document_snapshot"]["text"], first["text"])

    def test_restore_checks_snapshot_hash_segment_offsets_and_summary_before_writing(self):
        document = document_from_text(TEXT)
        identifier = self.library.save_document("owner", document)
        self.library.save_analysis("owner", identifier, analyze_document(document))
        original = json.loads(self.library.export_backup("owner"))
        mutations = [
            lambda result: result["document_snapshot"].update(content_hash="0" * 64),
            lambda result: result["segments"][0].update(start=-1),
            lambda result: result["segments"][0].update(text="다른 원문"),
            lambda result: result["summary"][0].update(text="원문에 없는 요약"),
            lambda result: result["metrics"].update(segment_count=9999),
        ]
        for mutate in mutations:
            bad = deepcopy(original)
            mutate(bad["analyses"][0]["analysis"])
            with self.subTest(mutation=mutate), self.assertRaises(LibraryError):
                self.library.import_backup("new-owner", json.dumps(bad))
            self.assertEqual(self.library.list_documents("new-owner"), [])

    def test_fictional_demo_has_no_model_scores_and_separate_owners(self):
        seed_demo(self.library, "demo-owner")
        self.assertEqual(len(self.library.list_documents("demo-owner")), 3)
        rows = self.library.list_analyses("demo-owner")
        self.assertEqual(len(rows), 3)
        self.assertTrue(all(row["analysis"]["engine"] == "demo" for row in rows))
        self.assertTrue(all(row["analysis"]["metrics"]["negative_score"] is None for row in rows))
        self.assertEqual(self.library.list_documents("actual-owner"), [])
        seed_demo(self.library, "demo-owner")
        self.assertEqual(len(self.library.list_analyses("demo-owner")), 3)

    def test_real_cpu_training_artifact_inference_storage_and_restore(self):
        from train import train_model
        records = []
        for month in range(1, 11):
            for label in (0, 1):
                phrase = "수요 둔화 실적 부진 이익 감소" if label else "수요 성장 실적 개선 이익 회복"
                records.append({"id": f"{month}-{label}", "text": f"{phrase} 고유 분석 {month}월 문서 {label}",
                                "label_id": label, "report_date": f"2024-{month:02d}-01",
                                "label_end_date": f"2024-{month:02d}-02", "status": "labeled"})
        with tempfile.TemporaryDirectory() as directory:
            manifest = train_model(records=records, output_dir=directory, max_tokens=32, overlap=8)
            document = document_from_text("실적 부진 우려와 이익 감소를 반영하여 전망을 하향합니다.")
            result = analyze_document(document, engine="model", model_path=manifest["model_dir"])
            self.assertTrue(0 <= result["metrics"]["negative_score"] <= 1)
            identifier = self.library.save_document("owner", document)
            run = self.library.save_analysis("owner", identifier, result)
            self.assertEqual(self.library.import_backup("restored", self.library.export_backup("owner"))["analyses"], 1)
            restored = self.library.list_analyses("restored")[0]["analysis"]
            self.assertEqual(restored, self.library.get_analysis("owner", run)["analysis"])

    def test_financial_facts_and_units_survive_all_exports(self):
        document = document_from_text("가상전자 (005930)\n목표주가: 120,000원\n\n매출액: 1,234억원\n영업이익률: -2.5%")
        result = analyze_document(document)
        identifier = self.library.save_document("owner", document)
        run = self.library.save_analysis("owner", identifier, result)
        saved = self.library.get_analysis("owner", run)["analysis"]
        self.assertEqual(saved["metadata"]["target_price"], "120000 원")
        self.assertEqual(saved["document_snapshot"]["financial_facts"], document["financial_facts"])
        html = analysis_to_html(document, saved)
        self.assertIn("재무 수치와 원문 근거", html)
        self.assertIn("-2.5", html)
        self.assertIn("억원", html)


if __name__ == "__main__":
    unittest.main()
