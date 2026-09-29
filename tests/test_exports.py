from copy import deepcopy
import csv
import io
import json
import unittest

from exports import analyses_to_csv, analyses_to_json, analysis_to_html
from tests.test_storage import analysis, document


def record(**changes):
    result = {"id": "run-1", "document_id": "document-1", "created_at": "2026-01-01T00:00:00+00:00",
              "analysis": analysis(metadata=document()["metadata"], document_filename="샘플.pdf")}
    result.update(changes)
    return result


class ExportTests(unittest.TestCase):
    def test_csv_bom_metadata_leading_zero_codes_and_blank_rule_scores(self):
        payload = analyses_to_csv([record()])
        self.assertTrue(payload.startswith(b"\xef\xbb\xbf"))
        rows = list(csv.DictReader(io.StringIO(payload.decode("utf-8-sig"))))
        self.assertEqual(rows[0]["ticker"], "005930")
        self.assertEqual(rows[0]["company"], "가상기업")
        self.assertEqual(rows[0]["engine"], "rules")
        self.assertEqual(rows[0]["negative_class_score"], "")
        self.assertEqual(rows[0]["status"], "ready")

    def test_csv_formula_safety_all_user_labels_and_literal_code(self):
        row = record()
        row["analysis"]["metadata"].update(company="=HYPERLINK(1)", broker=" +SUM(1)", opinion="\t@bad")
        row["analysis"]["document_filename"] = "-file.pdf"
        row["analysis"]["warnings"] = ["@unsafe"]
        exported = next(csv.DictReader(io.StringIO(analyses_to_csv([row]).decode("utf-8-sig"))))
        self.assertEqual(exported["company"], "'=HYPERLINK(1)")
        self.assertEqual(exported["broker"], "' +SUM(1)")
        self.assertEqual(exported["opinion"], "'\t@bad")
        self.assertEqual(exported["filename"], "'-file.pdf")
        self.assertEqual(exported["warnings"], "'@unsafe")
        self.assertEqual(exported["ticker"], "005930")

    def test_model_zero_is_exported_and_rules_demo_never_get_fake_scores(self):
        rows = []
        for engine, score in (("model", 0.0), ("model", .7345), ("rules", .95), ("demo", .99)):
            row = record()
            row["analysis"].update(engine=engine, metrics={"negative_score": score})
            rows.append(row)
        exported = list(csv.DictReader(io.StringIO(analyses_to_csv(rows).decode("utf-8-sig"))))
        self.assertEqual([row["negative_class_score"] for row in exported], ["0.0", "0.7345", "", ""])

    def test_json_preserves_all_fields_none_scores_codes_and_immutable_snapshot(self):
        rows = [record()]
        rows[0]["analysis"]["extra"] = {"nested": [1, True, None, "한글"]}
        original = deepcopy(rows)
        self.assertEqual(json.loads(analyses_to_json(rows)), original)
        self.assertEqual(rows, original)
        self.assertIn('"005930"', analyses_to_json(rows))
        with self.assertRaises(ValueError):
            analyses_to_json([record(analysis={"metrics": {"negative_score": float("nan")}})])

    def test_html_escapes_every_source_and_has_no_remote_assets_or_executable_markup(self):
        malicious = '<script>alert("x")</script><img src="https://example.com/x" onerror="alert(1)">'
        doc = document(text=malicious, filename=malicious, metadata={"company": malicious},
                       evidence={"company": {"page": 1, "text": malicious}}, warnings=[malicious])
        result = analysis(metadata={"company": malicious, "ticker": "005930"}, document_filename=malicious,
                          segments=[{"page": 1, "text": malicious, "negative_score": None,
                                     "rule_hits": [{"label": malicious}]}], summary=[{"text": malicious, "page": 1}], warnings=[malicious])
        html = analysis_to_html(doc, result)
        self.assertNotIn("<script", html)
        self.assertNotIn("<img", html)
        self.assertIn("&lt;script&gt;", html)
        self.assertIn("Content-Security-Policy", html)
        self.assertIn("default-src 'none'", html)
        self.assertNotIn('src="https://', html)
        self.assertIn("005930", html)
        self.assertIn("규칙 기반 문구 탐색", html)

    def test_html_uses_analysis_metadata_and_describes_scores_as_class_scores(self):
        doc = document(metadata={"company": "현재 기업"})
        result = analysis(engine="model", metadata={"company": "분석 당시 기업"}, metrics={"negative_score": .85},
                          segments=[{"page": 1, "text": "검토 문장", "negative_score": .6, "rule_hits": []}])
        html = analysis_to_html(doc, result)
        self.assertIn("분석 당시 기업", html)
        self.assertNotIn("현재 기업", html)
        self.assertIn("부정 클래스 점수: 0.8500", html)
        self.assertNotIn("85%", html)
        self.assertIn("미보정 분류 점수", html)
        self.assertIn("확률이 아닙니다", html)
        self.assertIn("0.6000", html)

    def test_html_rules_and_demo_never_display_injected_model_score(self):
        for engine in ("rules", "demo"):
            result = analysis(engine=engine, metrics={"negative_score": .9876},
                              segments=[{"page": 1, "text": "근거", "negative_score": .9876, "rule_hits": []}])
            html = analysis_to_html(document(), result)
            self.assertIn("모델 점수 없음", html)
            self.assertNotIn("0.9876", html)

    def test_empty_export_has_headers_and_valid_json(self):
        self.assertEqual(json.loads(analyses_to_json([])), [])
        self.assertEqual(len(analyses_to_csv([]).decode("utf-8-sig").splitlines()), 1)


if __name__ == "__main__":
    unittest.main()
