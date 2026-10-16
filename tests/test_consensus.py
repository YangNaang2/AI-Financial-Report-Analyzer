import json
from pathlib import Path
import tempfile
import unittest

import consensus
from data_processor import DocumentError, document_from_text


def collect(directory, reports):
    rows = {}
    for index, (listing, _) in enumerate(reports):
        filename = f"r{index}.pdf"
        (directory / filename).write_bytes(b"%PDF-1.4")
        rows[str(index)] = dict(listing, status="downloaded", filename=filename, sha256=f"hash{index}")
    (directory / "collection_manifest.json").write_text(json.dumps({"schema_version": 1, "reports": rows}), encoding="utf-8")
    return {f"r{index}.pdf": text for index, (_, text) in enumerate(reports)}


def cover(target, price="29,450원"):
    return f"대한항공(003490)\n투자의견: 매수\n목표주가: {target}\n현재주가: {price}\n"


class ConsensusTests(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.directory = Path(self.folder.name)
        self.texts = collect(self.directory, [
            (dict(ticker="003490", broker="하나증권", published_at="2026-09-01", title="이전"), cover("38,000원")),
            (dict(ticker="003490", broker="하나증권", published_at="2026-10-16", title="상향"), cover("41,000원")),
            (dict(ticker="003490", broker="유안타증권", published_at="2026-10-16", title="신규"), cover("40,000원")),
            (dict(ticker="003490", broker="신한투자증권", published_at="2026-05-01", title="오래됨"), cover("50,000원")),
            (dict(ticker="480370", broker="NICE평가정보", published_at="2026-10-16", title="기술분석"), "기술분석보고서\n현재주가: 2,115원\n"),
            (dict(ticker="000150", broker="교보증권", published_at="2026-10-15", title="달러"), cover("USD 120")),
        ])
        self.calls = []

    def tearDown(self):
        self.folder.cleanup()

    def extract(self, path):
        self.calls.append(path.name)
        return document_from_text(self.texts[path.name], path.name)

    def test_reports_changes_and_skips_reports_without_won_targets(self):
        reports, skipped = consensus.load_reports(self.directory, extract=self.extract)
        self.assertEqual({row["title"]: row["change"] for row in reports},
                         {"이전": "신규", "상향": "상향", "신규": "신규", "오래됨": "신규"})
        raised = next(row for row in reports if row["title"] == "상향")
        self.assertEqual((raised["target_price"], raised["previous_target"], raised["opinion"], raised["page"]), (41000, 38000, "BUY", 1))
        self.assertEqual({row["title"]: row["reason"] for row in skipped}, {"기술분석": "목표주가 미추출", "달러": "목표주가 단위 미확인"})

    def test_stated_previous_target_and_revision_are_used_without_history(self):
        self.texts["r2.pdf"] = cover("40,000원") + "직전 목표주가 43,000원\n"
        self.texts["r3.pdf"] = "대한항공(003490)\n목표주가 (하향) 50,000원\n"
        reports, _ = consensus.load_reports(self.directory, extract=self.extract)
        rows = {row["title"]: row for row in reports}
        self.assertEqual((rows["신규"]["change"], rows["신규"]["previous_target"], rows["신규"]["change_basis"]),
                         ("하향", 43000, "원문 직전 목표주가"))
        self.assertEqual((rows["오래됨"]["change"], rows["오래됨"]["previous_target"], rows["오래됨"]["change_basis"]),
                         ("하향", None, "원문 표기"))
        self.assertEqual(rows["상향"]["change_basis"], "수집 이력")

    def test_extraction_is_cached_by_file_hash(self):
        consensus.load_reports(self.directory, extract=self.extract)
        first = len(self.calls)
        consensus.load_reports(self.directory, extract=self.extract)
        self.assertEqual((first, len(self.calls)), (6, 6))

    def test_extraction_errors_are_skipped_with_reason(self):
        def broken(path):
            raise DocumentError("PDF가 손상되었거나 지원하지 않는 형식입니다.")

        reports, skipped = consensus.load_reports(self.directory, extract=broken)
        self.assertEqual(reports, [])
        self.assertTrue(all("손상" in row["reason"] for row in skipped))

    def test_summary_uses_latest_report_per_broker_within_window(self):
        reports, _ = consensus.load_reports(self.directory, extract=self.extract)
        (row,) = consensus.summarize(reports, as_of="2026-10-16", window_days=90)
        self.assertEqual((row["brokers"], row["mean_target"], row["low"], row["high"]), (2, 40500, 40000, 41000))
        self.assertEqual((row["price"], row["price_source"], row["ups"]), (29450, "리포트 기재가", 1))
        self.assertAlmostEqual(row["upside"], (40500 / 29450 - 1) * 100)
        (wide,) = consensus.summarize(reports, as_of="2026-10-16", window_days=365)
        self.assertEqual(wide["brokers"], 3)

    def test_latest_close_takes_precedence_over_report_price(self):
        reports, _ = consensus.load_reports(self.directory, extract=self.extract)
        prices = {"003490": [("2026-10-15", 29000.0), ("2026-10-16", 30000.0)]}
        price = consensus.current_price(reports, "003490", prices)
        self.assertEqual(price, dict(price=30000.0, date="2026-10-16", source="종가"))

    def test_figures_build_with_and_without_prices(self):
        reports, _ = consensus.load_reports(self.directory, extract=self.extract)
        latest = consensus.latest_by_broker(reports, "003490", "2026-10-16")
        target = consensus.target_figure(latest, consensus.current_price(reports, "003490"))
        self.assertEqual({trace.name for trace in target.data}, {"상향", "신규"})
        self.assertEqual(len(target.layout.shapes), 2)
        bare = consensus.history_figure(reports, "003490")
        self.assertIn("리포트 기재 주가", {trace.name for trace in bare.data})
        priced = consensus.history_figure(reports, "003490", [("2026-10-15", 30000.0), ("2026-10-16", 31000.0)])
        self.assertIn("종가", {trace.name for trace in priced.data})
        self.assertEqual(priced.layout.xaxis.type, "date")
        self.assertEqual(priced.layout.xaxis.tickformat, "%y.%m.%d")
        # Reports before the first close have no base; 10-16 reports use that day's close.
        labels = sorted(annotation.text for annotation in priced.layout.annotations if "원</b>" in annotation.text)
        self.assertEqual(labels, ["<b>+10,000원</b><br>+32.3%", "<b>+9,000원</b><br>+29.0%"])
        # The top panel holds only the close and publication markers, so it scales to the price.
        top = {trace.name for trace in priced.data if trace.yaxis == "y"}
        self.assertEqual(top, {"종가", "리포트 발간"})
        self.assertEqual({annotation.text for annotation in priced.layout.annotations if "원</b>" not in annotation.text},
                         {"주가 추이", "목표주가와 발간일 종가"})
        self.assertIn("<b>+11,550원</b><br>+39.2%", {annotation.text for annotation in bare.layout.annotations})

    def test_won_units_are_converted_explicitly(self):
        self.assertEqual([consensus._won(value) for value in ("41000 원", "41 천원", "120 USD", None)], [41000, 41000, None, None])


if __name__ == "__main__":
    unittest.main()
