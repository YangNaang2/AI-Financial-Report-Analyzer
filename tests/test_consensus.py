from datetime import timedelta
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
        self.assertEqual({row["title"]: row["reason"] for row in skipped}, {"기술분석": "목표주가 없음", "달러": "목표주가 단위 미확인"})

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

    def test_listing_values_take_precedence_and_rescue_unreadable_pdfs(self):
        manifest_path = self.directory / "collection_manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["reports"]["1"].update(company="대한항공", listing_target_price=41000, listing_opinion="Buy")
        manifest["reports"]["2"].update(listing_target_price=39000, listing_opinion="중립")
        manifest["reports"]["4"].update(company="한국비엔씨", listing_target_price=3000)
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
        reports, skipped = consensus.load_reports(self.directory, extract=self.extract)
        rows = {row["title"]: row for row in reports}
        agreed, differs, rescued = rows["상향"], rows["신규"], rows["기술분석"]
        self.assertEqual((agreed["target_origin"], agreed["company"], consensus.source_note(agreed)),
                         ("네이버 목록", "대한항공", "네이버 목록 · PDF 1페이지 일치"))
        self.assertEqual((differs["target_price"], differs["opinion"], consensus.source_note(differs)),
                         (39000, "HOLD", "네이버 목록 (PDF 40,000원과 다름, PDF 1페이지)"))
        self.assertEqual((rescued["target_price"], consensus.source_note(rescued)), (3000, "네이버 목록 (PDF에서 확인 못 함)"))
        self.assertEqual(consensus.source_note(rows["이전"]), "PDF 1페이지")
        self.assertEqual([row["title"] for row in skipped], ["달러"])

    def test_gap_labels_of_nearby_reports_do_not_overlap(self):
        def item(day, base, target):
            return dict(row=dict(date=day, target_price=target), base=base)

        items = [item("2026-10-15", 61000, 112000), item("2026-10-16", 64000, 120000), item("2026-06-01", 90000, 100000)]
        placed = consensus._label_positions(items, timedelta(days=120), 60000)
        self.assertEqual([side for side, _ in placed], ["left", "right", "right"])
        self.assertEqual(placed[0][1], (61000 + 112000) / 2)
        three = items[:2] + [item("2026-10-16", 64000, 118000)]
        (_, first), (_, second), (_, third) = consensus._label_positions(three, timedelta(days=120), 60000)
        self.assertGreaterEqual(abs(second - third), 60000 * 0.12)

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
        self.assertEqual({trace.name for trace in target.data if trace.showlegend is not False}, {"상향", "비교 없음"})
        # One current-price line; each broker and the mean get a line from that price to the target.
        self.assertEqual(len(target.layout.shapes), 1)
        gaps = [trace for trace in target.data if trace.mode == "lines"]
        self.assertEqual(sorted(tuple(trace.x) for trace in gaps), [(29450, 40000), (29450, 40500), (29450, 41000)])
        mean = next(trace for trace in target.data if trace.marker.symbol == "diamond")
        self.assertEqual((mean.y[0], mean.text[0]), ("<b>평균 목표 (2곳)</b>", "<b>40,500원 · +37.5%</b>"))
        bare_target = consensus.target_figure(latest)
        self.assertFalse([trace for trace in bare_target.data if trace.mode == "lines"])
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
        top = {trace.name for trace in priced.data if trace.yaxis == "y" and trace.name}
        self.assertEqual(top, {"종가", "리포트 발간"})
        self.assertEqual({annotation.text for annotation in priced.layout.annotations if "원</b>" not in annotation.text},
                         {"주가 추이", "목표주가와 발간일 종가"})
        self.assertIn("<b>+11,550원</b><br>+39.2%", {annotation.text for annotation in bare.layout.annotations})

    def test_won_units_are_converted_explicitly(self):
        self.assertEqual([consensus._won(value) for value in ("41000 원", "41 천원", "120 USD", None)], [41000, 41000, None, None])


if __name__ == "__main__":
    unittest.main()
