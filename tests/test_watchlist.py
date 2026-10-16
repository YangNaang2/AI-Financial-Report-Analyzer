import json
from pathlib import Path
import tempfile
import unittest

import consensus
import watchlist


def report(ticker, broker, day, target, **extra):
    return dict(dict(ticker=ticker, broker=broker, date=day, title=f"{broker} {day}", filename=f"{broker}-{day}.pdf",
                     company=None, target_price=target, opinion="BUY", report_price=None, page=1, source="",
                     collected_at=f"{day}T09:00:00+00:00"), **extra)


class WatchlistTests(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.path = Path(self.folder.name)

    def tearDown(self):
        self.folder.cleanup()

    def test_missing_file_is_empty_and_changes_persist(self):
        self.assertEqual(watchlist.load(self.path), dict(tickers=[], seen_at=None))
        watchlist.set_tickers(self.path, ["003490", "000100", "003490"])
        watchlist.mark_seen(self.path, "2026-10-16T09:00:00+00:00")
        self.assertEqual(watchlist.load(self.path), dict(tickers=["003490", "000100"], seen_at="2026-10-16T09:00:00+00:00"))
        watchlist.set_tickers(self.path, ["000100"])
        self.assertEqual(watchlist.load(self.path)["seen_at"], "2026-10-16T09:00:00+00:00")

    def test_invalid_codes_and_files_are_rejected_without_overwriting(self):
        watchlist.set_tickers(self.path, ["003490"])
        with self.assertRaises(watchlist.WatchlistError):
            watchlist.set_tickers(self.path, ["003490", "대한항공"])
        self.assertEqual(watchlist.load(self.path)["tickers"], ["003490"])
        (self.path / watchlist.FILENAME).write_text("{broken", encoding="utf-8")
        with self.assertRaises(watchlist.WatchlistError):
            watchlist.load(self.path)

    def test_unseen_is_based_on_collection_time(self):
        event = dict(collected_at="2026-10-16T09:00:00+00:00")
        self.assertTrue(watchlist.is_unseen(event, None))
        self.assertTrue(watchlist.is_unseen(event, "2026-10-15T23:00:00+00:00"))
        self.assertFalse(watchlist.is_unseen(event, "2026-10-16T09:00:00+00:00"))


class ChangeEventTests(unittest.TestCase):
    def setUp(self):
        self.reports = consensus.annotate_changes([
            report("003490", "하나증권", "2026-09-01", 38000),
            report("003490", "하나증권", "2026-10-16", 41000, opinion="HOLD"),
            report("003490", "유안타증권", "2026-10-16", 40000, stated_previous=40000),
            report("000100", "IBK투자증권", "2026-10-16", 110000, stated_change="하향"),
            report("028050", "삼성증권", "2026-10-15", 73000, stated_change="신규"),
            report("035900", "iM증권", "2026-10-14", 50000),
        ])

    def test_feed_lists_revisions_opinion_changes_and_initiations_only(self):
        events = consensus.change_events(self.reports)
        summary = [(e["ticker"], e["kind"], e["detail"]) for e in events]
        self.assertEqual(sorted(summary), sorted([
            ("003490", "목표가 상향", "목표주가 38,000 → 41,000원"),
            ("003490", "투자의견 변경", "투자의견 BUY → HOLD"),
            ("000100", "목표가 하향", "목표주가 110,000원 (하향)"),
            ("028050", "커버리지 개시", "목표주가 73,000원"),
        ]))
        raised = next(e for e in events if e["kind"] == "목표가 상향")
        self.assertAlmostEqual(raised["pct"], (41000 / 38000 - 1) * 100)
        self.assertEqual([e["date"] for e in events], sorted((e["date"] for e in events), reverse=True))

    def test_revision_breadth_and_flow_count_upgrades_and_downgrades(self):
        breadth = consensus.revision_breadth(self.reports)
        self.assertEqual((breadth["ups"], breadth["downs"], breadth["share"]), (1, 1, 50.0))
        self.assertEqual(consensus.revision_breadth(self.reports, days=1)["share"], 50.0)
        self.assertIsNone(consensus.revision_breadth([])["share"])
        flow = consensus.revision_flow_figure(self.reports)
        bars = {trace.name: dict(zip(trace.x, trace.y)) for trace in flow.data}
        self.assertEqual(bars, {"▲ 목표가 상향": {"2026-10-16": 1}, "▼ 목표가 하향": {"2026-10-16": -1}})
        gauge = consensus.breadth_figure(breadth)
        self.assertEqual(gauge.data[0].value, 50.0)

    def test_feed_can_be_limited_to_watched_tickers(self):
        events = consensus.change_events(self.reports, tickers={"000100"})
        self.assertEqual([e["kind"] for e in events], ["목표가 하향"])


class AppFlowTests(unittest.TestCase):
    def test_watchlist_and_feed_in_app(self):
        from streamlit.testing.v1 import AppTest
        import os
        with tempfile.TemporaryDirectory() as folder:
            data = Path(folder)
            reports = data / "reports"
            reports.mkdir()
            rows = {}
            for index, (ticker, broker, day, text) in enumerate([
                ("003490", "하나증권", "2026-09-01", "목표주가 38,000원\n투자의견: 매수"),
                ("003490", "하나증권", "2026-10-16", "목표주가 41,000원\n투자의견: 매수"),
                ("000100", "IBK투자증권", "2026-10-16", "목표주가 (하향) 110,000원"),
            ]):
                name = f"r{index}.txt"
                (reports / name).write_text(text, encoding="utf-8")
                rows[str(index)] = dict(status="downloaded", filename=name, ticker=ticker, broker=broker,
                                        published_at=day, title=f"리포트 {index}", collected_at=f"{day}T09:00:00+00:00")
            (reports / "collection_manifest.json").write_text(json.dumps({"schema_version": 1, "reports": rows}), encoding="utf-8")
            environment = dict(REPORT_LENS_DB=str(data / "library.sqlite3"), REPORT_LENS_DATA_DIR=str(data),
                               REPORT_LENS_MODELS_DIR=str(data / "models"))
            previous = {key: os.environ.get(key) for key in environment}
            os.environ.update(environment)
            try:
                app = AppTest.from_file(str(Path(__file__).resolve().parents[1] / "app.py"), default_timeout=30).run()
                self.assertFalse(app.exception)
                text = " ".join(m.value for m in app.markdown)
                self.assertIn("38,000 → 41,000원", text)
                self.assertIn("110,000원 (하향)", text)
                self.assertIn("NEW", text)
                next(b for b in app.button if b.label == "모두 확인함으로 표시").click().run()
                self.assertNotIn("NEW", " ".join(m.value for m in app.markdown))
                next(b for b in app.button if b.label == "보기").click().run()
                self.assertFalse(app.exception)
                self.assertEqual(app.session_state["nav"], "종목 컨센서스")
                app.multiselect[0].select("000100").run()
                self.assertFalse(app.exception)
                self.assertEqual(watchlist.load(reports)["tickers"], ["000100"])
                picker = next(s for s in app.selectbox if s.label.startswith("종목"))
                self.assertTrue(picker.options[0].startswith("★ "), picker.options)
                self.assertIn("(000100)", picker.options[0])
            finally:
                for key, value in previous.items():
                    if value is None:
                        os.environ.pop(key, None)
                    else:
                        os.environ[key] = value


if __name__ == "__main__":
    unittest.main()
