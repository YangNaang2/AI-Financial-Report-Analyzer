"""Cover formats seen in collected Naver research PDFs (October 2026)."""
import unittest

from data_processor import document_from_text, opinion_class


def metadata(text):
    return {key: value for key, value in document_from_text(text)["metadata"].items() if value}


class CoverMetadataTests(unittest.TestCase):
    def test_hana_cover_with_standalone_opinion_and_code(self):
        found = metadata("2026년 10월 16일 I 기업분석\n대한항공\nBUY (유지) (003490)\n목표주가(12M) 41,000원\n현재주가(10.15) 29,450원")
        self.assertEqual(found, dict(ticker="003490", report_date="2026-10-16", opinion="BUY",
                                     target_price="41000 원", current_price="29450 원"))

    def test_previous_target_is_not_a_conflicting_candidate(self):
        found = metadata("Company Report 2026.10.16\n(047040)\nBUY\n목표주가 24,000원 (M)\n"
                         "컨센서스에 부합할 것으로 전망한다. 직전 목표주가 22,000원\n현재주가 (10/15) 14,670원")
        self.assertEqual((found["target_price"], found["previous_target_price"], found["target_change"]),
                         ("24000 원", "22000 원", "유지"))

    def test_stated_revision_and_tp_label(self):
        self.assertEqual(metadata("목표주가 (하향) 110,000원")["target_change"], "하향")
        found = metadata("Kyobo Company Analysis\n462870\nOct 15 2026\nBUY\nTP 42,000원 유지 3Q26 매출은\n현재가(10/14) 30,650 원")
        self.assertEqual((found["ticker"], found["report_date"], found["target_price"], found["target_change"]),
                         ("462870", "2026-10-15", "42000 원", "유지"))

    def test_kiwoom_price_label_and_maintain_opinion(self):
        found = metadata(" 실적 Preview 2026. 10. 16\n한국전력\nBUY(Maintain)\n(015760)\n목표주가: 41,000원\n주가(10/15): 28,450원")
        self.assertEqual((found["opinion"], found["current_price"]), ("BUY", "28450 원"))
        # "목표주가(" must not be read as the current-price label "주가(".
        self.assertNotIn("current_price", metadata("목표주가(12M) 41,000원"))

    def test_equivalent_opinions_and_repeated_targets_do_not_conflict(self):
        found = metadata("교보증권\nBUY\n목표주가 42,000원\n투자의견 Buy & 목표주가는 42,000원 유지\n목표주가 42,000 42,000 0.0%")
        self.assertEqual((found["opinion"], found["target_price"]), ("BUY", "42000 원"))
        conflict = document_from_text("투자의견: 매수\n투자의견: 중립")
        self.assertIsNone(conflict["metadata"]["opinion"])
        self.assertEqual([opinion_class(value) for value in ("Strong Buy", "매수", "중립", "Underperform", "?")],
                         ["BUY", "BUY", "HOLD", "SELL", None])

    def test_body_words_are_not_read_as_opinions_or_brokers(self):
        found = metadata("매수세가 유입되며 주가가 반등했다\n주가수익률 (%) 1개월 3개월\niM증권 리서치센터")
        self.assertEqual(found, dict(broker="iM증권"))


if __name__ == "__main__":
    unittest.main()
