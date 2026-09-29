from datetime import date
import unittest
from unittest import mock

import pandas as pd

from labeler import calculate_post_report_return


def frame(dates, prices):
    return pd.DataFrame({"Close": prices}, index=pd.to_datetime(dates))


class LabelerTests(unittest.TestCase):
    def test_next_session_and_holiday_exit_are_observed_without_future_requests(self):
        loader = mock.Mock(return_value=frame(["2024-05-03", "2024-05-07", "2024-06-03", "2024-07-01"], [10, 100, 94, 200]))
        result = calculate_post_report_return("005930", "2024.05.03", as_of="2024-06-03", price_loader=loader)
        self.assertEqual(result["status"], "labeled")
        self.assertEqual(result["start_date"], "2024-05-07")
        self.assertEqual(result["target_date"], "2024-06-02")
        self.assertEqual(result["end_date"], "2024-06-03")
        self.assertEqual(result["return_pct"], -6)
        self.assertEqual(result["수익률"], -6)
        self.assertEqual(result["label_id"], 1)
        self.assertEqual(result["window_anchor"], "report_date")
        self.assertEqual(result["adjustment_policy"], "unknown")
        self.assertEqual(loader.call_args.args[-1], "2024-06-03")

    def test_unmatured_or_future_report_never_requests_a_price(self):
        loader = mock.Mock(side_effect=AssertionError("future query"))
        for published in ("2024-05-15", "2027-01-01"):
            result = calculate_post_report_return("005930", published, as_of="2024-05-31", price_loader=loader)
            self.assertEqual(result["status"], "pending")
            self.assertIsNone(result["label_id"])
            self.assertIsNone(result["return_pct"])
        loader.assert_not_called()

    def test_extra_provider_future_rows_never_create_a_label(self):
        loader = mock.Mock(return_value=frame(["2024-05-07", "2024-06-03"], [100, 80]))
        result = calculate_post_report_return("005930", "2024-05-03", as_of="2024-06-02", price_loader=loader)
        self.assertEqual(result["status"], "pending")
        self.assertIsNone(result["label_id"])
        self.assertIsNone(result["end_date"])

    def test_report_day_policy_is_explicit_and_changes_entry(self):
        loader = mock.Mock(return_value=frame(["2024-05-03", "2024-05-07", "2024-06-03"], [90, 100, 95]))
        next_day = calculate_post_report_return("005930", "2024-05-03", as_of="2024-06-03", price_loader=loader)
        same_day = calculate_post_report_return("005930", "2024-05-03", entry_policy="report_day", as_of="2024-06-03", price_loader=loader)
        self.assertEqual(next_day["label_id"], 1)
        self.assertEqual(same_day["label_id"], 0)
        self.assertEqual(same_day["start_date"], "2024-05-03")
        self.assertIn("발행 시각 미반영", same_day["policy_note"])

    def test_inclusive_threshold_uses_unrounded_return(self):
        loader = mock.Mock(return_value=frame(["2024-01-02", "2024-02-01"], [100, 105]))
        result = calculate_post_report_return("005930", "2024-01-01", threshold=5, as_of="2024-02-02", price_loader=loader)
        self.assertEqual(result["label_id"], 1)
        loader.return_value = frame(["2024-01-02", "2024-02-01"], [100, 95.001])
        result = calculate_post_report_return("005930", "2024-01-01", as_of="2024-02-02", price_loader=loader)
        self.assertEqual(result["수익률"], -5)
        self.assertEqual(result["label_id"], 0)  # Rounded display must not determine the label.

    def test_prices_sort_and_skip_nonfinite_zero_or_negative_closes(self):
        loader = mock.Mock(return_value=frame(["2024-02-02", "2024-01-03", "2024-01-02", "2024-02-01"], [80, 100, 0, float("nan")]))
        result = calculate_post_report_return("005930", "2024-01-01", as_of="2024-02-03", price_loader=loader)
        self.assertEqual(result["start_date"], "2024-01-03")
        self.assertEqual(result["end_date"], "2024-02-02")
        self.assertEqual(result["return_pct"], -20)

    def test_provider_failures_are_unavailable_not_neutral(self):
        for response in (None, pd.DataFrame(), pd.DataFrame({"Open": [100]})):
            result = calculate_post_report_return("005930", "2024-01-01", as_of="2024-03-01", price_loader=lambda *args: response)
            self.assertEqual(result["status"], "unavailable")
            self.assertIsNone(result["label_id"])
        loader = mock.Mock(side_effect=TimeoutError("offline"))
        result = calculate_post_report_return("005930", "2024-01-01", as_of="2024-03-01", price_loader=loader)
        self.assertEqual(result["status"], "unavailable")
        self.assertIn("TimeoutError", result["reason"])

    def test_invalid_configuration_is_rejected_before_network(self):
        for kwargs in ({"ticker": "5930"}, {"report_date_str": "2024-02-30"}, {"window_days": 0},
                       {"window_days": True}, {"entry_policy": "same_price"}, {"threshold": float("nan")}):
            arguments = dict(ticker="005930", report_date_str="2024-01-01", as_of=date(2024, 3, 1))
            arguments.update(kwargs)
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                calculate_post_report_return(**arguments)


if __name__ == "__main__":
    unittest.main()
