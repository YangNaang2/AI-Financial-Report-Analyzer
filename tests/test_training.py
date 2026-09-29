from copy import deepcopy
from datetime import date, datetime, timedelta
import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import pandas as pd

from data_processor import document_from_text
from inference import analyze_document, list_models
from main import extract_metadata_from_raw_text, main, prepare_dataset
from train import TrainingError, temporal_split, train_model


def records(count=20):
    result = []
    for index in range(count):
        day = date(2020, 1, 1) + timedelta(days=60 * index)
        result.append({"id": f"doc-{index}",
                       "text": ("매출 증가 실적 성장 신규 수주 긍정 기대 " if index % 2 == 0 else
                                "매출 감소 실적 부진 손실 비용 위험 확대 ") + f"가상 문서 {index}",
                       "report_date": day.isoformat(), "label_end_date": (day + timedelta(days=30)).isoformat(),
                       "label_id": index % 2})
    return result


def records_with_policy():
    result = records()
    for row in result:
        row["label_info"] = {"status": "labeled", "label_id": row["label_id"],
                             "window_days": 30, "window_kind": "calendar_days", "window_anchor": "report_date",
                             "entry_policy": "next_session", "threshold": -5,
                             "source": "FinanceDataReader.Close", "adjustment_policy": "unknown",
                             "adjustment_note": "분할·배당 조정 여부 미확인", "policy_note": "발행 후 첫 관측 거래일",
                             "start_date": (date.fromisoformat(row["report_date"]) + timedelta(days=1)).isoformat(),
                             "end_date": row["label_end_date"], "as_of": "2024-01-01",
                             "base_price": 100, "future_price": 90 if row["label_id"] else 110,
                             "return_pct": -10 if row["label_id"] else 10}
    return result


class TrainingTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name)

    def test_temporal_split_precedes_chunks_and_purges_touching_label_horizons(self):
        rows = records()
        original_groups, _ = temporal_split(rows)
        rows[11]["label_end_date"] = original_groups["validation"][0]["report_date"]
        rows[15]["label_end_date"] = original_groups["test"][0]["report_date"]
        groups, audit = temporal_split(rows)
        self.assertEqual(audit["purged_ids"], {"train": ["doc-11"], "validation": ["doc-15"]})
        self.assertLess(max(row["label_end_date"] for row in groups["train"]), min(row["report_date"] for row in groups["validation"]))
        self.assertLess(max(row["label_end_date"] for row in groups["validation"]), min(row["report_date"] for row in groups["test"]))
        identifiers = [{row["id"] for row in group} for group in groups.values()]
        self.assertEqual(len(set.union(*identifiers)), sum(map(len, identifiers)))

    def test_same_day_documents_stay_together_and_duplicates_are_input_order_independent(self):
        rows = records()
        rows.append(dict(rows[0], id="duplicate-later", report_date="2024-01-01", label_end_date="2024-02-01"))
        rows.append(dict(rows[2], id="same-day-new", text=rows[2]["text"] + "추가 고유 내용"))
        first, audit = temporal_split(rows)
        second, reverse_audit = temporal_split(list(reversed(rows)))
        self.assertEqual(first, second)
        self.assertEqual(audit, reverse_audit)
        self.assertIn("duplicate-later", audit["duplicate_ids"])
        self.assertTrue(any({"doc-2", "same-day-new"} <= {row["id"] for row in group} for group in first.values()))

    def test_conflicting_duplicate_labels_and_future_labels_are_rejected(self):
        rows = records()
        rows.append(dict(rows[0], id="conflict", label_id=1))
        with self.assertRaisesRegex(TrainingError, "서로 다른 라벨"):
            temporal_split(rows)
        for key, value in [("report_date", "2099-01-01"), ("label_end_date", "2099-01-01"),
                           ("report_date", datetime(2020, 1, 1))]:
            rows = records()
            rows[0][key] = value
            with self.subTest(key=key, value=value), self.assertRaises(TrainingError):
                temporal_split(rows)

    def test_insufficient_data_classes_and_undated_legacy_inputs_never_create_models(self):
        for rows in [[], records(2), [dict(row, label_id=0) for row in records()]]:
            with self.subTest(count=len(rows)), self.assertRaises(TrainingError):
                train_model(records=rows, output_dir=self.path)
        with self.assertRaisesRegex(TrainingError, "시간 누출"):
            train_model(["좋은 실적입니다", "매우 나쁩니다"], [0, 1], output_dir=self.path)
        with self.assertRaises(TrainingError):
            train_model(records=records(), max_tokens="bad", output_dir=self.path)
        self.assertEqual(list(self.path.iterdir()), [])

    def test_pending_records_are_excluded_without_becoming_class_zero(self):
        rows = records() + [dict(id="pending", text="아직 관측이 완료되지 않은 문서", status="pending", label_id=None)]
        groups, audit = temporal_split(rows)
        self.assertEqual(audit["unavailable_ids"], ["pending"])
        self.assertEqual(sum(map(len, groups.values())), 20)

    def test_real_cpu_baseline_manifest_hashes_and_inference_are_compatible(self):
        manifest = train_model(records=records(), output_dir=self.path, max_tokens=32, overlap=4, seed=7)
        model_dir = Path(manifest["model_dir"])
        self.assertTrue(manifest["completed"])
        self.assertEqual(manifest["evaluation_status"], "insufficient")
        self.assertTrue(manifest["evaluation_reasons"])
        self.assertEqual(manifest["metrics"]["test"]["count"], 4)
        self.assertEqual(manifest["metrics"]["majority_baseline"]["test"]["count"], 4)
        for filename, digest in manifest["files"].items():
            self.assertEqual(hashlib.sha256((model_dir / filename).read_bytes()).hexdigest(), digest)
        self.assertEqual(list_models(self.path)[0]["status"], "ready")
        negative = analyze_document(document_from_text("매출 감소 실적 부진 손실 비용 위험 확대"), engine="model", model_path=model_dir)
        positive = analyze_document(document_from_text("매출 증가 실적 성장 신규 수주 긍정 기대"), engine="model", model_path=model_dir)
        self.assertGreater(negative["metrics"]["negative_score"], positive["metrics"]["negative_score"])
        self.assertEqual(negative["settings"]["threshold"], manifest["threshold"])
        self.assertEqual(negative["status"], "ready")
        self.assertFalse(list(model_dir.glob("*.pkl")))
        self.assertFalse(list(self.path.glob(".*.tmp")))

    def test_seed_fingerprint_and_weights_reproduce_while_test_labels_cannot_select_threshold(self):
        rows = records()
        first = train_model(records=rows, output_dir=self.path, seed=11)
        second = train_model(records=list(reversed(rows)), output_dir=self.path, seed=11)
        self.assertEqual(first["fingerprint"], second["fingerprint"])
        self.assertEqual(first["files"], second["files"])
        self.assertEqual(first["threshold"], second["threshold"])
        modified = deepcopy(rows)
        for row in modified[16:]:
            row["label_id"] = 1 - row["label_id"]
        third = train_model(records=modified, output_dir=self.path, seed=11)
        self.assertEqual(first["files"], third["files"])
        self.assertEqual(first["threshold"], third["threshold"])
        self.assertNotEqual(first["metrics"]["test"]["accuracy"], third["metrics"]["test"]["accuracy"])

    def test_heldout_only_words_never_enter_training_vocabulary(self):
        rows = records()
        for row in rows[12:]:
            row["text"] += " 검증시험전용비밀토큰"
        result = train_model(records=rows, output_dir=self.path)
        saved = json.loads((Path(result["model_dir"]) / "baseline_model.json").read_text())
        self.assertNotIn("검증시험전용비밀토큰", saved["vocabulary"])

    def test_single_class_heldout_metrics_use_na_and_do_not_invent_pr_auc(self):
        rows = records()
        for row in rows[12:]:
            row["label_id"] = 0
        result = train_model(records=rows, output_dir=self.path)
        self.assertIsNone(result["metrics"]["validation"]["pr_auc"])
        self.assertIsNone(result["metrics"]["test"]["pr_auc"])
        self.assertEqual(result["threshold"], 0.5)
        self.assertEqual(result["evaluation_status"], "insufficient")

    def test_failed_training_removes_incomplete_artifacts(self):
        with mock.patch("train._train_baseline", side_effect=RuntimeError("simulated disk/training failure")):
            with self.assertRaises(RuntimeError):
                train_model(records=records(), output_dir=self.path)
        self.assertEqual(list(self.path.iterdir()), [])

    def test_manifest_preserves_return_policy_sources_and_observations_separate_from_classifier_threshold(self):
        rows = records_with_policy()
        rows[-1]["label_info"]["source"] = "reviewed_exchange_close"
        rows[-1]["label_info"]["adjustment_policy"] = "split_adjusted"
        result = train_model(records=rows, output_dir=self.path)
        policy = result["label_policy"]
        self.assertEqual(policy["status"], "complete")
        self.assertEqual(policy["definition"], {"window_days": 30, "window_kind": "calendar_days",
                                              "window_anchor": "report_date", "entry_policy": "next_session",
                                              "return_threshold_pct": -5.0})
        self.assertEqual(policy["known_record_count"], 20)
        self.assertEqual(policy["unknown_record_count"], 0)
        self.assertEqual(policy["sources"], ["FinanceDataReader.Close", "reviewed_exchange_close"])
        self.assertEqual(policy["adjustment_policies"], ["split_adjusted", "unknown"])
        self.assertEqual(policy["provenance"][-1]["record_id"], "doc-19")
        self.assertEqual(policy["provenance"][-1]["as_of"], "2024-01-01")
        self.assertEqual(policy["provenance"][-1]["future_price"], 90)
        self.assertEqual(policy["provenance"][-1]["adjustment_note"], "분할·배당 조정 여부 미확인")
        self.assertEqual(result["classifier_threshold"], result["threshold"])
        self.assertNotEqual(result["classifier_threshold"], policy["definition"]["return_threshold_pct"])
        stored = json.loads((Path(result["model_dir"]) / "model_manifest.json").read_text())
        self.assertEqual(stored["label_policy"], policy)

    def test_mixing_distinct_known_return_definitions_is_rejected_before_training(self):
        for field, value in (("window_days", 60), ("window_kind", "trading_sessions"),
                             ("window_anchor", "entry_date"), ("entry_policy", "report_day"), ("threshold", -10)):
            rows = records_with_policy()
            rows[-1]["label_info"][field] = value
            with self.subTest(field=field), mock.patch("train._train_baseline") as backend:
                with self.assertRaisesRegex(TrainingError, "서로 다른 수익률 라벨 정책"):
                    train_model(records=rows, output_dir=self.path)
                backend.assert_not_called()
        self.assertEqual(list(self.path.iterdir()), [])

    def test_unknown_label_definitions_are_explicit_and_partial_provenance_is_incomplete(self):
        unknown = train_model(records=records(), output_dir=self.path)
        self.assertEqual(unknown["label_policy"]["status"], "unknown")
        self.assertIsNone(unknown["label_policy"]["definition"])
        self.assertEqual(unknown["label_policy"]["unknown_record_count"], 20)
        self.assertEqual(unknown["label_policy"]["sources"], ["unknown"])
        self.assertIn("라벨 정의", " ".join(unknown["evaluation_reasons"]))
        rows = records_with_policy()
        rows[-1].pop("label_info")
        partial = train_model(records=rows, output_dir=self.path)
        self.assertEqual(partial["label_policy"]["status"], "incomplete")
        self.assertEqual(partial["label_policy"]["unknown_record_ids"], ["doc-19"])
        self.assertEqual(partial["label_policy"]["known_record_count"], 19)
        self.assertEqual(partial["evaluation_status"], "insufficient")


class PipelineTests(unittest.TestCase):
    def test_metadata_accepts_year_2027_english_dates_and_refuses_ambiguous_values(self):
        self.assertEqual(extract_metadata_from_raw_text("가상기업 (005930)\n발간일: 2027.05.18\n매출 증가"), ("005930", "2027.05.18"))
        self.assertEqual(extract_metadata_from_raw_text("Ticker: 005930\nMay 18, 2027"), ("005930", "2027.05.18"))
        self.assertEqual(extract_metadata_from_raw_text("가상기업 (005930)\n2027.05.18\n2027.05.19")[1], None)
        self.assertEqual(extract_metadata_from_raw_text("가격 123456원, 매출 234567원"), (None, None))

    def test_prepare_retains_pending_and_unavailable_records_and_does_not_train(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            (path / "ready.txt").write_text("가상기업 (005930)\n발행일: 2024.01.01\n매출 증가 예상", encoding="utf-8")
            (path / "pending.txt").write_text("가상기업 (005930)\n발행일: 2027.01.01\n미래 발행 문서", encoding="utf-8")
            (path / "unknown.txt").write_text("코드와 날짜가 없는 문서는 라벨을 만들지 않습니다.", encoding="utf-8")
            loader = mock.Mock(return_value=pd.DataFrame({"Close": [100, 90]}, index=pd.to_datetime(["2024-01-02", "2024-02-01"])))
            result = prepare_dataset(path, as_of="2024-03-01", price_loader=loader, output_path=path / "dataset.json")
            self.assertEqual(result["counts"]["labeled"], 1)
            self.assertEqual(result["counts"]["pending"], 1)
            self.assertEqual(result["counts"]["unavailable"], 1)
            self.assertEqual(loader.call_count, 1)
            self.assertEqual(json.loads((path / "dataset.json").read_text())["records"], result["records"])
            self.assertFalse((path / "models").exists())

    def test_empty_prepare_and_cli_dry_run_do_not_train_or_contact_network(self):
        with tempfile.TemporaryDirectory() as directory:
            self.assertEqual(prepare_dataset(directory)["status"], "no_labeled_data")
            with mock.patch("crawler.requests.get", side_effect=AssertionError("dry run network")), mock.patch("sys.stdout", new_callable=io.StringIO) as output:
                self.assertEqual(main(["collect", "--dry-run", "--save-dir", directory]), 0)
            self.assertEqual(json.loads(output.getvalue())["status"], "dry_run")


if __name__ == "__main__":
    unittest.main()
