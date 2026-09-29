from copy import deepcopy
from concurrent.futures import ThreadPoolExecutor
from hashlib import sha256
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

from data_processor import document_from_text
import inference


def write_baseline(directory, name="model", *, created="2024-01-02T00:00:00+00:00", coefficient=2.0, threshold=0.7):
    path = Path(directory) / name
    path.mkdir()
    payload = dict(vocabulary={"실적": 0, "부진": 1, "성장": 2, "실적 부진": 3}, idf=[1.0] * 4,
                   coef=[1.0, coefficient, -1.0, 3.0], intercept=-0.2, lowercase=True,
                   token_pattern=r"(?u)\b\w\w+\b", ngram_range=[1, 2], sublinear_tf=True, norm="l2")
    weights = path / "baseline_model.json"
    weights.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    manifest = dict(schema_version=1, status="complete", completed=True, model_type="baseline", model_id=name,
                    created_at=created, negative_id=1, id2label={"0": "non_negative", "1": "negative"},
                    max_tokens=256, overlap=32, preprocessing_version="2", threshold=threshold,
                    files={"baseline_model.json": sha256(weights.read_bytes()).hexdigest()})
    (path / "model_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return path


class InferenceTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        inference.clear_model_cache()
        self.addCleanup(inference.clear_model_cache)

    def test_preview_has_real_source_matches_but_never_model_scores(self):
        document = document_from_text("매출이 성장했습니다.\n\n실적 부진 우려로 추정치를 하향했습니다.")
        original = deepcopy(document)
        result = inference.analyze_document(document)
        self.assertEqual(result["engine"], "rules")
        self.assertIsNone(result["metrics"]["negative_score"])
        self.assertEqual(result["metrics"]["total_chars"], result["metrics"]["analyzed_chars"])
        self.assertEqual(result["status"], "ready")
        hits = []
        for segment in result["segments"]:
            self.assertIsNone(segment["negative_score"])
            for hit in segment["rule_hits"]:
                self.assertEqual(segment["text"][hit["start"]:hit["end"]], hit["keyword"])
                hits.append(hit["keyword"])
        self.assertIn("하향", hits)
        self.assertTrue(result["summary"])
        for summary in result["summary"]:
            self.assertIn(summary["text"], document["pages"][summary["page"] - 1]["text"])
        self.assertEqual(document, original)
        json.dumps(result)

    def test_preview_import_does_not_require_or_import_ml_frameworks(self):
        command = "import sys; import inference; assert 'torch' not in sys.modules; assert 'transformers' not in sys.modules; from data_processor import document_from_text; assert inference.analyze_document(document_from_text('실적 부진'))['metrics']['negative_score'] is None"
        completed = subprocess.run([sys.executable, "-c", command], cwd=Path(inference.__file__).parent,
                                   capture_output=True, text=True, timeout=15)
        self.assertEqual(completed.returncode, 0, completed.stderr)

    def test_baseline_prediction_matches_documented_tfidf_logistic_math(self):
        path = write_baseline(self.directory.name)
        result = inference.analyze_document(document_from_text("실적 부진"), engine="model", model_path=path)
        expected = 1 / (1 + math.exp(-((1 + 2 + 3) / math.sqrt(3) - 0.2)))
        self.assertAlmostEqual(result["metrics"]["negative_score"], expected, places=12)
        self.assertEqual(result["settings"]["threshold"], 0.7)
        self.assertEqual(result["settings"]["aggregation"], "segment_mean")
        self.assertEqual(result["model_id"], "model")
        self.assertEqual(len(result["model_fingerprint"]), 64)

    def test_multiple_segments_use_mean_and_report_partial_coverage(self):
        path = write_baseline(self.directory.name)
        text = " ".join(["실적 부진"] * 50)
        result = inference.analyze_document(document_from_text(text), engine="model", model_path=path,
                                            settings=dict(max_tokens=8, overlap=2, max_segments=2))
        scores = [segment["negative_score"] for segment in result["segments"]]
        self.assertEqual(len(scores), 2)
        self.assertAlmostEqual(result["metrics"]["negative_score"], sum(scores) / 2)
        self.assertEqual(result["status"], "partial")
        self.assertEqual(result["metrics"]["unprocessed_pages"], [1])
        self.assertTrue(any("학습 때의 구간" in warning for warning in result["warnings"]))

    def test_model_selection_uses_completed_timestamp_not_mtime(self):
        older = write_baseline(self.directory.name, "old", created="2024-01-01T00:00:00Z")
        newer = write_baseline(self.directory.name, "new", created="2024-01-02T00:00:00Z")
        os.utime(older, (2_000_000_000, 2_000_000_000))
        checkpoint = Path(self.directory.name) / "checkpoint-incomplete"
        checkpoint.mkdir()
        (checkpoint / "config.json").write_text("{}")
        models = inference.list_models(self.directory.name)
        self.assertEqual(inference.get_latest_model_path(self.directory.name), str(newer.resolve()))
        self.assertEqual(next(model for model in models if model["id"] == "checkpoint-incomplete")["status"], "invalid")

    def test_unfinished_or_ambiguous_label_models_are_invalid(self):
        path = write_baseline(self.directory.name)
        manifest_path = path / "model_manifest.json"
        manifest = json.loads(manifest_path.read_text())
        for change in (dict(completed=False), dict(id2label={"0": "LABEL_0", "1": "LABEL_1"}), dict(negative_id=None), dict(preprocessing_version="1")):
            with self.subTest(change=change):
                manifest_path.write_text(json.dumps(dict(manifest, **change)))
                self.assertEqual(inference.list_models(self.directory.name)[0]["status"], "invalid")
                with self.assertRaises(inference.InferenceError):
                    inference.analyze_document(document_from_text("실적 부진"), engine="model", model_path=path)

    def test_malformed_transformer_config_and_shard_index_are_listed_invalid(self):
        for bad_file in ("config.json", "model.safetensors.index.json"):
            with self.subTest(bad_file=bad_file):
                path = write_baseline(self.directory.name, bad_file.replace(".", "-"))
                manifest_path = path / "model_manifest.json"
                manifest = json.loads(manifest_path.read_text())
                manifest["model_type"] = "transformer"
                payloads = {"config.json": b"{}", "tokenizer_config.json": b"{}", "tokenizer.json": b"{}"}
                payloads["model.safetensors" if bad_file == "config.json" else "model.safetensors.index.json"] = b"placeholder"
                payloads[bad_file] = b"[]"
                for name, value in payloads.items():
                    (path / name).write_bytes(value)
                manifest["files"] = {name: sha256(value).hexdigest() for name, value in payloads.items()}
                manifest_path.write_text(json.dumps(manifest))
                result = next(model for model in inference.list_models(self.directory.name) if model["id"] == path.name)
                self.assertEqual(result["status"], "invalid")
                self.assertIn("형식", result["reason"])

    def test_model_checksum_change_is_rejected_until_manifest_updated(self):
        path = write_baseline(self.directory.name)
        document = document_from_text("실적 부진")
        first = inference.analyze_document(document, engine="model", model_path=path)
        weights = path / "baseline_model.json"
        payload = json.loads(weights.read_text())
        payload["coef"] = [-5.0] * 4
        weights.write_text(json.dumps(payload))
        with self.assertRaisesRegex(inference.InferenceError, "체크섬"):
            inference.analyze_document(document, engine="model", model_path=path)
        manifest_path = path / "model_manifest.json"
        manifest = json.loads(manifest_path.read_text())
        manifest["files"]["baseline_model.json"] = sha256(weights.read_bytes()).hexdigest()
        manifest_path.write_text(json.dumps(manifest))
        second = inference.analyze_document(document, engine="model", model_path=path)
        self.assertNotEqual(first["model_fingerprint"], second["model_fingerprint"])
        self.assertNotEqual(first["metrics"]["negative_score"], second["metrics"]["negative_score"])

    def test_model_cache_reuses_revision_and_retains_only_two_models(self):
        paths = [write_baseline(self.directory.name, f"model-{index}") for index in range(3)]
        first_descriptor = inference._inspect_model(paths[0])
        runtime = inference._load_runtime(first_descriptor)
        self.assertIs(runtime, inference._load_runtime(first_descriptor))
        for path in paths[1:]:
            inference.analyze_document(document_from_text("실적 부진"), engine="model", model_path=path)
        self.assertEqual(len(inference._MODEL_CACHE), 2)
        self.assertNotIn(first_descriptor["fingerprint"], inference._MODEL_CACHE)

    def test_parallel_analysis_loads_one_runtime_and_keeps_outputs_independent(self):
        path = write_baseline(self.directory.name)
        document = document_from_text("실적 부진")
        with mock.patch("inference._BaselineRuntime", wraps=inference._BaselineRuntime) as factory:
            with ThreadPoolExecutor(max_workers=4) as executor:
                results = list(executor.map(lambda _: inference.analyze_document(document, engine="model", model_path=path), range(8)))
            self.assertEqual(factory.call_count, 1)
        self.assertEqual(len({result["model_fingerprint"] for result in results}), 1)
        results[0]["metadata"]["company"] = "changed"
        self.assertIsNone(results[1]["metadata"]["company"])

    def test_transformer_loader_stays_local_and_rejects_random_missing_weights(self):
        tokenizer = SimpleNamespace(model_max_length=128)
        model = mock.Mock()
        model.config = SimpleNamespace(num_labels=2, max_position_embeddings=128)
        auto_tokenizer = mock.Mock()
        auto_tokenizer.from_pretrained.return_value = tokenizer
        auto_model = mock.Mock()
        auto_model.from_pretrained.return_value = (model, {})
        modules = {"torch": SimpleNamespace(), "transformers": SimpleNamespace(AutoTokenizer=auto_tokenizer, AutoModelForSequenceClassification=auto_model)}
        descriptor = dict(path=self.directory.name, manifest=dict(negative_id=1))
        with mock.patch.dict(sys.modules, modules):
            runtime = inference._TransformerRuntime(descriptor)
            self.assertEqual(runtime.max_tokens, 128)
            self.assertTrue(auto_tokenizer.from_pretrained.call_args.kwargs["local_files_only"])
            self.assertTrue(auto_model.from_pretrained.call_args.kwargs["local_files_only"])
            self.assertFalse(auto_model.from_pretrained.call_args.kwargs["trust_remote_code"])
            self.assertTrue(auto_model.from_pretrained.call_args.kwargs["use_safetensors"])
            auto_model.from_pretrained.return_value = (model, {"missing_keys": ["classifier.weight"]})
            with self.assertRaisesRegex(inference.InferenceError, "임의 초기화"):
                inference._TransformerRuntime(descriptor)

    def test_settings_are_validated_against_model_input_limit(self):
        path = write_baseline(self.directory.name)
        document = document_from_text("실적 부진")
        for settings in (dict(max_tokens=257), dict(threshold=float("nan")), dict(threshold=2)):
            with self.subTest(settings=settings), self.assertRaises(inference.InferenceError):
                inference.analyze_document(document, engine="model", model_path=path, settings=settings)
        with self.assertRaises(inference.InferenceError):
            inference.analyze_document(document, engine="unknown")

    def test_missing_model_never_falls_back_to_remote_or_random_predictions(self):
        with self.assertRaises(FileNotFoundError):
            inference.get_latest_model_path(self.directory.name)
        with mock.patch("inference._load_runtime") as load:
            with self.assertRaises(inference.InferenceError):
                inference.analyze_document(document_from_text("실적 부진"), engine="model", model_path=Path(self.directory.name) / "missing")
            load.assert_not_called()

    def test_legacy_wrapper_uses_manifest_threshold(self):
        path = write_baseline(self.directory.name, threshold=0.999)
        label, confidence = inference.predict_hidden_sell_signal("실적 부진", model_path=path)
        self.assertIn("Non-negative", label)
        self.assertGreater(confidence, 50)
        score = inference.analyze_document(document_from_text("실적 부진"), engine="model", model_path=path)["metrics"]["negative_score"]
        self.assertAlmostEqual(confidence, score * 100)


if __name__ == "__main__":
    unittest.main()
