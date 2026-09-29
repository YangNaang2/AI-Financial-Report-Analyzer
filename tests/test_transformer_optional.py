"""Optional local Transformer plumbing test, not a financial-quality benchmark.

Creates random tiny BERT weights, a fast tokenizer and synthetic labeled text in
a temporary directory. It never downloads a pretrained model or retains files.
"""

from datetime import date, timedelta
from importlib.util import find_spec
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch


@unittest.skipUnless(find_spec("torch") is not None and find_spec("transformers") is not None,
                     "Optional torch/transformers dependencies are not installed")
class OptionalTransformerTests(unittest.TestCase):
    def test_local_tiny_transformer_train_infer_store_and_restore(self):
        offline = {"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
                   "HF_HUB_DISABLE_TELEMETRY": "1", "TOKENIZERS_PARALLELISM": "false"}
        with patch.dict(os.environ, offline):
            import torch
            from transformers import BertConfig, BertForSequenceClassification, BertTokenizerFast

            from data_processor import document_from_text
            from inference import analyze_document, clear_model_cache
            from storage import Library
            from train import train_model

            original_threads = torch.get_num_threads()
            torch.set_num_threads(1)
            self.addCleanup(torch.set_num_threads, original_threads)
            self.addCleanup(clear_model_cache)
            torch.manual_seed(37)

            with tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                base = root / "tiny-local-base"
                base.mkdir()
                vocabulary = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", "실적", "부진", "감소", "위험",
                              "성장", "개선", "회복", "상승", "보고서", "기업", "문서", "평가"]
                vocabulary.extend(str(index) for index in range(50))
                vocab_file = base / "vocab.txt"
                vocab_file.write_text("\n".join(vocabulary) + "\n", encoding="utf-8")
                tokenizer = BertTokenizerFast(vocab_file=str(vocab_file), do_lower_case=False, model_max_length=64)
                self.assertTrue(tokenizer.is_fast)
                tokenizer.save_pretrained(base)
                config = BertConfig(
                    vocab_size=len(tokenizer), hidden_size=16, num_hidden_layers=1,
                    num_attention_heads=2, intermediate_size=32, max_position_embeddings=64,
                    num_labels=2, id2label={0: "non_negative", 1: "negative"},
                    label2id={"non_negative": 0, "negative": 1},
                )
                BertForSequenceClassification(config).save_pretrained(base, safe_serialization=True)
                self.assertTrue((base / "model.safetensors").exists())

                records = []
                for index in range(20):
                    published = date(2020, 1, 1) + timedelta(days=index * 10)
                    for label in (0, 1):
                        phrase = "실적 부진 감소 위험" if label else "성장 개선 회복 상승"
                        records.append({"id": f"{index}-{label}",
                                        "text": f"{phrase} 기업 {index} 문서 {label} 보고서 평가",
                                        "report_date": published.isoformat(),
                                        "label_end_date": (published + timedelta(days=1)).isoformat(),
                                        "label_id": label, "status": "labeled"})
                # Keep this small regression test on CPU even on GPU CI hosts.
                with patch("torch.cuda.is_available", return_value=False):
                    manifest = train_model(
                        records=records, output_dir=root / "trained", backend="transformer",
                        base_model=str(base), epochs=1, batch_size=4, max_tokens=32, overlap=4, seed=37,
                    )
                self.assertTrue(manifest["completed"])
                self.assertEqual(manifest["model_type"], "transformer")
                self.assertEqual({key: value["count"] for key, value in manifest["splits"].items()},
                                 {"train": 24, "validation": 8, "test": 8})
                self.assertIn("model.safetensors", manifest["files"])

                # A longer paragraph exercises source-mapped fast-tokenizer chunks.
                document = document_from_text(" ".join(["실적 부진 감소 위험 보고서 평가"] * 20))
                result = analyze_document(document, engine="model", model_path=manifest["model_dir"])
                self.assertEqual(result["engine"], "model")
                self.assertEqual(result["status"], "ready")
                self.assertGreater(len(result["segments"]), 1)
                self.assertTrue(0 <= result["metrics"]["negative_score"] <= 1)
                self.assertEqual(result["metrics"]["analyzed_chars"], result["metrics"]["total_chars"])
                for segment in result["segments"]:
                    self.assertEqual(segment["text"], document["pages"][0]["text"][segment["start"]:segment["end"]])

                library = Library(root / "library.sqlite3")
                document_id = library.save_document("smoke-owner", document)
                run_id = library.save_analysis("smoke-owner", document_id, result)
                saved = library.get_analysis("smoke-owner", run_id)
                self.assertEqual(saved["analysis"]["model_fingerprint"], result["model_fingerprint"])
                counts = library.import_backup("restored-owner", library.export_backup("smoke-owner"))
                self.assertEqual(counts["documents"], 1)
                self.assertEqual(counts["analyses"], 1)
                self.assertEqual(library.list_analyses("restored-owner")[0]["analysis"], saved["analysis"])
                clear_model_cache()


if __name__ == "__main__":
    unittest.main()
