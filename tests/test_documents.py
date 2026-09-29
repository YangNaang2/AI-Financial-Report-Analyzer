from hashlib import sha256
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest import mock

from data_processor import (
    DocumentError, chunk_document, chunk_text, clean_financial_report,
    document_from_text, extract_document, extract_text_from_pdf,
)


class CharacterTokenizer:
    def num_special_tokens_to_add(self, pair=False):
        return 2

    def __call__(self, text, add_special_tokens=False, return_offsets_mapping=False, truncation=False, **kwargs):
        assert truncation is False
        offsets = [(index, index + 1) for index, character in enumerate(text) if not character.isspace()]
        result = {"input_ids": ([101] if add_special_tokens else []) + [1] * len(offsets) + ([102] if add_special_tokens else [])}
        if return_offsets_mapping:
            result["offset_mapping"] = offsets
        return result


class DocumentTests(unittest.TestCase):
    def test_financial_numbers_and_source_offsets_are_preserved(self):
        text = "삼성전자 (005930)\n발간일: 2024.02.29\n목표주가 120,000원\n현재주가 100,000원\n투자의견: 매수\n미래에셋증권\n\n실적 추정치 -5.5% 하향. 매출은 123억원."
        document = document_from_text(text, "report.txt")
        self.assertEqual(document["content_hash"], sha256(text.encode()).hexdigest())
        self.assertEqual(document["metadata"]["ticker"], "005930")
        self.assertEqual(document["metadata"]["report_date"], "2024-02-29")
        self.assertEqual(document["metadata"]["target_price"], "120000 원")
        self.assertEqual(document["metadata"]["current_price"], "100000 원")
        self.assertEqual(document["evidence"]["target_price"]["unit"], "원")
        self.assertEqual(document["metadata"]["company"], "삼성전자")
        self.assertEqual(document["metadata"]["opinion"], "매수")
        self.assertEqual(document["evidence"]["target_price"]["page"], 1)
        self.assertIn("-5.5%", clean_financial_report(text))
        for paragraph in document["pages"][0]["paragraphs"]:
            self.assertEqual(document["pages"][0]["text"][paragraph["start"]:paragraph["end"]], paragraph["text"])

    def test_conflicting_metadata_and_invalid_dates_remain_unknown(self):
        document = document_from_text("삼성전자 (005930)\n카카오 (035720)\n발간일: 2025.02.30\n목표주가 100,000원\n목표주가 90,000원")
        self.assertIsNone(document["metadata"]["ticker"])
        self.assertIsNone(document["metadata"]["company"])
        self.assertIsNone(document["metadata"]["target_price"])
        self.assertIsNone(document["metadata"]["report_date"])
        self.assertTrue(document["warnings"])
        self.assertEqual(document_from_text("발간일: 1998년 1월 2일")["metadata"]["report_date"], "1998-01-02")

    def test_price_number_is_not_misidentified_as_ticker(self):
        document = document_from_text("목표주가 100000원\n영업이익률 12.3%")
        self.assertIsNone(document["metadata"]["ticker"])
        self.assertIsNone(document["metadata"]["company"])

    def test_price_units_are_preserved_and_never_guessed(self):
        document = document_from_text("목표주가 12.5만원\n현재주가 82000")
        self.assertEqual(document["metadata"]["target_price"], "12.5 만원")
        self.assertEqual(document["evidence"]["target_price"]["unit"], "만원")
        self.assertEqual(document["metadata"]["current_price"], "82000")
        self.assertEqual(document["evidence"]["current_price"]["unit"], "미확인")
        self.assertEqual(document_from_text("목표주가(원) 100,000")["metadata"]["target_price"], "100000 원")

    def test_financial_facts_require_explicit_units_and_keep_exact_sources(self):
        document = document_from_text("매출액 1,200억원\n영업이익 -50억원\nEPS(원) 3,200\nROE 12.5%\n매출 2024년 추정")
        facts = document["financial_facts"]
        self.assertEqual([(fact["value"], fact["unit"]) for fact in facts], [("1200", "억원"), ("-50", "억원"), ("3200", "원"), ("12.5", "%")])
        for fact in facts:
            self.assertEqual(document["pages"][0]["text"][fact["start"]:fact["end"]], fact["text"])

    def test_malformed_and_compound_amounts_are_not_reduced_to_numeric_prefixes(self):
        bad_price = document_from_text("목표주가: 12,34원")
        self.assertIsNone(bad_price["metadata"]["target_price"])
        self.assertTrue(bad_price["warnings"])
        spaced_unit = document_from_text("목표주가: 12만 원")
        self.assertEqual(spaced_unit["metadata"]["target_price"], "12 만원")
        for text in ("매출액(억원): 1,00", "매출액: 1조원 2,000억원"):
            with self.subTest(text=text):
                document = document_from_text(text)
                self.assertEqual(document["financial_facts"], [])
                self.assertTrue(document["warnings"])

    def test_empty_oversized_and_damaged_documents_raise_clear_errors(self):
        with self.assertRaises(DocumentError):
            document_from_text("  \n")
        for content in (b"", b"not a PDF", b"%PDF-1.7 damaged"):
            with self.subTest(content=content), self.assertRaises(DocumentError):
                extract_document(content)
        with self.assertRaises(DocumentError):
            extract_document(b"%PDF-123456", max_bytes=4)
        with self.assertRaises(DocumentError):
            extract_document(b"\xff\xfe", filename="report.txt")

    def fake_pdf(self, texts, *, encrypted=False):
        pages = []
        for text in texts:
            page = mock.Mock()
            if isinstance(text, Exception):
                page.extract_text.side_effect = text
            else:
                page.extract_text.return_value = text
            pages.append(page)
        document = SimpleNamespace(pages=pages, doc=SimpleNamespace(encryption={} if encrypted else None, is_extractable=True))
        context = mock.MagicMock()
        context.__enter__.return_value = document
        return context

    def test_scanned_and_partial_pdf_pages_are_explicit(self):
        with mock.patch("pdfplumber.open", return_value=self.fake_pdf([None, ""])):
            document = extract_document(b"%PDF-1.7 placeholder")
        self.assertEqual(document["status"], "ocr_required")
        self.assertEqual(chunk_document(document)["unprocessed_pages"], [1, 2])
        with mock.patch("pdfplumber.open", return_value=self.fake_pdf(["매출 100억원", RuntimeError("failed page")])):
            document = extract_document(b"%PDF-1.7 placeholder")
        self.assertEqual(document["status"], "partial")
        self.assertEqual(document["pages"][0]["text"], "매출 100억원")
        self.assertEqual(chunk_document(document)["unprocessed_pages"], [2])

    def test_encrypted_and_page_limit_pdf_rejected_before_analysis(self):
        with mock.patch("pdfplumber.open", return_value=self.fake_pdf(["text"], encrypted=True)):
            with self.assertRaisesRegex(DocumentError, "암호화"):
                extract_document(b"%PDF-1.7 placeholder")
        with mock.patch("pdfplumber.open", return_value=self.fake_pdf(["one", "two"])):
            with self.assertRaisesRegex(DocumentError, "페이지"):
                extract_document(b"%PDF-1.7 placeholder", max_pages=1)

    def test_text_bytes_and_path_share_hash_and_legacy_pdf_wrapper(self):
        raw = "삼성전자 005930 100,000원".encode()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "report.txt"
            path.write_bytes(raw)
            one, two = extract_document(path), extract_document(raw, filename="report.txt")
            self.assertEqual(one["content_hash"], two["content_hash"])
            with mock.patch("pdfplumber.open", return_value=self.fake_pdf(["원문 100원"])):
                self.assertEqual(extract_text_from_pdf(self._pdf_path(directory)), "원문 100원")

    def _pdf_path(self, directory):
        path = Path(directory) / "report.pdf"
        path.write_bytes(b"%PDF-1.7 placeholder")
        return path

    def test_true_token_chunks_cover_long_input_without_truncation(self):
        document = document_from_text("한글과 숫자 12345 내용이 이어집니다.\n\n두 번째 문단은 실적 하향입니다.")
        tokenizer = CharacterTokenizer()
        chunks = chunk_document(document, tokenizer=tokenizer, max_tokens=10, overlap=2)
        self.assertGreater(len(chunks["segments"]), 2)
        self.assertEqual(chunks["analyzed_chars"], chunks["total_chars"])
        self.assertEqual(chunks["status"], "ready")
        self.assertEqual(chunks["unprocessed_pages"], [])
        page = document["pages"][0]["text"]
        for segment in chunks["segments"]:
            self.assertEqual(page[segment["start"]:segment["end"]], segment["text"])
            self.assertLessEqual(len(tokenizer(segment["text"], add_special_tokens=True)["input_ids"]), 10)
        self.assertGreater(sum(len(segment["text"]) for segment in chunks["segments"]), chunks["analyzed_chars"])

    def test_segment_limit_records_unprocessed_coverage(self):
        chunks = chunk_text(" ".join(f"단어{index}" for index in range(30)), max_tokens=8, overlap=2, max_segments=1)
        self.assertEqual(len(chunks["segments"]), 1)
        self.assertEqual(chunks["status"], "partial")
        self.assertEqual(chunks["unprocessed_pages"], [1])
        self.assertLess(chunks["analyzed_chars"], chunks["total_chars"])
        self.assertTrue(any("미처리" in warning for warning in chunks["warnings"]))

    def test_bad_chunk_settings_and_non_mapping_tokenizer_fail(self):
        document = document_from_text("분석할 텍스트")
        for settings in (dict(max_tokens=0), dict(max_tokens=8, overlap=8), dict(max_segments=0), dict(max_tokens=True)):
            with self.subTest(settings=settings), self.assertRaises(DocumentError):
                chunk_document(document, **settings)
        tokenizer = mock.Mock()
        tokenizer.num_special_tokens_to_add.return_value = 2
        tokenizer.side_effect = NotImplementedError()
        with self.assertRaisesRegex(DocumentError, "토크나이저"):
            chunk_document(document, tokenizer=tokenizer)


if __name__ == "__main__":
    unittest.main()
