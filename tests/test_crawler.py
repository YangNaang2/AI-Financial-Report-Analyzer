import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import requests

import crawler


class Response:
    def __init__(self, body=b"", status=200, headers=None):
        self.body, self.status_code = body, status
        self.headers = headers or {}
        self.closed = False

    def iter_content(self, size):
        for start in range(0, len(self.body), 5):
            yield self.body[start:start + 5]

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(f"HTTP {self.status_code}", response=self)

    def close(self):
        self.closed = True


def listing(items, has_next=False):
    return Response(json.dumps({"items": items, "hasNext": has_next}).encode())


def report(identifier="1", title="같은 제목", **extra):
    return dict(nid=identifier, title=title, writeDate="2026-09-29", itemCode="005930", **extra)


class CrawlerTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name)
        self.sleep = mock.patch("crawler.time.sleep").start()
        self.addCleanup(mock.patch.stopall)

    def test_current_api_detail_pdf_provenance_and_resume(self):
        pdf = b"%PDF-1.7\nsynthetic test only\n%%EOF"
        pdf_url = "https://stock.pstatic.net/stock-research/test.pdf"
        responses = [listing([report()]), Response(json.dumps({"attachUrl": pdf_url}).encode()), Response(pdf)]
        with mock.patch("crawler.requests.get", side_effect=responses) as get:
            result = crawler.download_naver_reports(self.path)
        self.assertEqual((result["status"], result["count"]), ("complete", 1))
        self.assertIn("index=0&size=15", get.call_args_list[0].args[0])
        self.assertEqual(get.call_args_list[1].args[0], crawler.API_URL + "/1")
        self.assertTrue(all(call.kwargs["timeout"] == (5, 20) for call in get.call_args_list))
        entry = result["downloaded"][0]
        self.assertEqual((self.path / entry["filename"]).read_bytes(), pdf)
        self.assertEqual(entry["sha256"], hashlib.sha256(pdf).hexdigest())
        self.assertEqual(entry["published_at"], "2026-09-29")
        self.assertEqual(entry["source_url"], crawler.LIST_URL + "/1")
        self.assertTrue(all(response.closed for response in responses))
        with mock.patch("crawler.requests.get", return_value=listing([report()])) as get:
            resumed = crawler.download_naver_reports(self.path)
        self.assertEqual(resumed["count"], 0)
        self.assertEqual(len(resumed["skipped"]), 1)
        self.assertEqual(get.call_count, 1)

    def test_identical_titles_get_distinct_sanitized_names(self):
        items = [report(str(i), "../../같은:제목", attachUrl=f"https://stock.pstatic.net/{i}.pdf") for i in (1, 2)]
        with mock.patch("crawler.requests.get", side_effect=[listing(items), Response(b"%PDF-1 test"), Response(b"%PDF-2 test")]):
            result = crawler.download_naver_reports(self.path)
        names = [row["filename"] for row in result["downloaded"]]
        self.assertEqual(len(set(names)), 2)
        self.assertTrue(all(Path(name).name == name and ":" not in name for name in names))
        self.assertEqual(len(list(self.path.glob("*.pdf"))), 2)

    def test_html_error_and_oversized_download_never_become_pdf_files(self):
        items = [report("1", attachUrl="https://stock.pstatic.net/1.pdf"), report("2", attachUrl="https://stock.pstatic.net/2.pdf")]
        with mock.patch("crawler.requests.get", side_effect=[listing(items), Response(b"<html>error</html>"),
                          Response(b"%PDF-bad", headers={"Content-Length": str(crawler.MAX_PDF_BYTES + 1)})]):
            result = crawler.download_naver_reports(self.path)
        self.assertEqual(result["status"], "failed")
        self.assertEqual(len(result["errors"]), 2)
        self.assertEqual(list(self.path.glob("*.pdf")), [])
        manifest = json.loads((self.path / "collection_manifest.json").read_text())
        self.assertTrue(all(row["status"] == "error" for row in manifest["reports"].values()))

    def test_stream_limit_is_enforced_without_content_length(self):
        with mock.patch("crawler.MAX_PDF_BYTES", 10), mock.patch("crawler.requests.get", side_effect=[
            listing([report(attachUrl="https://stock.pstatic.net/1.pdf")]), Response(b"%PDF-" + b"x" * 20)
        ]):
            result = crawler.download_naver_reports(self.path)
        self.assertEqual(result["count"], 0)
        self.assertEqual(list(self.path.glob("*.pdf")), [])

    def test_corrupt_existing_pdf_is_preserved_and_recovered_under_new_name(self):
        item = report(attachUrl="https://stock.pstatic.net/1.pdf")
        with mock.patch("crawler.requests.get", side_effect=[listing([item]), Response(b"%PDF-original")]):
            first = crawler.download_naver_reports(self.path)
        old = self.path / first["downloaded"][0]["filename"]
        old.write_bytes(b"damaged local data")
        with mock.patch("crawler.requests.get", side_effect=[listing([item]), Response(b"%PDF-recovered")]):
            second = crawler.download_naver_reports(self.path)
        self.assertEqual(old.read_bytes(), b"damaged local data")
        self.assertNotEqual(second["downloaded"][0]["filename"], old.name)

    def test_unsafe_links_redirects_and_authentication_are_not_followed(self):
        with mock.patch("crawler.requests.get", return_value=listing([report(attachUrl="http://127.0.0.1/private")])) as get:
            result = crawler.download_naver_reports(self.path)
        self.assertEqual(get.call_count, 1)
        self.assertEqual(result["status"], "failed")
        with mock.patch("crawler.requests.get", return_value=Response(status=302, headers={"Location": "http://127.0.0.1/"})) as get:
            with self.assertRaises(crawler.CollectionError):
                crawler._get(crawler.API_URL, max_bytes=100)
        self.assertEqual(get.call_count, 1)
        with mock.patch("crawler.requests.get", return_value=Response(status=403)) as get:
            result = crawler.download_naver_reports(self.path)
        self.assertEqual(get.call_count, 1)
        self.assertEqual(result["status"], "failed")

    def test_request_failures_retry_only_bounded_number_and_report_error(self):
        with mock.patch("crawler.requests.get", side_effect=requests.Timeout("offline")) as get:
            result = crawler.download_naver_reports(self.path)
        self.assertEqual(get.call_count, crawler.MAX_RETRIES + 1)
        self.assertEqual(result["status"], "failed")
        self.assertEqual(result["count"], 0)

    def test_changed_endpoint_is_explicit_and_legacy_html_remains_parseable(self):
        with mock.patch("crawler.requests.get", side_effect=[Response(b'{}'), Response(b'<html>new application</html>')]):
            result = crawler.download_naver_reports(self.path)
        self.assertEqual(result["status"], "failed")
        self.assertIn("사이트 구조", result["errors"][0]["reason"])
        html = '<table class="type_1"><tr><td>기업</td><td><a href="/research/company_read.naver?nid=1">제목</a></td><td class="file"><a href="https://stock.pstatic.net/a.pdf">PDF</a></td><td>26.09.29</td></tr></table>'
        parsed = crawler._parse_legacy(html, crawler.LEGACY_URL)
        self.assertEqual(parsed[0]["published_at"], "2026-09-29")

    def test_bad_manifest_and_invalid_pages_preserve_original_files(self):
        manifest = self.path / "collection_manifest.json"
        manifest.write_text("broken")
        with self.assertRaises(crawler.CollectionError):
            crawler.download_naver_reports(self.path)
        self.assertEqual(manifest.read_text(), "broken")
        for start, end in [(0, 1), (2, 1), (True, 1), (1, 101)]:
            with self.subTest(start=start, end=end), self.assertRaises(crawler.CollectionError):
                crawler.download_naver_reports(self.path, start, end)


if __name__ == "__main__":
    unittest.main()
