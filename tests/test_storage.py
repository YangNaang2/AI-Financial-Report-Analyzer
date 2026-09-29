from concurrent.futures import ThreadPoolExecutor
from contextlib import closing
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import sqlite3
import tempfile
import unittest
from unittest.mock import patch

from storage import Library, LibraryError


def document(text="매출 증가 전망", **changes):
    result = {"content_hash": hashlib.sha256(text.encode()).hexdigest(), "filename": "샘플.pdf", "text": text,
              "pages": [{"number": 1, "text": text, "paragraphs": [{"id": "p1", "page": 1, "text": text, "start": 0, "end": len(text)}]}],
              "metadata": {"company": "가상기업", "ticker": "005930", "broker": "가상증권", "report_date": "2026-01-01"},
              "evidence": {"ticker": {"page": 1, "text": "005930"}}, "warnings": [], "status": "ready", "extraction_version": "2"}
    result.update(changes)
    return result


def analysis(**changes):
    result = {"engine": "rules", "model_id": "keyword-v2", "model_fingerprint": "rules-v2", "preprocessing_version": "2",
              "settings": {"max_length": 256, "overlap": 32}, "status": "ready", "created_at": "2026-01-01T00:00:00+00:00",
              "segments": [], "summary": [], "metrics": {"negative_score": None, "segment_count": 0}, "warnings": []}
    result.update(changes)
    return result


class StorageTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name) / "private" / "library.sqlite3"
        self.library = Library(self.path)

    def test_all_reads_and_writes_are_scoped_by_owner(self):
        first = self.library.save_document("alice", document())
        second = self.library.save_document("bob", document())
        self.assertNotEqual(first, second)
        run = self.library.save_analysis("alice", first, analysis())
        self.assertIsNone(self.library.get_document("bob", first))
        self.assertIsNone(self.library.get_analysis("bob", run))
        self.assertEqual(self.library.list_analyses("bob", first), [])
        self.assertEqual([row["id"] for row in self.library.list_documents("bob")], [second])
        for operation in (lambda: self.library.update_document("bob", first, note="overwrite"),
                          lambda: self.library.delete_document("bob", first),
                          lambda: self.library.save_analysis("bob", first, analysis())):
            with self.assertRaises(LibraryError):
                operation()
        self.assertIsNotNone(self.library.get_document("alice", first))
        self.assertEqual(len(json.loads(self.library.export_backup("bob"))["analyses"]), 0)

    def test_sql_quotes_and_search_are_literal_and_cannot_cross_owners(self):
        owner = "alice' OR 1=1 --"
        identifier = self.library.save_document(owner, document(text="이 회사의 '주가'_%"))
        self.library.update_document(owner, identifier, note="' OR 1=1 --", tags=["기술", "검색"])
        self.library.save_document("bob", document(text="비공개"))
        self.assertEqual(len(self.library.list_documents(owner, query="'주가'_%")), 1)
        self.assertEqual(len(self.library.list_documents(owner, query="검색")), 1)
        self.assertEqual(self.library.list_documents(owner, query="비공개"), [])
        self.assertIsNone(self.library.get_document("bob", "' OR 1=1 --"))

    def test_duplicate_document_updates_body_but_preserves_annotations_and_runs(self):
        identifier = self.library.save_document("alice", document())
        self.library.update_document("alice", identifier, favorite=True, note="검토 필요", tags=["반도체"])
        run = self.library.save_analysis("alice", identifier, analysis())
        revised = document(filename="수정 이름.pdf", metadata={"company": "정정기업", "ticker": "005930"})
        self.assertEqual(self.library.save_document("alice", revised), identifier)
        saved = self.library.get_document("alice", identifier)
        self.assertEqual(saved["filename"], "수정 이름.pdf")
        self.assertTrue(saved["favorite"])
        self.assertEqual(saved["note"], "검토 필요")
        self.assertEqual(saved["tags"], ["반도체"])
        self.assertEqual(self.library.get_analysis("alice", run)["analysis"]["metadata"]["company"], "가상기업")
        self.assertEqual(self.library.get_analysis("alice", run)["analysis"]["document_filename"], "샘플.pdf")

    def test_metadata_corrections_merge_and_analysis_snapshots_remain_immutable(self):
        identifier = self.library.save_document("alice", document())
        snapshot = {"company": "분석 시점 기업", "ticker": "005930"}
        original = analysis(metadata=snapshot)
        first = self.library.save_analysis("alice", identifier, original)
        self.library.update_document("alice", identifier, metadata={"company": "새기업"})
        self.assertEqual(self.library.get_document("alice", identifier)["metadata"]["ticker"], "005930")
        self.assertEqual(self.library.save_analysis("alice", identifier, original), first)
        second = self.library.save_analysis("alice", identifier, analysis())
        self.assertNotEqual(first, second)
        self.assertEqual(self.library.get_analysis("alice", first)["analysis"]["metadata"], snapshot)
        self.assertEqual(self.library.get_analysis("alice", second)["analysis"]["metadata"]["company"], "새기업")
        self.assertNotIn("document_filename", original)

    def test_dedup_is_canonical_and_depends_on_every_execution_identity_field(self):
        identifier = self.library.save_document("alice", document())
        original = analysis()
        first = self.library.save_analysis("alice", identifier, original)
        reordered = analysis(settings={"overlap": 32, "max_length": 256}, created_at="2026-02-01T00:00:00Z")
        self.assertEqual(self.library.save_analysis("alice", identifier, reordered), first)
        for changes in ({"settings": {"max_length": 128, "overlap": 32}}, {"model_fingerprint": "new-model"},
                        {"preprocessing_version": "3"}, {"engine": "demo"}, {"metadata": {"ticker": "000660"}}):
            self.assertNotEqual(self.library.save_analysis("alice", identifier, analysis(**changes)), first)
        self.assertEqual(len(self.library.list_analyses("alice")), 6)

    def test_annotations_favorite_filter_and_validation(self):
        identifier = self.library.save_document("alice", document())
        self.assertEqual(self.library.list_documents("alice", favorite_only=True), [])
        self.library.update_document("alice", identifier, note="=test", tags=["연구"], favorite=True)
        self.assertEqual(len(self.library.list_documents("alice", favorite_only=True)), 1)
        for update in ({"favorite": 1}, {"tags": "tag"}, {"tags": [None]}, {"note": 42}, {"metadata": {"ticker": 5930}}):
            with self.subTest(update=update), self.assertRaises(LibraryError):
                self.library.update_document("alice", identifier, **update)
        self.assertEqual(self.library.get_document("alice", identifier)["note"], "=test")

    def test_delete_cascades_only_the_owners_document_runs(self):
        first = self.library.save_document("alice", document())
        second = self.library.save_document("bob", document())
        self.library.save_analysis("alice", first, analysis())
        self.library.save_analysis("bob", second, analysis())
        self.library.delete_document("alice", first)
        self.assertEqual(self.library.list_documents("alice"), [])
        self.assertEqual(self.library.list_analyses("alice"), [])
        self.assertEqual(len(self.library.list_analyses("bob")), 1)

    def test_backup_is_owner_only_remaps_ids_preserves_dates_and_merges_without_overwrite(self):
        source_id = self.library.save_document("alice", document())
        self.library.update_document("alice", source_id, note="old note", favorite=True, tags=["original"])
        self.library.save_analysis("alice", source_id, analysis())
        self.library.save_document("bob", document(text="타인의 원문"))
        payload = self.library.export_backup("alice")
        self.assertNotIn("타인의 원문", payload)
        self.assertNotIn('"owner"', payload)
        result = self.library.import_backup("charlie", payload)
        self.assertEqual(result, {"documents": 1, "analyses": 1, "skipped_documents": 0, "skipped_analyses": 0})
        restored = self.library.list_documents("charlie")[0]
        self.assertNotEqual(restored["id"], source_id)
        self.assertEqual(restored["created_at"], self.library.get_document("alice", source_id)["created_at"])
        self.assertTrue(restored["favorite"])
        self.library.update_document("charlie", restored["id"], note="new note", metadata={"company": "현재기업"})
        repeat = self.library.import_backup("charlie", payload)
        self.assertEqual(repeat, {"documents": 0, "analyses": 0, "skipped_documents": 1, "skipped_analyses": 1})
        self.assertEqual(self.library.get_document("charlie", restored["id"])["note"], "new note")
        self.assertEqual(self.library.get_document("charlie", restored["id"])["metadata"]["company"], "현재기업")
        self.assertEqual(len(self.library.list_documents("bob")), 1)

    def test_invalid_final_backup_row_prevents_every_write(self):
        identifier = self.library.save_document("alice", document())
        self.library.save_analysis("alice", identifier, analysis())
        backup = json.loads(self.library.export_backup("alice"))
        corruptions = []
        foreign = deepcopy(backup)
        foreign["analyses"][0]["document_id"] = "not-in-backup"
        corruptions.append(foreign)
        nan = deepcopy(backup)
        nan["analyses"][0]["analysis"]["settings"]["threshold"] = float("nan")
        corruptions.append(nan)
        duplicate = deepcopy(backup)
        duplicate["documents"].append(duplicate["documents"][0])
        corruptions.append(duplicate)
        malformed = deepcopy(backup)
        malformed["analyses"][0]["document_id"] = []
        corruptions.append(malformed)
        invalid_time = deepcopy(backup)
        invalid_time["analyses"][0]["created_at"] = "yesterday"
        corruptions.append(invalid_time)
        for bad in corruptions:
            with self.subTest(bad=bad), self.assertRaises(LibraryError):
                self.library.import_backup("new-owner", json.dumps(bad))
            self.assertEqual(self.library.list_documents("new-owner"), [])
            self.assertEqual(self.library.list_analyses("new-owner"), [])

    def test_restore_database_failure_rolls_back_preceding_document_insert(self):
        source = Library(":memory:")
        self.addCleanup(source.close)
        identifier = source.save_document("alice", document())
        source.save_analysis("alice", identifier, analysis())
        with closing(sqlite3.connect(self.path)) as connection:
            connection.execute("CREATE TRIGGER reject_run BEFORE INSERT ON analyses BEGIN SELECT RAISE(FAIL,'simulated disk failure'); END")
        with self.assertRaises(LibraryError):
            self.library.import_backup("alice", source.export_backup("alice"))
        self.assertEqual(self.library.list_documents("alice"), [])

    def test_future_schema_unknown_schema_and_corruption_are_preserved(self):
        for version in (2, 999):
            path = Path(self.directory.name) / f"future{version}.db"
            with closing(sqlite3.connect(path)) as connection:
                connection.execute(f"PRAGMA user_version={version}")
            with self.assertRaisesRegex(LibraryError, "최신"):
                Library(path)
            with closing(sqlite3.connect(path)) as connection:
                self.assertEqual(connection.execute("PRAGMA user_version").fetchone()[0], version)
        unknown = Path(self.directory.name) / "unknown.db"
        with closing(sqlite3.connect(unknown)) as connection:
            connection.execute("CREATE TABLE unrelated(secret TEXT)")
        with self.assertRaises(LibraryError):
            Library(unknown)
        corrupt = Path(self.directory.name) / "corrupt.db"
        original = b"corrupt database bytes"
        corrupt.write_bytes(original)
        with self.assertRaises(LibraryError):
            Library(corrupt)
        self.assertEqual(corrupt.read_bytes(), original)

    def test_document_and_analysis_safe_json_and_model_score_validation(self):
        for candidate in (document(content_hash="bad"), document(metadata={"ticker": 5930}), document(warnings=[float("nan")]),
                          document(extra={1: "numeric key"}), document(extra=(1, 2))):
            with self.subTest(candidate=candidate), self.assertRaises(LibraryError):
                self.library.save_document("alice", candidate)
        identifier = self.library.save_document("alice", document())
        for candidate in (analysis(metrics={"negative_score": .99}), analysis(settings={"value": float("nan")}),
                          analysis(document_hash="wrong"), analysis(engine="model", metrics={"negative_score": 1.1})):
            with self.subTest(candidate=candidate), self.assertRaises(LibraryError):
                self.library.save_analysis("alice", identifier, candidate)
        run = self.library.save_analysis("alice", identifier, analysis(engine="model", metrics={"negative_score": 0.0}))
        self.assertEqual(self.library.get_analysis("alice", run)["analysis"]["metrics"]["negative_score"], 0.0)

    def test_empty_owner_rejected_and_in_memory_library_remains_available(self):
        for owner in ("", " ", None, 3):
            with self.subTest(owner=owner), self.assertRaises(LibraryError):
                self.library.list_documents(owner)
        memory = Library(":memory:")
        self.addCleanup(memory.close)
        identifier = memory.save_document("demo", document())
        self.assertIsNotNone(memory.get_document("demo", identifier))

    def test_env_location_permissions_and_concurrent_duplicate_insert(self):
        alternate = Path(self.directory.name) / "environment.sqlite3"
        with patch.dict(os.environ, {"REPORT_LENS_DB": str(alternate)}):
            env_library = Library()
        self.assertTrue(alternate.exists())
        if os.name == "posix":
            self.assertEqual(alternate.stat().st_mode & 0o777, 0o600)
        def save(_):
            return Library(alternate).save_document("same-owner", document())
        with ThreadPoolExecutor(max_workers=4) as pool:
            ids = list(pool.map(save, range(8)))
        self.assertEqual(len(set(ids)), 1)
        self.assertEqual(len(env_library.list_documents("same-owner")), 1)


if __name__ == "__main__":
    unittest.main()
