"""Owner-scoped SQLite report library and portable, validated JSON backups."""

from contextlib import contextmanager
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import sqlite3
import threading
from uuid import uuid4


SCHEMA_VERSION = 1
MAX_TEXT_CHARS = 2_000_000
MAX_BACKUP_BYTES = 20 * 1024 * 1024
MAX_DOCUMENTS = 2000
MAX_ANALYSES = 10000
_RESERVED = {"id", "owner", "favorite", "note", "tags", "created_at", "updated_at"}


class LibraryError(ValueError):
    """A library error safe to display without exposing document contents."""


def _now():
    return datetime.now(timezone.utc).isoformat()


def _text(value, label, maximum=2000, *, empty=False):
    if not isinstance(value, str) or len(value) > maximum or (not empty and not value.strip()):
        raise LibraryError(f"{label}: 올바른 문자열을 입력해 주세요 (최대 {maximum:,}자).")
    try:
        value.encode("utf-8")
    except UnicodeError as exc:
        raise LibraryError(f"{label}: 올바른 UTF-8 문자열을 입력해 주세요.") from exc
    return value


def _owner(value):
    return _text(value, "보관함 키", 256)


def _safe_json(value, seen=None, depth=0):
    if depth > 80:
        raise LibraryError("데이터 중첩이 너무 깊습니다.")
    if isinstance(value, str):
        _text(value, "JSON 문자열", MAX_BACKUP_BYTES, empty=True)
        return
    if value is None or isinstance(value, (bool, int)):
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise LibraryError("NaN이나 무한대는 저장할 수 없습니다.")
        return
    if not isinstance(value, (dict, list)):
        raise LibraryError("JSON으로 저장 가능한 데이터만 사용할 수 있습니다.")
    seen = set() if seen is None else seen
    if id(value) in seen:
        raise LibraryError("자기 자신을 참조하는 데이터는 저장할 수 없습니다.")
    seen.add(id(value))
    if isinstance(value, dict) and any(not isinstance(key, str) for key in value):
        raise LibraryError("JSON 항목 이름은 문자열이어야 합니다.")
    if isinstance(value, dict):
        for key in value:
            _text(key, "JSON 항목 이름", 2000, empty=True)
    for child in value.values() if isinstance(value, dict) else value:
        _safe_json(child, seen, depth + 1)
    seen.remove(id(value))


def _dump(value):
    _safe_json(value)
    try:
        return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)
    except (ValueError, TypeError, UnicodeError, RecursionError) as exc:
        raise LibraryError("JSON 데이터 형식이나 크기를 확인해 주세요.") from exc


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise LibraryError("JSON에 중복된 항목 이름이 있습니다.")
        result[key] = value
    return result


def _load(value):
    try:
        result = json.loads(value, object_pairs_hook=_unique_object,
                            parse_constant=lambda _: (_ for _ in ()).throw(LibraryError("JSON에 유효하지 않은 숫자가 있습니다.")))
        _safe_json(result)
        return result
    except (ValueError, TypeError, RecursionError) as exc:
        raise LibraryError("저장된 JSON을 읽을 수 없습니다. 원본과 백업을 확인해 주세요.") from exc


def _hash(value):
    return hashlib.sha256(_dump(value).encode("utf-8")).hexdigest()


def _timestamp(value):
    _text(value, "기록 시각", 128)
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        if "T" not in value or parsed.utcoffset() is None:
            raise ValueError
    except ValueError as exc:
        raise LibraryError("기록 시각은 시간대가 포함된 ISO 형식이어야 합니다.") from exc
    return value


def _metadata(value):
    if not isinstance(value, dict):
        raise LibraryError("리포트 메타데이터는 객체 형식이어야 합니다.")
    _safe_json(value)
    for key in ("company", "ticker", "broker", "report_date", "opinion", "target_price", "current_price"):
        if key in value and value[key] is not None:
            _text(value[key], f"메타데이터 {key}", 2000, empty=True)
    return deepcopy(value)


def _warnings(value):
    if not isinstance(value, list) or len(value) > 2000:
        raise LibraryError("경고는 문자열 목록이어야 합니다.")
    for warning in value:
        _text(warning, "경고", 20000, empty=True)


def _document(value):
    if not isinstance(value, dict):
        raise LibraryError("리포트 데이터는 객체 형식이어야 합니다.")
    _safe_json(value)
    document = {key: deepcopy(item) for key, item in value.items() if key not in _RESERVED}
    required = {"content_hash", "filename", "text", "pages", "metadata", "evidence", "warnings", "status", "extraction_version"}
    if not required <= document.keys():
        raise LibraryError("리포트 필수 항목이 누락되었습니다.")
    if not isinstance(document["content_hash"], str) or not re.fullmatch(r"[a-f0-9]{64}", document["content_hash"]):
        raise LibraryError("리포트 해시는 SHA-256 형식이어야 합니다.")
    _text(document["filename"], "파일 이름", 512)
    _text(document["text"], "원문", MAX_TEXT_CHARS, empty=True)
    _text(document["extraction_version"], "추출 버전", 128)
    if document["status"] not in ("ready", "partial", "ocr_required"):
        raise LibraryError("지원하지 않는 리포트 추출 상태입니다.")
    _metadata(document["metadata"])
    _warnings(document["warnings"])
    if not isinstance(document["evidence"], dict):
        raise LibraryError("메타데이터 근거는 객체 형식이어야 합니다.")
    pages = document["pages"]
    if not isinstance(pages, list) or len(pages) > 2000:
        raise LibraryError("리포트 페이지는 최대 2,000개여야 합니다.")
    seen, total = set(), 0
    for page in pages:
        if not isinstance(page, dict) or not {"number", "text", "paragraphs"} <= page.keys():
            raise LibraryError("페이지의 번호, 원문, 문단 정보가 필요합니다.")
        number = page["number"]
        if type(number) is not int or number <= 0 or number in seen:
            raise LibraryError("페이지 번호는 중복되지 않는 양의 정수여야 합니다.")
        seen.add(number)
        text = _text(page["text"], "페이지 원문", MAX_TEXT_CHARS, empty=True)
        total += len(text)
        if total > MAX_TEXT_CHARS:
            raise LibraryError("페이지 원문 총 길이가 보관 한도를 초과했습니다.")
        if not isinstance(page["paragraphs"], list):
            raise LibraryError("문단 정보는 목록이어야 합니다.")
        for paragraph in page["paragraphs"]:
            if not isinstance(paragraph, dict) or not {"id", "page", "text", "start", "end"} <= paragraph.keys():
                raise LibraryError("문단 필수 항목이 누락되었습니다.")
            _text(paragraph["id"], "문단 ID", 256)
            _text(paragraph["text"], "문단 원문", MAX_TEXT_CHARS, empty=True)
            if (paragraph["page"] != number or type(paragraph["start"]) is not int or
                    type(paragraph["end"]) is not int or not 0 <= paragraph["start"] <= paragraph["end"] <= len(text)):
                raise LibraryError("문단의 페이지 또는 원문 위치가 올바르지 않습니다.")
            if paragraph["text"] != text[paragraph["start"]:paragraph["end"]]:
                raise LibraryError("문단 텍스트가 저장된 원문 위치와 다릅니다.")
    for evidence in document["evidence"].values():
        if not isinstance(evidence, dict) or not {"page", "text"} <= evidence.keys():
            raise LibraryError("추출 근거에는 페이지와 원문 정보가 필요합니다.")
        if type(evidence["page"]) is not int or evidence["page"] not in seen:
            raise LibraryError("추출 근거의 페이지가 존재하지 않습니다.")
        _text(evidence["text"], "추출 근거", MAX_TEXT_CHARS, empty=True)
    if document["text"] != "\n\n".join(page["text"] for page in pages):
        raise LibraryError("전체 추출문이 저장된 페이지 원문과 다릅니다.")
    if "financial_facts" in document:
        facts = document["financial_facts"]
        if not isinstance(facts, list):
            raise LibraryError("재무 수치 근거는 목록이어야 합니다.")
        by_number = {page["number"]: page for page in pages}
        for fact in facts:
            required_fact = {"metric", "value", "unit", "page", "paragraph_id", "start", "end", "text"}
            if not isinstance(fact, dict) or not required_fact <= fact.keys():
                raise LibraryError("재무 수치 근거의 필수 항목이 누락되었습니다.")
            for key in ("metric", "value", "unit", "paragraph_id"):
                _text(fact[key], "재무 수치 항목", 2000)
            if type(fact["page"]) is not int or fact["page"] not in by_number:
                raise LibraryError("재무 수치의 원문 페이지가 존재하지 않습니다.")
            paragraph = next((item for item in by_number[fact["page"]]["paragraphs"] if item["id"] == fact["paragraph_id"]), None)
            if (paragraph is None or type(fact["start"]) is not int or type(fact["end"]) is not int or
                    not paragraph["start"] <= fact["start"] <= fact["end"] <= paragraph["end"] or
                    fact["text"] != by_number[fact["page"]]["text"][fact["start"]:fact["end"]]):
                raise LibraryError("재무 수치의 원문 위치가 올바르지 않습니다.")
    if len(_dump(document).encode("utf-8")) > MAX_BACKUP_BYTES // 2:
        raise LibraryError("리포트 한 건의 저장 크기가 너무 큽니다.")
    return document


def _annotations(note, tags, favorite):
    _text(note, "메모", 20000, empty=True)
    if not isinstance(tags, list) or len(tags) > 100:
        raise LibraryError("태그는 최대 100개의 문자열 목록이어야 합니다.")
    for tag in tags:
        _text(tag, "태그", 100)
    if type(favorite) is not bool:
        raise LibraryError("즐겨찾기 값은 참 또는 거짓이어야 합니다.")


def _analysis(value, document):
    if not isinstance(value, dict):
        raise LibraryError("분석 결과는 객체 형식이어야 합니다.")
    _safe_json(value)
    result = deepcopy(value)
    for key in ("engine", "model_fingerprint", "preprocessing_version", "settings"):
        if key not in result:
            raise LibraryError(f"분석 결과에 {key} 항목이 없습니다.")
    if result["engine"] not in ("rules", "model", "demo"):
        raise LibraryError("지원하지 않는 분석 엔진입니다.")
    _text(result["model_fingerprint"], "모델 지문", 2048)
    _text(result["preprocessing_version"], "전처리 버전", 128)
    if not isinstance(result["settings"], dict):
        raise LibraryError("분석 설정은 객체 형식이어야 합니다.")
    result.setdefault("metadata", deepcopy(document["metadata"]))
    _metadata(result["metadata"])
    result.setdefault("document_filename", document["filename"])
    result.setdefault("document_content_hash", document["content_hash"])
    if result["document_content_hash"] != document["content_hash"] or result.get("document_hash", document["content_hash"]) != document["content_hash"]:
        raise LibraryError("분석 결과와 원본 리포트의 해시가 다릅니다.")
    _text(result["document_filename"], "분석 당시 파일 이름", 512)
    snapshot = _document(result.get("document_snapshot", document))
    if snapshot["content_hash"] != document["content_hash"]:
        raise LibraryError("분석 당시 원문 스냅샷과 리포트의 해시가 다릅니다.")
    result["document_snapshot"] = snapshot
    if "created_at" in result:
        _timestamp(result["created_at"])
    if "status" in result and result["status"] not in ("ready", "partial", "ocr_required"):
        raise LibraryError("분석 상태가 올바르지 않습니다.")
    if "warnings" in result:
        _warnings(result["warnings"])
    metrics = result.get("metrics", {})
    if not isinstance(metrics, dict):
        raise LibraryError("분석 지표는 객체 형식이어야 합니다.")
    segments = result.get("segments", [])
    if not isinstance(segments, list) or len(segments) > 100000:
        raise LibraryError("분석 문장 목록의 형식 또는 크기를 확인해 주세요.")
    source_pages = {page["number"]: page for page in snapshot["pages"]}
    segment_ids = set()
    for segment in segments:
        required = {"id", "page", "paragraph_id", "start", "end", "text", "token_count", "negative_score", "rule_hits"}
        if not isinstance(segment, dict) or not required <= segment.keys():
            raise LibraryError("분석 문장의 필수 항목이 누락되었습니다.")
        identifier = _text(segment["id"], "분석 문장 ID", 256)
        if identifier in segment_ids:
            raise LibraryError("분석 문장 ID가 중복되었습니다.")
        segment_ids.add(identifier)
        if type(segment["page"]) is not int or segment["page"] not in source_pages:
            raise LibraryError("분석 문장의 원문 페이지가 존재하지 않습니다.")
        page = source_pages[segment["page"]]
        _text(segment["paragraph_id"], "원문 문단 ID", 256)
        paragraph = next((item for item in page["paragraphs"] if item["id"] == segment["paragraph_id"]), None)
        if (paragraph is None or type(segment["start"]) is not int or type(segment["end"]) is not int or
                not paragraph["start"] <= segment["start"] <= segment["end"] <= paragraph["end"] or
                segment["text"] != page["text"][segment["start"]:segment["end"]]):
            raise LibraryError("분석 문장과 원문 위치가 일치하지 않습니다.")
        if type(segment["token_count"]) is not int or segment["token_count"] < 0:
            raise LibraryError("분석 토큰 수는 0 이상의 정수여야 합니다.")
        if not isinstance(segment["rule_hits"], list):
            raise LibraryError("문구 탐색 결과는 목록이어야 합니다.")
        for hit in segment["rule_hits"]:
            if not isinstance(hit, dict) or not {"label", "keyword", "start", "end"} <= hit.keys():
                raise LibraryError("문구 탐색 결과의 필수 항목이 누락되었습니다.")
            _text(hit["label"], "문구 유형", 2000)
            _text(hit["keyword"], "탐색 문구", 2000)
            if (type(hit["start"]) is not int or type(hit["end"]) is not int or
                    not 0 <= hit["start"] <= hit["end"] <= len(segment["text"]) or
                    segment["text"][hit["start"]:hit["end"]] != hit["keyword"]):
                raise LibraryError("탐색 문구와 원문 위치가 일치하지 않습니다.")
    for key in ("total_chars", "analyzed_chars", "segment_count"):
        if key in metrics and (type(metrics[key]) is not int or metrics[key] < 0):
            raise LibraryError("분석 범위와 문장 수는 0 이상의 정수여야 합니다.")
    if metrics.get("analyzed_chars", 0) > metrics.get("total_chars", MAX_TEXT_CHARS):
        raise LibraryError("분석 문자 수가 전체 원문 길이를 초과합니다.")
    if "segment_count" in metrics and metrics["segment_count"] != len(segments):
        raise LibraryError("분석 문장 수가 실제 저장된 문장 수와 다릅니다.")
    if "unprocessed_pages" in metrics:
        unprocessed = metrics["unprocessed_pages"]
        if not isinstance(unprocessed, list) or any(type(page) is not int or page not in source_pages for page in unprocessed):
            raise LibraryError("미처리 페이지 목록이 올바르지 않습니다.")
    summaries = result.get("summary", [])
    if not isinstance(summaries, list):
        raise LibraryError("발췌 요약은 목록이어야 합니다.")
    for summary in summaries:
        if not isinstance(summary, dict) or not {"text", "page", "paragraph_id"} <= summary.keys():
            raise LibraryError("발췌 요약의 필수 항목이 누락되었습니다.")
        _text(summary["text"], "발췌 요약", MAX_TEXT_CHARS, empty=True)
        if type(summary["page"]) is not int or summary["page"] not in source_pages:
            raise LibraryError("발췌 요약의 원문 페이지가 존재하지 않습니다.")
        page = source_pages[summary["page"]]
        if not any(item["id"] == summary["paragraph_id"] and summary["text"] in item["text"] for item in page["paragraphs"]):
            raise LibraryError("발췌 요약이 저장된 원문 문단에 없습니다.")
    for row in [metrics] + segments:
        if not isinstance(row, dict):
            raise LibraryError("분석 문장 정보는 객체 형식이어야 합니다.")
        score = row.get("negative_score")
        if score is not None:
            if result["engine"] != "model" or type(score) not in (int, float) or not 0 <= score <= 1:
                raise LibraryError("모델 분류 점수는 모델 결과에서만 0~1 범위로 저장할 수 있습니다.")
    if len(_dump(result).encode("utf-8")) > MAX_BACKUP_BYTES // 2:
        raise LibraryError("분석 결과 한 건의 저장 크기가 너무 큽니다.")
    return result


def _dedup(analysis):
    return _hash({key: analysis[key] for key in
                  ("engine", "model_fingerprint", "preprocessing_version", "settings", "metadata")})


class Library:
    """Every operation requires an owner; content is never globally cached."""

    def __init__(self, path=None):
        selected = path if path is not None else os.environ.get("REPORT_LENS_DB") or Path(__file__).resolve().parent / "data" / "library.sqlite3"
        self.path = ":memory:" if str(selected) == ":memory:" else Path(selected).expanduser().resolve()
        self._lock = threading.RLock()
        self._memory = None
        try:
            if self.path != ":memory:":
                self.path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
            else:
                self._memory = self._open()
            with self._connection() as connection:
                version = connection.execute("PRAGMA user_version").fetchone()[0]
                if version > SCHEMA_VERSION or version < 0:
                    raise LibraryError("보관함이 더 최신 버전에서 생성되었습니다. 앱을 업데이트해 주세요.")
                if connection.execute("PRAGMA quick_check").fetchone()[0] != "ok":
                    raise LibraryError("보관함 데이터베이스가 손상되었습니다. 원본을 보존하고 JSON 백업을 복구해 주세요.")
                connection.execute("BEGIN IMMEDIATE")
                # Re-read inside the write lock so concurrent first opens cannot race.
                version = connection.execute("PRAGMA user_version").fetchone()[0]
                if version == 0:
                    tables = connection.execute("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'").fetchall()
                    if tables:
                        raise LibraryError("버전을 확인할 수 없는 데이터베이스입니다. 별도 파일 또는 JSON 백업을 사용해 주세요.")
                    connection.execute("""CREATE TABLE documents (
                        id TEXT PRIMARY KEY, owner TEXT NOT NULL, content_hash TEXT NOT NULL,
                        body TEXT NOT NULL, favorite INTEGER NOT NULL DEFAULT 0,
                        note TEXT NOT NULL DEFAULT '', tags TEXT NOT NULL DEFAULT '[]',
                        created_at TEXT NOT NULL, updated_at TEXT NOT NULL,
                        UNIQUE(owner, content_hash), UNIQUE(owner, id))""")
                    connection.execute("""CREATE TABLE analyses (
                        id TEXT PRIMARY KEY, owner TEXT NOT NULL, document_id TEXT NOT NULL,
                        dedup_key TEXT NOT NULL, body TEXT NOT NULL, created_at TEXT NOT NULL,
                        UNIQUE(owner, document_id, dedup_key),
                        FOREIGN KEY(owner, document_id) REFERENCES documents(owner,id) ON DELETE CASCADE)""")
                    connection.execute("CREATE INDEX analyses_owner_created ON analyses(owner, created_at)")
                    connection.execute(f"PRAGMA user_version={SCHEMA_VERSION}")
                elif version != SCHEMA_VERSION:
                    raise LibraryError("지원하지 않는 보관함 버전입니다.")
                for table, required in (("documents", {"id", "owner", "content_hash", "body", "favorite", "note", "tags", "created_at", "updated_at"}),
                                        ("analyses", {"id", "owner", "document_id", "dedup_key", "body", "created_at"})):
                    columns = {row[1] for row in connection.execute(f"PRAGMA table_info({table})")}
                    if not required <= columns:
                        raise LibraryError("보관함 구조가 올바르지 않습니다. 원본을 보존하고 JSON 백업을 복구해 주세요.")
                connection.commit()
                connection.execute("PRAGMA journal_mode=WAL")
        except OSError as exc:
            raise LibraryError("보관함 폴더를 만들 수 없습니다. 경로와 쓰기 권한을 확인해 주세요.") from exc

    def _open(self):
        connection = sqlite3.connect(str(self.path), timeout=10, check_same_thread=False)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys=ON")
        connection.execute("PRAGMA busy_timeout=10000")
        return connection

    def _permissions(self):
        if self.path == ":memory:":
            return
        for path in (self.path, Path(str(self.path) + "-wal"), Path(str(self.path) + "-shm")):
            try:
                path.chmod(0o600)
            except OSError:
                pass

    @contextmanager
    def _connection(self):
        with self._lock:
            connection = None
            try:
                connection = self._memory if self._memory is not None else self._open()
                with connection:
                    yield connection
            except sqlite3.DatabaseError as exc:
                raise LibraryError("보관함을 읽거나 저장하지 못했습니다. 파일 손상, 잠금, 저장 공간과 권한을 확인하고 JSON 백업을 사용해 주세요.") from exc
            finally:
                if connection is not None and connection is not self._memory:
                    connection.close()
                self._permissions()

    def close(self):
        with self._lock:
            if self._memory is not None:
                self._memory.close()
                self._memory = None

    @staticmethod
    def _document_row(row):
        result = _load(row["body"])
        result.update(id=row["id"], favorite=bool(row["favorite"]), note=row["note"], tags=_load(row["tags"]),
                      created_at=row["created_at"], updated_at=row["updated_at"])
        return result

    @staticmethod
    def _analysis_row(row):
        return {"id": row["id"], "document_id": row["document_id"], "analysis": _load(row["body"]), "created_at": row["created_at"]}

    def save_document(self, owner, document):
        owner, document = _owner(owner), _document(document)
        body, now = _dump(document), _now()
        with self._connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute("SELECT id FROM documents WHERE owner=? AND content_hash=?", (owner, document["content_hash"])).fetchone()
            if row:
                identifier = row["id"]
                connection.execute("UPDATE documents SET body=?, updated_at=? WHERE owner=? AND id=?", (body, now, owner, identifier))
            else:
                identifier = str(uuid4())
                connection.execute("INSERT INTO documents(id,owner,content_hash,body,created_at,updated_at) VALUES(?,?,?,?,?,?)",
                                   (identifier, owner, document["content_hash"], body, now, now))
        return identifier

    def list_documents(self, owner, query="", favorite_only=False):
        owner = _owner(owner)
        query = _text(query, "검색어", 2000, empty=True).casefold()
        if type(favorite_only) is not bool:
            raise LibraryError("즐겨찾기 필터는 참 또는 거짓이어야 합니다.")
        with self._connection() as connection:
            rows = connection.execute("SELECT * FROM documents WHERE owner=? AND (?=0 OR favorite=1) ORDER BY updated_at DESC,id", (owner, int(favorite_only))).fetchall()
            documents = [self._document_row(row) for row in rows]
        if query:
            documents = [document for document in documents if query in " ".join(
                [document["filename"], document["text"], _dump(document["metadata"]), document["note"], " ".join(document["tags"])]).casefold()]
        return documents

    def get_document(self, owner, id):
        owner, id = _owner(owner), _text(id, "리포트 ID", 128)
        with self._connection() as connection:
            row = connection.execute("SELECT * FROM documents WHERE owner=? AND id=?", (owner, id)).fetchone()
            return self._document_row(row) if row else None

    def update_document(self, owner, id, metadata=None, note=None, tags=None, favorite=None):
        owner, id = _owner(owner), _text(id, "리포트 ID", 128)
        if metadata is not None:
            metadata = _metadata(metadata)
        with self._connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute("SELECT * FROM documents WHERE owner=? AND id=?", (owner, id)).fetchone()
            if row is None:
                raise LibraryError("이 보관함에서 리포트를 찾을 수 없습니다.")
            document = self._document_row(row)
            if metadata is not None:
                document["metadata"].update(metadata)
            note = document["note"] if note is None else note
            tags = document["tags"] if tags is None else tags
            favorite = document["favorite"] if favorite is None else favorite
            _annotations(note, tags, favorite)
            connection.execute("UPDATE documents SET body=?,note=?,tags=?,favorite=?,updated_at=? WHERE owner=? AND id=?",
                               (_dump(_document(document)), note, _dump(tags), int(favorite), _now(), owner, id))

    def delete_document(self, owner, id):
        owner, id = _owner(owner), _text(id, "리포트 ID", 128)
        with self._connection() as connection:
            cursor = connection.execute("DELETE FROM documents WHERE owner=? AND id=?", (owner, id))
            if cursor.rowcount == 0:
                raise LibraryError("이 보관함에서 리포트를 찾을 수 없습니다.")

    def save_analysis(self, owner, document_id, analysis):
        owner, document_id = _owner(owner), _text(document_id, "리포트 ID", 128)
        with self._connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute("SELECT * FROM documents WHERE owner=? AND id=?", (owner, document_id)).fetchone()
            if row is None:
                raise LibraryError("이 보관함에서 분석할 리포트를 찾을 수 없습니다.")
            analysis = _analysis(analysis, self._document_row(row))
            key = _dedup(analysis)
            previous = connection.execute("SELECT id FROM analyses WHERE owner=? AND document_id=? AND dedup_key=?", (owner, document_id, key)).fetchone()
            if previous:
                return previous["id"]
            identifier = str(uuid4())
            connection.execute("INSERT INTO analyses(id,owner,document_id,dedup_key,body,created_at) VALUES(?,?,?,?,?,?)",
                               (identifier, owner, document_id, key, _dump(analysis), _now()))
            return identifier

    def list_analyses(self, owner, document_id=None):
        owner = _owner(owner)
        if document_id is not None:
            _text(document_id, "리포트 ID", 128)
        with self._connection() as connection:
            rows = connection.execute("SELECT * FROM analyses WHERE owner=? AND (? IS NULL OR document_id=?) ORDER BY created_at DESC,id", (owner, document_id, document_id)).fetchall()
            return [self._analysis_row(row) for row in rows]

    def get_analysis(self, owner, run_id):
        owner, run_id = _owner(owner), _text(run_id, "분석 ID", 128)
        with self._connection() as connection:
            row = connection.execute("SELECT * FROM analyses WHERE owner=? AND id=?", (owner, run_id)).fetchone()
            return self._analysis_row(row) if row else None

    def export_backup(self, owner):
        owner = _owner(owner)
        with self._connection() as connection:
            connection.execute("BEGIN")
            documents = [self._document_row(row) for row in connection.execute("SELECT * FROM documents WHERE owner=? ORDER BY id", (owner,))]
            analyses = [self._analysis_row(row) for row in connection.execute("SELECT * FROM analyses WHERE owner=? ORDER BY id", (owner,))]
        payload = _dump({"format": "report-lens", "schema_version": SCHEMA_VERSION, "exported_at": _now(),
                         "documents": documents, "analyses": analyses})
        if len(documents) > MAX_DOCUMENTS or len(analyses) > MAX_ANALYSES or len(payload.encode("utf-8")) > MAX_BACKUP_BYTES:
            raise LibraryError("전체 백업이 한도(20MB, 문서 2,000건, 분석 10,000건)를 넘었습니다. 개별 분석 JSON을 내보내 보관함을 정리해 주세요.")
        return payload

    def import_backup(self, owner, payload):
        owner = _owner(owner)
        if not isinstance(payload, str):
            raise LibraryError("JSON 백업은 UTF-8 텍스트 20MB 이하여야 합니다.")
        _text(payload, "JSON 백업", MAX_BACKUP_BYTES, empty=True)
        if len(payload.encode("utf-8")) > MAX_BACKUP_BYTES:
            raise LibraryError("JSON 백업은 UTF-8 텍스트 20MB 이하여야 합니다.")
        backup = _load(payload)
        if (not isinstance(backup, dict) or backup.get("format") != "report-lens" or
                type(backup.get("schema_version")) is not int or backup["schema_version"] != SCHEMA_VERSION):
            raise LibraryError("지원하지 않는 백업 형식 또는 버전입니다.")
        _timestamp(backup.get("exported_at"))
        documents, analyses = backup.get("documents"), backup.get("analyses")
        if not isinstance(documents, list) or len(documents) > MAX_DOCUMENTS or not isinstance(analyses, list) or len(analyses) > MAX_ANALYSES:
            raise LibraryError("백업은 리포트 2,000건, 분석 10,000건 이하여야 합니다.")
        prepared_documents, prepared_analyses, source_ids, hashes, run_ids = [], [], {}, set(), set()
        for row in documents:
            if not isinstance(row, dict):
                raise LibraryError("백업의 리포트 형식이 올바르지 않습니다.")
            identifier = _text(row.get("id"), "백업 리포트 ID", 128)
            document = _document(row)
            if identifier in source_ids or document["content_hash"] in hashes:
                raise LibraryError("백업에 중복된 리포트 ID 또는 해시가 있습니다.")
            _annotations(row.get("note", ""), row.get("tags", []), row.get("favorite", False))
            source_ids[identifier] = document
            hashes.add(document["content_hash"])
            created_at = _timestamp(row.get("created_at"))
            updated_at = _timestamp(row.get("updated_at"))
            prepared_documents.append((identifier, document, row.get("note", ""), row.get("tags", []), row.get("favorite", False), created_at, updated_at))
        for row in analyses:
            if not isinstance(row, dict):
                raise LibraryError("백업의 분석 형식이 올바르지 않습니다.")
            identifier = _text(row.get("id"), "백업 분석 ID", 128)
            document_id = _text(row.get("document_id"), "백업 원본 리포트 ID", 128)
            if identifier in run_ids or document_id not in source_ids:
                raise LibraryError("백업 분석의 ID 또는 원본 리포트 연결이 올바르지 않습니다.")
            run_ids.add(identifier)
            prepared_analyses.append((document_id, _analysis(row.get("analysis"), source_ids[document_id]), _timestamp(row.get("created_at"))))
        counts = {"documents": 0, "analyses": 0, "skipped_documents": 0, "skipped_analyses": 0}
        remap = {}
        # Only after every record passes validation can any write begin.
        with self._connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            for source_id, document, note, tags, favorite, created_at, updated_at in prepared_documents:
                previous = connection.execute("SELECT id FROM documents WHERE owner=? AND content_hash=?", (owner, document["content_hash"])).fetchone()
                if previous:
                    remap[source_id] = previous["id"]
                    counts["skipped_documents"] += 1
                    continue
                identifier = str(uuid4())
                remap[source_id] = identifier
                connection.execute("INSERT INTO documents(id,owner,content_hash,body,note,tags,favorite,created_at,updated_at) VALUES(?,?,?,?,?,?,?,?,?)",
                                   (identifier, owner, document["content_hash"], _dump(document), note, _dump(tags), int(favorite), created_at, updated_at))
                counts["documents"] += 1
            for source_id, analysis, created_at in prepared_analyses:
                document_id, key = remap[source_id], _dedup(analysis)
                previous = connection.execute("SELECT id FROM analyses WHERE owner=? AND document_id=? AND dedup_key=?", (owner, document_id, key)).fetchone()
                if previous:
                    counts["skipped_analyses"] += 1
                    continue
                connection.execute("INSERT INTO analyses(id,owner,document_id,dedup_key,body,created_at) VALUES(?,?,?,?,?,?)",
                                   (str(uuid4()), owner, document_id, key, _dump(analysis), created_at))
                counts["analyses"] += 1
        return counts
