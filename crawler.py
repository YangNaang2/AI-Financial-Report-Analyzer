"""Bounded, resumable public Naver research collection.

The current official research page uses /api/stockSecurity/researches/v2.
Legacy HTML parsing remains a fallback for an older deployment. A changed or
unavailable endpoint is reported explicitly, never counted as a successful PDF.
"""

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import time
from urllib.parse import urlencode, urljoin, urlsplit, urlunsplit
from uuid import uuid4

import requests
from bs4 import BeautifulSoup

API_URL = "https://stock.naver.com/api/stockSecurity/researches/v2/company"
LIST_URL = "https://stock.naver.com/research/company"
LEGACY_URL = "https://finance.naver.com/research/company_list.naver"
MAX_PDF_BYTES = 20 * 1024 * 1024
MAX_PAGE_BYTES = 2 * 1024 * 1024
REQUEST_TIMEOUT = (5, 20)
REQUEST_DELAY = 1.0
MAX_RETRIES = 1


class CollectionError(ValueError):
    """A source response or local collection manifest is not usable."""


def _now():
    return datetime.now(timezone.utc).isoformat()


def _safe_url(url, base=LIST_URL):
    absolute = urljoin(base, str(url))
    parsed = urlsplit(absolute)
    host = (parsed.hostname or "").lower()
    if (parsed.scheme not in ("http", "https") or parsed.username or parsed.password
            or parsed.port not in (None, 80, 443)
            or not any(host == domain or host.endswith("." + domain) for domain in ("naver.com", "pstatic.net"))):
        raise CollectionError("공식 네이버/첨부 CDN 주소가 아닌 링크는 다운로드하지 않습니다.")
    return urlunsplit(("https", host, parsed.path, parsed.query, ""))


def _filename(title, source_url):
    safe = re.sub(r'[\\/:*?"<>|\x00-\x1f\x7f]', "_", str(title))
    safe = re.sub(r"\s+", " ", safe).strip(" .")[:80].strip(" .") or "report"
    if safe.upper().split(".")[0] in {"CON", "PRN", "AUX", "NUL", *(f"COM{i}" for i in range(1, 10)), *(f"LPT{i}" for i in range(1, 10))}:
        safe = "report_" + safe
    return f"{safe}--{hashlib.sha256(source_url.encode()).hexdigest()[:16]}.pdf"


def _atomic_json(path, data):
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _get(url, *, max_bytes, pdf=False):
    """Bound network waits, redirects, retries, streamed bytes and allowed hosts."""
    url = _safe_url(url)
    headers = {"User-Agent": "Mozilla/5.0 (compatible; LocalResearchCollector/2.0)",
               "Referer": LIST_URL, "Accept": "application/pdf" if pdf else "application/json,text/html"}
    for attempt in range(MAX_RETRIES + 1):
        current = url
        try:
            for redirect in range(4):
                time.sleep(REQUEST_DELAY)
                response = requests.get(current, headers=headers, timeout=REQUEST_TIMEOUT,
                                        stream=True, allow_redirects=False)
                try:
                    if response.status_code in (301, 302, 303, 307, 308):
                        if redirect == 3 or not response.headers.get("Location"):
                            raise CollectionError("리디렉션이 반복되거나 대상 주소가 없습니다.")
                        current = _safe_url(response.headers["Location"], current)
                        continue
                    if response.status_code in (429, 500, 502, 503, 504) and attempt < MAX_RETRIES:
                        raise requests.ConnectionError(f"HTTP {response.status_code}")
                    response.raise_for_status()
                    length = response.headers.get("Content-Length")
                    if length and int(length) > max_bytes:
                        raise CollectionError("응답이 허용된 최대 파일 크기를 초과했습니다.")
                    content, size = [], 0
                    for chunk in response.iter_content(64 * 1024):
                        if not chunk:
                            continue
                        size += len(chunk)
                        if size > max_bytes:
                            raise CollectionError("응답이 허용된 최대 파일 크기를 초과했습니다.")
                        content.append(chunk)
                    raw = b"".join(content)
                    if pdf and not raw.lstrip().startswith(b"%PDF-"):
                        raise CollectionError("다운로드 응답이 PDF가 아닙니다. HTML 오류 페이지를 저장하지 않았습니다.")
                    return raw, current, response.headers.get("Content-Type", "")
                finally:
                    response.close()
        except (requests.Timeout, requests.ConnectionError):
            if attempt == MAX_RETRIES:
                raise
            time.sleep(min(2 ** attempt, 2))
    raise CollectionError("자료 응답을 가져오지 못했습니다.")


def _parse_legacy(html, source_url):
    soup = BeautifulSoup(html, "html.parser")
    reports = []
    for row in soup.select("table.type_1 tr"):
        attachment, title = row.select_one("td.file a[href]"), row.select_one("td:nth-child(2) a")
        if not attachment or not title:
            continue
        day = re.search(r"(?<![0-9])([0-9]{2,4})[./-]([0-9]{2})[./-]([0-9]{2})(?![0-9])", row.get_text(" ", strip=True))
        published = None
        if day:
            year = day[1] if len(day[1]) == 4 else "20" + day[1]
            published = f"{year}-{day[2]}-{day[3]}"
        code = re.search(r"[?&](?:code|itemCode)=([0-9]{6})(?:&|$)", str(row))
        reports.append({"source_url": _safe_url(title.get("href") or source_url, source_url),
                        "pdf_url": _safe_url(attachment["href"], source_url),
                        "title": title.get_text(" ", strip=True), "published_at": published,
                        "ticker": code[1] if code else None})
    return reports


def _listing(page):
    url = API_URL + "?" + urlencode({"index": page - 1, "size": 15})
    try:
        raw, _, _ = _get(url, max_bytes=MAX_PAGE_BYTES)
        payload = json.loads(raw.decode("utf-8-sig"))
        if not isinstance(payload, dict) or not isinstance(payload.get("items"), list):
            raise CollectionError("현재 네이버 리포트 API의 응답 구조가 변경되었습니다.")
        reports = []
        for item in payload["items"]:
            if not isinstance(item, dict) or not item.get("nid") or not item.get("title"):
                raise CollectionError("현재 리포트 목록의 필수 항목이 없습니다.")
            identifier = str(item["nid"])
            if not re.fullmatch(r"[0-9]+", identifier):
                raise CollectionError("리포트 식별자 형식이 변경되었습니다.")
            reports.append({"source_url": f"{LIST_URL}/{identifier}", "nid": identifier,
                            "pdf_url": item.get("attachUrl"), "title": item["title"],
                            "published_at": item.get("writeDate"), "ticker": item.get("itemCode"),
                            "broker": item.get("brokerName")})
        return reports, bool(payload.get("hasNext", bool(reports)))
    except (json.JSONDecodeError, UnicodeError, CollectionError, requests.HTTPError) as original:
        # The old listing is only a public compatibility source. Authentication
        # errors are surfaced directly instead of attempting another route.
        if isinstance(original, requests.HTTPError) and getattr(original.response, "status_code", None) in (401, 403):
            raise
        raw, final_url, _ = _get(LEGACY_URL + "?page=" + str(page), max_bytes=MAX_PAGE_BYTES)
        reports = _parse_legacy(raw, final_url)
        if not reports:
            raise CollectionError(f"리포트 목록을 확인할 수 없습니다. 사이트 구조/응답을 확인해 주세요. {original}") from original
        return reports, True


def download_naver_reports(save_dir="./pdf_data", start_page=1, end_page=1, progress=None):
    """Collect pages into a resume manifest, retaining origin and outcome per file."""
    if (type(start_page) is not int or type(end_page) is not int or start_page < 1
            or end_page < start_page or end_page - start_page >= 100 or end_page > 10000):
        raise CollectionError("페이지는 1~10000 범위에서 한 번에 최대 100페이지를 순서대로 지정하세요.")
    directory = Path(save_dir).expanduser().resolve()
    directory.mkdir(parents=True, exist_ok=True)
    manifest_path = directory / "collection_manifest.json"
    if manifest_path.exists():
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if (manifest.get("schema_version") != 1 or not isinstance(manifest.get("reports"), dict)
                    or any(not isinstance(row, dict) for row in manifest["reports"].values())):
                raise ValueError
        except (ValueError, OSError, AttributeError) as exc:
            raise CollectionError("기존 수집 이력이 손상되었거나 지원하지 않는 버전입니다. 덮어쓰지 않았습니다.") from exc
    else:
        manifest = {"schema_version": 1, "source": LIST_URL, "reports": {}}
    result = {"status": "complete", "count": 0, "downloaded": [], "skipped": [], "errors": [],
              "source": API_URL, "manifest_path": str(manifest_path), "pages_visited": 0}
    for page in range(start_page, end_page + 1):
        if progress:
            progress(f"네이버 기업 리포트 {page}페이지 확인")
        try:
            reports, has_next = _listing(page)
            result["pages_visited"] += 1
        except (requests.RequestException, CollectionError, ValueError, OSError) as exc:
            result["errors"].append({"page": page, "reason": str(exc)})
            break
        for report_index, source in enumerate(reports, 1):
            if progress:
                progress(f"PDF {report_index}/{len(reports)} · {source['title']}")
            key = hashlib.sha256(source["source_url"].encode()).hexdigest()
            previous = manifest["reports"].get(key, {})
            filename = previous.get("filename")
            existing = directory / filename if isinstance(filename, str) and Path(filename).name == filename else None
            if existing and existing.is_file() and previous.get("status") == "downloaded":
                if existing.stat().st_size <= MAX_PDF_BYTES:
                    raw = existing.read_bytes()
                    if raw.lstrip().startswith(b"%PDF-") and hashlib.sha256(raw).hexdigest() == previous.get("sha256"):
                        result["skipped"].append(dict(previous, reason="이미 검증하여 저장된 자료"))
                        continue
            entry = dict(source, status="pending", collected_at=_now())
            try:
                pdf_url = source.get("pdf_url")
                if not pdf_url:
                    raw, _, _ = _get(API_URL + "/" + source["nid"], max_bytes=MAX_PAGE_BYTES)
                    detail = json.loads(raw.decode("utf-8-sig"))
                    pdf_url = detail.get("attachUrl") if isinstance(detail, dict) else None
                if not pdf_url:
                    raise CollectionError("첨부 PDF 주소가 없습니다.")
                pdf_url = _safe_url(pdf_url)
                filename = _filename(source["title"], source["source_url"])
                target = directory / filename
                # A conflicting or corrupt existing file is retained for review.
                if target.exists():
                    target = target.with_name(target.stem + "-" + uuid4().hex[:8] + ".pdf")
                raw, final_url, _ = _get(pdf_url, max_bytes=MAX_PDF_BYTES, pdf=True)
                temporary = target.with_name("." + target.name + ".part")
                try:
                    with temporary.open("xb") as handle:
                        handle.write(raw)
                        handle.flush()
                        os.fsync(handle.fileno())
                    # target includes a URL hash, and any existing name was
                    # disambiguated above; do not replace unrelated local files.
                    os.link(temporary, target)
                finally:
                    temporary.unlink(missing_ok=True)
                entry.update(status="downloaded", filename=target.name, pdf_url=final_url,
                             size_bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest())
                result["downloaded"].append(entry)
                result["count"] += 1
            except (requests.RequestException, CollectionError, ValueError, OSError) as exc:
                entry.update(status="error", reason=str(exc))
                result["errors"].append(entry)
            manifest["reports"][key] = entry
            manifest["updated_at"] = _now()
            _atomic_json(manifest_path, manifest)
        if not has_next:
            break
    if not manifest_path.exists():
        manifest["updated_at"] = _now()
        _atomic_json(manifest_path, manifest)
    if result["errors"]:
        result["status"] = "partial" if result["downloaded"] or result["skipped"] else "failed"
    return result


collect_reports = download_naver_reports


if __name__ == "__main__":
    from main import main
    raise SystemExit(main(["collect", *__import__("sys").argv[1:]]))
