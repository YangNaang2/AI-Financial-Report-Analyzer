"""Source-preserving PDF/text extraction and shared, coverage-aware chunking."""
from datetime import date
from decimal import Decimal
from hashlib import sha256
from io import BytesIO
from pathlib import Path
import re

EXTRACTION_VERSION = "2"
# Header metadata rules change independently of the text/chunking that trained
# models depend on, so they carry their own version for caches.
METADATA_VERSION = "2"
DEFAULT_MAX_BYTES = 20 * 1024 * 1024


class DocumentError(ValueError):
    """A document cannot be safely extracted or divided for analysis."""


def clean_financial_report(text):
    """Normalize spacing while preserving dates, codes, prices and percentages."""
    if not isinstance(text, str):
        raise DocumentError("텍스트 입력 형식이 올바르지 않습니다.")
    text = text.replace("\r\n", "\n").replace("\r", "\n").replace("\x00", "")
    text = re.sub(r"[\u200b\ufeff]", "", text)
    text = re.sub(r"[^\S\n]+", " ", text)
    text = "\n".join(line.strip() for line in text.split("\n"))
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def _paragraphs(text, page):
    result = []
    for index, match in enumerate(re.finditer(r".+?(?:\n\n|$)", text, flags=re.S), 1):
        if match.group().strip():
            result.append(dict(id=f"p{page}-para{index}", page=page,
                               text=match.group(), start=match.start(), end=match.end()))
    return result


def _date_value(value):
    match = re.search(r"(\d{4})\s*[./년-]\s*(\d{1,2})\s*[./월-]\s*(\d{1,2})", value)
    if not match:
        return None
    try:
        return date(*(int(part) for part in match.groups())).isoformat()
    except ValueError:
        return None


def _english_date(value):
    months = ("jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec")
    match = re.search(r"\b(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.?\s+(\d{1,2}),?\s+(\d{4})\b", value, re.I)
    if not match:
        return None
    try:
        return date(int(match[3]), months.index(match[1].lower()) + 1, int(match[2])).isoformat()
    except ValueError:
        return None


BROKERS = ("미래에셋증권", "삼성증권", "한국투자증권", "NH투자증권", "KB증권", "신한투자증권",
           "하나증권", "키움증권", "대신증권", "한화투자증권", "유안타증권", "메리츠증권",
           "유진투자증권", "현대차증권", "교보증권", "IBK투자증권", "SK증권", "LS증권", "iM증권",
           "DS투자증권", "DB증권", "DB금융투자", "BNK투자증권", "다올투자증권", "부국증권", "상상인증권",
           "케이프투자증권", "한양증권", "흥국증권", "카카오페이증권", "토스증권", "리딩투자증권", "Kiwoom Securities")
OPINION_WORDS = (r"Strong\s*Buy|Trading\s*Buy|Buy|Outperform|Overweight|Hold|Neutral|Marketperform|"
                 r"Underperform|Underweight|Reduce|Sell|Not\s*Rated|매수|중립|보유|매도")
OPINION_CLASSES = {"strongbuy": "BUY", "tradingbuy": "BUY", "buy": "BUY", "outperform": "BUY", "overweight": "BUY",
                   "매수": "BUY", "hold": "HOLD", "neutral": "HOLD", "marketperform": "HOLD", "중립": "HOLD",
                   "보유": "HOLD", "underperform": "SELL", "underweight": "SELL", "reduce": "SELL", "sell": "SELL",
                   "매도": "SELL", "notrated": "NR"}
# Stated revision next to a target price, e.g. "목표주가 (하향)", "24,000원 (M)", "TP 42,000원 유지".
TARGET_CHANGES = (("상향", r"상향|\((?:U|Up|Upgrade|Raise)\)|▲"), ("하향", r"하향|\((?:D|Down|Downgrade|Cut)\)|▼"),
                  ("유지", r"유지|\((?:M|Maintain|Unchanged)\)"), ("신규", r"신규|\((?:I|Initiate|New)\)"))


def opinion_class(value):
    """Group equivalent opinions (BUY, Buy, 매수) so they do not count as a conflict."""
    return OPINION_CLASSES.get(re.sub(r"\s+", "", str(value or "")).lower())


def _metadata(pages):
    candidates = {key: [] for key in ("company", "ticker", "broker", "report_date", "opinion", "target_price",
                                      "current_price", "target_change", "previous_target_price")}
    parse_warnings = []

    def add(key, value, page, excerpt, unit=None):
        if value:
            candidates[key].append((str(value).strip(), page, excerpt.strip()[:300], unit))

    # Metadata belongs to the report header, not historical figures in its body.
    for page in pages[:2]:
        for index, line in enumerate(page["text"].splitlines()[:35]):
            number = page["number"]
            cover = number == 1 and index < 15
            pair = re.search(r"([가-힣A-Za-z][가-힣A-Za-z0-9& .·-]{0,35})\s*\(([0-9]{6})(?:\.(?:KS|KQ))?\)", line)
            if pair:
                add("company", pair[1], number, line)
                add("ticker", pair[2], number, line)
            explicit = re.search(r"(?:종목코드|티커|Ticker)\s*[:：]?\s*([0-9]{6})(?![0-9])", line, re.I)
            if explicit:
                add("ticker", explicit[1], number, line)
            # Codes printed without a company name: "BUY (유지) (003490)" or "462870" on its own line.
            for code in re.findall(r"\(\s*A?([0-9]{6})(?:\.K[SQ])?\s*\)", line):
                add("ticker", code, number, line)
            bare = re.fullmatch(r"\s*A?([0-9]{6})\s*", line) if cover else None
            if bare:
                add("ticker", bare[1], number, line)
            explicit = re.search(r"(?:기업명|종목명)\s*[:：]\s*(.{1,40})$", line)
            if explicit:
                add("company", explicit[1], number, line)
            for broker in BROKERS:
                if broker in line:
                    add("broker", broker, number, line)
            if re.search(r"발간일|발행일|작성일|보고서일|Report\s*Date|^Date\b", line, re.I):
                add("report_date", _date_value(line), number, line)
            elif number == 1 and re.fullmatch(r"\s*\d{4}\s*[./년-]\s*\d{1,2}\s*[./월-]\s*\d{1,2}\s*일?\s*", line):
                add("report_date", _date_value(line), number, line)
            elif cover and len(line) <= 60:
                # Cover lines such as "2026년 10월 16일 I 기업분석" or "Company Report 2026.10.16".
                add("report_date", _date_value(line) or _english_date(line), number, line)
            opinion = re.search(rf"(?:투자의견|Investment\s*Opinion)\s*[:：]?\s*({OPINION_WORDS})(?![A-Za-z가-힣])", line, re.I)
            if opinion:
                add("opinion", opinion[1], number, line)
            elif cover:
                # Covers usually print the opinion alone: "BUY (유지)", "BUY(Maintain)".
                standalone = re.match(rf"\s*({OPINION_WORDS})(?=\s|\(|$)", line, re.I)
                if standalone:
                    add("opinion", standalone[1], number, line)
            for key, label in (("target_price", r"목표\s*주가|Target\s*Price|(?<![A-Za-z])TP(?![A-Za-z])"),
                               ("current_price", r"현재\s*(?:주가|가)|Current\s*Price|(?<![가-힣A-Za-z])주가(?=\s*\()")):
                found = False
                for match in re.finditer(
                    rf"(?:{label})\s*(?:\((?P<label_unit>[^)]{{0,25}})\))?\s*[:：]?\s*"
                    r"(?P<prefix>KRW|USD|\$)?\s*(?P<value>(?:[0-9]{1,3}(?:,[0-9]{3})+|[0-9]+)(?:\.[0-9]+)?)(?![0-9,.])"
                    r"[ \t]*(?P<unit>백만\s*원|천\s*원|만\s*원|원|KRW|USD|달러|\$)?(?![0-9A-Za-z가-힣,.])", line, re.I,
                ):
                    if Decimal(match["value"].replace(",", "")) <= 0:
                        continue
                    found = True
                    if re.match(r"\s*[0-9][0-9,.]*\s*(?:조\s*원|억\s*원|백만\s*원|천\s*원|만\s*원|원)", line[match.end():]):
                        parse_warnings.append(f"{key}: 복합 금액 표현은 원문에서 직접 확인하세요.")
                        continue
                    unit = match["unit"] or match["prefix"]
                    label_unit = re.sub(r"\s+", "", match["label_unit"] or "")
                    if not unit and label_unit in ("원", "천원", "만원", "KRW", "USD", "$"):
                        unit = label_unit
                    if unit:
                        unit = re.sub(r"\s+", "", unit)
                    value = match["value"].replace(",", "")
                    # "직전 목표주가 24,000원" is the previous target, not a second candidate.
                    previous = key == "target_price" and re.search(r"(?:직전|이전|기존|종전|Previous|Prior|Old)\s*$",
                                                                    line[max(0, match.start() - 12):match.start()], re.I)
                    add("previous_target_price" if previous else key, f"{value} {unit}" if unit else value,
                        number, line, unit=unit or "미확인")
                    if key == "target_price" and not previous:
                        nearby = (match["label_unit"] or "") + " " + line[match.end():match.end() + 16]
                        stated = next((name for name, pattern in TARGET_CHANGES if re.search(pattern, nearby, re.I)), None)
                        add("target_change", stated, number, line)
                if not found and re.search(rf"(?:{label}).*[0-9]", line, re.I):
                    parse_warnings.append(f"{key}: 숫자와 단위 형식이 불명확해 자동 입력하지 않았습니다.")
    values, evidence, warnings = {}, {}, parse_warnings
    prices = ("target_price", "current_price", "previous_target_price")
    for key, matches in candidates.items():
        if key == "opinion":
            unique = {opinion_class(value) or value for value, _, _, _ in matches}
        elif key in prices:
            # A unit-less table cell repeating the cover's "41,000원" is the same figure.
            unique = {value.split()[0] for value, _, _, _ in matches}
            matches = sorted(matches, key=lambda match: match[3] == "미확인")
        else:
            unique = {value for value, _, _, _ in matches}
        values[key] = matches[0][0] if len(unique) == 1 else None
        if len(unique) == 1:
            value, page, excerpt, unit = matches[0]
            evidence[key] = dict(page=page, text=excerpt)
            if unit is not None:
                evidence[key]["unit"] = unit
        elif len(unique) > 1:
            warnings.append(f"{key}: 서로 다른 후보가 있어 자동 입력하지 않았습니다. 원문을 확인하세요.")
    return values, evidence, warnings


def _financial_facts(pages):
    """Extract only same-line labeled figures with explicit units, without conversion."""
    units = r"조\s*원|억\s*원|백만\s*원|천\s*원|만\s*원|원|%|배|KRW|USD|달러"
    pattern = re.compile(
        r"(?P<metric>당기순이익|영업이익률|영업이익|영업손실|매출액|매출|주당순이익|순이익|EPS|PER|PBR|ROE|부채비율)"
        rf"[ \t]*(?:\((?P<label_unit>{units})\))?[ \t]*[:：]?[ \t]*"
        r"(?P<value>[+-]?(?:[0-9]{1,3}(?:,[0-9]{3})+|[0-9]+)(?:\.[0-9]+)?)(?![0-9,.])"
        rf"[ \t]*(?P<unit>{units})?(?![0-9A-Za-z가-힣,.])", re.I,
    )
    facts, warnings = [], []
    for page in pages:
        for paragraph in page["paragraphs"]:
            for line in paragraph["text"].splitlines():
                if re.search(r"(?:매출액|매출|영업이익|영업손실|순이익|EPS|PER|PBR|ROE|부채비율).*?[0-9]+,[0-9]{1,2}(?![0-9])", line, re.I):
                    warnings.append(f"{page['number']}페이지: 재무 수치의 구분 기호가 불명확해 원문 확인이 필요합니다.")
            for match in pattern.finditer(paragraph["text"]):
                unit = match["unit"] or match["label_unit"]
                if not unit:
                    continue
                if re.match(rf"[ \t]*[+-]?[0-9][0-9,.]*[ \t]*(?:{units})", paragraph["text"][match.end():]):
                    warnings.append(f"{page['number']}페이지 {match['metric']}: 복합 금액 표현은 원문에서 직접 확인하세요.")
                    continue
                facts.append(dict(metric=match["metric"], value=match["value"].replace(",", ""),
                                  unit=re.sub(r"\s+", "", unit), page=page["number"], paragraph_id=paragraph["id"],
                                  start=paragraph["start"] + match.start(), end=paragraph["start"] + match.end(),
                                  text=match.group()))
    return facts, warnings


def _document(raw, filename, texts, *, warnings=None):
    pages = []
    for number, text in enumerate(texts, 1):
        normalized = clean_financial_report(text or "")
        pages.append(dict(number=number, text=normalized, paragraphs=_paragraphs(normalized, number)))
    metadata, evidence, notes = _metadata(pages)
    financial_facts, fact_warnings = _financial_facts(pages)
    notes.extend(fact_warnings)
    notes = list(warnings or []) + notes
    missing = [page["number"] for page in pages if not page["text"]]
    if missing:
        notes.append(f"텍스트를 추출하지 못한 페이지: {', '.join(map(str, missing))}. 스캔 문서라면 OCR이 필요합니다.")
    status = "ocr_required" if len(missing) == len(pages) else "partial" if missing else "ready"
    return dict(content_hash=sha256(raw).hexdigest(), filename=str(filename),
                text="\n\n".join(page["text"] for page in pages), pages=pages,
                metadata=metadata, evidence=evidence, warnings=notes, status=status,
                extraction_version=EXTRACTION_VERSION, financial_facts=financial_facts)


def document_from_text(text, filename="입력.txt"):
    if not isinstance(text, str) or not text.strip():
        raise DocumentError("분석할 텍스트를 입력하세요.")
    raw = text.encode("utf-8")
    if len(raw) > DEFAULT_MAX_BYTES:
        raise DocumentError("텍스트가 최대 크기 20MB를 초과했습니다.")
    return _document(raw, filename, [text])


def extract_document(source, filename=None, max_bytes=DEFAULT_MAX_BYTES, max_pages=100):
    """Extract a local PDF or explicitly named UTF-8 text file; never invoke OCR."""
    if not isinstance(max_bytes, int) or max_bytes <= 0 or not isinstance(max_pages, int) or max_pages <= 0:
        raise DocumentError("파일 크기와 페이지 제한은 양의 정수여야 합니다.")
    if isinstance(source, (str, Path)):
        path = Path(source)
        filename = filename or path.name
        try:
            if path.stat().st_size > max_bytes:
                raise DocumentError("파일이 허용된 최대 크기를 초과했습니다.")
            raw = path.read_bytes()
        except OSError as exc:
            raise DocumentError("문서 파일을 읽을 수 없습니다.") from exc
    elif isinstance(source, bytes):
        raw = source
        filename = filename or "입력.pdf"
    else:
        raise DocumentError("파일 경로 또는 파일 내용을 전달하세요.")
    if not raw or len(raw) > max_bytes:
        raise DocumentError("파일이 비어 있거나 허용된 최대 크기를 초과했습니다.")
    if Path(filename).suffix.lower() in (".txt", ".md"):
        try:
            text = raw.decode("utf-8-sig")
        except UnicodeError as exc:
            raise DocumentError("텍스트 파일은 UTF-8 형식이어야 합니다.") from exc
        if not text.strip():
            raise DocumentError("문서에 분석할 텍스트가 없습니다.")
        return _document(raw, filename, [text])
    if not raw.lstrip().startswith(b"%PDF-"):
        raise DocumentError("올바른 PDF 파일이 아닙니다.")
    try:
        try:
            import pdfplumber
        except ImportError as exc:
            raise DocumentError("PDF 추출에는 pdfplumber 설치가 필요합니다.") from exc
        with pdfplumber.open(BytesIO(raw)) as pdf:
            if getattr(pdf.doc, "encryption", None) is not None or not getattr(pdf.doc, "is_extractable", True):
                raise DocumentError("암호화되었거나 텍스트 추출이 제한된 PDF입니다.")
            if len(pdf.pages) > max_pages:
                raise DocumentError(f"PDF가 최대 {max_pages}페이지를 초과했습니다.")
            texts, warnings = [], []
            for number, page in enumerate(pdf.pages, 1):
                try:
                    texts.append(page.extract_text() or "")
                except Exception:
                    texts.append("")
                    warnings.append(f"{number}페이지를 읽는 중 오류가 발생했습니다. 해당 페이지는 분석하지 않았습니다.")
    except DocumentError:
        raise
    except Exception as exc:
        message = "암호화된 PDF는 열 수 없습니다." if "password" in type(exc).__name__.lower() else "PDF가 손상되었거나 지원하지 않는 형식입니다."
        raise DocumentError(message) from exc
    if not texts:
        raise DocumentError("PDF에 페이지가 없습니다.")
    return _document(raw, filename, texts, warnings=warnings)


def extract_text_from_pdf(file_path):
    """Compatibility wrapper; source-preserving extraction replaces regex deletion."""
    return extract_document(file_path)["text"]


def _token_offsets(text, tokenizer):
    if tokenizer is None:
        return [match.span() for match in re.finditer(r"\S{1,32}", text)]
    try:
        encoded = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True, truncation=False, verbose=False)
        offsets = [(int(start), int(end)) for start, end in encoded["offset_mapping"] if end > start]
    except Exception as exc:
        raise DocumentError("원문 위치를 보존하려면 offset_mapping을 지원하는 빠른 토크나이저가 필요합니다.") from exc
    if not offsets and text.strip():
        raise DocumentError("토크나이저가 입력 텍스트를 처리하지 못했습니다.")
    return offsets


def _covered(intervals):
    total, end = 0, 0
    for start, stop in sorted(intervals):
        total += max(0, stop - max(start, end))
        end = max(end, stop)
    return total


def chunk_document(document, tokenizer=None, max_tokens=256, overlap=32, max_segments=500):
    """Return source-mapped segments and union coverage, never silent truncation.

    Offsets are relative to each normalized page. Tokenizer-free chunking uses
    whitespace and 32-character units; transformer chunks use real token offsets.
    max_tokens includes special tokens; overlap counts content tokens.
    """
    if any(isinstance(value, bool) or not isinstance(value, int) for value in (max_tokens, overlap, max_segments)):
        raise DocumentError("구간 설정은 정수여야 합니다.")
    specials = tokenizer.num_special_tokens_to_add(pair=False) if tokenizer is not None else 0
    budget = max_tokens - specials
    if not 1 <= max_segments <= 10000 or not 1 <= budget <= 8192 or not 0 <= overlap < budget:
        raise DocumentError("구간 크기·중첩·최대 구간 수 설정을 확인하세요.")
    pages = document.get("pages", [])
    segments, warnings, coverage = [], [], {page["number"]: [] for page in pages}
    if tokenizer is None:
        warnings.append("모델 토크나이저 없이 공백과 32자 단위로 구간을 나눴습니다.")
    limit_reached = False
    for page in pages:
        for paragraph in page.get("paragraphs", []):
            text = paragraph["text"]
            offsets = _token_offsets(text, tokenizer)
            cursor = 0
            while cursor < len(offsets):
                if len(segments) >= max_segments:
                    limit_reached = True
                    break
                stop = min(cursor + budget, len(offsets))
                start_char = 0 if cursor == 0 else offsets[cursor][0]
                end_char = len(text) if stop == len(offsets) else offsets[stop][0]
                # Subword tokens can change at a new left boundary. Recheck the
                # final substring rather than asking inference to truncate it.
                if tokenizer is not None:
                    while True:
                        token_ids = tokenizer(text[start_char:end_char], add_special_tokens=True, truncation=False)["input_ids"]
                        if len(token_ids) <= max_tokens:
                            break
                        stop -= 1
                        if stop <= cursor:
                            raise DocumentError("하나의 토큰 구간이 모델의 입력 한도를 초과했습니다.")
                        end_char = offsets[stop][0]
                segment_text = text[start_char:end_char]
                start, end = paragraph["start"] + start_char, paragraph["start"] + end_char
                segments.append(dict(id=f"segment-{len(segments) + 1}", page=page["number"],
                                     paragraph_id=paragraph["id"], start=start, end=end, text=segment_text,
                                     token_count=stop - cursor))
                coverage[page["number"]].append((start, end))
                if stop == len(offsets):
                    break
                cursor = max(cursor + 1, stop - overlap)
            if limit_reached:
                break
        if limit_reached:
            break
    total_chars = sum(len(page["text"]) for page in pages)
    analyzed_chars = sum(_covered(intervals) for intervals in coverage.values())
    unprocessed = [page["number"] for page in pages
                   if not page["text"] or _covered(coverage[page["number"]]) < len(page["text"])]
    if limit_reached:
        warnings.append(f"최대 {max_segments}개 구간까지만 분석했습니다. 미처리 내용이 있습니다.")
    status = "ocr_required" if document.get("status") == "ocr_required" else "partial" if unprocessed else "ready"
    return dict(segments=segments, total_chars=total_chars, analyzed_chars=analyzed_chars,
                unprocessed_pages=unprocessed, warnings=warnings, status=status)


def chunk_text(text, max_tokens=256, overlap=32, tokenizer=None, max_segments=500):
    return chunk_document(document_from_text(text), tokenizer=tokenizer, max_tokens=max_tokens,
                          overlap=overlap, max_segments=max_segments)
