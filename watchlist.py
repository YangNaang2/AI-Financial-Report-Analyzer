"""Watchlist and feed read marker stored next to a collection folder.

The app issues a new workspace key per browser session, so a watchlist kept
in the workspace would vanish on reload. Collected reports are shared on this
machine, and the watchlist belongs to that collection: data/reports/watchlist.json.
"""

from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
from uuid import uuid4

FILENAME = "watchlist.json"
MAX_TICKERS = 200


class WatchlistError(ValueError):
    """The watchlist file or a requested change is invalid."""


def _ticker(value):
    ticker = str(value or "").strip()
    if not re.fullmatch(r"[0-9]{6}", ticker):
        raise WatchlistError("종목코드는 6자리 숫자여야 합니다.")
    return ticker


def load(folder):
    """Return {tickers: [...], seen_at: iso | None}; a missing file is an empty list."""
    path = Path(folder) / FILENAME
    try:
        stored = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return dict(tickers=[], seen_at=None)
    except (OSError, ValueError) as exc:
        raise WatchlistError(f"관심 종목 파일을 읽을 수 없습니다: {path}") from exc
    if not isinstance(stored, dict) or not isinstance(stored.get("tickers"), list):
        raise WatchlistError(f"관심 종목 파일 형식이 올바르지 않습니다: {path}")
    seen_at = stored.get("seen_at")
    return dict(tickers=[_ticker(ticker) for ticker in stored["tickers"]][:MAX_TICKERS],
                seen_at=seen_at if isinstance(seen_at, str) else None)


def _save(folder, data):
    path = Path(folder) / FILENAME
    staged = path.with_name(f".{FILENAME}.{uuid4().hex}.tmp")
    try:
        staged.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        os.replace(staged, path)
    finally:
        staged.unlink(missing_ok=True)


def set_tickers(folder, tickers):
    """Replace the watchlist, keeping order and dropping duplicates."""
    cleaned = list(dict.fromkeys(_ticker(ticker) for ticker in tickers))
    if len(cleaned) > MAX_TICKERS:
        raise WatchlistError(f"관심 종목은 최대 {MAX_TICKERS}개까지 저장할 수 있습니다.")
    data = load(folder)
    data["tickers"] = cleaned
    _save(folder, data)
    return data


def mark_seen(folder, when=None):
    """Record that every change collected up to ``when`` has been reviewed."""
    data = load(folder)
    data["seen_at"] = when or datetime.now(timezone.utc).isoformat()
    _save(folder, data)
    return data


def is_unseen(event, seen_at):
    """An event is new when it was collected after the last review."""
    return not seen_at or (event.get("collected_at") or "") > seen_at
