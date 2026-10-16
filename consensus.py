"""Ticker-level target-price consensus built from collected reports.

Ticker, name, broker, date, target price and opinion come from the Naver
listing recorded in collection_manifest.json, which is more reliable than
header parsing. The PDF supplies the source page, the report's stated price,
previous target and revision wording, and is the fallback for older entries
without listing values. Extraction results are cached by file hash.
"""

from collections import Counter, defaultdict
from datetime import date, timedelta
import hashlib
import json
import os
from pathlib import Path
import re
from uuid import uuid4

from audit_extraction import load_entries
from data_processor import EXTRACTION_VERSION, METADATA_VERSION, opinion_class

CACHE_NAME = "consensus_cache.json"
CACHE_VERSION = f"{EXTRACTION_VERSION}/{METADATA_VERSION}"
UP, DOWN, SAME, NEW = "상향", "하향", "유지", "신규"
# Korean market convention: red for up, blue for down. Symbols carry the same
# meaning so direction never depends on color alone.
CHANGE_STYLE = {UP: ("#e66767", "triangle-up"), DOWN: ("#3987e5", "triangle-down"),
                SAME: ("#9aa4b2", "circle"), NEW: ("#cbd5e1", "circle-open")}
# Dark terminal surface: light ink, faint grid, cyan for the price line.
INK, INK_SECONDARY, INK_MUTED, GRID = "#e6edf7", "#a9b6c9", "#6b7a90", "#1c2636"
RULE, RING, PRICE, ACCENT = "#334155", "#0b1220", "#7dd3fc", "#22d3ee"
UNITS = {"원": 1, "KRW": 1, "천원": 1000, "만원": 10000}


def _won(value):
    """Parse an extracted '41000 원' value; other currencies or unknown units stay None."""
    match = re.fullmatch(r"([0-9]+(?:\.[0-9]+)?)\s*(원|KRW|천원|만원)", str(value or "").strip())
    return round(float(match[1]) * UNITS[match[2]]) if match else None


def _iso(value):
    digits = re.sub(r"[^0-9]", "", str(value or ""))
    try:
        return date(int(digits[:4]), int(digits[4:6]), int(digits[6:8])).isoformat() if len(digits) == 8 else None
    except ValueError:
        return None


def _extracted(document):
    metadata, evidence = document.get("metadata", {}), document.get("evidence", {})
    return dict(company=metadata.get("company"), target_price=metadata.get("target_price"),
                opinion=metadata.get("opinion"), current_price=metadata.get("current_price"),
                target_change=metadata.get("target_change"), previous_target=metadata.get("previous_target_price"),
                target_page=evidence.get("target_price", {}).get("page"),
                target_source=evidence.get("target_price", {}).get("text"))


def load_reports(pdf_dir, extract=None, progress=None):
    """Return (reports with a target price, reports skipped with a reason)."""
    if extract is None:
        from data_processor import extract_document as extract
    directory = Path(pdf_dir).expanduser()
    cache_path = directory / CACHE_NAME
    cache = {}
    try:
        stored = json.loads(cache_path.read_text(encoding="utf-8"))
        if stored.get("version") == CACHE_VERSION and isinstance(stored.get("items"), dict):
            cache = stored["items"]
    except (OSError, ValueError, AttributeError):
        pass
    entries = load_entries(directory)
    reports, skipped, changed = [], [], False
    for index, entry in enumerate(entries, 1):
        key = entry.get("sha256") or hashlib.sha256(entry["path"].read_bytes()).hexdigest()
        if key not in cache:
            if progress:
                progress(index / len(entries), entry.get("title") or entry["filename"])
            try:
                cache[key] = _extracted(extract(entry["path"]))
            except ValueError as exc:
                cache[key] = dict(error=str(exc))
            changed = True
        item = cache[key]
        base = dict(ticker=entry.get("ticker"), broker=entry.get("broker") or "미상",
                    date=_iso(entry.get("published_at")), title=entry.get("title") or "",
                    filename=entry["filename"], collected_at=entry.get("collected_at") or "")
        listed = entry.get("listing_target_price")
        listed = listed if isinstance(listed, int) and not isinstance(listed, bool) and listed > 0 else None
        extracted = _won(item.get("target_price"))
        target = listed or extracted
        if not target:
            reason = item.get("error") or ("목표주가 단위 미확인" if item.get("target_price") else "목표주가 없음")
        elif not base["ticker"] or not base["date"]:
            reason = "종목코드 또는 발간일 없음"
        else:
            reason = None
        if reason:
            skipped.append(dict(base, reason=reason))
            continue
        reports.append(dict(base, company=entry.get("company") or item.get("company"), target_price=target,
                            target_origin="네이버 목록" if listed else "PDF 원문", pdf_target=extracted,
                            opinion=opinion_class(entry.get("listing_opinion")) or opinion_class(item.get("opinion")),
                            report_price=_won(item.get("current_price")),
                            stated_change=item.get("target_change"), stated_previous=_won(item.get("previous_target")),
                            page=item.get("target_page"), source=item.get("target_source") or ""))
    if changed:
        staged = cache_path.with_name(f".{CACHE_NAME}.{uuid4().hex}.tmp")
        try:
            staged.write_text(json.dumps(dict(version=CACHE_VERSION, items=cache), ensure_ascii=False), encoding="utf-8")
            os.replace(staged, cache_path)
        finally:
            staged.unlink(missing_ok=True)
    return annotate_changes(reports), skipped


def annotate_changes(reports):
    """Mark each report's revision against the broker's previous target.

    The previous target printed in the report is the broker's own figure and
    wins over the collection history, which may have gaps. Without either, a
    stated "(하향)"/"유지" next to the target is used; otherwise it is new.
    A stated "신규" (coverage initiation) keeps the 원문 표기 basis. Opinion
    changes are only known from the collection history.
    """
    previous, opinions = {}, {}
    annotated = []
    for report in sorted(reports, key=lambda row: (row["ticker"], row["broker"], row["date"], row["filename"])):
        key = (report["ticker"], report["broker"])
        before, basis = report.get("stated_previous"), "원문 직전 목표주가"
        if before is None:
            before, basis = previous.get(key), "수집 이력"
        if before is not None:
            target = report["target_price"]
            change = UP if target > before else DOWN if target < before else SAME
        elif report.get("stated_change") in (UP, DOWN, SAME, NEW):
            change, basis = report["stated_change"], "원문 표기"
        else:
            change, basis = NEW, None
        annotated.append(dict(report, previous_target=before, change=change, change_basis=basis,
                              previous_opinion=opinions.get(key)))
        previous[key] = report["target_price"]
        if report.get("opinion"):
            opinions[key] = report["opinion"]
    return sorted(annotated, key=lambda row: (row["date"], row["filename"]), reverse=True)


def change_events(reports, tickers=None):
    """Feed rows for target revisions, opinion changes and coverage initiations.

    Unchanged targets are not events. ``tickers`` limits the feed. Rows are
    newest first by publication date, then by collection time.
    """
    names = company_names(reports)
    events = []
    for row in reports:
        if tickers is not None and row["ticker"] not in tickers:
            continue
        common = dict(ticker=row["ticker"], company=names.get(row["ticker"]) or row["ticker"], broker=row["broker"],
                      date=row["date"], collected_at=row.get("collected_at") or "", page=row.get("page"),
                      title=row["title"], basis=row.get("change_basis"))
        before, target = row.get("previous_target"), row["target_price"]
        if row["change"] in (UP, DOWN):
            detail = (f"목표주가 {before:,} → {target:,}원" if before else f"목표주가 {target:,}원 ({row['change']})")
            events.append(dict(common, kind=f"목표가 {row['change']}", change=row["change"], detail=detail,
                               pct=(target / before - 1) * 100 if before else None))
        elif row["change"] == NEW and row.get("change_basis") == "원문 표기":
            events.append(dict(common, kind="커버리지 개시", change=NEW, detail=f"목표주가 {target:,}원", pct=None))
        if row.get("previous_opinion") and row.get("opinion") and row["previous_opinion"] != row["opinion"]:
            events.append(dict(common, kind="투자의견 변경", change=None,
                               detail=f"투자의견 {row['previous_opinion']} → {row['opinion']}", pct=None))
    return sorted(events, key=lambda event: (event["date"], event["collected_at"]), reverse=True)


def company_names(reports):
    names = defaultdict(Counter)
    for report in reports:
        if report.get("company"):
            names[report["ticker"]][report["company"]] += 1
    return {ticker: counter.most_common(1)[0][0] for ticker, counter in names.items()}


def latest_by_broker(reports, ticker, as_of, window_days=90):
    """Each broker's most recent report for the ticker within the window."""
    start = (date.fromisoformat(as_of) - timedelta(days=window_days)).isoformat()
    latest = {}
    for report in reports:
        if report["ticker"] == ticker and start <= report["date"] <= as_of:
            if report["broker"] not in latest or report["date"] > latest[report["broker"]]["date"]:
                latest[report["broker"]] = report
    return sorted(latest.values(), key=lambda row: row["target_price"])


def current_price(reports, ticker, prices=None):
    """Latest close when available, else the price stated in the newest report."""
    history = (prices or {}).get(ticker)
    if history:
        day, close = history[-1]
        return dict(price=close, date=day, source="종가")
    for report in sorted((row for row in reports if row["ticker"] == ticker and row.get("report_price")),
                         key=lambda row: row["date"], reverse=True):
        return dict(price=report["report_price"], date=report["date"], source="리포트 기재가")
    return None


def summarize(reports, prices=None, as_of=None, window_days=90, recent_days=30):
    """One row per ticker: mean target, upside and recent revisions."""
    if not reports:
        return []
    as_of = as_of or max(row["date"] for row in reports)
    recent = (date.fromisoformat(as_of) - timedelta(days=recent_days)).isoformat()
    names = company_names(reports)
    rows = []
    for ticker in sorted({row["ticker"] for row in reports}):
        latest = latest_by_broker(reports, ticker, as_of, window_days)
        if not latest:
            continue
        mean = sum(row["target_price"] for row in latest) / len(latest)
        price = current_price(reports, ticker, prices)
        revisions = Counter(row["change"] for row in reports if row["ticker"] == ticker and row["date"] >= recent)
        rows.append(dict(ticker=ticker, company=names.get(ticker) or ticker, brokers=len(latest),
                         mean_target=round(mean), low=latest[0]["target_price"], high=latest[-1]["target_price"],
                         price=price["price"] if price else None, price_date=price["date"] if price else None,
                         price_source=price["source"] if price else None,
                         upside=(mean / price["price"] - 1) * 100 if price else None,
                         ups=revisions[UP], downs=revisions[DOWN],
                         latest_date=max(row["date"] for row in latest)))
    return sorted(rows, key=lambda row: (row["brokers"], row["latest_date"]), reverse=True)


def load_prices(tickers, start, end=None):
    """Daily closes per ticker via FinanceDataReader; None when it is not installed."""
    try:
        import FinanceDataReader as fdr
    except ImportError:
        return None
    prices = {}
    for ticker in tickers:
        try:
            frame = fdr.DataReader(ticker, start, end)
        except Exception:
            continue
        if frame is not None and "Close" in frame:
            prices[ticker] = [(index.date().isoformat(), float(close)) for index, close in frame["Close"].dropna().items()]
    return prices


def _hover(row):
    previous = f"{row['previous_target']:,}원" if row.get("previous_target") else "확인 불가"
    basis = f" · 근거: {row['change_basis']}" if row.get("change_basis") else ""
    return (f"<b>{row['broker']}</b> · {row['date']}<br>목표주가 {row['target_price']:,}원 ({row['change']}{basis})"
            f"<br>직전 {previous}<br>투자의견 {row.get('opinion') or '미확인'}<br>{row['title']}<br>출처: {source_note(row)}")


def source_note(row):
    """Where the target came from and whether the PDF cover agrees."""
    origin = row.get("target_origin") or "PDF 원문"
    page = f"PDF {row['page']}페이지" if row.get("page") else "PDF 위치 미확인"
    pdf = row.get("pdf_target")
    if origin == "PDF 원문":
        return page
    if pdf is None:
        return f"{origin} (PDF에서 확인 못 함)"
    if pdf != row["target_price"]:
        return f"{origin} (PDF {pdf:,}원과 다름, {page})"
    return f"{origin} · {page} 일치"


def _layout(figure, height):
    figure.update_layout(height=height, margin=dict(l=8, r=24, t=56, b=8), paper_bgcolor="rgba(0,0,0,0)",
                         plot_bgcolor="rgba(0,0,0,0)", font=dict(color=INK_SECONDARY, size=13),
                         hoverlabel=dict(bgcolor="#0e1626", bordercolor=ACCENT, font_color=INK),
                         legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0, title=None))
    figure.update_xaxes(gridcolor=GRID, zeroline=False, linecolor=RULE)
    figure.update_yaxes(gridcolor=GRID, zeroline=False, linecolor=RULE)
    return figure


def _label_positions(items, span, value_range):
    """Anchor side and y for gap labels so labels of nearby reports do not overlap.

    Reports within ~8% of the date span form a cluster. The earliest label in
    a cluster goes left of its line and the rest go right; labels on the same
    side are pushed apart vertically by ~12% of the value range.
    """
    near = max(timedelta(days=2), span * 0.08)
    gap = max(value_range * 0.12, 1)
    order = sorted(range(len(items)), key=lambda index: items[index]["row"]["date"])
    placed = [None] * len(items)
    cluster = []
    for index in order + [None]:
        day = date.fromisoformat(items[index]["row"]["date"]) if index is not None else None
        if cluster and (index is None or day - date.fromisoformat(items[cluster[-1]]["row"]["date"]) > near):
            sides = {"right": cluster[:1], "left": []} if len(cluster) == 1 else {"right": cluster[1:], "left": cluster[:1]}
            for anchor, members in sides.items():
                taken = []
                for member in sorted(members, key=lambda m: -(items[m]["base"] + items[m]["row"]["target_price"]) / 2):
                    y = (items[member]["base"] + items[member]["row"]["target_price"]) / 2
                    while any(abs(y - other) < gap for other in taken):
                        y -= gap
                    taken.append(y)
                    placed[member] = (anchor, y)
            cluster = []
        if index is not None:
            cluster.append(index)
    return placed


def _close_on(history, day):
    """Last close on or before the report date; reports cite the prior close."""
    before = [close for when, close in history if when <= day]
    return before[-1] if before else None


def target_figure(latest, price=None):
    """① How far each broker's latest target sits from the current price.

    Each row is a line from the current price to the target, so its length is
    the upside; the mean target gets its own row on top. Marker symbols show
    the revision against the broker's previous target, labelled as such.
    """
    import plotly.graph_objects as go
    figure = go.Figure()
    mean = sum(row["target_price"] for row in latest) / len(latest)
    reference = price["price"] if price else None
    mean_label = f"<b>평균 목표 ({len(latest)}곳)</b>"

    def label(value, bold=False):
        text = f"{value:,.0f}원" + (f" · {(value / reference - 1) * 100:+.1f}%" if reference else "")
        return f"<b>{text}</b>" if bold else text

    def side(value):
        return "middle left" if reference and value < reference else "middle right"

    if reference:
        for row in latest:
            figure.add_trace(go.Scatter(x=[reference, row["target_price"]], y=[row["broker"]] * 2, mode="lines",
                                        line=dict(color=RULE, width=3), hoverinfo="skip", showlegend=False))
        figure.add_trace(go.Scatter(x=[reference, mean], y=[mean_label] * 2, mode="lines",
                                    line=dict(color=INK_SECONDARY, width=3, dash="dot"), hoverinfo="skip", showlegend=False))
    for change, (color, symbol) in CHANGE_STYLE.items():
        rows = [row for row in latest if row["change"] == change]
        if not rows:
            continue
        figure.add_trace(go.Scatter(
            x=[row["target_price"] for row in rows], y=[row["broker"] for row in rows], mode="markers+text",
            name={UP: "상향", DOWN: "하향", SAME: "유지", NEW: "비교 없음"}[change],
            text=[label(row["target_price"]) for row in rows], textposition=[side(row["target_price"]) for row in rows],
            textfont=dict(color=INK_SECONDARY),
            marker=dict(size=14, color=color, symbol=symbol, line=dict(width=2, color=RING if symbol != "circle-open" else color)),
            hovertext=[_hover(row) for row in rows], hoverinfo="text"))
    figure.add_trace(go.Scatter(x=[mean], y=[mean_label], mode="markers+text", showlegend=False,
                                text=[label(mean, bold=True)], textposition=side(mean), textfont=dict(color=INK),
                                marker=dict(size=13, color=ACCENT, symbol="diamond", line=dict(width=2, color=RING)),
                                hovertemplate=f"평균 목표주가 {mean:,.0f}원<br>증권사 {len(latest)}곳의 최신 리포트 평균<extra></extra>"))
    values = [row["target_price"] for row in latest] + ([reference] if reference else [])
    span = max(values) - min(values) or max(values) * 0.1
    # Room on both sides for labels that sit outside the lines.
    figure.update_xaxes(range=[min(values) - span * 0.3, max(values) + span * 0.45], title=None, tickformat=",")
    figure.update_yaxes(categoryorder="array", categoryarray=[row["broker"] for row in latest] + [mean_label], showgrid=False)
    if reference:
        figure.add_vline(x=reference, line=dict(color=ACCENT, width=2))
        figure.add_annotation(x=reference, y=1, yref="paper", yanchor="bottom", xanchor="center", showarrow=False,
                              text=f"<b>현재가 {reference:,.0f}원</b> ({price['source']} {price['date'][5:]})",
                              font=dict(color=ACCENT, size=12))
    _layout(figure, 170 + 46 * (len(latest) + 1))
    figure.update_layout(showlegend=True, margin=dict(t=88),
                         legend=dict(y=0.99, yanchor="top", yref="container", orientation="h",
                                     title=dict(text="직전 대비 ", font=dict(color=INK_MUTED))))
    return figure


def history_figure(reports, ticker, history=None):
    """② Close price with every broker's target-price path and revision markers.

    With a price history the figure has two panels on a shared date axis: the
    top panel scales to the close alone so the trend stays readable even when
    targets sit far above it; the bottom panel compares targets with the close.
    """
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    rows = sorted((row for row in reports if row["ticker"] == ticker), key=lambda row: row["date"])
    if history:
        figure = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.09, row_heights=[0.42, 0.58],
                               subplot_titles=("주가 추이", "목표주가와 발간일 종가"))
        for annotation in figure.layout.annotations:
            annotation.update(x=0, xanchor="left", font=dict(size=13, color=INK))
        main = dict(row=2, col=1)
    else:
        figure = go.Figure()
        main = {}
    end = max([row["date"] for row in rows] + ([history[-1][0]] if history else []))
    by_broker = defaultdict(list)
    for row in rows:
        by_broker[row["broker"]].append(row)
    for broker, items in by_broker.items():
        figure.add_trace(go.Scatter(x=[row["date"] for row in items] + [end], y=[row["target_price"] for row in items] + [items[-1]["target_price"]],
                                    mode="lines", line=dict(color=RULE, width=1.5, shape="hv"),
                                    hoverinfo="skip", showlegend=False), **main)
    if history:
        days, prices = [day for day, _ in history], [close for _, close in history]
        closes = dict(x=days, y=prices, mode="lines",
                      line=dict(color=PRICE, width=2), hovertemplate="%{x}<br>종가 %{y:,.0f}원<extra></extra>")
        floor = min(prices) - (max(prices) - min(prices)) * 0.08
        figure.add_trace(go.Scatter(x=days, y=[floor] * len(days), mode="lines", line=dict(width=0),
                                    hoverinfo="skip", showlegend=False), row=1, col=1)
        figure.add_trace(go.Scatter(x=days, y=prices, mode="lines", line=dict(width=0), fill="tonexty",
                                    fillgradient=dict(type="vertical", colorscale=[[0, "rgba(125,211,252,0)"], [1, "rgba(125,211,252,0.28)"]]),
                                    hoverinfo="skip", showlegend=False), row=1, col=1)
        figure.add_trace(go.Scatter(x=days, y=prices, mode="lines", line=dict(color="rgba(125,211,252,0.18)", width=9),
                                    hoverinfo="skip", showlegend=False), row=1, col=1)
        figure.add_trace(go.Scatter(name="종가", legendgroup="close", **closes), row=1, col=1)
        figure.add_trace(go.Scatter(name="종가", legendgroup="close", showlegend=False, **closes), **main)
        # Mark publication days on the trend so report timing is visible there too.
        published = [(row["date"], _close_on(history, row["date"]), row["broker"]) for row in rows]
        published = [item for item in published if item[1]]
        if published:
            figure.add_trace(go.Scatter(x=[day for day, _, _ in published], y=[close for _, close, _ in published], mode="markers",
                                        name="리포트 발간", marker=dict(size=9, color=RING, line=dict(width=2, color=ACCENT)),
                                        customdata=[broker for _, _, broker in published],
                                        hovertemplate="%{x} · %{customdata} 리포트<br>종가 %{y:,.0f}원<extra></extra>"), row=1, col=1)
    else:
        stated = [row for row in rows if row.get("report_price")]
        if stated:
            figure.add_trace(go.Scatter(x=[row["date"] for row in stated], y=[row["report_price"] for row in stated],
                                        mode="markers", name="리포트 기재 주가",
                                        marker=dict(size=9, color=INK_MUTED, symbol="x-thin", line=dict(width=2, color=INK_MUTED)),
                                        hovertemplate="%{x}<br>리포트 기재 주가 %{y:,.0f}원<extra></extra>"))
    # Connect each target to the close at publication; label the gap for each
    # broker's latest report so older revisions do not crowd the chart.
    latest = {row["broker"]: row for row in rows}
    connectors = []
    for row in rows:
        base = _close_on(history, row["date"]) if history else row.get("report_price")
        if not base:
            continue
        gap = row["target_price"] - base
        color = CHANGE_STYLE[UP][0] if gap >= 0 else CHANGE_STYLE[DOWN][0]
        figure.add_trace(go.Scatter(x=[row["date"], row["date"]], y=[base, row["target_price"]], mode="lines",
                                    line=dict(color=color, width=1.5, dash="dot"), hoverinfo="skip",
                                    name="목표가 − 발간일 종가", legendgroup="gap", showlegend=not connectors), **main)
        connectors.append(dict(row=row, base=base, gap=gap))
    if connectors:
        figure.add_trace(go.Scatter(x=[item["row"]["date"] for item in connectors], y=[item["base"] for item in connectors],
                                    mode="markers", name="발간일 종가", marker=dict(size=8, color=INK_SECONDARY, line=dict(width=2, color=RING)),
                                    customdata=[[item["gap"], item["gap"] / item["base"] * 100, item["row"]["broker"]] for item in connectors],
                                    hovertemplate="%{x|%Y-%m-%d} · %{customdata[2]}<br>발간일 종가 %{y:,.0f}원"
                                                  "<br>목표가까지 %{customdata[0]:+,.0f}원 (%{customdata[1]:+.1f}%)<extra></extra>"), **main)
    labelled = [item for item in connectors if latest[item["row"]["broker"]] is item["row"]]
    first = min([row["date"] for row in rows] + ([history[0][0]] if history else []))
    span = date.fromisoformat(end) - date.fromisoformat(first)
    values = [row["target_price"] for row in rows] + [item["base"] for item in connectors]
    for item, (anchor, y) in zip(labelled, _label_positions(labelled, span, max(values) - min(values))):
        figure.add_annotation(x=item["row"]["date"], y=y, xanchor=anchor, xshift=8 if anchor == "left" else -8,
                              text=f"<b>{item['gap']:+,.0f}원</b><br>{item['gap'] / item['base'] * 100:+.1f}%",
                              showarrow=False, align=anchor, font=dict(color=INK, size=12),
                              bgcolor="rgba(6,10,18,0.85)", **main)
    for change, (color, symbol) in CHANGE_STYLE.items():
        items = [row for row in rows if row["change"] == change]
        if items:
            figure.add_trace(go.Scatter(x=[row["date"] for row in items], y=[row["target_price"] for row in items],
                                        mode="markers", name=f"목표주가 {change}", hovertext=[_hover(row) for row in items], hoverinfo="text",
                                        marker=dict(size=13, color=color, symbol=symbol,
                                                    line=dict(width=2, color=RING if symbol != "circle-open" else color))), **main)
    # Leave room on the right so labels on the newest report are not clipped.
    pad = max(timedelta(days=3), span * 0.11)
    figure.update_yaxes(tickformat=",", title=None)
    figure.update_xaxes(type="date", tickformat="%y.%m.%d", hoverformat="%Y-%m-%d", title=None, showgrid=False,
                        range=[(date.fromisoformat(first) - span * 0.01).isoformat(), (date.fromisoformat(end) + pad).isoformat()])
    if history:
        # Shared axes hide the top panel's dates; show them on both panels.
        figure.update_xaxes(showticklabels=True, row=1, col=1)
    figure.update_layout(hovermode="closest")
    _layout(figure, 720 if history else 480)
    figure.update_layout(margin=dict(l=8, r=24, t=64, b=8), legend=dict(y=0.995, yanchor="top", yref="container"))
    return figure


def revision_breadth(reports, as_of=None, days=30):
    """Upgrades and downgrades of target prices within ``days`` before ``as_of``."""
    if not reports:
        return dict(ups=0, downs=0, share=None, start=None, end=None)
    as_of = as_of or max(row["date"] for row in reports)
    start = (date.fromisoformat(as_of) - timedelta(days=days)).isoformat()
    counts = Counter(row["change"] for row in reports if start < row["date"] <= as_of)
    total = counts[UP] + counts[DOWN]
    return dict(ups=counts[UP], downs=counts[DOWN], share=counts[UP] / total * 100 if total else None, start=start, end=as_of)


def breadth_figure(breadth):
    """Gauge of the share of target revisions that were upgrades (50% = balanced)."""
    import plotly.graph_objects as go
    share = breadth["share"]
    figure = go.Figure(go.Indicator(
        mode="gauge+number", value=share if share is not None else 0,
        number=dict(suffix="%", valueformat=".0f", font=dict(color=INK if share is not None else INK_MUTED, size=40)),
        gauge=dict(axis=dict(range=[0, 100], tickvals=[0, 50, 100], ticktext=["하향", "균형", "상향"],
                             tickcolor=RULE, tickfont=dict(color=INK_SECONDARY, size=11)),
                   bar=dict(color=ACCENT, thickness=0.28), bgcolor="rgba(0,0,0,0)", borderwidth=0,
                   steps=[dict(range=[0, 40], color="rgba(57,135,229,0.22)"), dict(range=[40, 60], color="rgba(154,164,178,0.12)"),
                          dict(range=[60, 100], color="rgba(230,103,103,0.22)")],
                   threshold=dict(line=dict(color=INK_SECONDARY, width=2), thickness=0.8, value=50))))
    _layout(figure, 260)
    figure.update_layout(margin=dict(l=64, r=64, t=28, b=8))
    return figure


def revision_flow_figure(reports, days=60):
    """Daily target upgrades (up, red) and downgrades (down, blue) on one zero line."""
    import plotly.graph_objects as go
    end = max(row["date"] for row in reports)
    start = (date.fromisoformat(end) - timedelta(days=days)).isoformat()
    ups, downs = Counter(), Counter()
    for row in reports:
        if start < row["date"] <= end and row["change"] == UP:
            ups[row["date"]] += 1
        elif start < row["date"] <= end and row["change"] == DOWN:
            downs[row["date"]] += 1
    dates = sorted(set(ups) | set(downs))
    figure = go.Figure()
    figure.add_trace(go.Bar(x=dates, y=[ups[day] for day in dates], name="▲ 목표가 상향", marker=dict(color=CHANGE_STYLE[UP][0], cornerradius=4),
                            hovertemplate="%{x}<br>상향 %{y}건<extra></extra>"))
    figure.add_trace(go.Bar(x=dates, y=[-downs[day] for day in dates], name="▼ 목표가 하향", marker=dict(color=CHANGE_STYLE[DOWN][0], cornerradius=4),
                            customdata=[downs[day] for day in dates], hovertemplate="%{x}<br>하향 %{customdata}건<extra></extra>"))
    # Fixed bar width (0.6 day) so a short history does not render slab-wide bars.
    for trace in figure.data:
        trace.width = [86_400_000 * 0.6] * len(dates)
    figure.update_layout(barmode="relative")
    if len(dates) < 14:
        # Show at least two weeks so a few days of data read as days, not slabs.
        last = date.fromisoformat(dates[-1] if dates else end)
        figure.update_xaxes(range=[(last - timedelta(days=14)).isoformat(), (last + timedelta(days=1)).isoformat()])
    figure.update_xaxes(type="date", tickformat="%m.%d", showgrid=False)
    figure.update_yaxes(zeroline=True, zerolinecolor=INK_MUTED, zerolinewidth=1, tickformat="d")
    _layout(figure, 260)
    figure.update_layout(margin=dict(l=8, r=16, t=40, b=8), legend=dict(y=1.12))
    return figure
