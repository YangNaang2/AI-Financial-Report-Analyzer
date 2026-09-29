"""Observed forward-return labels; unobserved future outcomes stay pending.

Default entry is the first observed session strictly after publication. Exit is
the first later observed session at least ``window_days`` after publication.
Labels describe price outcomes, not analyst intent or recommendation honesty.
"""

from datetime import date, datetime, timedelta
from decimal import Decimal, localcontext
import math
import re


def _day(value, name):
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    if isinstance(value, str):
        try:
            return date.fromisoformat(value.strip().replace(".", "-").replace("/", "-"))
        except ValueError:
            pass
    raise ValueError(f"{name}은 YYYY-MM-DD 또는 YYYY.MM.DD 날짜여야 합니다.")


def calculate_post_report_return(ticker, report_date_str, window_days=30,
                                 threshold=-5, entry_policy="next_session",
                                 as_of=None, price_loader=None):
    """Return structured labeled/pending/unavailable results.

    Inject ``price_loader(ticker, start_iso, end_iso)`` for offline tests. Provider
    rows are sorted and clipped to ``as_of``, even if the provider returns extra
    data. FinanceDataReader Close corporate-action adjustment is unknown.
    """
    if not isinstance(ticker, str) or not re.fullmatch(r"[0-9]{6}", ticker):
        raise ValueError("종목코드는 숫자 6자리여야 합니다.")
    if isinstance(window_days, bool) or not isinstance(window_days, int) or not 1 <= window_days <= 3650:
        raise ValueError("예측 기간은 1~3650일의 정수여야 합니다.")
    try:
        if isinstance(threshold, bool):
            raise ValueError
        threshold = float(threshold)
    except (ValueError, TypeError):
        raise ValueError("하락 기준은 유한한 백분율이어야 합니다.") from None
    if not math.isfinite(threshold) or not -100 <= threshold <= 100:
        raise ValueError("하락 기준은 -100~100% 범위여야 합니다.")
    if entry_policy not in ("next_session", "report_day"):
        raise ValueError("진입 정책은 next_session 또는 report_day여야 합니다.")
    published = _day(report_date_str, "발행일")
    observed = date.today() if as_of is None else _day(as_of, "관측 기준일")
    target = published + timedelta(days=window_days)
    result = {
        "status": "pending", "ticker": ticker, "report_date": published.isoformat(),
        "as_of": observed.isoformat(), "target_date": target.isoformat(),
        "start_date": None, "end_date": None, "base_price": None, "future_price": None,
        "return_pct": None, "수익률": None, "label_id": None,
        "window_days": window_days, "threshold": threshold, "entry_policy": entry_policy,
        "window_kind": "calendar_days", "window_anchor": "report_date",
        "source": "injected_price_loader" if price_loader else "FinanceDataReader.Close",
        "adjustment_policy": "unknown",
        "adjustment_note": "종가의 분할·배당 조정 여부를 보장하지 않습니다. 기업행동 자료는 별도 확인이 필요합니다.",
        "policy_note": "발행 후 다음 관측 거래일 진입" if entry_policy == "next_session" else "발행일 또는 이후 첫 관측 거래일 진입(발행 시각 미반영)",
        "reason": "예측 기간이 아직 지나지 않았습니다.",
    }
    if observed < target:
        return result
    request_end = min(observed, target + timedelta(days=45))
    try:
        if price_loader is None:
            import FinanceDataReader as fdr
            price_loader = fdr.DataReader
        frame = price_loader(ticker, published.isoformat(), request_end.isoformat())
        if frame is None or getattr(frame, "empty", True) or "Close" not in frame:
            return dict(result, status="unavailable", reason="종가 자료가 없습니다.")
        prices = {}
        for index, value in frame["Close"].items():
            try:
                session, price = _day(index, "가격 날짜"), float(value)
            except (TypeError, ValueError, OverflowError):
                continue
            if published <= session <= observed and math.isfinite(price) and price > 0:
                prices[session] = price
        ordered = sorted(prices)
        entries = [session for session in ordered
                   if session > published or (entry_policy == "report_day" and session == published)]
        if not entries:
            return dict(result, status="unavailable", reason="정책에 맞는 유효한 진입 거래일이 없습니다.")
        start = entries[0]
        result.update(start_date=start.isoformat(), base_price=prices[start])
        exits = [session for session in ordered if session >= target and session > start]
        if not exits:
            return dict(result, reason="기간 종료 이후의 유효한 거래일 종가가 아직 관측되지 않았습니다.")
        end = exits[0]
        with localcontext() as context:
            context.prec = 40
            base, future = Decimal(str(prices[start])), Decimal(str(prices[end]))
            exact_return = (future - base) / base * 100
        returned = float(exact_return)
        result.update(status="labeled", end_date=end.isoformat(), future_price=prices[end],
                      return_pct=returned, 수익률=round(returned, 2),
                      label_id=int(exact_return <= Decimal(str(threshold))), reason="관측된 종가로 계산했습니다.")
        return result
    except Exception as exc:
        return dict(result, status="unavailable", reason=f"가격 자료 조회 실패: {type(exc).__name__}: {exc}")


if __name__ == "__main__":
    import argparse
    import json
    parser = argparse.ArgumentParser(description="리포트 이후 실제 관측 수익률 라벨 계산")
    parser.add_argument("ticker")
    parser.add_argument("report_date")
    parser.add_argument("--window-days", type=int, default=30)
    parser.add_argument("--as-of")
    args = parser.parse_args()
    print(json.dumps(calculate_post_report_return(args.ticker, args.report_date,
                                                 args.window_days, as_of=args.as_of),
                     ensure_ascii=False, indent=2))
