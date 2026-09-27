from __future__ import annotations

import math
from typing import Mapping

import numpy as np
import pandas as pd


def _finite(value: float | int | None, digits: int = 4) -> float | None:
    if value is None or not math.isfinite(float(value)):
        return None
    return round(float(value), digits)


def _return_over(series: pd.Series, sessions: int) -> float | None:
    clean = series.dropna()
    if len(clean) <= sessions:
        return None
    return float(clean.iloc[-1] / clean.iloc[-sessions - 1] - 1.0)


def score_watchlist(
    histories: Mapping[str, pd.DataFrame],
    benchmark_history: pd.DataFrame,
    market_risk: float,
) -> list[dict[str, object]]:
    """Build explainable end-of-day states for a technology watchlist.

    Market risk acts as a gate.  A stock-specific state still requires trend and
    relative-strength confirmation; this function never emits an order.
    """

    benchmark = benchmark_history["adj_close"].dropna().sort_index()
    benchmark_r60 = _return_over(benchmark, 60)
    results: list[dict[str, object]] = []

    for symbol, history in histories.items():
        close = history["adj_close"].dropna().sort_index()
        if len(close) < 200:
            results.append(
                {
                    "symbol": symbol,
                    "status": "insufficient_data",
                    "reasons": ["less_than_200_sessions"],
                    "as_of": close.index[-1].date().isoformat() if len(close) else None,
                }
            )
            continue

        value = float(close.iloc[-1])
        sma20 = float(close.rolling(20).mean().iloc[-1])
        sma50 = float(close.rolling(50).mean().iloc[-1])
        sma200 = float(close.rolling(200).mean().iloc[-1])
        r20 = _return_over(close, 20)
        r60 = _return_over(close, 60)
        rs60 = None if r60 is None or benchmark_r60 is None else r60 - benchmark_r60
        hv20 = float(np.log(close / close.shift(1)).rolling(20).std().iloc[-1] * np.sqrt(252))
        high252 = float(close.tail(252).max())
        drawdown = value / high252 - 1.0

        above_50 = value > sma50
        above_200 = value > sma200
        trend_ordered = value > sma20 > sma50 > sma200
        relative_leader = rs60 is not None and rs60 > 0
        reasons: list[str] = []

        if market_risk >= 65:
            reasons.append("market_risk_off")
            if not above_200 or not relative_leader:
                status = "reduce_watch"
                reasons.append("weak_stock_confirmation")
            else:
                status = "hold_no_new_position"
                reasons.append("stock_trend_still_intact")
        elif trend_ordered and relative_leader:
            status = "buy_eligible" if market_risk < 40 else "small_position_only"
            reasons.extend(["trend_aligned", "relative_strength_positive"])
        elif above_50 and above_200:
            status = "hold"
            reasons.append("long_term_trend_intact")
            if not relative_leader:
                reasons.append("relative_strength_not_confirmed")
        elif not above_200 and not relative_leader:
            status = "exit_condition_triggered"
            reasons.extend(["below_200dma", "relative_strength_negative"])
        else:
            status = "observe_or_reduce"
            reasons.append("mixed_stock_signals")

        results.append(
            {
                "symbol": symbol,
                "as_of": close.index[-1].date().isoformat(),
                "status": status,
                "reasons": reasons,
                "close": _finite(value, 2),
                "sma20": _finite(sma20, 2),
                "sma50": _finite(sma50, 2),
                "sma200": _finite(sma200, 2),
                "return_20d": _finite(r20),
                "return_60d": _finite(r60),
                "relative_strength_60d": _finite(rs60),
                "realized_vol_20d": _finite(hv20),
                "drawdown_from_52w_high": _finite(drawdown),
            }
        )

    return sorted(results, key=lambda item: str(item["symbol"]))
