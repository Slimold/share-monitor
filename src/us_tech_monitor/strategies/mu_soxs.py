"""MU long / SOXS defensive confidence model.

The model converts the original hard RSI(6) rules into two independent
0--100 evidence scores.  The scores are ordinal research signals, not
probabilities.  Every rolling statistic and percentile uses only information
available through the scored close.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from ..config import MuSoxsStrategyConfig
from ..indicators import historical_percentile


def wilder_rsi(price: pd.Series, periods: int = 6) -> pd.Series:
    """Return a Wilder-style RSI using only current and earlier closes."""

    if periods < 2:
        raise ValueError("periods must be at least 2")
    close = pd.to_numeric(price, errors="coerce")
    change = close.diff()
    gain = change.clip(lower=0.0)
    loss = -change.clip(upper=0.0)
    average_gain = gain.ewm(
        alpha=1.0 / periods, adjust=False, min_periods=periods
    ).mean()
    average_loss = loss.ewm(
        alpha=1.0 / periods, adjust=False, min_periods=periods
    ).mean()
    relative_strength = average_gain / average_loss.replace(0.0, np.nan)
    rsi = 100.0 - 100.0 / (1.0 + relative_strength)
    rsi = rsi.where(average_loss.ne(0.0), 100.0)
    rsi = rsi.where(average_gain.ne(0.0), 0.0)
    rsi.name = f"rsi_{periods}"
    return rsi


def _adjusted_close(history: pd.DataFrame, label: str) -> pd.Series:
    if "adj_close" not in history:
        raise ValueError(f"{label} history is missing adj_close")
    close = pd.to_numeric(history["adj_close"], errors="coerce").sort_index()
    close = close[~close.index.duplicated(keep="last")]
    return close.where(close > 0).rename(label)


def _clip01(value: pd.Series) -> pd.Series:
    return value.clip(lower=0.0, upper=1.0)


def build_confidence_history(
    mu_history: pd.DataFrame,
    smh_history: pd.DataFrame,
    soxs_history: pd.DataFrame,
    market_risk: pd.Series,
    config: MuSoxsStrategyConfig | None = None,
) -> pd.DataFrame:
    """Build daily MU-buy and MU-sell/SOXS confidence without look-ahead.

    Only exact common trading dates are scored.  Missing dates are not
    forward- or backward-filled, preventing a stale instrument from being
    combined with a newer market-risk observation.
    """

    policy = config or MuSoxsStrategyConfig()
    mu = _adjusted_close(mu_history, "mu_close")
    smh = _adjusted_close(smh_history, "smh_close")
    soxs = _adjusted_close(soxs_history, "soxs_close")
    risk = pd.to_numeric(market_risk, errors="coerce").sort_index().rename("market_risk")
    risk = risk[~risk.index.duplicated(keep="last")]

    common_index = mu.index.intersection(smh.index).intersection(soxs.index).intersection(risk.index)
    frame = pd.concat(
        [
            mu.reindex(common_index),
            smh.reindex(common_index),
            soxs.reindex(common_index),
            risk.reindex(common_index),
        ],
        axis=1,
    ).dropna()
    if frame.empty:
        return frame

    frame["mu_return_1d"] = frame["mu_close"].pct_change(fill_method=None)
    frame["smh_return_1d"] = frame["smh_close"].pct_change(fill_method=None)
    frame["soxs_return_1d"] = frame["soxs_close"].pct_change(fill_method=None)
    frame["smh_return_5d"] = frame["smh_close"].pct_change(5, fill_method=None)
    frame["mu_sma20"] = frame["mu_close"].rolling(20, min_periods=20).mean()
    frame["mu_gap_20"] = frame["mu_close"] / frame["mu_sma20"] - 1.0
    frame["mu_smh_relative_20d"] = (
        frame["mu_close"] / frame["smh_close"]
    ).pct_change(20, fill_method=None)
    frame["smh_ema10"] = frame["smh_close"].ewm(span=10, adjust=False).mean()
    frame["smh_gap_ema10"] = frame["smh_close"] / frame["smh_ema10"] - 1.0
    frame["soxs_ema5"] = frame["soxs_close"].ewm(span=5, adjust=False).mean()

    rsi_column = f"mu_rsi_{policy.rsi_period}"
    frame[rsi_column] = wilder_rsi(frame["mu_close"], policy.rsi_period)
    frame["recent_rsi_max_5d"] = frame[rsi_column].rolling(5, min_periods=1).max()
    frame["recent_rsi_min_5d"] = frame[rsi_column].rolling(5, min_periods=1).min()

    percentile_options = {
        "window": policy.percentile_window,
        "min_periods": policy.percentile_min_periods,
    }
    frame["mu_gap_20_percentile"] = historical_percentile(
        frame["mu_gap_20"], **percentile_options
    )
    frame["mu_smh_relative_20d_percentile"] = historical_percentile(
        frame["mu_smh_relative_20d"], **percentile_options
    )
    frame["smh_return_5d_percentile"] = historical_percentile(
        frame["smh_return_5d"], **percentile_options
    )
    frame["smh_gap_ema10_percentile"] = historical_percentile(
        frame["smh_gap_ema10"], **percentile_options
    )

    # Buy score: extreme oversold + historically stretched downside, with a
    # smaller reward for evidence that the sell-off is starting to reverse.
    frame["buy_rsi_oversold"] = 100.0 * _clip01(
        (40.0 - frame[rsi_column]) / 15.0
    )
    frame["buy_downside_stretch"] = (
        100.0
        - 0.5 * frame["mu_gap_20_percentile"]
        - 0.5 * frame["mu_smh_relative_20d_percentile"]
    ).clip(0.0, 100.0)
    frame["buy_reversal"] = (
        50.0 * _clip01((frame[rsi_column] - frame[rsi_column].shift(1)) / 12.0)
        + 30.0 * frame["mu_return_1d"].gt(0.0).astype(float)
        + 20.0 * frame["smh_return_1d"].gt(0.0).astype(float)
    ).clip(0.0, 100.0)
    frame["buy_sector_capitulation"] = (
        100.0 - frame["smh_return_5d_percentile"]
    ).clip(0.0, 100.0)
    buy_components = (
        "buy_rsi_oversold",
        "buy_downside_stretch",
        "buy_reversal",
        "buy_sector_capitulation",
    )
    buy_complete = frame[list(buy_components)].notna().all(axis=1)
    frame["buy_confidence"] = (
        frame["buy_rsi_oversold"] * policy.buy_oversold_weight
        + frame["buy_downside_stretch"] * policy.buy_downside_weight
        + frame["buy_reversal"] * policy.buy_reversal_weight
        + frame["buy_sector_capitulation"] * policy.buy_sector_weight
    ).where(buy_complete).clip(0.0, 100.0)

    # Sell/defensive score: recent overbought condition + MU rollover +
    # confirmed semiconductor weakness.  SOXS itself is a hard execution gate,
    # not a directional factor, avoiding daily-reset path noise in the score.
    frame["sell_recent_overbought"] = 100.0 * _clip01(
        (frame["recent_rsi_max_5d"] - 65.0) / 15.0
    )
    frame["sell_stock_extension"] = (
        0.5 * frame["mu_gap_20_percentile"]
        + 0.5 * frame["mu_smh_relative_20d_percentile"]
    ).clip(0.0, 100.0)
    frame["sell_sector_weakness"] = (
        100.0
        - 0.5 * frame["smh_gap_ema10_percentile"]
        - 0.5 * frame["smh_return_5d_percentile"]
    ).clip(0.0, 100.0)
    crossed_down_70 = frame[rsi_column].le(70.0) & frame[rsi_column].shift(1).gt(70.0)
    frame["sell_rollover"] = (
        60.0 * _clip01((frame[rsi_column].shift(1) - frame[rsi_column]) / 15.0)
        + 20.0 * crossed_down_70.astype(float)
        + 20.0 * frame["mu_return_1d"].lt(0.0).astype(float)
    ).clip(0.0, 100.0)
    frame["sell_market_risk"] = frame["market_risk"].clip(0.0, 100.0)
    sell_components = (
        "sell_recent_overbought",
        "sell_stock_extension",
        "sell_sector_weakness",
        "sell_rollover",
        "sell_market_risk",
    )
    sell_complete = frame[list(sell_components)].notna().all(axis=1)
    frame["sell_confidence"] = (
        frame["sell_recent_overbought"] * policy.sell_overbought_weight
        + frame["sell_stock_extension"] * policy.sell_extension_weight
        + frame["sell_sector_weakness"] * policy.sell_sector_weight
        + frame["sell_rollover"] * policy.sell_rollover_weight
        + frame["sell_market_risk"] * policy.sell_market_weight
    ).where(sell_complete).clip(0.0, 100.0)

    shock = frame["mu_return_1d"].le(policy.shock_mu_return) | frame[
        "smh_return_1d"
    ].le(policy.shock_sector_return)
    frame["shock_day"] = shock
    frame["shock_cooldown_active"] = (
        shock.rolling(policy.shock_cooldown_sessions, min_periods=1).max().astype(bool)
    )
    frame["smh_weakness_confirmed"] = frame["smh_close"].lt(
        frame["smh_ema10"]
    ) & frame["smh_return_5d"].lt(0.0)
    frame["soxs_trend_confirmed"] = frame["soxs_close"].gt(frame["soxs_ema5"])
    confidence_spread = frame["buy_confidence"] - frame["sell_confidence"]

    frame["mu_entry_allowed"] = (
        frame["buy_confidence"].ge(policy.buy_threshold)
        & confidence_spread.ge(policy.confidence_margin)
        & frame[rsi_column].lt(30.0)
    )
    frame["mu_add_allowed"] = (
        frame["buy_confidence"].ge(policy.buy_threshold)
        & confidence_spread.ge(policy.confidence_margin)
        & frame["recent_rsi_min_5d"].lt(30.0)
        & frame[rsi_column].between(30.0, 50.0, inclusive="left")
        & frame[rsi_column].gt(frame[rsi_column].shift(1))
        & frame["mu_return_1d"].gt(0.0)
        & frame["mu_return_1d"].lt(0.08)
        & frame["smh_return_1d"].gt(0.0)
    )
    frame["mu_exit_condition"] = frame[rsi_column].gt(50.0)
    frame["soxs_entry_allowed"] = (
        frame["sell_confidence"].ge(policy.sell_threshold)
        & (-confidence_spread).ge(policy.confidence_margin)
        & frame["recent_rsi_max_5d"].gt(70.0)
        & frame[rsi_column].between(50.0, 70.0, inclusive="both")
        & frame["smh_weakness_confirmed"]
        & frame["soxs_trend_confirmed"]
        & ~frame["shock_cooldown_active"]
    )
    smh_recovered = frame["smh_close"].gt(frame["smh_ema10"])
    frame["soxs_exit_condition"] = frame[rsi_column].lt(50.0) | (
        smh_recovered & smh_recovered.shift(1, fill_value=False)
    )

    decisions = np.select(
        [
            frame["soxs_entry_allowed"],
            frame["mu_entry_allowed"],
            frame["mu_add_allowed"],
            frame["mu_exit_condition"],
            frame["soxs_exit_condition"],
        ],
        [
            "reduce_mu_buy_soxs",
            "buy_mu_starter",
            "add_mu_after_reversal",
            "exit_mu_if_held",
            "exit_soxs_if_held",
        ],
        default="wait_for_confirmation",
    )
    frame["decision"] = decisions
    return frame


def _number(row: pd.Series, key: str, digits: int = 1) -> float | None:
    value = row.get(key)
    if value is None or pd.isna(value) or not np.isfinite(float(value)):
        return None
    return round(float(value), digits)


def _strategy_reasons(row: pd.Series, rsi_column: str) -> list[str]:
    reasons: list[str] = []
    if row.get("mu_entry_allowed"):
        reasons.extend(["mu_rsi_oversold", "downside_stretch_extreme"])
    elif row.get("mu_add_allowed"):
        reasons.extend(["mu_reversal_confirmed", "sector_rebound_confirmed"])
    elif row.get("soxs_entry_allowed"):
        reasons.extend(["mu_overbought_rolled_over", "smh_weakness_confirmed"])
    elif row.get("shock_cooldown_active") and _number(row, "sell_confidence", 1) is not None:
        reasons.append("shock_cooldown_active")
    elif row.get("mu_exit_condition"):
        reasons.append("mu_exit_rsi_above_50")
    elif row.get("soxs_exit_condition"):
        reasons.append("soxs_exit_condition_met")

    if row.get("smh_weakness_confirmed"):
        reasons.append("smh_below_ema10_and_5d_negative")
    if row.get("soxs_trend_confirmed"):
        reasons.append("soxs_above_ema5")
    risk = _number(row, "market_risk", 1)
    if risk is not None:
        reasons.append("market_risk_elevated" if risk >= 65.0 else "market_risk_not_extreme")
    if not reasons and _number(row, rsi_column, 1) is not None:
        reasons.append("waiting_for_extreme_and_confirmation")
    return list(dict.fromkeys(reasons))[:5]


def latest_confidence_payload(
    confidence: pd.DataFrame,
    *,
    as_of: pd.Timestamp,
    config: MuSoxsStrategyConfig | None = None,
) -> dict[str, Any]:
    """Return one JSON-friendly strategy record for an exact market date."""

    policy = config or MuSoxsStrategyConfig()
    as_of = pd.Timestamp(as_of)
    instruments = {
        "long": policy.signal_symbol,
        "inverse": policy.inverse_symbol,
        "sector": policy.sector_proxy,
    }
    base: dict[str, Any] = {
        "as_of": as_of.date().isoformat(),
        "version": policy.version,
        "status": "unavailable",
        "buy_confidence": None,
        "sell_confidence": None,
        "decision": "data_unavailable",
        "instruments": instruments,
        "development_window": {
            "start": policy.development_start,
            "end": policy.development_end,
            "role": "development_calibration_not_out_of_sample",
        },
        "execution_timing": "close_signal_next_session_open",
        "interpretation": "ordinal_evidence_score_not_probability",
        "reasons": ["strategy_inputs_not_aligned"],
    }
    if as_of not in confidence.index:
        return base
    row = confidence.loc[as_of]
    buy = _number(row, "buy_confidence")
    sell = _number(row, "sell_confidence")
    if buy is None or sell is None:
        return base

    rsi_column = f"mu_rsi_{policy.rsi_period}"
    base.update(
        {
            "status": "available",
            "buy_confidence": buy,
            "sell_confidence": sell,
            "decision": str(row["decision"]),
            "mu_close": _number(row, "mu_close", 2),
            "soxs_close": _number(row, "soxs_close", 2),
            "smh_close": _number(row, "smh_close", 2),
            "mu_rsi6": _number(row, rsi_column),
            "market_risk": _number(row, "market_risk"),
            "thresholds": {
                "buy": policy.buy_threshold,
                "sell": policy.sell_threshold,
                "margin": policy.confidence_margin,
            },
            "gates": {
                "mu_entry_allowed": bool(row["mu_entry_allowed"]),
                "mu_add_allowed": bool(row["mu_add_allowed"]),
                "mu_exit_condition": bool(row["mu_exit_condition"]),
                "soxs_entry_allowed": bool(row["soxs_entry_allowed"]),
                "soxs_exit_condition": bool(row["soxs_exit_condition"]),
                "shock_cooldown_active": bool(row["shock_cooldown_active"]),
                "smh_weakness_confirmed": bool(row["smh_weakness_confirmed"]),
                "soxs_trend_confirmed": bool(row["soxs_trend_confirmed"]),
            },
            "components": {
                "buy": {
                    "rsi_oversold": _number(row, "buy_rsi_oversold"),
                    "downside_stretch": _number(row, "buy_downside_stretch"),
                    "reversal": _number(row, "buy_reversal"),
                    "sector_capitulation": _number(row, "buy_sector_capitulation"),
                },
                "sell": {
                    "recent_overbought": _number(row, "sell_recent_overbought"),
                    "stock_extension": _number(row, "sell_stock_extension"),
                    "sector_weakness": _number(row, "sell_sector_weakness"),
                    "rollover": _number(row, "sell_rollover"),
                    "market_risk": _number(row, "sell_market_risk"),
                },
            },
            "risk_policy": {
                "account_risk_budget": policy.account_risk_budget,
                "mu_max_weight": policy.mu_max_weight,
                "mu_starter_fraction": policy.mu_starter_fraction,
                "soxs_max_weight": policy.soxs_max_weight,
                "soxs_max_holding_sessions": policy.soxs_max_holding_sessions,
            },
            "reasons": _strategy_reasons(row, rsi_column),
            "limitations": [
                "earnings_calendar_not_included",
                "score_does_not_include_personal_position_or_tax_context",
            ],
        }
    )
    return base
