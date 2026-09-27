"""Explainable market regime model for the U.S. technology MVP."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from .indicators import build_indicators


@dataclass(frozen=True)
class ModelConfig:
    """Pre-declared model policy; no weight is fitted to the backtest sample."""

    macro_weight: float = 0.30
    volatility_weight: float = 0.30
    trend_weight: float = 0.40
    risk_on_threshold: float = 40.0
    risk_off_threshold: float = 65.0

    def __post_init__(self) -> None:
        weights = (self.macro_weight, self.volatility_weight, self.trend_weight)
        if any(weight < 0 for weight in weights) or not np.isclose(sum(weights), 1.0):
            raise ValueError(
                "macro, volatility, and trend weights must be non-negative and sum to 1"
            )
        if not 0 <= self.risk_on_threshold < self.risk_off_threshold <= 100:
            raise ValueError("regime thresholds must satisfy 0 <= risk_on < risk_off <= 100")


def classify_risk(
    score: float | None, config: ModelConfig | None = None
) -> tuple[str, str]:
    """Map a risk score to a stable regime and a non-execution action label."""

    policy = config or ModelConfig()
    if score is None or not np.isfinite(float(score)):
        return "unavailable", "wait_for_data"
    if float(score) < policy.risk_on_threshold:
        return "risk_on", "buy_eligible_with_stock_confirmation"
    if float(score) < policy.risk_off_threshold:
        return "neutral", "selective_or_smaller_positions"
    return "risk_off", "reduce_risk_and_avoid_new_buys"


def position_for_risk(score: float | None) -> float:
    """Return the candidate QQQ exposure band for a 0--100 risk score."""

    if score is None or not np.isfinite(float(score)):
        return np.nan
    value = float(score)
    if value < 25.0:
        return 0.95
    if value < 40.0:
        return 0.80
    if value < 55.0:
        return 0.55
    if value < 70.0:
        return 0.30
    return 0.10


def _reasons(row: pd.Series, regime: str) -> tuple[str, ...]:
    if regime == "unavailable":
        return ("insufficient_history",)

    reasons: list[str] = []
    sleeves = (
        ("macro_risk", "macro_conditions_tight", "macro_conditions_supportive"),
        ("volatility_risk", "volatility_elevated", "volatility_subdued"),
        ("trend_risk", "technology_trend_weak", "technology_trend_supportive"),
    )
    for column, high_reason, low_reason in sleeves:
        value = row.get(column)
        if pd.notna(value) and float(value) >= 65.0:
            reasons.append(high_reason)
        elif pd.notna(value) and float(value) <= 35.0:
            reasons.append(low_reason)

    if pd.notna(row.get("qqq_gap_200")):
        reasons.append("qqq_above_200dma" if row["qqq_gap_200"] >= 0 else "qqq_below_200dma")
    if pd.notna(row.get("smh_qqq_relative_60d")):
        reasons.append(
            "semiconductors_leading_qqq"
            if row["smh_qqq_relative_60d"] >= 0
            else "semiconductors_lagging_qqq"
        )
    if not reasons:
        reasons.append(f"composite_{regime}")
    return tuple(reasons)


def score_indicators(
    indicators: pd.DataFrame,
    config: ModelConfig | None = None,
) -> pd.DataFrame:
    """Add composite score, regime, action, position band, and explanations."""

    policy = config or ModelConfig()
    required = {"macro_risk", "volatility_risk", "trend_risk"}
    missing = sorted(required.difference(indicators.columns))
    if missing:
        raise ValueError(f"indicator frame is missing: {', '.join(missing)}")

    result = indicators.copy()
    complete = result[list(required)].notna().all(axis=1)
    result["risk_score"] = (
        result["macro_risk"] * policy.macro_weight
        + result["volatility_risk"] * policy.volatility_weight
        + result["trend_risk"] * policy.trend_weight
    ).where(complete).clip(0.0, 100.0)

    classifications = [classify_risk(value, policy) for value in result["risk_score"]]
    result["regime"] = [item[0] for item in classifications]
    result["action"] = [item[1] for item in classifications]
    result["target_position"] = [position_for_risk(value) for value in result["risk_score"]]
    result["reasons"] = [
        _reasons(row, regime) for (_, row), regime in zip(result.iterrows(), result["regime"])
    ]
    return result


def score_market(
    data: pd.DataFrame,
    config: ModelConfig | None = None,
    percentile_window: int = 252,
    min_periods: int = 60,
) -> pd.DataFrame:
    """Build indicators and score every trading day with one point-in-time path."""

    indicators = build_indicators(
        data, percentile_window=percentile_window, min_periods=min_periods
    )
    return score_indicators(indicators, config=config)


def latest_snapshot(scored: pd.DataFrame) -> dict[str, Any]:
    """Return the latest valid model observation as a JSON-friendly mapping."""

    required = {"risk_score", "regime", "action", "target_position", "reasons"}
    missing = sorted(required.difference(scored.columns))
    if missing:
        raise ValueError(f"scored frame is missing: {', '.join(missing)}")
    valid = scored.loc[scored["risk_score"].notna()]
    if valid.empty:
        return {
            "as_of": None,
            "risk_score": None,
            "regime": "unavailable",
            "action": "wait_for_data",
            "target_position": None,
            "reasons": ["insufficient_history"],
        }
    as_of, row = valid.iloc[-1].name, valid.iloc[-1]
    as_of_value = as_of.isoformat() if hasattr(as_of, "isoformat") else str(as_of)
    return {
        "as_of": as_of_value,
        "risk_score": round(float(row["risk_score"]), 2),
        "macro_risk": round(float(row["macro_risk"]), 2),
        "volatility_risk": round(float(row["volatility_risk"]), 2),
        "trend_risk": round(float(row["trend_risk"]), 2),
        "regime": str(row["regime"]),
        "action": str(row["action"]),
        "target_position": float(row["target_position"]),
        "reasons": list(row["reasons"]),
    }
