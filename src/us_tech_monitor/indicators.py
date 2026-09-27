"""Point-in-time indicators for the U.S. technology thermometer.

All percentile transforms compare today's observation with observations that
were already known *before* today.  This seemingly small detail keeps the same
indicator implementation safe for both the dashboard and the backtest.
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pandas as pd


REQUIRED_COLUMNS: tuple[str, ...] = (
    "QQQ",
    "SPY",
    "XLK",
    "SMH",
    "VXN",
    "REAL10Y",
    "NFCI",
    "CASH3M",
)


def validate_market_data(data: pd.DataFrame) -> pd.DataFrame:
    """Return a sorted, numeric copy of the canonical market-data frame.

    The data pipeline is expected to align observations to U.S. trading days.
    Slow-moving observations such as NFCI are forward-filled here; values are
    never back-filled, so a publication cannot leak into an earlier date.
    """

    if not isinstance(data, pd.DataFrame):
        raise TypeError("data must be a pandas DataFrame")
    missing = sorted(set(REQUIRED_COLUMNS).difference(data.columns))
    if missing:
        raise ValueError(f"market data is missing required columns: {', '.join(missing)}")
    if data.index.has_duplicates:
        raise ValueError("market data index must not contain duplicate trading days")

    result = data.loc[:, list(REQUIRED_COLUMNS)].copy().sort_index()
    for column in REQUIRED_COLUMNS:
        result[column] = pd.to_numeric(result[column], errors="coerce")

    # Market and macro releases arrive at different frequencies.  Forward fill
    # is point-in-time safe; backward fill would create look-ahead bias.
    result.loc[:, ["VXN", "REAL10Y", "NFCI", "CASH3M"]] = result.loc[
        :, ["VXN", "REAL10Y", "NFCI", "CASH3M"]
    ].ffill()
    for column in ("QQQ", "SPY", "XLK", "SMH"):
        result.loc[result[column] <= 0, column] = np.nan
    return result


def historical_percentile(
    series: pd.Series,
    window: int = 252,
    min_periods: int = 60,
) -> pd.Series:
    """Rank each value against the preceding window, on a 0--100 scale.

    The current row is deliberately excluded from its reference distribution.
    Ties use a midpoint rank, which maps a constant history to 50 rather than
    incorrectly treating it as maximally risky.  No expanding full-sample
    statistics are used.
    """

    if window < 1:
        raise ValueError("window must be positive")
    if min_periods < 1 or min_periods > window:
        raise ValueError("min_periods must be between 1 and window")

    values = pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)
    output = np.full(values.shape, np.nan, dtype=float)
    for position, current in enumerate(values):
        if not np.isfinite(current):
            continue
        history = values[max(0, position - window) : position]
        history = history[np.isfinite(history)]
        if history.size < min_periods:
            continue
        less = np.count_nonzero(history < current)
        equal = np.count_nonzero(history == current)
        output[position] = 100.0 * (less + 0.5 * equal) / history.size
    return pd.Series(output, index=series.index, name=f"{series.name}_percentile")


def _weighted_available_score(
    frame: pd.DataFrame,
    weights: Mapping[str, float],
    *,
    minimum_components: int,
) -> pd.Series:
    """Combine available 0--100 components without rewarding missing data."""

    columns = list(weights)
    weight_series = pd.Series(weights, dtype=float)
    available = frame[columns].notna()
    numerator = frame[columns].mul(weight_series, axis="columns").sum(axis=1, min_count=1)
    denominator = available.mul(weight_series, axis="columns").sum(axis=1)
    score = numerator.div(denominator.where(denominator > 0))
    score[available.sum(axis=1) < minimum_components] = np.nan
    return score.clip(0.0, 100.0)


def build_indicators(
    data: pd.DataFrame,
    percentile_window: int = 252,
    min_periods: int = 60,
) -> pd.DataFrame:
    """Calculate raw factors, historical percentiles, and three risk sleeves.

    Candidate sleeve construction (fixed before seeing backtest results):

    * macro risk: real-yield level 50%, 20-day real-yield change 20%, NFCI 30%;
    * volatility risk: VXN 60%, QQQ 20-day realized volatility 40%;
    * trend risk: QQQ 200d gap 30%, QQQ 50d gap 25%, QQQ/SPY relative
      strength 20%, XLK/SPY 10%, and SMH/QQQ 15%.

    Higher values always mean greater risk.  These weights are intentionally
    policy constants, not parameters optimized on the supplied sample.
    """

    result = validate_market_data(data)
    qqq_log_return = np.log(result["QQQ"] / result["QQQ"].shift(1))

    result["qqq_return_1d"] = result["QQQ"].pct_change(fill_method=None)
    result["qqq_return_20d"] = result["QQQ"].pct_change(20, fill_method=None)
    result["qqq_return_60d"] = result["QQQ"].pct_change(60, fill_method=None)
    result["qqq_realized_vol_20d"] = qqq_log_return.rolling(20, min_periods=20).std() * np.sqrt(
        252.0
    )
    result["qqq_sma_50"] = result["QQQ"].rolling(50, min_periods=50).mean()
    result["qqq_sma_200"] = result["QQQ"].rolling(200, min_periods=200).mean()
    result["qqq_gap_50"] = result["QQQ"] / result["qqq_sma_50"] - 1.0
    result["qqq_gap_200"] = result["QQQ"] / result["qqq_sma_200"] - 1.0

    qqq_spy_ratio = result["QQQ"] / result["SPY"]
    xlk_spy_ratio = result["XLK"] / result["SPY"]
    smh_qqq_ratio = result["SMH"] / result["QQQ"]
    result["qqq_spy_relative_60d"] = qqq_spy_ratio.pct_change(60, fill_method=None)
    result["xlk_spy_relative_60d"] = xlk_spy_ratio.pct_change(60, fill_method=None)
    result["smh_qqq_relative_60d"] = smh_qqq_ratio.pct_change(60, fill_method=None)
    result["real10y_change_20d"] = result["REAL10Y"].diff(20)
    result["vxn_rv_spread"] = result["VXN"] / 100.0 - result["qqq_realized_vol_20d"]

    percentile_inputs = (
        "REAL10Y",
        "real10y_change_20d",
        "NFCI",
        "VXN",
        "qqq_realized_vol_20d",
        "qqq_gap_50",
        "qqq_gap_200",
        "qqq_spy_relative_60d",
        "xlk_spy_relative_60d",
        "smh_qqq_relative_60d",
    )
    for column in percentile_inputs:
        result[f"{column.lower()}_percentile"] = historical_percentile(
            result[column], window=percentile_window, min_periods=min_periods
        )

    result["macro_risk"] = _weighted_available_score(
        result,
        {
            "real10y_percentile": 0.50,
            "real10y_change_20d_percentile": 0.20,
            "nfci_percentile": 0.30,
        },
        minimum_components=3,
    )
    result["volatility_risk"] = _weighted_available_score(
        result,
        {"vxn_percentile": 0.60, "qqq_realized_vol_20d_percentile": 0.40},
        minimum_components=2,
    )

    # High trend/relative-strength percentiles are supportive, so invert them
    # before aggregation to keep the common convention: high == risky.
    trend_support_columns = (
        "qqq_gap_50_percentile",
        "qqq_gap_200_percentile",
        "qqq_spy_relative_60d_percentile",
        "xlk_spy_relative_60d_percentile",
        "smh_qqq_relative_60d_percentile",
    )
    for column in trend_support_columns:
        result[f"{column.removesuffix('_percentile')}_risk"] = 100.0 - result[column]
    result["trend_risk"] = _weighted_available_score(
        result,
        {
            "qqq_gap_200_risk": 0.30,
            "qqq_gap_50_risk": 0.25,
            "qqq_spy_relative_60d_risk": 0.20,
            "xlk_spy_relative_60d_risk": 0.10,
            "smh_qqq_relative_60d_risk": 0.15,
        },
        minimum_components=5,
    )
    return result
