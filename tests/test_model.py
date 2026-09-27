import numpy as np
import pandas as pd
import pandas.testing as pdt

from us_tech_monitor.indicators import historical_percentile
from us_tech_monitor.model import ModelConfig, classify_risk, score_market


def _market_data(periods: int = 520) -> pd.DataFrame:
    index = pd.bdate_range("2024-01-02", periods=periods)
    x = np.arange(periods, dtype=float)
    return pd.DataFrame(
        {
            "QQQ": 100 * np.exp(0.0006 * x + 0.025 * np.sin(x / 17)),
            "SPY": 100 * np.exp(0.0004 * x + 0.015 * np.sin(x / 21)),
            "XLK": 100 * np.exp(0.00055 * x + 0.020 * np.sin(x / 19)),
            "SMH": 100 * np.exp(0.0007 * x + 0.035 * np.sin(x / 13)),
            "VXN": 22 + 4 * np.sin(x / 23),
            "REAL10Y": 1.5 + 0.4 * np.sin(x / 71),
            "NFCI": -0.2 + 0.2 * np.sin(x / 83),
            "CASH3M": 4.5 + 0.2 * np.sin(x / 91),
        },
        index=index,
    )


def test_historical_percentile_never_reads_future_values() -> None:
    original = pd.Series(np.arange(200, dtype=float))
    changed_future = original.copy()
    changed_future.iloc[150:] = -1_000_000
    left = historical_percentile(original, window=80, min_periods=20)
    right = historical_percentile(changed_future, window=80, min_periods=20)
    pdt.assert_series_equal(left.iloc[:150], right.iloc[:150])


def test_full_model_is_invariant_to_future_mutation() -> None:
    original = _market_data()
    changed = original.copy()
    changed.iloc[450:, changed.columns.get_loc("QQQ")] *= 3
    before = score_market(original, percentile_window=126, min_periods=40)
    after = score_market(changed, percentile_window=126, min_periods=40)
    pdt.assert_series_equal(before["risk_score"].iloc[:450], after["risk_score"].iloc[:450])


def test_threshold_boundaries_are_explicit() -> None:
    policy = ModelConfig(risk_on_threshold=40, risk_off_threshold=65)
    assert classify_risk(39.999, policy)[0] == "risk_on"
    assert classify_risk(40.0, policy)[0] == "neutral"
    assert classify_risk(64.999, policy)[0] == "neutral"
    assert classify_risk(65.0, policy)[0] == "risk_off"
    assert classify_risk(np.nan, policy)[0] == "unavailable"


def test_scored_output_is_explainable_and_bounded() -> None:
    scored = score_market(_market_data(), percentile_window=126, min_periods=40)
    valid = scored.dropna(subset=["risk_score"])
    assert not valid.empty
    assert valid["risk_score"].between(0, 100).all()
    assert valid["target_position"].between(0, 1).all()
    assert {"macro_risk", "volatility_risk", "trend_risk", "regime", "action", "reasons"} <= set(
        valid.columns
    )
    assert all(isinstance(item, tuple) and item for item in valid["reasons"])
