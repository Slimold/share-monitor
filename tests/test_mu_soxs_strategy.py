from __future__ import annotations

import numpy as np
import pandas as pd

from us_tech_monitor.config import MuSoxsStrategyConfig
from us_tech_monitor.strategies import (
    build_confidence_history,
    latest_confidence_payload,
    run_mu_soxs_backtest,
)


def _history(values: np.ndarray, index: pd.DatetimeIndex) -> pd.DataFrame:
    values = np.asarray(values, dtype=float)
    return pd.DataFrame(
        {
            "open": values,
            "high": values * 1.01,
            "low": values * 0.99,
            "close": values,
            "adj_close": values,
            "volume": np.full(len(values), 1_000_000.0),
        },
        index=index,
    )


def _oversold_inputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series]:
    index = pd.bdate_range("2025-01-02", periods=330)
    mu = np.linspace(100, 140, len(index))
    mu[-10:] = [139, 136, 132, 127, 121, 114, 108, 102, 96, 91]
    smh = np.linspace(100, 125, len(index))
    smh[-10:] = [124, 123, 122, 121, 120, 119, 118, 117, 116, 115]
    soxs = np.linspace(60, 35, len(index))
    soxs[-10:] = [36, 37, 38, 39, 40, 41, 42, 43, 44, 45]
    risk = pd.Series(70.0, index=index)
    return _history(mu, index), _history(smh, index), _history(soxs, index), risk


def _rollover_inputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series]:
    index = pd.bdate_range("2025-01-02", periods=330)
    mu = np.linspace(100, 140, len(index))
    mu[-12:] = [138, 140, 143, 147, 151, 155, 159, 162, 160, 158, 156, 154]
    smh = np.linspace(100, 125, len(index))
    smh[-12:] = [126, 127, 128, 129, 130, 131, 132, 131, 130, 129, 128, 127]
    soxs = np.linspace(60, 35, len(index))
    soxs[-12:] = [35, 34, 33, 32, 31, 30, 29, 30, 31, 32, 33, 34]
    risk = pd.Series(75.0, index=index)
    return _history(mu, index), _history(smh, index), _history(soxs, index), risk


def test_oversold_mu_produces_high_buy_confidence() -> None:
    mu, smh, soxs, risk = _oversold_inputs()
    scored = build_confidence_history(mu, smh, soxs, risk)
    latest = scored.iloc[-1]

    assert 0 <= latest["buy_confidence"] <= 100
    assert 0 <= latest["sell_confidence"] <= 100
    assert latest["buy_confidence"] > latest["sell_confidence"]
    assert bool(latest["mu_entry_allowed"]) is True


def test_overbought_rollover_needs_sector_and_soxs_confirmation() -> None:
    mu, smh, soxs, risk = _rollover_inputs()
    scored = build_confidence_history(mu, smh, soxs, risk)
    latest = scored.iloc[-1]

    assert latest["sell_confidence"] > latest["buy_confidence"]
    assert bool(latest["smh_weakness_confirmed"]) is True
    assert bool(latest["soxs_trend_confirmed"]) is True
    assert bool(latest["soxs_entry_allowed"]) is True


def test_future_prices_do_not_change_an_earlier_confidence_score() -> None:
    mu, smh, soxs, risk = _rollover_inputs()
    cutoff = mu.index[-8]
    full = build_confidence_history(mu, smh, soxs, risk)
    truncated = build_confidence_history(
        mu.loc[:cutoff], smh.loc[:cutoff], soxs.loc[:cutoff], risk.loc[:cutoff]
    )

    columns = ["buy_confidence", "sell_confidence", "market_risk"]
    pd.testing.assert_series_equal(full.loc[cutoff, columns], truncated.loc[cutoff, columns])


def test_missing_exact_date_returns_unavailable_payload() -> None:
    mu, smh, soxs, risk = _rollover_inputs()
    as_of = mu.index[-1]
    scored = build_confidence_history(mu, smh, soxs.iloc[:-1], risk)
    payload = latest_confidence_payload(scored, as_of=as_of)

    assert payload["status"] == "unavailable"
    assert payload["buy_confidence"] is None
    assert payload["sell_confidence"] is None


def test_backtest_executes_close_signal_at_next_open() -> None:
    index = pd.bdate_range("2026-06-01", periods=35)
    mu_values = np.linspace(100, 120, len(index))
    smh_values = np.linspace(200, 205, len(index))
    soxs_values = np.linspace(50, 48, len(index))
    mu = _history(mu_values, index)
    smh = _history(smh_values, index)
    soxs = _history(soxs_values, index)
    confidence = pd.DataFrame(
        {
            "mu_close": mu_values,
            "smh_close": smh_values,
            "soxs_close": soxs_values,
            "mu_entry_allowed": False,
            "mu_add_allowed": False,
            "mu_exit_condition": False,
            "soxs_entry_allowed": False,
            "soxs_exit_condition": False,
        },
        index=index,
    )
    signal_date = index[20]
    exit_signal_date = index[24]
    confidence.loc[signal_date, "mu_entry_allowed"] = True
    confidence.loc[exit_signal_date, "mu_exit_condition"] = True

    result = run_mu_soxs_backtest(
        confidence,
        mu,
        smh,
        soxs,
        start=index[0],
        end=index[-1],
        config=MuSoxsStrategyConfig(),
    )

    assert result["metrics"]["completed_trades"] == 1
    trade = result["trades"][0]
    assert trade["entry_date"] == index[21].date().isoformat()
    assert trade["exit_date"] == index[25].date().isoformat()
    assert trade["return"] > 0


def test_existing_gap_stop_executes_before_a_new_add_order() -> None:
    index = pd.bdate_range("2026-06-01", periods=40)
    mu_values = np.full(len(index), 100.0)
    smh_values = np.full(len(index), 200.0)
    soxs_values = np.full(len(index), 50.0)
    mu = _history(mu_values, index)
    smh = _history(smh_values, index)
    soxs = _history(soxs_values, index)
    confidence = pd.DataFrame(
        {
            "mu_close": mu_values,
            "smh_close": smh_values,
            "soxs_close": soxs_values,
            "mu_entry_allowed": False,
            "mu_add_allowed": False,
            "mu_exit_condition": False,
            "soxs_entry_allowed": False,
            "soxs_exit_condition": False,
        },
        index=index,
    )
    entry_signal_date = index[20]
    add_signal_date = index[21]
    gap_date = index[22]
    confidence.loc[entry_signal_date, "mu_entry_allowed"] = True
    confidence.loc[add_signal_date, "mu_add_allowed"] = True
    mu.loc[gap_date, ["open", "high", "low", "close", "adj_close"]] = [
        90.0,
        91.0,
        89.0,
        90.0,
        90.0,
    ]

    result = run_mu_soxs_backtest(
        confidence,
        mu,
        smh,
        soxs,
        start=index[0],
        end=index[-1],
        config=MuSoxsStrategyConfig(),
    )

    assert result["metrics"]["completed_trades"] == 1
    assert result["metrics"]["open_position"] is None
    trade = result["trades"][0]
    assert trade["exit_date"] == gap_date.date().isoformat()
    assert trade["exit_reason"] == "gap_stop"
    assert trade["exit_price"] == 90.0
