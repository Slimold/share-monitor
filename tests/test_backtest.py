import numpy as np
import pandas as pd

from us_tech_monitor.backtest import (
    positions_from_risk,
    qqq_200dma_positions,
    run_backtest,
    run_full_and_holdout,
    run_sensitivity_experiments,
)


def _scored(positions: list[float], returns: list[float] | None = None) -> pd.DataFrame:
    index = pd.bdate_range("2026-01-02", periods=len(positions))
    if returns is None:
        prices = np.full(len(positions), 100.0)
    else:
        prices = 100 * np.cumprod(1 + np.asarray(returns, dtype=float))
    return pd.DataFrame(
        {"QQQ": prices, "CASH3M": 0.0, "target_position": positions}, index=index
    )


def test_signal_executes_next_close_and_earns_following_close_to_close_return() -> None:
    data = _scored([1.0, 0.0, 0.0, 0.0], returns=[0.0, 0.10, 0.20, 0.0])
    result = run_backtest(data, transaction_cost_bps=0)
    # The day-one close signal misses the next day's +10% move: it is filled at
    # that next close and first participates in the subsequent +20% move.
    assert result.equity.index.tolist() == data.index[2:].tolist()
    assert result.equity["applied_position"].tolist() == [1.0, 0.0]
    assert np.isclose(result.equity["strategy_return"].iloc[0], 0.20)


def test_ten_basis_point_cost_is_charged_on_each_position_change() -> None:
    data = _scored([0.0, 1.0, 1.0, 0.0, 0.0, 0.0])
    result = run_backtest(data, transaction_cost_bps=10)
    assert np.isclose(result.equity["position_change"].sum(), 2.0)
    assert np.isclose(result.equity["transaction_cost"].sum(), 0.002)
    assert np.isclose(result.equity["strategy_equity"].iloc[-1], 0.999 * 0.999)
    assert np.isclose(result.metrics["strategy"]["max_drawdown"], 0.999 * 0.999 - 1.0)


def test_daily_rebalancing_turnover_uses_the_drifted_pretrade_weight() -> None:
    data = _scored([0.5, 0.5, 0.5, 0.5], returns=[0.0, 0.0, 0.10, 0.0])
    result = run_backtest(data, transaction_cost_bps=10)

    drifted_weight = 0.5 * 1.10 / 1.05
    expected_changes = [0.5, abs(0.5 - drifted_weight)]
    assert np.allclose(result.equity["pretrade_position"], [0.0, drifted_weight])
    assert np.allclose(result.equity["position_change"], expected_changes)
    assert np.isclose(result.metrics["strategy"]["turnover"], sum(expected_changes))
    assert np.isclose(result.equity["transaction_cost"].sum(), sum(expected_changes) / 1000)


def test_benchmark_pays_initial_purchase_cost_for_every_reported_slice() -> None:
    data = _scored([0.0] * 8)
    runs = run_full_and_holdout(data, holdout_sessions=3, transaction_cost_bps=10)

    for result in runs.values():
        assert result.equity["benchmark_transaction_cost"].iloc[0] == 0.001
        assert result.equity["benchmark_transaction_cost"].iloc[1:].eq(0.0).all()
        assert result.equity["benchmark_return"].iloc[0] == -0.001
        assert np.isclose(result.equity["benchmark_equity"].iloc[-1], 0.999)
        assert result.metrics["benchmark"]["turnover"] == 1.0


def test_cash_return_uses_annual_percentage_yield_divided_by_252() -> None:
    data = _scored([0.0, 0.0, 0.0, 0.0])
    data["CASH3M"] = 5.04
    result = run_backtest(data, transaction_cost_bps=0)
    assert np.allclose(result.equity["cash_return"], 0.0504 / 252)
    assert np.allclose(result.equity["strategy_return"], 0.0504 / 252)


def test_cash_yield_is_known_before_the_return_period() -> None:
    data = _scored([0.0, 0.0, 0.0, 0.0])
    data["CASH3M"] = [0.0, 2.52, 5.04, 7.56]
    result = run_backtest(data, transaction_cost_bps=0)
    assert np.allclose(result.equity["cash_return"], [0.0252 / 252, 0.0504 / 252])


def test_full_and_holdout_have_requested_sample_boundaries() -> None:
    positions = [0.5] * 30
    result = run_full_and_holdout(_scored(positions), holdout_sessions=10)
    assert result["full"].metrics["strategy"]["observations"] == 28
    assert result["holdout"].metrics["strategy"]["observations"] == 10
    required = {"cagr", "volatility", "sharpe", "max_drawdown", "calmar", "turnover"}
    assert required <= set(result["holdout"].metrics["strategy"])


def test_predeclared_policy_buckets_include_exact_boundaries() -> None:
    scores = pd.Series([0, 24.999, 25, 39.999, 40, 54.999, 55, 69.999, 70, 100, np.nan])
    assert positions_from_risk(scores, policy="conservative").tolist()[:10] == [
        0.95,
        0.95,
        0.80,
        0.80,
        0.55,
        0.55,
        0.30,
        0.30,
        0.10,
        0.10,
    ]
    assert positions_from_risk(scores, policy="balanced").tolist()[:10] == [
        1.0,
        1.0,
        0.9,
        0.9,
        0.75,
        0.75,
        0.5,
        0.5,
        0.25,
        0.25,
    ]
    assert positions_from_risk(scores, policy="growth").tolist()[:10] == [
        1.0,
        1.0,
        1.0,
        1.0,
        0.85,
        0.85,
        0.65,
        0.65,
        0.35,
        0.35,
    ]
    assert np.isnan(positions_from_risk(scores).iloc[-1])


def test_ten_day_smoothing_only_uses_current_and_past_scores() -> None:
    scores = pd.Series(np.linspace(0, 100, 40))
    changed_future = scores.copy()
    changed_future.iloc[30:] = 0
    before = positions_from_risk(scores, smoothing_days=10)
    after = positions_from_risk(changed_future, smoothing_days=10)
    assert before.iloc[:30].equals(after.iloc[:30])


def test_qqq_200dma_signal_executes_at_next_close_for_following_return() -> None:
    index = pd.bdate_range("2025-01-02", periods=205)
    data = pd.DataFrame(
        {
            "QQQ": np.r_[np.full(200, 100.0), np.full(5, 120.0)],
            "CASH3M": 0.0,
            "target_position": 0.0,
        },
        index=index,
    )
    data["target_position"] = qqq_200dma_positions(data)
    result = run_backtest(data, transaction_cost_bps=0)
    # The first signal trades at the next close, then earns the return ending
    # two sessions after the signal date.
    first_signal_date = data["target_position"].first_valid_index()
    execution_date = data.index[data.index.get_loc(first_signal_date) + 1]
    first_return_date = data.index[data.index.get_loc(first_signal_date) + 2]
    assert first_signal_date not in result.equity.index
    assert execution_date not in result.equity.index
    assert result.equity.index[0] == first_return_date
    assert result.equity.loc[first_return_date, "applied_position"] == 1.0


def test_qqq_200dma_reuses_warmup_column_at_evaluation_start() -> None:
    index = pd.bdate_range("2026-01-02", periods=5)
    scored = pd.DataFrame(
        {
            "QQQ": [100.0, 101.0, 102.0, 103.0, 104.0],
            # This column was calculated before the formal evaluation slice.
            "qqq_sma_200": [95.0, 95.2, 95.4, 95.6, 95.8],
            "CASH3M": 0.0,
            "target_position": 0.0,
        },
        index=index,
    )
    scored["target_position"] = qqq_200dma_positions(scored)
    assert scored["target_position"].notna().all()
    assert scored["target_position"].iloc[0] == 1.0
    result = run_backtest(scored, transaction_cost_bps=0)
    assert result.metrics["strategy"]["observations"] == len(scored) - 2


def test_full_and_holdout_start_keeps_warmup_and_uses_last_executable_days() -> None:
    data = _scored([0.5] * 15)
    formal_start = data.index[5]
    runs = run_full_and_holdout(data, start=formal_start, holdout_sessions=3)

    assert runs["full"].equity.index[0] == formal_start
    assert runs["full"].metrics["strategy"]["observations"] == 10
    assert runs["holdout"].equity.index.tolist() == data.index[-3:].tolist()
    assert runs["holdout"].metrics["strategy"]["observations"] == 3


def test_sensitivity_experiments_report_full_and_holdout_metrics() -> None:
    index = pd.bdate_range("2024-01-02", periods=320)
    x = np.arange(len(index), dtype=float)
    scored = pd.DataFrame(
        {
            "QQQ": 100 * np.exp(0.0005 * x + 0.02 * np.sin(x / 20)),
            "CASH3M": 4.0,
            "risk_score": 50 + 35 * np.sin(x / 31),
            "target_position": 0.5,
        },
        index=index,
    )
    experiments = run_sensitivity_experiments(scored, holdout_sessions=20)
    assert set(experiments) == {
        "conservative",
        "conservative_ma10",
        "balanced",
        "balanced_ma10",
        "growth",
        "growth_ma10",
        "qqq_200dma",
    }
    for runs in experiments.values():
        assert runs["holdout"].metrics["strategy"]["observations"] == 20
        assert "cagr" in runs["full"].metrics["strategy"]


def test_sensitivity_start_calculates_ma10_on_the_complete_warmup_history() -> None:
    index = pd.bdate_range("2026-01-02", periods=20)
    risk_score = pd.Series(np.linspace(0.0, 100.0, len(index)), index=index)
    scored = pd.DataFrame(
        {
            "QQQ": np.linspace(100.0, 110.0, len(index)),
            "qqq_sma_200": 90.0,
            "CASH3M": 0.0,
            "risk_score": risk_score,
            "target_position": 0.5,
        },
        index=index,
    )
    formal_start = index[12]
    experiments = run_sensitivity_experiments(
        scored,
        start=formal_start,
        holdout_sessions=3,
    )

    ma10 = experiments["conservative_ma10"]["full"]
    expected = positions_from_risk(risk_score, smoothing_days=10).shift(2).loc[formal_start]
    assert ma10.equity.index[0] == formal_start
    assert ma10.equity.loc[formal_start, "applied_position"] == expected
    for runs in experiments.values():
        assert runs["full"].equity.index[0] == formal_start
        assert runs["holdout"].equity.index.tolist() == index[-3:].tolist()
