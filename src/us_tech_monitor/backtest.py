"""Reproducible end-of-day backtest for the market thermometer."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


RISK_BUCKET_EDGES: tuple[float, ...] = (25.0, 40.0, 55.0, 70.0)
POSITION_POLICIES: dict[str, tuple[float, ...]] = {
    # Current capital-preservation policy used by the MVP model.
    "conservative": (0.95, 0.80, 0.55, 0.30, 0.10),
    # Pre-declared sensitivity variants; neither is selected from results.
    "balanced": (1.00, 0.90, 0.75, 0.50, 0.25),
    "growth": (1.00, 1.00, 0.85, 0.65, 0.35),
}


@dataclass(frozen=True)
class BacktestResult:
    """Backtest timeseries and an auditable metric summary."""

    equity: pd.DataFrame
    metrics: dict[str, Any]


def positions_from_risk(
    risk_score: pd.Series,
    *,
    policy: str = "conservative",
    smoothing_days: int = 1,
) -> pd.Series:
    """Map point-in-time risk scores to one of three pre-declared policies.

    ``smoothing_days=10`` is a planned sensitivity experiment using the current
    and previous nine scores. It is not tuned or selected based on performance.
    The returned value remains a close-``t`` signal. :func:`run_backtest`
    executes it at the close of ``t+1`` and applies the resulting position to
    the close-to-close return from ``t+1`` to ``t+2``.
    """

    if policy not in POSITION_POLICIES:
        choices = ", ".join(sorted(POSITION_POLICIES))
        raise ValueError(f"unknown position policy {policy!r}; choose from {choices}")
    if smoothing_days < 1:
        raise ValueError("smoothing_days must be positive")

    score = pd.to_numeric(risk_score, errors="coerce")
    if smoothing_days > 1:
        score = score.rolling(smoothing_days, min_periods=smoothing_days).mean()
    exposures = POSITION_POLICIES[policy]
    values = np.select(
        [
            score < RISK_BUCKET_EDGES[0],
            score < RISK_BUCKET_EDGES[1],
            score < RISK_BUCKET_EDGES[2],
            score < RISK_BUCKET_EDGES[3],
        ],
        exposures[:4],
        default=exposures[4],
    ).astype(float)
    values[score.isna().to_numpy()] = np.nan
    return pd.Series(values, index=risk_score.index, name="target_position")


def qqq_200dma_positions(scored: pd.DataFrame, window: int = 200) -> pd.Series:
    """Return an untuned 100%-QQQ/100%-cash trend baseline signal.

    Prefer the point-in-time moving average already calculated on the warm-up
    history. Recomputing it after an evaluation slice would impose an extra
    200-session warm-up on this baseline and make comparisons unfair.
    """

    if "QQQ" not in scored:
        raise ValueError("scored frame is missing: QQQ")
    if window < 2:
        raise ValueError("window must be at least 2")
    price = pd.to_numeric(scored["QQQ"], errors="coerce")
    if "qqq_sma_200" in scored:
        moving_average = pd.to_numeric(scored["qqq_sma_200"], errors="coerce")
    else:
        moving_average = price.rolling(window, min_periods=window).mean()
    signal = (price >= moving_average).astype(float).where(moving_average.notna())
    signal.name = "target_position"
    return signal


def _metric_set(
    returns: pd.Series,
    equity: pd.Series,
    cash_returns: pd.Series,
    *,
    annualization: int,
    turnover: float,
) -> dict[str, float | int | None]:
    observations = int(returns.notna().sum())
    if observations == 0:
        return {
            "observations": 0,
            "total_return": None,
            "cagr": None,
            "volatility": None,
            "sharpe": None,
            "max_drawdown": None,
            "calmar": None,
            "turnover": float(turnover),
        }

    ending = float(equity.iloc[-1])
    total_return = ending - 1.0
    cagr = ending ** (annualization / observations) - 1.0 if ending > 0 else -1.0
    volatility = float(returns.std(ddof=1) * np.sqrt(annualization)) if observations > 1 else 0.0
    excess = returns - cash_returns.reindex(returns.index).fillna(0.0)
    excess_std = float(excess.std(ddof=1)) if observations > 1 else 0.0
    sharpe = (
        float(excess.mean() / excess_std * np.sqrt(annualization))
        if excess_std > 0
        else None
    )
    # Include the starting capital (1.0) in the high-water mark so an initial
    # loss or entry cost is not hidden by making the first lower equity value
    # its own peak.
    drawdown = equity / equity.cummax().clip(lower=1.0) - 1.0
    max_drawdown = float(drawdown.min())
    calmar = float(cagr / abs(max_drawdown)) if max_drawdown < 0 else None
    return {
        "observations": observations,
        "total_return": total_return,
        "cagr": cagr,
        "volatility": volatility,
        "sharpe": sharpe,
        "max_drawdown": max_drawdown,
        "calmar": calmar,
        "turnover": float(turnover),
    }


def run_backtest(
    scored: pd.DataFrame,
    *,
    start: str | pd.Timestamp | None = None,
    end: str | pd.Timestamp | None = None,
    annualization: int = 252,
    transaction_cost_bps: float = 10.0,
) -> BacktestResult:
    """Backtest QQQ exposure using conservative next-close execution.

    A signal produced with session ``t`` closing data is executed at the close
    of ``t+1`` and therefore becomes the position for the return from close
    ``t+1`` to close ``t+2``. Uninvested capital earns the FRED-style annual
    percentage yield ``CASH3M / 100 / 252``. Transaction costs are proportional
    to one-way trades from the drifted pre-trade weight to the new target.
    Every reported slice starts in cash; the strategy's first allocation and
    the benchmark's initial QQQ purchase are both charged at the configured
    rate.
    """

    required = {"QQQ", "CASH3M", "target_position"}
    missing = sorted(required.difference(scored.columns))
    if missing:
        raise ValueError(f"scored frame is missing: {', '.join(missing)}")
    if annualization < 1:
        raise ValueError("annualization must be positive")
    if transaction_cost_bps < 0:
        raise ValueError("transaction_cost_bps must be non-negative")

    frame = scored.copy().sort_index()
    qqq_return = pd.to_numeric(frame["QQQ"], errors="coerce").pct_change(fill_method=None)
    target = pd.to_numeric(frame["target_position"], errors="coerce").clip(0.0, 1.0)
    # target[t] is decided after close t, trades at close t+1, and first earns
    # the return recorded at t+2. Thus it aligns with returns via shift(2).
    applied_raw = target.shift(2)

    # Exclude indicator warm-up from the performance sample.  If a later score
    # is unavailable, move to cash rather than silently carrying a stale signal.
    first_valid = applied_raw.first_valid_index()
    if first_valid is None:
        raise ValueError("no executable target position is available after indicator warm-up")
    eligible = pd.Series(frame.index >= first_valid, index=frame.index)
    if start is not None:
        eligible &= frame.index >= pd.Timestamp(start)
    if end is not None:
        eligible &= frame.index <= pd.Timestamp(end)
    eligible &= qqq_return.notna()
    if not eligible.any():
        raise ValueError("selected backtest slice has no executable observations")

    selected_index = frame.index[eligible]
    applied_position = applied_raw.reindex(selected_index).fillna(0.0)
    asset_return = qqq_return.reindex(selected_index)
    # The return ending on day t can only earn the yield observable at t-1.
    # This is also conservative for H.15 rates, which are published after the
    # cash close on their observation date.
    annual_cash_yield = pd.to_numeric(frame["CASH3M"], errors="coerce").ffill().shift(1)
    cash_return = annual_cash_yield.reindex(selected_index).fillna(0.0) / 100.0 / annualization

    gross_return = applied_position * asset_return + (1.0 - applied_position) * cash_return

    # Rebalancing is against the risky-asset weight after the previous day's
    # asset and cash returns, not merely against yesterday's target. A constant
    # target therefore still trades after relative price movement.
    pretrade_position = (
        applied_position.shift(1)
        * (1.0 + asset_return.shift(1))
        / (1.0 + gross_return.shift(1))
    )
    position_change = (applied_position - pretrade_position).abs()
    # Each reported slice is a standalone portfolio that begins fully in cash.
    pretrade_position.iloc[0] = 0.0
    position_change.iloc[0] = abs(float(applied_position.iloc[0]))
    transaction_cost = position_change * transaction_cost_bps / 10_000.0
    strategy_return = gross_return - transaction_cost

    benchmark_transaction_cost = pd.Series(0.0, index=selected_index)
    benchmark_transaction_cost.iloc[0] = transaction_cost_bps / 10_000.0
    benchmark_return = asset_return - benchmark_transaction_cost
    strategy_equity = (1.0 + strategy_return).cumprod()
    benchmark_equity = (1.0 + benchmark_return).cumprod()

    equity = pd.DataFrame(
        {
            "qqq_return": asset_return,
            "cash_return": cash_return,
            "signal_position": target.reindex(selected_index),
            "applied_position": applied_position,
            "pretrade_position": pretrade_position,
            "position_change": position_change,
            "transaction_cost": transaction_cost,
            "benchmark_transaction_cost": benchmark_transaction_cost,
            "gross_return": gross_return,
            "strategy_return": strategy_return,
            "benchmark_return": benchmark_return,
            "strategy_equity": strategy_equity,
            "benchmark_equity": benchmark_equity,
        },
        index=selected_index,
    )

    strategy_metrics = _metric_set(
        strategy_return,
        strategy_equity,
        cash_return,
        annualization=annualization,
        turnover=float(position_change.sum()),
    )
    benchmark_metrics = _metric_set(
        benchmark_return,
        benchmark_equity,
        cash_return,
        annualization=annualization,
        turnover=1.0,
    )
    metrics: dict[str, Any] = {
        "start": selected_index[0].isoformat()
        if hasattr(selected_index[0], "isoformat")
        else str(selected_index[0]),
        "end": selected_index[-1].isoformat()
        if hasattr(selected_index[-1], "isoformat")
        else str(selected_index[-1]),
        "annualization": annualization,
        "transaction_cost_bps": float(transaction_cost_bps),
        "strategy": strategy_metrics,
        "benchmark": benchmark_metrics,
        "comparison": {
            "cagr_difference": (
                strategy_metrics["cagr"] - benchmark_metrics["cagr"]
                if strategy_metrics["cagr"] is not None and benchmark_metrics["cagr"] is not None
                else None
            ),
            "max_drawdown_improvement": (
                strategy_metrics["max_drawdown"] - benchmark_metrics["max_drawdown"]
                if strategy_metrics["max_drawdown"] is not None
                and benchmark_metrics["max_drawdown"] is not None
                else None
            ),
        },
    }
    return BacktestResult(equity=equity, metrics=metrics)


def run_full_and_holdout(
    scored: pd.DataFrame,
    *,
    start: str | pd.Timestamp | None = None,
    holdout_sessions: int = 252,
    annualization: int = 252,
    transaction_cost_bps: float = 10.0,
) -> dict[str, BacktestResult]:
    """Run the formal sample and its trailing executable holdout slice.

    ``scored`` may include history before ``start``. Target positions are
    expected to have been calculated on that complete history; execution
    alignment is likewise calculated before the formal evaluation boundary.
    The holdout is the final ``holdout_sessions`` executable observations on or
    after ``start``.
    """

    if holdout_sessions < 1:
        raise ValueError("holdout_sessions must be positive")
    frame = scored.copy().sort_index()
    target = pd.to_numeric(frame["target_position"], errors="coerce").shift(2)
    qqq_return = pd.to_numeric(frame["QQQ"], errors="coerce").pct_change(fill_method=None)
    first_valid = target.first_valid_index()
    if first_valid is None:
        raise ValueError("no executable observations are available")
    executable = pd.Series(frame.index >= first_valid, index=frame.index) & qqq_return.notna()
    if start is not None:
        executable &= frame.index >= pd.Timestamp(start)
    executable_index = frame.index[executable]
    if executable_index.empty:
        raise ValueError("no executable observations are available")
    holdout_start = executable_index[-min(holdout_sessions, len(executable_index))]

    common = {
        "annualization": annualization,
        "transaction_cost_bps": transaction_cost_bps,
    }
    return {
        "full": run_backtest(frame, start=start, **common),
        "holdout": run_backtest(frame, start=holdout_start, **common),
    }


def run_sensitivity_experiments(
    scored: pd.DataFrame,
    *,
    start: str | pd.Timestamp | None = None,
    holdout_sessions: int = 252,
    annualization: int = 252,
    transaction_cost_bps: float = 10.0,
    include_ten_day_smoothing: bool = True,
) -> dict[str, dict[str, BacktestResult]]:
    """Run the fixed policy grid and a simple QQQ 200-day baseline.

    Returned keys are experiment names; every experiment contains ``full`` and
    ``holdout`` :class:`BacktestResult` objects. If ``start`` is supplied, all
    policy positions and moving averages are still calculated on the complete
    input history before results are sliced at that formal boundary. This API
    intentionally reports every pre-declared variant and makes no attempt to
    choose a winner.
    """

    if "risk_score" not in scored:
        raise ValueError("scored frame is missing: risk_score")

    history = scored.copy().sort_index()
    experiments: dict[str, dict[str, BacktestResult]] = {}
    common = {
        "start": start,
        "holdout_sessions": holdout_sessions,
        "annualization": annualization,
        "transaction_cost_bps": transaction_cost_bps,
    }
    smoothing_windows = (1, 10) if include_ten_day_smoothing else (1,)
    for policy_name in POSITION_POLICIES:
        for smoothing_days in smoothing_windows:
            name = policy_name if smoothing_days == 1 else f"{policy_name}_ma10"
            candidate = history.copy()
            candidate["target_position"] = positions_from_risk(
                candidate["risk_score"],
                policy=policy_name,
                smoothing_days=smoothing_days,
            )
            experiments[name] = run_full_and_holdout(candidate, **common)

    trend_baseline = history.copy()
    trend_baseline["target_position"] = qqq_200dma_positions(trend_baseline)
    experiments["qqq_200dma"] = run_full_and_holdout(trend_baseline, **common)
    return experiments
