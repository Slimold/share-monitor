"""Execution-aware backtest for the MU-SOXS confidence strategy."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from ..config import MuSoxsStrategyConfig


def _adjusted_ohlc(history: pd.DataFrame, label: str) -> pd.DataFrame:
    required = {"open", "high", "low", "close", "adj_close"}
    missing = sorted(required.difference(history.columns))
    if missing:
        raise ValueError(f"{label} history is missing: {', '.join(missing)}")
    values = history.loc[:, ["open", "high", "low", "close", "adj_close"]].apply(
        pd.to_numeric, errors="coerce"
    )
    scale = values["adj_close"] / values["close"].replace(0.0, np.nan)
    adjusted = pd.DataFrame(index=values.index)
    for column in ("open", "high", "low", "close"):
        adjusted[column] = values[column] * scale
    adjusted = adjusted.sort_index()
    adjusted = adjusted[~adjusted.index.duplicated(keep="last")]
    return adjusted.rename(columns={column: f"{label}_{column}" for column in adjusted})


def _atr(frame: pd.DataFrame, label: str, periods: int = 14) -> pd.Series:
    high = frame[f"{label}_high"]
    low = frame[f"{label}_low"]
    close = frame[f"{label}_close"]
    previous_close = close.shift(1)
    true_range = pd.concat(
        [high - low, (high - previous_close).abs(), (low - previous_close).abs()],
        axis=1,
    ).max(axis=1)
    return true_range.rolling(periods, min_periods=periods).mean()


def _slice_metrics(equity: pd.Series) -> dict[str, float | int | None]:
    if equity.empty:
        return {"sessions": 0, "total_return": None, "max_drawdown": None}
    start_equity = 1.0
    total_return = float(equity.iloc[-1] / start_equity - 1.0)
    high_water = pd.concat(
        [pd.Series([start_equity], index=[equity.index[0] - pd.Timedelta(nanoseconds=1)]), equity]
    ).cummax()
    drawdown = equity / high_water.reindex(equity.index, method="ffill") - 1.0
    return {
        "sessions": int(len(equity)),
        "total_return": total_return,
        "max_drawdown": float(drawdown.min()),
    }


def run_mu_soxs_backtest(
    confidence: pd.DataFrame,
    mu_history: pd.DataFrame,
    smh_history: pd.DataFrame,
    soxs_history: pd.DataFrame,
    *,
    start: str | pd.Timestamp,
    end: str | pd.Timestamp,
    config: MuSoxsStrategyConfig | None = None,
    transaction_cost_bps: float = 10.0,
) -> dict[str, Any]:
    """Backtest confidence signals with next-open execution and gap-aware stops.

    The portfolio begins in cash.  Position weights are determined at the
    signal close from a fixed account-risk budget divided by the volatility
    stop distance, subject to the configured MU/SOXS caps.
    """

    policy = config or MuSoxsStrategyConfig()
    if transaction_cost_bps < 0:
        raise ValueError("transaction_cost_bps must be non-negative")
    start = pd.Timestamp(start)
    end = pd.Timestamp(end)
    if start > end:
        raise ValueError("start must not be after end")

    mu = _adjusted_ohlc(mu_history, "mu")
    smh = _adjusted_ohlc(smh_history, "smh")
    soxs = _adjusted_ohlc(soxs_history, "soxs")
    frame = (
        confidence.join(mu.drop(columns="mu_close"), how="inner")
        .join(smh.drop(columns="smh_close"), how="inner")
        .join(soxs.drop(columns="soxs_close"), how="inner")
    )
    frame["mu_atr14"] = _atr(frame, "mu")
    frame["smh_atr14"] = _atr(frame, "smh")
    frame["mu_stop_distance"] = (
        1.5 * frame["mu_atr14"] / frame["mu_close"]
    ).clip(0.08, 0.15)
    frame["soxs_stop_distance"] = (
        3.0 * frame["smh_atr14"] / frame["smh_close"]
    ).clip(0.10, 0.15)
    frame["mu_target_weight"] = (
        policy.account_risk_budget / frame["mu_stop_distance"]
    ).clip(upper=policy.mu_max_weight)
    frame["soxs_target_weight"] = (
        policy.account_risk_budget / frame["soxs_stop_distance"]
    ).clip(upper=policy.soxs_max_weight)

    for column in ("mu_entry_allowed", "mu_add_allowed", "soxs_entry_allowed"):
        frame[f"{column}_event"] = frame[column].astype(bool) & ~frame[column].astype(bool).shift(
            1, fill_value=False
        )

    evaluation = frame.loc[(frame.index >= start) & (frame.index <= end)].copy()
    if len(evaluation) < 2:
        raise ValueError("selected MU-SOXS backtest window needs at least two sessions")

    fee_rate = transaction_cost_bps / 10_000.0
    cash = 1.0
    units = 0.0
    asset: str | None = None
    stop_price: float | None = None
    holding_sessions = 0
    trade_cost_basis = 0.0
    trade_entry_equity = 0.0
    trade_entry_date: pd.Timestamp | None = None
    trade_entry_price = 0.0
    trade_initial_weight = 0.0
    trades: list[dict[str, Any]] = []
    equity_values: list[float] = []
    equity_dates: list[pd.Timestamp] = []
    exposure_sessions = 0

    def price(row: pd.Series, instrument: str, field: str) -> float:
        return float(row[f"{instrument}_{field}"])

    def portfolio_value(row: pd.Series, field: str) -> float:
        if asset is None:
            return cash
        return cash + units * price(row, asset, field)

    def buy_to_weight(
        row: pd.Series,
        instrument: str,
        target_weight: float,
        distance: float,
        date: pd.Timestamp,
    ) -> None:
        nonlocal cash, units, asset, stop_price, holding_sessions
        nonlocal trade_cost_basis, trade_entry_equity, trade_entry_date
        nonlocal trade_entry_price, trade_initial_weight
        open_price = price(row, instrument, "open")
        equity = portfolio_value(row, "open")
        existing_stop = stop_price if asset == instrument else None
        target_notional = max(0.0, min(float(target_weight), 1.0)) * equity
        current_notional = units * open_price if asset == instrument else 0.0
        additional_notional = max(0.0, target_notional - current_notional)
        affordable = cash / (1.0 + fee_rate)
        additional_notional = min(additional_notional, affordable)
        if additional_notional <= 0:
            return
        additional_units = additional_notional / open_price
        fee = additional_notional * fee_rate
        if asset is None:
            asset = instrument
            trade_entry_equity = equity
            trade_entry_date = date
            trade_entry_price = open_price
            trade_initial_weight = target_notional / equity if equity > 0 else 0.0
            trade_cost_basis = 0.0
            holding_sessions = 0
        else:
            trade_entry_price = (
                trade_entry_price * units + open_price * additional_units
            ) / (units + additional_units)
        units += additional_units
        cash -= additional_notional + fee
        trade_cost_basis += additional_notional + fee
        candidate_stop = trade_entry_price * (1.0 - float(distance))
        # Adding to a long position may tighten its protective stop, but must
        # never move an already-live stop lower and silently increase risk.
        stop_price = (
            max(float(existing_stop), candidate_stop)
            if existing_stop is not None
            else candidate_stop
        )

    def sell_all(row: pd.Series, field: str, date: pd.Timestamp, reason: str) -> None:
        nonlocal cash, units, asset, stop_price, holding_sessions
        nonlocal trade_cost_basis, trade_entry_equity, trade_entry_date
        nonlocal trade_entry_price, trade_initial_weight
        if asset is None or units <= 0:
            return
        exit_price = price(row, asset, field) if field != "stop" else float(stop_price)
        proceeds = units * exit_price
        fee = proceeds * fee_rate
        cash += proceeds - fee
        net_return = (proceeds - fee) / trade_cost_basis - 1.0 if trade_cost_basis else 0.0
        trades.append(
            {
                "asset": asset.upper(),
                "entry_date": trade_entry_date.date().isoformat() if trade_entry_date else None,
                "exit_date": date.date().isoformat(),
                "entry_price": trade_entry_price,
                "exit_price": exit_price,
                "return": net_return,
                "portfolio_pnl": cash - trade_entry_equity,
                "initial_weight": trade_initial_weight,
                "holding_sessions": holding_sessions,
                "exit_reason": reason,
            }
        )
        units = 0.0
        asset = None
        stop_price = None
        holding_sessions = 0
        trade_cost_basis = 0.0
        trade_entry_equity = 0.0
        trade_entry_date = None
        trade_entry_price = 0.0
        trade_initial_weight = 0.0

    previous_row: pd.Series | None = None
    for date, row in evaluation.iterrows():
        # A protective stop that was already live before today's open takes
        # priority over any new add/switch order generated at yesterday's
        # close.  Otherwise a gap below the old stop could be hidden by first
        # averaging down and recalculating the stop from the larger position.
        stopped_at_open = False
        if asset is not None and stop_price is not None:
            open_price = price(row, asset, "open")
            if open_price <= stop_price:
                sell_all(row, "open", date, "gap_stop")
                stopped_at_open = True

        # A close-t signal may only trade at the following session's open.
        if previous_row is not None and not stopped_at_open:
            if asset == "mu":
                if bool(previous_row["soxs_entry_allowed_event"]):
                    sell_all(row, "open", date, "switch_to_soxs")
                    buy_to_weight(
                        row,
                        "soxs",
                        float(previous_row["soxs_target_weight"]),
                        float(previous_row["soxs_stop_distance"]),
                        date,
                    )
                elif bool(previous_row["mu_exit_condition"]):
                    sell_all(row, "open", date, "mu_rsi_exit")
                elif bool(previous_row["mu_add_allowed_event"]):
                    buy_to_weight(
                        row,
                        "mu",
                        float(previous_row["mu_target_weight"]),
                        float(previous_row["mu_stop_distance"]),
                        date,
                    )
            elif asset == "soxs":
                time_exit = holding_sessions >= policy.soxs_max_holding_sessions
                if bool(previous_row["mu_entry_allowed_event"]):
                    sell_all(row, "open", date, "switch_to_mu")
                    buy_to_weight(
                        row,
                        "mu",
                        float(previous_row["mu_target_weight"])
                        * policy.mu_starter_fraction,
                        float(previous_row["mu_stop_distance"]),
                        date,
                    )
                elif bool(previous_row["soxs_exit_condition"]) or time_exit:
                    sell_all(
                        row,
                        "open",
                        date,
                        "soxs_signal_exit" if not time_exit else "soxs_time_exit",
                    )
            else:
                if bool(previous_row["soxs_entry_allowed_event"]):
                    buy_to_weight(
                        row,
                        "soxs",
                        float(previous_row["soxs_target_weight"]),
                        float(previous_row["soxs_stop_distance"]),
                        date,
                    )
                elif bool(previous_row["mu_entry_allowed_event"]):
                    buy_to_weight(
                        row,
                        "mu",
                        float(previous_row["mu_target_weight"])
                        * policy.mu_starter_fraction,
                        float(previous_row["mu_stop_distance"]),
                        date,
                    )

        # After open actions, new and surviving positions remain protected
        # against an intraday breach at the declared stop price.
        if asset is not None and stop_price is not None:
            low_price = price(row, asset, "low")
            if low_price <= stop_price:
                sell_all(row, "stop", date, "intraday_stop")

        close_equity = portfolio_value(row, "close")
        equity_dates.append(date)
        equity_values.append(close_equity)
        if asset is not None:
            holding_sessions += 1
            exposure_sessions += 1
        previous_row = row

    equity = pd.Series(equity_values, index=pd.DatetimeIndex(equity_dates), name="equity")
    metrics = _slice_metrics(equity)
    completed = len(trades)
    wins = sum(float(trade["return"]) > 0 for trade in trades)
    gross_profit = sum(max(float(trade["portfolio_pnl"]), 0.0) for trade in trades)
    gross_loss = -sum(min(float(trade["portfolio_pnl"]), 0.0) for trade in trades)
    metrics.update(
        {
            "completed_trades": completed,
            "winning_trades": wins,
            "win_rate": wins / completed if completed else None,
            "profit_factor": gross_profit / gross_loss if gross_loss > 0 else None,
            "exposure_sessions": exposure_sessions,
            "exposure_fraction": exposure_sessions / len(evaluation),
            "ending_equity": float(equity.iloc[-1]),
            "open_position": asset.upper() if asset else None,
        }
    )

    development_end = min(pd.Timestamp(policy.development_end), end)
    development_equity = equity.loc[equity.index <= development_end]
    forward_equity = equity.loc[equity.index > pd.Timestamp(policy.development_end)]
    if not forward_equity.empty and not development_equity.empty:
        forward_equity = forward_equity / float(development_equity.iloc[-1])

    return {
        "version": policy.version,
        "window": {
            "start": evaluation.index[0].date().isoformat(),
            "end": evaluation.index[-1].date().isoformat(),
            "development_end": policy.development_end,
        },
        "assumptions": {
            "signal_timing": "close_t_signal_executes_at_open_t_plus_1",
            "transaction_cost_bps": float(transaction_cost_bps),
            "cash_return": 0.0,
            "gap_stop": "execute_at_open_when_open_crosses_stop",
            "mu_stop": "1.5_atr14_clipped_8_to_15_percent",
            "soxs_stop": "3_smh_atr14_clipped_10_to_15_percent",
            "position_sizing": "account_risk_budget_divided_by_stop_distance_with_caps",
        },
        "metrics": metrics,
        "development_metrics": _slice_metrics(development_equity),
        "post_development_metrics": _slice_metrics(forward_equity),
        "trades": trades,
        "equity_curve": [
            {"date": date.date().isoformat(), "equity": float(value)}
            for date, value in equity.items()
        ],
        "limitations": [
            "june_through_august_was_used_to_design_the_rules",
            "post_development_window_is_short_and_not_statistically_conclusive",
            "earnings_calendar_is_not_modeled",
        ],
    }
