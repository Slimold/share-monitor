import numpy as np
import pandas as pd

from us_tech_monitor.watchlist import score_watchlist


def _history(start: float, stop: float, periods: int = 260) -> pd.DataFrame:
    index = pd.bdate_range("2025-01-02", periods=periods)
    return pd.DataFrame({"adj_close": np.linspace(start, stop, periods)}, index=index)


def test_leader_is_buy_eligible_when_market_allows_risk() -> None:
    qqq = _history(100, 130)
    leader = _history(100, 180)
    result = score_watchlist({"LEAD": leader}, qqq, market_risk=30)[0]
    assert result["status"] == "buy_eligible"
    assert "relative_strength_positive" in result["reasons"]


def test_market_gate_does_not_force_intact_leader_to_exit() -> None:
    qqq = _history(100, 130)
    leader = _history(100, 180)
    result = score_watchlist({"LEAD": leader}, qqq, market_risk=80)[0]
    assert result["status"] == "hold_no_new_position"


def test_neutral_market_limits_an_otherwise_eligible_leader() -> None:
    qqq = _history(100, 130)
    leader = _history(100, 180)
    result = score_watchlist({"LEAD": leader}, qqq, market_risk=48)[0]
    assert result["status"] == "small_position_only"


def test_weak_stock_can_trigger_exit_condition() -> None:
    qqq = _history(100, 130)
    weak = _history(180, 80)
    result = score_watchlist({"WEAK": weak}, qqq, market_risk=40)[0]
    assert result["status"] == "exit_condition_triggered"
