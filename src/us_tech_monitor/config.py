from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class MuSoxsStrategyConfig:
    """Fixed policy for the MU long / SOXS defensive confidence model."""

    enabled: bool = True
    version: str = "mu-soxs-confidence-v2.1"
    signal_symbol: str = "MU"
    inverse_symbol: str = "SOXS"
    sector_proxy: str = "SMH"
    development_start: str = "2026-06-01"
    development_end: str = "2026-08-31"
    rsi_period: int = 6
    percentile_window: int = 252
    percentile_min_periods: int = 60
    buy_threshold: float = 65.0
    sell_threshold: float = 70.0
    confidence_margin: float = 15.0
    shock_mu_return: float = -0.08
    shock_sector_return: float = -0.05
    shock_cooldown_sessions: int = 5
    buy_oversold_weight: float = 0.45
    buy_downside_weight: float = 0.25
    buy_reversal_weight: float = 0.20
    buy_sector_weight: float = 0.10
    sell_overbought_weight: float = 0.30
    sell_extension_weight: float = 0.15
    sell_sector_weight: float = 0.25
    sell_rollover_weight: float = 0.20
    sell_market_weight: float = 0.10
    account_risk_budget: float = 0.0075
    mu_max_weight: float = 0.40
    soxs_max_weight: float = 0.20
    mu_starter_fraction: float = 0.50
    soxs_max_holding_sessions: int = 5

    def __post_init__(self) -> None:
        symbols = (self.signal_symbol, self.inverse_symbol, self.sector_proxy)
        if any(not symbol.strip() for symbol in symbols):
            raise ValueError("MU-SOXS strategy symbols must not be empty")
        if self.signal_symbol == self.inverse_symbol:
            raise ValueError("signal_symbol and inverse_symbol must differ")
        if self.rsi_period < 2:
            raise ValueError("rsi_period must be at least 2")
        if self.percentile_window < 2:
            raise ValueError("percentile_window must be at least 2")
        if not 1 <= self.percentile_min_periods <= self.percentile_window:
            raise ValueError(
                "percentile_min_periods must be between 1 and percentile_window"
            )
        for name, value in (
            ("buy_threshold", self.buy_threshold),
            ("sell_threshold", self.sell_threshold),
            ("confidence_margin", self.confidence_margin),
        ):
            if not 0 <= value <= 100:
                raise ValueError(f"{name} must be between 0 and 100")
        if self.shock_mu_return >= 0 or self.shock_sector_return >= 0:
            raise ValueError("shock-return thresholds must be negative")
        if self.shock_cooldown_sessions < 1:
            raise ValueError("shock_cooldown_sessions must be positive")

        buy_weights = (
            self.buy_oversold_weight,
            self.buy_downside_weight,
            self.buy_reversal_weight,
            self.buy_sector_weight,
        )
        sell_weights = (
            self.sell_overbought_weight,
            self.sell_extension_weight,
            self.sell_sector_weight,
            self.sell_rollover_weight,
            self.sell_market_weight,
        )
        if any(weight < 0 for weight in (*buy_weights, *sell_weights)):
            raise ValueError("confidence weights must be non-negative")
        if abs(sum(buy_weights) - 1.0) > 1e-9:
            raise ValueError("MU buy-confidence weights must sum to 1")
        if abs(sum(sell_weights) - 1.0) > 1e-9:
            raise ValueError("MU-SOXS sell-confidence weights must sum to 1")

        for name, value in (
            ("account_risk_budget", self.account_risk_budget),
            ("mu_max_weight", self.mu_max_weight),
            ("soxs_max_weight", self.soxs_max_weight),
            ("mu_starter_fraction", self.mu_starter_fraction),
        ):
            if not 0 < value <= 1:
                raise ValueError(f"{name} must be in (0, 1]")
        if self.soxs_max_holding_sessions < 1:
            raise ValueError("soxs_max_holding_sessions must be positive")


@dataclass(frozen=True)
class Settings:
    schema_version: str
    lookback_years: int
    benchmark: str
    broad_market: str
    sector_proxies: tuple[str, ...]
    watchlist: tuple[str, ...]
    fred_series: dict[str, str]
    warmup_calendar_days: int
    holdout_sessions: int
    transaction_cost_bps: float
    annualization: int
    maximum_market_age_sessions: int
    minimum_history_sessions: int
    mu_soxs_strategy: MuSoxsStrategyConfig | None = None


def _mu_soxs_strategy(payload: dict[str, Any]) -> MuSoxsStrategyConfig | None:
    strategy = payload.get("strategies", {}).get("mu_soxs")
    if strategy is None:
        return None
    weights = strategy.get("weights", {})
    buy_weights = weights.get("buy", {})
    sell_weights = weights.get("sell", {})
    risk = strategy.get("risk_policy", {})
    development = strategy.get("development_window", {})
    return MuSoxsStrategyConfig(
        enabled=bool(strategy.get("enabled", True)),
        version=str(strategy.get("version", "mu-soxs-confidence-v2.1")),
        signal_symbol=str(strategy.get("signal_symbol", "MU")).upper(),
        inverse_symbol=str(strategy.get("inverse_symbol", "SOXS")).upper(),
        sector_proxy=str(strategy.get("sector_proxy", "SMH")).upper(),
        development_start=str(development.get("start", "2026-06-01")),
        development_end=str(development.get("end", "2026-08-31")),
        rsi_period=int(strategy.get("rsi_period", 6)),
        percentile_window=int(strategy.get("percentile_window", 252)),
        percentile_min_periods=int(strategy.get("percentile_min_periods", 60)),
        buy_threshold=float(strategy.get("buy_threshold", 65.0)),
        sell_threshold=float(strategy.get("sell_threshold", 70.0)),
        confidence_margin=float(strategy.get("confidence_margin", 15.0)),
        shock_mu_return=float(strategy.get("shock_mu_return", -0.08)),
        shock_sector_return=float(strategy.get("shock_sector_return", -0.05)),
        shock_cooldown_sessions=int(strategy.get("shock_cooldown_sessions", 5)),
        buy_oversold_weight=float(buy_weights.get("oversold", 0.45)),
        buy_downside_weight=float(buy_weights.get("downside_stretch", 0.25)),
        buy_reversal_weight=float(buy_weights.get("reversal", 0.20)),
        buy_sector_weight=float(buy_weights.get("sector_capitulation", 0.10)),
        sell_overbought_weight=float(sell_weights.get("recent_overbought", 0.30)),
        sell_extension_weight=float(sell_weights.get("stock_extension", 0.15)),
        sell_sector_weight=float(sell_weights.get("sector_weakness", 0.25)),
        sell_rollover_weight=float(sell_weights.get("rollover", 0.20)),
        sell_market_weight=float(sell_weights.get("market_risk", 0.10)),
        account_risk_budget=float(risk.get("account_risk_budget", 0.0075)),
        mu_max_weight=float(risk.get("mu_max_weight", 0.40)),
        soxs_max_weight=float(risk.get("soxs_max_weight", 0.20)),
        mu_starter_fraction=float(risk.get("mu_starter_fraction", 0.50)),
        soxs_max_holding_sessions=int(risk.get("soxs_max_holding_sessions", 5)),
    )


def load_settings(path: str | Path) -> Settings:
    source = Path(path)
    payload: dict[str, Any] = json.loads(source.read_text(encoding="utf-8"))
    backtest = payload["backtest"]
    quality = payload["quality"]
    return Settings(
        schema_version=str(payload["schema_version"]),
        lookback_years=int(payload["lookback_years"]),
        benchmark=str(payload["benchmark"]),
        broad_market=str(payload["broad_market"]),
        sector_proxies=tuple(payload["sector_proxies"]),
        watchlist=tuple(payload["watchlist"]),
        fred_series={str(k): str(v) for k, v in payload["fred_series"].items()},
        warmup_calendar_days=int(backtest["warmup_calendar_days"]),
        holdout_sessions=int(backtest["holdout_sessions"]),
        transaction_cost_bps=float(backtest["transaction_cost_bps"]),
        annualization=int(backtest["annualization"]),
        maximum_market_age_sessions=int(quality["maximum_market_age_sessions"]),
        minimum_history_sessions=int(quality["minimum_history_sessions"]),
        mu_soxs_strategy=_mu_soxs_strategy(payload),
    )
