from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, time as clock_time, timedelta, timezone
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from .backtest import (
    POSITION_POLICIES,
    RISK_BUCKET_EDGES,
    BacktestResult,
    run_full_and_holdout,
    run_sensitivity_experiments,
)
from .config import Settings, load_settings
from .data import FredCsvClient, YahooDailyClient
from .io import atomic_write_json, to_jsonable
from .model import latest_snapshot, score_market
from .strategies import (
    build_confidence_history,
    latest_confidence_payload,
    run_mu_soxs_backtest,
)
from .watchlist import score_watchlist


@dataclass(frozen=True)
class BuildArtifacts:
    snapshot_path: Path
    backtest_path: Path
    report_path: Path
    snapshot: dict[str, Any]
    backtest: dict[str, Any]


def repository_root() -> Path:
    return Path(__file__).resolve().parents[2]


def default_dates(
    years: int,
    end: str | date | None = None,
    *,
    now: datetime | None = None,
) -> tuple[date, date]:
    """Return a formal evaluation window ending on a completed U.S. session.

    An explicit ``end`` is always respected.  The default is conservative:
    before 18:00 New York time on a weekday, the current calendar day is not
    requested because Yahoo's daily bar may still be incomplete and cached as
    though it were final.
    """

    if end is not None:
        end_date = pd.Timestamp(end).date()
    else:
        current = now or datetime.now(ZoneInfo("America/New_York"))
        if current.tzinfo is None:
            current = current.replace(tzinfo=ZoneInfo("America/New_York"))
        else:
            current = current.astimezone(ZoneInfo("America/New_York"))
        end_date = current.date()
        if current.weekday() < 5 and current.time() < clock_time(18, 0):
            end_date -= timedelta(days=1)
    start_date = (pd.Timestamp(end_date) - pd.DateOffset(years=years)).date()
    return start_date, end_date


def _series_metadata(value: pd.Series | pd.DataFrame) -> dict[str, Any]:
    return dict(value.attrs.get("metadata", {}))


def collect_inputs(
    settings: Settings,
    *,
    start: date,
    end: date,
    cache_dir: str | Path,
    refresh: bool = False,
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame], dict[str, dict[str, Any]]]:
    yahoo = YahooDailyClient(cache_dir)
    fred = FredCsvClient(cache_dir)
    core_symbols = tuple(
        dict.fromkeys(
            (settings.benchmark, settings.broad_market, *settings.sector_proxies)
        )
    )
    strategy_symbols: tuple[str, ...] = ()
    if settings.mu_soxs_strategy is not None and settings.mu_soxs_strategy.enabled:
        strategy_symbols = (
            settings.mu_soxs_strategy.signal_symbol,
            settings.mu_soxs_strategy.inverse_symbol,
            settings.mu_soxs_strategy.sector_proxy,
        )
    all_symbols = tuple(
        dict.fromkeys((*core_symbols, *settings.watchlist, *strategy_symbols))
    )
    price_history: dict[str, pd.DataFrame] = {}
    source_metadata: dict[str, dict[str, Any]] = {}

    for symbol in all_symbols:
        history = yahoo.fetch(symbol, start, end, refresh=refresh)
        if history.empty:
            raise RuntimeError(f"no Yahoo observations returned for {symbol}")
        history = history.copy()
        history["adj_close"] = history["adj_close"].fillna(history["close"])
        price_history[symbol] = history
        source_metadata[f"YAHOO:{symbol}"] = _series_metadata(history)

    benchmark_history = price_history[settings.benchmark]
    sessions = benchmark_history.index
    market = pd.DataFrame(index=sessions)
    symbol_mapping = {
        settings.benchmark: "QQQ",
        settings.broad_market: "SPY",
        settings.sector_proxies[0]: "XLK",
        settings.sector_proxies[1]: "SMH",
    }
    for symbol, column in symbol_mapping.items():
        market[column] = price_history[symbol]["adj_close"].reindex(sessions)

    for canonical, series_id in settings.fred_series.items():
        values = fred.fetch(series_id, start, end, refresh=refresh)
        metadata = _series_metadata(values)
        aligned = values.copy()
        # NFCI observation dates are week-ending Fridays but the release is
        # normally the following Wednesday.  Apply the release lag before any
        # forward-fill so the backtest cannot see it early.
        if canonical == "NFCI":
            aligned.index = aligned.index + pd.Timedelta(days=5)
            metadata["alignment"] = "week_ending_friday_plus_5_calendar_days"
        # H.15 real yields are normally published just after the cash close.
        # A one-calendar-day delay is a conservative next-session convention.
        elif canonical == "REAL10Y":
            aligned.index = aligned.index + pd.Timedelta(days=1)
            metadata["alignment"] = "observation_plus_1_calendar_day"
        else:
            metadata["alignment"] = "observation_date"
        available = aligned.dropna()
        if not available.empty:
            metadata["model_available_through"] = available.index[-1].date().isoformat()
        # Keep release dates that land on weekends/holidays in the temporary
        # index, forward-fill through them, and only then select trading days.
        # Reindexing directly to sessions would silently discard such releases.
        alignment_index = aligned.index.union(sessions).sort_values()
        market[canonical] = aligned.reindex(alignment_index).ffill().reindex(sessions)
        source_metadata[f"FRED:{series_id}"] = metadata

    market = market.sort_index()
    return market, price_history, source_metadata


def _quality_summary(
    market: pd.DataFrame,
    metadata: dict[str, dict[str, Any]],
    *,
    requested_end: date,
    maximum_market_age_sessions: int,
    minimum_sessions: int,
) -> dict[str, Any]:
    enriched_metadata: dict[str, dict[str, Any]] = {}
    stale_sources: list[str] = []
    for key, source in metadata.items():
        details = dict(source)
        # Cache locations are implementation details and may contain a local
        # username or CI workspace path.  Keep them inside the data client,
        # never in the dashboard snapshot that can be published publicly.
        details.pop("cache_path", None)
        observed = pd.to_datetime(details.get("last_observation_date"), errors="coerce")
        if pd.notna(observed):
            observed_date = observed.date()
            age_sessions = len(
                pd.bdate_range(observed_date + timedelta(days=1), requested_end)
            )
            # Weekly NFCI is normally released with a lag; daily FRED series
            # also occasionally pause around holidays.  Price bars retain the
            # configured tighter limit.
            if key == "FRED:NFCI":
                allowed_age = max(maximum_market_age_sessions, 8)
            elif key.startswith("FRED:"):
                allowed_age = max(maximum_market_age_sessions, 5)
            else:
                allowed_age = maximum_market_age_sessions
            details["observation_age_sessions"] = age_sessions
            details["maximum_observation_age_sessions"] = allowed_age
            details["observation_stale"] = age_sessions > allowed_age
        else:
            details["observation_age_sessions"] = None
            details["observation_stale"] = True
        if details.get("stale") or details["observation_stale"]:
            stale_sources.append(key)
        enriched_metadata[key] = details

    stale_sources = sorted(stale_sources)
    missing_latest = sorted(
        column for column in market.columns if pd.isna(market[column].iloc[-1])
    )
    history_short = len(market) < minimum_sessions
    latest_market_date = market.index[-1].date()
    calendar_age_days = (requested_end - latest_market_date).days
    market_age_sessions = len(
        pd.bdate_range(latest_market_date + timedelta(days=1), requested_end)
    )
    status = "fresh"
    if (
        stale_sources
        or missing_latest
        or history_short
        or market_age_sessions > maximum_market_age_sessions
    ):
        status = "degraded"
    return {
        "status": status,
        "stale_sources": stale_sources,
        "missing_latest": missing_latest,
        "history_short": history_short,
        "calendar_age_days": calendar_age_days,
        "market_age_sessions": market_age_sessions,
        "maximum_market_age_sessions": maximum_market_age_sessions,
        "source_metadata": enriched_metadata,
    }


def _history_payload(scored: pd.DataFrame, limit: int = 260) -> list[dict[str, Any]]:
    columns = ["QQQ", "risk_score", "target_position", "regime"]
    points: list[dict[str, Any]] = []
    for index, row in scored.loc[:, columns].tail(limit).iterrows():
        points.append(
            {
                "date": index.date().isoformat(),
                "qqq": row["QQQ"],
                "risk_score": row["risk_score"],
                "target_position": row["target_position"],
                "regime": row["regime"],
            }
        )
    return to_jsonable(points)


def _equity_payload(result: BacktestResult) -> list[dict[str, Any]]:
    points: list[dict[str, Any]] = []
    audit_columns = (
        "signal_position",
        "applied_position",
        "pretrade_position",
        "position_change",
        "transaction_cost",
        "benchmark_transaction_cost",
        "qqq_return",
        "cash_return",
        "strategy_return",
        "benchmark_return",
    )
    for index, row in result.equity.iterrows():
        point = {
            "date": index.date().isoformat(),
            "strategy": row["strategy_equity"],
            "benchmark": row["benchmark_equity"],
            "position": row["applied_position"],
        }
        point.update({column: row[column] for column in audit_columns if column in row})
        points.append(point)
    return to_jsonable(points)


def _run_payload(result: BacktestResult) -> dict[str, Any]:
    metrics = result.metrics
    return {
        "start": metrics["start"],
        "end": metrics["end"],
        "strategy": {"metrics": metrics["strategy"]},
        "benchmark": {"metrics": metrics["benchmark"]},
        "comparison": metrics["comparison"],
    }


def _experiments_payload(
    experiments: dict[str, dict[str, BacktestResult]],
) -> dict[str, dict[str, Any]]:
    return {
        name: {
            "full": _run_payload(runs["full"]),
            "holdout": _run_payload(runs["holdout"]),
        }
        for name, runs in experiments.items()
    }


def _shifted_threshold_positions(
    risk_score: pd.Series,
    shift: float,
) -> pd.Series:
    """Apply the conservative policy after shifting every risk-bucket edge."""

    score = pd.to_numeric(risk_score, errors="coerce")
    edges = tuple(edge + shift for edge in RISK_BUCKET_EDGES)
    exposures = POSITION_POLICIES["conservative"]
    values = np.select(
        [score < edges[0], score < edges[1], score < edges[2], score < edges[3]],
        exposures[:4],
        default=exposures[4],
    ).astype(float)
    values[score.isna().to_numpy()] = np.nan
    return pd.Series(values, index=score.index, name="target_position")


def _validation_payload(
    holdout: BacktestResult,
    experiments: dict[str, dict[str, BacktestResult]],
) -> dict[str, Any]:
    strategy = holdout.metrics["strategy"]
    benchmark = holdout.metrics["benchmark"]
    strategy_cagr = strategy["cagr"]
    benchmark_cagr = benchmark["cagr"]
    cagr_retention = (
        strategy_cagr / benchmark_cagr
        if strategy_cagr is not None and benchmark_cagr is not None and benchmark_cagr > 0
        else None
    )
    drawdown_pass = (
        strategy["max_drawdown"] > benchmark["max_drawdown"]
        if strategy["max_drawdown"] is not None and benchmark["max_drawdown"] is not None
        else None
    )
    retention_pass = cagr_retention >= 0.80 if cagr_retention is not None else None
    sharpe_pass = (
        strategy["sharpe"] >= benchmark["sharpe"]
        if strategy["sharpe"] is not None and benchmark["sharpe"] is not None
        else None
    )
    simple = experiments["qqq_200dma"]["holdout"].metrics["strategy"]
    return {
        "holdout_gates": {
            "drawdown_better_than_qqq": drawdown_pass,
            "cagr_retention": cagr_retention,
            "cagr_retention_at_least_80pct": retention_pass,
            "sharpe_not_below_qqq": sharpe_pass,
            "all_passed": all(
                value is True for value in (drawdown_pass, retention_pass, sharpe_pass)
            ),
        },
        "qqq_200dma_comparison": {
            "default_cagr": strategy["cagr"],
            "simple_cagr": simple["cagr"],
            "default_sharpe": strategy["sharpe"],
            "simple_sharpe": simple["sharpe"],
            "default_max_drawdown": strategy["max_drawdown"],
            "simple_max_drawdown": simple["max_drawdown"],
            "default_dominates_simple_baseline": (
                strategy["cagr"] >= simple["cagr"]
                and strategy["sharpe"] >= simple["sharpe"]
                and strategy["max_drawdown"] >= simple["max_drawdown"]
            ),
        },
    }


def _percentage(value: float | None) -> str:
    return "n/a" if value is None else f"{value * 100:.2f}%"


def _number(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.2f}"


def render_backtest_report(payload: dict[str, Any]) -> str:
    lines = [
        "# U.S. Tech Monitor MVP Backtest",
        "",
        f"Generated: `{payload['generated_at_utc']}`",
        "",
        "The candidate policy uses fixed, non-fitted weights. A close-t signal executes at the "
        "next session's close and starts earning the following close-to-close return. Strategy "
        "and benchmark both pay the configured initial-entry cost; unallocated capital earns "
        "the 3-month Treasury proxy.",
        "",
    ]
    for label, key in (
        ("Full executable sample", "full"),
        ("Trailing evaluation slice", "holdout"),
    ):
        section = payload[key]
        strategy = section["strategy"]["metrics"]
        benchmark = section["benchmark"]["metrics"]
        lines.extend(
            [
                f"## {label}",
                "",
                f"Window: `{section['start']}` to `{section['end']}`",
                "",
                "| Metric | Thermometer policy | QQQ buy & hold |",
                "|---|---:|---:|",
                (
                    f"| Total return | {_percentage(strategy['total_return'])} | "
                    f"{_percentage(benchmark['total_return'])} |"
                ),
                f"| CAGR | {_percentage(strategy['cagr'])} | {_percentage(benchmark['cagr'])} |",
                (
                    f"| Annual volatility | {_percentage(strategy['volatility'])} | "
                    f"{_percentage(benchmark['volatility'])} |"
                ),
                f"| Sharpe | {_number(strategy['sharpe'])} | {_number(benchmark['sharpe'])} |",
                (
                    f"| Maximum drawdown | {_percentage(strategy['max_drawdown'])} | "
                    f"{_percentage(benchmark['max_drawdown'])} |"
                ),
                f"| Calmar | {_number(strategy['calmar'])} | {_number(benchmark['calmar'])} |",
                (
                    f"| One-way turnover | {_number(strategy['turnover'])} | "
                    f"{_number(benchmark['turnover'])} |"
                ),
                "",
            ]
        )
    lines.extend(
        [
            "## Pre-declared policy sensitivity",
            "",
            (
                "All variants are reported; the experiment does not choose a winner after "
                "seeing results. The dashboard policy remains `conservative`."
            ),
            "",
            (
                "| Policy | Full CAGR | Full max DD | Trailing CAGR | Trailing max DD | "
                "Trailing turnover |"
            ),
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for name, experiment in payload.get("experiments", {}).items():
        full_metrics = experiment["full"]["strategy"]["metrics"]
        holdout_metrics = experiment["holdout"]["strategy"]["metrics"]
        lines.append(
            f"| {name} | {_percentage(full_metrics['cagr'])} | "
            f"{_percentage(full_metrics['max_drawdown'])} | "
            f"{_percentage(holdout_metrics['cagr'])} | "
            f"{_percentage(holdout_metrics['max_drawdown'])} | "
            f"{_number(holdout_metrics['turnover'])} |"
        )
    gates = payload["validation"]["holdout_gates"]
    simple = payload["validation"]["qqq_200dma_comparison"]
    holdout_payload = payload["holdout"]
    holdout_strategy = holdout_payload["strategy"]["metrics"]
    holdout_benchmark = holdout_payload["benchmark"]["metrics"]
    drawdown_improvement = holdout_payload["comparison"]["max_drawdown_improvement"]
    drawdown_gate = "yes" if gates["drawdown_better_than_qqq"] else "no"
    retention_gate = "yes" if gates["cagr_retention_at_least_80pct"] else "no"
    sharpe_gate = "yes" if gates["sharpe_not_below_qqq"] else "no"
    lines.extend(
        [
            "",
            "## Trailing-slice validation gates",
            "",
            "| Pre-declared gate | Result | Pass |",
            "|---|---:|:---:|",
            (
                "| Maximum drawdown shallower than QQQ | "
                f"{_percentage(drawdown_improvement)} improvement | {drawdown_gate} |"
            ),
            (
                "| Retain at least 80% of QQQ CAGR | "
                f"{_percentage(gates['cagr_retention'])} retained | {retention_gate} |"
            ),
            (
                f"| Sharpe not below QQQ | {_number(holdout_strategy['sharpe'])} vs "
                f"{_number(holdout_benchmark['sharpe'])} | {sharpe_gate} |"
            ),
            "",
            f"Overall trailing-slice gate result: **{'PASS' if gates['all_passed'] else 'FAIL'}**.",
            "",
            "The default composite does not dominate the simple QQQ 200DMA baseline. "
            f"On the trailing slice, their CAGRs are {_percentage(simple['default_cagr'])} and "
            f"{_percentage(simple['simple_cagr'])}; maximum drawdowns are "
            f"{_percentage(simple['default_max_drawdown'])} and "
            f"{_percentage(simple['simple_max_drawdown'])}, respectively.",
            "",
            "## Transaction-cost stress",
            "",
            "| One-way cost | Full CAGR | Full max DD | Trailing CAGR | Trailing max DD |",
            "|---:|---:|---:|---:|---:|",
        ]
    )
    for label, experiment in payload.get("cost_sensitivity", {}).items():
        full_metrics = experiment["full"]["strategy"]["metrics"]
        holdout_metrics = experiment["holdout"]["strategy"]["metrics"]
        lines.append(
            f"| {label} | {_percentage(full_metrics['cagr'])} | "
            f"{_percentage(full_metrics['max_drawdown'])} | "
            f"{_percentage(holdout_metrics['cagr'])} | "
            f"{_percentage(holdout_metrics['max_drawdown'])} |"
        )
    lines.extend(
        [
            "",
            "## Risk-threshold stress",
            "",
            "All four position-bucket boundaries are shifted together; no variant is selected.",
            "",
            "| Boundary shift | Full CAGR | Full max DD | Trailing CAGR | Trailing max DD |",
            "|---:|---:|---:|---:|---:|",
        ]
    )
    for label, experiment in payload.get("threshold_sensitivity", {}).items():
        full_metrics = experiment["full"]["strategy"]["metrics"]
        holdout_metrics = experiment["holdout"]["strategy"]["metrics"]
        lines.append(
            f"| {label} | {_percentage(full_metrics['cagr'])} | "
            f"{_percentage(full_metrics['max_drawdown'])} | "
            f"{_percentage(holdout_metrics['cagr'])} | "
            f"{_percentage(holdout_metrics['max_drawdown'])} |"
        )
    lines.extend(
        [
            "",
            "## Assumptions and limitations",
            "",
            (
                f"- Transaction cost: {payload['assumptions']['transaction_cost_bps']:.1f} "
                "bps per one-way exposure change."
            ),
            f"- Annualization: {payload['assumptions']['annualization']} trading sessions.",
            f"- The {payload['assumptions']['warmup_calendar_days']} calendar days before the "
            "formal evaluation window are used only for indicator warm-up and are excluded "
            "from all performance metrics.",
            f"- The final {payload['assumptions']['holdout_sessions']} executable sessions are "
            "reported separately as a trailing evaluation slice.",
            (
                "- The ETF-level test avoids constituent survivorship bias, but FRED series use "
                "today's vintage; NFCI revisions can therefore create revision bias."
            ),
            (
                "- Yahoo Chart v8 is an unofficial MVP price source. Production use should "
                "switch to a licensed consolidated feed."
            ),
            (
                "- Results are a research demonstration, not evidence of future performance "
                "or an investment recommendation."
            ),
            "",
        ]
    )
    return "\n".join(lines)


def build_mvp(
    *,
    config_path: str | Path | None = None,
    start: str | date | None = None,
    end: str | date | None = None,
    cache_dir: str | Path | None = None,
    refresh: bool = False,
) -> BuildArtifacts:
    root = repository_root()
    settings = load_settings(config_path or root / "config" / "us_tech_mvp.json")
    default_start, default_end = default_dates(settings.lookback_years, end=end)
    start_date = pd.Timestamp(start).date() if start else default_start
    end_date = pd.Timestamp(end).date() if end else default_end
    if start_date >= end_date:
        raise ValueError("start must be before end")
    cache_path = Path(cache_dir or root / "data" / "cache")

    data_start = start_date - timedelta(days=settings.warmup_calendar_days)
    market, histories, metadata = collect_inputs(
        settings,
        start=data_start,
        end=end_date,
        cache_dir=cache_path,
        refresh=refresh,
    )
    scored = score_market(market)
    evaluation_scored = scored.loc[scored.index.date >= start_date].copy()
    if len(evaluation_scored.index) < 2:
        raise RuntimeError("the selected window needs at least two market sessions")
    # The first formal session supplies the starting close. Performance begins
    # with the return ending on the following session, keeping pre-window price
    # movement out of the five-year evaluation.
    performance_start = evaluation_scored.index[1]
    latest = latest_snapshot(evaluation_scored)
    if latest["risk_score"] is None:
        raise RuntimeError("the selected window did not produce a valid market score")
    quality = _quality_summary(
        market,
        metadata,
        requested_end=end_date,
        maximum_market_age_sessions=settings.maximum_market_age_sessions,
        minimum_sessions=settings.minimum_history_sessions,
    )
    latest_index = evaluation_scored["risk_score"].last_valid_index()
    if latest_index is None:
        raise RuntimeError("the selected window did not produce a valid market score")
    latest_row = evaluation_scored.loc[latest_index]
    generated_at = datetime.now(timezone.utc).isoformat()
    watchlist = score_watchlist(
        {symbol: histories[symbol] for symbol in settings.watchlist},
        histories[settings.benchmark],
        float(latest["risk_score"]),
    )
    strategies: dict[str, Any] = {}
    if settings.mu_soxs_strategy is not None and settings.mu_soxs_strategy.enabled:
        strategy = settings.mu_soxs_strategy
        confidence_history = build_confidence_history(
            histories[strategy.signal_symbol],
            histories[strategy.sector_proxy],
            histories[strategy.inverse_symbol],
            scored["risk_score"],
            strategy,
        )
        strategy_snapshot = latest_confidence_payload(
            confidence_history,
            as_of=latest_index,
            config=strategy,
        )
        strategy_snapshot["recent_backtest"] = run_mu_soxs_backtest(
            confidence_history,
            histories[strategy.signal_symbol],
            histories[strategy.sector_proxy],
            histories[strategy.inverse_symbol],
            start=strategy.development_start,
            end=latest_index,
            config=strategy,
            transaction_cost_bps=settings.transaction_cost_bps,
        )
        strategies["mu_soxs"] = strategy_snapshot
    snapshot = {
        "schema_version": settings.schema_version,
        "model_version": "us-tech-mvp-v1.1-mu-soxs-confidence-v2.1",
        "generated_at_utc": generated_at,
        "as_of_market_date": latest_index.date().isoformat(),
        "window": {
            "start": start_date.isoformat(),
            "end": end_date.isoformat(),
            "sessions": len(evaluation_scored),
            "data_start": data_start.isoformat(),
            "warmup_calendar_days": settings.warmup_calendar_days,
        },
        "quality": quality,
        "market": {
            "risk_score": latest["risk_score"],
            "regime": latest["regime"],
            "action": latest["action"],
            "target_position": latest["target_position"],
            "reasons": latest["reasons"],
            "components": {
                "macro_risk": latest["macro_risk"],
                "volatility_risk": latest["volatility_risk"],
                "trend_risk": latest["trend_risk"],
            },
            "latest": {
                column: latest_row[column]
                for column in ("QQQ", "SPY", "XLK", "SMH", "VXN", "REAL10Y", "NFCI")
            },
        },
        "history": _history_payload(evaluation_scored.loc[:latest_index]),
        "strategies": strategies,
        "watchlist": watchlist,
        "disclaimer": "Research-only risk monitor; it does not place orders or guarantee outcomes.",
    }

    runs = run_full_and_holdout(
        scored,
        start=performance_start,
        holdout_sessions=settings.holdout_sessions,
        annualization=settings.annualization,
        transaction_cost_bps=settings.transaction_cost_bps,
    )
    full = runs["full"]
    holdout = runs["holdout"]
    experiments = run_sensitivity_experiments(
        scored,
        start=performance_start,
        holdout_sessions=settings.holdout_sessions,
        annualization=settings.annualization,
        transaction_cost_bps=settings.transaction_cost_bps,
    )
    cost_experiments: dict[str, dict[str, BacktestResult]] = {}
    for cost in (0.0, settings.transaction_cost_bps, 25.0):
        label = f"{cost:g} bp"
        cost_experiments[label] = (
            runs
            if np.isclose(cost, settings.transaction_cost_bps)
            else run_full_and_holdout(
                scored,
                start=performance_start,
                holdout_sessions=settings.holdout_sessions,
                annualization=settings.annualization,
                transaction_cost_bps=cost,
            )
        )

    threshold_experiments: dict[str, dict[str, BacktestResult]] = {}
    for shift in (-5.0, 0.0, 5.0):
        label = f"{shift:+g} points"
        if np.isclose(shift, 0.0):
            threshold_experiments[label] = runs
            continue
        candidate = scored.copy()
        candidate["target_position"] = _shifted_threshold_positions(
            candidate["risk_score"], shift
        )
        threshold_experiments[label] = run_full_and_holdout(
            candidate,
            start=performance_start,
            holdout_sessions=settings.holdout_sessions,
            annualization=settings.annualization,
            transaction_cost_bps=settings.transaction_cost_bps,
        )
    validation = _validation_payload(holdout, experiments)
    backtest_payload = {
        "schema_version": settings.schema_version,
        "model_version": "us-tech-mvp-v1-next-close",
        "generated_at_utc": generated_at,
        "window": {
            "requested_start": start_date.isoformat(),
            "requested_end": end_date.isoformat(),
            "data_start": data_start.isoformat(),
            "performance_start": performance_start.date().isoformat(),
            "start": full.metrics["start"],
            "end": full.metrics["end"],
            "sessions": full.metrics["strategy"]["observations"],
            "holdout_start": holdout.metrics["start"],
        },
        "assumptions": {
            "signal_timing": "close_t_signal_executes_at_close_t_plus_1",
            "transaction_cost_bps": settings.transaction_cost_bps,
            "annualization": settings.annualization,
            "cash_proxy": settings.fred_series["CASH3M"],
            "weights_fitted_to_sample": False,
            "warmup_calendar_days": settings.warmup_calendar_days,
            "holdout_sessions": settings.holdout_sessions,
        },
        "full": _run_payload(full),
        "holdout": _run_payload(holdout),
        "experiments": _experiments_payload(experiments),
        "cost_sensitivity": _experiments_payload(cost_experiments),
        "threshold_sensitivity": _experiments_payload(threshold_experiments),
        "validation": validation,
        "equity_curve": _equity_payload(full),
        "holdout_equity_curve": _equity_payload(holdout),
        "limitations": [
            "FRED current-vintage macro history may include revisions.",
            "Yahoo Chart v8 is an unofficial, research-only MVP price source.",
            "The ETF-level backtest does not validate individual-stock entry and exit states.",
            "The trailing 252-session slice is not a preregistered untouched holdout.",
            "Five years include only a limited number of distinct market regimes.",
        ],
    }

    snapshot_path = root / "data" / "us_tech_snapshot.json"
    backtest_path = root / "data" / "backtest_results.json"
    report_path = root / "reports" / "backtest_latest.md"
    atomic_write_json(snapshot_path, snapshot)
    atomic_write_json(backtest_path, backtest_payload)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(render_backtest_report(backtest_payload), encoding="utf-8")
    return BuildArtifacts(
        snapshot_path=snapshot_path,
        backtest_path=backtest_path,
        report_path=report_path,
        snapshot=to_jsonable(snapshot),
        backtest=to_jsonable(backtest_payload),
    )
