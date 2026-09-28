from __future__ import annotations

from datetime import date, datetime
from zoneinfo import ZoneInfo

import pandas as pd

from us_tech_monitor.config import Settings
from us_tech_monitor.pipeline import _quality_summary, collect_inputs, default_dates


def test_default_dates_avoids_incomplete_weekday_bar() -> None:
    new_york = ZoneInfo("America/New_York")
    before_close = datetime(2026, 9, 28, 12, 0, tzinfo=new_york)
    after_cutoff = datetime(2026, 9, 28, 19, 0, tzinfo=new_york)

    _, safe_end = default_dates(5, now=before_close)
    _, completed_end = default_dates(5, now=after_cutoff)

    assert safe_end == date(2026, 9, 27)
    assert completed_end == date(2026, 9, 28)


def test_explicit_end_is_not_rewritten() -> None:
    _, end = default_dates(5, end="2026-09-28")
    assert end == date(2026, 9, 28)


def test_quality_uses_observation_dates_and_configured_market_age() -> None:
    index = pd.DatetimeIndex(["2026-09-18", "2026-09-25"])
    market = pd.DataFrame({"QQQ": [100.0, 101.0]}, index=index)
    metadata = {
        "YAHOO:QQQ": {
            "stale": False,
            "last_observation_date": "2026-09-18",
            "cache_path": "C:/Users/example/private-cache/qqq.json",
            "error": "request failed at C:/Users/example/private-cache?token=secret",
        },
        "FRED:NFCI": {
            "stale": False,
            "last_observation_date": "2026-09-18",
        },
    }

    quality = _quality_summary(
        market,
        metadata,
        requested_end=date(2026, 9, 27),
        maximum_market_age_sessions=3,
        minimum_sessions=1,
    )

    assert quality["status"] == "degraded"
    assert quality["market_age_sessions"] == 0
    assert "YAHOO:QQQ" in quality["stale_sources"]
    assert "FRED:NFCI" not in quality["stale_sources"]
    assert quality["source_metadata"]["YAHOO:QQQ"]["observation_stale"] is True
    assert quality["source_metadata"]["FRED:NFCI"]["observation_stale"] is False
    assert "cache_path" not in quality["source_metadata"]["YAHOO:QQQ"]
    assert "error" not in quality["source_metadata"]["YAHOO:QQQ"]
    assert (
        quality["source_metadata"]["YAHOO:QQQ"]["error_code"]
        == "upstream_refresh_failed_using_cache"
    )


def test_release_lag_on_weekend_reaches_next_market_session(monkeypatch, tmp_path) -> None:
    sessions = pd.DatetimeIndex(["2026-09-25", "2026-09-28"], name="date")

    class FakeYahoo:
        def __init__(self, _cache_dir) -> None:
            pass

        def fetch(self, _symbol, _start, _end, *, refresh=False):
            del refresh
            frame = pd.DataFrame(
                {
                    "open": [100.0, 101.0],
                    "high": [101.0, 102.0],
                    "low": [99.0, 100.0],
                    "close": [100.0, 101.0],
                    "adj_close": [100.0, 101.0],
                    "volume": [1_000.0, 1_100.0],
                },
                index=sessions,
            )
            frame.attrs["metadata"] = {}
            return frame

    class FakeFred:
        def __init__(self, _cache_dir) -> None:
            pass

        def fetch(self, series_id, _start, _end, *, refresh=False):
            del refresh
            value = {
                "DFII10": 1.5,
                "NFCI": -0.5,
                "VXNCLS": 20.0,
                "DGS3MO": 4.0,
            }[series_id]
            series = pd.Series(
                [value], index=pd.DatetimeIndex(["2026-09-25"]), name=series_id
            )
            series.attrs["metadata"] = {}
            return series

    monkeypatch.setattr("us_tech_monitor.pipeline.YahooDailyClient", FakeYahoo)
    monkeypatch.setattr("us_tech_monitor.pipeline.FredCsvClient", FakeFred)
    settings = Settings(
        schema_version="1.0",
        lookback_years=5,
        benchmark="QQQ",
        broad_market="SPY",
        sector_proxies=("XLK", "SMH"),
        watchlist=(),
        fred_series={
            "REAL10Y": "DFII10",
            "NFCI": "NFCI",
            "VXN": "VXNCLS",
            "CASH3M": "DGS3MO",
        },
        warmup_calendar_days=450,
        holdout_sessions=252,
        transaction_cost_bps=10.0,
        annualization=252,
        maximum_market_age_sessions=3,
        minimum_history_sessions=1,
    )

    market, _, _ = collect_inputs(
        settings,
        start=date(2026, 9, 25),
        end=date(2026, 9, 28),
        cache_dir=tmp_path,
    )

    assert pd.isna(market.loc["2026-09-25", "REAL10Y"])
    assert market.loc["2026-09-28", "REAL10Y"] == 1.5
