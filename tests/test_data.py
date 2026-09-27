from __future__ import annotations

import json
from typing import Any

import pandas as pd
import pytest
import requests

from us_tech_monitor.data import (
    DataFetchError,
    fetch_fred_series,
    fetch_yahoo_daily,
)


class FakeResponse:
    def __init__(
        self,
        *,
        payload: dict[str, Any] | None = None,
        text: str = "",
        status_code: int = 200,
    ) -> None:
        self._payload = payload
        self.text = text
        self.status_code = status_code

    def json(self) -> dict[str, Any]:
        if self._payload is None:
            raise ValueError("no JSON payload")
        return self._payload

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise requests.HTTPError(f"HTTP {self.status_code}", response=self)


class FakeSession:
    def __init__(self, responses: list[FakeResponse | BaseException]) -> None:
        self.responses = list(responses)
        self.calls: list[dict[str, Any]] = []

    def get(self, url: str, **kwargs: Any) -> FakeResponse:
        self.calls.append({"url": url, **kwargs})
        if not self.responses:
            raise AssertionError("unexpected network request")
        response = self.responses.pop(0)
        if isinstance(response, BaseException):
            raise response
        return response


def _yahoo_payload() -> dict[str, Any]:
    return {
        "chart": {
            "error": None,
            "result": [
                {
                    "timestamp": [1609860600, 1609947000],
                    "indicators": {
                        "quote": [
                            {
                                "open": [100.0, 101.0],
                                "high": [102.0, 103.0],
                                "low": [99.0, 100.0],
                                "close": [101.0, 102.0],
                                "volume": [1_000_000, 1_200_000],
                            }
                        ],
                        "adjclose": [{"adjclose": [100.5, 101.5]}],
                    },
                }
            ],
        }
    }


def test_yahoo_parses_daily_bars_and_writes_raw_json_cache(tmp_path) -> None:
    session = FakeSession([FakeResponse(payload=_yahoo_payload())])
    frame = fetch_yahoo_daily(
        "qqq",
        "2021-01-05",
        "2021-01-06",
        cache_dir=tmp_path,
        session=session,  # type: ignore[arg-type]
        max_retries=0,
    )

    assert list(frame.columns) == ["open", "high", "low", "close", "adj_close", "volume"]
    assert list(frame.index) == [pd.Timestamp("2021-01-05"), pd.Timestamp("2021-01-06")]
    assert frame.loc["2021-01-06", "adj_close"] == pytest.approx(101.5)
    metadata = frame.attrs["metadata"]
    assert metadata["source"] == "network"
    assert metadata["stale"] is False
    assert metadata["first_observation_date"] == "2021-01-05"
    assert metadata["last_observation_date"] == "2021-01-06"
    cache_path = tmp_path / "yahoo" / "QQQ" / "2021-01-05_2021-01-06.json"
    assert json.loads(cache_path.read_text(encoding="utf-8")) == _yahoo_payload()
    assert session.calls[0]["headers"]["User-Agent"].startswith("us-tech-monitor/")


def test_yahoo_cache_hit_avoids_network_and_refresh_failure_is_marked_stale(tmp_path) -> None:
    warm = FakeSession([FakeResponse(payload=_yahoo_payload())])
    fetch_yahoo_daily(
        "QQQ",
        "2021-01-05",
        "2021-01-06",
        cache_dir=tmp_path,
        session=warm,  # type: ignore[arg-type]
        max_retries=0,
    )

    no_network = FakeSession([])
    cached = fetch_yahoo_daily(
        "QQQ",
        "2021-01-05",
        "2021-01-06",
        cache_dir=tmp_path,
        session=no_network,  # type: ignore[arg-type]
        max_retries=0,
    )
    assert not no_network.calls
    assert cached.attrs["metadata"]["source"] == "cache"
    assert cached.attrs["metadata"]["cache_hit"] is True
    assert cached.attrs["metadata"]["stale"] is False
    assert cached.attrs["metadata"]["first_observation_date"] == "2021-01-05"
    assert cached.attrs["metadata"]["last_observation_date"] == "2021-01-06"

    failing = FakeSession([requests.ConnectionError("offline")])
    stale = fetch_yahoo_daily(
        "QQQ",
        "2021-01-05",
        "2021-01-06",
        cache_dir=tmp_path,
        refresh=True,
        session=failing,  # type: ignore[arg-type]
        max_retries=0,
    )
    assert stale.attrs["metadata"]["source"] == "cache_fallback"
    assert stale.attrs["metadata"]["stale"] is True
    assert "offline" in stale.attrs["metadata"]["error"]


def test_yahoo_offline_fallback_uses_prior_overlapping_cache_and_skips_damage(
    tmp_path,
) -> None:
    warm = FakeSession([FakeResponse(payload=_yahoo_payload())])
    fetch_yahoo_daily(
        "QQQ",
        "2021-01-05",
        "2021-01-07",
        cache_dir=tmp_path,
        session=warm,  # type: ignore[arg-type]
        max_retries=0,
    )
    series_dir = tmp_path / "yahoo" / "QQQ"
    prior_cache = series_dir / "2021-01-05_2021-01-07.json"
    (series_dir / "2021-01-05_2021-01-08.json").write_text(
        "not json", encoding="utf-8"
    )
    (series_dir / "2021-01-06_2021-01-08.meta.json").write_text(
        json.dumps(_yahoo_payload()), encoding="utf-8"
    )

    failing = FakeSession([requests.ConnectionError("offline")])
    stale = fetch_yahoo_daily(
        "QQQ",
        "2021-01-06",
        "2021-01-08",
        cache_dir=tmp_path,
        session=failing,  # type: ignore[arg-type]
        max_retries=0,
    )

    assert list(stale.index) == [pd.Timestamp("2021-01-06")]
    metadata = stale.attrs["metadata"]
    assert metadata["source"] == "cache_fallback"
    assert metadata["stale"] is True
    assert metadata["cache_path"] == str(prior_cache.resolve())
    assert metadata["first_observation_date"] == "2021-01-06"
    assert metadata["last_observation_date"] == "2021-01-06"
    assert "offline" in metadata["error"]


def test_yahoo_failure_without_cache_is_explicit(tmp_path) -> None:
    session = FakeSession([requests.Timeout("timed out")])
    with pytest.raises(DataFetchError, match="Yahoo fetch failed"):
        fetch_yahoo_daily(
            "QQQ",
            "2021-01-05",
            "2021-01-06",
            cache_dir=tmp_path,
            session=session,  # type: ignore[arg-type]
            max_retries=0,
        )


def test_fred_parses_missing_values_and_caches_raw_csv(tmp_path) -> None:
    csv_text = (
        "observation_date,DFII10\n"
        "2024-01-02,1.75\n"
        "2024-01-03,.\n"
        "2024-01-04,1.80\n"
        "2024-01-05,.\n"
    )
    session = FakeSession([FakeResponse(text=csv_text)])
    series = fetch_fred_series(
        "DFII10",
        "2024-01-02",
        "2024-01-05",
        cache_dir=tmp_path,
        session=session,  # type: ignore[arg-type]
        max_retries=0,
    )

    assert series.name == "DFII10"
    assert series.loc["2024-01-02"] == pytest.approx(1.75)
    assert pd.isna(series.loc["2024-01-03"])
    metadata = series.attrs["metadata"]
    assert metadata["source"] == "network"
    assert metadata["first_observation_date"] == "2024-01-02"
    assert metadata["last_observation_date"] == "2024-01-04"
    cache_path = tmp_path / "fred" / "DFII10" / "2024-01-02_2024-01-05.csv"
    assert cache_path.read_text(encoding="utf-8") == csv_text


def test_fred_offline_fallback_uses_prior_overlapping_cache_and_crops(
    tmp_path,
) -> None:
    csv_text = (
        "DATE,NFCI\n"
        "2024-01-01,-0.55\n"
        "2024-01-02,-0.50\n"
        "2024-01-03,.\n"
        "2024-01-04,-0.45\n"
    )
    warm = FakeSession([FakeResponse(text=csv_text)])
    fetch_fred_series(
        "NFCI",
        "2024-01-01",
        "2024-01-04",
        cache_dir=tmp_path,
        session=warm,  # type: ignore[arg-type]
        max_retries=0,
    )
    series_dir = tmp_path / "fred" / "NFCI"
    prior_cache = series_dir / "2024-01-01_2024-01-04.csv"
    (series_dir / "2024-01-01_2024-01-05.csv").write_text(
        "this,is,not,the,requested,series\n", encoding="utf-8"
    )
    (series_dir / "2024-01-02_2024-01-05.meta.json").write_text(
        csv_text, encoding="utf-8"
    )

    failing = FakeSession([requests.ConnectionError("offline")])
    stale = fetch_fred_series(
        "NFCI",
        "2024-01-02",
        "2024-01-05",
        cache_dir=tmp_path,
        session=failing,  # type: ignore[arg-type]
        max_retries=0,
    )

    assert list(stale.index) == [
        pd.Timestamp("2024-01-02"),
        pd.Timestamp("2024-01-03"),
        pd.Timestamp("2024-01-04"),
    ]
    assert stale.iloc[0] == pytest.approx(-0.5)
    assert pd.isna(stale.iloc[1])
    assert stale.iloc[2] == pytest.approx(-0.45)
    metadata = stale.attrs["metadata"]
    assert metadata["source"] == "cache_fallback"
    assert metadata["cache_hit"] is True
    assert metadata["stale"] is True
    assert metadata["cache_path"] == str(prior_cache.resolve())
    assert metadata["first_observation_date"] == "2024-01-02"
    assert metadata["last_observation_date"] == "2024-01-04"
    assert "offline" in metadata["error"]


def test_fred_cache_fallback_is_marked_stale(tmp_path) -> None:
    csv_text = "DATE,NFCI\n2024-01-05,-0.50\n2024-01-12,-0.45\n"
    warm = FakeSession([FakeResponse(text=csv_text)])
    fetch_fred_series(
        "NFCI",
        "2024-01-05",
        "2024-01-12",
        cache_dir=tmp_path,
        session=warm,  # type: ignore[arg-type]
        max_retries=0,
    )

    failing = FakeSession([requests.ConnectionError("offline")])
    stale = fetch_fred_series(
        "NFCI",
        "2024-01-05",
        "2024-01-12",
        cache_dir=tmp_path,
        refresh=True,
        session=failing,  # type: ignore[arg-type]
        max_retries=0,
    )

    assert stale.to_list() == [-0.5, -0.45]
    assert stale.attrs["metadata"]["source"] == "cache_fallback"
    assert stale.attrs["metadata"]["cache_hit"] is True
    assert stale.attrs["metadata"]["stale"] is True
    assert "offline" in stale.attrs["metadata"]["error"]


def test_retryable_http_status_is_retried(tmp_path) -> None:
    session = FakeSession(
        [FakeResponse(status_code=503), FakeResponse(payload=_yahoo_payload())]
    )
    result = fetch_yahoo_daily(
        "QQQ",
        "2021-01-05",
        "2021-01-06",
        cache_dir=tmp_path,
        session=session,  # type: ignore[arg-type]
        max_retries=1,
    )
    assert len(session.calls) == 2
    assert result.attrs["metadata"]["source"] == "network"
