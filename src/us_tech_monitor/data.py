from __future__ import annotations

import json
import os
import re
import tempfile
import time
from datetime import date, datetime, timedelta, timezone
from io import StringIO
from pathlib import Path
from typing import Any
from urllib.parse import quote

import pandas as pd
import requests


DEFAULT_USER_AGENT = "us-tech-monitor/0.1 (+https://github.com/Slimold/share-monitor)"
YAHOO_CHART_URL = "https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"
FRED_CSV_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv"
_RETRYABLE_STATUS_CODES = {429, 500, 502, 503, 504}
_OHLCV_COLUMNS = ["open", "high", "low", "close", "adj_close", "volume"]


class DataFetchError(RuntimeError):
    """Raised when a provider request fails and no usable cache is available."""


def _normalise_date(value: str | date | datetime | pd.Timestamp) -> date:
    timestamp = pd.Timestamp(value)
    if pd.isna(timestamp):
        raise ValueError("date must not be missing")
    return timestamp.date()


def _date_range(
    start: str | date | datetime | pd.Timestamp,
    end: str | date | datetime | pd.Timestamp,
) -> tuple[date, date]:
    start_date = _normalise_date(start)
    end_date = _normalise_date(end)
    if start_date > end_date:
        raise ValueError("start must be on or before end")
    return start_date, end_date


def _safe_component(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("._") or "series"


def _atomic_write_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as handle:
            handle.write(value)
        os.replace(temporary_name, path)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def _read_cache_metadata(path: Path) -> dict[str, Any]:
    try:
        metadata = json.loads(path.read_text(encoding="utf-8"))
        return metadata if isinstance(metadata, dict) else {}
    except (FileNotFoundError, json.JSONDecodeError, OSError, UnicodeError):
        return {}


def _write_cache_metadata(path: Path, *, fetched_at: str, source_url: str) -> None:
    _atomic_write_text(
        path,
        json.dumps(
            {"fetched_at": fetched_at, "source_url": source_url},
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
    )


def _attach_metadata(
    value: pd.DataFrame | pd.Series,
    *,
    provider: str,
    source: str,
    source_url: str,
    cache_path: Path,
    cache_hit: bool,
    stale: bool,
    fetched_at: str | None,
    error: str | None = None,
) -> pd.DataFrame | pd.Series:
    if isinstance(value, pd.Series):
        valid_rows = value.notna()
    else:
        valid_rows = value.notna().any(axis=1)
    valid_index = pd.DatetimeIndex(
        pd.to_datetime(value.index[valid_rows], errors="coerce")
    ).dropna()
    first_observation_date = (
        valid_index.min().date().isoformat() if not valid_index.empty else None
    )
    last_observation_date = (
        valid_index.max().date().isoformat() if not valid_index.empty else None
    )

    value.attrs["metadata"] = {
        "provider": provider,
        "source": source,
        "source_url": source_url,
        "cache_path": str(cache_path.resolve()),
        "cache_hit": cache_hit,
        "stale": stale,
        "fetched_at": fetched_at,
        "error": error,
        "first_observation_date": first_observation_date,
        "last_observation_date": last_observation_date,
    }
    return value


def _fallback_cache_candidates(
    exact_path: Path, *, start_date: date, end_date: date
) -> list[Path]:
    """Return recent, substantially overlapping data caches from one series directory."""

    requested_days = (end_date - start_date).days + 1
    filename_pattern = re.compile(
        rf"^(\d{{4}}-\d{{2}}-\d{{2}})_(\d{{4}}-\d{{2}}-\d{{2}})"
        rf"{re.escape(exact_path.suffix)}$"
    )
    candidates: list[tuple[date, int, date, str, Path]] = []
    try:
        directory_entries = list(exact_path.parent.iterdir())
    except OSError:
        return []

    for candidate in directory_entries:
        if candidate == exact_path or candidate.is_symlink() or not candidate.is_file():
            continue
        match = filename_pattern.fullmatch(candidate.name)
        if match is None:
            continue
        try:
            candidate_start = date.fromisoformat(match.group(1))
            candidate_end = date.fromisoformat(match.group(2))
        except ValueError:
            continue
        if candidate_start > candidate_end:
            continue

        overlap_start = max(start_date, candidate_start)
        overlap_end = min(end_date, candidate_end)
        overlap_days = max(0, (overlap_end - overlap_start).days + 1)
        if overlap_days * 2 < requested_days:
            continue
        try:
            modified_at = candidate.stat().st_mtime_ns
        except OSError:
            modified_at = 0
        candidates.append(
            (candidate_end, modified_at, candidate_start, candidate.name, candidate)
        )

    candidates.sort(reverse=True)
    return [candidate[-1] for candidate in candidates]


def _has_valid_observations(value: pd.DataFrame | pd.Series) -> bool:
    if isinstance(value, pd.Series):
        return bool(value.notna().any())
    return bool(value.notna().any(axis=1).any())


def _request(
    session: requests.Session,
    url: str,
    *,
    params: dict[str, Any],
    timeout: float,
    max_retries: int,
    user_agent: str,
) -> requests.Response:
    if max_retries < 0:
        raise ValueError("max_retries must be non-negative")

    last_error: BaseException | None = None
    for attempt in range(max_retries + 1):
        try:
            response = session.get(
                url,
                params=params,
                headers={"User-Agent": user_agent, "Accept": "*/*"},
                timeout=timeout,
            )
            if response.status_code in _RETRYABLE_STATUS_CODES:
                raise requests.HTTPError(
                    f"retryable HTTP status {response.status_code}", response=response
                )
            response.raise_for_status()
            return response
        except requests.RequestException as exc:
            last_error = exc
            if attempt >= max_retries:
                break
            time.sleep(min(0.25 * (2**attempt), 2.0))

    raise DataFetchError(f"request failed after {max_retries + 1} attempt(s): {last_error}")


def _parse_yahoo_payload(
    payload: dict[str, Any], *, start_date: date, end_date: date
) -> pd.DataFrame:
    try:
        chart = payload["chart"]
        if chart.get("error"):
            raise DataFetchError(f"Yahoo returned an error: {chart['error']}")
        result = chart["result"][0]
        timestamps = result["timestamp"]
        quote_values = result["indicators"]["quote"][0]
    except (KeyError, IndexError, TypeError) as exc:
        raise DataFetchError("Yahoo response did not contain chart observations") from exc

    if not timestamps:
        empty = pd.DataFrame(columns=_OHLCV_COLUMNS, index=pd.DatetimeIndex([], name="date"))
        return empty.astype(float)

    index = (
        pd.to_datetime(timestamps, unit="s", utc=True)
        .tz_convert("America/New_York")
        .normalize()
        .tz_localize(None)
    )
    adjusted_blocks = result.get("indicators", {}).get("adjclose", [])
    adjusted_values = adjusted_blocks[0].get("adjclose", []) if adjusted_blocks else []
    if len(adjusted_values) != len(timestamps):
        adjusted_values = [None] * len(timestamps)

    def values_for(name: str) -> list[Any]:
        values = quote_values.get(name, [])
        return values if len(values) == len(timestamps) else [None] * len(timestamps)

    frame = pd.DataFrame(
        {
            "open": values_for("open"),
            "high": values_for("high"),
            "low": values_for("low"),
            "close": values_for("close"),
            "adj_close": adjusted_values,
            "volume": values_for("volume"),
        },
        index=index,
    )
    frame.index.name = "date"
    frame = frame.apply(pd.to_numeric, errors="coerce")
    frame = frame.loc[
        (frame.index.date >= start_date) & (frame.index.date <= end_date), _OHLCV_COLUMNS
    ]
    return frame[~frame.index.duplicated(keep="last")].sort_index()


def _parse_fred_csv(
    csv_text: str, *, series_id: str, start_date: date, end_date: date
) -> pd.Series:
    try:
        frame = pd.read_csv(StringIO(csv_text))
    except Exception as exc:
        raise DataFetchError("FRED response was not valid CSV") from exc

    date_column = "observation_date" if "observation_date" in frame.columns else "DATE"
    if date_column not in frame.columns or series_id not in frame.columns:
        raise DataFetchError(
            f"FRED response must contain {date_column!r} and {series_id!r} columns"
        )

    index = pd.to_datetime(frame[date_column], errors="coerce")
    values = pd.to_numeric(frame[series_id].replace(".", pd.NA), errors="coerce")
    series = pd.Series(values.to_numpy(), index=index, name=series_id, dtype="float64")
    series.index.name = "date"
    series = series.loc[
        series.index.notna()
        & (series.index.date >= start_date)
        & (series.index.date <= end_date)
    ]
    return series[~series.index.duplicated(keep="last")].sort_index()


class YahooDailyClient:
    """Download and cache unadjusted/adjusted daily bars from Yahoo chart v8."""

    def __init__(
        self,
        cache_dir: str | Path = "data/cache",
        *,
        session: requests.Session | None = None,
        timeout: float = 20.0,
        max_retries: int = 3,
        user_agent: str = DEFAULT_USER_AGENT,
    ) -> None:
        self.cache_dir = Path(cache_dir)
        self.session = session or requests.Session()
        self.timeout = timeout
        self.max_retries = max_retries
        self.user_agent = user_agent

    def fetch(
        self,
        symbol: str,
        start: str | date | datetime | pd.Timestamp,
        end: str | date | datetime | pd.Timestamp,
        *,
        refresh: bool = False,
    ) -> pd.DataFrame:
        start_date, end_date = _date_range(start, end)
        symbol = symbol.strip().upper()
        if not symbol:
            raise ValueError("symbol must not be empty")

        cache_path = (
            self.cache_dir
            / "yahoo"
            / _safe_component(symbol)
            / f"{start_date.isoformat()}_{end_date.isoformat()}.json"
        )
        metadata_path = cache_path.with_suffix(".meta.json")
        source_url = YAHOO_CHART_URL.format(symbol=quote(symbol, safe=""))
        cache_error: BaseException | None = None

        if cache_path.exists() and not refresh:
            try:
                payload = json.loads(cache_path.read_text(encoding="utf-8"))
                frame = _parse_yahoo_payload(
                    payload, start_date=start_date, end_date=end_date
                )
                cached_meta = _read_cache_metadata(metadata_path)
                return _attach_metadata(
                    frame,
                    provider="yahoo_chart_v8",
                    source="cache",
                    source_url=str(cached_meta.get("source_url", source_url)),
                    cache_path=cache_path,
                    cache_hit=True,
                    stale=False,
                    fetched_at=cached_meta.get("fetched_at"),
                )
            except (OSError, UnicodeError, json.JSONDecodeError, DataFetchError) as exc:
                cache_error = exc

        period1 = int(datetime.combine(start_date, datetime.min.time(), timezone.utc).timestamp())
        period2 = int(
            datetime.combine(
                end_date + timedelta(days=1), datetime.min.time(), timezone.utc
            ).timestamp()
        )
        try:
            response = _request(
                self.session,
                source_url,
                params={
                    "period1": period1,
                    "period2": period2,
                    "interval": "1d",
                    "events": "div,splits,capitalGains",
                    "includeAdjustedClose": "true",
                },
                timeout=self.timeout,
                max_retries=self.max_retries,
                user_agent=self.user_agent,
            )
            payload = response.json()
            frame = _parse_yahoo_payload(payload, start_date=start_date, end_date=end_date)
            fetched_at = datetime.now(timezone.utc).isoformat()
            _atomic_write_text(cache_path, json.dumps(payload, ensure_ascii=False))
            _write_cache_metadata(
                metadata_path, fetched_at=fetched_at, source_url=source_url
            )
            return _attach_metadata(
                frame,
                provider="yahoo_chart_v8",
                source="network",
                source_url=source_url,
                cache_path=cache_path,
                cache_hit=False,
                stale=False,
                fetched_at=fetched_at,
            )
        except (requests.RequestException, ValueError, DataFetchError) as exc:
            fallback_paths = [cache_path] if cache_path.exists() else []
            fallback_paths.extend(
                _fallback_cache_candidates(
                    cache_path, start_date=start_date, end_date=end_date
                )
            )
            for fallback_path in fallback_paths:
                try:
                    payload = json.loads(fallback_path.read_text(encoding="utf-8"))
                    frame = _parse_yahoo_payload(
                        payload, start_date=start_date, end_date=end_date
                    )
                    if fallback_path != cache_path and not _has_valid_observations(frame):
                        raise DataFetchError(
                            f"fallback cache {fallback_path.name!r} has no valid observations"
                        )
                    fallback_metadata_path = fallback_path.with_suffix(".meta.json")
                    cached_meta = _read_cache_metadata(fallback_metadata_path)
                    return _attach_metadata(
                        frame,
                        provider="yahoo_chart_v8",
                        source="cache_fallback",
                        source_url=str(cached_meta.get("source_url", source_url)),
                        cache_path=fallback_path,
                        cache_hit=True,
                        stale=True,
                        fetched_at=cached_meta.get("fetched_at"),
                        error=str(exc),
                    )
                except (
                    OSError,
                    UnicodeError,
                    json.JSONDecodeError,
                    DataFetchError,
                ) as fallback_exc:
                    cache_error = fallback_exc
            detail = f"Yahoo fetch failed for {symbol}: {exc}"
            if cache_error is not None:
                detail += f"; cached copy was unusable: {cache_error}"
            raise DataFetchError(detail) from exc


class FredCsvClient:
    """Download and cache public FRED series through the fredgraph CSV endpoint."""

    def __init__(
        self,
        cache_dir: str | Path = "data/cache",
        *,
        session: requests.Session | None = None,
        timeout: float = 20.0,
        max_retries: int = 3,
        user_agent: str = DEFAULT_USER_AGENT,
    ) -> None:
        self.cache_dir = Path(cache_dir)
        self.session = session or requests.Session()
        self.timeout = timeout
        self.max_retries = max_retries
        self.user_agent = user_agent

    def fetch(
        self,
        series_id: str,
        start: str | date | datetime | pd.Timestamp,
        end: str | date | datetime | pd.Timestamp,
        *,
        refresh: bool = False,
    ) -> pd.Series:
        start_date, end_date = _date_range(start, end)
        series_id = series_id.strip().upper()
        if not series_id:
            raise ValueError("series_id must not be empty")

        cache_path = (
            self.cache_dir
            / "fred"
            / _safe_component(series_id)
            / f"{start_date.isoformat()}_{end_date.isoformat()}.csv"
        )
        metadata_path = cache_path.with_suffix(".meta.json")
        cache_error: BaseException | None = None

        if cache_path.exists() and not refresh:
            try:
                csv_text = cache_path.read_text(encoding="utf-8")
                series = _parse_fred_csv(
                    csv_text,
                    series_id=series_id,
                    start_date=start_date,
                    end_date=end_date,
                )
                cached_meta = _read_cache_metadata(metadata_path)
                return _attach_metadata(
                    series,
                    provider="fred_fredgraph_csv",
                    source="cache",
                    source_url=str(cached_meta.get("source_url", FRED_CSV_URL)),
                    cache_path=cache_path,
                    cache_hit=True,
                    stale=False,
                    fetched_at=cached_meta.get("fetched_at"),
                )
            except (OSError, UnicodeError, DataFetchError) as exc:
                cache_error = exc

        try:
            response = _request(
                self.session,
                FRED_CSV_URL,
                params={
                    "id": series_id,
                    "cosd": start_date.isoformat(),
                    "coed": end_date.isoformat(),
                },
                timeout=self.timeout,
                max_retries=self.max_retries,
                user_agent=self.user_agent,
            )
            csv_text = response.text
            series = _parse_fred_csv(
                csv_text,
                series_id=series_id,
                start_date=start_date,
                end_date=end_date,
            )
            fetched_at = datetime.now(timezone.utc).isoformat()
            _atomic_write_text(cache_path, csv_text)
            _write_cache_metadata(
                metadata_path, fetched_at=fetched_at, source_url=FRED_CSV_URL
            )
            return _attach_metadata(
                series,
                provider="fred_fredgraph_csv",
                source="network",
                source_url=FRED_CSV_URL,
                cache_path=cache_path,
                cache_hit=False,
                stale=False,
                fetched_at=fetched_at,
            )
        except (requests.RequestException, ValueError, DataFetchError) as exc:
            fallback_paths = [cache_path] if cache_path.exists() else []
            fallback_paths.extend(
                _fallback_cache_candidates(
                    cache_path, start_date=start_date, end_date=end_date
                )
            )
            for fallback_path in fallback_paths:
                try:
                    csv_text = fallback_path.read_text(encoding="utf-8")
                    series = _parse_fred_csv(
                        csv_text,
                        series_id=series_id,
                        start_date=start_date,
                        end_date=end_date,
                    )
                    if fallback_path != cache_path and not _has_valid_observations(series):
                        raise DataFetchError(
                            f"fallback cache {fallback_path.name!r} has no valid observations"
                        )
                    fallback_metadata_path = fallback_path.with_suffix(".meta.json")
                    cached_meta = _read_cache_metadata(fallback_metadata_path)
                    return _attach_metadata(
                        series,
                        provider="fred_fredgraph_csv",
                        source="cache_fallback",
                        source_url=str(cached_meta.get("source_url", FRED_CSV_URL)),
                        cache_path=fallback_path,
                        cache_hit=True,
                        stale=True,
                        fetched_at=cached_meta.get("fetched_at"),
                        error=str(exc),
                    )
                except (OSError, UnicodeError, DataFetchError) as fallback_exc:
                    cache_error = fallback_exc
            detail = f"FRED fetch failed for {series_id}: {exc}"
            if cache_error is not None:
                detail += f"; cached copy was unusable: {cache_error}"
            raise DataFetchError(detail) from exc


def fetch_yahoo_daily(
    symbol: str,
    start: str | date | datetime | pd.Timestamp,
    end: str | date | datetime | pd.Timestamp,
    *,
    cache_dir: str | Path = "data/cache",
    refresh: bool = False,
    session: requests.Session | None = None,
    timeout: float = 20.0,
    max_retries: int = 3,
) -> pd.DataFrame:
    """Fetch Yahoo daily bars; see ``DataFrame.attrs['metadata']`` for provenance."""

    client = YahooDailyClient(
        cache_dir,
        session=session,
        timeout=timeout,
        max_retries=max_retries,
    )
    return client.fetch(symbol, start, end, refresh=refresh)


def fetch_fred_series(
    series_id: str,
    start: str | date | datetime | pd.Timestamp,
    end: str | date | datetime | pd.Timestamp,
    *,
    cache_dir: str | Path = "data/cache",
    refresh: bool = False,
    session: requests.Session | None = None,
    timeout: float = 20.0,
    max_retries: int = 3,
) -> pd.Series:
    """Fetch a FRED series; see ``Series.attrs['metadata']`` for provenance."""

    client = FredCsvClient(
        cache_dir,
        session=session,
        timeout=timeout,
        max_retries=max_retries,
    )
    return client.fetch(series_id, start, end, refresh=refresh)
