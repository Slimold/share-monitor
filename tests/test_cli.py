from __future__ import annotations

import json
from pathlib import Path

import pytest

from us_tech_monitor import cli


def _write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _valid_payloads() -> dict[str, dict[str, object]]:
    config = {
        "schema_version": "1.0",
        "lookback_years": 5,
        "benchmark": "QQQ",
        "broad_market": "SPY",
        "sector_proxies": ["XLK", "SMH"],
        "watchlist": ["AAPL"],
        "strategies": {
            "mu_soxs": {
                "enabled": True,
                "signal_symbol": "MU",
                "inverse_symbol": "SOXS",
                "sector_proxy": "SMH",
            }
        },
        "fred_series": {"CASH3M": "DGS3MO"},
        "backtest": {
            "warmup_calendar_days": 500,
            "holdout_sessions": 252,
            "transaction_cost_bps": 5.0,
            "annualization": 252,
        },
        "quality": {
            "maximum_market_age_sessions": 3,
            "minimum_history_sessions": 500,
        },
    }
    metrics = {"total_return": 0.1}
    run = {
        "strategy": {"metrics": metrics},
        "benchmark": {"metrics": metrics},
    }
    snapshot = {
        "schema_version": "1.0",
        "model_version": "test-model",
        "generated_at_utc": "2026-09-27T00:00:00+00:00",
        "as_of_market_date": "2026-09-25",
        "window": {"start": "2021-09-25"},
        "quality": {"status": "ok"},
        "market": {
            "risk_score": 42.0,
            "regime": "neutral",
            "action": "hold",
            "target_position": 0.55,
            "components": {"trend_risk": 40.0},
            "latest": {"QQQ": 500.0},
        },
        "history": [{"date": "2026-09-25", "risk_score": 42.0}],
        "watchlist": [{"symbol": "AAPL"}],
        "strategies": {"mu_soxs": {"status": "available"}},
    }
    backtest = {
        "schema_version": "1.0",
        "model_version": "test-model",
        "generated_at_utc": "2026-09-27T00:00:00+00:00",
        "window": {"start": "2021-09-25"},
        "assumptions": {"annualization": 252},
        "full": run,
        "holdout": run,
        "experiments": {"conservative": {"full": run, "holdout": run}},
        "equity_curve": [{"date": "2026-09-25", "strategy": 1.1}],
        "holdout_equity_curve": [{"date": "2026-09-25", "strategy": 1.1}],
    }
    return {
        "config/us_tech_mvp.json": config,
        "data/us_tech_snapshot.json": snapshot,
        "data/backtest_results.json": backtest,
    }


def _make_artifacts(root: Path) -> None:
    for relative_path in (
        "web/index.html",
        "web/app.js",
        "web/styles.css",
        "reports/backtest_latest.md",
    ):
        _write(root / relative_path, "generated artifact")
    for relative_path, payload in _valid_payloads().items():
        _write(root / relative_path, json.dumps(payload))


def _run_doctor(monkeypatch: pytest.MonkeyPatch, root: Path) -> int:
    monkeypatch.setattr(cli, "repository_root", lambda: root)
    return cli.main(["doctor"])


def test_doctor_accepts_complete_valid_artifacts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    _make_artifacts(tmp_path)

    assert _run_doctor(monkeypatch, tmp_path) == 0
    assert "present and valid" in capsys.readouterr().out


@pytest.mark.parametrize("strategy_mode", ["absent", "disabled"])
def test_doctor_accepts_optional_mu_soxs_strategy_when_not_enabled(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    strategy_mode: str,
) -> None:
    _make_artifacts(tmp_path)
    config = _valid_payloads()["config/us_tech_mvp.json"]
    snapshot = _valid_payloads()["data/us_tech_snapshot.json"]
    if strategy_mode == "absent":
        config.pop("strategies")
    else:
        config["strategies"] = {"mu_soxs": {"enabled": False}}
    snapshot.pop("strategies")
    _write(tmp_path / "config/us_tech_mvp.json", json.dumps(config))
    _write(tmp_path / "data/us_tech_snapshot.json", json.dumps(snapshot))

    assert _run_doctor(monkeypatch, tmp_path) == 0
    assert "present and valid" in capsys.readouterr().out


def test_doctor_requires_mu_soxs_snapshot_when_strategy_is_enabled(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    _make_artifacts(tmp_path)
    snapshot = _valid_payloads()["data/us_tech_snapshot.json"]
    snapshot.pop("strategies")
    _write(tmp_path / "data/us_tech_snapshot.json", json.dumps(snapshot))

    assert _run_doctor(monkeypatch, tmp_path) == 1
    assert "strategies.mu_soxs" in capsys.readouterr().out


@pytest.mark.parametrize(
    "relative_path",
    ["web/app.js", "web/styles.css", "reports/backtest_latest.md"],
)
def test_doctor_rejects_missing_frontend_or_report_artifact(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    relative_path: str,
) -> None:
    _make_artifacts(tmp_path)
    (tmp_path / relative_path).unlink()

    assert _run_doctor(monkeypatch, tmp_path) == 1
    assert relative_path in capsys.readouterr().out


@pytest.mark.parametrize("content", ["{not-json", "{}"])
def test_doctor_rejects_malformed_or_empty_json_object(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    content: str,
) -> None:
    _make_artifacts(tmp_path)
    _write(tmp_path / "data/us_tech_snapshot.json", content)

    assert _run_doctor(monkeypatch, tmp_path) == 1
    assert "us_tech_snapshot.json" in capsys.readouterr().out


def test_doctor_rejects_missing_nested_schema_field(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    _make_artifacts(tmp_path)
    payload = _valid_payloads()["data/backtest_results.json"]
    del payload["holdout"]
    _write(tmp_path / "data/backtest_results.json", json.dumps(payload))

    assert _run_doctor(monkeypatch, tmp_path) == 1
    assert "holdout.strategy.metrics" in capsys.readouterr().out


def test_doctor_rejects_inconsistent_schema_versions(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    _make_artifacts(tmp_path)
    payload = _valid_payloads()["data/us_tech_snapshot.json"]
    payload["schema_version"] = "2.0"
    _write(tmp_path / "data/us_tech_snapshot.json", json.dumps(payload))

    assert _run_doctor(monkeypatch, tmp_path) == 1
    assert "inconsistent schema_version" in capsys.readouterr().out
