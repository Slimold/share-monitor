from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

from .pipeline import build_mvp, repository_root
from .server import serve


_TEXT_ARTIFACTS = (
    Path("web/index.html"),
    Path("web/app.js"),
    Path("web/styles.css"),
    Path("reports/backtest_latest.md"),
)

_JSON_SCHEMAS: dict[Path, dict[str, type | tuple[type, ...]]] = {
    Path("config/us_tech_mvp.json"): {
        "schema_version": str,
        "lookback_years": int,
        "benchmark": str,
        "broad_market": str,
        "sector_proxies": list,
        "watchlist": list,
        "fred_series": dict,
        "backtest.warmup_calendar_days": int,
        "backtest.holdout_sessions": int,
        "backtest.transaction_cost_bps": (int, float),
        "backtest.annualization": int,
        "quality.maximum_market_age_sessions": int,
        "quality.minimum_history_sessions": int,
    },
    Path("data/us_tech_snapshot.json"): {
        "schema_version": str,
        "model_version": str,
        "generated_at_utc": str,
        "as_of_market_date": str,
        "window": dict,
        "quality.status": str,
        "market.risk_score": (int, float),
        "market.regime": str,
        "market.action": str,
        "market.target_position": (int, float),
        "market.components": dict,
        "market.latest": dict,
        "history": list,
        "watchlist": list,
    },
    Path("data/backtest_results.json"): {
        "schema_version": str,
        "model_version": str,
        "generated_at_utc": str,
        "window": dict,
        "assumptions": dict,
        "full.strategy.metrics": dict,
        "full.benchmark.metrics": dict,
        "holdout.strategy.metrics": dict,
        "holdout.benchmark.metrics": dict,
        "experiments": dict,
        "equity_curve": list,
        "holdout_equity_curve": list,
    },
}

_MISSING = object()


def _nested_value(payload: dict[str, Any], field: str) -> Any:
    value: Any = payload
    for part in field.split("."):
        if not isinstance(value, dict) or part not in value:
            return _MISSING
        value = value[part]
    return value


def _is_empty(value: Any) -> bool:
    if isinstance(value, str):
        return not value.strip()
    if isinstance(value, (dict, list)):
        return not value
    return False


def _type_label(expected: type | tuple[type, ...]) -> str:
    types = expected if isinstance(expected, tuple) else (expected,)
    return " or ".join(item.__name__ for item in types)


def _validate_json(
    path: Path,
    schema: dict[str, type | tuple[type, ...]],
    *,
    display_path: str | None = None,
) -> tuple[dict[str, Any] | None, list[str]]:
    label = display_path or str(path)
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as exc:
        return None, [f"{label}: cannot be read ({exc})"]
    if not text.strip():
        return None, [f"{label}: file is empty"]
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        return None, [f"{label}: invalid JSON ({exc.msg} at line {exc.lineno})"]
    if not isinstance(payload, dict) or not payload:
        return None, [f"{label}: JSON root must be a non-empty object"]

    problems: list[str] = []
    for field, expected_type in schema.items():
        value = _nested_value(payload, field)
        if value is _MISSING:
            problems.append(f"{label}: missing required field '{field}'")
        elif not isinstance(value, expected_type) or _is_empty(value):
            problems.append(
                f"{label}: field '{field}' must be a non-empty {_type_label(expected_type)}"
            )
    return payload, problems


def _doctor(root: Path) -> int:
    problems: list[str] = []
    parsed: dict[Path, dict[str, Any]] = {}

    for relative_path in (*_TEXT_ARTIFACTS, *_JSON_SCHEMAS):
        path = root / relative_path
        if not path.is_file():
            problems.append(f"{relative_path.as_posix()}: missing file")

    for relative_path in _TEXT_ARTIFACTS:
        path = root / relative_path
        if not path.is_file():
            continue
        try:
            if not path.read_text(encoding="utf-8").strip():
                problems.append(f"{relative_path.as_posix()}: file is empty")
        except (OSError, UnicodeError) as exc:
            problems.append(f"{relative_path.as_posix()}: cannot be read ({exc})")

    for relative_path, schema in _JSON_SCHEMAS.items():
        path = root / relative_path
        if not path.is_file():
            continue
        payload, json_problems = _validate_json(
            path,
            schema,
            display_path=relative_path.as_posix(),
        )
        problems.extend(json_problems)
        if payload is not None:
            parsed[relative_path] = payload

    schema_versions = {
        str(payload["schema_version"])
        for payload in parsed.values()
        if isinstance(payload.get("schema_version"), str) and payload["schema_version"].strip()
    }
    if len(parsed) == len(_JSON_SCHEMAS) and len(schema_versions) > 1:
        problems.append("JSON artifacts use inconsistent schema_version values")

    config_payload = parsed.get(Path("config/us_tech_mvp.json"))
    snapshot_payload = parsed.get(Path("data/us_tech_snapshot.json"))
    if config_payload is not None and snapshot_payload is not None:
        strategy_config = _nested_value(config_payload, "strategies.mu_soxs")
        strategy_enabled = isinstance(strategy_config, dict) and bool(
            strategy_config.get("enabled", True)
        )
        if strategy_enabled:
            strategy_snapshot = _nested_value(snapshot_payload, "strategies.mu_soxs")
            if not isinstance(strategy_snapshot, dict) or _is_empty(strategy_snapshot):
                problems.append(
                    "data/us_tech_snapshot.json: missing required field "
                    "'strategies.mu_soxs' for the enabled strategy"
                )

    if problems:
        print("Artifact check failed:")
        for problem in problems:
            print(f"- {problem}")
        print("Run: python run_us_monitor.py build")
        return 1
    print("All MVP artifacts are present and valid.")
    return 0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="us-tech-monitor",
        description="Build and view the U.S. technology market thermometer MVP.",
    )
    subcommands = parser.add_subparsers(dest="command", required=True)

    build = subcommands.add_parser("build", help="fetch data, score the market, and backtest")
    build.add_argument("--config", type=Path)
    build.add_argument("--start", help="inclusive YYYY-MM-DD (default: five years ago)")
    build.add_argument(
        "--end",
        help="inclusive YYYY-MM-DD (default: conservatively completed U.S. market date)",
    )
    build.add_argument("--cache-dir", type=Path)
    build.add_argument(
        "--refresh",
        action="store_true",
        help="refresh providers before cache fallback",
    )
    build.add_argument("--json", action="store_true", help="print a machine-readable summary")

    server = subcommands.add_parser("serve", help="serve the local dashboard")
    server.add_argument("--host", default="127.0.0.1")
    server.add_argument("--port", type=int, default=8788)
    server.add_argument("--no-browser", action="store_true")

    subcommands.add_parser("doctor", help="check generated dashboard artifacts and JSON schemas")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    root = repository_root()
    if args.command == "build":
        artifacts = build_mvp(
            config_path=args.config,
            start=args.start,
            end=args.end,
            cache_dir=args.cache_dir,
            refresh=args.refresh,
        )
        market = artifacts.snapshot["market"]
        full = artifacts.backtest["full"]
        holdout = artifacts.backtest["holdout"]
        summary = {
            "as_of": artifacts.snapshot["as_of_market_date"],
            "risk_score": market["risk_score"],
            "regime": market["regime"],
            "action": market["action"],
            "full": full,
            "holdout": holdout,
            "snapshot_path": str(artifacts.snapshot_path),
            "backtest_path": str(artifacts.backtest_path),
            "report_path": str(artifacts.report_path),
        }
        if args.json:
            print(json.dumps(summary, ensure_ascii=False, indent=2))
        else:
            print(
                f"Built {summary['as_of']}: risk={summary['risk_score']:.1f} "
                f"regime={summary['regime']} action={summary['action']}"
            )
            print(f"Snapshot: {artifacts.snapshot_path}")
            print(f"Backtest: {artifacts.backtest_path}")
            print(f"Report:   {artifacts.report_path}")
        return 0
    if args.command == "serve":
        serve(root, host=args.host, port=args.port, open_browser=not args.no_browser)
        return 0
    if args.command == "doctor":
        return _doctor(root)
    return 2
