#!/usr/bin/env python3
"""Generate an EV report for Kalshi weather markets.

This script combines:
1. Current/historical forecast data
2. Error PMF models
3. Kalshi market prices

To produce a report showing expected value opportunities.

Supports point-in-time analysis for backtesting:
- Use --as-of to simulate what the EV would have been at a specific time
- Use --snapshot-hour to target specific forecast capture times

Example usage:
    # Live EV report for NYC today
    python generate_ev_report.py --city NHIGH

    # Historical EV report (what we would have seen at 8am ET on Jan 28)
    python generate_ev_report.py --city NHIGH --date 2026-01-28 --as-of "2026-01-28T13:00:00+00:00"

    # Report for all cities
    python generate_ev_report.py
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from zoneinfo import ZoneInfo

import yaml

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from weather.lib import db as db_lib
from weather.lib.fair import (
    build_ladder_from_forecast_and_error,
    compute_contract_probability,
    normalize_pmf,
    shift_pmf,
    summarize_pmf,
)
from weather.lib.probability_ladder import ProbabilityLadder
from weather.lib.fees import FeeSchedule, ev_no, ev_yes
from weather.lib.kalshi_weather import (
    best_buy_prices_from_snapshot_row,
    parse_event_spec_from_title,
    prob_event,
)
from weather.backtesting.data_loader import AsOfDataLoader, trading_time_to_utc

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = ROOT / "weather" / "config" / "cities.yml"
DEFAULT_DB = Path(os.getenv("WEATHER_DB_PATH", str(ROOT / "weather" / "data" / "weather_forecast_accuracy.db")))
OUT_BASE = ROOT / "weather" / "outputs" / "ev_reports"


@dataclass(frozen=True)
class City:
    key: str
    label: str
    tz: str


@dataclass
class TradeCandidate:
    market_ticker: str
    title: str
    event_display: str
    side: str  # "YES" or "NO"
    model_prob: float
    market_price: float
    ev: float
    volume: int
    open_interest: int


def _now_utc() -> datetime:
    return datetime.now(tz=ZoneInfo("UTC"))


def load_cities(config_path: Path) -> List[City]:
    data = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    out: List[City] = []
    for row in data.get("cities", []):
        out.append(
            City(
                key=str(row["key"]).strip(),
                label=str(row.get("label", row["key"])).strip(),
                tz=str(row.get("tz", "America/New_York")).strip(),
            )
        )
    return out


def _event_display(spec_kind: str, a: int, b: Optional[int]) -> str:
    k = (spec_kind or "").strip().lower()
    if k in ("between", "range", "in") and b is not None:
        return f"{min(a, b)}-{max(a, b)}"
    if k in ("ge", "gte"):
        return f">={a}"
    if k in ("gt",):
        return f">{a}"
    if k in ("le", "lte"):
        return f"<={a}"
    if k in ("lt",):
        return f"<{a}"
    return str(a) if b is None else f"{a}-{b}"


def _market_title_from_raw(raw: Dict[str, Any]) -> str:
    for k in ("title", "market_title", "marketTitle", "name"):
        v = raw.get(k)
        if isinstance(v, str) and v.strip():
            return v.strip()
    return ""


def compute_ev_for_city(
    db_path: Path,
    city: City,
    target_date_local: str,
    as_of_utc: Optional[str],
    forecast_source: str,
    error_model_source: str,
    error_model_hour: int,
    fee_cents: int,
    min_ev: float,
    min_q: float,
    verbose: bool = False,
) -> Tuple[Dict[str, Any], List[TradeCandidate]]:
    """
    Compute EV for all Kalshi markets for a city.

    Returns:
        (metadata dict, list of trade candidates with positive EV)
    """
    loader = AsOfDataLoader(db_path)
    fees = FeeSchedule(open_fee_cents=fee_cents)
    fee_open = fees.open_fee_dollars()

    month = int(target_date_local.split("-")[1])
    now_utc_iso = as_of_utc or _now_utc().isoformat()

    # Get forecast
    if as_of_utc:
        forecast = loader.get_forecast_at_time(
            city_key=city.key,
            target_date_local=target_date_local,
            as_of_utc=as_of_utc,
            source=forecast_source,
        )
    else:
        forecast = db_lib.fetch_latest_forecast_snapshot(
            db_path,
            city_key=city.key,
            target_date_local=target_date_local,
            source=forecast_source,
        )

    if not forecast:
        return {
            "city_key": city.key,
            "label": city.label,
            "target_date_local": target_date_local,
            "as_of_utc": now_utc_iso,
            "error": "no_forecast",
        }, []

    forecast_high = int(forecast["forecast_high_f"])

    # Get error model
    error_model = loader.get_error_model_at_time(
        city_key=city.key,
        month=month,
        snapshot_hour_local=error_model_hour,
        source=error_model_source,
        as_of_utc=now_utc_iso,
    )

    if not error_model:
        return {
            "city_key": city.key,
            "label": city.label,
            "target_date_local": target_date_local,
            "as_of_utc": now_utc_iso,
            "forecast_high_f": forecast_high,
            "error": "no_error_model",
        }, []

    # Build probability distribution
    pmf_error = error_model["pmf"]
    pmf_high = normalize_pmf(shift_pmf(pmf_error, forecast_high))
    summary = summarize_pmf(pmf_high)

    # Get Kalshi markets
    if as_of_utc:
        markets = loader.get_kalshi_market_at_time(
            city_key=city.key,
            target_date_local=target_date_local,
            as_of_utc=as_of_utc,
        )
    else:
        snap_time = db_lib.fetch_latest_kalshi_weather_snapshot_time(
            db_path,
            city_key=city.key,
            target_date_local=target_date_local,
        )
        if not snap_time:
            return {
                "city_key": city.key,
                "label": city.label,
                "target_date_local": target_date_local,
                "as_of_utc": now_utc_iso,
                "forecast_high_f": forecast_high,
                "pmf_summary": {
                    "mean": round(summary.mean, 1),
                    "p10": summary.p10,
                    "p50": summary.p50,
                    "p90": summary.p90,
                },
                "error": "no_market_data",
            }, []
        markets = list(db_lib.fetch_kalshi_weather_markets_at_snapshot(
            db_path,
            snapshot_time_utc=snap_time,
            city_key=city.key,
            target_date_local=target_date_local,
        ))

    if not markets:
        return {
            "city_key": city.key,
            "label": city.label,
            "target_date_local": target_date_local,
            "as_of_utc": now_utc_iso,
            "forecast_high_f": forecast_high,
            "pmf_summary": {
                "mean": round(summary.mean, 1),
                "p10": summary.p10,
                "p50": summary.p50,
                "p90": summary.p90,
            },
            "error": "no_markets",
        }, []

    # Evaluate each market
    candidates: List[TradeCandidate] = []

    for row in markets:
        raw = row.get("raw") if isinstance(row.get("raw"), dict) else {}
        title = _market_title_from_raw(raw)
        if not title:
            continue

        spec = parse_event_spec_from_title(title)
        if not spec:
            continue

        # Model probability
        q = float(prob_event(pmf_high, spec))

        # Skip extreme probabilities
        if q < min_q or q > (1.0 - min_q):
            continue

        # Get prices
        p_yes, p_no = best_buy_prices_from_snapshot_row(row)

        # Calculate EV for both sides
        best_side = None
        best_ev = -1e9
        best_price = None

        if p_yes is not None:
            ev = ev_yes(q, float(p_yes), fee_open)
            if ev > best_ev:
                best_ev = ev
                best_side = "YES"
                best_price = float(p_yes)

        if p_no is not None:
            ev = ev_no(q, float(p_no), fee_open)
            if ev > best_ev:
                best_ev = ev
                best_side = "NO"
                best_price = float(p_no)

        if best_side is None or best_price is None:
            continue

        if best_ev >= min_ev:
            candidates.append(TradeCandidate(
                market_ticker=row.get("market_ticker", ""),
                title=title,
                event_display=_event_display(spec.kind, spec.a, spec.b),
                side=best_side,
                model_prob=q,
                market_price=best_price,
                ev=best_ev,
                volume=int(row.get("volume") or 0),
                open_interest=int(row.get("open_interest") or 0),
            ))

    # Sort by EV
    candidates.sort(key=lambda c: c.ev, reverse=True)

    market_snap_time = markets[0].get("snapshot_time_utc") if markets else None

    metadata = {
        "city_key": city.key,
        "label": city.label,
        "tz": city.tz,
        "target_date_local": target_date_local,
        "as_of_utc": now_utc_iso,
        "forecast_source": forecast_source,
        "forecast_high_f": forecast_high,
        "forecast_snapshot_time_utc": forecast.get("snapshot_time_utc"),
        "error_model_source": error_model_source,
        "error_model_n_samples": error_model.get("n_samples", 0),
        "pmf_summary": {
            "mean": round(summary.mean, 1),
            "p10": summary.p10,
            "p50": summary.p50,
            "p90": summary.p90,
        },
        "kalshi_snapshot_time_utc": market_snap_time,
        "n_markets_evaluated": len(markets),
        "n_candidates": len(candidates),
        "fee_cents": fee_cents,
        "min_ev": min_ev,
        "min_q": min_q,
    }

    return metadata, candidates


def format_report(
    metadata: Dict[str, Any],
    candidates: List[TradeCandidate],
    format_type: str = "text",
) -> str:
    """Format EV report for display."""

    if format_type == "json":
        return json.dumps({
            **metadata,
            "candidates": [
                {
                    "market_ticker": c.market_ticker,
                    "title": c.title,
                    "event": c.event_display,
                    "side": c.side,
                    "model_prob": round(c.model_prob, 4),
                    "market_price": c.market_price,
                    "ev": round(c.ev, 4),
                    "volume": c.volume,
                    "open_interest": c.open_interest,
                }
                for c in candidates
            ]
        }, indent=2)

    # Text format
    lines = [
        f"=== EV Report: {metadata.get('label', metadata['city_key'])} ===",
        f"Date: {metadata['target_date_local']}",
        f"As-of: {metadata['as_of_utc']}",
        "",
    ]

    if "error" in metadata:
        lines.append(f"Error: {metadata['error']}")
        return "\n".join(lines)

    lines.extend([
        f"Forecast: {metadata['forecast_high_f']}°F (source: {metadata['forecast_source']})",
        f"Model: mean={metadata['pmf_summary']['mean']}°F, "
        f"p10={metadata['pmf_summary']['p10']}°F, "
        f"p50={metadata['pmf_summary']['p50']}°F, "
        f"p90={metadata['pmf_summary']['p90']}°F",
        f"Error model: {metadata['error_model_source']} (n={metadata['error_model_n_samples']})",
        "",
    ])

    if not candidates:
        lines.append("No positive EV opportunities found.")
        return "\n".join(lines)

    lines.append(f"Found {len(candidates)} opportunities with EV >= ${metadata['min_ev']:.2f}:")
    lines.append("")
    lines.append(f"{'Event':<12} {'Side':<4} {'Model q':>8} {'Price':>6} {'EV':>8} Market")
    lines.append("-" * 70)

    for c in candidates[:10]:  # Top 10
        lines.append(
            f"{c.event_display:<12} {c.side:<4} {c.model_prob:>7.1%} "
            f"${c.market_price:>4.2f} ${c.ev:>+6.3f} {c.market_ticker}"
        )

    if len(candidates) > 10:
        lines.append(f"... and {len(candidates) - 10} more")

    return "\n".join(lines)


def main() -> int:
    p = argparse.ArgumentParser(description="Generate EV report for Kalshi weather markets")
    p.add_argument("--config", default=str(DEFAULT_CONFIG))
    p.add_argument("--db", default=str(DEFAULT_DB))
    p.add_argument("--city", default="", help="City key filter (default: all cities)")
    p.add_argument("--date", default="", help="Target date YYYY-MM-DD (default: today)")
    p.add_argument(
        "--as-of",
        default="",
        help="Point-in-time for backtesting (ISO UTC timestamp). "
             "Simulates what data was available at that time.",
    )
    p.add_argument(
        "--trading-hour",
        type=int,
        default=0,
        help="Local trading hour (0-23). Combined with --date to compute as-of time. "
             "E.g., --date 2026-01-28 --trading-hour 8 means 8am local on Jan 28.",
    )
    p.add_argument("--forecast-source", default=os.getenv("WEATHER_FORECAST_SOURCE", "nws_hourly_max"))
    p.add_argument(
        "--error-model-source",
        default=os.getenv("WEATHER_ERROR_MODEL_SOURCE", "mos_gfs_18z_archive"),
    )
    p.add_argument(
        "--error-model-hour",
        type=int,
        default=int(os.getenv("WEATHER_ERROR_MODEL_HOUR", "12")),
    )
    p.add_argument("--fee-cents", type=int, default=int(os.getenv("WEATHER_BUY_FEE_CENTS", "2")))
    p.add_argument("--min-ev", type=float, default=float(os.getenv("WEATHER_MIN_EV", "0.02")))
    p.add_argument("--min-q", type=float, default=float(os.getenv("WEATHER_MIN_Q", "0.05")))
    p.add_argument("--format", choices=["text", "json"], default="text")
    p.add_argument("--output", default="", help="Output file path (default: stdout)")
    p.add_argument("--save", action="store_true", help="Save to outputs/ev_reports/")
    p.add_argument("-v", "--verbose", action="store_true")
    args = p.parse_args()

    db_path = Path(args.db)
    cities = load_cities(Path(args.config))

    # Determine as-of time
    as_of_utc = args.as_of.strip() or None

    all_reports = []

    for city in cities:
        if args.city and city.key != args.city:
            continue

        now_local = _now_utc().astimezone(ZoneInfo(city.tz))
        target_date = args.date.strip() or now_local.date().isoformat()

        # If trading hour specified without explicit as-of, compute as-of
        city_as_of = as_of_utc
        if not city_as_of and args.trading_hour > 0:
            city_as_of = trading_time_to_utc(
                date_local=target_date,
                hour_local=args.trading_hour,
                minute_local=0,
                timezone=city.tz,
            )

        metadata, candidates = compute_ev_for_city(
            db_path=db_path,
            city=city,
            target_date_local=target_date,
            as_of_utc=city_as_of,
            forecast_source=args.forecast_source,
            error_model_source=args.error_model_source,
            error_model_hour=args.error_model_hour,
            fee_cents=args.fee_cents,
            min_ev=args.min_ev,
            min_q=args.min_q,
            verbose=args.verbose,
        )

        report = format_report(metadata, candidates, args.format)
        all_reports.append((city, metadata, candidates, report))

        # Print immediately in text mode
        if args.format == "text" and not args.output:
            print(report)
            print()

    # Handle output
    if args.format == "json":
        combined = {
            "generated_at_utc": _now_utc().isoformat(),
            "reports": [
                {
                    **m,
                    "candidates": [
                        {
                            "market_ticker": c.market_ticker,
                            "event": c.event_display,
                            "side": c.side,
                            "model_prob": round(c.model_prob, 4),
                            "market_price": c.market_price,
                            "ev": round(c.ev, 4),
                        }
                        for c in candidates
                    ]
                }
                for _, m, candidates, _ in all_reports
            ]
        }
        output = json.dumps(combined, indent=2)

        if args.output:
            Path(args.output).write_text(output, encoding="utf-8")
            print(f"Wrote {args.output}")
        else:
            print(output)

    # Save to outputs directory
    if args.save:
        for city, metadata, candidates, _ in all_reports:
            date_str = metadata.get("target_date_local", "unknown")
            out_dir = OUT_BASE / date_str
            out_dir.mkdir(parents=True, exist_ok=True)

            out_path = out_dir / f"{city.key}.json"
            out_data = {
                **metadata,
                "candidates": [
                    {
                        "market_ticker": c.market_ticker,
                        "title": c.title,
                        "event": c.event_display,
                        "side": c.side,
                        "model_prob": round(c.model_prob, 4),
                        "market_price": c.market_price,
                        "ev": round(c.ev, 4),
                        "volume": c.volume,
                        "open_interest": c.open_interest,
                    }
                    for c in candidates
                ]
            }
            out_path.write_text(json.dumps(out_data, indent=2), encoding="utf-8")

        print(f"[saved] {len(all_reports)} reports to {OUT_BASE}/")

    # Summary
    total_opportunities = sum(len(c) for _, _, c, _ in all_reports)
    total_ev = sum(sum(x.ev for x in c) for _, _, c, _ in all_reports)

    if args.verbose or len(all_reports) > 1:
        print(f"\n=== Summary ===")
        print(f"Cities evaluated: {len(all_reports)}")
        print(f"Total opportunities: {total_opportunities}")
        print(f"Total EV: ${total_ev:.2f}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
