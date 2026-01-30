#!/usr/bin/env python3
"""
Collect NBM (National Blend of Models) probabilistic forecasts.

NBM provides native percentiles (p10, p25, p50, p75, p90) which are ideal
for probability-based trading. This script:

1. Fetches the latest NBM forecast for each configured city
2. Extracts temperature percentiles and computes exceedance probabilities
3. Stores results in the nbm_probabilistic_forecasts table

Usage:
  python weather/scripts/collect_nbm_forecast.py
  python weather/scripts/collect_nbm_forecast.py --city NHIGH
  python weather/scripts/collect_nbm_forecast.py --target-date 2026-01-30
"""

from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
from zoneinfo import ZoneInfo

import yaml

from weather.lib import db as db_lib
from weather.lib import nbm as nbm_lib


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = ROOT / "weather" / "config" / "cities.yml"
DEFAULT_DB = Path(os.getenv(
    "WEATHER_DB_PATH",
    str(ROOT / "weather" / "data" / "weather_forecast_accuracy.db")
))


def load_cities(config_path: Path) -> List[Dict[str, Any]]:
    """Load city configurations from YAML."""
    data = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    return data.get("cities", [])


def _now_utc() -> datetime:
    return datetime.now(tz=ZoneInfo("UTC"))


def _now_in_tz(tz_name: str) -> datetime:
    return _now_utc().astimezone(ZoneInfo(tz_name))


def collect_nbm_for_city(
    session,
    db_path: Path,
    city: Dict[str, Any],
    target_date: str,
    overwrite: bool = False,
    use_fallback: bool = True,
) -> bool:
    """
    Collect NBM forecast for a single city.

    Args:
        session: Requests session
        db_path: Database path
        city: City configuration dict
        target_date: Target date (YYYY-MM-DD)
        overwrite: Whether to overwrite existing data
        use_fallback: If true, estimate percentiles from point forecast if NBM unavailable

    Returns:
        True if forecast was collected/stored, False otherwise
    """
    city_key = city["key"]
    lat = city["lat"]
    lon = city["lon"]
    tz = city.get("tz", "America/New_York")

    # Check for existing forecast
    if not overwrite:
        existing = db_lib.fetch_latest_nbm_forecast(
            db_path,
            city_key=city_key,
            target_date_local=target_date,
            metric="tmax",
        )
        if existing:
            # Check if it's from today's model run
            model_run_dt = datetime.fromisoformat(existing["model_run_utc"].replace("Z", "+00:00"))
            hours_old = (_now_utc() - model_run_dt).total_seconds() / 3600
            if hours_old < 6:  # Less than 6 hours old
                print(f"  [skip] {city_key}: recent NBM forecast exists (model run {hours_old:.1f}h ago)")
                return False

    # Try to fetch NBM forecast
    forecast = nbm_lib.fetch_nbm_probabilistic(
        session,
        city_key=city_key,
        lat=lat,
        lon=lon,
        target_date=target_date,
        timezone=tz,
    )

    if forecast:
        # Store the forecast
        db_lib.upsert_nbm_probabilistic_forecast(
            db_path,
            city_key=city_key,
            target_date_local=target_date,
            snapshot_time_utc=_now_utc().isoformat(),
            model_run_utc=forecast.model_run_utc,
            lead_hours=forecast.lead_hours,
            metric="tmax",
            p10=forecast.p10,
            p25=forecast.p25,
            p50=forecast.p50,
            p75=forecast.p75,
            p90=forecast.p90,
            exceedance=forecast.exceedance,
            raw_source=forecast.raw_source,
            qc_flags=[],
        )
        print(
            f"  [ok] {city_key}: NBM forecast stored "
            f"(p10={forecast.p10}, p50={forecast.p50}, p90={forecast.p90}, lead={forecast.lead_hours}h)"
        )
        return True

    # Fallback: try to estimate from existing point forecast
    if use_fallback:
        # Check if we have a recent point forecast
        point_forecast = db_lib.fetch_latest_forecast_snapshot(
            db_path,
            city_key=city_key,
            target_date_local=target_date,
            source="open_meteo",
        )
        if not point_forecast:
            point_forecast = db_lib.fetch_latest_forecast_snapshot(
                db_path,
                city_key=city_key,
                target_date_local=target_date,
                source="nws_hourly_max",
            )

        if point_forecast and point_forecast.get("forecast_high_f"):
            point_temp = point_forecast["forecast_high_f"]
            target_dt = datetime.strptime(target_date, "%Y-%m-%d")
            month = target_dt.month

            # Estimate lead hours from snapshot time
            snapshot_utc = datetime.fromisoformat(
                point_forecast["snapshot_time_utc"].replace("Z", "+00:00")
            )
            target_afternoon = datetime(
                target_dt.year, target_dt.month, target_dt.day,
                18, 0, 0, tzinfo=ZoneInfo("UTC")
            )
            lead_hours = max(0, int((target_afternoon - snapshot_utc).total_seconds() / 3600))

            # Estimate percentiles
            p10, p25, p50, p75, p90 = nbm_lib.estimate_percentiles_from_point_forecast(
                point_temp, month, lead_hours
            )

            # Build exceedance probabilities
            exceedance = nbm_lib.interpolate_exceedance_from_percentiles(
                p10, p25, p50, p75, p90
            )

            # Store with fallback flag
            db_lib.upsert_nbm_probabilistic_forecast(
                db_path,
                city_key=city_key,
                target_date_local=target_date,
                snapshot_time_utc=_now_utc().isoformat(),
                model_run_utc=point_forecast["snapshot_time_utc"],
                lead_hours=lead_hours,
                metric="tmax",
                p10=p10,
                p25=p25,
                p50=p50,
                p75=p75,
                p90=p90,
                exceedance=exceedance,
                raw_source=f"fallback:{point_forecast.get('source', 'unknown')}",
                qc_flags=["ESTIMATED_FROM_POINT_FORECAST"],
            )
            print(
                f"  [ok] {city_key}: Estimated NBM from point forecast "
                f"(p10={p10}, p50={p50}, p90={p90}, spread={p90-p10}F)"
            )
            return True

    print(f"  [warn] {city_key}: Could not collect NBM forecast")
    return False


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Collect NBM probabilistic forecasts for configured cities."
    )
    parser.add_argument(
        "--config",
        default=str(DEFAULT_CONFIG),
        help="Path to cities.yml configuration",
    )
    parser.add_argument(
        "--db",
        default=str(DEFAULT_DB),
        help="Path to SQLite database",
    )
    parser.add_argument(
        "--city",
        default="",
        help="Specific city key to collect (default: all cities)",
    )
    parser.add_argument(
        "--target-date",
        default="",
        help="Target date YYYY-MM-DD (default: today in each city's timezone)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing forecasts",
    )
    parser.add_argument(
        "--no-fallback",
        action="store_true",
        help="Don't estimate from point forecasts if NBM unavailable",
    )
    args = parser.parse_args()

    config_path = Path(args.config)
    db_path = Path(args.db)

    if not config_path.exists():
        print(f"[error] Config file not found: {config_path}", file=sys.stderr)
        return 1

    cities = load_cities(config_path)
    if not cities:
        print("[error] No cities found in config", file=sys.stderr)
        return 1

    # Filter to specific city if requested
    if args.city:
        cities = [c for c in cities if c["key"] == args.city.upper()]
        if not cities:
            print(f"[error] City not found: {args.city}", file=sys.stderr)
            return 1

    db_lib.ensure_schema(db_path)
    session = nbm_lib.make_session()

    print(f"Collecting NBM forecasts for {len(cities)} cities...")
    collected = 0

    for city in cities:
        city_key = city["key"]
        tz = city.get("tz", "America/New_York")

        # Determine target date
        if args.target_date:
            target_date = args.target_date
        else:
            target_date = _now_in_tz(tz).date().isoformat()

        if collect_nbm_for_city(
            session,
            db_path,
            city,
            target_date,
            overwrite=args.overwrite,
            use_fallback=not args.no_fallback,
        ):
            collected += 1

    print(f"\n[done] Collected {collected}/{len(cities)} NBM forecasts")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
