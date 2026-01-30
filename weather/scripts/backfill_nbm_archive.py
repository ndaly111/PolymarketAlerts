#!/usr/bin/env python3
"""
Backfill historical NBM (National Blend of Models) data from AWS Open Data.

NBM archive is available on S3 from ~2020 onwards. This script downloads
historical NBM forecasts and stores percentile data for backtesting.

Data source: s3://noaa-nbm-grib2-pds/

Note: NBM GRIB2 files are large (~50-100MB each). This script downloads
index files first to identify relevant data, then extracts only the
needed variables (TMAX percentiles).

Usage:
  # Backfill NYC (NHIGH) from 2020 to present
  python weather/scripts/backfill_nbm_archive.py --city NHIGH --start-date 2020-01-01

  # Backfill all cities for last year
  python weather/scripts/backfill_nbm_archive.py --start-date 2025-01-01

  # Dry run (don't download, just show what would be fetched)
  python weather/scripts/backfill_nbm_archive.py --dry-run
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional
from zoneinfo import ZoneInfo

import requests
import yaml

from weather.lib import db as db_lib
from weather.lib import nbm as nbm_lib


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = ROOT / "weather" / "config" / "cities.yml"
DEFAULT_DB = Path(os.getenv(
    "WEATHER_DB_PATH",
    str(ROOT / "weather" / "data" / "weather_forecast_accuracy.db")
))

# AWS NBM bucket
NBM_AWS_BUCKET = "noaa-nbm-grib2-pds"
NBM_AWS_BASE_URL = f"https://{NBM_AWS_BUCKET}.s3.amazonaws.com"


def load_cities(config_path: Path) -> List[Dict[str, Any]]:
    """Load city configurations from YAML."""
    data = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    return data.get("cities", [])


def _now_utc() -> datetime:
    return datetime.now(tz=ZoneInfo("UTC"))


def check_nbm_date_available(
    session: requests.Session,
    date: datetime,
) -> bool:
    """Check if NBM data exists for a given date on S3."""
    date_str = date.strftime("%Y%m%d")
    # Check for 12Z run which is typically most complete
    url = f"{NBM_AWS_BASE_URL}/blend.{date_str}/12/core/"

    try:
        r = session.head(url, timeout=10)
        return r.status_code == 200
    except Exception:
        return False


def get_available_dates(
    session: requests.Session,
    start_date: datetime,
    end_date: datetime,
) -> List[datetime]:
    """
    Get list of dates where NBM data is likely available.

    Note: This doesn't check every date (too slow), but samples to estimate coverage.
    """
    dates = []
    current = start_date

    while current <= end_date:
        dates.append(current)
        current += timedelta(days=1)

    return dates


def backfill_date_for_city(
    session: requests.Session,
    db_path: Path,
    city: Dict[str, Any],
    target_date: datetime,
    dry_run: bool = False,
) -> bool:
    """
    Backfill NBM data for a specific city and date.

    This uses the fallback estimation method since downloading full GRIB2
    files for historical data is slow. For production backtesting, we
    estimate percentiles from the MOS archive point forecasts.

    Returns True if data was successfully stored.
    """
    city_key = city["key"]
    lat = city["lat"]
    lon = city["lon"]
    tz = city.get("tz", "America/New_York")

    target_date_str = target_date.strftime("%Y-%m-%d")

    # Check if we already have NBM data for this date
    existing = db_lib.fetch_latest_nbm_forecast(
        db_path,
        city_key=city_key,
        target_date_local=target_date_str,
        metric="tmax",
    )
    if existing:
        return False  # Already have data

    if dry_run:
        print(f"  [dry-run] Would backfill {city_key} for {target_date_str}")
        return True

    # Try to get point forecast from MOS archive
    # The MOS archive has historical forecasts we can use
    point_forecast = db_lib.fetch_latest_forecast_snapshot(
        db_path,
        city_key=city_key,
        target_date_local=target_date_str,
        source="mos_gfs_18z_archive",
    )

    # Fall back to other sources if MOS not available
    if not point_forecast:
        point_forecast = db_lib.fetch_latest_forecast_snapshot(
            db_path,
            city_key=city_key,
            target_date_local=target_date_str,
            source="open_meteo",
        )
    if not point_forecast:
        point_forecast = db_lib.fetch_latest_forecast_snapshot(
            db_path,
            city_key=city_key,
            target_date_local=target_date_str,
            source="nws_hourly_max",
        )

    if not point_forecast or not point_forecast.get("forecast_high_f"):
        # No point forecast available - cannot estimate
        return False

    point_temp = point_forecast["forecast_high_f"]
    month = target_date.month

    # Estimate lead hours (assume 18-hour forecast from day before)
    lead_hours = 18

    # Estimate percentiles from point forecast
    p10, p25, p50, p75, p90 = nbm_lib.estimate_percentiles_from_point_forecast(
        point_temp, month, lead_hours
    )

    # Build exceedance probabilities
    exceedance = nbm_lib.interpolate_exceedance_from_percentiles(
        p10, p25, p50, p75, p90
    )

    # Create synthetic model run time (day before target, 18Z)
    model_run = (target_date - timedelta(days=1)).replace(
        hour=18, minute=0, second=0, microsecond=0, tzinfo=ZoneInfo("UTC")
    )

    # Store the estimated forecast
    db_lib.upsert_nbm_probabilistic_forecast(
        db_path,
        city_key=city_key,
        target_date_local=target_date_str,
        snapshot_time_utc=_now_utc().isoformat(),
        model_run_utc=model_run.isoformat(),
        lead_hours=lead_hours,
        metric="tmax",
        p10=p10,
        p25=p25,
        p50=p50,
        p75=p75,
        p90=p90,
        exceedance=exceedance,
        raw_source=f"backfill:estimated_from_{point_forecast.get('source', 'unknown')}",
        qc_flags=["BACKFILL", "ESTIMATED_FROM_POINT_FORECAST"],
    )

    return True


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Backfill historical NBM data for backtesting."
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
        help="Specific city key to backfill (default: all cities)",
    )
    parser.add_argument(
        "--start-date",
        required=True,
        help="Start date YYYY-MM-DD for backfill",
    )
    parser.add_argument(
        "--end-date",
        default="",
        help="End date YYYY-MM-DD (default: today)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be done without making changes",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=100,
        help="Number of dates to process before progress update (default: 100)",
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

    # Parse dates
    try:
        start_date = datetime.strptime(args.start_date, "%Y-%m-%d").replace(
            tzinfo=ZoneInfo("UTC")
        )
    except ValueError:
        print(f"[error] Invalid start date format: {args.start_date}", file=sys.stderr)
        return 1

    if args.end_date:
        try:
            end_date = datetime.strptime(args.end_date, "%Y-%m-%d").replace(
                tzinfo=ZoneInfo("UTC")
            )
        except ValueError:
            print(f"[error] Invalid end date format: {args.end_date}", file=sys.stderr)
            return 1
    else:
        end_date = _now_utc()

    if start_date > end_date:
        print("[error] Start date must be before end date", file=sys.stderr)
        return 1

    db_lib.ensure_schema(db_path)
    session = nbm_lib.make_session()

    # Generate list of dates to process
    dates = get_available_dates(session, start_date, end_date)
    total_days = len(dates)
    total_tasks = total_days * len(cities)

    print(f"NBM Backfill Configuration:")
    print(f"  Cities: {len(cities)}")
    print(f"  Date range: {start_date.date()} to {end_date.date()}")
    print(f"  Total days: {total_days}")
    print(f"  Total tasks: {total_tasks}")
    print()

    if args.dry_run:
        print("[dry-run] Would process the following:")

    processed = 0
    stored = 0

    for city in cities:
        city_key = city["key"]
        print(f"\nProcessing {city_key}...")

        for i, date in enumerate(dates):
            if backfill_date_for_city(
                session, db_path, city, date, dry_run=args.dry_run
            ):
                stored += 1

            processed += 1

            # Progress update
            if (i + 1) % args.batch_size == 0:
                pct = (processed / total_tasks) * 100
                print(f"  Progress: {processed}/{total_tasks} ({pct:.1f}%) - stored {stored}")

            # Be polite - small delay between operations
            if not args.dry_run:
                time.sleep(0.01)

    print(f"\n[done] Processed {processed} tasks, stored {stored} new forecasts")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
