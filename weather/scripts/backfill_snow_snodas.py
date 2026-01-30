#!/usr/bin/env python3
"""
Backfill historical snow observations from SNODAS.

This script fetches snow depth data for a date range to build
training data for the ML model.

Usage:
    # Backfill 2020-present for all cities
    python weather/scripts/backfill_snow_snodas.py --start 2020-01-01

    # Backfill specific date range for a city
    python weather/scripts/backfill_snow_snodas.py --city DENHIGH --start 2023-01-01 --end 2023-12-31

    # Winter months only (more relevant for snow)
    python weather/scripts/backfill_snow_snodas.py --start 2020-01-01 --winter-only

    # Dry run
    python weather/scripts/backfill_snow_snodas.py --start 2024-01-01 --dry-run
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List

import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from weather.lib import db as db_lib
from weather.lib.snodas import fetch_snow_depth_cached, SnowObservation


DEFAULT_CONFIG = ROOT / "weather" / "config" / "cities.yml"
DEFAULT_DB = Path(os.getenv(
    "WEATHER_DB_PATH",
    str(ROOT / "weather" / "data" / "weather_forecast_accuracy.db")
))
CACHE_DIR = ROOT / "weather" / "data" / "snodas_cache"


def load_cities(config_path: Path) -> List[Dict[str, Any]]:
    """Load city configurations from YAML."""
    data = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    return data.get("cities", [])


def date_range(start: datetime, end: datetime, winter_only: bool = False):
    """Generate dates between start and end."""
    current = start
    while current <= end:
        if winter_only:
            # Only yield dates in winter months (Nov-Mar)
            if current.month in (11, 12, 1, 2, 3):
                yield current
        else:
            yield current
        current += timedelta(days=1)


def backfill_city(
    city: Dict[str, Any],
    start_date: datetime,
    end_date: datetime,
    db_path: Path,
    dry_run: bool = False,
    winter_only: bool = False,
    delay_seconds: float = 0.5,
) -> tuple[int, int]:
    """
    Backfill snow observations for a single city.

    Returns (success_count, total_count)
    """
    city_key = city["key"]
    lat = city["lat"]
    lon = city["lon"]

    print(f"\n[{city_key}] Backfilling snow data...")
    print(f"  Lat: {lat}, Lon: {lon}")
    print(f"  Range: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
    if winter_only:
        print(f"  Mode: Winter months only (Nov-Mar)")

    success = 0
    total = 0

    for date in date_range(start_date, end_date, winter_only):
        date_str = date.strftime("%Y-%m-%d")
        total += 1

        # Check if we already have data for this date
        existing = db_lib.fetch_snow_observation(db_path, city_key=city_key, date_local=date_str)
        if existing:
            if total % 100 == 0:
                print(f"  [{city_key}] {date_str}: Already have data, skipping...")
            success += 1
            continue

        try:
            snow = fetch_snow_depth_cached(
                lat=lat,
                lon=lon,
                date=date_str,
                cache_dir=CACHE_DIR,
                timeout=120,
            )
        except Exception as e:
            print(f"  [{city_key}] {date_str}: Error - {e}")
            continue

        if snow is None:
            if total % 50 == 0:
                print(f"  [{city_key}] {date_str}: No data available")
            continue

        if dry_run:
            print(f"  [{city_key}] {date_str}: {snow.depth_inches:.1f} in [dry-run]")
            success += 1
        else:
            try:
                db_lib.upsert_snow_observation(
                    db_path,
                    city_key=city_key,
                    date_local=date_str,
                    snapshot_time_utc=datetime.utcnow().isoformat() + "Z",
                    snow_depth_inches=snow.depth_inches,
                    swe_inches=snow.swe_inches,
                    source=snow.source,
                    lat=lat,
                    lon=lon,
                    raw={
                        "depth_inches": snow.depth_inches,
                        "swe_inches": snow.swe_inches,
                    },
                )
                success += 1

                if total % 30 == 0:
                    print(f"  [{city_key}] Progress: {total} dates, {success} saved, "
                          f"latest: {date_str} = {snow.depth_inches:.1f} in")
            except Exception as e:
                print(f"  [{city_key}] {date_str}: DB error - {e}")

        # Rate limit to be nice to NOAA servers
        # (Cache reduces actual requests significantly)
        if delay_seconds > 0:
            time.sleep(delay_seconds)

    return success, total


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Backfill historical snow observations from SNODAS."
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
        "--start",
        required=True,
        help="Start date (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--end",
        help="End date (YYYY-MM-DD, default: yesterday)",
    )
    parser.add_argument(
        "--city",
        help="Specific city key (default: all cities)",
    )
    parser.add_argument(
        "--winter-only",
        action="store_true",
        help="Only backfill winter months (Nov-Mar)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Don't save to database",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=0.1,
        help="Delay between requests in seconds (default: 0.1)",
    )
    args = parser.parse_args()

    config_path = Path(args.config)
    db_path = Path(args.db)

    start_date = datetime.strptime(args.start, "%Y-%m-%d")
    if args.end:
        end_date = datetime.strptime(args.end, "%Y-%m-%d")
    else:
        end_date = datetime.now() - timedelta(days=1)

    cities = load_cities(config_path)

    if args.city:
        cities = [c for c in cities if c["key"] == args.city.upper()]
        if not cities:
            print(f"[error] City not found: {args.city}", file=sys.stderr)
            return 1

    print("=" * 60)
    print("SNODAS Snow Data Backfill")
    print("=" * 60)
    print(f"  Start: {start_date.strftime('%Y-%m-%d')}")
    print(f"  End: {end_date.strftime('%Y-%m-%d')}")
    print(f"  Cities: {len(cities)}")
    print(f"  Database: {db_path}")
    print(f"  Cache: {CACHE_DIR}")
    if args.winter_only:
        print(f"  Mode: Winter months only")
    if args.dry_run:
        print(f"  Mode: DRY RUN")

    # Create cache directory
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    total_success = 0
    total_dates = 0

    for city in cities:
        success, total = backfill_city(
            city,
            start_date,
            end_date,
            db_path,
            dry_run=args.dry_run,
            winter_only=args.winter_only,
            delay_seconds=args.delay,
        )
        total_success += success
        total_dates += total

    print()
    print("=" * 60)
    print(f"Backfill complete")
    print(f"  Total dates processed: {total_dates}")
    print(f"  Successfully saved: {total_success}")
    print("=" * 60)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
