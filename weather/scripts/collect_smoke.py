#!/usr/bin/env python3
"""
Collect smoke observations from NOAA HMS Fire and Smoke Product.

This script fetches smoke plume data for West Coast cities during fire season.

Usage:
    # Collect for yesterday (default, during fire season)
    python weather/scripts/collect_smoke.py

    # Collect for a specific date
    python weather/scripts/collect_smoke.py --date 2024-08-15

    # Force collection outside fire season
    python weather/scripts/collect_smoke.py --date 2024-01-15 --force

    # Backfill a date range
    python weather/scripts/collect_smoke.py --start 2024-08-01 --end 2024-10-31
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
from weather.lib.hms_smoke import (
    fetch_smoke_batch,
    is_fire_season,
    is_smoke_relevant_city,
    SMOKE_RELEVANT_CITIES,
)


DEFAULT_CONFIG = ROOT / "weather" / "config" / "cities.yml"
DEFAULT_DB = Path(os.getenv(
    "WEATHER_DB_PATH",
    str(ROOT / "weather" / "data" / "weather_forecast_accuracy.db")
))


def load_cities(config_path: Path) -> List[Dict[str, Any]]:
    """Load city configurations from YAML."""
    data = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    return data.get("cities", [])


def collect_smoke_for_date(
    cities: List[Dict[str, Any]],
    date_str: str,
    db_path: Path,
    dry_run: bool = False,
    force: bool = False,
    verbose: bool = False,
) -> int:
    """
    Collect smoke observations for all relevant cities on a date.

    Returns count of successfully saved observations.
    """
    date_dt = datetime.strptime(date_str, "%Y-%m-%d")

    # Check fire season
    if not is_fire_season(date_dt) and not force:
        if verbose:
            print(f"  [{date_str}] Outside fire season (Jul-Nov), skipping. Use --force to override.")
        return 0

    # Filter to smoke-relevant cities
    relevant_cities = [c for c in cities if is_smoke_relevant_city(c["key"])]

    if not relevant_cities:
        if verbose:
            print(f"  [{date_str}] No smoke-relevant cities")
        return 0

    # Build location list
    locations = [(c["key"], c["lat"], c["lon"]) for c in relevant_cities]

    if verbose:
        print(f"  [{date_str}] Fetching smoke data for {len(locations)} cities...")

    try:
        results = fetch_smoke_batch(locations, date_str, timeout=60)
    except Exception as e:
        print(f"  [{date_str}] Error: {e}")
        return 0

    saved = 0
    for city_key, smoke in results.items():
        city = next(c for c in relevant_cities if c["key"] == city_key)

        if verbose:
            status = f"SMOKE ({smoke.smoke_density})" if smoke.smoke_present else "clear"
            print(f"    [{city_key}] {status}")

        if dry_run:
            continue

        try:
            db_lib.upsert_smoke_observation(
                db_path,
                city_key=city_key,
                date_local=date_str,
                observation_time_utc=smoke.observation_time_utc,
                smoke_present=smoke.smoke_present,
                smoke_density=smoke.smoke_density,
                source=smoke.source,
                lat=city["lat"],
                lon=city["lon"],
                raw={
                    "smoke_present": smoke.smoke_present,
                    "smoke_density": smoke.smoke_density,
                    "source": smoke.source,
                },
            )
            saved += 1
        except Exception as e:
            print(f"    [{city_key}] DB error: {e}")

    return saved


def date_range(start: datetime, end: datetime):
    """Generate dates between start and end."""
    current = start
    while current <= end:
        yield current
        current += timedelta(days=1)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Collect smoke observations from NOAA HMS."
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
        "--date",
        help="Date to collect (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--start",
        help="Start date for backfill (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--end",
        help="End date for backfill (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force collection outside fire season",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Don't save to database",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Verbose output",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=0.5,
        help="Delay between date requests in seconds (default: 0.5)",
    )
    args = parser.parse_args()

    config_path = Path(args.config)
    db_path = Path(args.db)

    # Determine date(s) to collect
    if args.start and args.end:
        start_date = datetime.strptime(args.start, "%Y-%m-%d")
        end_date = datetime.strptime(args.end, "%Y-%m-%d")
        dates = list(date_range(start_date, end_date))
    elif args.date:
        dates = [datetime.strptime(args.date, "%Y-%m-%d")]
    else:
        # Default to yesterday
        yesterday = datetime.now() - timedelta(days=1)
        dates = [yesterday]

    cities = load_cities(config_path)

    print("Smoke Observation Collection")
    print(f"  Dates: {dates[0].strftime('%Y-%m-%d')} to {dates[-1].strftime('%Y-%m-%d')}")
    print(f"  Smoke-relevant cities: {SMOKE_RELEVANT_CITIES}")
    print(f"  Database: {db_path}")
    if args.force:
        print("  Mode: FORCE (collecting outside fire season)")
    if args.dry_run:
        print("  Mode: DRY RUN")
    print()

    total_saved = 0
    total_attempts = len(dates)

    for date in dates:
        date_str = date.strftime("%Y-%m-%d")

        saved = collect_smoke_for_date(
            cities,
            date_str,
            db_path,
            dry_run=args.dry_run,
            force=args.force,
            verbose=args.verbose,
        )
        total_saved += saved

        if args.delay > 0 and len(dates) > 1:
            time.sleep(args.delay)

    print()
    print(f"Collection complete: {total_saved} observations saved across {total_attempts} dates")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
