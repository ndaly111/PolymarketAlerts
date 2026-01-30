#!/usr/bin/env python3
"""
Collect daily visibility observations from METAR data.

This script fetches visibility and fog data from airport METAR observations
for all configured cities.

Usage:
    # Collect for yesterday (default)
    python weather/scripts/collect_visibility.py

    # Collect for a specific date
    python weather/scripts/collect_visibility.py --date 2024-01-15

    # Collect for a specific city
    python weather/scripts/collect_visibility.py --city SFOHIGH

    # Backfill a date range
    python weather/scripts/collect_visibility.py --start 2024-01-01 --end 2024-01-31
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
from weather.lib.metar import fetch_metar_visibility, get_station_for_city


DEFAULT_CONFIG = ROOT / "weather" / "config" / "cities.yml"
DEFAULT_DB = Path(os.getenv(
    "WEATHER_DB_PATH",
    str(ROOT / "weather" / "data" / "weather_forecast_accuracy.db")
))


def load_cities(config_path: Path) -> List[Dict[str, Any]]:
    """Load city configurations from YAML."""
    data = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    return data.get("cities", [])


def collect_visibility_for_city(
    city: Dict[str, Any],
    date_str: str,
    db_path: Path,
    dry_run: bool = False,
    verbose: bool = False,
) -> bool:
    """
    Collect visibility observations for a single city and date.

    Returns True if observation was collected/saved successfully.
    """
    city_key = city["key"]
    station_id = get_station_for_city(city)

    if not station_id:
        if verbose:
            print(f"  [{city_key}] No METAR station configured")
        return False

    if verbose:
        print(f"  [{city_key}] Fetching visibility from {station_id} for {date_str}...")

    try:
        summary = fetch_metar_visibility(station_id, date_str, timeout=60)
    except Exception as e:
        print(f"  [{city_key}] Error: {e}")
        return False

    if summary is None:
        if verbose:
            print(f"  [{city_key}] No visibility data available for {date_str}")
        return False

    if verbose:
        print(f"  [{city_key}] Min vis: {summary.visibility_min_miles:.1f} mi, "
              f"Fog hours: {summary.fog_hours}, "
              f"Obs count: {summary.obs_count}")

    if dry_run:
        print(f"  [{city_key}] [dry-run] Would save visibility observation")
        return True

    # Save individual observations or summary
    # For simplicity, we store a single summary observation per day
    try:
        # Use noon as representative time for daily summary
        obs_time = f"{date_str}T12:00:00Z"

        db_lib.upsert_visibility_observation(
            db_path,
            city_key=city_key,
            date_local=date_str,
            observation_time_utc=obs_time,
            visibility_miles=summary.visibility_min_miles,
            ceiling_ft=None,  # Not stored in summary
            fog_present=summary.fog_hours > 0,
            station_id=station_id,
            source=summary.source,
            raw={
                "visibility_min_miles": summary.visibility_min_miles,
                "visibility_avg_miles": summary.visibility_avg_miles,
                "fog_hours": summary.fog_hours,
                "low_visibility_hours": summary.low_visibility_hours,
                "obs_count": summary.obs_count,
            },
        )
        return True
    except Exception as e:
        print(f"  [{city_key}] DB error: {e}")
        return False


def date_range(start: datetime, end: datetime):
    """Generate dates between start and end."""
    current = start
    while current <= end:
        yield current
        current += timedelta(days=1)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Collect visibility observations from METAR data."
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
        "--city",
        help="Specific city key to collect (default: all cities)",
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
        help="Delay between requests in seconds (default: 0.5)",
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

    if args.city:
        cities = [c for c in cities if c["key"] == args.city.upper()]
        if not cities:
            print(f"[error] City not found: {args.city}", file=sys.stderr)
            return 1

    print("Visibility Observation Collection")
    print(f"  Dates: {dates[0].strftime('%Y-%m-%d')} to {dates[-1].strftime('%Y-%m-%d')}")
    print(f"  Cities: {len(cities)}")
    print(f"  Database: {db_path}")
    if args.dry_run:
        print("  Mode: DRY RUN")
    print()

    total_success = 0
    total_attempts = 0

    for date in dates:
        date_str = date.strftime("%Y-%m-%d")

        if len(dates) > 1:
            print(f"\n--- {date_str} ---")

        for city in cities:
            total_attempts += 1
            if collect_visibility_for_city(
                city,
                date_str,
                db_path,
                dry_run=args.dry_run,
                verbose=args.verbose,
            ):
                total_success += 1

            if args.delay > 0:
                time.sleep(args.delay)

    print()
    print(f"Collection complete: {total_success}/{total_attempts} observations")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
