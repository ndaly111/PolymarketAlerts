#!/usr/bin/env python3
"""
Collect daily snow observations from SNODAS for all cities.

This script fetches snow depth and SWE (Snow Water Equivalent) data
from NOAA's SNODAS product for each configured city.

Usage:
    # Collect for today (default)
    python weather/scripts/collect_snow_observations.py

    # Collect for a specific date
    python weather/scripts/collect_snow_observations.py --date 2024-01-15

    # Collect for a specific city
    python weather/scripts/collect_snow_observations.py --city DENHIGH

    # Dry run (don't save to DB)
    python weather/scripts/collect_snow_observations.py --dry-run
"""

from __future__ import annotations

import argparse
import os
import sys
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


def collect_snow_for_city(
    city: Dict[str, Any],
    date_str: str,
    db_path: Path,
    dry_run: bool = False,
    verbose: bool = False,
) -> bool:
    """
    Collect snow observation for a single city and date.

    Returns True if observation was collected/saved successfully.
    """
    city_key = city["key"]
    lat = city["lat"]
    lon = city["lon"]

    if verbose:
        print(f"  [{city_key}] Fetching snow data for {date_str}...")

    try:
        snow = fetch_snow_depth_cached(
            lat=lat,
            lon=lon,
            date=date_str,
            cache_dir=CACHE_DIR,
            timeout=60,
        )
    except Exception as e:
        print(f"  [{city_key}] Error: {e}")
        return False

    if snow is None:
        if verbose:
            print(f"  [{city_key}] No snow data available for {date_str}")
        return False

    if verbose:
        print(f"  [{city_key}] Snow depth: {snow.depth_inches:.1f} in, "
              f"SWE: {snow.swe_inches:.1f} in" if snow.swe_inches else
              f"  [{city_key}] Snow depth: {snow.depth_inches:.1f} in")

    if dry_run:
        print(f"  [{city_key}] [dry-run] Would save snow observation")
        return True

    # Save to database
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
        return True
    except Exception as e:
        print(f"  [{city_key}] DB error: {e}")
        return False


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Collect snow observations from SNODAS."
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
        help="Date to collect (YYYY-MM-DD, default: yesterday)",
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
    args = parser.parse_args()

    config_path = Path(args.config)
    db_path = Path(args.db)

    # Default to yesterday (SNODAS data has ~1 day latency)
    if args.date:
        date_str = args.date
    else:
        yesterday = datetime.now() - timedelta(days=1)
        date_str = yesterday.strftime("%Y-%m-%d")

    cities = load_cities(config_path)

    # Filter cities if specified
    if args.city:
        cities = [c for c in cities if c["key"] == args.city.upper()]
        if not cities:
            print(f"[error] City not found: {args.city}", file=sys.stderr)
            return 1

    print(f"Snow Observation Collection")
    print(f"  Date: {date_str}")
    print(f"  Cities: {len(cities)}")
    print(f"  Database: {db_path}")
    if args.dry_run:
        print(f"  Mode: DRY RUN")
    print()

    success_count = 0
    for city in cities:
        if collect_snow_for_city(
            city,
            date_str,
            db_path,
            dry_run=args.dry_run,
            verbose=args.verbose,
        ):
            success_count += 1

    print()
    print(f"Collection complete: {success_count}/{len(cities)} cities")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
