#!/usr/bin/env python3
"""
Backfill historical SST observations from NDBC buoys.

This script fetches historical SST data for coastal cities to build
training data for the ML model.

Usage:
    # Backfill 2020-present for all coastal cities
    python weather/scripts/backfill_sst.py --start 2020

    # Backfill specific years for a city
    python weather/scripts/backfill_sst.py --city SFOHIGH --start 2022 --end 2024

    # Dry run
    python weather/scripts/backfill_sst.py --start 2024 --dry-run
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from weather.lib import db as db_lib
from weather.lib.ndbc import (
    fetch_buoy_sst_historical,
    get_buoy_for_city,
    BUOY_MAPPING,
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


def backfill_city(
    city: Dict[str, Any],
    start_year: int,
    end_year: int,
    db_path: Path,
    dry_run: bool = False,
    delay_seconds: float = 1.0,
) -> tuple[int, int]:
    """
    Backfill SST observations for a single city.

    Returns (success_count, total_count)
    """
    city_key = city["key"]
    buoy_id = get_buoy_for_city(city_key)

    if not buoy_id:
        print(f"[{city_key}] No buoy mapping, skipping")
        return 0, 0

    print(f"\n[{city_key}] Backfilling SST from buoy {buoy_id}...")
    print(f"  Years: {start_year} to {end_year}")

    success = 0
    total = 0

    for year in range(start_year, end_year + 1):
        print(f"  [{city_key}] Fetching {year}...")

        try:
            historical = fetch_buoy_sst_historical(buoy_id, year, timeout=120)
        except Exception as e:
            print(f"  [{city_key}] {year}: Error - {e}")
            continue

        if not historical:
            print(f"  [{city_key}] {year}: No data available")
            continue

        year_success = 0
        for date_str, sst_f, anomaly in historical:
            total += 1

            # Check if we already have data
            existing = db_lib.fetch_sst_observation(
                db_path, city_key=city_key, date_local=date_str
            )
            if existing:
                success += 1
                year_success += 1
                continue

            if dry_run:
                success += 1
                year_success += 1
                continue

            try:
                db_lib.upsert_sst_observation(
                    db_path,
                    city_key=city_key,
                    date_local=date_str,
                    observation_time_utc=f"{date_str}T12:00:00Z",
                    sst_f=sst_f,
                    sst_anomaly_f=anomaly,
                    buoy_id=buoy_id,
                    source="ndbc_historical",
                    raw={
                        "sst_f": sst_f,
                        "sst_anomaly_f": anomaly,
                        "buoy_id": buoy_id,
                        "year": year,
                    },
                )
                success += 1
                year_success += 1
            except Exception as e:
                print(f"  [{city_key}] {date_str}: DB error - {e}")

        print(f"  [{city_key}] {year}: Saved {year_success}/{len(historical)} days")

        # Rate limit
        if delay_seconds > 0:
            time.sleep(delay_seconds)

    return success, total


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Backfill historical SST observations from NDBC buoys."
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
        type=int,
        required=True,
        help="Start year (e.g., 2020)",
    )
    parser.add_argument(
        "--end",
        type=int,
        help="End year (default: current year)",
    )
    parser.add_argument(
        "--city",
        help="Specific city key (default: all coastal cities)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Don't save to database",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=1.0,
        help="Delay between year requests in seconds (default: 1.0)",
    )
    args = parser.parse_args()

    config_path = Path(args.config)
    db_path = Path(args.db)

    start_year = args.start
    end_year = args.end if args.end else datetime.now().year

    cities = load_cities(config_path)

    # Filter to coastal cities only
    coastal_cities = [c for c in cities if get_buoy_for_city(c["key"]) is not None]

    if args.city:
        coastal_cities = [c for c in coastal_cities if c["key"] == args.city.upper()]
        if not coastal_cities:
            print(f"[error] City not found or not coastal: {args.city}", file=sys.stderr)
            print(f"Coastal cities: {list(BUOY_MAPPING.keys())}")
            return 1

    print("=" * 60)
    print("NDBC SST Data Backfill")
    print("=" * 60)
    print(f"  Years: {start_year} to {end_year}")
    print(f"  Coastal cities: {len(coastal_cities)}")
    print(f"  Database: {db_path}")
    if args.dry_run:
        print(f"  Mode: DRY RUN")

    total_success = 0
    total_dates = 0

    for city in coastal_cities:
        success, total = backfill_city(
            city,
            start_year,
            end_year,
            db_path,
            dry_run=args.dry_run,
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
