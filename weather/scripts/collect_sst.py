#!/usr/bin/env python3
"""
Collect daily SST (Sea Surface Temperature) observations from NDBC buoys.

This script fetches SST data for coastal cities from nearby NDBC buoys.

Usage:
    # Collect for today
    python weather/scripts/collect_sst.py

    # Collect for a specific date
    python weather/scripts/collect_sst.py --date 2024-01-15

    # Collect for a specific city
    python weather/scripts/collect_sst.py --city SFOHIGH
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
from weather.lib.ndbc import fetch_buoy_sst, get_buoy_for_city, BUOY_MAPPING


DEFAULT_CONFIG = ROOT / "weather" / "config" / "cities.yml"
DEFAULT_DB = Path(os.getenv(
    "WEATHER_DB_PATH",
    str(ROOT / "weather" / "data" / "weather_forecast_accuracy.db")
))


def load_cities(config_path: Path) -> List[Dict[str, Any]]:
    """Load city configurations from YAML."""
    data = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    return data.get("cities", [])


def collect_sst_for_city(
    city: Dict[str, Any],
    date_str: str,
    db_path: Path,
    dry_run: bool = False,
    verbose: bool = False,
) -> bool:
    """
    Collect SST observation for a single city and date.

    Returns True if observation was collected/saved successfully.
    """
    city_key = city["key"]

    buoy_id = get_buoy_for_city(city_key)
    if not buoy_id:
        if verbose:
            print(f"  [{city_key}] No buoy mapping (non-coastal city)")
        return False

    if verbose:
        print(f"  [{city_key}] Fetching SST from buoy {buoy_id} for {date_str}...")

    try:
        sst = fetch_buoy_sst(buoy_id, date_str, timeout=60)
    except Exception as e:
        print(f"  [{city_key}] Error: {e}")
        return False

    if sst is None:
        if verbose:
            print(f"  [{city_key}] No SST data available for {date_str}")
        return False

    if verbose:
        anomaly_str = f", anomaly: {sst.sst_anomaly_f:+.1f}°F" if sst.sst_anomaly_f else ""
        print(f"  [{city_key}] SST: {sst.sst_f}°F{anomaly_str}")

    if dry_run:
        print(f"  [{city_key}] [dry-run] Would save SST observation")
        return True

    # Save to database
    try:
        db_lib.upsert_sst_observation(
            db_path,
            city_key=city_key,
            date_local=date_str,
            observation_time_utc=sst.observation_time_utc,
            sst_f=sst.sst_f,
            sst_anomaly_f=sst.sst_anomaly_f,
            buoy_id=sst.buoy_id,
            source=sst.source,
            raw={
                "sst_f": sst.sst_f,
                "sst_anomaly_f": sst.sst_anomaly_f,
                "buoy_id": sst.buoy_id,
            },
        )
        return True
    except Exception as e:
        print(f"  [{city_key}] DB error: {e}")
        return False


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Collect SST observations from NDBC buoys."
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
        help="Date to collect (YYYY-MM-DD, default: today)",
    )
    parser.add_argument(
        "--city",
        help="Specific city key to collect (default: all coastal cities)",
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

    if args.date:
        date_str = args.date
    else:
        date_str = datetime.now().strftime("%Y-%m-%d")

    cities = load_cities(config_path)

    # Filter to coastal cities only (those with buoy mappings)
    coastal_cities = [c for c in cities if get_buoy_for_city(c["key"]) is not None]

    if args.city:
        coastal_cities = [c for c in coastal_cities if c["key"] == args.city.upper()]
        if not coastal_cities:
            print(f"[error] City not found or not coastal: {args.city}", file=sys.stderr)
            print(f"Coastal cities with buoys: {list(BUOY_MAPPING.keys())}")
            return 1

    print(f"SST Observation Collection")
    print(f"  Date: {date_str}")
    print(f"  Coastal cities: {len(coastal_cities)}")
    print(f"  Database: {db_path}")
    if args.dry_run:
        print(f"  Mode: DRY RUN")
    print()

    success_count = 0
    for city in coastal_cities:
        if collect_sst_for_city(
            city,
            date_str,
            db_path,
            dry_run=args.dry_run,
            verbose=args.verbose,
        ):
            success_count += 1

    print()
    print(f"Collection complete: {success_count}/{len(coastal_cities)} cities")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
