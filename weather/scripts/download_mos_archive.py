#!/usr/bin/env python3
"""
Download MOS archive data from Iowa Environmental Mesonet (IEM).

Downloads 3 years of GFS MOS forecasts for all configured cities.

Usage:
  python weather/scripts/download_mos_archive.py
  python weather/scripts/download_mos_archive.py --stations KDEN KAUS
  python weather/scripts/download_mos_archive.py --years 3
"""

from __future__ import annotations

import argparse
import os
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Optional
from zoneinfo import ZoneInfo

import requests
import yaml


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = ROOT / "weather" / "config" / "cities.yml"
OUTPUT_DIR = ROOT / "weather" / "data" / "forecast_archive"

IEM_MOS_URL = "https://mesonet.agron.iastate.edu/cgi-bin/request/mos.py"


def load_stations_from_config(config_path: Path) -> List[str]:
    """Load MOS station codes from cities.yml."""
    data = yaml.safe_load(config_path.read_text())
    stations = []
    for city in data.get("cities", []):
        mos_station = city.get("mos_station")
        if mos_station:
            stations.append(mos_station)
    return stations


def download_mos_data(
    station: str,
    start_date: datetime,
    end_date: datetime,
    model: str = "GFS",
    output_dir: Path = OUTPUT_DIR,
) -> Optional[Path]:
    """
    Download MOS archive data for a single station.

    Returns path to downloaded CSV file, or None on failure.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Format dates with timezone
    sts = start_date.strftime("%Y-%m-%dT00:00:00Z")
    ets = end_date.strftime("%Y-%m-%dT00:00:00Z")

    # Output filename
    start_str = start_date.strftime("%Y%m%d")
    end_str = end_date.strftime("%Y%m%d")
    output_file = output_dir / f"{station.lower()}_{start_str}_{end_str}.csv"

    # Check if already exists
    if output_file.exists() and output_file.stat().st_size > 1000:
        print(f"  [skip] {output_file.name} already exists ({output_file.stat().st_size:,} bytes)")
        return output_file

    params = {
        "station": station,
        "sts": sts,
        "ets": ets,
        "model": model,
        "format": "csv",
    }

    print(f"  Downloading {station} ({start_str} to {end_str})...")

    try:
        r = requests.get(IEM_MOS_URL, params=params, timeout=120)
        r.raise_for_status()

        # Check if we got actual CSV data
        content = r.text
        if content.startswith("[{"):
            # Got JSON error response
            print(f"    [error] API returned error: {content[:200]}")
            return None

        if len(content) < 100:
            print(f"    [warn] Very small response ({len(content)} bytes)")
            return None

        # Write to file
        output_file.write_text(content)
        print(f"    [ok] Saved {output_file.name} ({len(content):,} bytes)")
        return output_file

    except Exception as e:
        print(f"    [error] Failed to download {station}: {e}")
        return None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--stations",
        nargs="*",
        help="Specific stations to download (default: all from cities.yml)",
    )
    parser.add_argument(
        "--years",
        type=int,
        default=3,
        help="Number of years of history to download (default: 3)",
    )
    parser.add_argument(
        "--config",
        default=str(DEFAULT_CONFIG),
        help="Path to cities.yml config",
    )
    parser.add_argument(
        "--output-dir",
        default=str(OUTPUT_DIR),
        help="Output directory for CSV files",
    )
    args = parser.parse_args()

    config_path = Path(args.config)
    output_dir = Path(args.output_dir)

    # Get stations
    if args.stations:
        stations = [s.upper() for s in args.stations]
    else:
        stations = load_stations_from_config(config_path)

    if not stations:
        print("No stations found. Check --stations or cities.yml")
        return 1

    print(f"Downloading MOS archive for {len(stations)} stations:")
    print(f"  Stations: {', '.join(stations)}")

    # Calculate date range
    end_date = datetime.now(ZoneInfo("UTC"))
    start_date = end_date - timedelta(days=args.years * 365)

    print(f"  Date range: {start_date.date()} to {end_date.date()}")
    print(f"  Output: {output_dir}")
    print()

    success = 0
    failed = 0

    for station in stations:
        result = download_mos_data(
            station=station,
            start_date=start_date,
            end_date=end_date,
            output_dir=output_dir,
        )
        if result:
            success += 1
        else:
            failed += 1

        # Be polite to IEM servers
        time.sleep(2)

    print()
    print(f"Done: {success} succeeded, {failed} failed")

    if success > 0:
        print()
        print("Next steps:")
        print("  1. Ingest the CSVs into the database:")
        print(f"     python weather/scripts/ingest_mos_archive_csvs.py --files '{output_dir}/*.csv'")
        print("  2. Build error models:")
        print("     python weather/scripts/build_error_models.py")

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
