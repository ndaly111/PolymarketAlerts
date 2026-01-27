#!/usr/bin/env python3
"""
Download historical observed daily high temperatures from IEM.

Uses the IEM daily summary API to get observed max temps for building error models.

Usage:
  python weather/scripts/download_iem_observed.py
  python weather/scripts/download_iem_observed.py --stations KDEN KAUS
  python weather/scripts/download_iem_observed.py --years 3
"""

from __future__ import annotations

import argparse
import os
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional
from zoneinfo import ZoneInfo

import requests
import yaml

from weather.lib import db as db_lib


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = ROOT / "weather" / "config" / "cities.yml"
DEFAULT_DB = Path(os.getenv("WEATHER_DB_PATH", str(ROOT / "weather" / "data" / "weather_forecast_accuracy.db")))

# IEM API base URL for daily summaries
IEM_DAILY_API = "https://mesonet.agron.iastate.edu/api/1/daily.json"

# Station -> IEM network mapping
# Format: (IEM station code without K prefix, IEM network)
STATION_NETWORKS = {
    # Original cities
    "KMIA": ("MIA", "FL_ASOS"),
    "KLAX": ("LAX", "CA_ASOS"),
    "KMDW": ("MDW", "IL_ASOS"),
    "KNYC": ("NYC", "NY_ASOS"),
    # New cities
    "KDEN": ("DEN", "CO_ASOS"),
    "KAUS": ("AUS", "TX_ASOS"),
    "KSFO": ("SFO", "CA_ASOS"),
    "KPHL": ("PHL", "PA_ASOS"),
    "KSEA": ("SEA", "WA_ASOS"),
    "KDCA": ("DCA", "VA_ASOS"),  # Reagan National is in VA
    "KLAS": ("LAS", "NV_ASOS"),
    "KMSY": ("MSY", "LA_ASOS"),
}


def load_cities_from_config(config_path: Path) -> List[Dict[str, Any]]:
    """Load cities from config with MOS station codes."""
    data = yaml.safe_load(config_path.read_text())
    cities = []
    for city in data.get("cities", []):
        mos_station = city.get("mos_station")
        if mos_station:
            cities.append({
                "key": city["key"],
                "mos_station": mos_station,
                "tz": city.get("tz", "America/New_York"),
            })
    return cities


def fetch_daily_high(
    station: str,
    date_str: str,
    network: str,
) -> Optional[int]:
    """
    Fetch daily high temperature from IEM for a single date.

    Returns temperature in Fahrenheit, or None if not available.
    """
    params = {
        "station": station,
        "date": date_str,
        "network": network,
    }

    try:
        r = requests.get(IEM_DAILY_API, params=params, timeout=30)
        r.raise_for_status()
        data = r.json()
        records = data.get("data", [])
        if records:
            max_tmpf = records[0].get("max_tmpf")
            if max_tmpf is not None:
                return int(round(float(max_tmpf)))
    except Exception:
        pass

    return None


def fetch_daily_range(
    station: str,
    network: str,
    start_date: datetime,
    end_date: datetime,
) -> List[Dict[str, Any]]:
    """
    Fetch daily highs for a date range.

    IEM API only supports single-date queries, so we batch by month.
    """
    results = []

    # Use sdate/edate parameters for date range
    url = "https://mesonet.agron.iastate.edu/api/1/daily.json"
    params = {
        "station": station,
        "network": network,
        "sdate": start_date.strftime("%Y-%m-%d"),
        "edate": end_date.strftime("%Y-%m-%d"),
    }

    try:
        r = requests.get(url, params=params, timeout=120)
        r.raise_for_status()
        data = r.json()
        for rec in data.get("data", []):
            date_str = rec.get("date")
            max_tmpf = rec.get("max_tmpf")
            if date_str and max_tmpf is not None:
                results.append({
                    "date": date_str,
                    "tmax_f": int(round(float(max_tmpf))),
                })
    except Exception as e:
        print(f"    [error] Failed to fetch {station}: {e}")

    return results


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--stations",
        nargs="*",
        help="Specific MOS stations to download (default: all from cities.yml)",
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
    )
    parser.add_argument(
        "--db",
        default=str(DEFAULT_DB),
    )
    args = parser.parse_args()

    db_path = Path(args.db)
    db_lib.ensure_schema(db_path)

    cities = load_cities_from_config(Path(args.config))

    # Filter by stations if specified
    if args.stations:
        station_set = {s.upper() for s in args.stations}
        cities = [c for c in cities if c["mos_station"] in station_set]

    if not cities:
        print("No cities found. Check --stations or cities.yml")
        return 1

    # Calculate date range
    end_date = datetime.now(ZoneInfo("UTC")) - timedelta(days=1)  # Yesterday
    start_date = end_date - timedelta(days=args.years * 365)

    print(f"Downloading observed daily highs for {len(cities)} cities:")
    print(f"  Date range: {start_date.date()} to {end_date.date()}")
    print()

    total_wrote = 0
    fetched_at_utc = datetime.now(ZoneInfo("UTC")).isoformat()

    for city in cities:
        mos_station = city["mos_station"]
        city_key = city["key"]
        tz = city["tz"]

        # Get IEM station info
        if mos_station not in STATION_NETWORKS:
            print(f"  [skip] {city_key}: No IEM network mapping for {mos_station}")
            continue

        iem_station, network = STATION_NETWORKS[mos_station]

        print(f"  Fetching {city_key} ({iem_station}/{network})...")

        # Fetch all daily data in one request
        records = fetch_daily_range(iem_station, network, start_date, end_date)

        if not records:
            print(f"    [warn] No data returned")
            continue

        # Insert into database
        wrote = 0
        for rec in records:
            try:
                db_lib.upsert_observed_cli(
                    db_path,
                    city_key=city_key,
                    date_local=rec["date"],
                    tmax_f=rec["tmax_f"],
                    fetched_at_utc=fetched_at_utc,
                    source_url=f"{IEM_DAILY_API}?station={iem_station}&network={network}",
                    version_used=0,
                    report_date_local=rec["date"],
                    is_preliminary=False,
                    qc_flags=["source:iem_daily"],
                    raw_text="",
                )
                wrote += 1
            except Exception as e:
                pass  # Skip duplicates

        print(f"    [ok] Wrote {wrote}/{len(records)} observations")
        total_wrote += wrote

        # Be polite to IEM servers
        time.sleep(1)

    print()
    print(f"Done: {total_wrote} total observations written to {db_path}")

    print()
    print("Next step - build error models:")
    print("  PYTHONPATH=. python3 weather/scripts/build_error_models.py --forecast-source mos_gfs_18z_archive")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
