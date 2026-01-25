#!/usr/bin/env python3
"""
Backfill historical observed high temperatures from NOAA Climate Data Online (CDO) API.

This script fetches TMAX (daily maximum temperature) from NOAA's GHCND dataset
to populate the observed_cli table with historical data needed for error model building.

Usage:
    python -m weather.scripts.backfill_observed_noaa --start-date 2023-01-01 --end-date 2026-01-23

Requires NOAA_WEATHER_API environment variable (get token from: https://www.ncdc.noaa.gov/cdo-web/token)
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests
import yaml

from weather.lib import db as db_lib

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = ROOT / "weather" / "config" / "cities.yml"
DEFAULT_DB = Path(os.getenv("WEATHER_DB_PATH", str(ROOT / "weather" / "data" / "weather_forecast_accuracy.db")))

# NOAA CDO API base URL
NOAA_API_BASE = "https://www.ncdc.noaa.gov/cdo-web/api/v2"

# Map city keys to NOAA GHCND station IDs
# These are the primary airport weather stations used for official temperature records
CITY_STATION_MAP = {
    "MIAHIGH": "GHCND:USW00012839",  # Miami International Airport
    "LAXHIGH": "GHCND:USW00023174",  # Los Angeles International Airport
    "CHIHIGH": "GHCND:USW00094846",  # Chicago Midway Airport (matches CLI MDW)
    "NHIGH": "GHCND:USW00094728",    # NYC Central Park (primary record station)
}


@dataclass(frozen=True)
class City:
    key: str
    label: str
    tz: str


def load_cities(config_path: Path) -> List[City]:
    data = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    cities: List[City] = []
    for row in data.get("cities", []):
        cities.append(
            City(
                key=str(row["key"]).strip(),
                label=str(row.get("label", row["key"])).strip(),
                tz=str(row.get("tz", "America/New_York")).strip(),
            )
        )
    return cities


def fetch_noaa_tmax(
    session: requests.Session,
    token: str,
    station_id: str,
    start_date: str,
    end_date: str,
) -> Dict[str, int]:
    """
    Fetch daily TMAX values from NOAA CDO API.

    Returns dict mapping date (YYYY-MM-DD) to TMAX in Fahrenheit.
    NOAA returns values in tenths of degrees Celsius, so we convert.
    """
    results: Dict[str, int] = {}

    # NOAA API limits to 1 year per request, so chunk if needed
    start = datetime.strptime(start_date, "%Y-%m-%d").date()
    end = datetime.strptime(end_date, "%Y-%m-%d").date()

    current_start = start
    while current_start <= end:
        # Chunk by ~1 year (365 days) to stay within API limits
        current_end = min(current_start + timedelta(days=364), end)

        url = f"{NOAA_API_BASE}/data"
        params = {
            "datasetid": "GHCND",
            "stationid": station_id,
            "datatypeid": "TMAX",
            "startdate": current_start.isoformat(),
            "enddate": current_end.isoformat(),
            "units": "standard",  # Get Fahrenheit directly
            "limit": 1000,
        }
        headers = {"token": token}

        offset = 0
        while True:
            params["offset"] = offset
            resp = session.get(url, params=params, headers=headers, timeout=30)

            if resp.status_code == 429:
                # Rate limited - wait and retry
                print("[rate-limit] Waiting 5 seconds...")
                time.sleep(5)
                continue

            resp.raise_for_status()
            data = resp.json()

            records = data.get("results", [])
            if not records:
                break

            for rec in records:
                # Date format from NOAA: "2023-01-15T00:00:00"
                date_str = rec.get("date", "")[:10]
                value = rec.get("value")
                if date_str and value is not None:
                    # Value is already in Fahrenheit when units=standard
                    results[date_str] = int(round(value))

            # Check if there are more results
            metadata = data.get("metadata", {}).get("resultset", {})
            total = metadata.get("count", 0)
            offset += len(records)
            if offset >= total:
                break

            # Be nice to the API
            time.sleep(0.25)

        current_start = current_end + timedelta(days=1)
        time.sleep(0.5)  # Pause between year chunks

    return results


def main() -> int:
    p = argparse.ArgumentParser(description="Backfill observed temps from NOAA CDO API")
    p.add_argument("--config", default=str(DEFAULT_CONFIG))
    p.add_argument("--db", default=str(DEFAULT_DB))
    p.add_argument("--start-date", required=True, help="Start date YYYY-MM-DD")
    p.add_argument("--end-date", required=True, help="End date YYYY-MM-DD")
    p.add_argument("--city", default="", help="Optional city key filter")
    p.add_argument("--dry-run", action="store_true", help="Print data without writing to DB")
    args = p.parse_args()

    token = os.getenv("NOAA_WEATHER_API", "").strip()
    if not token:
        print("Error: NOAA_WEATHER_API environment variable not set", file=sys.stderr)
        print("Get a token from: https://www.ncdc.noaa.gov/cdo-web/token", file=sys.stderr)
        return 1

    db_path = Path(args.db)
    if not args.dry_run:
        db_lib.ensure_schema(db_path)

    cities = load_cities(Path(args.config))
    session = requests.Session()

    total_wrote = 0
    total_skipped = 0

    for city in cities:
        if args.city and city.key != args.city:
            continue

        station_id = CITY_STATION_MAP.get(city.key)
        if not station_id:
            print(f"[skip] {city.key}: No NOAA station mapping configured")
            continue

        print(f"[fetch] {city.key} ({city.label}): {station_id}")
        print(f"        Date range: {args.start_date} to {args.end_date}")

        try:
            tmax_data = fetch_noaa_tmax(
                session,
                token,
                station_id,
                args.start_date,
                args.end_date,
            )

            print(f"        Retrieved {len(tmax_data)} daily observations")

            if args.dry_run:
                # Just print a sample
                dates = sorted(tmax_data.keys())
                if dates:
                    print(f"        Sample: {dates[0]}={tmax_data[dates[0]]}°F ... {dates[-1]}={tmax_data[dates[-1]]}°F")
                continue

            wrote = 0
            for date_local, tmax_f in sorted(tmax_data.items()):
                db_lib.upsert_observed_cli(
                    db_path,
                    city_key=city.key,
                    date_local=date_local,
                    tmax_f=tmax_f,
                    fetched_at_utc=datetime.utcnow().isoformat() + "Z",
                    source_url=f"NOAA_CDO:{station_id}",
                    version_used=1,
                    report_date_local=date_local,
                    is_preliminary=False,
                    qc_flags=["NOAA_BACKFILL"],
                    raw_text=f"NOAA CDO GHCND backfill: station={station_id} date={date_local} TMAX={tmax_f}F",
                )
                wrote += 1

            print(f"        Wrote {wrote} rows to DB")
            total_wrote += wrote

        except Exception as exc:
            print(f"[error] {city.key}: {exc}", file=sys.stderr)
            total_skipped += 1

    if args.dry_run:
        print(f"\n[dry-run] Would have written observed data")
    else:
        print(f"\n[done] Wrote {total_wrote} observed rows, {total_skipped} cities had errors")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
