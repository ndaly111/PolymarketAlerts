#!/usr/bin/env python3
"""
Collect 12z GFS MOS forecast for today's high temperature.

This script fetches the latest 12z MOS forecast from IEM (Iowa Environmental Mesonet)
and stores it in the database for use in ML model predictions.

Usage:
    python weather/scripts/collect_mos_forecast.py
    python weather/scripts/collect_mos_forecast.py --cycle 12  # 12z MOS (default)
    python weather/scripts/collect_mos_forecast.py --cycle 18  # 18z MOS
"""

from __future__ import annotations

import argparse
import csv
import io
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional
from zoneinfo import ZoneInfo

import requests
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from weather.lib import db as db_lib

DEFAULT_CONFIG = ROOT / "weather" / "config" / "cities.yml"
DEFAULT_DB = Path(os.getenv("WEATHER_DB_PATH", str(ROOT / "weather" / "data" / "weather_forecast_accuracy.db")))

IEM_MOS_URL = "https://mesonet.agron.iastate.edu/cgi-bin/request/mos.py"

# Station code -> city_key mapping
STATION_TO_CITY_KEY: Dict[str, str] = {
    "KMDW": "CHIHIGH",
    "KMIA": "MIAHIGH",
    "KLAX": "LAXHIGH",
    "KNYC": "NHIGH",
    "KDEN": "DENHIGH",
    "KAUS": "AUSHIGH",
    "KSFO": "SFOHIGH",
    "KPHL": "PHLHIGH",
    "KSEA": "SEAHIGH",
    "KDCA": "DCHIGH",
    "KLAS": "LVHIGH",
    "KMSY": "NOLHIGH",
}


def load_cities(config_path: Path) -> Dict[str, Dict[str, Any]]:
    """Load city configurations from YAML."""
    data = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    cities = {}
    for row in data.get("cities", []):
        key = str(row["key"]).strip()
        cities[key] = {
            "key": key,
            "tz": str(row.get("tz", "America/New_York")).strip(),
            "mos_station": row.get("mos_station"),
        }
    return cities


def fetch_mos_forecast(
    station: str,
    target_date: datetime,
    cycle_hour: int = 12,
    timeout: int = 30,
) -> Optional[int]:
    """
    Fetch MOS forecast for a specific station, date, and cycle.

    Returns the forecast high temperature in Fahrenheit, or None if not available.
    """
    # IEM needs date range
    start_date = target_date - timedelta(days=1)
    end_date = target_date + timedelta(days=1)

    params = {
        "station": station,
        "sts": start_date.strftime("%Y-%m-%dT00:00:00Z"),
        "ets": end_date.strftime("%Y-%m-%dT23:59:59Z"),
        "model": "GFS",
        "format": "csv",
    }

    try:
        r = requests.get(IEM_MOS_URL, params=params, timeout=timeout)
        r.raise_for_status()

        content = r.text
        if not content or content.startswith("[{"):
            return None

        reader = csv.DictReader(io.StringIO(content))

        for row in reader:
            # Parse runtime to check cycle
            runtime_str = row.get("runtime", "").strip()
            ftime_str = row.get("ftime", "").strip()
            n_x = row.get("n_x", "").strip()

            if not runtime_str or not ftime_str or not n_x:
                continue

            try:
                runtime = datetime.strptime(runtime_str, "%Y-%m-%d %H:%M:%S")
                ftime = datetime.strptime(ftime_str, "%Y-%m-%d %H:%M:%S")
            except ValueError:
                continue

            # Check if this is the right cycle hour
            if runtime.hour != cycle_hour:
                continue

            # Daily HIGH is at ftime 00:00Z with n_x present
            # The HIGH for target_date is at ftime = target_date + 1 day @ 00Z
            # (midnight ending that day, per MOS convention)
            if ftime.hour != 0 or ftime.minute != 0:
                continue

            # Check if ftime matches target date + 1 day
            expected_ftime_date = (target_date + timedelta(days=1)).date()
            if ftime.date() != expected_ftime_date:
                continue

            # Parse temperature
            try:
                return int(round(float(n_x)))
            except ValueError:
                continue

        return None

    except Exception as e:
        print(f"    Error fetching MOS for {station}: {e}")
        return None


def main() -> int:
    parser = argparse.ArgumentParser(description="Collect 12z MOS forecast")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--db", default=str(DEFAULT_DB))
    parser.add_argument("--cycle", type=int, default=12, choices=[0, 6, 12, 18],
                        help="MOS cycle hour UTC (default: 12)")
    parser.add_argument("--target-date", default="",
                        help="Target date YYYY-MM-DD (default: today/tomorrow based on time)")
    args = parser.parse_args()

    config_path = Path(args.config)
    db_path = Path(args.db)
    cycle_hour = args.cycle

    cities = load_cities(config_path)

    # Determine target date
    now_utc = datetime.now(ZoneInfo("UTC"))

    if args.target_date:
        target_date = datetime.strptime(args.target_date, "%Y-%m-%d")
    else:
        # Today's 12z MOS run forecasts today's high (at ftime = tomorrow 00Z)
        # Default to today's date - this is what we need for daily collection
        target_date = now_utc

    target_date_str = target_date.strftime("%Y-%m-%d") if hasattr(target_date, 'strftime') else str(target_date)
    if hasattr(target_date, 'date'):
        target_date_dt = target_date
    else:
        target_date_dt = datetime.strptime(target_date_str, "%Y-%m-%d")

    source = f"mos_gfs_{cycle_hour}z"
    snapshot_hour = cycle_hour

    print(f"Collecting {cycle_hour}z GFS MOS forecasts")
    print(f"  Target date: {target_date_str}")
    print(f"  Source: {source}")
    print()

    db_lib.ensure_schema(db_path)

    collected = 0
    failed = 0

    for station, city_key in STATION_TO_CITY_KEY.items():
        if city_key not in cities:
            continue

        city = cities[city_key]
        tz = city["tz"]

        forecast_high = fetch_mos_forecast(station, target_date_dt, cycle_hour)

        if forecast_high is not None:
            print(f"  {city_key}: {forecast_high}°F")

            # Store in database
            snapshot_time = now_utc.isoformat()

            db_lib.upsert_forecast_snapshot(
                db_path=db_path,
                city_key=city_key,
                target_date_local=target_date_str,
                snapshot_time_utc=snapshot_time,
                snapshot_hour_local=snapshot_hour,
                snapshot_tz=tz,
                forecast_high_f=forecast_high,
                source=source,
                points_url="https://mesonet.agron.iastate.edu/mos/fe.phtml",
                forecast_url="https://mesonet.agron.iastate.edu/mos/fe.phtml",
                qc_flags=["mos_realtime", f"station:{station}"],
                raw={
                    "station": station,
                    "cycle_hour": cycle_hour,
                    "target_date": target_date_str,
                },
            )
            collected += 1
        else:
            print(f"  {city_key}: [no data]")
            failed += 1

    print()
    print(f"Collected: {collected}, Failed: {failed}")

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
