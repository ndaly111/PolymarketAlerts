#!/usr/bin/env python3
"""
Collect forecasts from multiple weather models at 4am for accuracy tracking.

This script collects daily high temperature forecasts from:
- Open-Meteo HRRR (US short-term, 3km resolution)
- Open-Meteo ECMWF (global, most accurate overall)
- Open-Meteo GFS (global, NOAA)
- NWS hourly max (government source)

All forecasts are stored in the database for later comparison with CLI settlements.
Run this daily at 4am local time to capture same-day forecasts.
"""

from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
from zoneinfo import ZoneInfo

import yaml

# Add parent to path for imports
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from weather.lib import db as db_lib
from weather.lib import nws as nws_lib
from weather.lib import open_meteo as om_lib

DEFAULT_CONFIG = ROOT / "weather" / "config" / "cities.yml"
DEFAULT_DB = Path(os.getenv("WEATHER_DB_PATH", str(ROOT / "weather" / "data" / "weather_forecast_accuracy.db")))

# Models to track
OPEN_METEO_MODELS = ["hrrr", "ecmwf", "gfs"]


def load_cities(config_path: Path) -> List[Dict[str, Any]]:
    """Load city configurations."""
    data = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    return data.get("cities", [])


def collect_open_meteo_forecast(
    session,
    lat: float,
    lon: float,
    target_date: str,
    timezone: str,
    model: str,
) -> Optional[int]:
    """Get daily high forecast from Open-Meteo for a specific model."""
    forecast = om_lib.get_daily_high_forecast(
        session, lat, lon, target_date, timezone, model
    )
    return forecast.high_f if forecast else None


def collect_nws_hourly_max(
    session,
    lat: float,
    lon: float,
    target_date: str,
    timezone: str,
) -> Optional[int]:
    """Get daily high from NWS hourly forecast."""
    try:
        endpoints = nws_lib.get_points_endpoints(session, lat, lon)
        hourly = nws_lib.fetch_json(session, endpoints.forecast_hourly_url)

        props = hourly.get("properties", {})
        periods = props.get("periods", [])

        tz = ZoneInfo(timezone)
        temps = []
        for p in periods:
            st = p.get("startTime")
            temp = p.get("temperature")
            if st and temp is not None:
                dt = datetime.fromisoformat(st).astimezone(tz)
                if dt.date().isoformat() == target_date:
                    temps.append(int(temp))

        return max(temps) if temps else None
    except Exception as e:
        print(f"    NWS error: {e}")
        return None


def main() -> int:
    parser = argparse.ArgumentParser(description="Collect multi-model forecasts for tracking")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--db", default=str(DEFAULT_DB))
    parser.add_argument("--target-date", default="", help="Target date YYYY-MM-DD (default: today)")
    args = parser.parse_args()

    cities = load_cities(Path(args.config))
    db_path = Path(args.db)
    db_lib.ensure_schema(db_path)

    # Create sessions
    om_session = om_lib.make_session()
    nws_session = nws_lib.make_session(
        user_agent=os.getenv("NWS_USER_AGENT", "weather-model-tracker/1.0")
    )

    now_utc = datetime.now(ZoneInfo("UTC"))
    snapshot_time = now_utc.isoformat()

    wrote = 0
    for city in cities:
        key = city["key"]
        lat = city["lat"]
        lon = city["lon"]
        tz = city.get("tz", "America/New_York")

        now_local = now_utc.astimezone(ZoneInfo(tz))
        target_date = args.target_date or now_local.date().isoformat()
        snapshot_hour = now_local.hour

        print(f"\n{key} ({city.get('label', key)}) - {target_date}:")

        # Collect Open-Meteo models
        for model in OPEN_METEO_MODELS:
            source = f"open_meteo_{model}"
            try:
                high_f = collect_open_meteo_forecast(
                    om_session, lat, lon, target_date, tz, model
                )
                if high_f is not None:
                    db_lib.upsert_forecast_snapshot(
                        db_path,
                        city_key=key,
                        target_date_local=target_date,
                        snapshot_time_utc=snapshot_time,
                        snapshot_hour_local=snapshot_hour,
                        snapshot_tz=tz,
                        forecast_high_f=high_f,
                        source=source,
                        points_url=f"open_meteo:{lat},{lon}",
                        forecast_url=f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&models={model}",
                        qc_flags=[],
                        raw={"model": model},
                    )
                    print(f"  {model.upper()}: {high_f}F")
                    wrote += 1
                else:
                    print(f"  {model.upper()}: no data")
            except Exception as e:
                print(f"  {model.upper()}: error - {e}")

        # Collect NWS hourly max
        try:
            nws_high = collect_nws_hourly_max(nws_session, lat, lon, target_date, tz)
            if nws_high is not None:
                db_lib.upsert_forecast_snapshot(
                    db_path,
                    city_key=key,
                    target_date_local=target_date,
                    snapshot_time_utc=snapshot_time,
                    snapshot_hour_local=snapshot_hour,
                    snapshot_tz=tz,
                    forecast_high_f=nws_high,
                    source="nws_hourly_max",
                    points_url=f"https://api.weather.gov/points/{lat},{lon}",
                    forecast_url="",
                    qc_flags=[],
                    raw={},
                )
                print(f"  NWS: {nws_high}F")
                wrote += 1
            else:
                print(f"  NWS: no data")
            nws_lib.polite_sleep()
        except Exception as e:
            print(f"  NWS: error - {e}")

    print(f"\n[done] Wrote {wrote} forecast snapshots")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
