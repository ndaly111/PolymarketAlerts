#!/usr/bin/env python3
"""
Collect intraday temperature observations from NWS stations.

This script:
  - Runs periodically (e.g., every 30 minutes) during trading hours
  - Fetches current temperature from the nearest NWS station for each city
  - Computes progress toward the daily high using 4am baseline
  - Stores observations in the database for PMF adjustment and backtesting

Progress formula:
  progress = (current_temp - baseline_4am) / (forecast_high - baseline_4am)

Usage:
  python -m weather.scripts.collect_intraday_observations --db weather/data/weather_forecast_accuracy.db
"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import List, Optional
from zoneinfo import ZoneInfo

import yaml

from weather.lib import db as db_lib
from weather.lib import nws as nws_lib
from weather.lib.fair import compute_progress


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = ROOT / "weather" / "config" / "cities.yml"
DEFAULT_DB = Path(os.getenv("WEATHER_DB_PATH", str(ROOT / "weather" / "data" / "weather_forecast_accuracy.db")))


@dataclass(frozen=True)
class City:
    key: str
    label: str
    tz: str
    lat: float
    lon: float


def _now_utc() -> datetime:
    return datetime.now(tz=ZoneInfo("UTC"))


def load_cities(config_path: Path) -> List[City]:
    data = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    cities: List[City] = []
    for row in data.get("cities", []):
        cities.append(
            City(
                key=str(row["key"]).strip(),
                label=str(row.get("label", row["key"])).strip(),
                tz=str(row.get("tz", "America/New_York")).strip(),
                lat=float(row["lat"]),
                lon=float(row["lon"]),
            )
        )
    return cities


def get_or_cache_station(
    s: nws_lib.requests.Session,
    db_path: Path,
    city: City,
) -> Optional[str]:
    """Get preferred station for city, caching in DB."""
    # Check cache first
    station_id = db_lib.fetch_preferred_station(db_path, city_key=city.key)
    if station_id:
        return station_id

    # Fetch from NWS and cache
    stations = nws_lib.get_nearby_stations(s, city.lat, city.lon, limit=3)
    if not stations:
        return None

    for i, st in enumerate(stations):
        db_lib.upsert_city_station(
            db_path,
            city_key=city.key,
            station_id=st.station_id,
            station_name=st.name,
            distance_km=st.distance_km,
            priority=i + 1,
            updated_at_utc=_now_utc().isoformat(),
        )

    return stations[0].station_id


def get_baseline_temp(
    db_path: Path,
    city_key: str,
    target_date_local: str,
    forecast_source: str,
) -> Optional[int]:
    """Get the 4am baseline temperature from forecast snapshots."""
    snap = db_lib.fetch_forecast_snapshot(
        db_path,
        city_key=city_key,
        target_date_local=target_date_local,
        snapshot_hour_local=4,
        source=forecast_source,
    )
    if snap:
        return int(snap["forecast_high_f"])
    return None


def get_max_observed_today(
    db_path: Path,
    city_key: str,
    target_date_local: str,
    current_temp: int,
) -> int:
    """Get running max including current observation."""
    prev_max = db_lib.fetch_max_intraday_observation(
        db_path,
        city_key=city_key,
        target_date_local=target_date_local,
    )
    if prev_max is None:
        return current_temp
    return max(prev_max, current_temp)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default=str(DEFAULT_CONFIG))
    p.add_argument("--db", default=str(DEFAULT_DB))
    p.add_argument("--forecast-source", default="nws_hourly_max")
    p.add_argument("--city", default="", help="Optional city_key filter")
    p.add_argument("--no-gate", action="store_true", help="Ignore time gate (run anytime)")
    args = p.parse_args()

    user_agent = os.getenv("NWS_USER_AGENT", "weather-edge-bot/0.1 (contact: you@example.com)")

    db_path = Path(args.db)
    db_lib.ensure_schema(db_path)

    cities = load_cities(Path(args.config))
    s = nws_lib.make_session(user_agent=user_agent)

    now_utc = _now_utc()
    wrote = 0

    for c in cities:
        if args.city and c.key != args.city:
            continue

        now_local = now_utc.astimezone(ZoneInfo(c.tz))
        target_date_local = now_local.date().isoformat()

        # Gate: only collect during daytime hours (6 AM - 8 PM local)
        if not args.no_gate and not (6 <= now_local.hour <= 20):
            print(f"[skip] {c.key}: outside collection hours ({now_local.hour}h local)")
            continue

        qc_flags: List[str] = []

        try:
            # Get station
            station_id = get_or_cache_station(s, db_path, c)
            if not station_id:
                qc_flags.append("NO_STATION")
                print(f"[err] {c.key}: no observation station found")
                continue

            # Get observation
            nws_lib.polite_sleep()
            obs = nws_lib.get_latest_observation(s, station_id)
            if not obs:
                qc_flags.append("NO_OBSERVATION")
                print(f"[err] {c.key}: no observation from {station_id}")
                continue

            current_temp = obs.temperature_f

            # Get 4am baseline temp
            baseline_temp = get_baseline_temp(db_path, c.key, target_date_local, args.forecast_source)
            if baseline_temp is None:
                qc_flags.append("NO_BASELINE")
                print(f"[warn] {c.key}: no 4am baseline for {target_date_local}, using current temp")
                baseline_temp = current_temp

            # Get current forecast high (latest snapshot)
            snap = db_lib.fetch_latest_forecast_snapshot(
                db_path,
                city_key=c.key,
                target_date_local=target_date_local,
                source=args.forecast_source,
            )
            if not snap:
                qc_flags.append("NO_FORECAST")
                print(f"[warn] {c.key}: no forecast snapshot for {target_date_local}")
                forecast_high = baseline_temp + 10  # fallback
            else:
                forecast_high = int(snap["forecast_high_f"])

            # Calculate running max
            max_observed = get_max_observed_today(db_path, c.key, target_date_local, current_temp)

            # Calculate progress using 4am baseline
            progress = compute_progress(current_temp, baseline_temp, forecast_high)

            # Store observation
            db_lib.upsert_intraday_observation(
                db_path,
                city_key=c.key,
                target_date_local=target_date_local,
                observation_time_utc=obs.timestamp_utc,
                observation_time_local=now_local.isoformat(),
                station_id=station_id,
                current_temp_f=current_temp,
                morning_temp_f=baseline_temp,
                max_observed_f=max_observed,
                forecast_high_f=forecast_high,
                progress=progress,
                qc_flags=qc_flags,
                raw=obs.raw,
            )
            wrote += 1

            print(
                f"[ok] {c.key}: current={current_temp}F baseline={baseline_temp}F "
                f"max={max_observed}F forecast={forecast_high}F progress={progress:.1%}"
            )

        except Exception as e:
            print(f"[err] {c.key}: {e}", file=sys.stderr)

    print(f"[done] wrote {wrote} intraday observations")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
