#!/usr/bin/env python3
"""
Backfill daily weather observations (precipitation, wind speed) from Open-Meteo Archive API.

This populates the weather_observations_daily table used by the ML training pipeline
to provide features that were previously dead (precip_prob, wind_speed, precip_last_3d_mm).

Usage:
    python3 weather/scripts/backfill_weather_context.py
    python3 weather/scripts/backfill_weather_context.py --start-date 2023-01-01
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import yaml

from weather.lib import db as db_lib
from weather.lib import open_meteo as om_lib


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = ROOT / "weather" / "config" / "cities.yml"
DEFAULT_DB = Path(os.getenv(
    "WEATHER_DB_PATH",
    str(ROOT / "weather" / "data" / "weather_forecast_accuracy.db")
))


def main() -> int:
    p = argparse.ArgumentParser(description="Backfill daily weather observations from Open-Meteo Archive.")
    p.add_argument("--config", default=str(DEFAULT_CONFIG))
    p.add_argument("--db", default=str(DEFAULT_DB))
    p.add_argument("--start-date", default="2023-01-01", help="Start date for backfill (YYYY-MM-DD)")
    p.add_argument("--end-date", default="", help="End date (default: yesterday)")
    args = p.parse_args()

    db_path = Path(args.db)
    db_lib.ensure_schema(db_path)

    data = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    cities = data.get("cities", [])

    end_date = args.end_date.strip()
    if not end_date:
        from datetime import timedelta
        end_date = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")

    session = om_lib.make_session(timeout=60.0)
    now_utc = datetime.utcnow().isoformat()

    total_wrote = 0
    for city in cities:
        key = city["key"]
        lat = float(city["lat"])
        lon = float(city["lon"])
        tz = city.get("tz", "America/New_York")

        print(f"\n--- {key} ({city.get('label', key)}) ---")
        print(f"  Fetching {args.start_date} to {end_date}...")

        # Open-Meteo Archive API can handle large date ranges (up to ~50 years)
        # but we chunk by year to be safe and show progress
        start_dt = datetime.strptime(args.start_date, "%Y-%m-%d")
        end_dt = datetime.strptime(end_date, "%Y-%m-%d")

        city_rows = []
        chunk_start = start_dt
        while chunk_start <= end_dt:
            chunk_end = min(
                chunk_start.replace(year=chunk_start.year + 1, month=1, day=1),
                end_dt,
            )
            chunk_start_str = chunk_start.strftime("%Y-%m-%d")
            chunk_end_str = chunk_end.strftime("%Y-%m-%d")

            results = om_lib.fetch_historical_daily(
                session, lat, lon, chunk_start_str, chunk_end_str, timezone=tz,
            )

            for r in results:
                city_rows.append((r["date"], r["precipitation_mm"], r["windspeed_max_mph"]))

            print(f"  Fetched {len(results)} days ({chunk_start_str} to {chunk_end_str})")

            chunk_start = chunk_end.replace(year=chunk_end.year, month=1, day=1)
            if chunk_start <= chunk_end:
                from datetime import timedelta
                chunk_start = chunk_end + timedelta(days=1)

            # Rate limit: Open-Meteo free tier has generous limits but be polite
            time.sleep(0.5)

        if city_rows:
            wrote = db_lib.upsert_weather_observations_daily(
                db_path,
                city_key=key,
                rows=city_rows,
                source="open_meteo_archive",
                fetched_at_utc=now_utc,
            )
            print(f"  Wrote {wrote} observations to DB")
            total_wrote += wrote

    print(f"\n[done] Total: {total_wrote} observations backfilled")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
