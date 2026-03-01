#!/usr/bin/env python3
"""
Collect upper-air features from Open-Meteo pressure-level API.

Fetches 500mb geopotential height (z500), 850mb temperature (t850), and
surface dewpoint (dewpoint_2m) for each city. These are daily averages of
the 12pm-3pm local values (peak heating hours).

Data is stored in the upper_air_snapshots table.

Usage:
    # Collect today's data for all cities
    python -m weather.scripts.collect_upper_air

    # Collect for a specific date
    python -m weather.scripts.collect_upper_air --date 2025-01-15

    # Collect for a specific city
    python -m weather.scripts.collect_upper_air --city CHIHIGH

    # Backfill historical data (2023-06-01 to today)
    python -m weather.scripts.collect_upper_air --backfill

    # Backfill from a specific start date
    python -m weather.scripts.collect_upper_air --backfill --start 2024-01-01
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import requests
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from weather.lib import db as db_lib

DEFAULT_CONFIG = ROOT / "weather" / "config" / "cities.yml"
DEFAULT_DB = Path(os.getenv(
    "WEATHER_DB_PATH",
    str(ROOT / "weather" / "data" / "weather_forecast_accuracy.db")
))

# Open-Meteo API endpoints
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"

# Peak heating hours (local): average over 12pm-3pm for max relevance to afternoon high
PEAK_HOURS = {12, 13, 14, 15}


def make_session() -> requests.Session:
    s = requests.Session()
    s.headers.update({"User-Agent": "weather-trading-bot/1.0"})
    adapter = requests.adapters.HTTPAdapter(max_retries=3)
    s.mount("https://", adapter)
    return s


def fetch_upper_air_range(
    session: requests.Session,
    lat: float,
    lon: float,
    timezone: str,
    start_date: str,
    end_date: str,
    use_archive: bool = True,
) -> Dict[str, Dict[str, Optional[float]]]:
    """
    Fetch upper-air variables for a date range from Open-Meteo.

    Notes:
    - Forecast API supports z500, t850, and dewpoint for current/future dates.
    - Archive API only supports dewpoint (z500/t850 return None for historical data).
    - z500 and t850 from forecast API are available for roughly the past 1-7 days.

    Returns dict of {date_str: {z500_m, t850_c, dewpoint_2m_c}}.
    """
    base_url = ARCHIVE_URL if use_archive else FORECAST_URL

    params = {
        "latitude": lat,
        "longitude": lon,
        "timezone": timezone,
        "hourly": "geopotential_height_500hPa,temperature_850hPa,dew_point_2m",
        "start_date": start_date,
        "end_date": end_date,
    }

    try:
        r = session.get(base_url, params=params, timeout=60)
        r.raise_for_status()
        data = r.json()
    except Exception as e:
        print(f"    [warn] Open-Meteo upper-air error: {e}")
        return {}

    hourly = data.get("hourly", {})
    times = hourly.get("time", [])
    z500_vals = hourly.get("geopotential_height_500hPa", [])
    t850_vals = hourly.get("temperature_850hPa", [])
    dew_vals = hourly.get("dew_point_2m", [])

    if not times:
        return {}

    # Group values by date and average over peak heating hours
    by_date: Dict[str, Dict[str, List[float]]] = {}

    for i, t_str in enumerate(times):
        try:
            date_part = t_str[:10]
            hour = int(t_str[11:13])
        except (ValueError, IndexError):
            continue

        if date_part not in by_date:
            by_date[date_part] = {"z500": [], "t850": [], "dew": []}

        if hour in PEAK_HOURS:
            if i < len(z500_vals) and z500_vals[i] is not None:
                by_date[date_part]["z500"].append(z500_vals[i])
            if i < len(t850_vals) and t850_vals[i] is not None:
                by_date[date_part]["t850"].append(t850_vals[i])
            if i < len(dew_vals) and dew_vals[i] is not None:
                by_date[date_part]["dew"].append(dew_vals[i])

    # Also collect all-hours data as fallback
    all_by_date: Dict[str, Dict[str, List[float]]] = {}
    for i, t_str in enumerate(times):
        try:
            date_part = t_str[:10]
        except (ValueError, IndexError):
            continue
        if date_part not in all_by_date:
            all_by_date[date_part] = {"z500": [], "t850": [], "dew": []}
        if i < len(z500_vals) and z500_vals[i] is not None:
            all_by_date[date_part]["z500"].append(z500_vals[i])
        if i < len(t850_vals) and t850_vals[i] is not None:
            all_by_date[date_part]["t850"].append(t850_vals[i])
        if i < len(dew_vals) and dew_vals[i] is not None:
            all_by_date[date_part]["dew"].append(dew_vals[i])

    results: Dict[str, Dict[str, Optional[float]]] = {}
    for date_part in by_date:
        peak = by_date[date_part]
        # Fall back to all-hours average if no peak-hour data
        fallback = all_by_date.get(date_part, {"z500": [], "t850": [], "dew": []})

        z500_list = peak["z500"] or fallback["z500"]
        t850_list = peak["t850"] or fallback["t850"]
        dew_list = peak["dew"] or fallback["dew"]

        results[date_part] = {
            "z500_m": sum(z500_list) / len(z500_list) if z500_list else None,
            "t850_c": sum(t850_list) / len(t850_list) if t850_list else None,
            "dewpoint_2m_c": sum(dew_list) / len(dew_list) if dew_list else None,
        }

    return results


def collect_for_city_date(
    session: requests.Session,
    city: Dict[str, Any],
    date_str: str,
    db_path: Path,
    is_historical: bool = False,
    dry_run: bool = False,
    verbose: bool = False,
) -> bool:
    """Collect and store upper-air data for one city/date (single-date mode)."""
    city_key = city["key"]
    lat = city.get("lat")
    lon = city.get("lon")
    tz = city.get("tz", "America/New_York")

    if lat is None or lon is None:
        if verbose:
            print(f"  [{city_key}] No lat/lon — skipping")
        return False

    if verbose:
        print(f"  [{city_key}] Fetching upper-air for {date_str}...")

    results = fetch_upper_air_range(
        session, lat, lon, tz, date_str, date_str, use_archive=is_historical
    )
    result = results.get(date_str)
    if result is None:
        print(f"  [{city_key}] No data returned for {date_str}")
        return False

    if verbose:
        print(
            f"  [{city_key}] z500={result['z500_m']:.0f}m  "
            f"t850={result['t850_c']:.1f}°C  "
            f"dew={result['dewpoint_2m_c']:.1f}°C"
        )

    if dry_run:
        print(f"  [{city_key}] [dry-run] Would save upper-air snapshot")
        return True

    try:
        db_lib.upsert_upper_air_snapshot(
            db_path,
            city_key=city_key,
            date_local=date_str,
            snapshot_time_utc=datetime.utcnow().isoformat(),
            z500_m=result["z500_m"],
            t850_c=result["t850_c"],
            dewpoint_2m_c=result["dewpoint_2m_c"],
            source="open_meteo",
        )
        return True
    except Exception as e:
        print(f"  [{city_key}] DB error: {e}")
        return False


def backfill_city(
    session: requests.Session,
    city: Dict[str, Any],
    start_date: str,
    end_date: str,
    db_path: Path,
    dry_run: bool = False,
    verbose: bool = False,
) -> int:
    """
    Backfill all dates for a city in one API call (efficient batch mode).

    Open-Meteo archive API supports arbitrary date ranges in a single request.
    We chunk into 1-year intervals to stay within API limits.

    Returns count of successfully stored dates.
    """
    city_key = city["key"]
    lat = city.get("lat")
    lon = city.get("lon")
    tz = city.get("tz", "America/New_York")

    if lat is None or lon is None:
        print(f"  [{city_key}] No lat/lon — skipping")
        return 0

    # Chunk into 1-year segments to avoid API timeout
    chunk_days = 365
    current_start = datetime.strptime(start_date, "%Y-%m-%d")
    end_dt = datetime.strptime(end_date, "%Y-%m-%d")

    total = 0
    while current_start <= end_dt:
        chunk_end = min(current_start + timedelta(days=chunk_days - 1), end_dt)
        chunk_start_str = current_start.strftime("%Y-%m-%d")
        chunk_end_str = chunk_end.strftime("%Y-%m-%d")

        if verbose:
            print(f"  [{city_key}] Fetching {chunk_start_str} → {chunk_end_str}...")

        results = fetch_upper_air_range(
            session, lat, lon, tz, chunk_start_str, chunk_end_str, use_archive=True
        )

        if not results:
            print(f"  [{city_key}] No results for chunk {chunk_start_str}→{chunk_end_str}")
            current_start = chunk_end + timedelta(days=1)
            time.sleep(1)
            continue

        chunk_stored = 0
        snap_time = datetime.utcnow().isoformat()
        for date_str, result in sorted(results.items()):
            if dry_run:
                chunk_stored += 1
                continue
            try:
                db_lib.upsert_upper_air_snapshot(
                    db_path,
                    city_key=city_key,
                    date_local=date_str,
                    snapshot_time_utc=snap_time,
                    z500_m=result.get("z500_m"),
                    t850_c=result.get("t850_c"),
                    dewpoint_2m_c=result.get("dewpoint_2m_c"),
                    source="open_meteo",
                )
                chunk_stored += 1
            except Exception as e:
                print(f"  [{city_key}] DB error for {date_str}: {e}")

        total += chunk_stored
        current_start = chunk_end + timedelta(days=1)
        time.sleep(0.5)  # Rate limit between chunks

    return total


def load_cities(config_path: Path) -> List[Dict[str, Any]]:
    data = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    return data.get("cities", [])


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Collect upper-air features (z500, t850, dewpoint) from Open-Meteo."
    )
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--db", default=str(DEFAULT_DB))
    parser.add_argument("--date", help="Date to collect (YYYY-MM-DD, default: today)")
    parser.add_argument("--city", help="Specific city key (default: all)")
    parser.add_argument(
        "--backfill",
        action="store_true",
        help="Backfill from --start to --end (uses archive API with batch requests)",
    )
    parser.add_argument(
        "--start",
        default="2023-06-01",
        help="Backfill start date (default: 2023-06-01)",
    )
    parser.add_argument(
        "--end",
        default="",
        help="Backfill end date (default: yesterday)",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    config_path = Path(args.config)
    db_path = Path(args.db)
    db_lib.ensure_schema(db_path)

    cities = load_cities(config_path)
    if args.city:
        cities = [c for c in cities if c["key"] == args.city.upper()]
        if not cities:
            print(f"[error] City not found: {args.city}", file=sys.stderr)
            return 1

    session = make_session()

    if args.backfill:
        start_date = args.start
        end_date = args.end or (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")
        print(f"Upper-Air Backfill: {start_date} → {end_date}, {len(cities)} cities")
        print(f"(Batch mode: one API call per year per city)")

        total = 0
        for city in cities:
            print(f"\n  {city['key']}...")
            city_count = backfill_city(
                session, city, start_date, end_date, db_path,
                dry_run=args.dry_run, verbose=args.verbose,
            )
            print(f"  {city['key']}: {city_count} dates stored")
            total += city_count

        print(f"\nBackfill complete: {total} total records")
        return 0

    # Single date mode
    date_str = args.date or datetime.now().strftime("%Y-%m-%d")
    today = datetime.now().strftime("%Y-%m-%d")
    is_historical = date_str < today

    print(f"Upper-Air Collection: {date_str} ({len(cities)} cities)")
    success = 0
    for city in cities:
        ok = collect_for_city_date(
            session, city, date_str, db_path,
            is_historical=is_historical, dry_run=args.dry_run, verbose=args.verbose,
        )
        if ok:
            success += 1

    print(f"Done: {success}/{len(cities)} cities")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
