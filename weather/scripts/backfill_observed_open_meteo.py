#!/usr/bin/env python3
"""
Backfill observed daily high temperatures into SQLite using Open-Meteo Archive.

Notes:
- This does NOT backfill historical forecasts (forecast_snapshots).
- Safe to re-run: upserts by (city_key, date_local).
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Dict, List, Tuple
from zoneinfo import ZoneInfo

import requests

from weather.lib import db as db_lib


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = ROOT / "weather" / "config" / "cities.yml"
DEFAULT_DB = ROOT / "weather" / "data" / "weather.db"


@dataclass(frozen=True)
class City:
    key: str
    label: str
    tz: str
    lat: float
    lon: float


def parse_date(s: str) -> date:
    return datetime.strptime(s, "%Y-%m-%d").date()


def load_cities(path: Path) -> List[City]:
    import yaml

    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    out: List[City] = []
    for row in data.get("cities", []):
        out.append(
            City(
                key=str(row["key"]).strip(),
                label=str(row.get("label", row["key"])).strip(),
                tz=str(row.get("tz", "America/New_York")).strip(),
                lat=float(row["lat"]),
                lon=float(row["lon"]),
            )
        )
    return out


def chunk_date_ranges(start: date, end: date, *, max_days: int) -> List[Tuple[str, str]]:
    ranges: List[Tuple[str, str]] = []
    cur = start
    while cur <= end:
        chunk_end = min(end, cur + timedelta(days=max_days - 1))
        ranges.append((cur.isoformat(), chunk_end.isoformat()))
        cur = chunk_end + timedelta(days=1)
    return ranges


def fetch_open_meteo_daily_max(
    city: City,
    start_date: str,
    end_date: str,
    session: requests.Session,
) -> Dict[str, float]:
    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": city.lat,
        "longitude": city.lon,
        "start_date": start_date,
        "end_date": end_date,
        "daily": "temperature_2m_max",
        "temperature_unit": "fahrenheit",
        "timezone": city.tz,
    }
    resp = session.get(url, params=params, timeout=30)
    resp.raise_for_status()
    data = resp.json()
    daily = data.get("daily") or {}
    dates = daily.get("time") or []
    tmax = daily.get("temperature_2m_max") or []
    out: Dict[str, float] = {}
    for d, v in zip(dates, tmax):
        if v is None:
            continue
        out[str(d)] = float(v)
    return out


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default=str(DEFAULT_CONFIG))
    p.add_argument("--db", default=str(DEFAULT_DB))
    p.add_argument("--city", default="", help="Optional city_key filter (blank = all)")
    p.add_argument("--start-date", required=True, help="YYYY-MM-DD")
    p.add_argument("--end-date", required=True, help="YYYY-MM-DD")
    p.add_argument("--sleep", type=float, default=0.25, help="Delay between API calls")
    p.add_argument("--max-days-per-call", type=int, default=365)
    args = p.parse_args()

    start = parse_date(args.start_date)
    end = parse_date(args.end_date)
    if end < start:
        raise SystemExit("end-date must be >= start-date")

    db_path = Path(args.db)
    db_lib.ensure_schema(db_path)

    cities = load_cities(Path(args.config))
    if args.city:
        cities = [c for c in cities if c.key == args.city]

    wrote = 0
    session = requests.Session()

    for c in cities:
        ranges = chunk_date_ranges(start, end, max_days=int(args.max_days_per_call))
        for s, e in ranges:
            mapping = fetch_open_meteo_daily_max(c, s, e, session)
            fetched_at_utc = datetime.now(tz=ZoneInfo("UTC")).isoformat()
            source_url = (
                "https://archive-api.open-meteo.com/v1/archive"
                f"?latitude={c.lat}&longitude={c.lon}"
                f"&start_date={s}&end_date={e}"
                "&daily=temperature_2m_max&temperature_unit=fahrenheit"
                f"&timezone={c.tz}"
            )

            for d_local, tmax_f in mapping.items():
                tmax_int = int(round(float(tmax_f)))
                db_lib.upsert_observed_cli(
                    db_path,
                    city_key=c.key,
                    date_local=d_local,
                    tmax_f=tmax_int,
                    fetched_at_utc=fetched_at_utc,
                    source_url=source_url,
                    version_used=0,
                    report_date_local=d_local,
                    is_preliminary=False,
                    qc_flags=[],
                    raw_text=json.dumps({"provider": "open-meteo", "tmax_f": tmax_f}),
                )
                wrote += 1

            time.sleep(float(args.sleep))

    print(f"[done] upserted {wrote} observed rows into {db_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
