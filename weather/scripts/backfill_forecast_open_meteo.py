#!/usr/bin/env python3
"""
Backfill forecast_snapshots using Open-Meteo archived (past days) forecast data.

What it does:
- Pulls hourly temperature forecast series for a date range.
- Computes each day's forecasted high (max hourly temp).
- Upserts into forecast_snapshots keyed by (city_key, target_date_local, snapshot_hour_local).

Notes / Caveats:
- This is not guaranteed to match Apple Weather / NWS issuance timing.
- It *does* give you consistent historical forecast data so you can start modeling now.
"""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Dict, List, Tuple
from zoneinfo import ZoneInfo

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

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


def fetch_open_meteo_hourly_temp(
    city: City,
    start_date: str,
    end_date: str,
    session: requests.Session,
    past_days: int,
) -> Dict[str, Dict[str, float]]:
    """
    Returns {date_local: {hour_iso_local: temp_f}}.
    """
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": city.lat,
        "longitude": city.lon,
        "start_date": start_date,
        "end_date": end_date,
        "hourly": "temperature_2m",
        "temperature_unit": "fahrenheit",
        "timezone": city.tz,
        # Critical: include past days so we can backfill now.
        # Open-Meteo describes this as archived forecasts for previous days.
        "past_days": int(past_days),
    }
    resp = session.get(url, params=params, timeout=45)
    resp.raise_for_status()
    data = resp.json()

    hourly = data.get("hourly") or {}
    times = hourly.get("time") or []
    temps = hourly.get("temperature_2m") or []

    out: Dict[str, Dict[str, float]] = {}
    for t, v in zip(times, temps):
        if v is None:
            continue
        # t is in local timezone already because we set timezone=city.tz
        d = str(t).split("T", 1)[0]
        out.setdefault(d, {})[str(t)] = float(v)
    return out


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default=str(DEFAULT_CONFIG))
    p.add_argument("--db", default=str(DEFAULT_DB))
    p.add_argument("--city", default="", help="Optional city_key filter (blank = all)")
    p.add_argument("--start-date", required=True, help="YYYY-MM-DD")
    p.add_argument("--end-date", required=True, help="YYYY-MM-DD")
    p.add_argument("--snapshot-hour-local", type=int, default=5)
    p.add_argument(
        "--past-days",
        type=int,
        default=92,
        help="How many days of archived forecast history to request from Open-Meteo.",
    )
    p.add_argument("--sleep", type=float, default=0.25, help="Delay between API calls")
    p.add_argument("--max-days-per-call", type=int, default=31)
    p.add_argument(
        "--user-agent",
        default="PolymarketAlerts Weather Backfill",
        help="User-Agent header for Open-Meteo requests.",
    )
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
    session.headers["User-Agent"] = args.user_agent
    retry = Retry(
        total=5,
        backoff_factor=0.5,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=("GET",),
    )
    session.mount("https://", HTTPAdapter(max_retries=retry))

    for c in cities:
        for s, e in chunk_date_ranges(start, end, max_days=int(args.max_days_per_call)):
            mapping = fetch_open_meteo_hourly_temp(
                c,
                s,
                e,
                session,
                past_days=int(args.past_days),
            )
            fetched_at_utc = datetime.now(tz=ZoneInfo("UTC")).isoformat()
            forecast_url = (
                "https://api.open-meteo.com/v1/forecast"
                f"?latitude={c.lat}&longitude={c.lon}"
                f"&start_date={s}&end_date={e}"
                "&hourly=temperature_2m&temperature_unit=fahrenheit"
                f"&timezone={c.tz}&past_days={int(args.past_days)}"
            )

            for d_local, hours in mapping.items():
                if d_local < s or d_local > e:
                    continue
                if not hours:
                    continue
                high_f = int(round(max(hours.values())))

                db_lib.upsert_forecast_snapshot(
                    db_path,
                    city_key=c.key,
                    target_date_local=d_local,
                    snapshot_time_utc=fetched_at_utc,
                    snapshot_hour_local=int(args.snapshot_hour_local),
                    snapshot_tz=c.tz,
                    forecast_high_f=high_f,
                    source="open-meteo-archived",
                    points_url=forecast_url,
                    forecast_url=forecast_url,
                    qc_flags=[
                        "backfilled_forecast_archived",
                        "snapshot_hour_label_only",
                    ],
                    raw={
                        "provider": "open-meteo",
                        "method": "hourly_max_temperature_2m",
                        "hours_count": len(hours),
                        "range_start_date": s,
                        "range_end_date": e,
                    },
                )
                wrote += 1

            time.sleep(float(args.sleep))

    print(f"[done] upserted {wrote} forecast snapshots into {db_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
