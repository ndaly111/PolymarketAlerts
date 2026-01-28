#!/usr/bin/env python3
"""
Part 1 / Step 1:
Collect trade-time forecast snapshots for each city.

Definition of "forecast high":
  - Use NWS forecastHourly endpoint
  - Compute the maximum hourly temperature for the target local calendar date (today) in the city's timezone

Why hourly-max?
  - Robust to daily-period naming quirks ("Today", "This Afternoon", etc.)
  - Always yields a deterministic per-day max if hourly periods exist

Gating:
  - Script may run multiple times per day (DST-safe schedule).
  - For each city, we only write when the city's local time falls within the
    snapshot window:
      * exact snapshot hour with minute <= SNAPSHOT_MINUTE_MAX, OR
      * within WEATHER_SNAPSHOT_LATE_WINDOW_HOURS after the snapshot hour.
  - Existing snapshots are preserved unless --overwrite is provided.
"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from zoneinfo import ZoneInfo

import yaml

from weather.lib import db as db_lib
from weather.lib import nws as nws_lib
from weather.lib import open_meteo as om_lib


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


def _now_in_tz(tz_name: str) -> datetime:
    return _now_utc().astimezone(ZoneInfo(tz_name))


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
    if not cities:
        raise ValueError(f"No cities found in {config_path}")
    return cities


def gate_city(
    now_local: datetime,
    snapshot_hour_local: int,
    minute_max: int,
    late_window_hours: int,
) -> bool:
    if now_local.hour == snapshot_hour_local:
        return now_local.minute <= minute_max
    if late_window_hours <= 0:
        return False
    snapshot_start = now_local.replace(
        hour=snapshot_hour_local,
        minute=0,
        second=0,
        microsecond=0,
    )
    if now_local < snapshot_start:
        return False
    delta_hours = (now_local - snapshot_start).total_seconds() / 3600.0
    return delta_hours <= late_window_hours


def compute_hourly_max_for_local_date(
    hourly_json: Dict[str, Any],
    tz_name: str,
    target_date_local: str,
) -> Tuple[Optional[int], List[str]]:
    qc: List[str] = []
    props = (hourly_json or {}).get("properties") or {}
    periods = props.get("periods") or []
    if not periods:
        qc.append("NO_HOURLY_PERIODS")
        return None, qc

    tz = ZoneInfo(tz_name)
    temps: List[int] = []
    for p in periods:
        st = p.get("startTime")
        temp = p.get("temperature")
        if st is None or temp is None:
            continue
        try:
            dt = datetime.fromisoformat(st)
            dt_local = dt.astimezone(tz)
            if dt_local.date().isoformat() != target_date_local:
                continue
            temps.append(int(temp))
        except Exception:
            continue

    if not temps:
        qc.append("NO_TEMPS_FOR_TARGET_DATE")
        return None, qc

    return max(temps), qc


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default=str(DEFAULT_CONFIG))
    p.add_argument("--db", default=str(DEFAULT_DB))
    p.add_argument("--no-gate", action="store_true", help="Ignore time gate (for local testing).")
    p.add_argument("--overwrite", action="store_true", help="Overwrite existing snapshots.")
    p.add_argument(
        "--source",
        default="",
        help="Optional override for forecast source (defaults to FORECAST_SOURCE env).",
    )
    p.add_argument(
        "--target-date",
        default="",
        help="Optional target date_local YYYY-MM-DD (default: today's local date per city).",
    )
    p.add_argument(
        "--use-local-hour",
        action="store_true",
        help="Use each city's current local hour as snapshot_hour_local (fallback mode).",
    )
    p.add_argument(
        "--skip-if-date-exists",
        action="store_true",
        help="Skip writing if any snapshot exists for the city/date/source (any hour).",
    )
    args = p.parse_args()

    # Default to open_meteo for better accuracy, fall back to nws_hourly_max
    forecast_source = (
        args.source.strip() or os.getenv("FORECAST_SOURCE", "open_meteo").strip() or "open_meteo"
    )
    # Model for Open-Meteo: best_match (auto), hrrr (best 0-48h US), gfs, ecmwf
    open_meteo_model = os.getenv("OPEN_METEO_MODEL", "best_match").strip()
    snapshot_hour_local = int(os.getenv("WEATHER_SNAPSHOT_HOUR_LOCAL", "6"))
    minute_max = int(os.getenv("WEATHER_SNAPSHOT_MINUTE_MAX", "25"))
    late_window_hours = int(os.getenv("WEATHER_SNAPSHOT_LATE_WINDOW_HOURS", "0"))

    user_agent = os.getenv("NWS_USER_AGENT", "").strip()
    if not user_agent:
        user_agent = "weather-edge-bot/0.1 (contact: you@example.com)"
        if forecast_source.startswith("nws"):
            print("[warn] NWS_USER_AGENT not set; using placeholder.", file=sys.stderr)

    cities = load_cities(Path(args.config))
    db_path = Path(args.db)
    db_lib.ensure_schema(db_path)

    # Create appropriate session based on forecast source
    if forecast_source.startswith("open_meteo"):
        s = om_lib.make_session()
        print(f"[info] Using Open-Meteo forecast source (model={open_meteo_model})")
    else:
        s = nws_lib.make_session(user_agent=user_agent)
        print(f"[info] Using NWS forecast source")
    fetched_at_utc = _now_utc().isoformat()

    wrote = 0
    for c in cities:
        now_local = _now_in_tz(c.tz)
        snapshot_hour_for_city = now_local.hour if args.use_local_hour else snapshot_hour_local
        if (not args.no_gate) and (
            not gate_city(now_local, snapshot_hour_for_city, minute_max, late_window_hours)
        ):
            print(
                "[skip] "
                f"{c.key} {c.label}: local={now_local.isoformat()} "
                "not in gate hour="
                f"{snapshot_hour_for_city}<=:{minute_max} "
                f"(late_window_hours={late_window_hours})"
            )
            continue

        target_date_local = args.target_date.strip() or now_local.date().isoformat()
        if args.skip_if_date_exists and db_lib.forecast_snapshot_exists(
            db_path,
            city_key=c.key,
            target_date_local=target_date_local,
            source=forecast_source,
        ):
            print(
                f"[skip] {c.key} {c.label}: snapshot exists for {target_date_local} source={forecast_source}"
            )
            continue

        qc: List[str] = []
        try:
            if not args.overwrite:
                existing = db_lib.fetch_forecast_snapshot(
                    db_path,
                    city_key=c.key,
                    target_date_local=target_date_local,
                    snapshot_hour_local=snapshot_hour_for_city,
                    source=forecast_source,
                )
                if existing:
                    print(
                        f"[skip] {c.key} {c.label}: snapshot exists for {target_date_local} hour={snapshot_hour_for_city}"
                    )
                    continue

            # Fetch forecast based on source
            if forecast_source.startswith("open_meteo"):
                # Use Open-Meteo API
                forecast = om_lib.get_daily_high_forecast(
                    s, c.lat, c.lon, target_date_local, c.tz, open_meteo_model
                )
                if forecast is None:
                    raise RuntimeError(f"Open-Meteo returned no forecast for {target_date_local}")

                fcst_high = forecast.high_f
                raw_trimmed = {
                    "target_date_local": target_date_local,
                    "city_key": c.key,
                    "model": forecast.model,
                    "source": "open_meteo",
                }
                points_url = f"open_meteo:{c.lat},{c.lon}"
                forecast_url = f"https://api.open-meteo.com/v1/forecast?latitude={c.lat}&longitude={c.lon}"

            else:
                # Use NWS API
                endpoints = nws_lib.get_points_endpoints(s, c.lat, c.lon)
                # Use city TZ from config for labeling; NWS also returns a tz we could compare
                if endpoints.time_zone and endpoints.time_zone != c.tz:
                    qc.append(f"TZ_MISMATCH_NWS:{endpoints.time_zone}")

                hourly = nws_lib.fetch_json(s, endpoints.forecast_hourly_url)
                fcst_high, qc2 = compute_hourly_max_for_local_date(hourly, c.tz, target_date_local)
                qc.extend(qc2)
                if fcst_high is None:
                    raise RuntimeError(f"Could not compute hourly max for {target_date_local}")

                # Store only a trimmed payload to keep DB smaller
                raw_trimmed = {
                    "target_date_local": target_date_local,
                    "city_key": c.key,
                    "points": endpoints.points_url,
                    "forecastHourly": endpoints.forecast_hourly_url,
                    "properties": {
                        "generatedAt": (hourly.get("properties") or {}).get("generatedAt"),
                        "updateTime": (hourly.get("properties") or {}).get("updateTime"),
                    },
                }
                points_url = endpoints.points_url
                forecast_url = endpoints.forecast_hourly_url

            db_lib.upsert_forecast_snapshot(
                db_path,
                city_key=c.key,
                target_date_local=target_date_local,
                snapshot_time_utc=fetched_at_utc,
                snapshot_hour_local=snapshot_hour_for_city,
                snapshot_tz=c.tz,
                forecast_high_f=int(fcst_high),
                source=forecast_source,
                points_url=points_url,
                forecast_url=forecast_url,
                qc_flags=qc,
                raw=raw_trimmed,
            )
            wrote += 1
            print(
                f"[ok] {c.key} {c.label}: {target_date_local} "
                f"forecast_high={fcst_high}F (hour={snapshot_hour_for_city}, source={forecast_source})"
            )
            if forecast_source.startswith("nws"):
                nws_lib.polite_sleep()
        except Exception as e:
            print(f"[err] {c.key} {c.label}: {e}", file=sys.stderr)

    if wrote == 0:
        print("[done] No snapshots written (likely due to gating).")
        return 0
    print(f"[done] wrote {wrote}/{len(cities)} forecast snapshots into {db_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
