#!/usr/bin/env python3
"""
Collect observed daily high temperatures via NWS API.

Alternative to collect_cli_observed.py - uses NWS station observations API
instead of CLI text reports.

This collector:
  - Uses city-local "yesterday" as the target date
  - Fetches all observations for that calendar day via NWS API
  - Computes max temperature from observations
  - Upserts observed_cli rows into SQLite (same table as CLI collector)

Usage:
  python weather/scripts/collect_observed_nws_api.py
  python weather/scripts/collect_observed_nws_api.py --date 2026-01-25
"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import List, Optional
from zoneinfo import ZoneInfo

import yaml

from weather.lib import db as db_lib
from weather.lib import nws as nws_lib


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = ROOT / "weather" / "config" / "cities.yml"
DEFAULT_DB = Path(os.getenv("WEATHER_DB_PATH", str(ROOT / "weather" / "data" / "weather_forecast_accuracy.db")))


@dataclass(frozen=True)
class City:
    key: str
    label: str
    tz: str
    mos_station: str  # METAR station code (e.g., KDEN)


def _now_utc() -> datetime:
    return datetime.now(tz=ZoneInfo("UTC"))


def _yesterday_local(tz_name: str) -> date:
    now_local = _now_utc().astimezone(ZoneInfo(tz_name))
    return now_local.date() - timedelta(days=1)


def load_cities(config_path: Path) -> List[City]:
    """Load cities from config, requiring mos_station field."""
    data = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    cities: List[City] = []
    for row in data.get("cities", []):
        mos_station = row.get("mos_station", "").strip()
        if not mos_station:
            # Skip cities without MOS station (can't fetch observations)
            continue
        cities.append(
            City(
                key=str(row["key"]).strip(),
                label=str(row.get("label", row["key"])).strip(),
                tz=str(row.get("tz", "America/New_York")).strip(),
                mos_station=mos_station,
            )
        )
    if not cities:
        raise ValueError(f"No cities with mos_station found in {config_path}")
    return cities


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default=str(DEFAULT_CONFIG))
    p.add_argument("--db", default=str(DEFAULT_DB))
    p.add_argument(
        "--date",
        default="",
        help="Specific date to collect (YYYY-MM-DD). Default: yesterday (local to each city)",
    )
    args = p.parse_args()

    db_path = Path(args.db)
    db_lib.ensure_schema(db_path)

    cities = load_cities(Path(args.config))

    # Create NWS session
    user_agent = os.getenv("WEATHER_USER_AGENT", "weather-collector/1.0 (github.com/ndaly111)")
    session = nws_lib.make_session(user_agent=user_agent)

    fetched_at_utc = _now_utc().isoformat()

    wrote = 0
    errors = 0

    for city in cities:
        # Determine target date
        if args.date:
            target_date_local = args.date.strip()
        else:
            target_date_local = _yesterday_local(city.tz).isoformat()

        qc_flags: List[str] = ["source:nws_api"]

        try:
            result = nws_lib.get_daily_observed_high(
                session,
                station_id=city.mos_station,
                date_local=target_date_local,
                timezone=city.tz,
            )

            if result is None:
                print(f"[warn] {city.key} {city.label}: No observations found for {target_date_local}")
                qc_flags.append("NO_OBSERVATIONS")
                errors += 1
                continue

            # Check observation count for quality
            if result.observation_count < 20:
                qc_flags.append(f"LOW_OBS_COUNT:{result.observation_count}")

            # Build source URL for reference
            source_url = f"https://api.weather.gov/stations/{city.mos_station}/observations"

            db_lib.upsert_observed_cli(
                db_path,
                city_key=city.key,
                date_local=target_date_local,
                tmax_f=result.tmax_f,
                fetched_at_utc=fetched_at_utc,
                source_url=source_url,
                version_used=0,  # Not applicable for API
                report_date_local=target_date_local,
                is_preliminary=False,  # API observations are real-time
                qc_flags=qc_flags,
                raw_text=str(result.raw_observations[:10]),  # Store sample of observations
            )
            wrote += 1
            print(
                f"[ok] {city.key} {city.label}: {target_date_local} "
                f"tmax={result.tmax_f}F (from {result.observation_count} observations)"
            )

            # Be polite to NWS servers
            nws_lib.polite_sleep()

        except Exception as exc:
            print(f"[err] {city.key} {city.label}: {exc}", file=sys.stderr)
            errors += 1

    print()
    if wrote == 0:
        print("[done] No observed rows written.")
    else:
        print(f"[done] wrote {wrote}/{len(cities)} observed rows into {db_path}")

    if errors > 0:
        print(f"[warn] {errors} cities had errors")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
