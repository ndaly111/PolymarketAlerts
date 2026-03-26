#!/usr/bin/env python3
"""
Ingest raw IEM NWS MOS archive CSVs into weather_forecast_accuracy.db as forecast_snapshots.

These MOS archive downloads contain many hourly rows. The daily high is published via the
MOS N/X element (here named `n_x` in the IEM CSV output). Empirically:
  - rows where ftime is 00:00Z and n_x is present represent the forecast DAILY HIGH ("X")
  - rows where ftime is 12:00Z and n_x is present represent the forecast DAILY LOW ("N")

We ingest ONLY the daily-high rows (ftime @ 00:00Z with n_x present).

Why:
  - This backfills forecast_snapshots so error joins (observed_cli ⟷ forecast_snapshots)
    work unchanged, enabling monthly error models and trailing-N fallback.

Usage:
  python weather/scripts/ingest_mos_archive_csvs.py \
    --files weather/data/forecast_archive/kmdw_*.csv weather/data/forecast_archive/kmia_*.csv \
    --source mos_gfs_18z_archive \
    --snapshot-hour 18
"""

from __future__ import annotations

import argparse
import csv
import glob
import os
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple
from zoneinfo import ZoneInfo

import yaml

from weather.lib import db as db_lib


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DB = Path(os.getenv("WEATHER_DB_PATH", str(ROOT / "weather" / "data" / "weather_forecast_accuracy.db")))
DEFAULT_CONFIG = ROOT / "weather" / "config" / "cities.yml"
DEFAULT_GLOB = str(ROOT / "weather" / "data" / "forecast_archive" / "*.csv")


# Station code in the MOS CSV -> city_key in your pipeline
# NOTE: NYC settlement (NHIGH) uses the NYC CLI; your MOS download here is KNYC (Central Park).
STATION_TO_CITY_KEY: Dict[str, str] = {
    # Original 4 cities
    "KMDW": "CHIHIGH",
    "KMIA": "MIAHIGH",
    "KLAX": "LAXHIGH",
    "KNYC": "NHIGH",
    # New 8 cities
    "KDEN": "DENHIGH",
    "KAUS": "AUSHIGH",
    "KSFO": "SFOHIGH",
    "KPHL": "PHLHIGH",
    "KSEA": "SEAHIGH",
    "KDCA": "DCHIGH",
    "KLAS": "LVHIGH",
    "KMSY": "NOLHIGH",
}


@dataclass(frozen=True)
class CityMeta:
    key: str
    tz: str


def _parse_dt_utc(s: str) -> datetime:
    # IEM CSV exports are naive timestamps; interpret as UTC.
    # Format: "YYYY-MM-DD HH:MM:SS"
    return datetime.strptime(s.strip(), "%Y-%m-%d %H:%M:%S").replace(tzinfo=ZoneInfo("UTC"))


def _parse_int_like(v: Optional[str]) -> Optional[int]:
    if v is None:
        return None
    s = str(v).strip()
    if not s:
        return None
    try:
        return int(round(float(s)))
    except Exception:
        return None


def _load_city_meta(config_path: Path) -> Dict[str, CityMeta]:
    data = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    out: Dict[str, CityMeta] = {}
    for row in data.get("cities", []):
        key = str(row["key"]).strip()
        tz = str(row.get("tz", "America/New_York")).strip()
        out[key] = CityMeta(key=key, tz=tz)
    return out


def _iter_files(patterns: List[str]) -> List[Path]:
    files: List[Path] = []
    for p in patterns:
        # allow explicit file paths or globs
        if any(ch in p for ch in ["*", "?", "["]):
            files.extend(Path(x) for x in glob.glob(p))
        else:
            files.append(Path(p))
    # de-dupe + stable ordering
    uniq = sorted({f.resolve() for f in files if f.exists()})
    return uniq


def _extract_daily_high_rows(
    csv_path: Path,
    *,
    model_filter: str,
    runtime_hour_utc: Optional[int],
) -> Iterable[Tuple[str, datetime, datetime, int]]:
    """
    Yields tuples: (station, runtime_utc, ftime_utc, forecast_high_f)
    """
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            station = (row.get("station") or "").strip().upper()
            if not station:
                continue

            model = (row.get("model") or "").strip()
            if model_filter and model != model_filter:
                continue

            n_x = _parse_int_like(row.get("n_x"))
            if n_x is None:
                continue

            try:
                runtime_utc = _parse_dt_utc(row["runtime"])
                ftime_utc = _parse_dt_utc(row["ftime"])
            except Exception:
                continue

            # Enforce a single MOS cycle (e.g., 18Z) so we don't overwrite the same day
            if runtime_hour_utc is not None and int(runtime_utc.hour) != int(runtime_hour_utc):
                continue

            # Daily HIGH: ftime at 00Z with n_x present
            if not (ftime_utc.hour == 0 and ftime_utc.minute == 0):
                continue

            yield (station, runtime_utc, ftime_utc, int(n_x))


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--db", default=str(DEFAULT_DB))
    p.add_argument("--config", default=str(DEFAULT_CONFIG))
    p.add_argument(
        "--files",
        nargs="*",
        default=[DEFAULT_GLOB],
        help="One or more file paths/globs for MOS archive CSVs.",
    )
    p.add_argument("--source", default="mos_gfs_18z_archive", help="forecast_snapshots.source label")
    p.add_argument(
        "--snapshot-hour",
        type=int,
        default=18,
        help="Stored snapshot_hour_local value used to query this dataset later (keep consistent).",
    )
    p.add_argument(
        "--model",
        default="GFS",
        help="MOS model filter (default: GFS). Set empty to disable filtering.",
    )
    p.add_argument(
        "--runtime-hour-utc",
        type=int,
        default=18,
        help="Only ingest rows from this MOS cycle hour in UTC (default: 18). Set to -1 to disable.",
    )
    args = p.parse_args()

    db_path = Path(args.db)
    config_path = Path(args.config)
    city_meta = _load_city_meta(config_path)

    files = _iter_files(list(args.files))
    if not files:
        raise SystemExit(f"No files found for patterns: {args.files}")

    db_lib.ensure_schema(db_path)

    wrote = 0
    skipped_station = 0
    skipped_city_meta = 0
    wrote_by_city = Counter()

    runtime_hour_utc: Optional[int]
    if int(args.runtime_hour_utc) < 0:
        runtime_hour_utc = None
    else:
        runtime_hour_utc = int(args.runtime_hour_utc)

    for csv_path in files:
        for station, runtime_utc, ftime_utc, high_f in _extract_daily_high_rows(
            csv_path,
            model_filter=str(args.model or ""),
            runtime_hour_utc=runtime_hour_utc,
        ):
            city_key = STATION_TO_CITY_KEY.get(station)
            if not city_key:
                skipped_station += 1
                continue
            meta = city_meta.get(city_key)
            if not meta:
                skipped_city_meta += 1
                continue

            # Convert ftime UTC to the city's local date for correct joins with observed_cli.date_local
            tz = ZoneInfo(meta.tz)
            target_date_local = ftime_utc.astimezone(tz).date().isoformat()

            db_lib.upsert_forecast_snapshot(
                db_path,
                city_key=city_key,
                target_date_local=target_date_local,
                snapshot_time_utc=runtime_utc.isoformat(),
                snapshot_hour_local=int(args.snapshot_hour),
                snapshot_tz="UTC",
                forecast_high_f=int(high_f),
                source=str(args.source),
                points_url="https://mesonet.agron.iastate.edu/mos/fe.phtml",
                forecast_url="https://mesonet.agron.iastate.edu/mos/fe.phtml",
                qc_flags=["mos_archive", "csv_ingest", f"station:{station}"],
                raw={
                    "ingest": "weather/scripts/ingest_mos_archive_csvs.py",
                    "csv_path": str(csv_path),
                    "station": station,
                    "model": str(args.model or ""),
                    "runtime_utc": runtime_utc.isoformat(),
                    "ftime_utc": ftime_utc.isoformat(),
                    "note": "daily_high derived from MOS n_x @ ftime 00Z",
                },
            )
            wrote += 1
            wrote_by_city[city_key] += 1

    print(
        f"[done] upserted {wrote} MOS daily-high forecast snapshots into {db_path} "
        f"(skipped_station={skipped_station}, skipped_city_meta={skipped_city_meta})"
    )
    if wrote_by_city:
        for k in sorted(wrote_by_city.keys()):
            print(f"  - {k}: {wrote_by_city[k]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
