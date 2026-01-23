#!/usr/bin/env python3
"""
Sanity-check that forecast snapshots exist for a given date/source/hour.

Why:
  The daily scan assumes the forecast snapshot workflow ran earlier and wrote
  one fresh snapshot per city. If that upstream job fails (NWS outage, rate-limit,
  etc.), pricing can silently run on stale/missing data.

This script fails the run (non-zero exit) when coverage is below a threshold.

Usage:
  python weather/scripts/check_forecast_snapshot_freshness.py \
    --forecast-source nws_hourly_max \
    --snapshot-hour 4 \
    --target-date 2026-01-23 \
    --min-pct 0.8
"""

from __future__ import annotations

import argparse
import sqlite3
import sys
from pathlib import Path
from typing import List, Set

import yaml


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DB = ROOT / "weather" / "data" / "weather.db"
DEFAULT_CONFIG = ROOT / "weather" / "config" / "cities.yml"


def _load_city_keys(config_path: Path) -> List[str]:
    data = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    keys: List[str] = []
    for row in data.get("cities", []):
        k = str(row.get("key", "")).strip()
        if k:
            keys.append(k)
    return keys


def _connect(db_path: Path) -> sqlite3.Connection:
    return sqlite3.connect(str(db_path))


def _fetch_snapshot_city_keys(
    conn: sqlite3.Connection,
    *,
    target_date_local: str,
    snapshot_hour_local: int,
    source: str,
) -> Set[str]:
    rows = conn.execute(
        """
        SELECT DISTINCT city_key
        FROM forecast_snapshots
        WHERE target_date_local = ?
          AND snapshot_hour_local = ?
          AND source = ?;
        """,
        (target_date_local, int(snapshot_hour_local), str(source)),
    ).fetchall()
    return {str(r[0]).strip() for r in rows if r and r[0]}


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--db", default=str(DEFAULT_DB))
    p.add_argument("--config", default=str(DEFAULT_CONFIG))
    p.add_argument("--forecast-source", required=True)
    p.add_argument("--snapshot-hour", type=int, required=True)
    p.add_argument("--target-date", required=True, help="YYYY-MM-DD (date_local used in DB)")
    p.add_argument("--min-pct", type=float, default=0.80, help="Minimum percent of cities with snapshots")
    p.add_argument("--min-count", type=int, default=1, help="Minimum absolute count of snapshots")
    args = p.parse_args()

    db_path = Path(args.db)
    config_path = Path(args.config)
    expected_keys = _load_city_keys(config_path)
    if not expected_keys:
        print(f"[freshness] No cities found in config: {config_path}", file=sys.stderr)
        return 2

    with _connect(db_path) as conn:
        have_keys = _fetch_snapshot_city_keys(
            conn,
            target_date_local=str(args.target_date),
            snapshot_hour_local=int(args.snapshot_hour),
            source=str(args.forecast_source),
        )

    expected_set = set(expected_keys)
    missing = sorted(expected_set - have_keys)
    have_n = len(have_keys)
    expected_n = len(expected_set)
    pct = have_n / expected_n if expected_n else 0.0

    print(
        f"[freshness] date={args.target_date} source={args.forecast_source} hour={args.snapshot_hour} "
        f"have={have_n}/{expected_n} ({pct:.1%})"
    )

    # Print a small missing list (keep logs readable)
    if missing:
        preview = missing[:25]
        more = "" if len(missing) <= 25 else f" (+{len(missing) - 25} more)"
        print(f"[freshness] missing city_keys: {', '.join(preview)}{more}")

    if have_n < int(args.min_count):
        print(
            f"[freshness] FAIL: only {have_n} snapshots found (< min_count={args.min_count}).",
            file=sys.stderr,
        )
        return 3

    if pct < float(args.min_pct):
        print(
            f"[freshness] FAIL: coverage {pct:.1%} < min_pct={float(args.min_pct):.1%}.",
            file=sys.stderr,
        )
        return 4

    print("[freshness] OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
