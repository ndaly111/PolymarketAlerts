from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Tuple


@dataclass(frozen=True)
class DbPaths:
    root: Path
    db_path: Path


def default_paths(repo_root: Path) -> DbPaths:
    return DbPaths(root=repo_root, db_path=repo_root / "weather" / "data" / "weather.db")


def connect(db_path: Path) -> sqlite3.Connection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    # Avoid WAL sidecar files (.db-wal/.db-shm) since we commit this DB to git.
    conn.execute("PRAGMA journal_mode=DELETE;")
    conn.execute("PRAGMA synchronous=FULL;")
    return conn


def ensure_schema(db_path: Path) -> None:
    with connect(db_path) as conn:
        # Forecast snapshots taken at trade-time (e.g., 06:00 local)
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS forecast_snapshots (
              id INTEGER PRIMARY KEY AUTOINCREMENT,
              city_key TEXT NOT NULL,
              target_date_local TEXT NOT NULL,      -- YYYY-MM-DD
              snapshot_time_utc TEXT NOT NULL,      -- ISO
              snapshot_hour_local INTEGER NOT NULL, -- e.g. 6
              snapshot_tz TEXT NOT NULL,            -- IANA tz
              forecast_high_f INTEGER NOT NULL,     -- integer F (our chosen definition)
              source TEXT NOT NULL,                 -- e.g. "nws_hourly_max"
              points_url TEXT NOT NULL,
              forecast_url TEXT NOT NULL,
              qc_flags TEXT NOT NULL,
              raw_json TEXT NOT NULL
            );
            """
        )
        conn.execute(
            """
            CREATE UNIQUE INDEX IF NOT EXISTS idx_forecast_unique
            ON forecast_snapshots(city_key, target_date_local, snapshot_hour_local);
            """
        )

        # Observed highs from CLI (you already create this in collect_cli_observed.py)
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS observed_cli (
              id INTEGER PRIMARY KEY AUTOINCREMENT,
              city_key TEXT NOT NULL,
              date_local TEXT NOT NULL,
              tmax_f INTEGER NOT NULL,
              fetched_at_utc TEXT NOT NULL,
              source_url TEXT NOT NULL,
              version_used INTEGER NOT NULL,
              report_date_local TEXT NOT NULL,
              is_preliminary INTEGER NOT NULL,
              qc_flags TEXT NOT NULL,
              raw_text TEXT NOT NULL
            );
            """
        )
        conn.execute(
            """
            CREATE UNIQUE INDEX IF NOT EXISTS idx_observed_unique
            ON observed_cli(city_key, date_local);
            """
        )

        # Monthly error PMFs by city/month/snapshot-hour
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS error_models (
              id INTEGER PRIMARY KEY AUTOINCREMENT,
              city_key TEXT NOT NULL,
              month INTEGER NOT NULL,               -- 1..12
              snapshot_hour_local INTEGER NOT NULL, -- aligns with forecast_snapshots
              n_samples INTEGER NOT NULL,
              pmf_json TEXT NOT NULL,               -- mapping error_int->prob
              updated_at_utc TEXT NOT NULL
            );
            """
        )
        conn.execute(
            """
            CREATE UNIQUE INDEX IF NOT EXISTS idx_error_models_unique
            ON error_models(city_key, month, snapshot_hour_local);
            """
        )


def upsert_forecast_snapshot(
    db_path: Path,
    *,
    city_key: str,
    target_date_local: str,
    snapshot_time_utc: str,
    snapshot_hour_local: int,
    snapshot_tz: str,
    forecast_high_f: int,
    source: str,
    points_url: str,
    forecast_url: str,
    qc_flags: Iterable[str],
    raw: Dict[str, Any],
) -> None:
    ensure_schema(db_path)
    qc = ",".join(sorted(set(qc_flags))) if qc_flags else ""
    raw_json = json.dumps(raw, separators=(",", ":"), ensure_ascii=False)
    with connect(db_path) as conn:
        conn.execute(
            """
            INSERT INTO forecast_snapshots
              (city_key, target_date_local, snapshot_time_utc, snapshot_hour_local, snapshot_tz,
               forecast_high_f, source, points_url, forecast_url, qc_flags, raw_json)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(city_key, target_date_local, snapshot_hour_local) DO UPDATE SET
              snapshot_time_utc=excluded.snapshot_time_utc,
              snapshot_tz=excluded.snapshot_tz,
              forecast_high_f=excluded.forecast_high_f,
              source=excluded.source,
              points_url=excluded.points_url,
              forecast_url=excluded.forecast_url,
              qc_flags=excluded.qc_flags,
              raw_json=excluded.raw_json;
            """,
            (
                city_key,
                target_date_local,
                snapshot_time_utc,
                int(snapshot_hour_local),
                snapshot_tz,
                int(forecast_high_f),
                source,
                points_url,
                forecast_url,
                qc,
                raw_json,
            ),
        )


def upsert_observed_cli(
    db_path: Path,
    *,
    city_key: str,
    date_local: str,
    tmax_f: int,
    fetched_at_utc: str,
    source_url: str,
    version_used: int,
    report_date_local: str,
    is_preliminary: bool,
    qc_flags: Iterable[str],
    raw_text: str,
) -> None:
    ensure_schema(db_path)
    qc = ",".join(sorted(set(qc_flags))) if qc_flags else ""
    with connect(db_path) as conn:
        conn.execute(
            """
            INSERT INTO observed_cli
              (city_key, date_local, tmax_f, fetched_at_utc, source_url, version_used,
               report_date_local, is_preliminary, qc_flags, raw_text)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(city_key, date_local) DO UPDATE SET
              tmax_f=excluded.tmax_f,
              fetched_at_utc=excluded.fetched_at_utc,
              source_url=excluded.source_url,
              version_used=excluded.version_used,
              report_date_local=excluded.report_date_local,
              is_preliminary=excluded.is_preliminary,
              qc_flags=excluded.qc_flags,
              raw_text=excluded.raw_text;
            """,
            (
                city_key,
                date_local,
                int(tmax_f),
                fetched_at_utc,
                source_url,
                int(version_used),
                report_date_local,
                1 if is_preliminary else 0,
                qc,
                raw_text,
            ),
        )


def fetch_joined_errors(
    db_path: Path,
    *,
    city_key: str,
    month: int,
    snapshot_hour_local: int,
) -> Tuple[int, Dict[int, int]]:
    """
    Returns (n_samples, counts_by_error) for the given city/month/snapshot_hour.
    error = observed - forecast
    """
    ensure_schema(db_path)
    with connect(db_path) as conn:
        rows = conn.execute(
            """
            SELECT o.tmax_f, f.forecast_high_f
            FROM observed_cli o
            JOIN forecast_snapshots f
              ON f.city_key=o.city_key
             AND f.target_date_local=o.date_local
             AND f.snapshot_hour_local=?
            WHERE o.city_key=?
              AND CAST(strftime('%m', o.date_local) AS INTEGER)=?
            """,
            (int(snapshot_hour_local), city_key, int(month)),
        ).fetchall()

    counts: Dict[int, int] = {}
    for (obs, fcst) in rows:
        e = int(obs) - int(fcst)
        counts[e] = counts.get(e, 0) + 1
    return (len(rows), counts)


def upsert_error_model(
    db_path: Path,
    *,
    city_key: str,
    month: int,
    snapshot_hour_local: int,
    n_samples: int,
    pmf: Dict[int, float],
    updated_at_utc: str,
) -> None:
    ensure_schema(db_path)
    pmf_json = json.dumps({str(k): v for k, v in pmf.items()}, separators=(",", ":"), ensure_ascii=False)
    with connect(db_path) as conn:
        conn.execute(
            """
            INSERT INTO error_models(city_key, month, snapshot_hour_local, n_samples, pmf_json, updated_at_utc)
            VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(city_key, month, snapshot_hour_local) DO UPDATE SET
              n_samples=excluded.n_samples,
              pmf_json=excluded.pmf_json,
              updated_at_utc=excluded.updated_at_utc;
            """,
            (city_key, int(month), int(snapshot_hour_local), int(n_samples), pmf_json, updated_at_utc),
        )


def fetch_forecast_snapshot(
    db_path: Path,
    *,
    city_key: str,
    target_date_local: str,
    snapshot_hour_local: int,
) -> Optional[Dict[str, Any]]:
    """
    Returns the snapshot row (dict) for a given city/date/hour, or None if missing.
    """
    ensure_schema(db_path)
    with connect(db_path) as conn:
        row = conn.execute(
            """
            SELECT city_key, target_date_local, snapshot_time_utc, snapshot_hour_local, snapshot_tz,
                   forecast_high_f, source, points_url, forecast_url, qc_flags, raw_json
            FROM forecast_snapshots
            WHERE city_key=? AND target_date_local=? AND snapshot_hour_local=?
            """,
            (city_key, target_date_local, int(snapshot_hour_local)),
        ).fetchone()

    if not row:
        return None
    return {
        "city_key": row[0],
        "target_date_local": row[1],
        "snapshot_time_utc": row[2],
        "snapshot_hour_local": int(row[3]),
        "snapshot_tz": row[4],
        "forecast_high_f": int(row[5]),
        "source": row[6],
        "points_url": row[7],
        "forecast_url": row[8],
        "qc_flags": row[9],
        "raw": json.loads(row[10]) if row[10] else {},
    }


def fetch_error_model(
    db_path: Path,
    *,
    city_key: str,
    month: int,
    snapshot_hour_local: int,
) -> Optional[Dict[str, Any]]:
    """
    Returns {"n_samples": int, "pmf": {error_int: prob}} or None if missing.
    """
    ensure_schema(db_path)
    with connect(db_path) as conn:
        row = conn.execute(
            """
            SELECT n_samples, pmf_json, updated_at_utc
            FROM error_models
            WHERE city_key=? AND month=? AND snapshot_hour_local=?
            """,
            (city_key, int(month), int(snapshot_hour_local)),
        ).fetchone()
    if not row:
        return None
    n_samples, pmf_json, updated_at_utc = row
    pmf_raw = json.loads(pmf_json) if pmf_json else {}
    pmf = {int(k): float(v) for k, v in pmf_raw.items()}
    return {"n_samples": int(n_samples), "pmf": pmf, "updated_at_utc": updated_at_utc}
