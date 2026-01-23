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
        conn.execute("DROP INDEX IF EXISTS idx_forecast_unique;")
        conn.execute(
            """
            CREATE UNIQUE INDEX IF NOT EXISTS idx_forecast_unique_v2
            ON forecast_snapshots(city_key, target_date_local, snapshot_hour_local, source);
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

        # Kalshi weather market snapshots captured at scan-time.
        # Append-only: we want historical quotes for audits/backtests.
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS kalshi_weather_market_snapshots (
              id INTEGER PRIMARY KEY AUTOINCREMENT,
              snapshot_time_utc TEXT NOT NULL,
              city_key TEXT NOT NULL,
              target_date_local TEXT NOT NULL,
              series_ticker TEXT NOT NULL,
              event_ticker TEXT NOT NULL,
              market_ticker TEXT NOT NULL,
              status TEXT NOT NULL,
              yes_bid INTEGER,
              yes_ask INTEGER,
              no_bid INTEGER,
              no_ask INTEGER,
              volume INTEGER,
              open_interest INTEGER,
              raw_json TEXT NOT NULL
            );
            """
        )
        conn.execute("DROP INDEX IF EXISTS idx_kalshi_weather_unique;")
        conn.execute(
            """
            CREATE UNIQUE INDEX IF NOT EXISTS idx_kalshi_weather_unique_v2
            ON kalshi_weather_market_snapshots(snapshot_time_utc, market_ticker);
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
              source TEXT NOT NULL,
              n_samples INTEGER NOT NULL,
              pmf_json TEXT NOT NULL,               -- mapping error_int->prob
              updated_at_utc TEXT NOT NULL
            );
            """
        )
        columns = {row[1] for row in conn.execute("PRAGMA table_info(error_models);").fetchall()}
        if "source" not in columns:
            conn.execute(
                "ALTER TABLE error_models ADD COLUMN source TEXT NOT NULL DEFAULT 'nws_hourly_max';"
            )
        conn.execute("DROP INDEX IF EXISTS idx_error_models_unique;")
        conn.execute(
            """
            CREATE UNIQUE INDEX IF NOT EXISTS idx_error_models_unique_v2
            ON error_models(city_key, month, snapshot_hour_local, source);
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
            ON CONFLICT(city_key, target_date_local, snapshot_hour_local, source) DO UPDATE SET
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


def insert_kalshi_weather_market_snapshot(
    db_path: Path,
    *,
    snapshot_time_utc: str,
    city_key: str,
    target_date_local: str,
    series_ticker: str,
    event_ticker: str,
    market_ticker: str,
    status: str,
    yes_bid: Optional[int],
    yes_ask: Optional[int],
    no_bid: Optional[int],
    no_ask: Optional[int],
    volume: Optional[int],
    open_interest: Optional[int],
    raw: Dict[str, Any],
) -> None:
    """Insert one Kalshi market snapshot row (append-only).

    Unique on (snapshot_time_utc, market_ticker) so a single run doesn't duplicate itself.
    """
    ensure_schema(db_path)
    raw_json = json.dumps(raw, separators=(",", ":"), ensure_ascii=False)
    with connect(db_path) as conn:
        conn.execute(
            """
            INSERT OR IGNORE INTO kalshi_weather_market_snapshots
              (snapshot_time_utc, city_key, target_date_local, series_ticker, event_ticker,
               market_ticker, status, yes_bid, yes_ask, no_bid, no_ask, volume, open_interest, raw_json)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
            """,
            (
                snapshot_time_utc,
                city_key,
                target_date_local,
                series_ticker,
                event_ticker,
                market_ticker,
                status,
                yes_bid,
                yes_ask,
                no_bid,
                no_ask,
                volume,
                open_interest,
                raw_json,
            ),
        )


def fetch_latest_kalshi_weather_snapshot_time(
    db_path: Path,
    *,
    city_key: str,
    target_date_local: str,
) -> Optional[str]:
    """Return latest snapshot_time_utc for a given city/date."""
    ensure_schema(db_path)
    with connect(db_path) as conn:
        row = conn.execute(
            """
            SELECT snapshot_time_utc
            FROM kalshi_weather_market_snapshots
            WHERE city_key = ? AND target_date_local = ?
            ORDER BY snapshot_time_utc DESC
            LIMIT 1;
            """,
            (city_key, target_date_local),
        ).fetchone()
    return str(row[0]) if row and row[0] else None


def fetch_kalshi_weather_markets_at_snapshot(
    db_path: Path,
    *,
    snapshot_time_utc: str,
    city_key: str,
    target_date_local: str,
) -> Iterable[Dict[str, Any]]:
    """Fetch all Kalshi weather markets for a city/date at a specific snapshot time."""
    ensure_schema(db_path)
    with connect(db_path) as conn:
        rows = conn.execute(
            """
            SELECT
              snapshot_time_utc, city_key, target_date_local, series_ticker, event_ticker, market_ticker,
              status, yes_bid, yes_ask, no_bid, no_ask, volume, open_interest, raw_json
            FROM kalshi_weather_market_snapshots
            WHERE snapshot_time_utc = ? AND city_key = ? AND target_date_local = ?
            ORDER BY market_ticker ASC;
            """,
            (snapshot_time_utc, city_key, target_date_local),
        ).fetchall()

    out: list[Dict[str, Any]] = []
    for r in rows:
        try:
            raw = json.loads(r[13]) if r[13] else {}
        except Exception:
            raw = {}
        out.append(
            {
                "snapshot_time_utc": r[0],
                "city_key": r[1],
                "target_date_local": r[2],
                "series_ticker": r[3],
                "event_ticker": r[4],
                "market_ticker": r[5],
                "status": r[6],
                "yes_bid": r[7],
                "yes_ask": r[8],
                "no_bid": r[9],
                "no_ask": r[10],
                "volume": r[11],
                "open_interest": r[12],
                "raw": raw,
            }
        )
    return out

def fetch_joined_errors(
    db_path: Path,
    *,
    city_key: str,
    month: int,
    snapshot_hour_local: int,
    forecast_source: str,
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
             AND f.source=?
            WHERE o.city_key=?
              AND CAST(strftime('%m', o.date_local) AS INTEGER)=?
            """,
            (int(snapshot_hour_local), str(forecast_source), city_key, int(month)),
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
    source: str,
    n_samples: int,
    pmf: Dict[int, float],
    updated_at_utc: str,
) -> None:
    ensure_schema(db_path)
    pmf_json = json.dumps({str(k): v for k, v in pmf.items()}, separators=(",", ":"), ensure_ascii=False)
    with connect(db_path) as conn:
        conn.execute(
            """
            INSERT INTO error_models(
              city_key, month, snapshot_hour_local, source, n_samples, pmf_json, updated_at_utc
            )
            VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(city_key, month, snapshot_hour_local, source) DO UPDATE SET
              n_samples=excluded.n_samples,
              pmf_json=excluded.pmf_json,
              updated_at_utc=excluded.updated_at_utc;
            """,
            (
                city_key,
                int(month),
                int(snapshot_hour_local),
                str(source),
                int(n_samples),
                pmf_json,
                updated_at_utc,
            ),
        )


def fetch_forecast_snapshot(
    db_path: Path,
    *,
    city_key: str,
    target_date_local: str,
    snapshot_hour_local: int,
    source: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    """
    Return an existing snapshot row for a given city/date/hour/source, or None.
    Used to enforce 'first snapshot wins' unless --overwrite is supplied.
    When source is None, this will match any source for the city/date/hour.
    """
    ensure_schema(db_path)
    with connect(db_path) as conn:
        if source is None:
            row = conn.execute(
                """
                SELECT
                  city_key,
                  target_date_local,
                  snapshot_time_utc,
                  snapshot_hour_local,
                  snapshot_tz,
                  forecast_high_f,
                  source,
                  points_url,
                  forecast_url,
                  qc_flags,
                  raw_json
                FROM forecast_snapshots
                WHERE city_key = ?
                  AND target_date_local = ?
                  AND snapshot_hour_local = ?
                ORDER BY snapshot_time_utc ASC
                LIMIT 1;
                """,
                (city_key, target_date_local, int(snapshot_hour_local)),
            ).fetchone()
        else:
            row = conn.execute(
                """
                SELECT
                  city_key,
                  target_date_local,
                  snapshot_time_utc,
                  snapshot_hour_local,
                  snapshot_tz,
                  forecast_high_f,
                  source,
                  points_url,
                  forecast_url,
                  qc_flags,
                  raw_json
                FROM forecast_snapshots
                WHERE city_key = ?
                  AND target_date_local = ?
                  AND snapshot_hour_local = ?
                  AND source = ?
                ORDER BY snapshot_time_utc ASC
                LIMIT 1;
                """,
                (city_key, target_date_local, int(snapshot_hour_local), str(source)),
            ).fetchone()
    if not row:
        return None

    return {
        "city_key": row[0],
        "target_date_local": row[1],
        "snapshot_time_utc": row[2],
        "snapshot_hour_local": int(row[3]) if row[3] is not None else None,
        "snapshot_tz": row[4],
        "forecast_high_f": int(row[5]) if row[5] is not None else None,
        "source": row[6],
        "points_url": row[7],
        "forecast_url": row[8],
        "qc_flags": row[9],
        "raw_json": row[10],
        "raw": json.loads(row[10]) if row[10] else {},
    }


def fetch_error_model(
    db_path: Path,
    *,
    city_key: str,
    month: int,
    snapshot_hour_local: int,
    source: str,
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
            WHERE city_key=? AND month=? AND snapshot_hour_local=? AND source=?
            """,
            (city_key, int(month), int(snapshot_hour_local), str(source)),
        ).fetchone()
    if not row:
        return None
    n_samples, pmf_json, updated_at_utc = row
    pmf_raw = json.loads(pmf_json) if pmf_json else {}
    pmf = {int(k): float(v) for k, v in pmf_raw.items()}
    return {"n_samples": int(n_samples), "pmf": pmf, "updated_at_utc": updated_at_utc}
