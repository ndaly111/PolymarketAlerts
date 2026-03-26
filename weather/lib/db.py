from __future__ import annotations

import json
import os
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


@dataclass(frozen=True)
class DbPaths:
    root: Path
    db_path: Path


def default_paths(repo_root: Path) -> DbPaths:
    env = os.getenv("WEATHER_DB_PATH", "").strip()
    if env:
        return DbPaths(root=repo_root, db_path=Path(env))
    return DbPaths(
        root=repo_root, db_path=repo_root / "weather" / "data" / "weather_forecast_accuracy.db"
    )


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
              forecast_low_f REAL,                  -- daily low forecast (F) — nullable for existing rows
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

        # Add forecast_context column if it doesn't exist (migration)
        forecast_cols = {row[1] for row in conn.execute("PRAGMA table_info(forecast_snapshots);").fetchall()}
        if "forecast_context_json" not in forecast_cols:
            conn.execute(
                "ALTER TABLE forecast_snapshots ADD COLUMN forecast_context_json TEXT;"
            )
        if "forecast_low_f" not in forecast_cols:
            conn.execute(
                "ALTER TABLE forecast_snapshots ADD COLUMN forecast_low_f REAL;"
            )

        # Observed highs from CLI (you already create this in collect_cli_observed.py)
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS observed_cli (
              id INTEGER PRIMARY KEY AUTOINCREMENT,
              city_key TEXT NOT NULL,
              date_local TEXT NOT NULL,
              tmax_f INTEGER NOT NULL,
              tmin_f REAL,                -- daily low observed (F) — nullable for existing rows
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
        obs_cols = {row[1] for row in conn.execute("PRAGMA table_info(observed_cli);").fetchall()}
        if "tmin_f" not in obs_cols:
            conn.execute("ALTER TABLE observed_cli ADD COLUMN tmin_f REAL;")

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

        # Offline fixtures table for local simulation runs (no external calls).
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS kalshi_weather_market_fixtures (
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
        conn.execute("DROP INDEX IF EXISTS idx_kalshi_weather_fixtures_unique;")
        conn.execute(
            """
            CREATE UNIQUE INDEX IF NOT EXISTS idx_kalshi_weather_fixtures_unique_v2
            ON kalshi_weather_market_fixtures(snapshot_time_utc, market_ticker);
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
        # Add valid_from_utc for temporal versioning of error models
        if "valid_from_utc" not in columns:
            conn.execute(
                "ALTER TABLE error_models ADD COLUMN valid_from_utc TEXT;"
            )
            # Backfill existing rows: use updated_at_utc as valid_from_utc
            conn.execute(
                "UPDATE error_models SET valid_from_utc = updated_at_utc WHERE valid_from_utc IS NULL;"
            )
        # Add training_end_date to track the latest data used for training
        if "training_end_date" not in columns:
            conn.execute(
                "ALTER TABLE error_models ADD COLUMN training_end_date TEXT;"
            )
        conn.execute("DROP INDEX IF EXISTS idx_error_models_unique;")
        conn.execute(
            """
            CREATE UNIQUE INDEX IF NOT EXISTS idx_error_models_unique_v2
            ON error_models(city_key, month, snapshot_hour_local, source);
            """
        )

        # Intraday temperature observations for progress tracking
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS intraday_observations (
              id INTEGER PRIMARY KEY AUTOINCREMENT,
              city_key TEXT NOT NULL,
              target_date_local TEXT NOT NULL,
              observation_time_utc TEXT NOT NULL,
              observation_time_local TEXT NOT NULL,
              station_id TEXT NOT NULL,
              current_temp_f INTEGER NOT NULL,
              morning_temp_f INTEGER,
              max_observed_f INTEGER NOT NULL,
              forecast_high_f INTEGER NOT NULL,
              progress REAL NOT NULL,
              qc_flags TEXT NOT NULL,
              raw_json TEXT NOT NULL
            );
            """
        )
        conn.execute(
            """
            CREATE UNIQUE INDEX IF NOT EXISTS idx_intraday_obs_unique
            ON intraday_observations(city_key, target_date_local, observation_time_utc);
            """
        )
        conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_intraday_obs_latest
            ON intraday_observations(city_key, target_date_local, observation_time_utc DESC);
            """
        )

        # Station mapping cache for NWS observation stations
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS city_stations (
              id INTEGER PRIMARY KEY AUTOINCREMENT,
              city_key TEXT NOT NULL,
              station_id TEXT NOT NULL,
              station_name TEXT NOT NULL,
              distance_km REAL,
              priority INTEGER NOT NULL DEFAULT 1,
              updated_at_utc TEXT NOT NULL
            );
            """
        )
        conn.execute(
            """
            CREATE UNIQUE INDEX IF NOT EXISTS idx_city_stations_unique
            ON city_stations(city_key, station_id);
            """
        )

        # NBM probabilistic forecasts with native percentiles and exceedance probabilities
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS nbm_probabilistic_forecasts (
              id INTEGER PRIMARY KEY AUTOINCREMENT,
              city_key TEXT NOT NULL,
              target_date_local TEXT NOT NULL,      -- YYYY-MM-DD
              snapshot_time_utc TEXT NOT NULL,      -- ISO, when we captured this
              model_run_utc TEXT NOT NULL,          -- ISO, NBM model initialization time
              lead_hours INTEGER NOT NULL,          -- hours from model_run to target date's afternoon
              metric TEXT NOT NULL DEFAULT 'tmax',  -- 'tmax' or 'tmin'
              p10 INTEGER,                          -- 10th percentile (°F)
              p25 INTEGER,                          -- 25th percentile (°F)
              p50 INTEGER,                          -- 50th percentile (median, °F)
              p75 INTEGER,                          -- 75th percentile (°F)
              p90 INTEGER,                          -- 90th percentile (°F)
              exceedance_json TEXT,                 -- {threshold_int: P(T >= threshold)}
              raw_source TEXT,                      -- S3 path or URL
              qc_flags TEXT NOT NULL DEFAULT ''
            );
            """
        )
        conn.execute(
            """
            CREATE UNIQUE INDEX IF NOT EXISTS idx_nbm_prob_forecasts_unique
            ON nbm_probabilistic_forecasts(city_key, target_date_local, model_run_utc, lead_hours, metric);
            """
        )
        conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_nbm_prob_forecasts_lookup
            ON nbm_probabilistic_forecasts(city_key, target_date_local, metric);
            """
        )

        # Snow observations from SNODAS
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS snow_observations (
              id INTEGER PRIMARY KEY AUTOINCREMENT,
              city_key TEXT NOT NULL,
              date_local TEXT NOT NULL,
              snapshot_time_utc TEXT NOT NULL,
              snow_depth_inches REAL,
              swe_inches REAL,
              source TEXT NOT NULL,
              lat REAL NOT NULL,
              lon REAL NOT NULL,
              raw_json TEXT
            );
            """
        )
        conn.execute(
            """
            CREATE UNIQUE INDEX IF NOT EXISTS idx_snow_unique
            ON snow_observations(city_key, date_local, source);
            """
        )

        # SST observations from NDBC buoys
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS sst_observations (
              id INTEGER PRIMARY KEY AUTOINCREMENT,
              city_key TEXT NOT NULL,
              date_local TEXT NOT NULL,
              observation_time_utc TEXT NOT NULL,
              sst_f REAL NOT NULL,
              sst_anomaly_f REAL,
              buoy_id TEXT NOT NULL,
              source TEXT NOT NULL,
              raw_json TEXT
            );
            """
        )
        conn.execute(
            """
            CREATE UNIQUE INDEX IF NOT EXISTS idx_sst_unique
            ON sst_observations(city_key, date_local, buoy_id);
            """
        )

        # Visibility observations from METAR
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS visibility_observations (
              id INTEGER PRIMARY KEY AUTOINCREMENT,
              city_key TEXT NOT NULL,
              date_local TEXT NOT NULL,
              observation_time_utc TEXT NOT NULL,
              visibility_miles REAL NOT NULL,
              ceiling_ft INTEGER,
              fog_present INTEGER NOT NULL DEFAULT 0,
              station_id TEXT NOT NULL,
              source TEXT NOT NULL,
              raw_json TEXT
            );
            """
        )
        conn.execute(
            """
            CREATE UNIQUE INDEX IF NOT EXISTS idx_visibility_unique
            ON visibility_observations(city_key, date_local, observation_time_utc, station_id);
            """
        )
        conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_visibility_lookup
            ON visibility_observations(city_key, date_local);
            """
        )

        # Smoke observations from NOAA HMS
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS smoke_observations (
              id INTEGER PRIMARY KEY AUTOINCREMENT,
              city_key TEXT NOT NULL,
              date_local TEXT NOT NULL,
              observation_time_utc TEXT NOT NULL,
              smoke_present INTEGER NOT NULL DEFAULT 0,
              smoke_density TEXT,
              source TEXT NOT NULL,
              lat REAL NOT NULL,
              lon REAL NOT NULL,
              raw_json TEXT
            );
            """
        )
        conn.execute(
            """
            CREATE UNIQUE INDEX IF NOT EXISTS idx_smoke_unique
            ON smoke_observations(city_key, date_local, source);
            """
        )

        # Daily weather observations from Open-Meteo Archive (for ML feature backfill)
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS weather_observations_daily (
              city_key TEXT NOT NULL,
              date_local TEXT NOT NULL,
              precipitation_mm REAL,
              windspeed_max_mph REAL,
              source TEXT DEFAULT 'open_meteo_archive',
              fetched_at_utc TEXT,
              PRIMARY KEY (city_key, date_local)
            );
            """
        )

        # Settlement alignment tracking: compare our CLI observation to Kalshi's official settlement
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS settlement_alignment (
              id INTEGER PRIMARY KEY AUTOINCREMENT,
              city_key TEXT NOT NULL,
              date_local TEXT NOT NULL,
              cli_tmax_f INTEGER,                    -- our observed_cli value
              kalshi_settlement_f INTEGER,            -- Kalshi's official settlement
              delta INTEGER,                          -- kalshi - cli (0 = match)
              checked_at_utc TEXT NOT NULL,
              notes TEXT
            );
            """
        )
        conn.execute(
            """
            CREATE UNIQUE INDEX IF NOT EXISTS idx_settlement_alignment_unique
            ON settlement_alignment(city_key, date_local);
            """
        )

        # Upper-air snapshots: 500mb geopotential height, 850mb temp, dewpoint
        # Fetched from Open-Meteo historical/forecast API (pressure level variables)
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS upper_air_snapshots (
              id INTEGER PRIMARY KEY AUTOINCREMENT,
              city_key TEXT NOT NULL,
              date_local TEXT NOT NULL,
              snapshot_time_utc TEXT NOT NULL,
              z500_m REAL,
              t850_c REAL,
              dewpoint_2m_c REAL,
              source TEXT DEFAULT 'open_meteo',
              UNIQUE(city_key, date_local, source)
            );
            """
        )

        # Climate teleconnection indices (monthly, forward-filled to daily)
        # NAO, AO, PNA, MEI/ENSO, PDO from NOAA; MJO from CPC daily file
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS climate_indices (
              date_local TEXT PRIMARY KEY,
              nao REAL,
              ao REAL,
              pna REAL,
              mei_enso REAL,
              pdo REAL,
              mjo_amplitude REAL,
              mjo_phase INTEGER,
              fetched_at_utc TEXT
            );
            """
        )


def upsert_settlement_alignment(
    db_path: Path,
    *,
    city_key: str,
    date_local: str,
    cli_tmax_f: int,
    kalshi_settlement_f: int,
    checked_at_utc: str,
    notes: str = "",
) -> None:
    """Record settlement alignment between CLI and Kalshi for validation."""
    ensure_schema(db_path)
    delta = kalshi_settlement_f - cli_tmax_f
    with connect(db_path) as conn:
        conn.execute(
            """
            INSERT INTO settlement_alignment(
              city_key, date_local, cli_tmax_f, kalshi_settlement_f, delta, checked_at_utc, notes
            )
            VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(city_key, date_local) DO UPDATE SET
              cli_tmax_f=excluded.cli_tmax_f,
              kalshi_settlement_f=excluded.kalshi_settlement_f,
              delta=excluded.delta,
              checked_at_utc=excluded.checked_at_utc,
              notes=excluded.notes;
            """,
            (city_key, date_local, cli_tmax_f, kalshi_settlement_f, delta, checked_at_utc, notes),
        )


def fetch_settlement_alignment_stats(
    db_path: Path,
    *,
    city_key: Optional[str] = None,
) -> Dict[str, Any]:
    """Get settlement alignment statistics."""
    ensure_schema(db_path)
    with connect(db_path) as conn:
        if city_key:
            rows = conn.execute(
                "SELECT delta FROM settlement_alignment WHERE city_key = ?",
                (city_key,),
            ).fetchall()
        else:
            rows = conn.execute("SELECT delta FROM settlement_alignment").fetchall()

    if not rows:
        return {"n": 0, "match_rate": None, "mean_delta": None, "max_abs_delta": None}

    deltas = [r[0] for r in rows]
    n = len(deltas)
    exact_matches = sum(1 for d in deltas if d == 0)
    within_1 = sum(1 for d in deltas if abs(d) <= 1)

    return {
        "n": n,
        "exact_match_rate": round(exact_matches / n, 3),
        "within_1f_rate": round(within_1 / n, 3),
        "mean_delta": round(sum(deltas) / n, 2),
        "mean_abs_delta": round(sum(abs(d) for d in deltas) / n, 2),
        "max_abs_delta": max(abs(d) for d in deltas),
    }


def upsert_weather_observations_daily(
    db_path: Path,
    *,
    city_key: str,
    rows: list[tuple[str, Optional[float], Optional[float]]],
    source: str = "open_meteo_archive",
    fetched_at_utc: str,
) -> int:
    """Bulk upsert daily weather observations. rows = [(date_local, precip_mm, windspeed_mph), ...]."""
    ensure_schema(db_path)
    wrote = 0
    with connect(db_path) as conn:
        for date_local, precip_mm, wind_mph in rows:
            conn.execute(
                """
                INSERT INTO weather_observations_daily
                  (city_key, date_local, precipitation_mm, windspeed_max_mph, source, fetched_at_utc)
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(city_key, date_local) DO UPDATE SET
                  precipitation_mm=excluded.precipitation_mm,
                  windspeed_max_mph=excluded.windspeed_max_mph,
                  source=excluded.source,
                  fetched_at_utc=excluded.fetched_at_utc;
                """,
                (city_key, date_local, precip_mm, wind_mph, source, fetched_at_utc),
            )
            wrote += 1
    return wrote


def fetch_weather_observations_daily(
    db_path: Path,
    *,
    city_key: str,
    dates: list[str],
) -> Dict[str, Dict[str, float]]:
    """Fetch daily weather obs for a list of dates. Returns {date: {precipitation_mm, windspeed_max_mph}}."""
    ensure_schema(db_path)
    if not dates:
        return {}
    with connect(db_path) as conn:
        placeholders = ",".join("?" for _ in dates)
        rows = conn.execute(
            f"""
            SELECT date_local, precipitation_mm, windspeed_max_mph
            FROM weather_observations_daily
            WHERE city_key = ? AND date_local IN ({placeholders})
            """,
            [city_key] + dates,
        ).fetchall()
    result: Dict[str, Dict[str, float]] = {}
    for date_local, precip_mm, wind_mph in rows:
        result[date_local] = {
            "precipitation_mm": precip_mm if precip_mm is not None else 0.0,
            "windspeed_max_mph": wind_mph if wind_mph is not None else 10.0,
        }
    return result


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
    forecast_context: Optional[Dict[str, Any]] = None,
    forecast_low_f: Optional[int] = None,
) -> None:
    ensure_schema(db_path)
    qc = ",".join(sorted(set(qc_flags))) if qc_flags else ""
    raw_json = json.dumps(raw, separators=(",", ":"), ensure_ascii=False)
    context_json = (
        json.dumps(forecast_context, separators=(",", ":"), ensure_ascii=False)
        if forecast_context
        else None
    )
    with connect(db_path) as conn:
        conn.execute(
            """
            INSERT INTO forecast_snapshots
              (city_key, target_date_local, snapshot_time_utc, snapshot_hour_local, snapshot_tz,
               forecast_high_f, forecast_low_f, source, points_url, forecast_url, qc_flags, raw_json, forecast_context_json)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(city_key, target_date_local, snapshot_hour_local, source) DO UPDATE SET
              snapshot_time_utc=excluded.snapshot_time_utc,
              snapshot_tz=excluded.snapshot_tz,
              forecast_high_f=excluded.forecast_high_f,
              forecast_low_f=excluded.forecast_low_f,
              source=excluded.source,
              points_url=excluded.points_url,
              forecast_url=excluded.forecast_url,
              qc_flags=excluded.qc_flags,
              raw_json=excluded.raw_json,
              forecast_context_json=excluded.forecast_context_json;
            """,
            (
                city_key,
                target_date_local,
                snapshot_time_utc,
                int(snapshot_hour_local),
                snapshot_tz,
                int(forecast_high_f),
                int(forecast_low_f) if forecast_low_f is not None else None,
                source,
                points_url,
                forecast_url,
                qc,
                raw_json,
                context_json,
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
    tmin_f: Optional[int] = None,
) -> None:
    ensure_schema(db_path)
    qc = ",".join(sorted(set(qc_flags))) if qc_flags else ""
    with connect(db_path) as conn:
        conn.execute(
            """
            INSERT INTO observed_cli
              (city_key, date_local, tmax_f, tmin_f, fetched_at_utc, source_url, version_used,
               report_date_local, is_preliminary, qc_flags, raw_text)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(city_key, date_local) DO UPDATE SET
              tmax_f=excluded.tmax_f,
              tmin_f=excluded.tmin_f,
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
                int(tmin_f) if tmin_f is not None else None,
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


def upsert_kalshi_weather_market_fixture(
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
    """Insert or replace a Kalshi market fixture row for offline simulations."""
    ensure_schema(db_path)
    raw_json = json.dumps(raw, separators=(",", ":"), ensure_ascii=False)
    with connect(db_path) as conn:
        conn.execute(
            """
            INSERT INTO kalshi_weather_market_fixtures
              (snapshot_time_utc, city_key, target_date_local, series_ticker, event_ticker,
               market_ticker, status, yes_bid, yes_ask, no_bid, no_ask, volume, open_interest, raw_json)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(snapshot_time_utc, market_ticker) DO UPDATE SET
              city_key=excluded.city_key,
              target_date_local=excluded.target_date_local,
              series_ticker=excluded.series_ticker,
              event_ticker=excluded.event_ticker,
              status=excluded.status,
              yes_bid=excluded.yes_bid,
              yes_ask=excluded.yes_ask,
              no_bid=excluded.no_bid,
              no_ask=excluded.no_ask,
              volume=excluded.volume,
              open_interest=excluded.open_interest,
              raw_json=excluded.raw_json;
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


def _offline_fixtures_enabled() -> bool:
    return os.getenv("WEATHER_OFFLINE_FIXTURES", "").strip().lower() in {"1", "true", "yes", "y"}


def _kalshi_market_table() -> str:
    table = "kalshi_weather_market_fixtures" if _offline_fixtures_enabled() else "kalshi_weather_market_snapshots"
    if table not in {"kalshi_weather_market_fixtures", "kalshi_weather_market_snapshots"}:
        raise RuntimeError(f"Unexpected kalshi market table: {table}")
    return table


def fetch_latest_kalshi_weather_snapshot_time(
    db_path: Path,
    *,
    city_key: str,
    target_date_local: str,
) -> Optional[str]:
    """Return latest snapshot_time_utc for a given city/date."""
    ensure_schema(db_path)
    table = _kalshi_market_table()
    with connect(db_path) as conn:
        row = conn.execute(
            f"""
            SELECT snapshot_time_utc
            FROM {table}
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
    table = _kalshi_market_table()
    with connect(db_path) as conn:
        rows = conn.execute(
            f"""
            SELECT
              snapshot_time_utc, city_key, target_date_local, series_ticker, event_ticker, market_ticker,
              status, yes_bid, yes_ask, no_bid, no_ask, volume, open_interest, raw_json
            FROM {table}
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
    valid_from_utc: Optional[str] = None,
    training_end_date: Optional[str] = None,
) -> None:
    ensure_schema(db_path)
    pmf_json = json.dumps({str(k): v for k, v in pmf.items()}, separators=(",", ":"), ensure_ascii=False)
    vf = valid_from_utc or updated_at_utc
    with connect(db_path) as conn:
        conn.execute(
            """
            INSERT INTO error_models(
              city_key, month, snapshot_hour_local, source, n_samples, pmf_json,
              updated_at_utc, valid_from_utc, training_end_date
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(city_key, month, snapshot_hour_local, source) DO UPDATE SET
              n_samples=excluded.n_samples,
              pmf_json=excluded.pmf_json,
              updated_at_utc=excluded.updated_at_utc,
              valid_from_utc=excluded.valid_from_utc,
              training_end_date=excluded.training_end_date;
            """,
            (
                city_key,
                int(month),
                int(snapshot_hour_local),
                str(source),
                int(n_samples),
                pmf_json,
                updated_at_utc,
                vf,
                training_end_date,
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
                  raw_json,
                  forecast_context_json,
                  forecast_low_f
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
                  raw_json,
                  forecast_context_json,
                  forecast_low_f
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
        "forecast_context_json": row[11],
        "forecast_context": json.loads(row[11]) if row[11] else None,
        "forecast_low_f": int(row[12]) if row[12] is not None else None,
    }


def fetch_latest_forecast_snapshot(
    db_path: Path,
    *,
    city_key: str,
    target_date_local: str,
    source: str,
) -> Optional[Dict[str, Any]]:
    """Return the latest snapshot row for a given city/date/source, or None."""
    ensure_schema(db_path)
    with connect(db_path) as conn:
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
              raw_json,
              forecast_context_json,
              forecast_low_f
            FROM forecast_snapshots
            WHERE city_key = ?
              AND target_date_local = ?
              AND source = ?
            ORDER BY snapshot_time_utc DESC
            LIMIT 1;
            """,
            (city_key, target_date_local, str(source)),
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
        "forecast_context_json": row[11],
        "forecast_context": json.loads(row[11]) if row[11] else None,
        "forecast_low_f": int(row[12]) if row[12] is not None else None,
    }


def forecast_snapshot_exists(
    db_path: Path,
    *,
    city_key: str,
    target_date_local: str,
    source: str,
) -> bool:
    """Return True if any snapshot exists for a city/date/source (any hour)."""
    ensure_schema(db_path)
    with connect(db_path) as conn:
        row = conn.execute(
            """
            SELECT 1
            FROM forecast_snapshots
            WHERE city_key = ?
              AND target_date_local = ?
              AND source = ?
            LIMIT 1;
            """,
            (city_key, target_date_local, str(source)),
        ).fetchone()
    return row is not None


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


def fetch_error_model_at_time(
    db_path: Path,
    *,
    city_key: str,
    month: int,
    snapshot_hour_local: int,
    source: str,
    as_of_utc: str,
) -> Optional[Dict[str, Any]]:
    """
    Fetch error model that was available at as_of_utc (versioned).

    Returns the most recent model whose valid_from_utc <= as_of_utc.
    Falls back to fetch_error_model() if no valid_from_utc data exists.
    """
    ensure_schema(db_path)
    with connect(db_path) as conn:
        row = conn.execute(
            """
            SELECT n_samples, pmf_json, updated_at_utc, valid_from_utc
            FROM error_models
            WHERE city_key=? AND month=? AND snapshot_hour_local=? AND source=?
              AND valid_from_utc IS NOT NULL
              AND valid_from_utc <= ?
            ORDER BY valid_from_utc DESC
            LIMIT 1
            """,
            (city_key, int(month), int(snapshot_hour_local), str(source), as_of_utc),
        ).fetchone()

    if not row:
        # Fall back to unversioned lookup
        return fetch_error_model(
            db_path,
            city_key=city_key,
            month=month,
            snapshot_hour_local=snapshot_hour_local,
            source=source,
        )

    n_samples, pmf_json, updated_at_utc, valid_from_utc = row
    pmf_raw = json.loads(pmf_json) if pmf_json else {}
    pmf = {int(k): float(v) for k, v in pmf_raw.items()}
    return {
        "n_samples": int(n_samples),
        "pmf": pmf,
        "updated_at_utc": updated_at_utc,
        "valid_from_utc": valid_from_utc,
    }


# ---------------------------------------------------------------------------
# Intraday observation functions
# ---------------------------------------------------------------------------


def upsert_intraday_observation(
    db_path: Path,
    *,
    city_key: str,
    target_date_local: str,
    observation_time_utc: str,
    observation_time_local: str,
    station_id: str,
    current_temp_f: int,
    morning_temp_f: int,
    max_observed_f: int,
    forecast_high_f: int,
    progress: float,
    qc_flags: Iterable[str],
    raw: Dict[str, Any],
) -> None:
    """Insert or update an intraday observation."""
    ensure_schema(db_path)
    qc = ",".join(sorted(set(qc_flags))) if qc_flags else ""
    raw_json = json.dumps(raw, separators=(",", ":"), ensure_ascii=False)
    with connect(db_path) as conn:
        conn.execute(
            """
            INSERT INTO intraday_observations
              (city_key, target_date_local, observation_time_utc, observation_time_local,
               station_id, current_temp_f, morning_temp_f, max_observed_f, forecast_high_f,
               progress, qc_flags, raw_json)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(city_key, target_date_local, observation_time_utc) DO UPDATE SET
              observation_time_local=excluded.observation_time_local,
              station_id=excluded.station_id,
              current_temp_f=excluded.current_temp_f,
              morning_temp_f=excluded.morning_temp_f,
              max_observed_f=excluded.max_observed_f,
              forecast_high_f=excluded.forecast_high_f,
              progress=excluded.progress,
              qc_flags=excluded.qc_flags,
              raw_json=excluded.raw_json;
            """,
            (
                city_key,
                target_date_local,
                observation_time_utc,
                observation_time_local,
                station_id,
                int(current_temp_f),
                int(morning_temp_f),
                int(max_observed_f),
                int(forecast_high_f),
                float(progress),
                qc,
                raw_json,
            ),
        )


def fetch_latest_intraday_state(
    db_path: Path,
    *,
    city_key: str,
    target_date_local: str,
) -> Optional[Dict[str, Any]]:
    """Fetch the most recent intraday observation for a city/date."""
    ensure_schema(db_path)
    with connect(db_path) as conn:
        row = conn.execute(
            """
            SELECT
              city_key, target_date_local, observation_time_utc, observation_time_local,
              station_id, current_temp_f, morning_temp_f, max_observed_f, forecast_high_f,
              progress, qc_flags
            FROM intraday_observations
            WHERE city_key = ? AND target_date_local = ?
            ORDER BY observation_time_utc DESC
            LIMIT 1;
            """,
            (city_key, target_date_local),
        ).fetchone()

    if not row:
        return None

    return {
        "city_key": row[0],
        "target_date_local": row[1],
        "observation_time_utc": row[2],
        "observation_time_local": row[3],
        "station_id": row[4],
        "current_temp_f": int(row[5]),
        "morning_temp_f": int(row[6]) if row[6] is not None else None,
        "max_observed_f": int(row[7]),
        "forecast_high_f": int(row[8]),
        "progress": float(row[9]),
        "qc_flags": row[10],
    }


def fetch_first_intraday_observation(
    db_path: Path,
    *,
    city_key: str,
    target_date_local: str,
) -> Optional[int]:
    """Return the first observed temperature of the day (morning reference)."""
    ensure_schema(db_path)
    with connect(db_path) as conn:
        row = conn.execute(
            """
            SELECT current_temp_f
            FROM intraday_observations
            WHERE city_key = ? AND target_date_local = ?
            ORDER BY observation_time_utc ASC
            LIMIT 1;
            """,
            (city_key, target_date_local),
        ).fetchone()
    return int(row[0]) if row else None


def fetch_max_intraday_observation(
    db_path: Path,
    *,
    city_key: str,
    target_date_local: str,
) -> Optional[int]:
    """Return the maximum observed temperature so far today."""
    ensure_schema(db_path)
    with connect(db_path) as conn:
        row = conn.execute(
            """
            SELECT MAX(max_observed_f)
            FROM intraday_observations
            WHERE city_key = ? AND target_date_local = ?;
            """,
            (city_key, target_date_local),
        ).fetchone()
    return int(row[0]) if row and row[0] is not None else None


# ---------------------------------------------------------------------------
# City station cache functions
# ---------------------------------------------------------------------------


def upsert_city_station(
    db_path: Path,
    *,
    city_key: str,
    station_id: str,
    station_name: str,
    distance_km: Optional[float],
    priority: int,
    updated_at_utc: str,
) -> None:
    """Cache a station mapping for a city."""
    ensure_schema(db_path)
    with connect(db_path) as conn:
        conn.execute(
            """
            INSERT INTO city_stations
              (city_key, station_id, station_name, distance_km, priority, updated_at_utc)
            VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(city_key, station_id) DO UPDATE SET
              station_name=excluded.station_name,
              distance_km=excluded.distance_km,
              priority=excluded.priority,
              updated_at_utc=excluded.updated_at_utc;
            """,
            (city_key, station_id, station_name, distance_km, priority, updated_at_utc),
        )


def fetch_preferred_station(
    db_path: Path,
    *,
    city_key: str,
) -> Optional[str]:
    """Return the preferred (lowest priority) station for a city."""
    ensure_schema(db_path)
    with connect(db_path) as conn:
        row = conn.execute(
            """
            SELECT station_id
            FROM city_stations
            WHERE city_key = ?
            ORDER BY priority ASC
            LIMIT 1;
            """,
            (city_key,),
        ).fetchone()
    return str(row[0]) if row else None


def fetch_all_city_stations(
    db_path: Path,
    *,
    city_key: str,
) -> List[Dict[str, Any]]:
    """Return all cached stations for a city, ordered by priority."""
    ensure_schema(db_path)
    with connect(db_path) as conn:
        rows = conn.execute(
            """
            SELECT station_id, station_name, priority
            FROM city_stations
            WHERE city_key = ?
            ORDER BY priority ASC;
            """,
            (city_key,),
        ).fetchall()
    return [
        {"station_id": r[0], "station_name": r[1], "priority": r[2]}
        for r in rows
    ]


def ensure_weather_trades_schema(db_path: Path) -> None:
    """Ensure the weather_trades schema exists in a trades database."""
    with connect(db_path) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS weather_trades (
              id INTEGER PRIMARY KEY AUTOINCREMENT,
              trade_date TEXT NOT NULL,
              city_key TEXT NOT NULL,
              market_ticker TEXT NOT NULL,
              event_display TEXT NOT NULL,
              side TEXT NOT NULL,
              quantity INTEGER NOT NULL,
              limit_price_cents INTEGER NOT NULL,
              fair_q REAL NOT NULL,
              ev REAL NOT NULL,
              forecast_high_f INTEGER,
              order_id TEXT,
              status TEXT NOT NULL,
              fill_price_cents INTEGER,
              placed_at_utc TEXT,
              last_checked_at_utc TEXT,
              filled_at_utc TEXT,
              created_at TEXT NOT NULL,
              updated_at TEXT NOT NULL,
              UNIQUE(trade_date, market_ticker)
            );
            """
        )
        columns = {row[1] for row in conn.execute("PRAGMA table_info(weather_trades);").fetchall()}
        for column in ("placed_at_utc", "last_checked_at_utc", "filled_at_utc"):
            if column not in columns:
                conn.execute(f"ALTER TABLE weather_trades ADD COLUMN {column} TEXT;")


# ---------------------------------------------------------------------------
# NBM probabilistic forecast functions
# ---------------------------------------------------------------------------


def upsert_nbm_probabilistic_forecast(
    db_path: Path,
    *,
    city_key: str,
    target_date_local: str,
    snapshot_time_utc: str,
    model_run_utc: str,
    lead_hours: int,
    metric: str = "tmax",
    p10: Optional[int] = None,
    p25: Optional[int] = None,
    p50: Optional[int] = None,
    p75: Optional[int] = None,
    p90: Optional[int] = None,
    exceedance: Optional[Dict[int, float]] = None,
    raw_source: Optional[str] = None,
    qc_flags: Optional[Iterable[str]] = None,
) -> None:
    """Insert or update an NBM probabilistic forecast."""
    ensure_schema(db_path)
    qc = ",".join(sorted(set(qc_flags))) if qc_flags else ""
    exceedance_json = (
        json.dumps({str(k): v for k, v in exceedance.items()}, separators=(",", ":"))
        if exceedance
        else None
    )
    with connect(db_path) as conn:
        conn.execute(
            """
            INSERT INTO nbm_probabilistic_forecasts
              (city_key, target_date_local, snapshot_time_utc, model_run_utc, lead_hours,
               metric, p10, p25, p50, p75, p90, exceedance_json, raw_source, qc_flags)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(city_key, target_date_local, model_run_utc, lead_hours, metric) DO UPDATE SET
              snapshot_time_utc=excluded.snapshot_time_utc,
              p10=excluded.p10,
              p25=excluded.p25,
              p50=excluded.p50,
              p75=excluded.p75,
              p90=excluded.p90,
              exceedance_json=excluded.exceedance_json,
              raw_source=excluded.raw_source,
              qc_flags=excluded.qc_flags;
            """,
            (
                city_key,
                target_date_local,
                snapshot_time_utc,
                model_run_utc,
                int(lead_hours),
                metric,
                p10,
                p25,
                p50,
                p75,
                p90,
                exceedance_json,
                raw_source,
                qc,
            ),
        )


def fetch_latest_nbm_forecast(
    db_path: Path,
    *,
    city_key: str,
    target_date_local: str,
    metric: str = "tmax",
) -> Optional[Dict[str, Any]]:
    """Fetch the most recent NBM forecast for a city/date/metric."""
    ensure_schema(db_path)
    with connect(db_path) as conn:
        row = conn.execute(
            """
            SELECT
              city_key, target_date_local, snapshot_time_utc, model_run_utc, lead_hours,
              metric, p10, p25, p50, p75, p90, exceedance_json, raw_source, qc_flags
            FROM nbm_probabilistic_forecasts
            WHERE city_key = ? AND target_date_local = ? AND metric = ?
            ORDER BY model_run_utc DESC
            LIMIT 1;
            """,
            (city_key, target_date_local, metric),
        ).fetchone()

    if not row:
        return None

    exceedance_raw = json.loads(row[11]) if row[11] else {}
    exceedance = {int(k): float(v) for k, v in exceedance_raw.items()}

    return {
        "city_key": row[0],
        "target_date_local": row[1],
        "snapshot_time_utc": row[2],
        "model_run_utc": row[3],
        "lead_hours": int(row[4]),
        "metric": row[5],
        "p10": int(row[6]) if row[6] is not None else None,
        "p25": int(row[7]) if row[7] is not None else None,
        "p50": int(row[8]) if row[8] is not None else None,
        "p75": int(row[9]) if row[9] is not None else None,
        "p90": int(row[10]) if row[10] is not None else None,
        "exceedance": exceedance,
        "raw_source": row[12],
        "qc_flags": row[13],
    }


def fetch_nbm_forecast_at_time(
    db_path: Path,
    *,
    city_key: str,
    target_date_local: str,
    as_of_utc: str,
    metric: str = "tmax",
) -> Optional[Dict[str, Any]]:
    """
    Fetch NBM forecast that was available at a specific time (for backtesting).
    Returns the most recent forecast where model_run_utc <= as_of_utc.
    """
    ensure_schema(db_path)
    with connect(db_path) as conn:
        row = conn.execute(
            """
            SELECT
              city_key, target_date_local, snapshot_time_utc, model_run_utc, lead_hours,
              metric, p10, p25, p50, p75, p90, exceedance_json, raw_source, qc_flags
            FROM nbm_probabilistic_forecasts
            WHERE city_key = ? AND target_date_local = ? AND metric = ?
              AND model_run_utc <= ?
            ORDER BY model_run_utc DESC
            LIMIT 1;
            """,
            (city_key, target_date_local, metric, as_of_utc),
        ).fetchone()

    if not row:
        return None

    exceedance_raw = json.loads(row[11]) if row[11] else {}
    exceedance = {int(k): float(v) for k, v in exceedance_raw.items()}

    return {
        "city_key": row[0],
        "target_date_local": row[1],
        "snapshot_time_utc": row[2],
        "model_run_utc": row[3],
        "lead_hours": int(row[4]),
        "metric": row[5],
        "p10": int(row[6]) if row[6] is not None else None,
        "p25": int(row[7]) if row[7] is not None else None,
        "p50": int(row[8]) if row[8] is not None else None,
        "p75": int(row[9]) if row[9] is not None else None,
        "p90": int(row[10]) if row[10] is not None else None,
        "exceedance": exceedance,
        "raw_source": row[12],
        "qc_flags": row[13],
    }


def fetch_nbm_forecast_history(
    db_path: Path,
    *,
    city_key: str,
    target_date_local: str,
    metric: str = "tmax",
) -> list[Dict[str, Any]]:
    """Fetch all NBM forecasts for a city/date/metric, ordered by model run time."""
    ensure_schema(db_path)
    with connect(db_path) as conn:
        rows = conn.execute(
            """
            SELECT
              city_key, target_date_local, snapshot_time_utc, model_run_utc, lead_hours,
              metric, p10, p25, p50, p75, p90, exceedance_json, raw_source, qc_flags
            FROM nbm_probabilistic_forecasts
            WHERE city_key = ? AND target_date_local = ? AND metric = ?
            ORDER BY model_run_utc ASC;
            """,
            (city_key, target_date_local, metric),
        ).fetchall()

    results = []
    for row in rows:
        exceedance_raw = json.loads(row[11]) if row[11] else {}
        exceedance = {int(k): float(v) for k, v in exceedance_raw.items()}
        results.append({
            "city_key": row[0],
            "target_date_local": row[1],
            "snapshot_time_utc": row[2],
            "model_run_utc": row[3],
            "lead_hours": int(row[4]),
            "metric": row[5],
            "p10": int(row[6]) if row[6] is not None else None,
            "p25": int(row[7]) if row[7] is not None else None,
            "p50": int(row[8]) if row[8] is not None else None,
            "p75": int(row[9]) if row[9] is not None else None,
            "p90": int(row[10]) if row[10] is not None else None,
            "exceedance": exceedance,
            "raw_source": row[12],
            "qc_flags": row[13],
        })
    return results


# ---------------------------------------------------------------------------
# Snow observation functions
# ---------------------------------------------------------------------------


def upsert_snow_observation(
    db_path: Path,
    *,
    city_key: str,
    date_local: str,
    snapshot_time_utc: str,
    snow_depth_inches: Optional[float],
    swe_inches: Optional[float],
    source: str,
    lat: float,
    lon: float,
    raw: Optional[Dict[str, Any]] = None,
) -> None:
    """Insert or update a snow observation."""
    ensure_schema(db_path)
    raw_json = json.dumps(raw, separators=(",", ":"), ensure_ascii=False) if raw else None
    with connect(db_path) as conn:
        conn.execute(
            """
            INSERT INTO snow_observations
              (city_key, date_local, snapshot_time_utc, snow_depth_inches, swe_inches,
               source, lat, lon, raw_json)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(city_key, date_local, source) DO UPDATE SET
              snapshot_time_utc=excluded.snapshot_time_utc,
              snow_depth_inches=excluded.snow_depth_inches,
              swe_inches=excluded.swe_inches,
              lat=excluded.lat,
              lon=excluded.lon,
              raw_json=excluded.raw_json;
            """,
            (
                city_key,
                date_local,
                snapshot_time_utc,
                snow_depth_inches,
                swe_inches,
                source,
                lat,
                lon,
                raw_json,
            ),
        )


def fetch_snow_observation(
    db_path: Path,
    *,
    city_key: str,
    date_local: str,
) -> Optional[Dict[str, Any]]:
    """Fetch snow observation for a city/date."""
    ensure_schema(db_path)
    with connect(db_path) as conn:
        row = conn.execute(
            """
            SELECT
              city_key, date_local, snapshot_time_utc, snow_depth_inches, swe_inches,
              source, lat, lon
            FROM snow_observations
            WHERE city_key = ? AND date_local = ?
            ORDER BY snapshot_time_utc DESC
            LIMIT 1;
            """,
            (city_key, date_local),
        ).fetchone()

    if not row:
        return None

    return {
        "city_key": row[0],
        "date_local": row[1],
        "snapshot_time_utc": row[2],
        "snow_depth_inches": float(row[3]) if row[3] is not None else None,
        "swe_inches": float(row[4]) if row[4] is not None else None,
        "source": row[5],
        "lat": float(row[6]),
        "lon": float(row[7]),
    }


# ---------------------------------------------------------------------------
# SST observation functions
# ---------------------------------------------------------------------------


def upsert_sst_observation(
    db_path: Path,
    *,
    city_key: str,
    date_local: str,
    observation_time_utc: str,
    sst_f: float,
    sst_anomaly_f: Optional[float],
    buoy_id: str,
    source: str,
    raw: Optional[Dict[str, Any]] = None,
) -> None:
    """Insert or update an SST observation."""
    ensure_schema(db_path)
    raw_json = json.dumps(raw, separators=(",", ":"), ensure_ascii=False) if raw else None
    with connect(db_path) as conn:
        conn.execute(
            """
            INSERT INTO sst_observations
              (city_key, date_local, observation_time_utc, sst_f, sst_anomaly_f,
               buoy_id, source, raw_json)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(city_key, date_local, buoy_id) DO UPDATE SET
              observation_time_utc=excluded.observation_time_utc,
              sst_f=excluded.sst_f,
              sst_anomaly_f=excluded.sst_anomaly_f,
              source=excluded.source,
              raw_json=excluded.raw_json;
            """,
            (
                city_key,
                date_local,
                observation_time_utc,
                sst_f,
                sst_anomaly_f,
                buoy_id,
                source,
                raw_json,
            ),
        )


def fetch_sst_observation(
    db_path: Path,
    *,
    city_key: str,
    date_local: str,
) -> Optional[Dict[str, Any]]:
    """Fetch SST observation for a city/date."""
    ensure_schema(db_path)
    with connect(db_path) as conn:
        row = conn.execute(
            """
            SELECT
              city_key, date_local, observation_time_utc, sst_f, sst_anomaly_f,
              buoy_id, source
            FROM sst_observations
            WHERE city_key = ? AND date_local = ?
            ORDER BY observation_time_utc DESC
            LIMIT 1;
            """,
            (city_key, date_local),
        ).fetchone()

    if not row:
        return None

    return {
        "city_key": row[0],
        "date_local": row[1],
        "observation_time_utc": row[2],
        "sst_f": float(row[3]),
        "sst_anomaly_f": float(row[4]) if row[4] is not None else None,
        "buoy_id": row[5],
        "source": row[6],
    }


# ---------------------------------------------------------------------------
# Visibility observation functions
# ---------------------------------------------------------------------------


def upsert_visibility_observation(
    db_path: Path,
    *,
    city_key: str,
    date_local: str,
    observation_time_utc: str,
    visibility_miles: float,
    ceiling_ft: Optional[int],
    fog_present: bool,
    station_id: str,
    source: str,
    raw: Optional[Dict[str, Any]] = None,
) -> None:
    """Insert or update a visibility observation."""
    ensure_schema(db_path)
    raw_json = json.dumps(raw, separators=(",", ":"), ensure_ascii=False) if raw else None
    with connect(db_path) as conn:
        conn.execute(
            """
            INSERT INTO visibility_observations
              (city_key, date_local, observation_time_utc, visibility_miles, ceiling_ft,
               fog_present, station_id, source, raw_json)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(city_key, date_local, observation_time_utc, station_id) DO UPDATE SET
              visibility_miles=excluded.visibility_miles,
              ceiling_ft=excluded.ceiling_ft,
              fog_present=excluded.fog_present,
              source=excluded.source,
              raw_json=excluded.raw_json;
            """,
            (
                city_key,
                date_local,
                observation_time_utc,
                visibility_miles,
                ceiling_ft,
                1 if fog_present else 0,
                station_id,
                source,
                raw_json,
            ),
        )


def fetch_visibility_summary(
    db_path: Path,
    *,
    city_key: str,
    date_local: str,
) -> Optional[Dict[str, Any]]:
    """Fetch visibility summary for a city/date (min visibility, fog hours)."""
    ensure_schema(db_path)
    with connect(db_path) as conn:
        row = conn.execute(
            """
            SELECT
              MIN(visibility_miles) as min_visibility,
              SUM(fog_present) as fog_hours,
              COUNT(*) as obs_count
            FROM visibility_observations
            WHERE city_key = ? AND date_local = ?;
            """,
            (city_key, date_local),
        ).fetchone()

    if not row or row[2] == 0:
        return None

    return {
        "city_key": city_key,
        "date_local": date_local,
        "visibility_min_miles": float(row[0]) if row[0] is not None else None,
        "fog_hours": int(row[1]) if row[1] is not None else 0,
        "obs_count": int(row[2]),
    }


# ---------------------------------------------------------------------------
# Smoke observation functions
# ---------------------------------------------------------------------------


def upsert_smoke_observation(
    db_path: Path,
    *,
    city_key: str,
    date_local: str,
    observation_time_utc: str,
    smoke_present: bool,
    smoke_density: Optional[str],
    source: str,
    lat: float,
    lon: float,
    raw: Optional[Dict[str, Any]] = None,
) -> None:
    """Insert or update a smoke observation."""
    ensure_schema(db_path)
    raw_json = json.dumps(raw, separators=(",", ":"), ensure_ascii=False) if raw else None
    with connect(db_path) as conn:
        conn.execute(
            """
            INSERT INTO smoke_observations
              (city_key, date_local, observation_time_utc, smoke_present, smoke_density,
               source, lat, lon, raw_json)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(city_key, date_local, source) DO UPDATE SET
              observation_time_utc=excluded.observation_time_utc,
              smoke_present=excluded.smoke_present,
              smoke_density=excluded.smoke_density,
              lat=excluded.lat,
              lon=excluded.lon,
              raw_json=excluded.raw_json;
            """,
            (
                city_key,
                date_local,
                observation_time_utc,
                1 if smoke_present else 0,
                smoke_density,
                source,
                lat,
                lon,
                raw_json,
            ),
        )


def fetch_smoke_observation(
    db_path: Path,
    *,
    city_key: str,
    date_local: str,
) -> Optional[Dict[str, Any]]:
    """Fetch smoke observation for a city/date."""
    ensure_schema(db_path)
    with connect(db_path) as conn:
        row = conn.execute(
            """
            SELECT
              city_key, date_local, observation_time_utc, smoke_present, smoke_density,
              source, lat, lon
            FROM smoke_observations
            WHERE city_key = ? AND date_local = ?
            ORDER BY observation_time_utc DESC
            LIMIT 1;
            """,
            (city_key, date_local),
        ).fetchone()

    if not row:
        return None

    return {
        "city_key": row[0],
        "date_local": row[1],
        "observation_time_utc": row[2],
        "smoke_present": bool(row[3]),
        "smoke_density": row[4],
        "source": row[5],
        "lat": float(row[6]),
        "lon": float(row[7]),
    }


def upsert_upper_air_snapshot(
    db_path: Path,
    *,
    city_key: str,
    date_local: str,
    snapshot_time_utc: str,
    z500_m: Optional[float],
    t850_c: Optional[float],
    dewpoint_2m_c: Optional[float],
    source: str = "open_meteo",
) -> None:
    """Upsert upper-air observation (z500, t850, dewpoint) for a city/date."""
    ensure_schema(db_path)
    with connect(db_path) as conn:
        conn.execute(
            """
            INSERT INTO upper_air_snapshots
              (city_key, date_local, snapshot_time_utc, z500_m, t850_c, dewpoint_2m_c, source)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(city_key, date_local, source) DO UPDATE SET
              snapshot_time_utc=excluded.snapshot_time_utc,
              z500_m=excluded.z500_m,
              t850_c=excluded.t850_c,
              dewpoint_2m_c=excluded.dewpoint_2m_c;
            """,
            (city_key, date_local, snapshot_time_utc, z500_m, t850_c, dewpoint_2m_c, source),
        )


def fetch_upper_air_snapshot(
    db_path: Path,
    *,
    city_key: str,
    date_local: str,
) -> Optional[Dict[str, Any]]:
    """Fetch latest upper-air snapshot for a city/date."""
    ensure_schema(db_path)
    with connect(db_path) as conn:
        row = conn.execute(
            """
            SELECT city_key, date_local, snapshot_time_utc, z500_m, t850_c, dewpoint_2m_c, source
            FROM upper_air_snapshots
            WHERE city_key = ? AND date_local = ?
            ORDER BY snapshot_time_utc DESC
            LIMIT 1;
            """,
            (city_key, date_local),
        ).fetchone()

    if not row:
        return None

    return {
        "city_key": row[0],
        "date_local": row[1],
        "snapshot_time_utc": row[2],
        "z500_m": row[3],
        "t850_c": row[4],
        "dewpoint_2m_c": row[5],
        "source": row[6],
    }


def upsert_climate_indices(
    db_path: Path,
    *,
    date_local: str,
    nao: Optional[float] = None,
    ao: Optional[float] = None,
    pna: Optional[float] = None,
    mei_enso: Optional[float] = None,
    pdo: Optional[float] = None,
    mjo_amplitude: Optional[float] = None,
    mjo_phase: Optional[int] = None,
    fetched_at_utc: Optional[str] = None,
) -> None:
    """Upsert climate teleconnection indices for a date."""
    ensure_schema(db_path)
    if fetched_at_utc is None:
        from datetime import datetime as _dt
        fetched_at_utc = _dt.utcnow().isoformat()
    with connect(db_path) as conn:
        conn.execute(
            """
            INSERT INTO climate_indices
              (date_local, nao, ao, pna, mei_enso, pdo, mjo_amplitude, mjo_phase, fetched_at_utc)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(date_local) DO UPDATE SET
              nao=COALESCE(excluded.nao, nao),
              ao=COALESCE(excluded.ao, ao),
              pna=COALESCE(excluded.pna, pna),
              mei_enso=COALESCE(excluded.mei_enso, mei_enso),
              pdo=COALESCE(excluded.pdo, pdo),
              mjo_amplitude=COALESCE(excluded.mjo_amplitude, mjo_amplitude),
              mjo_phase=COALESCE(excluded.mjo_phase, mjo_phase),
              fetched_at_utc=excluded.fetched_at_utc;
            """,
            (date_local, nao, ao, pna, mei_enso, pdo, mjo_amplitude, mjo_phase, fetched_at_utc),
        )


def fetch_climate_indices(
    db_path: Path,
    *,
    date_local: str,
) -> Optional[Dict[str, Any]]:
    """Fetch climate indices for a given date (looks up the exact date)."""
    ensure_schema(db_path)
    with connect(db_path) as conn:
        row = conn.execute(
            """
            SELECT date_local, nao, ao, pna, mei_enso, pdo, mjo_amplitude, mjo_phase
            FROM climate_indices
            WHERE date_local <= ?
            ORDER BY date_local DESC
            LIMIT 1;
            """,
            (date_local,),
        ).fetchone()

    if not row:
        return None

    return {
        "date_local": row[0],
        "nao": row[1],
        "ao": row[2],
        "pna": row[3],
        "mei_enso": row[4],
        "pdo": row[5],
        "mjo_amplitude": row[6],
        "mjo_phase": row[7],
    }
