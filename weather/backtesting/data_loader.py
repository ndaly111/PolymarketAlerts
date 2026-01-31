"""
As-of data loader for backtesting with strict temporal semantics.

The critical requirement for backtesting is to only use data that was
actually available at the decision time. This prevents look-ahead bias
and ensures backtest results reflect real-world performance.

Key concept: As-of queries
- "What forecast was available at 5:10am ET on Jan 15, 2025?"
- Must return the forecast that existed at that exact point in time
- Cannot use forecasts issued after the decision time
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional
from zoneinfo import ZoneInfo

from weather.lib import db as db_lib


@dataclass
class AsOfDataLoader:
    """
    Load historical data with strict as-of semantics.

    All data retrieval methods take an `as_of_utc` timestamp and only
    return data that was available at that point in time.
    """
    db_path: Path

    def __post_init__(self):
        self.db_path = Path(self.db_path)
        db_lib.ensure_schema(self.db_path)

    def get_forecast_at_time(
        self,
        city_key: str,
        target_date_local: str,
        as_of_utc: str,
        source: str = "open_meteo",
    ) -> Optional[Dict[str, Any]]:
        """
        Get forecast that was actually available at as_of_utc.

        This is the most critical query for backtesting. It returns the
        most recent forecast snapshot where snapshot_time_utc <= as_of_utc.

        Args:
            city_key: City identifier
            target_date_local: Target date (YYYY-MM-DD)
            as_of_utc: Point-in-time (ISO format)
            source: Forecast source

        Returns:
            Forecast dict or None if no forecast was available
        """
        conn = sqlite3.connect(self.db_path)

        row = conn.execute(
            """
            SELECT
                city_key, target_date_local, snapshot_time_utc, snapshot_hour_local,
                snapshot_tz, forecast_high_f, source, qc_flags
            FROM forecast_snapshots
            WHERE city_key = ?
                AND target_date_local = ?
                AND source = ?
                AND snapshot_time_utc <= ?
            ORDER BY snapshot_time_utc DESC
            LIMIT 1
            """,
            (city_key, target_date_local, source, as_of_utc),
        ).fetchone()

        conn.close()

        if not row:
            return None

        return {
            "city_key": row[0],
            "target_date_local": row[1],
            "snapshot_time_utc": row[2],
            "snapshot_hour_local": row[3],
            "snapshot_tz": row[4],
            "forecast_high_f": row[5],
            "source": row[6],
            "qc_flags": row[7],
        }

    def get_nbm_forecast_at_time(
        self,
        city_key: str,
        target_date_local: str,
        as_of_utc: str,
        metric: str = "tmax",
    ) -> Optional[Dict[str, Any]]:
        """
        Get NBM probabilistic forecast available at as_of_utc.

        Args:
            city_key: City identifier
            target_date_local: Target date
            as_of_utc: Point-in-time
            metric: "tmax" or "tmin"

        Returns:
            NBM forecast dict with percentiles and exceedance, or None
        """
        return db_lib.fetch_nbm_forecast_at_time(
            self.db_path,
            city_key=city_key,
            target_date_local=target_date_local,
            as_of_utc=as_of_utc,
            metric=metric,
        )

    def get_error_model_at_time(
        self,
        city_key: str,
        month: int,
        snapshot_hour_local: int,
        source: str,
        as_of_utc: str,
    ) -> Optional[Dict[str, Any]]:
        """
        Get error model that was available at as_of_utc.

        In practice, error models are updated infrequently (monthly or less),
        so we return the current model if it was trained before as_of_utc.

        For true temporal fidelity, we would need to version error models.
        This implementation assumes error models are stable over the backtest period.

        Args:
            city_key: City identifier
            month: Month (1-12)
            snapshot_hour_local: Hour of day for forecast snapshot
            source: Forecast source
            as_of_utc: Point-in-time

        Returns:
            Error model dict with PMF, or None
        """
        # For now, return current error model
        # TODO: Add versioning to error_models table for true as-of semantics
        return db_lib.fetch_error_model(
            self.db_path,
            city_key=city_key,
            month=month,
            snapshot_hour_local=snapshot_hour_local,
            source=source,
        )

    def get_kalshi_market_at_time(
        self,
        city_key: str,
        target_date_local: str,
        as_of_utc: str,
    ) -> List[Dict[str, Any]]:
        """
        Get Kalshi market snapshots available at as_of_utc.

        Returns markets from the most recent snapshot before as_of_utc.

        Args:
            city_key: City identifier
            target_date_local: Target date
            as_of_utc: Point-in-time

        Returns:
            List of market dicts with prices
        """
        conn = sqlite3.connect(self.db_path)

        # Find the most recent snapshot time before as_of_utc
        row = conn.execute(
            """
            SELECT snapshot_time_utc
            FROM kalshi_weather_market_snapshots
            WHERE city_key = ?
                AND target_date_local = ?
                AND snapshot_time_utc <= ?
            ORDER BY snapshot_time_utc DESC
            LIMIT 1
            """,
            (city_key, target_date_local, as_of_utc),
        ).fetchone()

        if not row:
            conn.close()
            return []

        snapshot_time = row[0]
        conn.close()

        # Get all markets at that snapshot
        return list(db_lib.fetch_kalshi_weather_markets_at_snapshot(
            self.db_path,
            snapshot_time_utc=snapshot_time,
            city_key=city_key,
            target_date_local=target_date_local,
        ))

    def get_observed_high(
        self,
        city_key: str,
        date_local: str,
    ) -> Optional[int]:
        """
        Get actual observed high temperature.

        This is the settlement truth used to evaluate predictions.
        Only call this for dates in the past.

        Args:
            city_key: City identifier
            date_local: Date (YYYY-MM-DD)

        Returns:
            Observed high temperature (F), or None if not available
        """
        conn = sqlite3.connect(self.db_path)

        row = conn.execute(
            """
            SELECT tmax_f
            FROM observed_cli
            WHERE city_key = ? AND date_local = ?
            LIMIT 1
            """,
            (city_key, date_local),
        ).fetchone()

        conn.close()

        return int(row[0]) if row else None

    def get_intraday_observation_at_time(
        self,
        city_key: str,
        target_date_local: str,
        as_of_utc: str,
    ) -> Optional[Dict[str, Any]]:
        """
        Get intraday observation available at as_of_utc.

        Used for simulating intraday trading with progress adjustments.

        Args:
            city_key: City identifier
            target_date_local: Target date
            as_of_utc: Point-in-time

        Returns:
            Intraday observation dict or None
        """
        conn = sqlite3.connect(self.db_path)

        row = conn.execute(
            """
            SELECT
                city_key, target_date_local, observation_time_utc,
                current_temp_f, max_observed_f, forecast_high_f, progress
            FROM intraday_observations
            WHERE city_key = ?
                AND target_date_local = ?
                AND observation_time_utc <= ?
            ORDER BY observation_time_utc DESC
            LIMIT 1
            """,
            (city_key, target_date_local, as_of_utc),
        ).fetchone()

        conn.close()

        if not row:
            return None

        return {
            "city_key": row[0],
            "target_date_local": row[1],
            "observation_time_utc": row[2],
            "current_temp_f": row[3],
            "max_observed_f": row[4],
            "forecast_high_f": row[5],
            "progress": row[6],
        }

    def get_date_range_with_data(
        self,
        city_key: str,
        start_date: str,
        end_date: str,
    ) -> List[str]:
        """
        Get list of dates with both forecast and observation data.

        Useful for identifying valid backtest dates.

        Args:
            city_key: City identifier
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)

        Returns:
            List of date strings
        """
        conn = sqlite3.connect(self.db_path)

        rows = conn.execute(
            """
            SELECT DISTINCT f.target_date_local
            FROM forecast_snapshots f
            JOIN observed_cli o
                ON f.city_key = o.city_key
                AND f.target_date_local = o.date_local
            WHERE f.city_key = ?
                AND f.target_date_local >= ?
                AND f.target_date_local <= ?
            ORDER BY f.target_date_local
            """,
            (city_key, start_date, end_date),
        ).fetchall()

        conn.close()

        return [row[0] for row in rows]

    def get_market_dates_with_data(
        self,
        city_key: str,
        start_date: str,
        end_date: str,
    ) -> List[str]:
        """
        Get list of dates with Kalshi market data.

        Kalshi weather markets started ~2023, so only recent dates
        will have market data for backtesting trading strategies.

        Args:
            city_key: City identifier
            start_date: Start date
            end_date: End date

        Returns:
            List of date strings with market data
        """
        conn = sqlite3.connect(self.db_path)

        rows = conn.execute(
            """
            SELECT DISTINCT target_date_local
            FROM kalshi_weather_market_snapshots
            WHERE city_key = ?
                AND target_date_local >= ?
                AND target_date_local <= ?
            ORDER BY target_date_local
            """,
            (city_key, start_date, end_date),
        ).fetchall()

        conn.close()

        return [row[0] for row in rows]


def trading_time_to_utc(
    date_local: str,
    hour_local: int,
    minute_local: int,
    timezone: str,
) -> str:
    """
    Convert local trading time to UTC string.

    Args:
        date_local: Date (YYYY-MM-DD)
        hour_local: Local hour (0-23)
        minute_local: Local minute (0-59)
        timezone: IANA timezone

    Returns:
        ISO format UTC timestamp
    """
    tz = ZoneInfo(timezone)
    dt = datetime.strptime(date_local, "%Y-%m-%d")
    local_dt = dt.replace(hour=hour_local, minute=minute_local, tzinfo=tz)
    utc_dt = local_dt.astimezone(ZoneInfo("UTC"))
    return utc_dt.isoformat()
