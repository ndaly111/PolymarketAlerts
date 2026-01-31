"""SQLite database for BTC 15-min data collection and paper trading."""

from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, List, Dict, Any

DEFAULT_DB_PATH = Path(__file__).parent.parent / "data" / "btc15m.db"


def ensure_schema(db_path: Path = DEFAULT_DB_PATH) -> None:
    """Create database tables if they don't exist."""
    db_path.parent.mkdir(parents=True, exist_ok=True)

    with sqlite3.connect(str(db_path)) as conn:
        conn.executescript("""
            -- Price snapshots collected every 15 seconds
            CREATE TABLE IF NOT EXISTS price_snapshots (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                binance_price REAL NOT NULL,
                chainlink_price REAL,
                price_diff REAL,
                UNIQUE(timestamp)
            );

            -- Technical indicators calculated from candle data
            CREATE TABLE IF NOT EXISTS indicator_snapshots (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                rsi_14 REAL,
                macd REAL,
                macd_signal REAL,
                macd_histogram REAL,
                vwap REAL,
                volatility_20 REAL,
                momentum_10 REAL,
                sma_20 REAL,
                price_vs_vwap REAL,
                UNIQUE(timestamp)
            );

            -- Kalshi market snapshots
            CREATE TABLE IF NOT EXISTS market_snapshots (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                market_ticker TEXT NOT NULL,
                expiry_time TEXT,
                strike_price REAL,
                up_yes_price REAL,
                up_no_price REAL,
                down_yes_price REAL,
                down_no_price REAL,
                time_to_expiry_sec INTEGER,
                UNIQUE(timestamp, market_ticker)
            );

            -- Outcomes for training (filled after market settles)
            CREATE TABLE IF NOT EXISTS outcomes (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                market_ticker TEXT NOT NULL,
                expiry_time TEXT NOT NULL,
                strike_price REAL,
                settlement_price REAL,
                outcome TEXT,  -- 'UP' or 'DOWN'
                UNIQUE(market_ticker)
            );

            -- Paper trades
            CREATE TABLE IF NOT EXISTS paper_trades (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                market_ticker TEXT NOT NULL,
                side TEXT NOT NULL,  -- 'UP' or 'DOWN'
                entry_price REAL NOT NULL,
                stake REAL NOT NULL,
                model_prob REAL,
                market_prob REAL,
                edge REAL,
                expiry_time TEXT,
                strike_price REAL,
                outcome TEXT,  -- 'WIN', 'LOSS', or NULL if pending
                pnl REAL,
                settled_at TEXT
            );

            -- Model predictions for tracking accuracy
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                market_ticker TEXT NOT NULL,
                model_up_prob REAL NOT NULL,
                model_down_prob REAL NOT NULL,
                market_up_prob REAL,
                market_down_prob REAL,
                recommended_side TEXT,
                edge REAL,
                actual_outcome TEXT,
                correct INTEGER
            );

            -- Indexes for faster queries
            CREATE INDEX IF NOT EXISTS idx_price_ts ON price_snapshots(timestamp);
            CREATE INDEX IF NOT EXISTS idx_indicator_ts ON indicator_snapshots(timestamp);
            CREATE INDEX IF NOT EXISTS idx_market_ts ON market_snapshots(timestamp);
            CREATE INDEX IF NOT EXISTS idx_trades_market ON paper_trades(market_ticker);
        """)
        conn.commit()


@contextmanager
def get_connection(db_path: Path = DEFAULT_DB_PATH):
    """Get a database connection context manager."""
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    try:
        yield conn
    finally:
        conn.close()


def insert_price_snapshot(
    binance_price: float,
    chainlink_price: Optional[float] = None,
    timestamp: Optional[str] = None,
    db_path: Path = DEFAULT_DB_PATH,
) -> None:
    """Insert a price snapshot."""
    if timestamp is None:
        timestamp = datetime.now(timezone.utc).isoformat()

    price_diff = None
    if chainlink_price is not None:
        price_diff = binance_price - chainlink_price

    with get_connection(db_path) as conn:
        conn.execute(
            """INSERT OR REPLACE INTO price_snapshots
               (timestamp, binance_price, chainlink_price, price_diff)
               VALUES (?, ?, ?, ?)""",
            (timestamp, binance_price, chainlink_price, price_diff),
        )
        conn.commit()


def insert_indicator_snapshot(
    rsi: Optional[float] = None,
    macd: Optional[float] = None,
    macd_signal: Optional[float] = None,
    macd_histogram: Optional[float] = None,
    vwap: Optional[float] = None,
    volatility: Optional[float] = None,
    momentum: Optional[float] = None,
    sma_20: Optional[float] = None,
    price_vs_vwap: Optional[float] = None,
    timestamp: Optional[str] = None,
    db_path: Path = DEFAULT_DB_PATH,
) -> None:
    """Insert an indicator snapshot."""
    if timestamp is None:
        timestamp = datetime.now(timezone.utc).isoformat()

    with get_connection(db_path) as conn:
        conn.execute(
            """INSERT OR REPLACE INTO indicator_snapshots
               (timestamp, rsi_14, macd, macd_signal, macd_histogram,
                vwap, volatility_20, momentum_10, sma_20, price_vs_vwap)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (timestamp, rsi, macd, macd_signal, macd_histogram,
             vwap, volatility, momentum, sma_20, price_vs_vwap),
        )
        conn.commit()


def insert_market_snapshot(
    market_ticker: str,
    expiry_time: Optional[str] = None,
    strike_price: Optional[float] = None,
    up_yes_price: Optional[float] = None,
    up_no_price: Optional[float] = None,
    down_yes_price: Optional[float] = None,
    down_no_price: Optional[float] = None,
    time_to_expiry_sec: Optional[int] = None,
    timestamp: Optional[str] = None,
    db_path: Path = DEFAULT_DB_PATH,
) -> None:
    """Insert a market snapshot."""
    if timestamp is None:
        timestamp = datetime.now(timezone.utc).isoformat()

    with get_connection(db_path) as conn:
        conn.execute(
            """INSERT OR REPLACE INTO market_snapshots
               (timestamp, market_ticker, expiry_time, strike_price,
                up_yes_price, up_no_price, down_yes_price, down_no_price,
                time_to_expiry_sec)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (timestamp, market_ticker, expiry_time, strike_price,
             up_yes_price, up_no_price, down_yes_price, down_no_price,
             time_to_expiry_sec),
        )
        conn.commit()


def insert_paper_trade(
    market_ticker: str,
    side: str,
    entry_price: float,
    stake: float,
    model_prob: Optional[float] = None,
    market_prob: Optional[float] = None,
    edge: Optional[float] = None,
    expiry_time: Optional[str] = None,
    strike_price: Optional[float] = None,
    timestamp: Optional[str] = None,
    db_path: Path = DEFAULT_DB_PATH,
) -> int:
    """Insert a paper trade and return its ID."""
    if timestamp is None:
        timestamp = datetime.now(timezone.utc).isoformat()

    with get_connection(db_path) as conn:
        cursor = conn.execute(
            """INSERT INTO paper_trades
               (timestamp, market_ticker, side, entry_price, stake,
                model_prob, market_prob, edge, expiry_time, strike_price)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (timestamp, market_ticker, side, entry_price, stake,
             model_prob, market_prob, edge, expiry_time, strike_price),
        )
        conn.commit()
        return cursor.lastrowid


def settle_paper_trade(
    trade_id: int,
    outcome: str,
    pnl: float,
    db_path: Path = DEFAULT_DB_PATH,
) -> None:
    """Settle a paper trade with outcome."""
    settled_at = datetime.now(timezone.utc).isoformat()

    with get_connection(db_path) as conn:
        conn.execute(
            """UPDATE paper_trades
               SET outcome = ?, pnl = ?, settled_at = ?
               WHERE id = ?""",
            (outcome, pnl, settled_at, trade_id),
        )
        conn.commit()


def get_paper_trade_summary(db_path: Path = DEFAULT_DB_PATH) -> Dict[str, Any]:
    """Get summary of paper trading performance."""
    with get_connection(db_path) as conn:
        row = conn.execute(
            """SELECT
                 COUNT(*) as total_trades,
                 SUM(CASE WHEN outcome = 'WIN' THEN 1 ELSE 0 END) as wins,
                 SUM(CASE WHEN outcome = 'LOSS' THEN 1 ELSE 0 END) as losses,
                 SUM(CASE WHEN outcome IS NULL THEN 1 ELSE 0 END) as pending,
                 SUM(pnl) as total_pnl,
                 AVG(edge) as avg_edge
               FROM paper_trades"""
        ).fetchone()

        return {
            "total_trades": row["total_trades"] or 0,
            "wins": row["wins"] or 0,
            "losses": row["losses"] or 0,
            "pending": row["pending"] or 0,
            "total_pnl": row["total_pnl"] or 0.0,
            "avg_edge": row["avg_edge"],
            "win_rate": (row["wins"] / (row["wins"] + row["losses"]))
                        if (row["wins"] or 0) + (row["losses"] or 0) > 0 else None,
        }


def get_recent_snapshots(
    limit: int = 100,
    db_path: Path = DEFAULT_DB_PATH,
) -> List[Dict[str, Any]]:
    """Get recent price and indicator snapshots joined."""
    with get_connection(db_path) as conn:
        rows = conn.execute(
            """SELECT p.timestamp, p.binance_price, p.chainlink_price,
                      i.rsi_14, i.macd, i.macd_histogram, i.vwap,
                      i.volatility_20, i.momentum_10
               FROM price_snapshots p
               LEFT JOIN indicator_snapshots i ON p.timestamp = i.timestamp
               ORDER BY p.timestamp DESC
               LIMIT ?""",
            (limit,),
        ).fetchall()

        return [dict(row) for row in rows]


def get_pending_trades(db_path: Path = DEFAULT_DB_PATH) -> List[Dict[str, Any]]:
    """Get all pending (unsettled) trades."""
    with get_connection(db_path) as conn:
        rows = conn.execute(
            """SELECT id, timestamp, market_ticker, side, entry_price,
                      stake, model_prob, market_prob, edge,
                      expiry_time, strike_price
               FROM paper_trades
               WHERE outcome IS NULL
               ORDER BY timestamp DESC"""
        ).fetchall()

        return [dict(row) for row in rows]


def get_all_trades(db_path: Path = DEFAULT_DB_PATH) -> List[Dict[str, Any]]:
    """Get all trades with full details."""
    with get_connection(db_path) as conn:
        rows = conn.execute(
            """SELECT id, timestamp, market_ticker, side, entry_price,
                      stake, model_prob, market_prob, edge,
                      expiry_time, strike_price, outcome, pnl, settled_at
               FROM paper_trades
               ORDER BY timestamp DESC"""
        ).fetchall()

        return [dict(row) for row in rows]


def get_daily_stats(db_path: Path = DEFAULT_DB_PATH) -> Dict[str, Any]:
    """Get trading stats for the current day."""
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    with get_connection(db_path) as conn:
        row = conn.execute(
            """SELECT
                 COUNT(*) as trades_today,
                 SUM(CASE WHEN outcome = 'WIN' THEN 1 ELSE 0 END) as wins,
                 SUM(CASE WHEN outcome = 'LOSS' THEN 1 ELSE 0 END) as losses,
                 SUM(CASE WHEN outcome IS NULL THEN 1 ELSE 0 END) as pending,
                 SUM(pnl) as pnl_today,
                 AVG(edge) as avg_edge,
                 AVG(model_prob) as avg_confidence
               FROM paper_trades
               WHERE date(timestamp) = ?""",
            (today,),
        ).fetchone()

        win_count = row["wins"] or 0
        loss_count = row["losses"] or 0

        return {
            "date": today,
            "trades_today": row["trades_today"] or 0,
            "wins": win_count,
            "losses": loss_count,
            "pending": row["pending"] or 0,
            "pnl_today": row["pnl_today"] or 0.0,
            "avg_edge": row["avg_edge"],
            "avg_confidence": row["avg_confidence"],
            "win_rate": win_count / (win_count + loss_count) if (win_count + loss_count) > 0 else None,
        }
