"""
Unified database schema and operations.

Consolidates DB operations from props_auto_trade.py and sports_auto_trade.py
with support for new columns (net_edge, outliers_removed, etc).
"""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
from zoneinfo import ZoneInfo


# Timezone for daily boundaries (Eastern Time)
ET = ZoneInfo("America/New_York")


def ensure_schema(db_path: Path, table_type: str = "props") -> None:
    """
    Create database tables if they don't exist.

    Also handles migrations to add new columns.

    Args:
        db_path: Path to SQLite database
        table_type: "props" or "sports" for schema type
    """
    conn = sqlite3.connect(str(db_path))
    cur = conn.cursor()

    if table_type == "props":
        cur.execute("""
            CREATE TABLE IF NOT EXISTS trades (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                trade_date TEXT NOT NULL,
                ticker TEXT NOT NULL,
                player_name TEXT NOT NULL,
                stat_type TEXT NOT NULL,
                line REAL NOT NULL,
                side TEXT NOT NULL,
                quantity INTEGER NOT NULL,
                limit_price_cents INTEGER NOT NULL,
                fair_prob REAL NOT NULL,
                edge REAL NOT NULL,
                books_count INTEGER NOT NULL,
                order_id TEXT,
                status TEXT NOT NULL,
                fill_price_cents INTEGER,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                settled INTEGER DEFAULT 0,
                won INTEGER,
                payout_cents INTEGER,
                settled_at TEXT,
                -- New columns for refactor
                net_edge REAL,
                outliers_removed INTEGER DEFAULT 0,
                consensus_spread REAL,
                validation_flags TEXT,
                UNIQUE(trade_date, ticker)
            )
        """)
    else:  # sports
        cur.execute("""
            CREATE TABLE IF NOT EXISTS trades (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                trade_date TEXT NOT NULL,
                ticker TEXT NOT NULL,
                event_title TEXT NOT NULL,
                line_type TEXT NOT NULL,
                line_label TEXT NOT NULL,
                side TEXT NOT NULL,
                quantity INTEGER NOT NULL,
                limit_price_cents INTEGER NOT NULL,
                fair_prob REAL NOT NULL,
                edge REAL NOT NULL,
                books_count INTEGER NOT NULL,
                order_id TEXT,
                status TEXT NOT NULL,
                fill_price_cents INTEGER,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                settled INTEGER DEFAULT 0,
                won INTEGER,
                payout_cents INTEGER,
                settled_at TEXT,
                -- New columns for refactor
                net_edge REAL,
                outliers_removed INTEGER DEFAULT 0,
                consensus_spread REAL,
                validation_flags TEXT,
                UNIQUE(trade_date, ticker)
            )
        """)

    # Common tables
    cur.execute("""
        CREATE TABLE IF NOT EXISTS daily_stats (
            date TEXT PRIMARY KEY,
            trades_attempted INTEGER DEFAULT 0,
            trades_filled INTEGER DEFAULT 0,
            trades_cancelled INTEGER DEFAULT 0,
            total_edge_cents REAL DEFAULT 0
        )
    """)

    if table_type == "props":
        cur.execute("""
            CREATE TABLE IF NOT EXISTS scanned_opportunities (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                scan_date TEXT NOT NULL,
                scan_time TEXT NOT NULL,
                ticker TEXT NOT NULL,
                player_name TEXT NOT NULL,
                stat_type TEXT NOT NULL,
                line REAL NOT NULL,
                side TEXT NOT NULL,
                kalshi_bid INTEGER,
                kalshi_ask INTEGER,
                kalshi_prob REAL NOT NULL,
                fair_prob REAL NOT NULL,
                edge REAL NOT NULL,
                books_count INTEGER NOT NULL,
                volume INTEGER,
                event_time TEXT,
                result TEXT,
                settled_at TEXT,
                -- New columns
                net_edge REAL,
                outliers_removed INTEGER DEFAULT 0,
                UNIQUE(scan_date, ticker, side)
            )
        """)
    else:  # sports
        cur.execute("""
            CREATE TABLE IF NOT EXISTS scanned_opportunities (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                scan_date TEXT NOT NULL,
                scan_time TEXT NOT NULL,
                ticker TEXT NOT NULL,
                event_title TEXT NOT NULL,
                line_type TEXT NOT NULL,
                line_label TEXT NOT NULL,
                side TEXT NOT NULL,
                kalshi_bid INTEGER,
                kalshi_ask INTEGER,
                fair_prob REAL NOT NULL,
                edge REAL NOT NULL,
                books_count INTEGER NOT NULL,
                commence_time TEXT,
                -- New columns
                net_edge REAL,
                outliers_removed INTEGER DEFAULT 0,
                UNIQUE(scan_date, ticker, side)
            )
        """)

    # Run migrations for new columns
    _migrate_schema(cur)

    conn.commit()
    conn.close()


def _migrate_schema(cur: sqlite3.Cursor) -> None:
    """Add new columns to existing tables (backward compatible)."""

    # Check existing columns in trades table
    try:
        columns = {row[1] for row in cur.execute("PRAGMA table_info(trades);").fetchall()}
    except Exception:
        columns = set()

    # Add new columns if missing
    migrations = [
        ("net_edge", "REAL"),
        ("outliers_removed", "INTEGER DEFAULT 0"),
        ("consensus_spread", "REAL"),
        ("validation_flags", "TEXT"),
        ("settled", "INTEGER DEFAULT 0"),
        ("won", "INTEGER"),
        ("payout_cents", "INTEGER"),
        ("settled_at", "TEXT"),
    ]

    for col_name, col_type in migrations:
        if col_name not in columns:
            try:
                cur.execute(f"ALTER TABLE trades ADD COLUMN {col_name} {col_type};")
            except sqlite3.OperationalError:
                pass  # Column might already exist

    # Also migrate scanned_opportunities
    try:
        opp_columns = {row[1] for row in cur.execute("PRAGMA table_info(scanned_opportunities);").fetchall()}
    except Exception:
        opp_columns = set()

    opp_migrations = [
        ("net_edge", "REAL"),
        ("outliers_removed", "INTEGER DEFAULT 0"),
    ]

    for col_name, col_type in opp_migrations:
        if col_name not in opp_columns:
            try:
                cur.execute(f"ALTER TABLE scanned_opportunities ADD COLUMN {col_name} {col_type};")
            except sqlite3.OperationalError:
                pass


def get_trades_today(db_path: Path) -> int:
    """
    Get number of filled trades today (ET timezone).

    Only counts filled trades, not dry_run or other statuses.
    """
    today = datetime.now(ET).strftime("%Y-%m-%d")
    conn = sqlite3.connect(str(db_path))
    cur = conn.cursor()
    cur.execute("""
        SELECT COUNT(*) FROM trades
        WHERE trade_date = ? AND status IN ('filled', 'FILLED')
    """, (today,))
    row = cur.fetchone()
    conn.close()
    return row[0] if row else 0


def already_traded_today(db_path: Path, ticker: str) -> bool:
    """Check if we already traded this ticker today (ET timezone)."""
    today = datetime.now(ET).strftime("%Y-%m-%d")
    conn = sqlite3.connect(str(db_path))
    cur = conn.cursor()
    cur.execute(
        "SELECT 1 FROM trades WHERE trade_date = ? AND ticker = ?",
        (today, ticker)
    )
    row = cur.fetchone()
    conn.close()
    return row is not None


def increment_trades_today(db_path: Path, filled: bool = False) -> None:
    """Increment trade counter for today (ET timezone)."""
    today = datetime.now(ET).strftime("%Y-%m-%d")
    conn = sqlite3.connect(str(db_path))
    cur = conn.cursor()

    if filled:
        cur.execute("""
            INSERT INTO daily_stats (date, trades_attempted, trades_filled)
            VALUES (?, 1, 1)
            ON CONFLICT(date) DO UPDATE SET
                trades_attempted = trades_attempted + 1,
                trades_filled = trades_filled + 1
        """, (today,))
    else:
        cur.execute("""
            INSERT INTO daily_stats (date, trades_attempted)
            VALUES (?, 1)
            ON CONFLICT(date) DO UPDATE SET trades_attempted = trades_attempted + 1
        """, (today,))

    conn.commit()
    conn.close()


def record_trade(
    db_path: Path,
    trade_data: Dict[str, Any],
    table_type: str = "props",
) -> None:
    """
    Record a trade in the database (ET timezone).

    Args:
        db_path: Path to database
        trade_data: Dict with trade fields
        table_type: "props" or "sports"
    """
    now = datetime.now(ET).isoformat()
    today = datetime.now(ET).strftime("%Y-%m-%d")

    conn = sqlite3.connect(str(db_path))
    cur = conn.cursor()

    if table_type == "props":
        cur.execute("""
            INSERT OR REPLACE INTO trades (
                trade_date, ticker, player_name, stat_type, line, side,
                quantity, limit_price_cents, fair_prob, edge, books_count,
                order_id, status, fill_price_cents, created_at, updated_at,
                net_edge, outliers_removed, consensus_spread, validation_flags
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            today,
            trade_data.get("ticker", ""),
            trade_data.get("player_name", ""),
            trade_data.get("stat_type", ""),
            trade_data.get("line", 0),
            trade_data.get("side", ""),
            trade_data.get("quantity", 1),
            trade_data.get("limit_price_cents", 0),
            trade_data.get("fair_prob", 0.5),
            trade_data.get("edge", 0),
            trade_data.get("books_count", 0),
            trade_data.get("order_id"),
            trade_data.get("status", "unknown"),
            trade_data.get("fill_price_cents"),
            now,
            now,
            trade_data.get("net_edge"),
            trade_data.get("outliers_removed", 0),
            trade_data.get("consensus_spread"),
            json.dumps(trade_data.get("validation_flags")) if trade_data.get("validation_flags") else None,
        ))
    else:  # sports
        cur.execute("""
            INSERT OR REPLACE INTO trades (
                trade_date, ticker, event_title, line_type, line_label, side,
                quantity, limit_price_cents, fair_prob, edge, books_count,
                order_id, status, fill_price_cents, created_at, updated_at,
                net_edge, outliers_removed, consensus_spread, validation_flags
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            today,
            trade_data.get("ticker", ""),
            trade_data.get("event_title", ""),
            trade_data.get("line_type", ""),
            trade_data.get("line_label", ""),
            trade_data.get("side", ""),
            trade_data.get("quantity", 1),
            trade_data.get("limit_price_cents", 0),
            trade_data.get("fair_prob", 0.5),
            trade_data.get("edge", 0),
            trade_data.get("books_count", 0),
            trade_data.get("order_id"),
            trade_data.get("status", "unknown"),
            trade_data.get("fill_price_cents"),
            now,
            now,
            trade_data.get("net_edge"),
            trade_data.get("outliers_removed", 0),
            trade_data.get("consensus_spread"),
            json.dumps(trade_data.get("validation_flags")) if trade_data.get("validation_flags") else None,
        ))

    conn.commit()
    conn.close()


def log_scanned_opportunity(
    db_path: Path,
    opportunity_data: Dict[str, Any],
    table_type: str = "props",
) -> None:
    """Log a scanned opportunity for later analysis (ET timezone)."""
    now = datetime.now(ET)
    scan_date = now.strftime("%Y-%m-%d")
    scan_time = now.isoformat()

    conn = sqlite3.connect(str(db_path))
    cur = conn.cursor()

    if table_type == "props":
        cur.execute("""
            INSERT OR REPLACE INTO scanned_opportunities (
                scan_date, scan_time, ticker, player_name, stat_type, line, side,
                kalshi_bid, kalshi_ask, kalshi_prob, fair_prob, edge, books_count,
                volume, event_time, net_edge, outliers_removed
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            scan_date,
            scan_time,
            opportunity_data.get("ticker", ""),
            opportunity_data.get("player_name", ""),
            opportunity_data.get("stat_type", ""),
            opportunity_data.get("line", 0),
            opportunity_data.get("side", ""),
            opportunity_data.get("kalshi_bid", 0),
            opportunity_data.get("kalshi_ask", 100),
            opportunity_data.get("kalshi_prob", 0.5),
            opportunity_data.get("fair_prob", 0.5),
            opportunity_data.get("edge", 0),
            opportunity_data.get("books_count", 0),
            opportunity_data.get("volume", 0),
            opportunity_data.get("event_time", ""),
            opportunity_data.get("net_edge"),
            opportunity_data.get("outliers_removed", 0),
        ))
    else:  # sports
        cur.execute("""
            INSERT OR REPLACE INTO scanned_opportunities (
                scan_date, scan_time, ticker, event_title, line_type, line_label, side,
                kalshi_bid, kalshi_ask, fair_prob, edge, books_count, commence_time,
                net_edge, outliers_removed
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            scan_date,
            scan_time,
            opportunity_data.get("ticker", ""),
            opportunity_data.get("event_title", ""),
            opportunity_data.get("line_type", ""),
            opportunity_data.get("line_label", ""),
            opportunity_data.get("side", ""),
            opportunity_data.get("kalshi_bid", 0),
            opportunity_data.get("kalshi_ask", 100),
            opportunity_data.get("fair_prob", 0.5),
            opportunity_data.get("edge", 0),
            opportunity_data.get("books_count", 0),
            opportunity_data.get("commence_time", ""),
            opportunity_data.get("net_edge"),
            opportunity_data.get("outliers_removed", 0),
        ))

    conn.commit()
    conn.close()


def get_recent_trades(
    db_path: Path,
    days: int = 7,
    status: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Get recent trades from database."""
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()

    query = "SELECT * FROM trades WHERE trade_date >= date('now', ?)"
    params: List[Any] = [f"-{days} days"]

    if status:
        query += " AND status = ?"
        params.append(status)

    query += " ORDER BY created_at DESC"

    cur.execute(query, params)
    rows = cur.fetchall()
    conn.close()

    return [dict(row) for row in rows]
