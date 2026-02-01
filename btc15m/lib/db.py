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

            -- Prediction samples for learning from ALL predictions (not just trades)
            CREATE TABLE IF NOT EXISTS prediction_samples (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                market_ticker TEXT NOT NULL,
                expiry_time TEXT NOT NULL,
                strike_price REAL NOT NULL,
                btc_price REAL NOT NULL,
                prediction TEXT NOT NULL,  -- 'UP' or 'DOWN'
                model_prob REAL NOT NULL,
                market_ask REAL,
                edge REAL,
                features_json TEXT,  -- JSON blob of all features
                actual_outcome TEXT,  -- Filled after settlement: 'UP' or 'DOWN'
                was_correct INTEGER,  -- 1 if correct, 0 if wrong
                was_traded INTEGER DEFAULT 0,  -- 1 if we actually traded this
                settled_at TEXT,
                UNIQUE(market_ticker)  -- One prediction per market
            );

            -- Scanner logs for dashboard
            CREATE TABLE IF NOT EXISTS scanner_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                level TEXT DEFAULT 'INFO',
                message TEXT NOT NULL
            );

            -- Indexes for faster queries
            CREATE INDEX IF NOT EXISTS idx_price_ts ON price_snapshots(timestamp);
            CREATE INDEX IF NOT EXISTS idx_indicator_ts ON indicator_snapshots(timestamp);
            CREATE INDEX IF NOT EXISTS idx_market_ts ON market_snapshots(timestamp);
            CREATE INDEX IF NOT EXISTS idx_trades_market ON paper_trades(market_ticker);
            CREATE INDEX IF NOT EXISTS idx_prediction_expiry ON prediction_samples(expiry_time);
            CREATE INDEX IF NOT EXISTS idx_prediction_outcome ON prediction_samples(actual_outcome);
            CREATE INDEX IF NOT EXISTS idx_logs_ts ON scanner_logs(timestamp);
        """)

        # Migrate: add columns to paper_trades if they don't exist
        cursor = conn.execute("PRAGMA table_info(paper_trades)")
        existing_cols = {row[1] for row in cursor.fetchall()}

        if "expiry_time" not in existing_cols:
            conn.execute("ALTER TABLE paper_trades ADD COLUMN expiry_time TEXT")
        if "strike_price" not in existing_cols:
            conn.execute("ALTER TABLE paper_trades ADD COLUMN strike_price REAL")

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


def insert_prediction_sample(
    market_ticker: str,
    expiry_time: str,
    strike_price: float,
    btc_price: float,
    prediction: str,
    model_prob: float,
    features: Dict[str, Any],
    market_ask: Optional[float] = None,
    edge: Optional[float] = None,
    was_traded: bool = False,
    timestamp: Optional[str] = None,
    db_path: Path = DEFAULT_DB_PATH,
) -> int:
    """Insert a prediction sample for learning (even without trading).

    Returns:
        Row ID of inserted sample, or 0 if already exists
    """
    import json
    if timestamp is None:
        timestamp = datetime.now(timezone.utc).isoformat()

    features_json = json.dumps(features)

    with get_connection(db_path) as conn:
        try:
            cursor = conn.execute(
                """INSERT OR REPLACE INTO prediction_samples
                   (timestamp, market_ticker, expiry_time, strike_price, btc_price,
                    prediction, model_prob, market_ask, edge, features_json, was_traded)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (timestamp, market_ticker, expiry_time, strike_price, btc_price,
                 prediction, model_prob, market_ask, edge, features_json, 1 if was_traded else 0),
            )
            conn.commit()
            return cursor.lastrowid
        except Exception as e:
            print(f"[db] Error inserting prediction: {e}")
            return 0


def get_unsettled_predictions(db_path: Path = DEFAULT_DB_PATH) -> List[Dict[str, Any]]:
    """Get all prediction samples that haven't been settled yet."""
    with get_connection(db_path) as conn:
        rows = conn.execute(
            """SELECT id, timestamp, market_ticker, expiry_time, strike_price,
                      btc_price, prediction, model_prob, market_ask, edge,
                      features_json, was_traded
               FROM prediction_samples
               WHERE actual_outcome IS NULL
               ORDER BY expiry_time ASC"""
        ).fetchall()

        return [dict(row) for row in rows]


def settle_prediction_sample(
    prediction_id: int,
    actual_outcome: str,
    db_path: Path = DEFAULT_DB_PATH,
) -> None:
    """Settle a prediction sample with actual outcome."""
    settled_at = datetime.now(timezone.utc).isoformat()

    with get_connection(db_path) as conn:
        # Get the prediction to check if it was correct
        row = conn.execute(
            "SELECT prediction FROM prediction_samples WHERE id = ?",
            (prediction_id,)
        ).fetchone()

        if row:
            was_correct = 1 if row["prediction"] == actual_outcome else 0
            conn.execute(
                """UPDATE prediction_samples
                   SET actual_outcome = ?, was_correct = ?, settled_at = ?
                   WHERE id = ?""",
                (actual_outcome, was_correct, settled_at, prediction_id),
            )
            conn.commit()


def get_prediction_stats(db_path: Path = DEFAULT_DB_PATH) -> Dict[str, Any]:
    """Get accuracy stats from all prediction samples."""
    with get_connection(db_path) as conn:
        row = conn.execute(
            """SELECT
                 COUNT(*) as total,
                 SUM(CASE WHEN actual_outcome IS NOT NULL THEN 1 ELSE 0 END) as settled,
                 SUM(was_correct) as correct,
                 SUM(CASE WHEN was_traded = 1 THEN 1 ELSE 0 END) as traded,
                 SUM(CASE WHEN was_traded = 1 AND was_correct = 1 THEN 1 ELSE 0 END) as traded_correct
               FROM prediction_samples"""
        ).fetchone()

        total = row["total"] or 0
        settled = row["settled"] or 0
        correct = row["correct"] or 0
        traded = row["traded"] or 0
        traded_correct = row["traded_correct"] or 0

        return {
            "total_predictions": total,
            "settled": settled,
            "pending": total - settled,
            "correct": correct,
            "accuracy": correct / settled if settled > 0 else None,
            "traded": traded,
            "traded_correct": traded_correct,
            "traded_accuracy": traded_correct / traded if traded > 0 else None,
        }


def push_log(
    message: str,
    level: str = "INFO",
    db_path: Path = DEFAULT_DB_PATH,
) -> None:
    """Push a log message to the database for dashboard display."""
    timestamp = datetime.now(timezone.utc).isoformat()

    try:
        with get_connection(db_path) as conn:
            conn.execute(
                "INSERT INTO scanner_logs (timestamp, level, message) VALUES (?, ?, ?)",
                (timestamp, level, message),
            )
            # Keep only last 500 logs
            conn.execute(
                "DELETE FROM scanner_logs WHERE id NOT IN (SELECT id FROM scanner_logs ORDER BY id DESC LIMIT 500)"
            )
            conn.commit()
    except Exception:
        pass  # Don't let logging errors crash the scanner


def get_recent_logs(
    limit: int = 50,
    db_path: Path = DEFAULT_DB_PATH,
) -> List[Dict[str, Any]]:
    """Get recent logs for dashboard."""
    try:
        with get_connection(db_path) as conn:
            rows = conn.execute(
                "SELECT timestamp, level, message FROM scanner_logs ORDER BY id DESC LIMIT ?",
                (limit,),
            ).fetchall()
            return [dict(row) for row in reversed(rows)]
    except Exception:
        return []


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


def ensure_settings_table(db_path: Path = DEFAULT_DB_PATH) -> None:
    """Create settings table if it doesn't exist."""
    with get_connection(db_path) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS scanner_settings (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
        """)
        conn.commit()


def get_setting(key: str, default: str = None, db_path: Path = DEFAULT_DB_PATH) -> Optional[str]:
    """Get a setting value."""
    ensure_settings_table(db_path)
    with get_connection(db_path) as conn:
        row = conn.execute(
            "SELECT value FROM scanner_settings WHERE key = ?", (key,)
        ).fetchone()
        return row["value"] if row else default


def set_setting(key: str, value: str, db_path: Path = DEFAULT_DB_PATH) -> None:
    """Set a setting value."""
    ensure_settings_table(db_path)
    updated_at = datetime.now(timezone.utc).isoformat()
    with get_connection(db_path) as conn:
        conn.execute(
            """INSERT OR REPLACE INTO scanner_settings (key, value, updated_at)
               VALUES (?, ?, ?)""",
            (key, value, updated_at),
        )
        conn.commit()


def get_pnl_by_confidence_bucket(
    bucket_size: float = 0.05,
    min_trades: int = 3,
    db_path: Path = DEFAULT_DB_PATH,
) -> List[Dict[str, Any]]:
    """Get P&L statistics grouped by model confidence buckets.

    Args:
        bucket_size: Size of each bucket (0.05 = 5% buckets like 55-60%, 60-65%)
        min_trades: Minimum settled trades required for bucket to be valid

    Returns:
        List of buckets with stats: bucket_min, bucket_max, trades, wins, losses, total_pnl, avg_pnl
    """
    with get_connection(db_path) as conn:
        rows = conn.execute(
            """SELECT
                 model_prob,
                 outcome,
                 pnl
               FROM paper_trades
               WHERE outcome IS NOT NULL AND model_prob IS NOT NULL
               ORDER BY model_prob"""
        ).fetchall()

    if not rows:
        return []

    # Group into buckets
    buckets = {}
    for row in rows:
        prob = row["model_prob"]
        # Round down to bucket (e.g., 0.57 -> 0.55 for 5% buckets)
        bucket_min = int(prob / bucket_size) * bucket_size
        bucket_max = bucket_min + bucket_size

        key = (bucket_min, bucket_max)
        if key not in buckets:
            buckets[key] = {"wins": 0, "losses": 0, "total_pnl": 0.0, "trades": []}

        buckets[key]["trades"].append(row["pnl"] or 0)
        if row["outcome"] == "WIN":
            buckets[key]["wins"] += 1
        else:
            buckets[key]["losses"] += 1
        buckets[key]["total_pnl"] += row["pnl"] or 0

    # Convert to list with stats
    result = []
    for (bucket_min, bucket_max), data in sorted(buckets.items()):
        trade_count = len(data["trades"])
        if trade_count >= min_trades:
            result.append({
                "bucket_min": bucket_min,
                "bucket_max": bucket_max,
                "trades": trade_count,
                "wins": data["wins"],
                "losses": data["losses"],
                "total_pnl": data["total_pnl"],
                "avg_pnl": data["total_pnl"] / trade_count,
                "win_rate": data["wins"] / trade_count if trade_count > 0 else 0,
            })

    return result


def calculate_optimal_threshold(
    min_bucket_trades: int = 3,
    db_path: Path = DEFAULT_DB_PATH,
) -> Dict[str, Any]:
    """Calculate optimal confidence threshold based on historical performance.

    Returns:
        Dict with: optimal_threshold, analysis (list of bucket stats), reason
    """
    buckets = get_pnl_by_confidence_bucket(bucket_size=0.05, min_trades=min_bucket_trades, db_path=db_path)

    if not buckets:
        return {
            "optimal_threshold": 0.60,  # Default
            "analysis": [],
            "reason": "Not enough data - using default 60%",
        }

    # Find lowest bucket with positive avg P&L
    profitable_buckets = [b for b in buckets if b["avg_pnl"] > 0]

    if not profitable_buckets:
        # No profitable buckets - use highest available
        highest = max(buckets, key=lambda b: b["bucket_min"])
        return {
            "optimal_threshold": min(highest["bucket_max"], 0.80),
            "analysis": buckets,
            "reason": f"No profitable buckets found - raising to {highest['bucket_max']*100:.0f}%",
        }

    # Use lowest profitable bucket
    lowest_profitable = min(profitable_buckets, key=lambda b: b["bucket_min"])

    return {
        "optimal_threshold": lowest_profitable["bucket_min"],
        "analysis": buckets,
        "reason": f"Bucket {lowest_profitable['bucket_min']*100:.0f}-{lowest_profitable['bucket_max']*100:.0f}% is profitable (avg ${lowest_profitable['avg_pnl']:+.2f})",
    }


def get_recent_pnl(lookback_trades: int = 10, db_path: Path = DEFAULT_DB_PATH) -> Dict[str, Any]:
    """Get P&L for recent trades to detect losing streaks.

    Returns:
        Dict with: recent_pnl, recent_trades, consecutive_losses, should_paper_mode
    """
    with get_connection(db_path) as conn:
        rows = conn.execute(
            """SELECT outcome, pnl
               FROM paper_trades
               WHERE outcome IS NOT NULL
               ORDER BY settled_at DESC
               LIMIT ?""",
            (lookback_trades,),
        ).fetchall()

    if not rows:
        return {
            "recent_pnl": 0.0,
            "recent_trades": 0,
            "consecutive_losses": 0,
            "should_paper_mode": False,
        }

    recent_pnl = sum(row["pnl"] or 0 for row in rows)

    # Count consecutive losses from most recent
    consecutive_losses = 0
    for row in rows:
        if row["outcome"] == "LOSS":
            consecutive_losses += 1
        else:
            break

    # Paper mode triggers:
    # - 5+ consecutive losses
    # - Lost more than $5 in last 10 trades
    should_paper = consecutive_losses >= 5 or recent_pnl < -5.0

    return {
        "recent_pnl": recent_pnl,
        "recent_trades": len(rows),
        "consecutive_losses": consecutive_losses,
        "should_paper_mode": should_paper,
    }
