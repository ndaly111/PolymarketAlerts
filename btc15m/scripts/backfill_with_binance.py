#!/usr/bin/env python3
"""Backfill market context features using Binance historical data.

Fetches BTC prices at trade timestamps to calculate:
- time_to_expiry_min
- distance_to_strike_pct
- above_strike
- time_distance_interaction
"""

import sqlite3
import json
import time
import requests
from datetime import datetime, timezone
from pathlib import Path

DB_PATH = Path(__file__).parent.parent / "data" / "btc15m.db"

# Cache for Binance klines to avoid repeated API calls
PRICE_CACHE = {}


def fetch_btc_price_at_time(timestamp_ms: int) -> float:
    """Fetch BTC price from Binance at a specific timestamp."""
    # Round to nearest minute for caching
    minute_ms = (timestamp_ms // 60000) * 60000

    if minute_ms in PRICE_CACHE:
        return PRICE_CACHE[minute_ms]

    # Fetch 1-minute kline that contains this timestamp
    url = "https://api.binance.us/api/v3/klines"
    params = {
        "symbol": "BTCUSDT",
        "interval": "1m",
        "startTime": minute_ms,
        "limit": 1,
    }

    try:
        resp = requests.get(url, params=params, timeout=10)
        resp.raise_for_status()
        data = resp.json()

        if data:
            # Use close price of the minute
            close_price = float(data[0][4])
            PRICE_CACHE[minute_ms] = close_price
            return close_price
    except Exception as e:
        print(f"  Error fetching price: {e}")

    return 0.0


def backfill_paper_trades():
    """Add BTC price and market context to paper_trades."""
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()

    # Check if btc_price column exists, add if not
    cur.execute("PRAGMA table_info(paper_trades)")
    existing_cols = {r[1] for r in cur.fetchall()}

    if "btc_price" not in existing_cols:
        print("Adding btc_price column to paper_trades")
        cur.execute("ALTER TABLE paper_trades ADD COLUMN btc_price REAL")

    # Get all trades that need btc_price
    rows = cur.execute("""
        SELECT id, timestamp, strike_price
        FROM paper_trades
        WHERE btc_price IS NULL OR btc_price = 0
        ORDER BY id
    """).fetchall()

    print(f"Found {len(rows)} paper trades to backfill")

    for i, row in enumerate(rows):
        try:
            # Parse timestamp
            ts = row["timestamp"]
            trade_time = datetime.fromisoformat(ts.replace("Z", "+00:00"))
            timestamp_ms = int(trade_time.timestamp() * 1000)

            # Fetch BTC price
            btc_price = fetch_btc_price_at_time(timestamp_ms)

            if btc_price > 0:
                cur.execute(
                    "UPDATE paper_trades SET btc_price = ? WHERE id = ?",
                    (btc_price, row["id"])
                )

                # Calculate distance
                strike = row["strike_price"]
                dist = ((btc_price - strike) / btc_price) * 100 if btc_price > 0 else 0

                if (i + 1) % 20 == 0:
                    print(f"  [{i+1}/{len(rows)}] ID {row['id']}: BTC=${btc_price:.0f}, strike=${strike:.0f}, dist={dist:+.2f}%")

            # Rate limit
            if (i + 1) % 10 == 0:
                time.sleep(0.5)

        except Exception as e:
            print(f"  Error processing trade {row['id']}: {e}")

    conn.commit()
    conn.close()
    print(f"Updated {len(rows)} paper trades with BTC prices")


def backfill_prediction_samples_with_context():
    """Update prediction_samples features_json with market context."""
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()

    # Get all samples
    rows = cur.execute("""
        SELECT id, timestamp, expiry_time, strike_price, btc_price, features_json
        FROM prediction_samples
        WHERE features_json IS NOT NULL
    """).fetchall()

    print(f"\nUpdating {len(rows)} prediction samples with market context")

    updated = 0
    for row in rows:
        try:
            features = json.loads(row["features_json"]) if row["features_json"] else {}

            # Calculate time to expiry
            ts = row["timestamp"]
            exp = row["expiry_time"]
            if ts and exp:
                trade_time = datetime.fromisoformat(ts.replace("Z", "+00:00"))
                expiry = datetime.fromisoformat(exp.replace("Z", "+00:00"))
                tte_sec = (expiry - trade_time).total_seconds()
                time_to_expiry_min = max(0, tte_sec / 60.0)
            else:
                time_to_expiry_min = 7.5

            # Get BTC price (fetch if needed)
            btc_price = row["btc_price"]
            if not btc_price or btc_price == 0:
                trade_time = datetime.fromisoformat(ts.replace("Z", "+00:00"))
                timestamp_ms = int(trade_time.timestamp() * 1000)
                btc_price = fetch_btc_price_at_time(timestamp_ms)

                # Update btc_price in table
                if btc_price > 0:
                    cur.execute(
                        "UPDATE prediction_samples SET btc_price = ? WHERE id = ?",
                        (btc_price, row["id"])
                    )

            # Calculate distance to strike
            strike_price = row["strike_price"]
            if btc_price and strike_price and btc_price > 0:
                distance_to_strike_pct = ((btc_price - strike_price) / btc_price) * 100
                above_strike = 1.0 if btc_price > strike_price else 0.0
            else:
                distance_to_strike_pct = 0.0
                above_strike = 0.5

            # Time-distance interaction
            time_distance_interaction = (time_to_expiry_min / 15.0) * (1.0 / (1.0 + abs(distance_to_strike_pct)))

            # Update features
            features.update({
                "time_to_expiry_min": time_to_expiry_min,
                "distance_to_strike_pct": distance_to_strike_pct,
                "above_strike": above_strike,
                "time_distance_interaction": time_distance_interaction,
            })

            cur.execute(
                "UPDATE prediction_samples SET features_json = ? WHERE id = ?",
                (json.dumps(features), row["id"])
            )
            updated += 1

        except Exception as e:
            print(f"  Error processing sample {row['id']}: {e}")

    conn.commit()
    conn.close()
    print(f"Updated {updated} prediction samples")


def analyze_backfilled_data():
    """Show analysis of backfilled data."""
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()

    print("\n" + "=" * 60)
    print("BACKFILLED DATA ANALYSIS")
    print("=" * 60)

    # Paper trades with context
    rows = cur.execute("""
        SELECT
            CASE
                WHEN btc_price > strike_price THEN 'above'
                ELSE 'below'
            END as position,
            outcome,
            COUNT(*) as cnt,
            AVG((btc_price - strike_price) / btc_price * 100) as avg_dist
        FROM paper_trades
        WHERE btc_price IS NOT NULL AND outcome IS NOT NULL
        GROUP BY position, outcome
        ORDER BY position, outcome
    """).fetchall()

    print("\nPaper trades by position vs strike:")
    for r in rows:
        print(f"  {r['position']:5} strike, {r['outcome']:4}: {r['cnt']:3} trades, avg dist: {r['avg_dist']:+.2f}%")

    # Time to expiry analysis
    rows = cur.execute("""
        SELECT
            CASE
                WHEN (julianday(expiry_time) - julianday(timestamp)) * 24 * 60 < 5 THEN '< 5 min'
                WHEN (julianday(expiry_time) - julianday(timestamp)) * 24 * 60 < 10 THEN '5-10 min'
                ELSE '10+ min'
            END as tte_bucket,
            outcome,
            COUNT(*) as cnt
        FROM paper_trades
        WHERE outcome IS NOT NULL
        GROUP BY tte_bucket, outcome
        ORDER BY tte_bucket, outcome
    """).fetchall()

    print("\nPaper trades by time to expiry:")
    for r in rows:
        print(f"  {r['tte_bucket']:10}, {r['outcome']:4}: {r['cnt']:3} trades")

    # Sample with all data
    print("\nSample backfilled trades:")
    rows = cur.execute("""
        SELECT id, timestamp, strike_price, btc_price, entry_price, side, outcome,
               (btc_price - strike_price) / btc_price * 100 as dist_pct,
               (julianday(expiry_time) - julianday(timestamp)) * 24 * 60 as tte_min
        FROM paper_trades
        WHERE btc_price IS NOT NULL
        ORDER BY id DESC
        LIMIT 10
    """).fetchall()

    for r in rows:
        print(f"  #{r['id']}: {r['side']} @ {r['entry_price']*100:.0f}¢ | BTC=${r['btc_price']:.0f} | strike=${r['strike_price']:.0f} | dist={r['dist_pct']:+.2f}% | tte={r['tte_min']:.0f}m | {r['outcome']}")

    conn.close()


if __name__ == "__main__":
    print("=" * 60)
    print("BACKFILLING WITH BINANCE HISTORICAL DATA")
    print("=" * 60)

    backfill_paper_trades()
    backfill_prediction_samples_with_context()
    analyze_backfilled_data()

    print("\n" + "=" * 60)
    print("Done! Now retrain the model:")
    print("  python btc15m/scripts/train_model.py")
    print("=" * 60)
