#!/usr/bin/env python3
"""Backfill market context features (time_to_expiry, distance_to_strike) for historical data.

This adds critical features that help the model understand:
- How much time was left when the trade was placed
- How far the price was from the strike

These features are essential for the model to learn that the same technical
setup has very different probabilities at 3 min vs 59 min remaining.
"""

import sqlite3
import json
from datetime import datetime
from pathlib import Path

DB_PATH = Path(__file__).parent.parent / "data" / "btc15m.db"


def backfill_prediction_samples():
    """Add market context features to prediction_samples."""
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()

    # Get all samples with features_json
    rows = cur.execute("""
        SELECT id, timestamp, expiry_time, strike_price, btc_price, features_json
        FROM prediction_samples
        WHERE features_json IS NOT NULL
    """).fetchall()

    print(f"Found {len(rows)} prediction samples to backfill")

    updated = 0
    for row in rows:
        try:
            features = json.loads(row["features_json"]) if row["features_json"] else {}

            # Skip if already has market context
            if "time_to_expiry_min" in features:
                continue

            # Calculate time to expiry
            ts = row["timestamp"]
            exp = row["expiry_time"]
            if ts and exp:
                trade_time = datetime.fromisoformat(ts.replace("Z", "+00:00"))
                expiry = datetime.fromisoformat(exp.replace("Z", "+00:00"))
                tte_sec = (expiry - trade_time).total_seconds()
                time_to_expiry_min = tte_sec / 60.0
            else:
                time_to_expiry_min = 7.5  # Default

            # Calculate distance to strike
            btc_price = row["btc_price"]
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

            # Save back
            cur.execute(
                "UPDATE prediction_samples SET features_json = ? WHERE id = ?",
                (json.dumps(features), row["id"])
            )
            updated += 1

        except Exception as e:
            print(f"Error processing sample {row['id']}: {e}")

    conn.commit()
    conn.close()
    print(f"Updated {updated} prediction samples with market context features")


def add_market_context_columns():
    """Add market context columns to historical_samples table if needed."""
    conn = sqlite3.connect(str(DB_PATH))
    cur = conn.cursor()

    # Check existing columns
    cur.execute("PRAGMA table_info(historical_samples)")
    existing_cols = {r[1] for r in cur.fetchall()}

    new_cols = [
        ("time_to_expiry_min", "REAL DEFAULT 7.5"),
        ("distance_to_strike_pct", "REAL DEFAULT 0"),
        ("above_strike", "REAL DEFAULT 0.5"),
        ("time_distance_interaction", "REAL DEFAULT 0.5"),
    ]

    for col_name, col_type in new_cols:
        if col_name not in existing_cols:
            print(f"Adding column {col_name} to historical_samples")
            cur.execute(f"ALTER TABLE historical_samples ADD COLUMN {col_name} {col_type}")

    conn.commit()
    conn.close()


def show_sample_data():
    """Show sample of backfilled data."""
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()

    rows = cur.execute("""
        SELECT id, timestamp, expiry_time, strike_price, btc_price, features_json
        FROM prediction_samples
        WHERE features_json IS NOT NULL
        ORDER BY id DESC
        LIMIT 5
    """).fetchall()

    print("\nSample backfilled data:")
    for row in rows:
        features = json.loads(row["features_json"]) if row["features_json"] else {}
        tte = features.get("time_to_expiry_min", "N/A")
        dist = features.get("distance_to_strike_pct", "N/A")
        above = features.get("above_strike", "N/A")

        print(f"  ID {row['id']}: TTE={tte:.1f}min, Dist={dist:+.2f}%, Above={above}")

    conn.close()


if __name__ == "__main__":
    print("=" * 60)
    print("BACKFILLING MARKET CONTEXT FEATURES")
    print("=" * 60)

    add_market_context_columns()
    backfill_prediction_samples()
    show_sample_data()

    print("\nDone! Now retrain the model with: python scripts/train_model.py")
