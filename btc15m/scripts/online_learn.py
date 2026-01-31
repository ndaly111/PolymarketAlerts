#!/usr/bin/env python3
"""Online learning - continuously retrain model with new data.

Saves live predictions with features, then periodically retrains
the model incorporating new data.

Usage:
    python online_learn.py [--interval 300] [--min-samples 10]
"""

from __future__ import annotations

import argparse
import pickle
import sqlite3
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from lib.db import DEFAULT_DB_PATH, get_paper_trade_summary
from lib.discord import send_ml_training_alert
from scripts.train_model import FEATURE_COLUMNS, train_model, save_model

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False


def ensure_live_samples_table(db_path: Path) -> None:
    """Create live_samples table if it doesn't exist."""
    cols = ", ".join([f"{col} REAL" for col in FEATURE_COLUMNS])

    with sqlite3.connect(str(db_path)) as conn:
        conn.execute(f"""
            CREATE TABLE IF NOT EXISTS live_samples (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                {cols},
                outcome TEXT NOT NULL,
                prediction TEXT,
                model_prob REAL,
                was_correct INTEGER,
                pnl REAL DEFAULT 0,
                entry_price REAL DEFAULT 0,
                edge REAL DEFAULT 0,
                UNIQUE(timestamp)
            )
        """)
        # Add pnl column if table exists but column doesn't
        try:
            conn.execute("ALTER TABLE live_samples ADD COLUMN pnl REAL DEFAULT 0")
        except:
            pass
        try:
            conn.execute("ALTER TABLE live_samples ADD COLUMN entry_price REAL DEFAULT 0")
        except:
            pass
        try:
            conn.execute("ALTER TABLE live_samples ADD COLUMN edge REAL DEFAULT 0")
        except:
            pass
        conn.commit()


def save_live_sample(
    db_path: Path,
    features: dict,
    outcome: str,
    prediction: str = None,
    model_prob: float = None,
    pnl: float = 0,
    entry_price: float = 0,
    edge: float = 0,
) -> int:
    """Save a live sample with features, outcome, and P&L for profit-weighted learning.

    Args:
        db_path: Database path
        features: Dict of feature name -> value
        outcome: 'UP' or 'DOWN' (actual market outcome)
        prediction: Model's prediction
        model_prob: Model's probability
        pnl: Profit/loss from this trade (for weighting)
        entry_price: Entry price paid
        edge: Edge at time of trade

    Returns:
        Row ID of inserted sample
    """
    ensure_live_samples_table(db_path)

    timestamp = datetime.now(timezone.utc).isoformat()
    was_correct = 1 if prediction == outcome else 0 if prediction else None

    cols = ["timestamp"] + FEATURE_COLUMNS + ["outcome", "prediction", "model_prob", "was_correct", "pnl", "entry_price", "edge"]
    vals = [timestamp] + [features.get(col, 0) for col in FEATURE_COLUMNS] + [outcome, prediction, model_prob, was_correct, pnl, entry_price, edge]

    placeholders = ", ".join(["?"] * len(cols))
    col_names = ", ".join(cols)

    with sqlite3.connect(str(db_path)) as conn:
        cursor = conn.execute(f"""
            INSERT OR REPLACE INTO live_samples ({col_names})
            VALUES ({placeholders})
        """, vals)
        conn.commit()
        return cursor.lastrowid


def load_all_training_data(db_path: Path, profit_weighted: bool = True) -> tuple:
    """Load both historical and live samples for training.

    Args:
        db_path: Database path
        profit_weighted: If True, return sample weights based on P&L

    Returns:
        (X, y, sample_weights) if profit_weighted else (X, y, None)
    """

    defaults = {
        "rsi": 50, "macd": 0, "macd_signal": 0, "macd_histogram": 0,
        "volatility": 0, "momentum": 0, "price_vs_vwap": 0,
        "bb_percent_b": 0.5, "bb_bandwidth": 0, "atr": 0, "stoch_k": 50,
        "williams_r": -50, "cci": 0, "obv_trend": 0,
        "trend_15m": 0, "trend_1h": 0, "change_15m": 0, "change_1h": 0,
        "hour": 12, "hour_sin": 0, "hour_cos": 1,
        "asia_session": 0, "europe_session": 0, "us_session": 0, "high_vol_hour": 0,
        "doji": 0, "hammer": 0, "shooting_star": 0,
        "bullish_engulfing": 0, "bearish_engulfing": 0,
        "adx": 25, "plus_di": 25, "minus_di": 25, "mfi": 50,
        "roc": 0, "trix": 0, "kc_position": 0.5, "kc_width": 0,
        "dc_position": 0.5, "dc_width": 0, "ichi_tk_cross": 0,
        "ichi_above_cloud": 0, "ichi_below_cloud": 0, "ichi_cloud_thickness": 0,
        "price_vs_vwma": 0, "cmf": 0, "chop": 50,
        "squeeze_on": 0, "squeeze_momentum": 0,
        "vol_ratio": 1, "vol_trend": 0, "vol_up_ratio": 0.5, "vol_spike": 0,
        "dist_to_high_pct": 0, "dist_to_low_pct": 0, "range_compression": 1,
        "higher_highs": 0, "lower_lows": 0, "up_streak": 0, "down_streak": 0,
        "above_pivot": 0, "dist_to_nearest_pivot": 0,
        "lr_slope": 0, "lr_deviation": 0, "lr_r_squared": 0,
        "comp_trend_composite": 0, "comp_weighted_trend": 0,
        "comp_ob_os_signal": 0, "comp_overbought_consensus": 0, "comp_oversold_consensus": 0,
        "comp_volatility_composite": 0, "comp_is_ranging": 0, "comp_is_trending": 0,
        "comp_breakout_potential": 0, "comp_vol_price_confirmation": 0,
        "comp_vol_price_divergence": 0, "comp_strong_move": 0,
        "comp_mtf_aligned": 0, "comp_mtf_direction": 0, "comp_mtf_acceleration": 0,
        "comp_breakout_score": 0, "comp_breakout_direction": 0,
        "comp_has_divergence": 0, "comp_divergence_strength": 0,
        "funding_rate": 0, "funding_annualized": 0, "funding_extreme": 0,
        "funding_bullish": 0, "funding_bearish": 0, "long_short_ratio": 1.0,
        "ls_ratio_extreme": 0, "eth_change_15m": 0, "eth_change_1h": 0,
        "eth_momentum": 0, "eth_btc_divergence": 0, "eth_leading": 0,
        "eth_lagging": 0, "btc_eth_ratio": 25, "cross_asset_bullish": 0,
        "cross_asset_bearish": 0,
        "fg_value": 50, "fg_normalized": 0, "fg_extreme_fear": 0,
        "fg_fear": 0, "fg_neutral": 1, "fg_greed": 0, "fg_extreme_greed": 0,
        "fg_contrarian_buy": 0, "fg_contrarian_sell": 0,
        "deribit_iv": 50, "deribit_iv_high": 0, "deribit_iv_low": 0,
        "deribit_pc_ratio": 1.0, "deribit_pc_bullish": 0, "deribit_pc_bearish": 0,
        "deribit_call_oi": 0, "deribit_put_oi": 0, "deribit_oi_imbalance": 0,
    }

    X = []
    y = []
    weights = []

    with sqlite3.connect(str(db_path)) as conn:
        conn.row_factory = sqlite3.Row

        # Load historical samples (no P&L, default weight)
        cols = ", ".join(FEATURE_COLUMNS)
        try:
            hist_rows = conn.execute(f"""
                SELECT {cols}, outcome
                FROM historical_samples
                WHERE rsi IS NOT NULL AND macd IS NOT NULL
                ORDER BY window_start
            """).fetchall()

            for row in hist_rows:
                features = []
                for col in FEATURE_COLUMNS:
                    val = row[col] if col in row.keys() else None
                    if val is None:
                        val = defaults.get(col, 0)
                    features.append(val)
                X.append(features)
                y.append(1 if row["outcome"] == "UP" else 0)
                weights.append(1.0)  # Default weight for historical

            print(f"[online] Loaded {len(hist_rows)} historical samples")
        except Exception as e:
            print(f"[online] Error loading historical: {e}")

        # Load live samples with P&L for profit weighting
        try:
            live_rows = conn.execute(f"""
                SELECT {cols}, outcome, pnl
                FROM live_samples
                ORDER BY timestamp
            """).fetchall()

            live_pnls = []
            for row in live_rows:
                features = []
                for col in FEATURE_COLUMNS:
                    val = row[col] if col in row.keys() else None
                    if val is None:
                        val = defaults.get(col, 0)
                    features.append(val)
                X.append(features)
                y.append(1 if row["outcome"] == "UP" else 0)

                # Get P&L for weighting
                pnl = row["pnl"] if "pnl" in row.keys() and row["pnl"] else 0
                live_pnls.append(pnl)

            # Compute profit-based weights for live samples
            # Weight = 1 + |pnl| * scale_factor, capped at 5x
            # This makes high-profit trades 5x more influential
            if profit_weighted and live_pnls:
                max_abs_pnl = max(abs(p) for p in live_pnls) if live_pnls else 1
                scale = 4.0 / max_abs_pnl if max_abs_pnl > 0 else 1  # Scale so max weight ~ 5

                for pnl in live_pnls:
                    # Higher |P&L| = more weight (both wins and losses matter)
                    weight = 1.0 + abs(pnl) * scale
                    weight = min(weight, 5.0)  # Cap at 5x
                    weights.append(weight)

                print(f"[online] Loaded {len(live_rows)} live samples (profit-weighted)")
                print(f"[online] Live P&L range: ${min(live_pnls):.2f} to ${max(live_pnls):.2f}")
            else:
                # No profit weighting, use default weights
                weights.extend([1.0] * len(live_rows))
                print(f"[online] Loaded {len(live_rows)} live samples")

        except Exception as e:
            print(f"[online] No live samples table yet: {e}")

    if not X:
        return None, None, None

    return np.array(X), np.array(y), np.array(weights) if profit_weighted else None


def get_live_sample_count(db_path: Path) -> int:
    """Get number of live samples."""
    try:
        with sqlite3.connect(str(db_path)) as conn:
            result = conn.execute("SELECT COUNT(*) FROM live_samples").fetchone()
            return result[0] if result else 0
    except:
        return 0


def get_model_stats(db_path: Path) -> dict:
    """Get recent model accuracy stats from live samples."""
    try:
        with sqlite3.connect(str(db_path)) as conn:
            result = conn.execute("""
                SELECT
                    COUNT(*) as total,
                    SUM(was_correct) as correct,
                    AVG(was_correct) as accuracy
                FROM live_samples
                WHERE was_correct IS NOT NULL
            """).fetchone()

            return {
                "total": result[0] or 0,
                "correct": result[1] or 0,
                "accuracy": result[2] or 0,
            }
    except:
        return {"total": 0, "correct": 0, "accuracy": 0}


def retrain_and_save(db_path: Path, model_path: Path, profit_weighted: bool = True) -> bool:
    """Retrain model with all available data and save.

    Args:
        db_path: Database path
        model_path: Where to save the model
        profit_weighted: Use P&L-based sample weights for training
    """

    print(f"\n[online] {'='*50}")
    print(f"[online] RETRAINING MODEL (profit_weighted={profit_weighted})")
    print(f"[online] {'='*50}")

    X, y, sample_weights = load_all_training_data(db_path, profit_weighted=profit_weighted)

    if X is None or len(X) < 100:
        print(f"[online] Not enough data ({len(X) if X is not None else 0} samples)")
        return False

    print(f"[online] Total samples: {len(X)}")
    print(f"[online] UP: {sum(y)} ({sum(y)/len(y)*100:.1f}%)")
    print(f"[online] DOWN: {len(y) - sum(y)} ({(len(y)-sum(y))/len(y)*100:.1f}%)")
    if sample_weights is not None:
        print(f"[online] Weight range: {sample_weights.min():.2f} to {sample_weights.max():.2f}")

    # Count samples
    live_count = get_live_sample_count(db_path)
    hist_count = len(X) - live_count

    # Train model with profit weights
    model, scaler, model_name, accuracy = train_model(X, y, sample_weights)

    # Backup old model
    if model_path.exists():
        backup_path = model_path.with_suffix(f".pkl.backup.{int(time.time())}")
        model_path.rename(backup_path)
        print(f"[online] Backed up old model to {backup_path.name}")

    # Save new model
    save_model(model, scaler, model_path)

    print(f"\n[online] Retrained with {model_name}, accuracy: {accuracy:.1%}")

    # Get trading stats for Discord
    stats = get_model_stats(db_path)
    try:
        summary = get_paper_trade_summary(db_path)
        cumulative_pnl = summary.get('total_pnl', 0)
        total_record = f"{summary.get('wins', 0)}W / {summary.get('losses', 0)}L"
    except:
        cumulative_pnl = None
        total_record = None

    # Send Discord update
    weight_range = (sample_weights.min(), sample_weights.max()) if sample_weights is not None else None
    send_ml_training_alert(
        total_samples=len(X),
        live_samples=live_count,
        historical_samples=hist_count,
        model_name=model_name,
        accuracy=accuracy,
        profit_weighted=profit_weighted,
        weight_range=weight_range,
        cumulative_pnl=cumulative_pnl,
        total_record=total_record,
        live_accuracy=stats.get('accuracy'),
    )
    print(f"[online] Discord alert sent")

    return True


def main():
    if not HAS_NUMPY:
        print("ERROR: numpy is required")
        sys.exit(1)

    parser = argparse.ArgumentParser(description="Online learning for BTC model")
    parser.add_argument("--db", type=str, default=str(DEFAULT_DB_PATH))
    parser.add_argument("--model", type=str,
                        default=str(Path(__file__).parent.parent / "models" / "btc15m_model.pkl"))
    parser.add_argument("--interval", type=int, default=300,
                        help="Retrain interval in seconds (default: 300 = 5 min)")
    parser.add_argument("--min-samples", type=int, default=10,
                        help="Minimum new samples before retraining (default: 10)")
    parser.add_argument("--once", action="store_true",
                        help="Retrain once and exit")
    args = parser.parse_args()

    db_path = Path(args.db)
    model_path = Path(args.model)

    print("[online] BTC 15-min Online Learning")
    print(f"[online] DB: {db_path}")
    print(f"[online] Model: {model_path}")
    print(f"[online] Interval: {args.interval}s")
    print(f"[online] Min samples: {args.min_samples}")

    ensure_live_samples_table(db_path)

    if args.once:
        retrain_and_save(db_path, model_path)
        return

    last_sample_count = get_live_sample_count(db_path)
    last_retrain = time.time()

    print("\n[online] Starting continuous learning loop (Ctrl+C to stop)...")

    try:
        while True:
            time.sleep(60)  # Check every minute

            current_count = get_live_sample_count(db_path)
            new_samples = current_count - last_sample_count
            elapsed = time.time() - last_retrain

            stats = get_model_stats(db_path)
            print(f"[online] Live samples: {current_count} (+{new_samples} new) | "
                  f"Accuracy: {stats['accuracy']*100:.1f}% ({stats['correct']}/{stats['total']})")

            # Retrain if enough new samples and enough time passed
            if new_samples >= args.min_samples and elapsed >= args.interval:
                if retrain_and_save(db_path, model_path):
                    last_sample_count = current_count
                    last_retrain = time.time()
                    print("[online] Model updated - scanner will use new model on next prediction")

    except KeyboardInterrupt:
        print("\n[online] Stopped by user")


if __name__ == "__main__":
    main()
