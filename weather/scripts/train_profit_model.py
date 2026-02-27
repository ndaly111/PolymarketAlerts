#!/usr/bin/env python3
"""
Train a profit-optimized ML model for weather trading.

Unlike the quantile regression model that predicts forecast error distribution,
this model predicts whether a trade will be PROFITABLE (pnl > 0).

Features:
- Model probability (from quantile model)
- Market price (yes_ask/no_ask)
- Edge (model_prob - market_price)
- EV estimate
- Event type (above/below/between)
- Temperature volatility indicators
- Time features (month, day_of_week)
- City-specific features

Usage:
    python train_profit_model.py --all        # Train on all cities
    python train_profit_model.py --city NHIGH # Train on NYC only
    python train_profit_model.py --backtest   # Run backtest to generate training data
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import sqlite3
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

try:
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from weather.lib import db as db_lib
from weather.lib.fees import FeeSchedule
from weather.lib.kalshi_weather import EventSpec

DB_PATH = Path(os.getenv("WEATHER_DB_PATH", str(ROOT / "weather" / "data" / "weather_forecast_accuracy.db")))
MODEL_DIR = ROOT / "weather" / "models"
FEATURE_IMPORTANCE_FILE = MODEL_DIR / "profit_feature_importance.json"

# Feature names for the profit model
PROFIT_FEATURES = [
    "model_prob",
    "market_price",
    "edge",
    "ev_estimate",
    "is_yes_side",
    "event_type_above",
    "event_type_below",
    "event_type_between",
    "threshold_distance",  # Distance from forecast to threshold
    "nbm_spread",          # NBM model uncertainty
    "month",
    "day_of_week",
    "hour_of_day",
    "forecast_high",
    "lead_hours",
    "precip_prob",
    "cloud_cover",
    "wind_speed",
    "city_idx",
]


def ensure_trades_table(db_path: Path) -> None:
    """Create the weather trades table if it doesn't exist."""
    conn = sqlite3.connect(db_path)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS weather_paper_trades (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            city_key TEXT NOT NULL,
            target_date TEXT NOT NULL,
            market_ticker TEXT NOT NULL,
            event_type TEXT NOT NULL,
            threshold_a INTEGER NOT NULL,
            threshold_b INTEGER,
            side TEXT NOT NULL,
            entry_price REAL NOT NULL,
            contracts INTEGER NOT NULL DEFAULT 1,
            model_prob REAL NOT NULL,
            edge REAL NOT NULL,
            ev_estimate REAL NOT NULL,
            profit_model_prob REAL,
            features_json TEXT,
            outcome TEXT,
            pnl REAL,
            settled_at TEXT,
            is_paper INTEGER NOT NULL DEFAULT 1
        )
    """)
    conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_weather_trades_city_date
        ON weather_paper_trades(city_key, target_date)
    """)
    conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_weather_trades_outcome
        ON weather_paper_trades(outcome)
    """)
    conn.commit()
    conn.close()


def get_dates_with_data(
    db_path: Path,
    city_key: str,
    start_date: str,
    end_date: str,
) -> List[str]:
    """Get dates that have both forecast and observation data."""
    conn = sqlite3.connect(db_path)
    rows = conn.execute("""
        SELECT DISTINCT f.target_date_local
        FROM forecast_snapshots f
        JOIN observed_cli o ON f.city_key = o.city_key AND f.target_date_local = o.date_local
        WHERE f.city_key = ?
          AND f.target_date_local >= ?
          AND f.target_date_local <= ?
        ORDER BY f.target_date_local
    """, (city_key, start_date, end_date)).fetchall()
    conn.close()
    return [r[0] for r in rows]


def get_observed_high(db_path: Path, city_key: str, date: str) -> Optional[int]:
    """Get observed high temperature for a date."""
    conn = sqlite3.connect(db_path)
    row = conn.execute("""
        SELECT tmax_f FROM observed_cli
        WHERE city_key = ? AND date_local = ?
    """, (city_key, date)).fetchone()
    conn.close()
    return row[0] if row else None


def generate_training_data_from_backtest(
    db_path: Path,
    start_date: str,
    end_date: str,
    fee_cents: int = 2,
    min_ev: float = 0.02,
) -> Tuple[np.ndarray, np.ndarray, List[Dict]]:
    """
    Generate training data by simulating trades on historical data.

    Returns:
        (X, y, metadata) where:
        - X: feature array
        - y: binary labels (1=profitable, 0=unprofitable)
        - metadata: list of dicts with trade details
    """
    from weather.lib.fair import normalize_pmf, shift_pmf
    from weather.lib.kalshi_weather import parse_event_spec_from_title, parse_event_spec_from_ticker, prob_event

    import yaml
    config_path = ROOT / "weather" / "config" / "cities.yml"
    with open(config_path) as f:
        config = yaml.safe_load(f)
    cities = {c["key"]: c for c in config.get("cities", [])}
    city_idx_map = {k: i for i, k in enumerate(sorted(cities.keys()))}

    fees = FeeSchedule(open_fee_cents=fee_cents)

    X_list = []
    y_list = []
    metadata_list = []

    for city_key, city_cfg in cities.items():
        print(f"Processing {city_key}...")

        # Get dates with data
        dates = get_dates_with_data(db_path, city_key, start_date, end_date)

        for date in dates:
            month = int(date.split("-")[1])
            date_obj = datetime.strptime(date, "%Y-%m-%d")

            # Get forecast - try multiple sources
            forecast = None
            for source in ["open_meteo", "nws_hourly_max", "mos_gfs_18z_archive", "mos_gfs_12z"]:
                forecast = db_lib.fetch_latest_forecast_snapshot(
                    db_path,
                    city_key=city_key,
                    target_date_local=date,
                    source=source,
                )
                if forecast:
                    break
            if not forecast:
                continue

            forecast_high = int(forecast["forecast_high_f"])

            # Parse forecast context
            forecast_context = {}
            if forecast.get("forecast_context_json"):
                try:
                    forecast_context = json.loads(forecast["forecast_context_json"])
                except (json.JSONDecodeError, TypeError):
                    pass

            precip_prob = forecast_context.get("precip_prob_max", 0) / 100.0
            cloud_cover = forecast_context.get("cloud_cover_avg", 50) / 100.0
            wind_speed = forecast_context.get("wind_speed_avg", 10)

            # Get error model - try multiple sources and hours
            error_model = None
            for source in ["combined", "mos_gfs_18z_archive", "nws_hourly_max"]:
                for hour in [6, 12, 18]:
                    error_model = db_lib.fetch_error_model(
                        db_path,
                        city_key=city_key,
                        month=month,
                        snapshot_hour_local=hour,
                        source=source,
                    )
                    if error_model:
                        break
                if error_model:
                    break
            if not error_model:
                continue

            # Build PMF
            pmf_error = error_model["pmf"]
            pmf_high = normalize_pmf(shift_pmf(pmf_error, forecast_high))

            # Get observed outcome
            observed = get_observed_high(db_path, city_key, date)
            if observed is None:
                continue

            # Get NBM spread if available
            nbm_row = db_lib.fetch_latest_nbm_forecast(
                db_path, city_key=city_key, target_date_local=date, metric="tmax"
            )

            nbm_spread = 8  # Default
            if nbm_row and nbm_row.get("p90") and nbm_row.get("p10"):
                nbm_spread = nbm_row["p90"] - nbm_row["p10"]

            # Get market data
            conn = sqlite3.connect(db_path)
            conn.row_factory = sqlite3.Row

            # Build event ticker pattern
            series_ticker = city_cfg.get("kalshi_series", f"KXHIGH{city_key[:3]}")
            event_pattern = f"{series_ticker}-{date_obj.strftime('%y%b%d').upper()}"

            markets = conn.execute("""
                SELECT DISTINCT market_ticker, yes_ask, no_ask, raw_json, snapshot_time_utc
                FROM kalshi_weather_market_snapshots
                WHERE event_ticker = ? AND status = 'active'
                ORDER BY snapshot_time_utc DESC
            """, (event_pattern,)).fetchall()

            conn.close()

            # Track seen tickers to avoid duplicates
            seen_tickers = set()

            for market in markets:
                ticker = market["market_ticker"]
                if ticker in seen_tickers:
                    continue
                seen_tickers.add(ticker)

                yes_ask = market["yes_ask"]
                no_ask = market["no_ask"]

                if yes_ask is None or no_ask is None:
                    continue

                # Parse market details
                try:
                    raw = json.loads(market["raw_json"])
                    title = raw.get("no_sub_title", raw.get("title", ""))
                except:
                    continue

                event_spec = parse_event_spec_from_title(title)
                if not event_spec:
                    ticker = row.get("market_ticker") or row.get("ticker", "")
                    event_spec = parse_event_spec_from_ticker(ticker)
                if not event_spec:
                    continue

                event_type = event_spec.kind
                threshold_a = event_spec.a
                threshold_b = event_spec.b

                # Compute model probability
                model_prob = prob_event(pmf_high, event_spec)

                # Check both sides
                for side in ["YES", "NO"]:
                    if side == "YES":
                        price = yes_ask / 100.0
                        prob = model_prob
                    else:
                        price = no_ask / 100.0
                        prob = 1.0 - model_prob

                    edge = prob - price - fees.open_fee_dollars()
                    ev = edge * (1 - price)  # Simplified EV

                    if ev < min_ev:
                        continue

                    # Determine actual outcome
                    if event_type == "ge":
                        event_happened = observed >= threshold_a
                    elif event_type == "le":
                        event_happened = observed <= threshold_a
                    elif event_type == "between":
                        event_happened = threshold_a <= observed <= threshold_b
                    else:
                        continue

                    # Did we win?
                    if side == "YES":
                        won = event_happened
                        pnl = (1 - price) * 0.98 if won else -price
                    else:
                        won = not event_happened
                        pnl = (1 - price) * 0.98 if won else -price

                    # Compute threshold distance
                    if event_type == "between":
                        mid = (threshold_a + threshold_b) / 2
                        threshold_distance = abs(forecast_high - mid)
                    else:
                        threshold_distance = abs(forecast_high - threshold_a)

                    # Parse snapshot time for hour
                    try:
                        snap_time = datetime.fromisoformat(market["snapshot_time_utc"].replace("Z", "+00:00"))
                        hour_of_day = snap_time.hour
                    except:
                        hour_of_day = 12

                    # Build feature vector
                    features = {
                        "model_prob": prob,
                        "market_price": price,
                        "edge": edge,
                        "ev_estimate": ev,
                        "is_yes_side": 1.0 if side == "YES" else 0.0,
                        "event_type_above": 1.0 if event_type == "ge" else 0.0,
                        "event_type_below": 1.0 if event_type == "le" else 0.0,
                        "event_type_between": 1.0 if event_type == "between" else 0.0,
                        "threshold_distance": threshold_distance,
                        "nbm_spread": nbm_spread,
                        "month": float(month),
                        "day_of_week": float(date_obj.weekday()),
                        "hour_of_day": float(hour_of_day),
                        "forecast_high": float(forecast_high),
                        "lead_hours": 18.0,  # Assume morning forecast
                        "precip_prob": precip_prob,
                        "cloud_cover": cloud_cover,
                        "wind_speed": wind_speed,
                        "city_idx": float(city_idx_map.get(city_key, 0)),
                    }

                    X_row = [features[name] for name in PROFIT_FEATURES]
                    X_list.append(X_row)
                    y_list.append(1 if pnl > 0 else 0)

                    metadata_list.append({
                        "city_key": city_key,
                        "date": date,
                        "ticker": ticker,
                        "side": side,
                        "price": price,
                        "model_prob": prob,
                        "edge": edge,
                        "ev": ev,
                        "pnl": pnl,
                        "won": won,
                        "observed": observed,
                        "forecast": forecast_high,
                    })

    return np.array(X_list), np.array(y_list), metadata_list


def train_profit_model(
    X: np.ndarray,
    y: np.ndarray,
    save_path: Optional[Path] = None,
) -> Tuple[Any, Dict[str, float]]:
    """
    Train a gradient boosting classifier to predict profitable trades.

    Returns:
        (model, metrics) tuple
    """
    if not HAS_SKLEARN:
        raise ImportError("sklearn required: pip install scikit-learn")

    # Split data - use stratify only if we have enough samples of each class
    min_class_count = min(np.sum(y == 0), np.sum(y == 1))
    use_stratify = min_class_count >= 5

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42,
        stratify=y if use_stratify else None
    )

    print(f"Training samples: {len(X_train)}")
    print(f"Test samples: {len(X_test)}")
    print(f"Positive rate: {y.mean()*100:.1f}%")

    # Train model
    model = GradientBoostingClassifier(
        n_estimators=200,
        max_depth=5,
        learning_rate=0.1,
        min_samples_leaf=10,
        subsample=0.8,
        random_state=42,
    )

    model.fit(X_train, y_train)

    # Evaluate
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1": f1_score(y_test, y_pred, zero_division=0),
        "baseline_accuracy": max(y.mean(), 1 - y.mean()),
    }

    print(f"\nTest Metrics:")
    print(f"  Accuracy:  {metrics['accuracy']*100:.1f}% (baseline: {metrics['baseline_accuracy']*100:.1f}%)")
    print(f"  Precision: {metrics['precision']*100:.1f}%")
    print(f"  Recall:    {metrics['recall']*100:.1f}%")
    print(f"  F1 Score:  {metrics['f1']*100:.1f}%")

    # Feature importance
    importance = list(zip(PROFIT_FEATURES, model.feature_importances_))
    importance.sort(key=lambda x: x[1], reverse=True)

    print(f"\nTop Features:")
    for name, imp in importance[:10]:
        print(f"  {name}: {imp:.4f}")

    # Save feature importance
    if save_path:
        importance_data = {
            "model": "GradientBoostingClassifier",
            "features": [{"name": n, "importance": float(i)} for n, i in importance],
            "accuracy": metrics["accuracy"],
            "timestamp": datetime.utcnow().isoformat(),
        }

        with open(FEATURE_IMPORTANCE_FILE, "w") as f:
            json.dump(importance_data, f, indent=2)

    # Save model
    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        with open(save_path, "wb") as f:
            pickle.dump({
                "model": model,
                "feature_names": PROFIT_FEATURES,
                "metrics": metrics,
                "trained_at": datetime.utcnow().isoformat(),
            }, f)
        print(f"\nModel saved to {save_path}")

    return model, metrics


def simulate_profit_model_trading(
    model: Any,
    X: np.ndarray,
    y: np.ndarray,
    metadata: List[Dict],
    threshold: float = 0.6,
) -> Dict[str, Any]:
    """
    Simulate trading using the profit model filter.

    Returns:
        Dict with simulation results
    """
    probs = model.predict_proba(X)[:, 1]

    # Compare: baseline (all positive EV trades) vs model-filtered
    baseline_pnl = sum(m["pnl"] for m in metadata)
    baseline_trades = len(metadata)
    baseline_wins = sum(1 for m in metadata if m["won"])

    filtered_indices = [i for i, p in enumerate(probs) if p >= threshold]
    filtered_pnl = sum(metadata[i]["pnl"] for i in filtered_indices)
    filtered_trades = len(filtered_indices)
    filtered_wins = sum(1 for i in filtered_indices if metadata[i]["won"])

    results = {
        "baseline": {
            "trades": baseline_trades,
            "wins": baseline_wins,
            "win_rate": baseline_wins / baseline_trades * 100 if baseline_trades > 0 else 0,
            "pnl": baseline_pnl,
            "avg_pnl": baseline_pnl / baseline_trades if baseline_trades > 0 else 0,
        },
        "filtered": {
            "threshold": threshold,
            "trades": filtered_trades,
            "wins": filtered_wins,
            "win_rate": filtered_wins / filtered_trades * 100 if filtered_trades > 0 else 0,
            "pnl": filtered_pnl,
            "avg_pnl": filtered_pnl / filtered_trades if filtered_trades > 0 else 0,
        },
    }

    print(f"\n{'='*60}")
    print("TRADING SIMULATION")
    print(f"{'='*60}")
    print(f"\nBaseline (all +EV trades):")
    print(f"  Trades: {results['baseline']['trades']}")
    print(f"  Win Rate: {results['baseline']['win_rate']:.1f}%")
    print(f"  P&L: ${results['baseline']['pnl']:.2f}")
    print(f"  Avg P&L: ${results['baseline']['avg_pnl']:.4f}/trade")

    print(f"\nFiltered (profit model > {threshold*100:.0f}%):")
    print(f"  Trades: {results['filtered']['trades']}")
    print(f"  Win Rate: {results['filtered']['win_rate']:.1f}%")
    print(f"  P&L: ${results['filtered']['pnl']:.2f}")
    print(f"  Avg P&L: ${results['filtered']['avg_pnl']:.4f}/trade")

    improvement = results['filtered']['pnl'] - results['baseline']['pnl']
    print(f"\nImprovement: ${improvement:+.2f}")

    return results


def main():
    parser = argparse.ArgumentParser(description="Train weather profit prediction model")
    parser.add_argument("--city", help="Single city to train on (e.g., NHIGH)")
    parser.add_argument("--all", action="store_true", help="Train on all cities")
    parser.add_argument("--start", default="2024-01-01", help="Start date for training data")
    parser.add_argument("--end", default="2025-12-31", help="End date for training data")
    parser.add_argument("--min-ev", type=float, default=0.02, help="Minimum EV for trade candidates")
    parser.add_argument("--threshold", type=float, default=0.6, help="Model confidence threshold")
    parser.add_argument("--backtest-only", action="store_true", help="Only generate backtest data, don't train")
    args = parser.parse_args()

    if not HAS_SKLEARN:
        print("ERROR: scikit-learn required. Install with: pip install scikit-learn")
        return 1

    print("="*60)
    print("WEATHER PROFIT MODEL TRAINING")
    print("="*60)
    print(f"Database: {DB_PATH}")
    print(f"Date range: {args.start} to {args.end}")
    print(f"Min EV: {args.min_ev}")

    # Ensure trades table exists
    ensure_trades_table(DB_PATH)

    # Generate training data
    print(f"\nGenerating training data from backtest...")
    X, y, metadata = generate_training_data_from_backtest(
        DB_PATH,
        args.start,
        args.end,
        min_ev=args.min_ev,
    )

    if len(X) == 0:
        print("ERROR: No training data generated")
        return 1

    print(f"\nTotal samples: {len(X)}")
    print(f"Profitable: {y.sum()} ({y.mean()*100:.1f}%)")

    if args.backtest_only:
        print("\nBacktest data generation complete (--backtest-only)")
        return 0

    # Check minimum samples
    min_samples = 50
    if len(X) < min_samples:
        print(f"\n⚠️  Insufficient samples ({len(X)} < {min_samples}) for reliable training")
        print("   Continue collecting paper trade data to build training set")
        print("   Run with --backtest-only to save data without training")
        return 1

    # Train model
    model_path = MODEL_DIR / "weather_profit_model.pkl"
    model, metrics = train_profit_model(X, y, save_path=model_path)

    # Simulate trading
    simulate_profit_model_trading(model, X, y, metadata, threshold=args.threshold)

    print(f"\n{'='*60}")
    print("Training complete!")
    print(f"Model saved to: {model_path}")
    print(f"Feature importance: {FEATURE_IMPORTANCE_FILE}")
    print(f"{'='*60}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
