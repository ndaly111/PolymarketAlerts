#!/usr/bin/env python3
"""Train ML model on historical BTC 15-min data.

Uses the backfilled historical data to train a classifier
that predicts UP/DOWN outcomes.

Usage:
    python train_model.py [--db path/to/db]
"""

from __future__ import annotations

import argparse
import pickle
import sqlite3
import sys
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from lib.db import DEFAULT_DB_PATH

try:
    import numpy as np
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.model_selection import train_test_split, cross_val_score
    from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
    from sklearn.preprocessing import StandardScaler
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False


FEATURE_COLUMNS = [
    "rsi",
    "macd",
    "macd_signal",
    "macd_histogram",
    "volatility",
    "momentum",
    "price_vs_vwap",
]


def load_training_data(db_path: Path) -> tuple:
    """Load training data from historical samples table."""
    with sqlite3.connect(str(db_path)) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute("""
            SELECT rsi, macd, macd_signal, macd_histogram,
                   volatility, momentum, price_vs_vwap, outcome
            FROM historical_samples
            WHERE rsi IS NOT NULL
              AND macd IS NOT NULL
            ORDER BY window_start
        """).fetchall()

    if not rows:
        return None, None

    X = []
    y = []

    for row in rows:
        features = [
            row["rsi"] or 50,
            row["macd"] or 0,
            row["macd_signal"] or 0,
            row["macd_histogram"] or 0,
            row["volatility"] or 0,
            row["momentum"] or 0,
            row["price_vs_vwap"] or 0,
        ]
        X.append(features)
        y.append(1 if row["outcome"] == "UP" else 0)

    return np.array(X), np.array(y)


def train_model(X: np.ndarray, y: np.ndarray) -> tuple:
    """Train and evaluate the model."""
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, shuffle=False  # Don't shuffle time series
    )

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train Random Forest
    print("[train] Training Random Forest...")
    rf_model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        min_samples_split=10,
        random_state=42,
        n_jobs=-1,
    )
    rf_model.fit(X_train_scaled, y_train)

    # Train Gradient Boosting
    print("[train] Training Gradient Boosting...")
    gb_model = GradientBoostingClassifier(
        n_estimators=100,
        max_depth=5,
        learning_rate=0.1,
        random_state=42,
    )
    gb_model.fit(X_train_scaled, y_train)

    # Evaluate both
    print("\n" + "=" * 50)
    print("RANDOM FOREST RESULTS")
    print("=" * 50)
    rf_pred = rf_model.predict(X_test_scaled)
    rf_acc = accuracy_score(y_test, rf_pred)
    print(f"Accuracy: {rf_acc:.3f}")
    print("\nClassification Report:")
    print(classification_report(y_test, rf_pred, target_names=["DOWN", "UP"]))

    print("\n" + "=" * 50)
    print("GRADIENT BOOSTING RESULTS")
    print("=" * 50)
    gb_pred = gb_model.predict(X_test_scaled)
    gb_acc = accuracy_score(y_test, gb_pred)
    print(f"Accuracy: {gb_acc:.3f}")
    print("\nClassification Report:")
    print(classification_report(y_test, gb_pred, target_names=["DOWN", "UP"]))

    # Feature importance
    print("\n" + "=" * 50)
    print("FEATURE IMPORTANCE (Random Forest)")
    print("=" * 50)
    for name, importance in sorted(
        zip(FEATURE_COLUMNS, rf_model.feature_importances_),
        key=lambda x: x[1],
        reverse=True,
    ):
        print(f"  {name:20s}: {importance:.3f}")

    # Cross-validation
    print("\n" + "=" * 50)
    print("CROSS-VALIDATION (5-fold)")
    print("=" * 50)
    cv_scores = cross_val_score(rf_model, X_train_scaled, y_train, cv=5)
    print(f"RF CV Scores: {cv_scores}")
    print(f"RF CV Mean: {cv_scores.mean():.3f} (+/- {cv_scores.std() * 2:.3f})")

    # Choose best model
    best_model = rf_model if rf_acc >= gb_acc else gb_model
    best_name = "RandomForest" if rf_acc >= gb_acc else "GradientBoosting"
    best_acc = max(rf_acc, gb_acc)

    return best_model, scaler, best_name, best_acc


def save_model(model, scaler, model_path: Path) -> None:
    """Save trained model and scaler."""
    model_path.parent.mkdir(parents=True, exist_ok=True)

    with open(model_path, "wb") as f:
        pickle.dump({"model": model, "scaler": scaler}, f)

    print(f"\n[train] Model saved to {model_path}")


def main():
    if not HAS_SKLEARN:
        print("ERROR: scikit-learn is required")
        print("Run: pip install scikit-learn numpy")
        sys.exit(1)

    parser = argparse.ArgumentParser(description="Train BTC 15-min prediction model")
    parser.add_argument("--db", type=str, default=str(DEFAULT_DB_PATH),
                        help="Database path")
    parser.add_argument("--output", type=str,
                        default=str(Path(__file__).parent.parent / "models" / "btc15m_model.pkl"),
                        help="Output model path")
    args = parser.parse_args()

    db_path = Path(args.db)
    model_path = Path(args.output)

    print("[train] Loading training data...")
    X, y = load_training_data(db_path)

    if X is None or len(X) == 0:
        print("ERROR: No training data found")
        print("Run: python backfill_historical.py --days 30")
        sys.exit(1)

    print(f"[train] Loaded {len(X):,} samples")
    print(f"[train] Features: {FEATURE_COLUMNS}")
    print(f"[train] UP: {sum(y)} ({sum(y)/len(y)*100:.1f}%)")
    print(f"[train] DOWN: {len(y) - sum(y)} ({(len(y)-sum(y))/len(y)*100:.1f}%)")
    print()

    # Train
    model, scaler, model_name, accuracy = train_model(X, y)

    # Save
    save_model(model, scaler, model_path)

    print("\n" + "=" * 50)
    print("SUMMARY")
    print("=" * 50)
    print(f"Best Model: {model_name}")
    print(f"Test Accuracy: {accuracy:.1%}")
    print(f"Model saved: {model_path}")

    # Note about edge
    print("\n[train] Note: 50% accuracy = random chance")
    print("[train] You need >52% accuracy to overcome trading fees")
    print("[train] >55% accuracy = potentially profitable")


if __name__ == "__main__":
    main()
