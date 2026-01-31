#!/usr/bin/env python3
"""Evaluate model accuracy at different confidence thresholds.

Shows trade-off between accuracy and number of trades.

Usage:
    python evaluate_confidence.py [--db path/to/db]
"""

from __future__ import annotations

import argparse
import pickle
import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from lib.db import DEFAULT_DB_PATH

try:
    import numpy as np
    from sklearn.metrics import accuracy_score
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False


def main():
    if not HAS_SKLEARN:
        print("ERROR: scikit-learn required")
        sys.exit(1)

    parser = argparse.ArgumentParser(description="Evaluate model at different confidence levels")
    parser.add_argument("--db", type=str, default=str(DEFAULT_DB_PATH))
    parser.add_argument("--model", type=str,
                        default=str(Path(__file__).parent.parent / "models" / "btc15m_model.pkl"))
    args = parser.parse_args()

    # Load model
    print("[eval] Loading model...")
    with open(args.model, "rb") as f:
        data = pickle.load(f)
        model = data["model"]
        scaler = data["scaler"]

    # Load test data
    print("[eval] Loading test data...")
    from train_model import FEATURE_COLUMNS, load_training_data

    X, y = load_training_data(Path(args.db))
    if X is None:
        print("ERROR: No data found")
        sys.exit(1)

    # Use last 20% as test set
    split = int(len(X) * 0.8)
    X_test = X[split:]
    y_test = y[split:]

    # Scale
    X_test_scaled = scaler.transform(X_test)

    # Get probabilities
    proba = model.predict_proba(X_test_scaled)

    # Evaluate at different thresholds
    print("\n" + "=" * 70)
    print("  ACCURACY BY CONFIDENCE THRESHOLD")
    print("=" * 70)
    print(f"  {'Threshold':<12} {'Trades':<10} {'% of Total':<12} {'Accuracy':<12} {'EV/Trade':<10}")
    print("-" * 70)

    thresholds = [0.50, 0.52, 0.55, 0.57, 0.60, 0.62, 0.65, 0.70]

    for thresh in thresholds:
        # Get predictions that meet threshold
        max_proba = np.max(proba, axis=1)
        mask = max_proba >= thresh

        if mask.sum() == 0:
            print(f"  {thresh*100:.0f}%{'':<9} {'0':<10} {'0.0%':<12} {'N/A':<12} {'N/A':<10}")
            continue

        # Get predictions and actual for filtered samples
        filtered_pred = np.argmax(proba[mask], axis=1)
        filtered_actual = y_test[mask]

        # Calculate metrics
        n_trades = mask.sum()
        pct_total = n_trades / len(y_test) * 100
        accuracy = accuracy_score(filtered_actual, filtered_pred)

        # Expected value per trade (assuming 50/50 market pricing)
        # Win = +$0.50, Lose = -$0.50
        ev_per_trade = (accuracy * 0.50) - ((1 - accuracy) * 0.50)

        print(f"  {thresh*100:.0f}%{'':<9} {n_trades:<10} {pct_total:.1f}%{'':<8} {accuracy*100:.1f}%{'':<8} ${ev_per_trade:+.3f}")

    print("=" * 70)

    # Show daily projections
    print("\n" + "=" * 70)
    print("  DAILY PROJECTION (assuming 96 markets/day)")
    print("=" * 70)
    print(f"  {'Threshold':<12} {'Trades/Day':<12} {'Expected Wins':<14} {'Daily EV':<12}")
    print("-" * 70)

    for thresh in thresholds:
        max_proba = np.max(proba, axis=1)
        mask = max_proba >= thresh

        if mask.sum() == 0:
            continue

        filtered_pred = np.argmax(proba[mask], axis=1)
        filtered_actual = y_test[mask]

        n_trades = mask.sum()
        pct_total = n_trades / len(y_test)
        accuracy = accuracy_score(filtered_actual, filtered_pred)

        trades_per_day = int(96 * pct_total)
        expected_wins = trades_per_day * accuracy
        ev_per_trade = (accuracy * 0.50) - ((1 - accuracy) * 0.50)
        daily_ev = trades_per_day * ev_per_trade

        print(f"  {thresh*100:.0f}%{'':<9} {trades_per_day:<12} {expected_wins:.1f}{'':<11} ${daily_ev:+.2f}")

    print("=" * 70)

    # Recommendation
    print("\n  RECOMMENDATION:")
    print("  - 55% threshold: Good balance of trades and accuracy")
    print("  - 60% threshold: Higher accuracy, fewer trades")
    print("  - 65% threshold: Best accuracy, but may skip many opportunities")
    print()


if __name__ == "__main__":
    main()
