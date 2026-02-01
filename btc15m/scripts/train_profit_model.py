#!/usr/bin/env python3
"""
Train a profit-optimized ML model for BTC 15-min trading.

Key differences from direction-only model:
1. Includes market prices (yes_ask, no_ask, spread) as features
2. Trains on profitability, not just direction accuracy
3. Uses profit-weighted samples
4. Outputs probability of profitable trade at given price
"""

import json
import pickle
import sqlite3
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional

import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix

try:
    import lightgbm as lgb
    HAS_LIGHTGBM = True
except ImportError:
    HAS_LIGHTGBM = False

# Paths
SCRIPT_DIR = Path(__file__).resolve().parent
BTC15M_DIR = SCRIPT_DIR.parent
DB_PATH = BTC15M_DIR / "data" / "btc15m.db"
HOURLY_DB_PATH = BTC15M_DIR / "hourly_trades.db"
MODEL_PATH = BTC15M_DIR / "models" / "btc15m_profit_model.pkl"

KALSHI_FEE_RATE = 0.07


def load_15min_samples(db_path: Path) -> List[Dict]:
    """Load 15-min prediction samples with outcomes."""
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    c = conn.cursor()

    c.execute("""
        SELECT * FROM prediction_samples
        WHERE actual_outcome IS NOT NULL
        AND features_json IS NOT NULL
    """)

    samples = []
    for row in c.fetchall():
        try:
            features = json.loads(row["features_json"])

            # Calculate P&L for this trade
            entry_price = row["market_ask"] or 0.5
            was_correct = row["was_correct"]

            if was_correct:
                pnl = (1 - entry_price) * (1 - KALSHI_FEE_RATE)
            else:
                pnl = -entry_price

            samples.append({
                "features": features,
                "market_ask": entry_price,
                "edge": row["edge"],
                "model_prob": row["model_prob"],
                "prediction": row["prediction"],
                "actual_outcome": row["actual_outcome"],
                "was_correct": was_correct,
                "pnl": pnl,
                "is_profitable": pnl > 0,
            })
        except:
            continue

    conn.close()
    return samples


def load_hourly_samples(db_path: Path) -> List[Dict]:
    """Load hourly prediction samples with outcomes."""
    if not db_path.exists():
        return []

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    c = conn.cursor()

    c.execute("""
        SELECT * FROM hourly_predictions
        WHERE settled = 1
        AND features_json IS NOT NULL
    """)

    samples = []
    for row in c.fetchall():
        try:
            features = json.loads(row["features_json"])

            # Calculate P&L
            entry_price = (row["yes_ask"] or 50) / 100
            was_correct = row["prediction_correct"]

            if was_correct:
                pnl = (1 - entry_price) * (1 - KALSHI_FEE_RATE)
            else:
                pnl = -entry_price

            samples.append({
                "features": features,
                "market_ask": entry_price,
                "edge": row["edge"] or 0,
                "model_prob": row["model_prob"] or 0.5,
                "prediction": row["prediction"],
                "actual_outcome": row["actual_outcome"],
                "was_correct": was_correct,
                "pnl": pnl,
                "is_profitable": pnl > 0,
                # Hourly-specific features
                "distance_pct": row["distance_pct"],
                "time_to_expiry_sec": row["time_to_expiry_sec"],
                "volatility_1h": row["volatility_1h"],
                "trend_1h": row["trend_1h"],
            })
        except:
            continue

    conn.close()
    return samples


def prepare_features(samples: List[Dict]) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Prepare feature matrix and target vector."""
    if not samples:
        return np.array([]), np.array([]), []

    # Get all unique feature names from first sample
    base_features = set(samples[0]["features"].keys())

    # Additional features we always include
    extra_features = [
        "market_ask",
        "edge",
        "model_prob",
        "distance_pct",
        "time_to_expiry_sec",
        "volatility_1h",
        "trend_1h",
    ]

    # Build feature list (sorted for consistency)
    feature_names = sorted(base_features) + extra_features

    X = []
    y = []
    weights = []

    for sample in samples:
        row = []

        # Base features from technical indicators
        for fname in sorted(base_features):
            val = sample["features"].get(fname, 0)
            if isinstance(val, bool):
                val = int(val)
            elif val is None:
                val = 0
            row.append(float(val))

        # Extra features
        row.append(sample.get("market_ask", 0.5))
        row.append(sample.get("edge", 0))
        row.append(sample.get("model_prob", 0.5))
        row.append(sample.get("distance_pct", 0) or 0)
        row.append(sample.get("time_to_expiry_sec", 600) or 600)
        row.append(sample.get("volatility_1h", 0) or 0)
        row.append(sample.get("trend_1h", 0) or 0)

        X.append(row)
        y.append(1 if sample["is_profitable"] else 0)

        # Profit-weighted: weight by absolute P&L
        weight = 1 + abs(sample["pnl"]) * 2
        weights.append(weight)

    return np.array(X), np.array(y), feature_names, np.array(weights)


def train_model(X: np.ndarray, y: np.ndarray, weights: np.ndarray, feature_names: List[str]) -> Dict:
    """Train multiple models and pick the best."""
    print(f"\nTraining on {len(X)} samples...")
    print(f"Class distribution: {sum(y)} profitable, {len(y) - sum(y)} unprofitable")

    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Split data
    X_train, X_test, y_train, y_test, w_train, w_test = train_test_split(
        X_scaled, y, weights, test_size=0.2, random_state=42, stratify=y
    )

    models = {}

    # Random Forest
    print("\nTraining Random Forest...")
    rf = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        min_samples_leaf=3,
        class_weight="balanced",
        random_state=42,
    )
    rf.fit(X_train, y_train, sample_weight=w_train)
    rf_score = rf.score(X_test, y_test)
    models["RandomForest"] = (rf, rf_score)
    print(f"  Accuracy: {rf_score*100:.1f}%")

    # Gradient Boosting
    print("\nTraining Gradient Boosting...")
    gb = GradientBoostingClassifier(
        n_estimators=100,
        max_depth=5,
        learning_rate=0.1,
        random_state=42,
    )
    gb.fit(X_train, y_train, sample_weight=w_train)
    gb_score = gb.score(X_test, y_test)
    models["GradientBoosting"] = (gb, gb_score)
    print(f"  Accuracy: {gb_score*100:.1f}%")

    # LightGBM if available
    if HAS_LIGHTGBM:
        print("\nTraining LightGBM...")
        lgbm = lgb.LGBMClassifier(
            n_estimators=100,
            max_depth=8,
            learning_rate=0.1,
            num_leaves=31,
            class_weight="balanced",
            random_state=42,
            verbose=-1,
        )
        lgbm.fit(X_train, y_train, sample_weight=w_train)
        lgbm_score = lgbm.score(X_test, y_test)
        models["LightGBM"] = (lgbm, lgbm_score)
        print(f"  Accuracy: {lgbm_score*100:.1f}%")

    # Pick best model
    best_name = max(models, key=lambda k: models[k][1])
    best_model, best_score = models[best_name]

    print(f"\nBest model: {best_name} ({best_score*100:.1f}%)")

    # Detailed evaluation
    y_pred = best_model.predict(X_test)
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=["Unprofitable", "Profitable"]))

    # Feature importance
    feature_importance = []
    if hasattr(best_model, "feature_importances_"):
        importances = list(zip(feature_names, best_model.feature_importances_))
        importances.sort(key=lambda x: x[1], reverse=True)
        feature_importance = importances

        print("\nTop 20 Features (most predictive of profit):")
        for fname, imp in importances[:20]:
            print(f"  {fname}: {imp:.4f}")

        # Identify features that might be noise (very low importance)
        low_importance = [f for f, i in importances if i < 0.001]
        if low_importance:
            print(f"\n{len(low_importance)} features with <0.1% importance (potential noise)")

    # Cross-validation
    cv_scores = cross_val_score(best_model, X_scaled, y, cv=5)
    print(f"\nCross-validation: {cv_scores.mean()*100:.1f}% (+/- {cv_scores.std()*100:.1f}%)")

    return {
        "model": best_model,
        "scaler": scaler,
        "feature_names": feature_names,
        "model_name": best_name,
        "accuracy": best_score,
        "cv_mean": cv_scores.mean(),
        "cv_std": cv_scores.std(),
        "feature_importance": feature_importance,
        "top_features": [f for f, _ in feature_importance[:10]],
    }


def save_model(model_data: Dict, path: Path):
    """Save model and metadata."""
    path.parent.mkdir(parents=True, exist_ok=True)

    # Backup existing model
    if path.exists():
        backup_path = path.with_suffix(f".pkl.backup.{int(datetime.now().timestamp())}")
        path.rename(backup_path)
        print(f"Backed up old model to {backup_path.name}")

    with open(path, "wb") as f:
        pickle.dump(model_data, f)

    print(f"Saved model to {path}")

    # Also save feature importance to JSON for easy analysis
    if model_data.get("feature_importance"):
        importance_path = path.parent / "feature_importance.json"
        importance_data = {
            "timestamp": datetime.now().isoformat(),
            "model": model_data.get("model_name", "Unknown"),
            "accuracy": model_data.get("accuracy", 0),
            "features": [
                {"name": name, "importance": float(imp)}
                for name, imp in model_data["feature_importance"]
            ],
            "top_10": model_data.get("top_features", []),
        }
        with open(importance_path, "w") as f:
            json.dump(importance_data, f, indent=2)
        print(f"Saved feature importance to {importance_path}")


def main():
    print("=" * 60)
    print("PROFIT-OPTIMIZED ML MODEL TRAINER")
    print("=" * 60)

    # Load all available samples
    print("\nLoading 15-min samples...")
    samples_15m = load_15min_samples(DB_PATH)
    print(f"  Loaded {len(samples_15m)} samples")

    print("\nLoading hourly samples...")
    samples_hourly = load_hourly_samples(HOURLY_DB_PATH)
    print(f"  Loaded {len(samples_hourly)} samples")

    # Combine all samples
    all_samples = samples_15m + samples_hourly
    print(f"\nTotal samples: {len(all_samples)}")

    if len(all_samples) < 20:
        print("ERROR: Need at least 20 samples to train")
        sys.exit(1)

    # Analyze profitability distribution
    profitable = sum(1 for s in all_samples if s["is_profitable"])
    print(f"Profitable trades: {profitable} ({profitable/len(all_samples)*100:.1f}%)")

    total_pnl = sum(s["pnl"] for s in all_samples)
    print(f"Total P&L if all traded: ${total_pnl:+.2f}")

    # Prepare features
    X, y, feature_names, weights = prepare_features(all_samples)
    print(f"\nFeature matrix: {X.shape}")

    # Train model
    model_data = train_model(X, y, weights, feature_names)

    # Save model
    save_model(model_data, MODEL_PATH)

    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)
    print(f"Model: {model_data['model_name']}")
    print(f"Accuracy: {model_data['accuracy']*100:.1f}%")
    print(f"Cross-val: {model_data['cv_mean']*100:.1f}% (+/- {model_data['cv_std']*100:.1f}%)")
    print(f"Features: {len(model_data['feature_names'])}")
    print(f"Saved to: {MODEL_PATH}")


if __name__ == "__main__":
    main()
