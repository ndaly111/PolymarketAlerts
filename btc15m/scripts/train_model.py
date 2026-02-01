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

try:
    import lightgbm as lgb
    HAS_LIGHTGBM = True
except ImportError:
    HAS_LIGHTGBM = False


FEATURE_COLUMNS = [
    # Original indicators
    "rsi",
    "macd",
    "macd_signal",
    "macd_histogram",
    "volatility",
    "momentum",
    "price_vs_vwap",
    # Bollinger Bands
    "bb_percent_b",
    "bb_bandwidth",
    # ATR
    "atr",
    # Stochastic
    "stoch_k",
    # Williams %R
    "williams_r",
    # CCI
    "cci",
    # OBV trend
    "obv_trend",
    # Higher timeframe
    "trend_15m",
    "trend_1h",
    "change_15m",
    "change_1h",
    # Time features
    "hour",
    "hour_sin",
    "hour_cos",
    "asia_session",
    "europe_session",
    "us_session",
    "high_vol_hour",
    # Candle patterns
    "doji",
    "hammer",
    "shooting_star",
    "bullish_engulfing",
    "bearish_engulfing",
    # ===== BATCH 2 INDICATORS =====
    # ADX
    "adx",
    "plus_di",
    "minus_di",
    # MFI
    "mfi",
    # ROC
    "roc",
    # TRIX
    "trix",
    # Keltner Channels
    "kc_position",
    "kc_width",
    # Donchian Channels
    "dc_position",
    "dc_width",
    # Ichimoku
    "ichi_tk_cross",
    "ichi_above_cloud",
    "ichi_below_cloud",
    "ichi_cloud_thickness",
    # VWMA
    "price_vs_vwma",
    # CMF
    "cmf",
    # Choppiness
    "chop",
    # Squeeze
    "squeeze_on",
    "squeeze_momentum",
    # Volume
    "vol_ratio",
    "vol_trend",
    "vol_up_ratio",
    "vol_spike",
    # Price action
    "dist_to_high_pct",
    "dist_to_low_pct",
    "range_compression",
    "higher_highs",
    "lower_lows",
    "up_streak",
    "down_streak",
    # Pivot
    "above_pivot",
    "dist_to_nearest_pivot",
    # Linear regression
    "lr_slope",
    "lr_deviation",
    "lr_r_squared",
    # ===== COMPOSITE INDICATORS =====
    "comp_trend_composite",
    "comp_weighted_trend",
    "comp_ob_os_signal",
    "comp_overbought_consensus",
    "comp_oversold_consensus",
    "comp_volatility_composite",
    "comp_is_ranging",
    "comp_is_trending",
    "comp_breakout_potential",
    "comp_vol_price_confirmation",
    "comp_vol_price_divergence",
    "comp_strong_move",
    "comp_mtf_aligned",
    "comp_mtf_direction",
    "comp_mtf_acceleration",
    "comp_breakout_score",
    "comp_breakout_direction",
    "comp_has_divergence",
    "comp_divergence_strength",
    # ===== NEW: FUNDING RATE FEATURES =====
    "funding_rate",
    "funding_annualized",
    "funding_extreme",
    "funding_bullish",
    "funding_bearish",
    "long_short_ratio",
    "ls_ratio_extreme",
    # ===== NEW: CROSS-ASSET FEATURES =====
    "eth_change_15m",
    "eth_change_1h",
    "eth_momentum",
    "eth_btc_divergence",
    "eth_leading",
    "eth_lagging",
    "btc_eth_ratio",
    "cross_asset_bullish",
    "cross_asset_bearish",
    # ===== NEW: FEAR & GREED INDEX =====
    "fg_value",
    "fg_normalized",
    "fg_extreme_fear",
    "fg_fear",
    "fg_neutral",
    "fg_greed",
    "fg_extreme_greed",
    "fg_contrarian_buy",
    "fg_contrarian_sell",
    # ===== NEW: DERIBIT OPTIONS =====
    "deribit_iv",
    "deribit_iv_high",
    "deribit_iv_low",
    "deribit_pc_ratio",
    "deribit_pc_bullish",
    "deribit_pc_bearish",
    "deribit_call_oi",
    "deribit_put_oi",
    "deribit_oi_imbalance",
    # ===== MARKET CONTEXT (critical for timing) =====
    "time_to_expiry_min",      # Minutes until contract expires (1-60)
    "distance_to_strike_pct",  # % distance from current price to strike
    "above_strike",            # 1 if price > strike, 0 otherwise
    "time_distance_interaction",  # time * distance (less time + far = bad)
    # ===== VELOCITY & MOMENTUM (immediate price action) =====
    "price_velocity_1m",       # Price change in last 1 minute (%)
    "price_velocity_5m",       # Price change in last 5 minutes (%)
    "price_acceleration",      # Is momentum increasing or decreasing?
    "distance_velocity",       # Is price moving toward or away from strike?
    "vol_ratio_5m",            # Recent volatility vs historical
    "strikes_crossed",         # How many strike levels crossed in last 15 min
]


def load_training_data(db_path: Path) -> tuple:
    """Load training data from historical samples table."""
    with sqlite3.connect(str(db_path)) as conn:
        conn.row_factory = sqlite3.Row
        # Build dynamic column list from FEATURE_COLUMNS
        cols = ", ".join(FEATURE_COLUMNS)
        rows = conn.execute(f"""
            SELECT {cols}, outcome
            FROM historical_samples
            WHERE rsi IS NOT NULL
              AND macd IS NOT NULL
            ORDER BY window_start
        """).fetchall()

    if not rows:
        return None, None

    X = []
    y = []

    # Default values for missing data
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
        # Batch 2
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
        # Composites
        "comp_trend_composite": 0, "comp_weighted_trend": 0,
        "comp_ob_os_signal": 0, "comp_overbought_consensus": 0, "comp_oversold_consensus": 0,
        "comp_volatility_composite": 0, "comp_is_ranging": 0, "comp_is_trending": 0,
        "comp_breakout_potential": 0, "comp_vol_price_confirmation": 0,
        "comp_vol_price_divergence": 0, "comp_strong_move": 0,
        "comp_mtf_aligned": 0, "comp_mtf_direction": 0, "comp_mtf_acceleration": 0,
        "comp_breakout_score": 0, "comp_breakout_direction": 0,
        "comp_has_divergence": 0, "comp_divergence_strength": 0,
        # New: Funding
        "funding_rate": 0, "funding_annualized": 0, "funding_extreme": 0,
        "funding_bullish": 0, "funding_bearish": 0, "long_short_ratio": 1.0,
        "ls_ratio_extreme": 0,
        # New: Cross-asset
        "eth_change_15m": 0, "eth_change_1h": 0, "eth_momentum": 0,
        "eth_btc_divergence": 0, "eth_leading": 0, "eth_lagging": 0,
        "btc_eth_ratio": 25, "cross_asset_bullish": 0, "cross_asset_bearish": 0,
        # New: Fear & Greed
        "fg_value": 50, "fg_normalized": 0, "fg_extreme_fear": 0,
        "fg_fear": 0, "fg_neutral": 1, "fg_greed": 0, "fg_extreme_greed": 0,
        "fg_contrarian_buy": 0, "fg_contrarian_sell": 0,
        # New: Deribit
        "deribit_iv": 50, "deribit_iv_high": 0, "deribit_iv_low": 0,
        "deribit_pc_ratio": 1.0, "deribit_pc_bullish": 0, "deribit_pc_bearish": 0,
        "deribit_call_oi": 0, "deribit_put_oi": 0, "deribit_oi_imbalance": 0,
    }

    for row in rows:
        features = []
        for col in FEATURE_COLUMNS:
            val = row[col] if col in row.keys() else None
            if val is None:
                val = defaults.get(col, 0)
            features.append(val)
        X.append(features)
        y.append(1 if row["outcome"] == "UP" else 0)

    return np.array(X), np.array(y)


def train_model(X: np.ndarray, y: np.ndarray, sample_weights: np.ndarray = None) -> tuple:
    """Train and evaluate the model.

    Args:
        X: Feature matrix
        y: Target labels
        sample_weights: Optional profit-based weights (higher weight = more influence)
    """
    # Split data (and weights if provided)
    if sample_weights is not None:
        X_train, X_test, y_train, y_test, w_train, w_test = train_test_split(
            X, y, sample_weights, test_size=0.2, random_state=42, shuffle=False
        )
        print(f"[train] Using profit-weighted training (weight range: {w_train.min():.2f}-{w_train.max():.2f})")
    else:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, shuffle=False  # Don't shuffle time series
        )
        w_train = None

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train Random Forest with sample weights
    print("[train] Training Random Forest...")
    rf_model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        min_samples_split=10,
        random_state=42,
        n_jobs=-1,
    )
    rf_model.fit(X_train_scaled, y_train, sample_weight=w_train)

    # Train Gradient Boosting with sample weights
    print("[train] Training Gradient Boosting...")
    gb_model = GradientBoostingClassifier(
        n_estimators=100,
        max_depth=5,
        learning_rate=0.1,
        random_state=42,
    )
    gb_model.fit(X_train_scaled, y_train, sample_weight=w_train)

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

    # Try LightGBM if available
    lgb_model = None
    lgb_acc = 0
    if HAS_LIGHTGBM:
        print("\n" + "=" * 50)
        print("LIGHTGBM RESULTS")
        print("=" * 50)
        lgb_model = lgb.LGBMClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.05,
            num_leaves=31,
            min_child_samples=20,
            reg_alpha=0.1,
            reg_lambda=0.1,
            random_state=42,
            verbose=-1,
        )
        lgb_model.fit(X_train_scaled, y_train, sample_weight=w_train)
        lgb_pred = lgb_model.predict(X_test_scaled)
        lgb_acc = accuracy_score(y_test, lgb_pred)
        print(f"Accuracy: {lgb_acc:.3f}")
        print("\nClassification Report:")
        print(classification_report(y_test, lgb_pred, target_names=["DOWN", "UP"]))

        print("\nFeature Importance (LightGBM):")
        for name, importance in sorted(
            zip(FEATURE_COLUMNS, lgb_model.feature_importances_),
            key=lambda x: x[1],
            reverse=True,
        )[:15]:
            print(f"  {name:20s}: {importance}")

    # Choose best model
    accuracies = [("RandomForest", rf_acc, rf_model), ("GradientBoosting", gb_acc, gb_model)]
    if lgb_model is not None:
        accuracies.append(("LightGBM", lgb_acc, lgb_model))

    best_name, best_acc, best_model = max(accuracies, key=lambda x: x[1])

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
