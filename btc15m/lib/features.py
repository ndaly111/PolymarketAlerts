"""Feature engineering for ML model training.

Extracts features from collected data for training a classifier
to predict BTC UP/DOWN outcomes.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Optional, Tuple

import numpy as np

try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False

from .db import DEFAULT_DB_PATH


FEATURE_COLUMNS = [
    # Price features
    "binance_price",
    "chainlink_price",
    "price_diff",  # binance - chainlink

    # Technical indicators
    "rsi_14",
    "macd",
    "macd_signal",
    "macd_histogram",
    "vwap",
    "volatility_20",
    "momentum_10",
    "sma_20",
    "price_vs_vwap",

    # Market features
    "market_up_prob",
    "market_down_prob",
    "time_to_expiry_sec",

    # Derived features
    "rsi_oversold",      # 1 if RSI < 30
    "rsi_overbought",    # 1 if RSI > 70
    "macd_bullish",      # 1 if histogram > 0
    "price_above_vwap",  # 1 if price > VWAP
    "momentum_positive", # 1 if momentum > 0
]


def load_training_data(
    db_path: Path = DEFAULT_DB_PATH,
    min_samples: int = 100,
) -> Optional[Tuple["pd.DataFrame", "pd.Series"]]:
    """Load training data from database.

    Joins price snapshots, indicators, and market data,
    then labels with actual outcomes.

    Returns:
        Tuple of (features DataFrame, labels Series) or None
    """
    if not HAS_PANDAS:
        raise ImportError("pandas is required for ML training")

    with sqlite3.connect(str(db_path)) as conn:
        # Load and join data
        query = """
        SELECT
            p.timestamp,
            p.binance_price,
            p.chainlink_price,
            p.price_diff,
            i.rsi_14,
            i.macd,
            i.macd_signal,
            i.macd_histogram,
            i.vwap,
            i.volatility_20,
            i.momentum_10,
            i.sma_20,
            i.price_vs_vwap,
            m.market_ticker,
            m.strike_price,
            m.up_yes_price,
            m.down_yes_price,
            m.time_to_expiry_sec,
            o.outcome
        FROM price_snapshots p
        LEFT JOIN indicator_snapshots i ON p.timestamp = i.timestamp
        LEFT JOIN market_snapshots m ON p.timestamp = m.timestamp
        LEFT JOIN outcomes o ON m.market_ticker = o.market_ticker
        WHERE o.outcome IS NOT NULL
        ORDER BY p.timestamp
        """

        df = pd.read_sql_query(query, conn)

    if len(df) < min_samples:
        print(f"[features] Insufficient data: {len(df)} < {min_samples}")
        return None

    # Create derived features
    df["market_up_prob"] = df["up_yes_price"] / 100.0
    df["market_down_prob"] = df["down_yes_price"] / 100.0
    df["rsi_oversold"] = (df["rsi_14"] < 30).astype(int)
    df["rsi_overbought"] = (df["rsi_14"] > 70).astype(int)
    df["macd_bullish"] = (df["macd_histogram"] > 0).astype(int)
    df["price_above_vwap"] = (df["price_vs_vwap"] > 0).astype(int)
    df["momentum_positive"] = (df["momentum_10"] > 0).astype(int)

    # Select feature columns (handle missing)
    available_features = [c for c in FEATURE_COLUMNS if c in df.columns]
    X = df[available_features].copy()

    # Fill NaN with column means
    X = X.fillna(X.mean())

    # Labels: 1 for UP, 0 for DOWN
    y = (df["outcome"] == "UP").astype(int)

    return X, y


def prepare_live_features(
    binance_price: float,
    chainlink_price: Optional[float],
    rsi: Optional[float],
    macd: Optional[float],
    macd_signal: Optional[float],
    macd_histogram: Optional[float],
    vwap: Optional[float],
    volatility: Optional[float],
    momentum: Optional[float],
    sma_20: Optional[float],
    market_up_prob: float,
    time_to_expiry_sec: int,
) -> dict:
    """Prepare features from live data for model prediction.

    Returns:
        Dict of feature name -> value
    """
    price_diff = None
    if chainlink_price is not None:
        price_diff = binance_price - chainlink_price

    price_vs_vwap = None
    if vwap is not None and vwap > 0:
        price_vs_vwap = (binance_price - vwap) / vwap * 100

    features = {
        "binance_price": binance_price,
        "chainlink_price": chainlink_price,
        "price_diff": price_diff,
        "rsi_14": rsi,
        "macd": macd,
        "macd_signal": macd_signal,
        "macd_histogram": macd_histogram,
        "vwap": vwap,
        "volatility_20": volatility,
        "momentum_10": momentum,
        "sma_20": sma_20,
        "price_vs_vwap": price_vs_vwap,
        "market_up_prob": market_up_prob,
        "market_down_prob": 1 - market_up_prob,
        "time_to_expiry_sec": time_to_expiry_sec,

        # Derived features
        "rsi_oversold": 1 if rsi and rsi < 30 else 0,
        "rsi_overbought": 1 if rsi and rsi > 70 else 0,
        "macd_bullish": 1 if macd_histogram and macd_histogram > 0 else 0,
        "price_above_vwap": 1 if price_vs_vwap and price_vs_vwap > 0 else 0,
        "momentum_positive": 1 if momentum and momentum > 0 else 0,
    }

    return features


def features_to_array(features: dict) -> "np.ndarray":
    """Convert feature dict to numpy array in correct order."""
    return np.array([features.get(col, 0) for col in FEATURE_COLUMNS])
