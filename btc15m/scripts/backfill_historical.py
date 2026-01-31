#!/usr/bin/env python3
"""Backfill historical data for ML training.

Downloads historical BTC price data and computes indicators
to create training samples without waiting weeks.

Usage:
    python backfill_historical.py [--days 30]
"""

from __future__ import annotations

import argparse
import sqlite3
import sys
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import List, Optional

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from lib.binance import fetch_klines, Candle, get_endpoint
from lib.indicators import (
    compute_rsi,
    compute_macd,
    compute_vwap,
    compute_volatility,
    compute_price_momentum,
    compute_sma,
    compute_bollinger_bands,
    compute_atr,
    compute_stochastic,
    compute_williams_r,
    compute_cci,
    compute_obv_trend,
    compute_higher_timeframe_trend,
    compute_time_features,
    compute_candle_patterns,
)
from lib.db import ensure_schema, DEFAULT_DB_PATH

import requests


def fetch_historical_klines(
    start_time: datetime,
    end_time: datetime,
    interval: str = "1m",
    symbol: str = "BTCUSDT",
) -> List[Candle]:
    """Fetch historical klines between two timestamps."""
    endpoint = get_endpoint()
    url = f"{endpoint}/api/v3/klines"

    all_candles = []
    current_start = int(start_time.timestamp() * 1000)
    end_ms = int(end_time.timestamp() * 1000)

    while current_start < end_ms:
        params = {
            "symbol": symbol,
            "interval": interval,
            "startTime": current_start,
            "endTime": end_ms,
            "limit": 1000,
        }

        resp = requests.get(url, params=params, timeout=30)
        resp.raise_for_status()
        data = resp.json()

        if not data:
            break

        for row in data:
            all_candles.append(Candle(
                open_time=int(row[0]),
                open=float(row[1]),
                high=float(row[2]),
                low=float(row[3]),
                close=float(row[4]),
                volume=float(row[5]),
                close_time=int(row[6]),
            ))

        # Move to next batch
        current_start = data[-1][6] + 1  # close_time + 1ms

        # Rate limit
        time.sleep(0.2)

    return all_candles


def generate_15min_samples(candles: List[Candle]) -> List[dict]:
    """Generate training samples for each 15-minute window.

    For each 15-min window, compute:
    - Indicators at the START of the window
    - Outcome (UP/DOWN) based on price at END of window
    """
    samples = []

    # Group candles into 15-minute windows
    # Each 1m candle has a timestamp, so we can bucket them
    windows = {}
    for c in candles:
        # Round down to 15-minute boundary
        ts_sec = c.open_time // 1000
        window_start = (ts_sec // 900) * 900  # 900 sec = 15 min
        if window_start not in windows:
            windows[window_start] = []
        windows[window_start].append(c)

    window_times = sorted(windows.keys())

    for i, window_start in enumerate(window_times):
        window_candles = windows[window_start]

        # Need at least 60 candles of history for indicators
        if i < 60:
            continue

        # Get 60 candles leading up to this window
        history_start = window_times[max(0, i - 60)]
        history_candles = []
        for j in range(max(0, i - 60), i):
            history_candles.extend(windows[window_times[j]])

        if len(history_candles) < 30:
            continue

        closes = [c.close for c in history_candles]
        candle_dicts = [
            {"high": c.high, "low": c.low, "close": c.close, "volume": c.volume, "open": c.open}
            for c in history_candles
        ]

        # Compute indicators at window start
        rsi = compute_rsi(closes, period=14)
        macd_result = compute_macd(closes, fast=12, slow=26, signal=9)
        vwap = compute_vwap(candle_dicts)
        volatility = compute_volatility(closes, period=20)
        momentum = compute_price_momentum(closes, period=10)
        sma_20 = compute_sma(closes, period=20)

        # New indicators
        bb = compute_bollinger_bands(closes, period=20, std_dev=2.0)
        atr = compute_atr(candle_dicts, period=14)
        stoch = compute_stochastic(candle_dicts, k_period=14, d_period=3)
        williams_r = compute_williams_r(candle_dicts, period=14)
        cci = compute_cci(candle_dicts, period=20)
        obv_trend = compute_obv_trend(candle_dicts, period=10)
        htf_trend = compute_higher_timeframe_trend(closes)

        # Time features
        dt = datetime.fromtimestamp(window_start, tz=timezone.utc)
        time_feats = compute_time_features(dt.hour)

        # Candle patterns
        patterns = compute_candle_patterns(candle_dicts)

        price_at_start = window_candles[0].open if window_candles else closes[-1]
        price_at_end = window_candles[-1].close if window_candles else closes[-1]

        price_vs_vwap = None
        if vwap:
            price_vs_vwap = (price_at_start - vwap) / vwap * 100

        # Determine outcome
        outcome = "UP" if price_at_end > price_at_start else "DOWN"

        timestamp = dt.isoformat()

        samples.append({
            "timestamp": timestamp,
            "window_start": window_start,
            "price_at_start": price_at_start,
            "price_at_end": price_at_end,
            "price_change": price_at_end - price_at_start,
            "price_change_pct": (price_at_end - price_at_start) / price_at_start * 100,
            # Original indicators
            "rsi": rsi,
            "macd": macd_result.macd if macd_result else None,
            "macd_signal": macd_result.signal if macd_result else None,
            "macd_histogram": macd_result.histogram if macd_result else None,
            "vwap": vwap,
            "volatility": volatility,
            "momentum": momentum,
            "sma_20": sma_20,
            "price_vs_vwap": price_vs_vwap,
            # Bollinger Bands
            "bb_percent_b": bb["percent_b"] if bb else None,
            "bb_bandwidth": bb["bandwidth"] if bb else None,
            # ATR
            "atr": atr,
            # Stochastic
            "stoch_k": stoch["k"] if stoch else None,
            # Williams %R
            "williams_r": williams_r,
            # CCI
            "cci": cci,
            # OBV trend
            "obv_trend": obv_trend,
            # Higher timeframe trends
            "trend_15m": htf_trend.get("trend_15m"),
            "trend_1h": htf_trend.get("trend_1h"),
            "change_15m": htf_trend.get("change_15m"),
            "change_1h": htf_trend.get("change_1h"),
            # Time features
            "hour": time_feats["hour"],
            "hour_sin": time_feats["hour_sin"],
            "hour_cos": time_feats["hour_cos"],
            "asia_session": time_feats["asia_session"],
            "europe_session": time_feats["europe_session"],
            "us_session": time_feats["us_session"],
            "high_vol_hour": time_feats["high_vol_hour"],
            # Candle patterns
            "doji": patterns.get("doji", 0),
            "hammer": patterns.get("hammer", 0),
            "shooting_star": patterns.get("shooting_star", 0),
            "bullish_engulfing": patterns.get("bullish_engulfing", 0),
            "bearish_engulfing": patterns.get("bearish_engulfing", 0),
            # Outcome
            "outcome": outcome,
        })

    return samples


def store_historical_samples(samples: List[dict], db_path: Path) -> int:
    """Store historical samples in a new table for training."""
    with sqlite3.connect(str(db_path)) as conn:
        # Drop old table to recreate with new schema
        conn.execute("DROP TABLE IF EXISTS historical_samples")

        conn.execute("""
            CREATE TABLE IF NOT EXISTS historical_samples (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                window_start INTEGER NOT NULL,
                price_at_start REAL,
                price_at_end REAL,
                price_change REAL,
                price_change_pct REAL,
                -- Original indicators
                rsi REAL,
                macd REAL,
                macd_signal REAL,
                macd_histogram REAL,
                vwap REAL,
                volatility REAL,
                momentum REAL,
                sma_20 REAL,
                price_vs_vwap REAL,
                -- Bollinger Bands
                bb_percent_b REAL,
                bb_bandwidth REAL,
                -- ATR
                atr REAL,
                -- Stochastic
                stoch_k REAL,
                -- Williams %R
                williams_r REAL,
                -- CCI
                cci REAL,
                -- OBV trend
                obv_trend REAL,
                -- Higher timeframe
                trend_15m REAL,
                trend_1h REAL,
                change_15m REAL,
                change_1h REAL,
                -- Time features
                hour INTEGER,
                hour_sin REAL,
                hour_cos REAL,
                asia_session INTEGER,
                europe_session INTEGER,
                us_session INTEGER,
                high_vol_hour INTEGER,
                -- Candle patterns
                doji INTEGER,
                hammer INTEGER,
                shooting_star INTEGER,
                bullish_engulfing INTEGER,
                bearish_engulfing INTEGER,
                -- Outcome
                outcome TEXT NOT NULL,
                UNIQUE(window_start)
            )
        """)

        inserted = 0
        for s in samples:
            try:
                conn.execute("""
                    INSERT OR IGNORE INTO historical_samples
                    (timestamp, window_start, price_at_start, price_at_end,
                     price_change, price_change_pct, rsi, macd, macd_signal,
                     macd_histogram, vwap, volatility, momentum, sma_20,
                     price_vs_vwap, bb_percent_b, bb_bandwidth, atr, stoch_k,
                     williams_r, cci, obv_trend, trend_15m, trend_1h,
                     change_15m, change_1h, hour, hour_sin, hour_cos,
                     asia_session, europe_session, us_session, high_vol_hour,
                     doji, hammer, shooting_star, bullish_engulfing,
                     bearish_engulfing, outcome)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                            ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                            ?, ?, ?, ?, ?)
                """, (
                    s["timestamp"], s["window_start"], s["price_at_start"],
                    s["price_at_end"], s["price_change"], s["price_change_pct"],
                    s["rsi"], s["macd"], s["macd_signal"], s["macd_histogram"],
                    s["vwap"], s["volatility"], s["momentum"], s["sma_20"],
                    s["price_vs_vwap"], s["bb_percent_b"], s["bb_bandwidth"],
                    s["atr"], s["stoch_k"], s["williams_r"], s["cci"],
                    s["obv_trend"], s["trend_15m"], s["trend_1h"],
                    s["change_15m"], s["change_1h"], s["hour"], s["hour_sin"],
                    s["hour_cos"], s["asia_session"], s["europe_session"],
                    s["us_session"], s["high_vol_hour"], s["doji"], s["hammer"],
                    s["shooting_star"], s["bullish_engulfing"],
                    s["bearish_engulfing"], s["outcome"],
                ))
                inserted += 1
            except Exception as e:
                continue

        conn.commit()

    return inserted


def main():
    parser = argparse.ArgumentParser(description="Backfill historical BTC data")
    parser.add_argument("--days", type=int, default=30,
                        help="Number of days to backfill (default: 30)")
    parser.add_argument("--db", type=str, default=str(DEFAULT_DB_PATH),
                        help="Database path")
    args = parser.parse_args()

    db_path = Path(args.db)
    ensure_schema(db_path)

    end_time = datetime.now(timezone.utc)
    start_time = end_time - timedelta(days=args.days)

    print(f"[backfill] Fetching {args.days} days of historical data...")
    print(f"[backfill] From: {start_time.strftime('%Y-%m-%d')}")
    print(f"[backfill] To:   {end_time.strftime('%Y-%m-%d')}")
    print()

    # Fetch historical 1-minute candles
    print("[backfill] Downloading 1-minute candles from Binance...")
    candles = fetch_historical_klines(start_time, end_time, interval="1m")
    print(f"[backfill] Downloaded {len(candles):,} candles")

    # Generate 15-minute samples
    print("[backfill] Generating 15-minute training samples...")
    samples = generate_15min_samples(candles)
    print(f"[backfill] Generated {len(samples):,} samples")

    # Show outcome distribution
    up_count = sum(1 for s in samples if s["outcome"] == "UP")
    down_count = len(samples) - up_count
    print(f"[backfill] UP: {up_count} ({up_count/len(samples)*100:.1f}%)")
    print(f"[backfill] DOWN: {down_count} ({down_count/len(samples)*100:.1f}%)")

    # Store in database
    print(f"\n[backfill] Storing samples in {db_path}...")
    inserted = store_historical_samples(samples, db_path)
    print(f"[backfill] Inserted {inserted:,} samples")

    print("\n[backfill] Done! You can now train an ML model on this data.")
    print(f"[backfill] Total 15-min samples: {len(samples):,}")


if __name__ == "__main__":
    main()
