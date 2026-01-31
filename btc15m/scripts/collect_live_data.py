#!/usr/bin/env python3
"""Collect live BTC price, indicators, and Kalshi market data.

Run continuously to gather training data for ML model.
Data is stored in SQLite for later analysis and model training.

Usage:
    python collect_live_data.py [--interval 15]
"""

from __future__ import annotations

import argparse
import signal
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from lib.binance import fetch_klines, fetch_current_price, Candle
from lib.chainlink import get_chainlink_btc_price
from lib.indicators import (
    compute_rsi,
    compute_macd,
    compute_vwap,
    compute_volatility,
    compute_price_momentum,
    compute_sma,
)
from lib.kalshi_btc import fetch_btc_15min_markets, calculate_market_implied_prob
from lib.db import (
    ensure_schema,
    insert_price_snapshot,
    insert_indicator_snapshot,
    insert_market_snapshot,
    DEFAULT_DB_PATH,
)


# Graceful shutdown
shutdown_requested = False


def signal_handler(signum, frame):
    global shutdown_requested
    print("\n[collector] Shutdown requested...")
    shutdown_requested = True


def candles_to_dicts(candles: list[Candle]) -> list[dict]:
    """Convert Candle objects to dicts for indicator functions."""
    return [
        {
            "open": c.open,
            "high": c.high,
            "low": c.low,
            "close": c.close,
            "volume": c.volume,
        }
        for c in candles
    ]


def collect_snapshot() -> dict:
    """Collect a single data snapshot."""
    timestamp = datetime.now(timezone.utc).isoformat()
    snapshot = {"timestamp": timestamp, "errors": []}

    # 1. Fetch prices
    try:
        binance_price = fetch_current_price()
        snapshot["binance_price"] = binance_price
    except Exception as e:
        snapshot["errors"].append(f"binance_price: {e}")
        snapshot["binance_price"] = None

    try:
        chainlink_price = get_chainlink_btc_price()
        snapshot["chainlink_price"] = chainlink_price
    except Exception as e:
        snapshot["errors"].append(f"chainlink_price: {e}")
        snapshot["chainlink_price"] = None

    # 2. Fetch candles for indicators
    try:
        candles_1m = fetch_klines(interval="1m", limit=60)
        closes = [c.close for c in candles_1m]
        candle_dicts = candles_to_dicts(candles_1m)

        # Calculate indicators
        snapshot["rsi"] = compute_rsi(closes, period=14)

        macd_result = compute_macd(closes, fast=12, slow=26, signal=9)
        if macd_result:
            snapshot["macd"] = macd_result.macd
            snapshot["macd_signal"] = macd_result.signal
            snapshot["macd_histogram"] = macd_result.histogram
        else:
            snapshot["macd"] = None
            snapshot["macd_signal"] = None
            snapshot["macd_histogram"] = None

        snapshot["vwap"] = compute_vwap(candle_dicts)
        snapshot["volatility"] = compute_volatility(closes, period=20)
        snapshot["momentum"] = compute_price_momentum(closes, period=10)
        snapshot["sma_20"] = compute_sma(closes, period=20)

        # Price vs VWAP
        if snapshot["vwap"] and snapshot["binance_price"]:
            snapshot["price_vs_vwap"] = (
                (snapshot["binance_price"] - snapshot["vwap"]) / snapshot["vwap"] * 100
            )
        else:
            snapshot["price_vs_vwap"] = None

    except Exception as e:
        snapshot["errors"].append(f"indicators: {e}")
        for key in ["rsi", "macd", "macd_signal", "macd_histogram",
                    "vwap", "volatility", "momentum", "sma_20", "price_vs_vwap"]:
            snapshot[key] = None

    # 3. Fetch Kalshi markets
    try:
        markets = fetch_btc_15min_markets()
        snapshot["markets"] = []

        for m in markets[:3]:  # Top 3 nearest expiry
            probs = calculate_market_implied_prob(m)
            time_to_expiry = (m.expiry_time - datetime.now(timezone.utc)).total_seconds()

            snapshot["markets"].append({
                "ticker": m.ticker,
                "expiry_time": m.expiry_time.isoformat(),
                "strike_price": m.strike_price,
                "up_yes_price": m.yes_bid,
                "down_yes_price": m.no_bid,
                "up_prob": probs["up_prob"],
                "down_prob": probs["down_prob"],
                "time_to_expiry_sec": int(time_to_expiry),
            })

    except Exception as e:
        snapshot["errors"].append(f"kalshi: {e}")
        snapshot["markets"] = []

    return snapshot


def store_snapshot(snapshot: dict) -> None:
    """Store snapshot in database."""
    ts = snapshot["timestamp"]

    # Store price snapshot
    if snapshot.get("binance_price"):
        insert_price_snapshot(
            binance_price=snapshot["binance_price"],
            chainlink_price=snapshot.get("chainlink_price"),
            timestamp=ts,
        )

    # Store indicator snapshot
    insert_indicator_snapshot(
        rsi=snapshot.get("rsi"),
        macd=snapshot.get("macd"),
        macd_signal=snapshot.get("macd_signal"),
        macd_histogram=snapshot.get("macd_histogram"),
        vwap=snapshot.get("vwap"),
        volatility=snapshot.get("volatility"),
        momentum=snapshot.get("momentum"),
        sma_20=snapshot.get("sma_20"),
        price_vs_vwap=snapshot.get("price_vs_vwap"),
        timestamp=ts,
    )

    # Store market snapshots
    for m in snapshot.get("markets", []):
        insert_market_snapshot(
            market_ticker=m["ticker"],
            expiry_time=m.get("expiry_time"),
            strike_price=m.get("strike_price"),
            up_yes_price=m.get("up_yes_price"),
            down_yes_price=m.get("down_yes_price"),
            time_to_expiry_sec=m.get("time_to_expiry_sec"),
            timestamp=ts,
        )


def print_snapshot(snapshot: dict) -> None:
    """Print snapshot summary to console."""
    ts = snapshot["timestamp"][:19]
    binance = snapshot.get("binance_price")
    chainlink = snapshot.get("chainlink_price")
    rsi = snapshot.get("rsi")
    macd_hist = snapshot.get("macd_histogram")

    price_str = f"${binance:,.2f}" if binance else "N/A"
    chain_str = f"${chainlink:,.2f}" if chainlink else "N/A"
    rsi_str = f"{rsi:.1f}" if rsi else "N/A"
    macd_str = f"{macd_hist:+.2f}" if macd_hist else "N/A"

    print(f"[{ts}] BTC: {price_str} | Chainlink: {chain_str} | RSI: {rsi_str} | MACD: {macd_str}")

    for m in snapshot.get("markets", []):
        tte = m.get("time_to_expiry_sec", 0)
        up_prob = m.get("up_prob", 0)
        print(f"  └─ {m['ticker']}: UP {up_prob*100:.1f}% | Expires in {tte//60}m {tte%60}s")

    if snapshot.get("errors"):
        for err in snapshot["errors"]:
            print(f"  [ERROR] {err}")


def main():
    parser = argparse.ArgumentParser(description="Collect live BTC data")
    parser.add_argument("--interval", type=int, default=15,
                        help="Collection interval in seconds (default: 15)")
    parser.add_argument("--db", type=str, default=str(DEFAULT_DB_PATH),
                        help="Database path")
    args = parser.parse_args()

    # Set up signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # Ensure database schema
    db_path = Path(args.db)
    ensure_schema(db_path)

    print(f"[collector] Starting data collection (interval: {args.interval}s)")
    print(f"[collector] Database: {db_path}")
    print(f"[collector] Press Ctrl+C to stop\n")

    while not shutdown_requested:
        try:
            snapshot = collect_snapshot()
            store_snapshot(snapshot)
            print_snapshot(snapshot)
            print()

        except Exception as e:
            print(f"[ERROR] Collection failed: {e}")

        # Wait for next interval
        time.sleep(args.interval)

    print("[collector] Shutdown complete")


if __name__ == "__main__":
    main()
