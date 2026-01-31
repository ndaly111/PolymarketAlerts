#!/usr/bin/env python3
"""Real-time data collection using WebSocket streams.

Collects data every 1-5 seconds for better ML feature granularity.
Uses WebSocket for sub-second price updates.

Usage:
    python collect_realtime.py [--interval 5] [--batch-size 10]
"""

from __future__ import annotations

import argparse
import signal
import sys
import time
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock
from typing import Optional

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from lib.binance import fetch_klines, fetch_current_price, Candle
from lib.binance_ws import start_price_stream, stop_price_stream, Trade, get_realtime_price
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

# Recent trades buffer for micro-features
recent_trades: deque[Trade] = deque(maxlen=1000)
trades_lock = Lock()

# Stats
stats = {
    "snapshots": 0,
    "trades_received": 0,
    "errors": 0,
}


def signal_handler(signum, frame):
    global shutdown_requested
    print("\n[realtime] Shutdown requested...")
    shutdown_requested = True


def on_trade(trade: Trade) -> None:
    """Called for each trade from WebSocket."""
    with trades_lock:
        recent_trades.append(trade)
    stats["trades_received"] += 1


def get_micro_features() -> dict:
    """Calculate micro-features from recent trades.

    These capture very short-term momentum that
    traditional indicators miss.
    """
    with trades_lock:
        trades = list(recent_trades)

    if len(trades) < 10:
        return {}

    # Last N seconds of trades
    now_ms = int(time.time() * 1000)

    # Trades in last 5 seconds
    recent_5s = [t for t in trades if now_ms - t.timestamp < 5000]
    # Trades in last 30 seconds
    recent_30s = [t for t in trades if now_ms - t.timestamp < 30000]
    # Trades in last 60 seconds
    recent_60s = [t for t in trades if now_ms - t.timestamp < 60000]

    features = {}

    # Buy/Sell pressure (is_buyer_maker=True means seller initiated)
    if recent_5s:
        buys = sum(1 for t in recent_5s if not t.is_buyer_maker)
        sells = sum(1 for t in recent_5s if t.is_buyer_maker)
        features["buy_sell_ratio_5s"] = buys / max(sells, 1)
        features["trade_count_5s"] = len(recent_5s)

    if recent_30s:
        buys = sum(1 for t in recent_30s if not t.is_buyer_maker)
        sells = sum(1 for t in recent_30s if t.is_buyer_maker)
        features["buy_sell_ratio_30s"] = buys / max(sells, 1)
        features["trade_count_30s"] = len(recent_30s)

        # Price momentum in last 30s
        if len(recent_30s) >= 2:
            first_price = recent_30s[0].price
            last_price = recent_30s[-1].price
            features["price_change_30s"] = (last_price - first_price) / first_price * 100

    if recent_60s:
        # Volume in last 60s
        features["volume_60s"] = sum(t.quantity for t in recent_60s)

        # Price range in last 60s
        prices = [t.price for t in recent_60s]
        features["price_range_60s"] = max(prices) - min(prices)

    return features


def collect_snapshot(use_ws_price: bool = True) -> dict:
    """Collect a single data snapshot."""
    timestamp = datetime.now(timezone.utc).isoformat()
    snapshot = {"timestamp": timestamp, "errors": []}

    # 1. Get price (WebSocket if available, else REST)
    try:
        if use_ws_price:
            ws_price = get_realtime_price()
            if ws_price:
                snapshot["binance_price"] = ws_price
            else:
                snapshot["binance_price"] = fetch_current_price()
        else:
            snapshot["binance_price"] = fetch_current_price()
    except Exception as e:
        snapshot["errors"].append(f"binance_price: {e}")
        snapshot["binance_price"] = None

    # 2. Chainlink (less frequent, cache-friendly)
    try:
        chainlink_price = get_chainlink_btc_price()
        snapshot["chainlink_price"] = chainlink_price
    except Exception as e:
        snapshot["errors"].append(f"chainlink_price: {e}")
        snapshot["chainlink_price"] = None

    # 3. Fetch candles for indicators (can cache for a few seconds)
    try:
        candles_1m = fetch_klines(interval="1m", limit=60)
        closes = [c.close for c in candles_1m]
        candle_dicts = [
            {"high": c.high, "low": c.low, "close": c.close, "volume": c.volume}
            for c in candles_1m
        ]

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

    # 4. Micro-features from WebSocket trades
    snapshot["micro"] = get_micro_features()

    # 5. Kalshi markets (every few snapshots to reduce API calls)
    snapshot["markets"] = []

    return snapshot


def fetch_markets_if_needed(last_fetch_time: float, interval: float = 30.0) -> list:
    """Fetch Kalshi markets, but not too frequently."""
    now = time.time()
    if now - last_fetch_time < interval:
        return []

    try:
        markets = fetch_btc_15min_markets()
        result = []
        for m in markets[:3]:
            probs = calculate_market_implied_prob(m)
            time_to_expiry = (m.expiry_time - datetime.now(timezone.utc)).total_seconds()
            result.append({
                "ticker": m.ticker,
                "expiry_time": m.expiry_time.isoformat(),
                "strike_price": m.strike_price,
                "up_yes_price": m.yes_bid,
                "down_yes_price": m.no_bid,
                "up_prob": probs["up_prob"],
                "down_prob": probs["down_prob"],
                "time_to_expiry_sec": int(time_to_expiry),
            })
        return result
    except Exception as e:
        print(f"[realtime] Market fetch error: {e}")
        return []


def store_snapshot(snapshot: dict, db_path: Path) -> None:
    """Store snapshot in database."""
    ts = snapshot["timestamp"]

    if snapshot.get("binance_price"):
        insert_price_snapshot(
            binance_price=snapshot["binance_price"],
            chainlink_price=snapshot.get("chainlink_price"),
            timestamp=ts,
            db_path=db_path,
        )

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
        db_path=db_path,
    )

    for m in snapshot.get("markets", []):
        insert_market_snapshot(
            market_ticker=m["ticker"],
            expiry_time=m.get("expiry_time"),
            strike_price=m.get("strike_price"),
            up_yes_price=m.get("up_yes_price"),
            down_yes_price=m.get("down_yes_price"),
            time_to_expiry_sec=m.get("time_to_expiry_sec"),
            timestamp=ts,
            db_path=db_path,
        )

    stats["snapshots"] += 1


def print_status(snapshot: dict) -> None:
    """Print compact status line."""
    ts = snapshot["timestamp"][11:19]  # Just time
    price = snapshot.get("binance_price")
    rsi = snapshot.get("rsi")
    micro = snapshot.get("micro", {})

    price_str = f"${price:,.2f}" if price else "N/A"
    rsi_str = f"{rsi:.0f}" if rsi else "--"

    buy_sell = micro.get("buy_sell_ratio_5s", 1.0)
    trades_5s = micro.get("trade_count_5s", 0)

    # Direction indicator
    if buy_sell > 1.2:
        direction = "↑"
    elif buy_sell < 0.8:
        direction = "↓"
    else:
        direction = "→"

    print(f"[{ts}] {price_str} | RSI:{rsi_str} | {direction} B/S:{buy_sell:.2f} ({trades_5s} trades) | Total: {stats['snapshots']}")


def main():
    parser = argparse.ArgumentParser(description="Real-time BTC data collection")
    parser.add_argument("--interval", type=int, default=5,
                        help="Snapshot interval in seconds (default: 5)")
    parser.add_argument("--db", type=str, default=str(DEFAULT_DB_PATH),
                        help="Database path")
    parser.add_argument("--no-websocket", action="store_true",
                        help="Disable WebSocket, use REST only")
    args = parser.parse_args()

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    db_path = Path(args.db)
    ensure_schema(db_path)

    print(f"[realtime] Starting real-time data collection")
    print(f"[realtime] Interval: {args.interval}s | DB: {db_path}")
    print(f"[realtime] WebSocket: {'disabled' if args.no_websocket else 'enabled'}")
    print(f"[realtime] Press Ctrl+C to stop\n")

    # Start WebSocket stream
    if not args.no_websocket:
        try:
            start_price_stream(on_trade=on_trade)
            print("[realtime] WebSocket stream started")
            time.sleep(1)  # Wait for connection
        except ImportError:
            print("[realtime] WebSocket not available, using REST")
            args.no_websocket = True

    last_market_fetch = 0

    while not shutdown_requested:
        try:
            snapshot = collect_snapshot(use_ws_price=not args.no_websocket)

            # Fetch markets every 30 seconds
            markets = fetch_markets_if_needed(last_market_fetch, interval=30.0)
            if markets:
                snapshot["markets"] = markets
                last_market_fetch = time.time()

            store_snapshot(snapshot, db_path)
            print_status(snapshot)

        except Exception as e:
            print(f"[ERROR] {e}")
            stats["errors"] += 1

        time.sleep(args.interval)

    # Cleanup
    if not args.no_websocket:
        stop_price_stream()

    print(f"\n[realtime] Final stats:")
    print(f"  Snapshots: {stats['snapshots']}")
    print(f"  Trades received: {stats['trades_received']}")
    print(f"  Errors: {stats['errors']}")


if __name__ == "__main__":
    main()
