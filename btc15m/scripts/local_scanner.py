#!/usr/bin/env python3
"""
Local continuous BTC 15-min scanner with Kalshi WebSocket.

Runs continuously, monitoring for edge opportunities and placing trades.

Usage:
    python local_scanner.py [--min-prob 0.54] [--contracts 1]

Environment:
    KALSHI_API_KEY_ID - Your Kalshi API key ID
    KALSHI_PRIVATE_KEY - Your Kalshi private key
    BTC_15MIN_DISCORD - Discord webhook URL (optional)
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Dict, Any, Set

# Get absolute paths for imports
SCRIPT_DIR = Path(__file__).resolve().parent
BTC15M_DIR = SCRIPT_DIR.parent
REPO_DIR = BTC15M_DIR.parent

sys.path.insert(0, str(REPO_DIR))
sys.path.insert(0, str(BTC15M_DIR))

from lib.binance import fetch_klines, fetch_current_price
from lib.chainlink import get_chainlink_btc_price
from lib.kalshi_btc import (
    fetch_btc_15min_markets,
    calculate_market_implied_prob,
    get_kalshi_client,
    BTCMarket,
)
from lib.db import (
    ensure_schema,
    insert_paper_trade,
    get_paper_trade_summary,
    get_pending_trades,
    settle_paper_trade,
    DEFAULT_DB_PATH,
)
from lib.ml_predictor import MLPredictor, compute_all_features
from lib.discord import send_trade_alert, send_settlement_alert, send_edge_analysis_alert

try:
    from kalshi_auth_client import KalshiAuthClient
except ImportError:
    KalshiAuthClient = None


KALSHI_FEE_RATE = 0.07
SCAN_INTERVAL_SECONDS = 15  # How often to scan for opportunities


def place_order_with_retry(
    client: KalshiAuthClient,
    ticker: str,
    direction: str,
    limit_price_cents: int,
    contracts: int,
    max_retries: int = 5,
    timeout_per_attempt: int = 60,
) -> tuple[bool, Optional[str]]:
    """
    Place order with retry logic.

    Returns:
        (filled, order_id)
    """
    side = "yes" if direction == "UP" else "no"

    for attempt in range(1, max_retries + 1):
        print(f"\n[ORDER] Attempt {attempt}/{max_retries}: {contracts}x {side.upper()} @ {limit_price_cents}c on {ticker}")

        try:
            order = client.place_order(
                ticker=ticker,
                side=side,
                quantity=contracts,
                limit_price=limit_price_cents,
            )
            order_id = order.get("order_id")
            status = order.get("status", "unknown").lower()
            print(f"[ORDER] Placed! ID: {order_id}, Status: {status}")

            if status in ("filled", "executed"):
                print(f"[ORDER] Immediately filled!")
                return True, order_id

            # Poll for fill
            start = time.time()
            while time.time() - start < timeout_per_attempt:
                time.sleep(10)
                try:
                    resp = client.get_order(order_id)
                    current = resp.get("status", "").lower()
                    elapsed = int(time.time() - start)
                    print(f"[ORDER] Status: {current} ({elapsed}s)")

                    if current in ("filled", "executed"):
                        print(f"[ORDER] Filled!")
                        return True, order_id
                    elif current in ("cancelled", "canceled", "expired"):
                        break
                except Exception as e:
                    print(f"[ORDER] Check error: {e}")

            # Cancel if not filled
            try:
                print(f"[ORDER] Timeout - cancelling")
                client.cancel_order(order_id)
            except Exception as e:
                print(f"[ORDER] Cancel error: {e}")

        except Exception as e:
            print(f"[ORDER] Error: {e}")

    return False, None


def check_settlements(db_path: Path, current_price: float):
    """Check and settle any expired trades."""
    pending = get_pending_trades(db_path)

    for trade in pending:
        expiry = datetime.fromisoformat(trade['expiry_time'].replace('Z', '+00:00'))
        now = datetime.now(timezone.utc)

        if now >= expiry:
            strike = trade['strike_price']
            actual = "UP" if current_price > strike else "DOWN"

            if trade['side'] == actual:
                gross_pnl = (1.0 - trade['entry_price']) * trade['stake']
                fee = gross_pnl * KALSHI_FEE_RATE
                pnl = gross_pnl - fee
                outcome = "WIN"
            else:
                pnl = -trade['entry_price'] * trade['stake']
                outcome = "LOSS"

            settle_paper_trade(trade['id'], outcome, pnl, db_path)

            summary = get_paper_trade_summary(db_path)
            print(f"\n[SETTLE] {trade['market_ticker']}: {outcome} (${pnl:+.2f})")
            print(f"[CUMULATIVE] P&L: ${summary['total_pnl']:+.2f} | {summary['wins']}W/{summary['losses']}L")

            send_settlement_alert(
                market_ticker=trade['market_ticker'],
                side=trade['side'],
                outcome=outcome,
                pnl=pnl,
                strike_price=strike,
                settlement_price=current_price,
                cumulative_pnl=summary['total_pnl'],
                total_record=f"{summary['wins']}W / {summary['losses']}L",
            )


def run_scan_cycle(
    predictor: MLPredictor,
    kalshi_client: KalshiAuthClient,
    db_path: Path,
    min_prob: float,
    contracts: int,
    traded_tickers: Set[str],
):
    """Run a single scan cycle."""
    print(f"\n{'='*60}")
    print(f"[SCAN] {datetime.now().strftime('%H:%M:%S')}")

    # Fetch current prices
    try:
        binance_price = fetch_current_price()
        chainlink_price = get_chainlink_btc_price()
        print(f"[SCAN] BTC: ${binance_price:,.2f}")
    except Exception as e:
        print(f"[SCAN] Price error: {e}")
        return

    # Check settlements
    settlement_price = chainlink_price or binance_price
    check_settlements(db_path, settlement_price)

    # Compute features
    try:
        candles = fetch_klines(interval="1m", limit=100)
        candle_dicts = [
            {"open": c.open, "high": c.high, "low": c.low, "close": c.close, "volume": c.volume}
            for c in candles
        ]
        current_hour = datetime.now(timezone.utc).hour
        features = compute_all_features(candle_dicts, current_hour)
    except Exception as e:
        print(f"[SCAN] Feature error: {e}")
        return

    # Get prediction
    prediction = predictor.predict(features)
    if not prediction:
        print("[SCAN] No prediction")
        return

    print(f"[SCAN] Prediction: {prediction.direction} ({prediction.probability*100:.1f}%)")

    # Check threshold
    if prediction.probability < min_prob:
        print(f"[SCAN] Below threshold ({min_prob*100:.0f}%)")
        return

    # Fetch markets
    try:
        markets = fetch_btc_15min_markets(current_btc_price=binance_price)
    except Exception as e:
        print(f"[SCAN] Market error: {e}")
        return

    if not markets:
        print("[SCAN] No markets near current price")
        return

    # Find best tradeable market
    for market in markets:
        if market.ticker in traded_tickers:
            continue

        probs = calculate_market_implied_prob(market)
        tte = (market.expiry_time - datetime.now(timezone.utc)).total_seconds()

        if tte < 120:  # Skip if less than 2 min to expiry
            continue

        # Calculate prices
        if prediction.direction == "UP":
            current_ask = probs["up_ask"]
        else:
            current_ask = probs["down_ask"]

        # Max +EV price
        prob = prediction.probability
        max_ev_price = (prob * (1 - KALSHI_FEE_RATE)) / (1 - prob * KALSHI_FEE_RATE)
        entry_price = min(max_ev_price, current_ask)
        edge = prob - entry_price

        print(f"[SCAN] Market: {market.ticker}")
        print(f"[SCAN] Strike: ${market.strike_price:,.0f}, Expires: {int(tte/60)}m")
        print(f"[SCAN] ML: {prob*100:.1f}%, Max: {max_ev_price*100:.1f}¢, Ask: {current_ask*100:.1f}¢")
        print(f"[SCAN] Edge: {edge*100:+.1f}%")

        if edge <= 0:
            print("[SCAN] No positive edge, skipping")
            continue

        # Place order
        limit_cents = int(entry_price * 100)
        filled, order_id = place_order_with_retry(
            client=kalshi_client,
            ticker=market.ticker,
            direction=prediction.direction,
            limit_price_cents=limit_cents,
            contracts=contracts,
        )

        if filled:
            traded_tickers.add(market.ticker)

            # Record trade
            trade_id = insert_paper_trade(
                market_ticker=market.ticker,
                side=prediction.direction,
                entry_price=entry_price,
                stake=contracts,
                model_prob=prediction.probability,
                market_prob=entry_price,
                edge=edge,
                expiry_time=market.expiry_time.isoformat(),
                strike_price=market.strike_price,
                db_path=db_path,
            )

            summary = get_paper_trade_summary(db_path)
            print(f"\n[TRADE] {prediction.direction} filled on {market.ticker}")
            print(f"[TRADE] Cumulative P&L: ${summary['total_pnl']:+.2f}")

            send_trade_alert(
                side=prediction.direction,
                market_ticker=market.ticker,
                strike_price=market.strike_price,
                model_prob=prediction.probability,
                market_prob=entry_price,
                edge=edge,
                btc_price=binance_price,
                time_to_expiry_sec=int(tte),
                cumulative_pnl=summary['total_pnl'],
                total_trades=summary['total_trades'],
                win_rate=summary['win_rate'],
            )

        break  # Only try one market per scan


def main():
    parser = argparse.ArgumentParser(description="Local BTC 15-min continuous scanner")
    parser.add_argument("--min-prob", type=float, default=0.54,
                        help="Minimum probability to trade (default: 0.54)")
    parser.add_argument("--contracts", type=int, default=1,
                        help="Contracts per trade (default: 1)")
    parser.add_argument("--db", type=str, default=str(DEFAULT_DB_PATH))
    parser.add_argument("--interval", type=int, default=SCAN_INTERVAL_SECONDS,
                        help="Scan interval in seconds (default: 15)")
    args = parser.parse_args()

    db_path = Path(args.db)
    ensure_schema(db_path)

    print("=" * 60)
    print("BTC 15-MIN LOCAL SCANNER")
    print("=" * 60)
    print(f"Min probability: {args.min_prob*100:.0f}%")
    print(f"Contracts: {args.contracts}")
    print(f"Scan interval: {args.interval}s")
    print("=" * 60)

    # Initialize Kalshi client
    kalshi_client = get_kalshi_client()
    if not kalshi_client:
        print("[ERROR] Failed to initialize Kalshi client")
        print("Make sure KALSHI_API_KEY_ID and KALSHI_PRIVATE_KEY are set")
        sys.exit(1)

    try:
        balance = kalshi_client.get_balance()
        balance_usd = balance.get('balance', 0) / 100
        print(f"Kalshi balance: ${balance_usd:.2f}")
    except Exception as e:
        print(f"Balance check error: {e}")

    # Load ML model
    try:
        predictor = MLPredictor()
        print("ML model loaded")
    except Exception as e:
        print(f"[ERROR] Failed to load model: {e}")
        sys.exit(1)

    # Track already traded tickers this session
    traded_tickers: Set[str] = set()

    print("\nStarting continuous scan (Ctrl+C to stop)...")

    try:
        while True:
            try:
                run_scan_cycle(
                    predictor=predictor,
                    kalshi_client=kalshi_client,
                    db_path=db_path,
                    min_prob=args.min_prob,
                    contracts=args.contracts,
                    traded_tickers=traded_tickers,
                )
            except KeyboardInterrupt:
                raise
            except Exception as e:
                print(f"[ERROR] Scan cycle error: {e}")

            time.sleep(args.interval)

    except KeyboardInterrupt:
        print("\n\nScanner stopped by user")

        # Print final stats
        summary = get_paper_trade_summary(db_path)
        print(f"\n{'='*60}")
        print("SESSION SUMMARY")
        print(f"{'='*60}")
        print(f"Total trades: {summary['total_trades']}")
        print(f"Wins: {summary['wins']}")
        print(f"Losses: {summary['losses']}")
        print(f"Win rate: {summary['win_rate']*100:.1f}%" if summary['win_rate'] else "Win rate: N/A")
        print(f"Total P&L: ${summary['total_pnl']:+.2f}")


if __name__ == "__main__":
    main()
