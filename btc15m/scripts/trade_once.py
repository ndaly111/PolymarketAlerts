#!/usr/bin/env python3
"""Single execution trade script for GitHub Actions.

Runs once, makes a prediction, executes trade if conditions met,
checks for settlements, and exits.

Usage:
    python trade_once.py [--min-prob 0.52]
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from lib.binance import fetch_klines, fetch_current_price
from lib.chainlink import get_chainlink_btc_price
from lib.kalshi_btc import (
    fetch_btc_15min_markets,
    calculate_market_implied_prob,
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
from lib.discord import send_trade_alert, send_settlement_alert
from lib.stats import update_all_stats


def check_settlements(db_path: Path, current_price: float) -> None:
    """Check and settle any expired trades."""
    pending = get_pending_trades(db_path)

    for trade in pending:
        # Parse expiry time
        expiry = datetime.fromisoformat(trade['expiry_time'].replace('Z', '+00:00'))
        now = datetime.now(timezone.utc)

        if now >= expiry:
            # Determine outcome
            strike = trade['strike_price']
            if current_price > strike:
                actual = "UP"
            else:
                actual = "DOWN"

            if trade['side'] == actual:
                outcome = "WIN"
                pnl = (1.0 - trade['entry_price']) * trade['stake']
            else:
                outcome = "LOSS"
                pnl = -trade['entry_price'] * trade['stake']

            settle_paper_trade(trade['id'], outcome, pnl, db_path)

            print(f"[SETTLED] {trade['market_ticker']}: {outcome} (${pnl:+.2f})")

            send_settlement_alert(
                market_ticker=trade['market_ticker'],
                side=trade['side'],
                outcome=outcome,
                pnl=pnl,
                strike_price=strike,
                settlement_price=current_price,
            )


def main():
    parser = argparse.ArgumentParser(description="Single trade execution")
    parser.add_argument("--min-prob", type=float, default=0.52,
                        help="Minimum probability to trade (default: 0.52)")
    parser.add_argument("--stake", type=float, default=1.0,
                        help="Contracts per trade (default: 1)")
    parser.add_argument("--db", type=str, default=str(DEFAULT_DB_PATH))
    args = parser.parse_args()

    db_path = Path(args.db)
    ensure_schema(db_path)

    print(f"[trade] Starting single trade cycle")
    print(f"[trade] Min probability: {args.min_prob*100:.0f}%")
    print(f"[trade] Stake: {args.stake} contracts")

    # Load ML model
    try:
        predictor = MLPredictor()
        print("[trade] ML model loaded")
    except Exception as e:
        print(f"[trade] Error loading model: {e}")
        sys.exit(1)

    # Fetch current prices
    try:
        binance_price = fetch_current_price()
        chainlink_price = get_chainlink_btc_price()
        print(f"[trade] BTC: ${binance_price:,.2f} (Chainlink: ${chainlink_price:,.2f})")
    except Exception as e:
        print(f"[trade] Error fetching price: {e}")
        sys.exit(1)

    # Check for settlements
    check_settlements(db_path, chainlink_price or binance_price)

    # Fetch candles and compute features
    try:
        candles = fetch_klines(interval="1m", limit=100)
        candle_dicts = [
            {"open": c.open, "high": c.high, "low": c.low, "close": c.close, "volume": c.volume}
            for c in candles
        ]
        current_hour = datetime.now(timezone.utc).hour
        features = compute_all_features(candle_dicts, current_hour)
    except Exception as e:
        print(f"[trade] Error computing features: {e}")
        sys.exit(1)

    # Get ML prediction
    prediction = predictor.predict(features)
    if not prediction:
        print("[trade] No prediction available")
        sys.exit(0)

    print(f"[trade] Prediction: {prediction.direction} ({prediction.probability*100:.1f}%)")

    # Fetch Kalshi markets
    try:
        markets = fetch_btc_15min_markets()
    except Exception as e:
        print(f"[trade] Error fetching markets: {e}")
        markets = []

    if not markets:
        print("[trade] No active markets found")
        # Still show prediction even without market
        print(f"\n--- Current Signal ---")
        print(f"Direction: {prediction.direction}")
        print(f"Confidence: {prediction.probability*100:.1f}%")
        print(f"Would trade: {'YES' if prediction.probability >= args.min_prob else 'NO'}")
        sys.exit(0)

    market = markets[0]
    probs = calculate_market_implied_prob(market)
    tte = (market.expiry_time - datetime.now(timezone.utc)).total_seconds()

    print(f"[trade] Market: {market.ticker}")
    print(f"[trade] Strike: ${market.strike_price:,.0f}")
    print(f"[trade] Expires in: {int(tte//60)}m {int(tte%60)}s")
    print(f"[trade] Market UP prob: {probs['up_prob']*100:.1f}%")

    # Check if we already traded this market
    pending = get_pending_trades(db_path)
    already_traded = any(t['market_ticker'] == market.ticker for t in pending)

    if already_traded:
        print(f"[trade] Already have position in {market.ticker}")
        sys.exit(0)

    # Check confidence threshold
    if prediction.probability < args.min_prob:
        print(f"[trade] Confidence {prediction.probability*100:.1f}% < threshold {args.min_prob*100:.0f}%")
        print("[trade] No trade")
        sys.exit(0)

    # Execute trade
    if prediction.direction == "UP":
        entry_price = probs["up_prob"]
    else:
        entry_price = probs["down_prob"]

    edge = prediction.probability - entry_price

    trade_id = insert_paper_trade(
        market_ticker=market.ticker,
        side=prediction.direction,
        entry_price=entry_price,
        stake=args.stake,
        model_prob=prediction.probability,
        market_prob=entry_price,
        edge=edge,
        expiry_time=market.expiry_time.isoformat(),
        strike_price=market.strike_price,
        db_path=db_path,
    )

    print(f"\n[TRADE] {prediction.direction} on {market.ticker}")
    print(f"  ML Confidence: {prediction.probability*100:.1f}%")
    print(f"  Market Price:  {entry_price*100:.1f}%")
    print(f"  Edge:          {edge*100:+.1f}%")
    print(f"  Trade ID:      {trade_id}")

    # Send Discord alert
    send_trade_alert(
        side=prediction.direction,
        market_ticker=market.ticker,
        strike_price=market.strike_price,
        model_prob=prediction.probability,
        market_prob=entry_price,
        edge=edge,
        btc_price=binance_price,
        time_to_expiry_sec=int(tte),
    )

    # Update and save all stats (JSON, dashboard, Discord)
    print("\n[trade] Updating stats...")
    stats = update_all_stats(db_path, send_discord=False)  # Don't spam Discord every run

    print(f"\n--- Session Stats ---")
    print(f"Total Trades: {stats['all_time']['total_trades']}")
    print(f"Win Rate: {stats['all_time']['win_rate']*100:.1f}%" if stats['all_time']['win_rate'] else "Win Rate: N/A")
    print(f"Total P&L: ${stats['all_time']['total_pnl']:+.2f}")
    print(f"Today's P&L: ${stats['today']['pnl']:+.2f}")


if __name__ == "__main__":
    main()
