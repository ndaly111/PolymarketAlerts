#!/usr/bin/env python3
"""Paper trading simulator for BTC 15-min markets.

Uses ML model to predict direction and trades 1 contract per market
when confidence exceeds threshold.

Usage:
    python paper_trade.py [--min-prob 0.55] [--stake 1]
"""

from __future__ import annotations

import argparse
import signal
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from lib.binance import fetch_klines, fetch_current_price
from lib.chainlink import get_chainlink_btc_price
from lib.kalshi_btc import (
    fetch_btc_15min_markets,
    calculate_market_implied_prob,
    BTCMarket,
)
from lib.db import (
    ensure_schema,
    insert_paper_trade,
    settle_paper_trade,
    get_paper_trade_summary,
    DEFAULT_DB_PATH,
)
from lib.ml_predictor import MLPredictor, compute_all_features, Prediction
from lib.discord import send_trade_alert, send_settlement_alert, send_summary_alert


# Graceful shutdown
shutdown_requested = False


def signal_handler(signum, frame):
    global shutdown_requested
    print("\n[paper] Shutdown requested...")
    shutdown_requested = True


@dataclass
class PendingTrade:
    """A trade waiting for settlement."""
    trade_id: int
    market_ticker: str
    side: str
    entry_price: float
    stake: float
    expiry_time: datetime
    strike_price: float
    model_prob: float


# Track pending trades
pending_trades: list[PendingTrade] = []


def check_pending_settlements(current_price: float, send_discord: bool = True) -> None:
    """Check and settle any pending trades that have expired."""
    global pending_trades

    now = datetime.now(timezone.utc)
    still_pending = []

    for trade in pending_trades:
        if now >= trade.expiry_time:
            # Market has expired, settle the trade
            if current_price > trade.strike_price:
                actual_outcome = "UP"
            else:
                actual_outcome = "DOWN"

            if trade.side == actual_outcome:
                outcome = "WIN"
                # Kalshi pays $1 for correct, we paid entry_price
                pnl = (1.0 - trade.entry_price) * trade.stake
            else:
                outcome = "LOSS"
                pnl = -trade.entry_price * trade.stake

            settle_paper_trade(trade.trade_id, outcome, pnl)

            emoji = "✅" if outcome == "WIN" else "❌"
            print(f"\n  {emoji} SETTLED: {trade.market_ticker}")
            print(f"     Side: {trade.side} @ {trade.model_prob*100:.1f}% confidence")
            print(f"     Strike: ${trade.strike_price:,.0f} | Settlement: ${current_price:,.2f}")
            print(f"     P&L: ${pnl:+.2f}")

            # Send Discord notification
            if send_discord:
                send_settlement_alert(
                    market_ticker=trade.market_ticker,
                    side=trade.side,
                    outcome=outcome,
                    pnl=pnl,
                    strike_price=trade.strike_price,
                    settlement_price=current_price,
                )
        else:
            still_pending.append(trade)

    pending_trades = still_pending


def execute_paper_trade(
    market: BTCMarket,
    prediction: Prediction,
    market_probs: dict,
    stake: float = 1.0,
) -> Optional[PendingTrade]:
    """Execute a paper trade based on ML prediction."""
    # Entry price is what we pay for the contract
    if prediction.direction == "UP":
        entry_price = market_probs["up_prob"]  # Buy YES on UP
    else:
        entry_price = market_probs["down_prob"]  # Buy YES on DOWN

    # Calculate edge
    edge = prediction.probability - entry_price

    trade_id = insert_paper_trade(
        market_ticker=market.ticker,
        side=prediction.direction,
        entry_price=entry_price,
        stake=stake,
        model_prob=prediction.probability,
        market_prob=entry_price,
        edge=edge,
    )

    return PendingTrade(
        trade_id=trade_id,
        market_ticker=market.ticker,
        side=prediction.direction,
        entry_price=entry_price,
        stake=stake,
        expiry_time=market.expiry_time,
        strike_price=market.strike_price,
        model_prob=prediction.probability,
    )


def print_status(
    binance_price: float,
    chainlink_price: Optional[float],
    market: Optional[BTCMarket],
    prediction: Optional[Prediction],
    summary: dict,
    min_prob: float,
) -> None:
    """Print current status to console."""
    now = datetime.now(timezone.utc)

    print("\n" + "=" * 65)
    print(f"  {now.strftime('%Y-%m-%d %H:%M:%S')} UTC")
    print("=" * 65)
    print(f"  BTC Price: ${binance_price:,.2f}")
    if chainlink_price:
        diff = binance_price - chainlink_price
        print(f"  Chainlink: ${chainlink_price:,.2f} (diff: ${diff:+.2f})")

    if market:
        probs = calculate_market_implied_prob(market)
        tte = (market.expiry_time - now).total_seconds()
        print(f"\n  Market:     {market.ticker}")
        print(f"  Strike:     ${market.strike_price:,.0f}")
        print(f"  Expires:    {int(tte//60)}m {int(tte%60)}s")
        print(f"  Market UP:  {probs['up_prob']*100:.1f}%")

    if prediction:
        meets_threshold = prediction.probability >= min_prob
        status = "✓ TRADING" if meets_threshold else "✗ SKIP (below threshold)"
        print(f"\n  ML Prediction:")
        print(f"    Direction:  {prediction.direction}")
        print(f"    Confidence: {prediction.probability*100:.1f}% ({prediction.confidence})")
        print(f"    Threshold:  {min_prob*100:.1f}%")
        print(f"    Status:     {status}")
    else:
        print(f"\n  ML Prediction: N/A (insufficient data)")

    print(f"\n  --- Paper Trading Stats ---")
    print(f"  Total P&L:  ${summary['total_pnl']:+.2f}")
    print(f"  Trades:     {summary['total_trades']} ({summary['wins']}W / {summary['losses']}L)")
    if summary['win_rate']:
        print(f"  Win Rate:   {summary['win_rate']*100:.1f}%")
    print(f"  Pending:    {len(pending_trades)}")
    print("=" * 65)


def main():
    parser = argparse.ArgumentParser(description="Paper trade BTC 15-min markets with ML")
    parser.add_argument("--min-prob", type=float, default=0.55,
                        help="Minimum ML probability to trade (default: 0.55 = 55%%)")
    parser.add_argument("--stake", type=float, default=1.0,
                        help="Number of contracts per trade (default: 1)")
    parser.add_argument("--interval", type=int, default=60,
                        help="Check interval in seconds (default: 60)")
    parser.add_argument("--db", type=str, default=str(DEFAULT_DB_PATH),
                        help="Database path")
    args = parser.parse_args()

    # Set up signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # Ensure database schema
    db_path = Path(args.db)
    ensure_schema(db_path)

    # Load ML model
    print("[paper] Loading ML model...")
    try:
        predictor = MLPredictor()
        print(f"[paper] Model loaded successfully")
    except Exception as e:
        print(f"[paper] Error loading model: {e}")
        print("[paper] Run train_model.py first to create a model")
        sys.exit(1)

    print(f"\n[paper] Starting paper trading")
    print(f"[paper] Contracts per trade: {args.stake}")
    print(f"[paper] Min probability: {args.min_prob*100:.0f}%")
    print(f"[paper] Database: {db_path}")
    print(f"[paper] Press Ctrl+C to stop")

    # Track which markets we've already traded
    traded_markets: set[str] = set()

    while not shutdown_requested:
        try:
            # Fetch current data
            binance_price = fetch_current_price()
            chainlink_price = get_chainlink_btc_price()

            # Check for settled trades
            check_pending_settlements(chainlink_price or binance_price)

            # Fetch candles for ML features
            candles = fetch_klines(interval="1m", limit=100)
            candle_dicts = [
                {
                    "open": c.open,
                    "high": c.high,
                    "low": c.low,
                    "close": c.close,
                    "volume": c.volume,
                }
                for c in candles
            ]

            # Compute features and get ML prediction
            current_hour = datetime.now(timezone.utc).hour
            features = compute_all_features(candle_dicts, current_hour)
            prediction = predictor.predict(features)

            # Fetch markets
            markets = fetch_btc_15min_markets()
            market = markets[0] if markets else None

            # Execute trade if conditions met
            if market and market.ticker not in traded_markets:
                probs = calculate_market_implied_prob(market)

                if prediction and prediction.probability >= args.min_prob:
                    # Execute trade
                    pending = execute_paper_trade(
                        market=market,
                        prediction=prediction,
                        market_probs=probs,
                        stake=args.stake,
                    )
                    if pending:
                        pending_trades.append(pending)
                        traded_markets.add(market.ticker)

                        edge = prediction.probability - pending.entry_price
                        tte = int((market.expiry_time - datetime.now(timezone.utc)).total_seconds())

                        print(f"\n  🎯 NEW TRADE: {prediction.direction} on {market.ticker}")
                        print(f"     ML Confidence: {prediction.probability*100:.1f}%")
                        print(f"     Market Price:  {pending.entry_price*100:.1f}%")
                        print(f"     Edge:          {edge*100:+.1f}%")

                        # Send Discord notification
                        send_trade_alert(
                            side=prediction.direction,
                            market_ticker=market.ticker,
                            strike_price=market.strike_price,
                            model_prob=prediction.probability,
                            market_prob=pending.entry_price,
                            edge=edge,
                            btc_price=binance_price,
                            time_to_expiry_sec=tte,
                        )

            # Get summary and print status
            summary = get_paper_trade_summary(db_path)
            print_status(
                binance_price=binance_price,
                chainlink_price=chainlink_price,
                market=market,
                prediction=prediction,
                summary=summary,
                min_prob=args.min_prob,
            )

        except Exception as e:
            print(f"[ERROR] {e}")
            import traceback
            traceback.print_exc()

        # Wait for next interval
        time.sleep(args.interval)

    # Final summary
    summary = get_paper_trade_summary(db_path)
    print("\n" + "=" * 65)
    print("  FINAL SUMMARY")
    print("=" * 65)
    print(f"  Total Trades: {summary['total_trades']}")
    print(f"  Wins:         {summary['wins']}")
    print(f"  Losses:       {summary['losses']}")
    print(f"  Win Rate:     {summary['win_rate']*100:.1f}%" if summary['win_rate'] else "  Win Rate:     N/A")
    print(f"  Total P&L:    ${summary['total_pnl']:+.2f}")
    print("=" * 65)

    # Send Discord summary
    if summary['total_trades'] > 0:
        send_summary_alert(
            total_trades=summary['total_trades'],
            wins=summary['wins'],
            losses=summary['losses'],
            total_pnl=summary['total_pnl'],
            win_rate=summary['win_rate'],
        )


if __name__ == "__main__":
    main()
