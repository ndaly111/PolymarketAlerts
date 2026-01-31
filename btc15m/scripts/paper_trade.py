#!/usr/bin/env python3
"""Paper trading simulator for BTC 15-min markets.

Simulates trading on Kalshi BTC markets using a simple strategy
or ML model predictions. Tracks P&L without real money.

Usage:
    python paper_trade.py [--stake 10] [--min-edge 0.05]
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
from lib.indicators import compute_rsi, compute_macd, compute_vwap
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


# Graceful shutdown
shutdown_requested = False


def signal_handler(signum, frame):
    global shutdown_requested
    print("\n[paper] Shutdown requested...")
    shutdown_requested = True


@dataclass
class Signal:
    """Trading signal from strategy."""
    side: str  # 'UP' or 'DOWN'
    model_prob: float
    market_prob: float
    edge: float
    confidence: str  # 'LOW', 'MEDIUM', 'HIGH'


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


# Track pending trades
pending_trades: list[PendingTrade] = []


def simple_strategy(
    rsi: Optional[float],
    macd_histogram: Optional[float],
    price_vs_vwap: Optional[float],
    market_up_prob: float,
) -> Optional[Signal]:
    """Simple rule-based strategy combining RSI, MACD, and VWAP.

    Returns a Signal if there's a trading opportunity, None otherwise.
    """
    if rsi is None or macd_histogram is None:
        return None

    # Score based on indicators (-1 to +1, positive = bullish)
    score = 0.0
    factors = 0

    # RSI signal
    if rsi < 30:
        score += 0.3  # Oversold = bullish
        factors += 1
    elif rsi > 70:
        score -= 0.3  # Overbought = bearish
        factors += 1
    elif rsi < 45:
        score += 0.1
        factors += 1
    elif rsi > 55:
        score -= 0.1
        factors += 1

    # MACD histogram signal
    if macd_histogram > 0:
        score += 0.2 if macd_histogram > 50 else 0.1
        factors += 1
    elif macd_histogram < 0:
        score -= 0.2 if macd_histogram < -50 else 0.1
        factors += 1

    # Price vs VWAP signal
    if price_vs_vwap is not None:
        if price_vs_vwap < -0.5:
            score += 0.2  # Below VWAP = potential bounce
            factors += 1
        elif price_vs_vwap > 0.5:
            score -= 0.2  # Above VWAP = potential pullback
            factors += 1

    if factors == 0:
        return None

    # Convert score to probability
    # Score range: -0.7 to +0.7, map to probability 0.3 to 0.7
    model_up_prob = 0.5 + (score / 1.4)  # Maps -0.7..0.7 to 0.0..1.0
    model_up_prob = max(0.35, min(0.65, model_up_prob))  # Clamp to realistic range

    # Determine side and calculate edge
    if model_up_prob > 0.5:
        side = "UP"
        model_prob = model_up_prob
        market_prob = market_up_prob
    else:
        side = "DOWN"
        model_prob = 1 - model_up_prob
        market_prob = 1 - market_up_prob

    edge = model_prob - market_prob

    # Determine confidence
    if abs(edge) >= 0.10:
        confidence = "HIGH"
    elif abs(edge) >= 0.05:
        confidence = "MEDIUM"
    else:
        confidence = "LOW"

    return Signal(
        side=side,
        model_prob=model_prob,
        market_prob=market_prob,
        edge=edge,
        confidence=confidence,
    )


def check_pending_settlements(current_price: float) -> None:
    """Check and settle any pending trades that have expired."""
    global pending_trades

    now = datetime.now(timezone.utc)
    still_pending = []

    for trade in pending_trades:
        if now >= trade.expiry_time:
            # Market has expired, settle the trade
            # Determine outcome based on price vs strike
            if current_price > trade.strike_price:
                actual_outcome = "UP"
            else:
                actual_outcome = "DOWN"

            if trade.side == actual_outcome:
                outcome = "WIN"
                # Kalshi pays out at $1 for a win
                pnl = trade.stake * (1 / trade.entry_price - 1)
            else:
                outcome = "LOSS"
                pnl = -trade.stake

            settle_paper_trade(trade.trade_id, outcome, pnl)

            print(f"  [SETTLED] {trade.market_ticker}: {outcome}")
            print(f"            Side: {trade.side}, Strike: ${trade.strike_price:,.0f}")
            print(f"            Settlement: ${current_price:,.2f}, P&L: ${pnl:+.2f}")
        else:
            still_pending.append(trade)

    pending_trades = still_pending


def execute_paper_trade(
    market: BTCMarket,
    signal: Signal,
    stake: float,
) -> Optional[PendingTrade]:
    """Execute a paper trade."""
    entry_price = signal.market_prob  # Buying at market probability

    trade_id = insert_paper_trade(
        market_ticker=market.ticker,
        side=signal.side,
        entry_price=entry_price,
        stake=stake,
        model_prob=signal.model_prob,
        market_prob=signal.market_prob,
        edge=signal.edge,
    )

    return PendingTrade(
        trade_id=trade_id,
        market_ticker=market.ticker,
        side=signal.side,
        entry_price=entry_price,
        stake=stake,
        expiry_time=market.expiry_time,
        strike_price=market.strike_price,
    )


def print_status(
    binance_price: float,
    chainlink_price: Optional[float],
    rsi: Optional[float],
    macd_hist: Optional[float],
    market: Optional[BTCMarket],
    signal: Optional[Signal],
    summary: dict,
) -> None:
    """Print current status to console."""
    print("=" * 60)
    print(f"  BTC Price: ${binance_price:,.2f}")
    if chainlink_price:
        diff = binance_price - chainlink_price
        print(f"  Chainlink: ${chainlink_price:,.2f} (diff: ${diff:+.2f})")

    print(f"  RSI(14):   {rsi:.1f}" if rsi else "  RSI(14):   N/A")
    print(f"  MACD Hist: {macd_hist:+.2f}" if macd_hist else "  MACD Hist: N/A")

    if market:
        probs = calculate_market_implied_prob(market)
        tte = (market.expiry_time - datetime.now(timezone.utc)).total_seconds()
        print(f"\n  Market:    {market.ticker}")
        print(f"  Strike:    ${market.strike_price:,.0f}")
        print(f"  Expires:   {int(tte//60)}m {int(tte%60)}s")
        print(f"  UP prob:   {probs['up_prob']*100:.1f}%")

    if signal:
        edge_pct = signal.edge * 100
        print(f"\n  Signal:    {signal.side} ({signal.confidence})")
        print(f"  Model:     {signal.model_prob*100:.1f}%")
        print(f"  Market:    {signal.market_prob*100:.1f}%")
        print(f"  Edge:      {edge_pct:+.1f}%")

    print(f"\n  Paper P&L: ${summary['total_pnl']:+.2f}")
    print(f"  Trades:    {summary['total_trades']} ({summary['wins']}W/{summary['losses']}L)")
    if summary['win_rate']:
        print(f"  Win Rate:  {summary['win_rate']*100:.1f}%")
    print(f"  Pending:   {len(pending_trades)}")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Paper trade BTC 15-min markets")
    parser.add_argument("--stake", type=float, default=10.0,
                        help="Stake per trade in dollars (default: 10)")
    parser.add_argument("--min-edge", type=float, default=0.05,
                        help="Minimum edge to trade (default: 0.05 = 5%%)")
    parser.add_argument("--interval", type=int, default=30,
                        help="Check interval in seconds (default: 30)")
    parser.add_argument("--db", type=str, default=str(DEFAULT_DB_PATH),
                        help="Database path")
    args = parser.parse_args()

    # Set up signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # Ensure database schema
    db_path = Path(args.db)
    ensure_schema(db_path)

    print(f"[paper] Starting paper trading")
    print(f"[paper] Stake: ${args.stake:.2f} | Min Edge: {args.min_edge*100:.1f}%")
    print(f"[paper] Database: {db_path}")
    print(f"[paper] Press Ctrl+C to stop\n")

    # Track which markets we've already traded
    traded_markets: set[str] = set()

    while not shutdown_requested:
        try:
            # Fetch current data
            binance_price = fetch_current_price()
            chainlink_price = get_chainlink_btc_price()

            # Check for settled trades
            check_pending_settlements(chainlink_price or binance_price)

            # Fetch candles and calculate indicators
            candles = fetch_klines(interval="1m", limit=60)
            closes = [c.close for c in candles]
            candle_dicts = [{"high": c.high, "low": c.low, "close": c.close, "volume": c.volume}
                           for c in candles]

            rsi = compute_rsi(closes, period=14)
            macd_result = compute_macd(closes, fast=12, slow=26, signal=9)
            macd_hist = macd_result.histogram if macd_result else None
            vwap = compute_vwap(candle_dicts)

            price_vs_vwap = None
            if vwap and binance_price:
                price_vs_vwap = (binance_price - vwap) / vwap * 100

            # Fetch markets
            markets = fetch_btc_15min_markets()
            market = markets[0] if markets else None

            signal_result = None
            if market and market.ticker not in traded_markets:
                probs = calculate_market_implied_prob(market)

                # Generate trading signal
                signal_result = simple_strategy(
                    rsi=rsi,
                    macd_histogram=macd_hist,
                    price_vs_vwap=price_vs_vwap,
                    market_up_prob=probs["up_prob"],
                )

                # Execute trade if edge is sufficient
                if signal_result and signal_result.edge >= args.min_edge:
                    pending = execute_paper_trade(market, signal_result, args.stake)
                    if pending:
                        pending_trades.append(pending)
                        traded_markets.add(market.ticker)
                        print(f"\n  [TRADE] {signal_result.side} on {market.ticker}")
                        print(f"          Edge: {signal_result.edge*100:+.1f}%")

            # Get summary and print status
            summary = get_paper_trade_summary(db_path)
            print_status(
                binance_price=binance_price,
                chainlink_price=chainlink_price,
                rsi=rsi,
                macd_hist=macd_hist,
                market=market,
                signal=signal_result,
                summary=summary,
            )

        except Exception as e:
            print(f"[ERROR] {e}")

        # Wait for next interval
        time.sleep(args.interval)

    # Final summary
    summary = get_paper_trade_summary(db_path)
    print("\n" + "=" * 60)
    print("  FINAL SUMMARY")
    print("=" * 60)
    print(f"  Total Trades: {summary['total_trades']}")
    print(f"  Wins:         {summary['wins']}")
    print(f"  Losses:       {summary['losses']}")
    print(f"  Win Rate:     {summary['win_rate']*100:.1f}%" if summary['win_rate'] else "  Win Rate:     N/A")
    print(f"  Total P&L:    ${summary['total_pnl']:+.2f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
