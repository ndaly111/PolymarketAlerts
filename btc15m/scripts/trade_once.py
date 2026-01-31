#!/usr/bin/env python3
"""Single execution trade script for GitHub Actions.

Runs once, makes a prediction, executes REAL trade on Kalshi if conditions met,
checks for settlements, and exits.

Usage:
    python trade_once.py [--min-prob 0.52] [--contracts 1]
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from lib.binance import fetch_klines, fetch_current_price
from lib.chainlink import get_chainlink_btc_price
from lib.kalshi_btc import (
    fetch_btc_15min_markets,
    calculate_market_implied_prob,
    get_execution_price,
    get_orderbook_best_prices,
    get_kalshi_client,
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
from lib.discord import send_trade_alert, send_settlement_alert, send_pending_status_alert
from lib.stats import update_all_stats

try:
    from kalshi_auth_client import KalshiAuthClient
except ImportError:
    KalshiAuthClient = None


def place_real_order(
    client: KalshiAuthClient,
    ticker: str,
    direction: str,  # "UP" or "DOWN"
    limit_price_cents: int,
    contracts: int = 1,
) -> dict:
    """Place a real order on Kalshi.

    For BTC markets:
    - UP prediction = buy YES (betting price goes above strike)
    - DOWN prediction = buy NO (betting price stays below strike)
    """
    # UP = buy YES, DOWN = buy NO
    side = "yes" if direction == "UP" else "no"

    print(f"[ORDER] Placing {contracts}x {side.upper()} @ {limit_price_cents}c on {ticker}")

    order = client.place_order(
        ticker=ticker,
        side=side,
        quantity=contracts,
        limit_price=limit_price_cents,
    )

    return order


def check_settlements(db_path: Path, current_price: float) -> int:
    """Check and settle any expired trades. Returns number settled."""
    pending = get_pending_trades(db_path)
    settled_count = 0

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
            settled_count += 1

            # Get updated cumulative stats
            summary = get_paper_trade_summary(db_path)
            total_record = f"{summary['wins']}W / {summary['losses']}L"

            print(f"[SETTLED] {trade['market_ticker']}: {outcome} (${pnl:+.2f})")
            print(f"[CUMULATIVE] Total P&L: ${summary['total_pnl']:+.2f} | Record: {total_record}")

            send_settlement_alert(
                market_ticker=trade['market_ticker'],
                side=trade['side'],
                outcome=outcome,
                pnl=pnl,
                strike_price=strike,
                settlement_price=current_price,
                cumulative_pnl=summary['total_pnl'],
                total_record=total_record,
            )

    return settled_count


def main():
    parser = argparse.ArgumentParser(description="Single trade execution - REAL TRADING")
    parser.add_argument("--min-prob", type=float, default=0.52,
                        help="Minimum probability to trade (default: 0.52)")
    parser.add_argument("--contracts", type=int, default=1,
                        help="Contracts per trade (default: 1)")
    parser.add_argument("--db", type=str, default=str(DEFAULT_DB_PATH))
    parser.add_argument("--paper", action="store_true",
                        help="Paper trade only (no real orders)")
    args = parser.parse_args()

    db_path = Path(args.db)
    ensure_schema(db_path)

    print(f"[trade] Starting single trade cycle")
    print(f"[trade] Mode: {'PAPER' if args.paper else 'REAL TRADING'}")
    print(f"[trade] Min probability: {args.min_prob*100:.0f}%")
    print(f"[trade] Contracts: {args.contracts}")

    # Initialize Kalshi client for real trading
    kalshi_client = None
    if not args.paper:
        kalshi_client = get_kalshi_client()
        if kalshi_client:
            try:
                balance = kalshi_client.get_balance()
                balance_usd = balance.get('balance', 0) / 100
                print(f"[trade] Kalshi balance: ${balance_usd:.2f}")
            except Exception as e:
                print(f"[trade] Warning: Could not get balance: {e}")
        else:
            print("[trade] WARNING: No Kalshi client - falling back to paper trading")
            args.paper = True

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
    settlement_price = chainlink_price or binance_price
    settled_count = check_settlements(db_path, settlement_price)
    if settled_count > 0:
        print(f"[trade] Settled {settled_count} trade(s)")

    # Send pending status if there are still pending trades
    still_pending = get_pending_trades(db_path)
    if still_pending:
        print(f"[trade] {len(still_pending)} trade(s) still pending")
        send_pending_status_alert(still_pending, settlement_price)

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

    # Try to get more accurate prices from orderbook
    orderbook_prices = get_orderbook_best_prices(market.ticker)
    if orderbook_prices:
        print(f"[trade] Got orderbook prices")
        probs.update({
            "up_ask": orderbook_prices["yes_ask"] or probs["up_ask"],
            "up_bid": orderbook_prices["yes_bid"] or probs["up_bid"],
            "down_ask": orderbook_prices["no_ask"] or probs["down_ask"],
            "down_bid": orderbook_prices["no_bid"] or probs["down_bid"],
        })

    print(f"[trade] Market: {market.ticker}")
    print(f"[trade] Strike: ${market.strike_price:,.0f}")
    print(f"[trade] Expires in: {int(tte//60)}m {int(tte%60)}s")
    print(f"[trade] UP: bid={probs['up_bid']*100:.1f}% / ask={probs['up_ask']*100:.1f}%")
    print(f"[trade] DOWN: bid={probs['down_bid']*100:.1f}% / ask={probs['down_ask']*100:.1f}%")

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

    # Execute trade at actual ASK price (what we'd pay)
    if prediction.direction == "UP":
        entry_price = probs["up_ask"]  # Pay the ask to buy YES on UP
    else:
        entry_price = probs["down_ask"]  # Pay the ask to buy YES on DOWN

    # Calculate edge vs entry price (not mid)
    edge = prediction.probability - entry_price

    # Skip if edge is negative after spread
    if edge < 0:
        print(f"[trade] Negative edge after spread ({edge*100:.1f}%), skipping")
        sys.exit(0)

    # Convert probability to cents for limit price
    limit_price_cents = int(entry_price * 100)

    # Place real order or paper trade
    order_id = None
    if not args.paper and kalshi_client:
        try:
            order = place_real_order(
                client=kalshi_client,
                ticker=market.ticker,
                direction=prediction.direction,
                limit_price_cents=limit_price_cents,
                contracts=args.contracts,
            )
            order_id = order.get("order_id")
            order_status = order.get("status", "unknown")
            print(f"[ORDER] Order placed! ID: {order_id}, Status: {order_status}")
        except Exception as e:
            print(f"[ORDER] ERROR placing order: {e}")
            print("[ORDER] Falling back to paper trade")
            args.paper = True

    # Record in database (for tracking both real and paper)
    trade_id = insert_paper_trade(
        market_ticker=market.ticker,
        side=prediction.direction,
        entry_price=entry_price,
        stake=args.contracts,
        model_prob=prediction.probability,
        market_prob=entry_price,
        edge=edge,
        expiry_time=market.expiry_time.isoformat(),
        strike_price=market.strike_price,
        db_path=db_path,
    )

    # Get current cumulative stats
    summary = get_paper_trade_summary(db_path)

    trade_type = "PAPER" if args.paper else "REAL"
    print(f"\n[{trade_type} TRADE] {prediction.direction} on {market.ticker}")
    print(f"  ML Confidence: {prediction.probability*100:.1f}%")
    print(f"  Market Price:  {entry_price*100:.1f}% ({limit_price_cents}c)")
    print(f"  Edge:          {edge*100:+.1f}%")
    print(f"  Contracts:     {args.contracts}")
    if order_id:
        print(f"  Order ID:      {order_id}")
    print(f"  DB Trade ID:   {trade_id}")
    print(f"  Cumulative P&L: ${summary['total_pnl']:+.2f}")

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
        cumulative_pnl=summary['total_pnl'],
        total_trades=summary['total_trades'],
        win_rate=summary['win_rate'],
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
