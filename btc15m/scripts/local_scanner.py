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

# Load .env file (check both btc15m dir and parent repo dir)
try:
    from dotenv import load_dotenv
    # Try btc15m directory first (for AWS deployment)
    env_path = BTC15M_DIR / ".env"
    if not env_path.exists():
        # Fall back to parent repo dir (for local development)
        env_path = REPO_DIR / ".env"
    if env_path.exists():
        load_dotenv(env_path)
        print(f"[ENV] Loaded {env_path}")
except ImportError:
    pass

from lib.binance import fetch_klines, fetch_current_price
from lib.binance_ws import start_price_stream, get_realtime_price, stop_price_stream
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
    insert_prediction_sample,
    get_unsettled_predictions,
    settle_prediction_sample,
    get_prediction_stats,
    push_log,
    get_setting,
    set_setting,
    calculate_optimal_threshold,
    get_recent_pnl,
    DEFAULT_DB_PATH,
)
from lib.ml_predictor import MLPredictor, compute_all_features
from lib.profit_predictor import get_profit_predictor, ProfitPredictor
from lib.discord import (
    send_trade_alert,
    send_settlement_alert,
    send_edge_analysis_alert,
    send_scan_log,
    send_threshold_update_alert,
    send_cumulative_results_alert,
    send_ml_training_alert,
)
from scripts.online_learn import save_live_sample, retrain_and_save

try:
    from kalshi_auth_client import KalshiAuthClient
except ImportError:
    KalshiAuthClient = None

# Store features for each trade to use when settling (for online learning)
TRADE_FEATURES: Dict[str, Dict[str, Any]] = {}

# Scan counter for periodic Discord updates
SCAN_COUNT = 0
DISCORD_LOG_INTERVAL = 20  # Send Discord log every N scans
RESULTS_ALERT_INTERVAL = 100  # Send cumulative results every N scans

# ML retraining tracker - retrain when predictions increase by 25%
LAST_RETRAIN_PREDICTION_COUNT = 0

# Dynamic threshold
CURRENT_THRESHOLD = 0.80  # Only trade at 80%+ confidence (profitable bucket)

KALSHI_FEE_RATE = 0.07
SCAN_INTERVAL_SECONDS = 2  # Base polling (event-driven as backup)

# Adaptive scanning intervals based on confidence proximity to threshold
LAST_CONFIDENCE = 0.50  # Track last seen ML confidence for adaptive scanning


def calculate_position_size(
    bankroll: float,
    edge: float,
    entry_price: float,
    min_contracts: int = 1,
    max_bet_small: float = 2.00,  # Max $2 per trade when bankroll < $25
    kelly_threshold: float = 25.00,  # Use Kelly sizing when bankroll >= $25
    max_fraction: float = 0.25,  # Never bet more than 25% of bankroll
) -> int:
    """Calculate position size based on bankroll level.

    - Below $25 bankroll: Fixed max $2 per trade (conservative growth)
    - At/above $25 bankroll: Half-Kelly criterion at 50%

    Args:
        bankroll: Available balance in dollars
        edge: Our edge (probability - entry_price)
        entry_price: Cost per contract as fraction (0-1)
        min_contracts: Minimum contracts to trade
        max_bet_small: Maximum dollar amount per trade when bankroll < $25
        kelly_threshold: Bankroll level to start using Kelly sizing
        max_fraction: Maximum fraction of bankroll to risk

    Returns:
        Number of contracts to trade (minimum 1)
    """
    if edge <= 0 or entry_price <= 0 or entry_price >= 1:
        return min_contracts

    cost_per_contract = entry_price  # e.g., 0.50 = 50 cents

    if bankroll < kelly_threshold:
        # Small bankroll: fixed max bet of $2
        bet_amount = min(max_bet_small, bankroll * 0.5)  # Max $2 or 50% of bankroll
        contracts = int(bet_amount / cost_per_contract)
        return max(contracts, min_contracts)

    # Large bankroll ($25+): use half-Kelly criterion
    # Kelly fraction: edge / (1 - entry_price)
    kelly_fraction = edge / (1 - entry_price)

    # Half-Kelly for conservative sizing
    half_kelly = kelly_fraction * 0.5

    # Cap at max_fraction to avoid over-betting
    betting_fraction = min(half_kelly, max_fraction)

    # Calculate dollar amount to bet
    bet_amount = bankroll * betting_fraction

    # Convert to contracts
    contracts = int(bet_amount / cost_per_contract)

    # Enforce minimum
    return max(contracts, min_contracts)


def get_adaptive_interval() -> float:
    """Get scan interval based on how close we are to the trading threshold.

    Returns shorter intervals when confidence is closer to threshold,
    saving compute when we're far from trading opportunities.
    """
    global LAST_CONFIDENCE, CURRENT_THRESHOLD

    confidence = LAST_CONFIDENCE
    threshold = CURRENT_THRESHOLD

    # Calculate how far we are from the threshold
    gap = threshold - confidence

    if gap <= 0:
        # Above threshold - should be trading, scan fast
        return 0.5
    elif gap <= 0.05:  # Within 5% of threshold (e.g., 75-80% when threshold is 80%)
        return 1.0
    elif gap <= 0.15:  # Within 15% of threshold (e.g., 65-80%)
        return 3.0
    elif gap <= 0.25:  # Within 25% of threshold (e.g., 55-80%)
        return 5.0
    else:
        # Far from threshold - scan less frequently
        return 15.0


def load_settings(db_path: Path) -> None:
    """Load threshold from database."""
    global CURRENT_THRESHOLD

    threshold_str = get_setting("threshold", "0.60", db_path)
    try:
        CURRENT_THRESHOLD = float(threshold_str)
    except ValueError:
        CURRENT_THRESHOLD = 0.60


def check_and_update_threshold(db_path: Path) -> None:
    """Check if threshold should be updated based on performance."""
    global CURRENT_THRESHOLD

    result = calculate_optimal_threshold(min_bucket_trades=3, db_path=db_path)
    new_threshold = result["optimal_threshold"]

    # Only update if significantly different (at least 5% change)
    if abs(new_threshold - CURRENT_THRESHOLD) >= 0.05:
        old_threshold = CURRENT_THRESHOLD
        CURRENT_THRESHOLD = new_threshold
        set_setting("threshold", str(new_threshold), db_path)

        summary = get_paper_trade_summary(db_path)
        print(f"\n[THRESHOLD] Updated: {old_threshold*100:.0f}% -> {new_threshold*100:.0f}%")
        print(f"[THRESHOLD] Reason: {result['reason']}")
        push_log(f"THRESHOLD: {old_threshold*100:.0f}% -> {new_threshold*100:.0f}%", db_path=db_path)

        send_threshold_update_alert(
            old_threshold=old_threshold,
            new_threshold=new_threshold,
            reason=result["reason"],
            bucket_analysis=result["analysis"],
            cumulative_pnl=summary["total_pnl"],
            total_record=f"{summary['wins']}W / {summary['losses']}L",
        )



def place_order_with_retry(
    client: KalshiAuthClient,
    ticker: str,
    direction: str,
    limit_price_cents: int,
    contracts: int,
    min_price_cents: int = 1,
    max_retries: int = 10,
    wait_seconds: int = 3,
) -> tuple[bool, Optional[str], int]:
    """
    Place order with aggressive retry logic - walks down price until filled.

    Args:
        client: Kalshi client
        ticker: Market ticker
        direction: 'UP' or 'DOWN'
        limit_price_cents: Starting price in cents
        contracts: Number of contracts
        min_price_cents: Minimum acceptable price (floor for +EV)
        max_retries: Max attempts before giving up
        wait_seconds: How long to wait for fill before trying lower price

    Returns:
        (filled, order_id, final_price_cents)
    """
    side = "yes" if direction == "UP" else "no"
    current_price = limit_price_cents

    for attempt in range(1, max_retries + 1):
        # Don't go below minimum price
        if current_price < min_price_cents:
            print(f"[ORDER] Price {current_price}c below min {min_price_cents}c - stopping")
            return False, None, 0

        print(f"\n[ORDER] Attempt {attempt}/{max_retries}: {contracts}x {side.upper()} @ {current_price}c on {ticker}")

        try:
            order = client.place_order(
                ticker=ticker,
                side=side,
                quantity=contracts,
                limit_price=current_price,
            )
            order_id = order.get("order_id")
            status = order.get("status", "unknown").lower()
            print(f"[ORDER] Placed! ID: {order_id}, Status: {status}")

            if status in ("filled", "executed"):
                print(f"[ORDER] Immediately filled @ {current_price}c!")
                return True, order_id, current_price

            # Wait briefly for fill
            time.sleep(wait_seconds)

            # Check status
            try:
                resp = client.get_order(order_id)
                current_status = resp.get("status", "").lower()
                print(f"[ORDER] Status after {wait_seconds}s: {current_status}")

                if current_status in ("filled", "executed"):
                    print(f"[ORDER] Filled @ {current_price}c!")
                    return True, order_id, current_price
                elif current_status in ("cancelled", "canceled", "expired"):
                    print(f"[ORDER] Order was cancelled/expired")
                else:
                    # Not filled yet - cancel and try lower price
                    print(f"[ORDER] Not filled - cancelling to try lower price")
                    try:
                        client.cancel_order(order_id)
                    except Exception as e:
                        print(f"[ORDER] Cancel error: {e}")
            except Exception as e:
                print(f"[ORDER] Status check error: {e}")

            # Try 1 cent lower next time
            current_price -= 1

        except Exception as e:
            print(f"[ORDER] Error: {e}")
            current_price -= 1  # Try lower price on error too

    print(f"[ORDER] Failed to fill after {max_retries} attempts")
    return False, None, 0


def check_prediction_settlements(db_path: Path, current_price: float):
    """Check and settle any expired prediction samples for learning."""
    import json
    unsettled = get_unsettled_predictions(db_path)
    settled_count = 0

    for pred in unsettled:
        expiry = datetime.fromisoformat(pred['expiry_time'].replace('Z', '+00:00'))
        now = datetime.now(timezone.utc)

        if now >= expiry:
            strike = pred['strike_price']
            actual = "UP" if current_price > strike else "DOWN"

            settle_prediction_sample(pred['id'], actual, db_path)
            settled_count += 1

            # Save to live_samples for online learning
            try:
                features = json.loads(pred['features_json']) if pred['features_json'] else {}
                save_live_sample(
                    db_path=db_path,
                    features=features,
                    outcome=actual,
                    prediction=pred['prediction'],
                    model_prob=pred['model_prob'],
                    pnl=0,  # No P&L for non-traded predictions (but still learns from outcome)
                    entry_price=pred.get('market_ask', 0.5),
                    edge=pred.get('edge', 0),
                )
            except Exception as e:
                print(f"[LEARN] Error saving prediction sample: {e}")

    if settled_count > 0:
        stats = get_prediction_stats(db_path)
        print(f"[LEARN] Settled {settled_count} predictions | "
              f"Accuracy: {stats['accuracy']*100:.1f}% ({stats['correct']}/{stats['settled']})")
        push_log(f"LEARN: {settled_count} predictions settled | Accuracy: {stats['accuracy']*100:.1f}%", db_path=db_path)


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

            # Save to live_samples for profit-weighted online learning
            ticker = trade['market_ticker']
            if ticker in TRADE_FEATURES:
                try:
                    save_live_sample(
                        db_path=db_path,
                        features=TRADE_FEATURES[ticker],
                        outcome=actual,  # Actual market outcome (UP/DOWN)
                        prediction=trade['side'],  # What we predicted
                        model_prob=trade['model_prob'],
                        pnl=pnl,  # For profit-weighted training
                        entry_price=trade['entry_price'],
                        edge=trade['edge'],
                    )
                    print(f"[LEARN] Saved sample: pred={trade['side']}, actual={actual}, pnl=${pnl:+.2f}")
                    del TRADE_FEATURES[ticker]  # Clean up
                except Exception as e:
                    print(f"[LEARN] Error saving sample: {e}")

            summary = get_paper_trade_summary(db_path)
            print(f"\n[SETTLE] {trade['market_ticker']}: {outcome} (${pnl:+.2f})")
            print(f"[CUMULATIVE] P&L: ${summary['total_pnl']:+.2f} | {summary['wins']}W/{summary['losses']}L")
            push_log(f"SETTLE: {outcome} ${pnl:+.2f} | Total: ${summary['total_pnl']:+.2f} ({summary['wins']}W/{summary['losses']}L)", db_path=db_path)

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
    contracts: int,
    traded_tickers: Set[str],
):
    """Run a single scan cycle."""
    print(f"\n{'='*60}")
    print(f"[SCAN] {datetime.now().strftime('%H:%M:%S')}")

    # Fetch current prices - try WebSocket first, fall back to REST
    try:
        ws_price = get_realtime_price()
        if ws_price:
            binance_price = ws_price
        else:
            binance_price = fetch_current_price()
        chainlink_price = get_chainlink_btc_price()
        print(f"[SCAN] BTC: ${binance_price:,.2f}")
    except Exception as e:
        print(f"[SCAN] Price error: {e}")
        push_log(f"ERROR: Price fetch failed - {e}", "ERROR", db_path)
        return

    # Check settlements for trades AND predictions
    settlement_price = chainlink_price or binance_price
    check_settlements(db_path, settlement_price)
    check_prediction_settlements(db_path, settlement_price)

    # Compute base features (without market context - will add per-market)
    try:
        candles = fetch_klines(interval="1m", limit=100)
        candle_dicts = [
            {"open": c.open, "high": c.high, "low": c.low, "close": c.close, "volume": c.volume}
            for c in candles
        ]
        current_hour = datetime.now(timezone.utc).hour
        base_features = compute_all_features(candle_dicts, current_hour)
    except Exception as e:
        print(f"[SCAN] Feature error: {e}")
        return

    # Fetch markets first (we need market context for predictions)
    try:
        markets = fetch_btc_15min_markets(current_btc_price=binance_price)
    except Exception as e:
        print(f"[SCAN] Market error: {e}")
        return

    if not markets:
        print("[SCAN] No markets near current price")
        return

    # Track which markets we've logged this cycle to avoid duplicates
    logged_tickers: Set[str] = set()

    # Update scan count for periodic Discord updates
    global SCAN_COUNT
    SCAN_COUNT += 1

    # Find best tradeable market AND log all predictions for learning
    for market in markets:
        probs = calculate_market_implied_prob(market)
        tte = (market.expiry_time - datetime.now(timezone.utc)).total_seconds()

        if tte < 120:  # Skip if less than 2 min to expiry
            continue

        # === SIMPLE RULE-BASED PREDICTION ===
        # Based on 685,000 simulated trades:
        # - Follow position (UP if above strike, DOWN if below) = 83% win rate
        # - With distance > 0.2% = 85-97% win rate
        # - ML model was only 29% win rate due to misaligned predictions

        time_to_expiry_min = tte / 60.0
        distance_to_strike_pct = ((binance_price - market.strike_price) / binance_price) * 100
        abs_distance = abs(distance_to_strike_pct)
        is_above_strike = binance_price > market.strike_price

        # SIMPLE RULE: Follow position
        # If price > strike, predict UP (bet it stays above)
        # If price < strike, predict DOWN (bet it stays below)
        if is_above_strike:
            direction = "UP"
            current_ask = probs["up_ask"]
        else:
            direction = "DOWN"
            current_ask = probs["down_ask"]

        # Estimate win probability based on distance (from backtest data)
        # Distance > 0.4%: 96-98% win rate
        # Distance 0.3-0.4%: 94% win rate
        # Distance 0.2-0.3%: 85-90% win rate
        # Distance < 0.2%: 70-80% win rate
        if abs_distance >= 0.4:
            estimated_prob = 0.97
        elif abs_distance >= 0.3:
            estimated_prob = 0.94
        elif abs_distance >= 0.2:
            estimated_prob = 0.87
        elif abs_distance >= 0.15:
            estimated_prob = 0.80
        else:
            estimated_prob = 0.70

        # Create a simple prediction object for compatibility
        class SimplePrediction:
            pass
        prediction = SimplePrediction()
        prediction.direction = direction
        prediction.probability = estimated_prob

        # Update global confidence for adaptive scanning
        global LAST_CONFIDENCE
        if market == markets[0]:
            LAST_CONFIDENCE = prediction.probability
            position_str = "ABOVE" if is_above_strike else "BELOW"
            print(f"[SCAN] Position: {position_str} strike by {abs_distance:.2f}% | Predict: {direction} ({estimated_prob*100:.0f}%) | TTE: {int(tte/60)}m")
            push_log(f"Rule: {direction} (dist={abs_distance:.2f}%)", db_path=db_path)

            # Send periodic Discord updates
            if SCAN_COUNT % DISCORD_LOG_INTERVAL == 0:
                summary = get_paper_trade_summary(db_path)
                send_scan_log(
                    f"Scan #{SCAN_COUNT} | P&L: ${summary['total_pnl']:+.2f} | {summary['wins']}W/{summary['losses']}L",
                    btc_price=binance_price,
                    prediction=prediction.direction,
                    probability=prediction.probability,
                )

        # Use estimated probability for edge calculation
        prob = prediction.probability
        max_ev_price = (prob * (1 - KALSHI_FEE_RATE)) / (1 - prob * KALSHI_FEE_RATE)

        # Use market ask - this will actually fill
        entry_price = current_ask
        edge = prob - entry_price

        # === LOG ALL PREDICTIONS FOR LEARNING ===
        # Save prediction for this market (whether we trade or not)
        if market.ticker not in logged_tickers:
            insert_prediction_sample(
                market_ticker=market.ticker,
                expiry_time=market.expiry_time.isoformat(),
                strike_price=market.strike_price,
                btc_price=binance_price,
                prediction=prediction.direction,
                model_prob=prediction.probability,
                features=base_features,
                market_ask=current_ask,
                edge=edge,
                was_traded=False,  # Will update if we trade
                db_path=db_path,
            )
            logged_tickers.add(market.ticker)

        # Skip if already traded
        if market.ticker in traded_tickers:
            continue

        print(f"[SCAN] Market: {market.ticker}")
        print(f"[SCAN] Strike: ${market.strike_price:,.0f}, Expires: {int(tte/60)}m")
        print(f"[SCAN] ML: {prob*100:.1f}%, Max +EV: {max_ev_price*100:.1f}¢, Ask: {current_ask*100:.1f}¢")
        print(f"[SCAN] Edge: {edge*100:+.1f}%")

        # === SIMPLE RULE-BASED TRADE FILTERING ===
        # Based on 685,000 simulated trades and 154 actual trades analysis:
        # - Aligned + >0.2% distance: 81.5% win rate, +$8.13 P&L
        # - Entry 30-70¢: Best performance
        # - Cheap entries (<30¢): 7% win rate even when aligned

        MIN_DISTANCE_LIVE = 0.20   # Minimum distance for live trades (81.5% win rate)
        MIN_BID_CENTS = 30         # <30¢ has 7% win rate - avoid!
        MAX_BID_CENTS = 80         # >80¢ - expanded from 70¢

        ask_cents = int(current_ask * 100)

        # Track why we might skip live trading
        live_skip_reasons = []

        # === NEW DISTANCE-BASED CRITERIA ===
        # (Alignment is automatic since we use the simple rule)

        # FILTER 1: Distance check - need >0.2% for good win rates
        if abs_distance < MIN_DISTANCE_LIVE:
            live_skip_reasons.append(f"distance {abs_distance:.2f}% < {MIN_DISTANCE_LIVE}% (need more margin)")

        # FILTER 2: Entry price must be 30-70¢ (sweet spot)
        if ask_cents < MIN_BID_CENTS:
            live_skip_reasons.append(f"price {ask_cents}¢ < {MIN_BID_CENTS}¢ (cheap entries lose)")

        if ask_cents > MAX_BID_CENTS:
            live_skip_reasons.append(f"price {ask_cents}¢ > {MAX_BID_CENTS}¢ (expensive entries risky)")

        # FILTER 3: Check edge is positive
        if edge < 0.05:
            live_skip_reasons.append(f"edge {edge*100:.1f}% < 5%")

        # No profit predictor needed - simple rules are more reliable!

        # Check if trade meets all live criteria
        is_live_eligible = len(live_skip_reasons) == 0

        if not is_live_eligible:
            if live_skip_reasons:
                print(f"[SCAN] Skip: {', '.join(live_skip_reasons[:2])}")
            continue

        print(f"[SCAN] ✓ All live criteria passed (edge={edge*100:+.1f}%, price={ask_cents}¢, dist={abs_distance:.2f}%)")

        # Place real order
        # Start at market ask, can go down to max_ev_price (floor where still +EV)
        start_price_cents = int(entry_price * 100)
        min_price_cents = max(1, int(max_ev_price * 100))  # Floor for +EV

        # Calculate position size using half-Kelly criterion
        try:
            balance_resp = kalshi_client.get_balance()
            bankroll = balance_resp.get('balance', 0) / 100.0  # Convert cents to dollars
        except Exception as e:
            print(f"[ORDER] Balance check failed: {e}, using 1 contract")
            bankroll = 0

        sized_contracts = calculate_position_size(
            bankroll=bankroll,
            edge=edge,
            entry_price=entry_price,
            min_contracts=1,  # Always trade at least 1 contract
        )
        # Use min of sized and user-specified max
        actual_contracts = min(sized_contracts, contracts) if contracts > 0 else sized_contracts

        sizing_mode = "Kelly" if bankroll >= 25 else f"Fixed(max $2)"
        print(f"[ORDER] {sizing_mode}: {sized_contracts} contracts (bankroll: ${bankroll:.2f}, edge: {edge*100:.1f}%)")
        print(f"[ORDER] Trading: {actual_contracts} contracts @ {start_price_cents}c down to {min_price_cents}c")

        filled, order_id, final_price_cents = place_order_with_retry(
            client=kalshi_client,
            ticker=market.ticker,
            direction=prediction.direction,
            limit_price_cents=start_price_cents,
            contracts=actual_contracts,
            min_price_cents=min_price_cents,
        )

        if filled:
            traded_tickers.add(market.ticker)

            # Use actual fill price for recording
            actual_entry_price = final_price_cents / 100.0
            actual_edge = prob - actual_entry_price

            # Store features for online learning (will be used at settlement)
            TRADE_FEATURES[market.ticker] = base_features.copy() if base_features else {}

            # Update prediction sample to mark as traded
            insert_prediction_sample(
                market_ticker=market.ticker,
                expiry_time=market.expiry_time.isoformat(),
                strike_price=market.strike_price,
                btc_price=binance_price,
                prediction=prediction.direction,
                model_prob=prediction.probability,
                features=base_features,
                market_ask=current_ask,
                edge=actual_edge,
                was_traded=True,
                db_path=db_path,
            )

            # Record trade with actual fill price
            trade_id = insert_paper_trade(
                market_ticker=market.ticker,
                side=prediction.direction,
                entry_price=actual_entry_price,
                stake=actual_contracts,
                model_prob=prediction.probability,
                market_prob=actual_entry_price,
                edge=actual_edge,
                expiry_time=market.expiry_time.isoformat(),
                strike_price=market.strike_price,
                db_path=db_path,
            )

            summary = get_paper_trade_summary(db_path)
            print(f"\n[TRADE] {prediction.direction} filled on {market.ticker} @ {final_price_cents}¢")
            print(f"[TRADE] Actual edge: {actual_edge*100:+.1f}%")
            print(f"[TRADE] Cumulative P&L: ${summary['total_pnl']:+.2f}")
            push_log(f"TRADE: {prediction.direction} @ {final_price_cents}¢ | Edge: {actual_edge*100:+.1f}% | P&L: ${summary['total_pnl']:+.2f}", db_path=db_path)

            send_trade_alert(
                side=prediction.direction,
                market_ticker=market.ticker,
                strike_price=market.strike_price,
                model_prob=prediction.probability,
                market_prob=actual_entry_price,
                edge=actual_edge,
                btc_price=binance_price,
                time_to_expiry_sec=int(tte),
                cumulative_pnl=summary['total_pnl'],
                total_trades=summary['total_trades'],
                win_rate=summary['win_rate'],
            )
            # Continue to allow multiple trades per scan window

    # Print learning stats periodically
    stats = get_prediction_stats(db_path)
    if stats['settled'] > 0:
        print(f"[LEARN] Predictions: {stats['total_predictions']} total, {stats['pending']} pending | "
              f"Accuracy: {stats['accuracy']*100:.1f}% ({stats['correct']}/{stats['settled']})")

    # Check if we should retrain ML model (when settled predictions increase by 25%)
    global LAST_RETRAIN_PREDICTION_COUNT
    current_predictions = stats['settled']  # Use settled predictions, not total

    # Initialize on first run
    if LAST_RETRAIN_PREDICTION_COUNT == 0:
        LAST_RETRAIN_PREDICTION_COUNT = current_predictions

    # Check if we should retrain (25% more settled predictions)
    prediction_increase = current_predictions >= LAST_RETRAIN_PREDICTION_COUNT * 1.25

    should_retrain = False
    retrain_reason = ""

    if prediction_increase and current_predictions >= 50:  # Minimum 50 settled samples
        should_retrain = True
        retrain_reason = f"settled predictions +25% ({LAST_RETRAIN_PREDICTION_COUNT} -> {current_predictions})"

    if should_retrain:
        print(f"\n[RETRAIN] Triggered: {retrain_reason}")
        push_log(f"RETRAIN: {retrain_reason}", db_path=db_path)

        try:
            model_path = Path(__file__).resolve().parent.parent / "models" / "btc15m_model.pkl"
            success = retrain_and_save(db_path, model_path, profit_weighted=True)
            if success:
                LAST_RETRAIN_PREDICTION_COUNT = current_predictions
                print(f"[RETRAIN] Model updated! Next at {int(current_predictions * 1.25)} settled predictions")
                push_log(f"RETRAIN: Success!", db_path=db_path)

                # Reload the model
                predictor.reload_model()
                print("[RETRAIN] Predictor reloaded with new model")

                # Update threshold based on new performance data
                check_and_update_threshold(db_path)

                # Send Discord alert about retraining
                summary = get_paper_trade_summary(db_path)
                send_ml_training_alert(
                    total_samples=current_predictions,
                    live_samples=current_predictions,
                    historical_samples=0,
                    model_name="RandomForest",
                    accuracy=stats['accuracy'],
                    profit_weighted=True,
                    cumulative_pnl=summary['total_pnl'],
                    total_record=f"{summary['wins']}W / {summary['losses']}L",
                    live_accuracy=stats['accuracy'],
                )
        except Exception as e:
            print(f"[RETRAIN] Error: {e}")
            push_log(f"RETRAIN ERROR: {e}", "ERROR", db_path)

    # Send cumulative results periodically
    if SCAN_COUNT % RESULTS_ALERT_INTERVAL == 0 and SCAN_COUNT > 0:
        summary = get_paper_trade_summary(db_path)
        send_cumulative_results_alert(
            total_pnl=summary['total_pnl'],
            total_trades=summary['total_trades'],
            wins=summary['wins'],
            losses=summary['losses'],
            pending=summary['pending'],
            win_rate=summary['win_rate'],
            avg_edge=summary['avg_edge'],
            current_threshold=CURRENT_THRESHOLD,
            paper_mode=False,
        )


def main():
    parser = argparse.ArgumentParser(description="Local BTC 15-min continuous scanner")
    parser.add_argument("--min-prob", type=float, default=0.60,
                        help="Initial minimum probability (default: 0.60, will be optimized)")
    parser.add_argument("--contracts", type=int, default=1,
                        help="Contracts per trade (default: 1)")
    parser.add_argument("--db", type=str, default=str(DEFAULT_DB_PATH))
    parser.add_argument("--interval", type=int, default=SCAN_INTERVAL_SECONDS,
                        help="Scan interval in seconds (default: 15)")
    args = parser.parse_args()

    db_path = Path(args.db)
    ensure_schema(db_path)

    # Load saved settings (paper mode, threshold)
    load_settings(db_path)

    # Initialize threshold from args if not set
    global CURRENT_THRESHOLD
    if get_setting("threshold", db_path=db_path) is None:
        CURRENT_THRESHOLD = args.min_prob
        set_setting("threshold", str(args.min_prob), db_path)

    print("=" * 60)
    print("BTC 15-MIN LOCAL SCANNER (LIVE)")
    print("=" * 60)
    print(f"Dynamic Threshold: {CURRENT_THRESHOLD*100:.0f}% (auto-optimized)")
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

    # Event-driven state
    last_scan_price: Optional[float] = None
    last_scan_time: float = 0
    known_strikes: Set[float] = set()
    PRICE_MOVE_THRESHOLD = 25  # Scan when price moves $25+
    MAX_SCAN_INTERVAL = 10  # Force scan at least every 10s
    MIN_SCAN_INTERVAL = 0.5  # Don't scan faster than 500ms

    def should_scan(current_price: float) -> tuple[bool, str]:
        """Check if we should trigger a scan based on price movement."""
        nonlocal last_scan_price, last_scan_time, known_strikes

        now = time.time()

        # Don't scan too fast
        if now - last_scan_time < MIN_SCAN_INTERVAL:
            return False, ""

        # Force periodic scan
        if now - last_scan_time >= MAX_SCAN_INTERVAL:
            return True, "periodic"

        # First scan
        if last_scan_price is None:
            return True, "initial"

        # Price moved significantly
        price_move = abs(current_price - last_scan_price)
        if price_move >= PRICE_MOVE_THRESHOLD:
            return True, f"price_move_${price_move:.0f}"

        # Price crossed a known strike
        for strike in known_strikes:
            if (last_scan_price < strike <= current_price) or (current_price <= strike < last_scan_price):
                return True, f"crossed_${strike:.0f}"

        return False, ""

    def on_price_update(trade):
        """Called on each WebSocket price update."""
        nonlocal last_scan_price, last_scan_time, known_strikes

        current_price = trade.price
        do_scan, reason = should_scan(current_price)

        if do_scan:
            print(f"\n[EVENT] Scan triggered: {reason} @ ${current_price:,.2f}")
            try:
                run_scan_cycle(
                    predictor=predictor,
                    kalshi_client=kalshi_client,
                    db_path=db_path,
                    contracts=args.contracts,
                    traded_tickers=traded_tickers,
                )
                last_scan_price = current_price
                last_scan_time = time.time()

                # Update known strikes from recent markets
                try:
                    markets = fetch_btc_15min_markets(current_btc_price=current_price)
                    known_strikes = {m.strike_price for m in markets}
                except:
                    pass

            except Exception as e:
                print(f"[ERROR] Scan cycle error: {e}")

    # Start Binance WebSocket for real-time prices with event callback
    print("Starting event-driven scanner with Binance WebSocket...")
    try:
        start_price_stream(on_trade=on_price_update)
        # Wait for first trade to come through
        print("Waiting for WebSocket to receive first trade...")
        for _ in range(10):
            time.sleep(0.5)
            if get_realtime_price() is not None:
                break
        if get_realtime_price():
            print(f"WebSocket ready - BTC ${get_realtime_price():,.2f}")
            print(f"  - Triggers: price move >${PRICE_MOVE_THRESHOLD}, strike cross, or every {MAX_SCAN_INTERVAL}s")
        else:
            print("WebSocket connected but no trades yet, will use fallback")
    except Exception as e:
        print(f"WebSocket failed ({e}) - falling back to polling mode")

    print("\nStarting event-driven scan (Ctrl+C to stop)...")

    try:
        # Keep main thread alive, websocket callback handles scanning
        while True:
            # Check if websocket is delivering prices
            ws_price = get_realtime_price()
            if ws_price is None:
                # WebSocket not working, do manual fallback scan with adaptive interval
                adaptive_interval = get_adaptive_interval()
                print(f"[FALLBACK] Polling (interval: {adaptive_interval:.1f}s, confidence: {LAST_CONFIDENCE*100:.0f}%)...")
                try:
                    run_scan_cycle(
                        predictor=predictor,
                        kalshi_client=kalshi_client,
                        db_path=db_path,
                        contracts=args.contracts,
                        traded_tickers=traded_tickers,
                    )
                except Exception as e:
                    print(f"[ERROR] Scan cycle error: {e}")
                time.sleep(adaptive_interval)
            else:
                # WebSocket working, just sleep and let callbacks handle it
                time.sleep(1)

    except KeyboardInterrupt:
        print("\n\nScanner stopped by user")
        stop_price_stream()

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
