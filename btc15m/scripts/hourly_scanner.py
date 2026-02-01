#!/usr/bin/env python3
"""
BTC 1-Hour Trading Scanner.

Trades on Kalshi KXBTC hourly markets with half-Kelly position sizing.
Logs comprehensive indicators for ML training.

Usage:
    python hourly_scanner.py [--interval 60] [--live]
"""

from __future__ import annotations

import argparse
import json
import os
import sqlite3
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Dict, Any, List, Set, Tuple

# Get absolute paths for imports
SCRIPT_DIR = Path(__file__).resolve().parent
BTC15M_DIR = SCRIPT_DIR.parent
REPO_DIR = BTC15M_DIR.parent

sys.path.insert(0, str(REPO_DIR))
sys.path.insert(0, str(BTC15M_DIR))

# Load .env file
try:
    from dotenv import load_dotenv
    env_path = BTC15M_DIR / ".env"
    if not env_path.exists():
        env_path = REPO_DIR / ".env"
    if env_path.exists():
        load_dotenv(env_path)
        print(f"[ENV] Loaded {env_path}")
except ImportError:
    pass

from lib.binance import fetch_klines, fetch_current_price
from lib.chainlink import get_chainlink_btc_price
from lib.kalshi_btc import get_kalshi_client, calculate_market_implied_prob
from lib.ml_predictor import compute_all_features
from lib.discord import send_discord_alert, send_hourly_trade_alert, send_hourly_settlement_alert

try:
    from kalshi_auth_client import KalshiAuthClient
except ImportError:
    KalshiAuthClient = None

# Calibration tracking file
CALIBRATION_STATE_FILE = BTC15M_DIR / "calibration_state.json"

# Database path for hourly trades
HOURLY_DB_PATH = BTC15M_DIR / "hourly_trades.db"

# Discord webhook
DISCORD_WEBHOOK = os.environ.get("BTC_15MIN_DISCORD")

# Trading constants
KALSHI_FEE_RATE = 0.07

# Recalibration threshold: retrain when settled trades increase by 25%
RECALIBRATION_THRESHOLD = 1.25


def load_calibration_state() -> Dict[str, Any]:
    """Load calibration state from file."""
    if CALIBRATION_STATE_FILE.exists():
        try:
            return json.loads(CALIBRATION_STATE_FILE.read_text())
        except:
            pass
    return {"last_calibration_count": 0, "last_calibration_time": None}


def save_calibration_state(settled_count: int) -> None:
    """Save calibration state to file."""
    state = {
        "last_calibration_count": settled_count,
        "last_calibration_time": datetime.now(timezone.utc).isoformat(),
    }
    CALIBRATION_STATE_FILE.write_text(json.dumps(state, indent=2))


def check_recalibration_needed(db_path: Path) -> Tuple[bool, int, int]:
    """Check if ML model recalibration is needed.

    Returns (should_recalibrate, current_count, last_count).
    Recalibrates when settled trades increase by 25% from last calibration.
    """
    conn = sqlite3.connect(db_path)
    c = conn.cursor()

    # Get count of settled predictions
    c.execute("SELECT COUNT(*) FROM hourly_predictions WHERE settled = 1")
    current_count = c.fetchone()[0]
    conn.close()

    state = load_calibration_state()
    last_count = state.get("last_calibration_count", 0)

    # Recalibrate when we have 25% more data than last calibration
    # Minimum 50 samples before first calibration
    if current_count >= 50 and (last_count == 0 or current_count >= last_count * RECALIBRATION_THRESHOLD):
        return True, current_count, last_count

    return False, current_count, last_count


def run_recalibration(db_path: Path) -> bool:
    """Run ML model recalibration based on settled predictions.

    This analyzes what features/conditions led to correct vs incorrect predictions
    and could in the future train a model. For now, it just updates the state.
    """
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    c = conn.cursor()

    # Get settled predictions with outcomes
    c.execute("""
        SELECT
            prediction_correct,
            distance_pct,
            time_to_expiry_sec,
            volatility_1h,
            trend_1h,
            model_prob,
            edge,
            strike_type
        FROM hourly_predictions
        WHERE settled = 1 AND prediction_correct IS NOT NULL
    """)
    rows = c.fetchall()

    if len(rows) < 50:
        print(f"[CALIBRATE] Not enough data ({len(rows)} samples, need 50+)")
        conn.close()
        return False

    # Analyze accuracy by different factors
    total = len(rows)
    correct = sum(1 for r in rows if r["prediction_correct"] == 1)
    accuracy = correct / total if total > 0 else 0

    # Accuracy by distance buckets
    close_correct = sum(1 for r in rows if r["prediction_correct"] == 1 and abs(r["distance_pct"] or 0) < 0.005)
    close_total = sum(1 for r in rows if abs(r["distance_pct"] or 0) < 0.005)

    mid_correct = sum(1 for r in rows if r["prediction_correct"] == 1 and 0.005 <= abs(r["distance_pct"] or 0) < 0.015)
    mid_total = sum(1 for r in rows if 0.005 <= abs(r["distance_pct"] or 0) < 0.015)

    far_correct = sum(1 for r in rows if r["prediction_correct"] == 1 and abs(r["distance_pct"] or 0) >= 0.015)
    far_total = sum(1 for r in rows if abs(r["distance_pct"] or 0) >= 0.015)

    print(f"\n[CALIBRATE] ===== ML RECALIBRATION =====")
    print(f"[CALIBRATE] Total samples: {total}")
    print(f"[CALIBRATE] Overall accuracy: {accuracy*100:.1f}%")
    print(f"[CALIBRATE] Close (<0.5%): {close_correct}/{close_total} = {(close_correct/close_total*100 if close_total else 0):.1f}%")
    print(f"[CALIBRATE] Mid (0.5-1.5%): {mid_correct}/{mid_total} = {(mid_correct/mid_total*100 if mid_total else 0):.1f}%")
    print(f"[CALIBRATE] Far (>1.5%): {far_correct}/{far_total} = {(far_correct/far_total*100 if far_total else 0):.1f}%")

    # Accuracy by time to expiry
    short_correct = sum(1 for r in rows if r["prediction_correct"] == 1 and (r["time_to_expiry_sec"] or 0) < 1800)
    short_total = sum(1 for r in rows if (r["time_to_expiry_sec"] or 0) < 1800)

    long_correct = sum(1 for r in rows if r["prediction_correct"] == 1 and (r["time_to_expiry_sec"] or 0) >= 1800)
    long_total = sum(1 for r in rows if (r["time_to_expiry_sec"] or 0) >= 1800)

    print(f"[CALIBRATE] Short (<30m): {short_correct}/{short_total} = {(short_correct/short_total*100 if short_total else 0):.1f}%")
    print(f"[CALIBRATE] Long (>=30m): {long_correct}/{long_total} = {(long_correct/long_total*100 if long_total else 0):.1f}%")
    print(f"[CALIBRATE] =============================\n")

    conn.close()

    # Save new calibration state
    save_calibration_state(total)

    return True


def calculate_half_kelly_contracts(
    bankroll: float,
    edge: float,
    entry_price: float,
    min_contracts: int = 1,
    max_fraction: float = 0.10,  # Conservative: max 10% per hourly trade
) -> int:
    """Calculate position size using half-Kelly criterion."""
    if edge <= 0 or entry_price <= 0 or entry_price >= 1:
        return min_contracts

    kelly_fraction = edge / (1 - entry_price)
    half_kelly = kelly_fraction * 0.5
    betting_fraction = min(half_kelly, max_fraction)
    bet_amount = bankroll * betting_fraction
    cost_per_contract = entry_price
    contracts = int(bet_amount / cost_per_contract)

    return max(contracts, min_contracts)


def place_hourly_order(
    client: KalshiAuthClient,
    ticker: str,
    side: str,  # 'yes' or 'no'
    limit_price_cents: int,
    contracts: int,
    max_retries: int = 3,
) -> tuple[bool, Optional[str], int]:
    """Place order for hourly market."""
    for attempt in range(1, max_retries + 1):
        try:
            print(f"[ORDER] Attempt {attempt}: {contracts}x {side.upper()} @ {limit_price_cents}c on {ticker}")

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
                return True, order_id, limit_price_cents

            # Wait briefly for fill
            time.sleep(2)

            # Check status
            try:
                resp = client.get_order(order_id)
                current_status = resp.get("status", "").lower()

                if current_status in ("filled", "executed"):
                    return True, order_id, limit_price_cents
                else:
                    # Not filled - try to cancel
                    try:
                        client.cancel_order(order_id)
                    except:
                        pass
            except Exception as e:
                print(f"[ORDER] Status check error: {e}")

        except Exception as e:
            print(f"[ORDER] Error: {e}")

    return False, None, 0


def ensure_hourly_schema(db_path: Path) -> None:
    """Create database tables for hourly trading."""
    conn = sqlite3.connect(db_path)
    c = conn.cursor()

    # Predictions table with comprehensive features
    c.execute("""
        CREATE TABLE IF NOT EXISTS hourly_predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP,
            market_ticker TEXT NOT NULL,
            expiry_time TEXT NOT NULL,
            strike_price REAL NOT NULL,
            strike_type TEXT NOT NULL,  -- 'above', 'below', or 'range'
            range_lower REAL,  -- For range markets: lower bound
            range_upper REAL,  -- For range markets: upper bound
            btc_price_at_prediction REAL NOT NULL,
            distance_to_strike REAL NOT NULL,  -- BTC price - strike (or distance to range)
            distance_pct REAL NOT NULL,  -- percentage distance
            time_to_expiry_sec INTEGER NOT NULL,

            -- Market data
            yes_ask REAL,
            yes_bid REAL,
            no_ask REAL,
            no_bid REAL,
            implied_prob REAL,
            spread REAL,
            volume INTEGER,
            open_interest INTEGER,

            -- Our prediction
            prediction TEXT,  -- 'above' or 'below'
            model_prob REAL,
            edge REAL,

            -- Technical indicators (from features)
            features_json TEXT,

            -- Additional indicators
            volatility_1h REAL,
            volatility_24h REAL,
            trend_1h REAL,
            trend_4h REAL,
            rsi_14 REAL,
            macd_signal REAL,
            bb_position REAL,  -- position within bollinger bands
            vwap_distance REAL,
            orderbook_imbalance REAL,

            -- Settlement
            settled INTEGER DEFAULT 0,
            final_btc_price REAL,
            actual_outcome TEXT,  -- 'above' or 'below'
            prediction_correct INTEGER,
            pnl_if_traded REAL,
            settled_at TEXT
        )
    """)

    # Paper trades table
    c.execute("""
        CREATE TABLE IF NOT EXISTS hourly_paper_trades (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP,
            prediction_id INTEGER REFERENCES hourly_predictions(id),
            market_ticker TEXT NOT NULL,
            side TEXT NOT NULL,  -- 'yes' or 'no'
            entry_price REAL NOT NULL,
            contracts INTEGER DEFAULT 1,
            model_prob REAL,
            edge REAL,
            expiry_time TEXT,
            strike_price REAL,
            settled INTEGER DEFAULT 0,
            outcome TEXT,  -- 'WIN' or 'LOSS'
            pnl REAL,
            settled_at TEXT
        )
    """)

    # Real trades table
    c.execute("""
        CREATE TABLE IF NOT EXISTS hourly_real_trades (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP,
            prediction_id INTEGER REFERENCES hourly_predictions(id),
            market_ticker TEXT NOT NULL,
            order_id TEXT,
            side TEXT NOT NULL,
            entry_price REAL NOT NULL,
            contracts INTEGER NOT NULL,
            model_prob REAL,
            edge REAL,
            expiry_time TEXT,
            strike_price REAL,
            btc_price_at_trade REAL,
            settled INTEGER DEFAULT 0,
            outcome TEXT,
            pnl REAL,
            settled_at TEXT
        )
    """)

    # Indexes
    c.execute("CREATE INDEX IF NOT EXISTS idx_hourly_pred_settled ON hourly_predictions(settled)")
    c.execute("CREATE INDEX IF NOT EXISTS idx_hourly_pred_expiry ON hourly_predictions(expiry_time)")
    c.execute("CREATE INDEX IF NOT EXISTS idx_hourly_trades_settled ON hourly_paper_trades(settled)")
    c.execute("CREATE INDEX IF NOT EXISTS idx_hourly_real_settled ON hourly_real_trades(settled)")

    conn.commit()
    conn.close()


def fetch_hourly_markets(client: KalshiAuthClient) -> List[Dict]:
    """Fetch KXBTC hourly RANGE markets expiring in 10-120 minutes.

    IMPORTANT: KXBTC markets are RANGE markets, not simple above/below.
    Example: "Bitcoin price range $77,500 to 77,749.99"
    YES wins only if BTC settles WITHIN the specified range.
    """
    try:
        result = client.list_markets(series_ticker="KXBTC", status="open")
    except Exception as e:
        print(f"[kalshi] Error fetching KXBTC: {e}")
        return []

    if not result:
        return []

    now = datetime.now(timezone.utc)
    markets = []

    for m in result:
        try:
            exp_str = m.get("close_time") or m.get("expiration_time")
            if not exp_str:
                continue

            expiry = datetime.fromisoformat(exp_str.replace("Z", "+00:00"))
            tte = (expiry - now).total_seconds()

            # Keep markets expiring in 10-120 minutes
            if not (600 <= tte <= 7200):
                continue

            ticker = m.get("ticker", "")
            title = m.get("title", "")

            # Parse range from title: "Bitcoin price range today at Xpm EST? $77,500 to 77,749.99"
            # or similar format with range bounds
            range_lower = None
            range_upper = None

            # Try to parse range from title
            import re
            # Pattern: "$XX,XXX to XX,XXX" or "$XX,XXX.XX to $XX,XXX.XX"
            range_match = re.search(r'\$?([\d,]+(?:\.\d+)?)\s+to\s+\$?([\d,]+(?:\.\d+)?)', title)
            if range_match:
                try:
                    range_lower = float(range_match.group(1).replace(',', ''))
                    range_upper = float(range_match.group(2).replace(',', ''))
                except:
                    pass

            # Fallback: parse from ticker and assume ~$250 range buckets
            # Format: KXBTC-26JAN3116-B91375 means range starting at $91,375
            if range_lower is None:
                parts = ticker.split("-")
                if len(parts) >= 3:
                    strike_part = parts[-1]
                    try:
                        # B prefix = bottom of range
                        range_lower = float(strike_part[1:])
                        # Assume $250 range buckets (common Kalshi structure)
                        range_upper = range_lower + 249.99
                    except:
                        continue
                else:
                    continue

            if range_lower is None or range_upper is None:
                continue

            markets.append({
                "ticker": ticker,
                "title": title,
                "expiry_time": expiry,
                "range_lower": range_lower,
                "range_upper": range_upper,
                "range_width": range_upper - range_lower,
                "market_type": "range",  # Explicitly mark as range market
                # Legacy fields for compatibility (use midpoint as "strike")
                "strike_price": (range_lower + range_upper) / 2,
                "strike_type": "range",
                "yes_ask": m.get("yes_ask"),
                "yes_bid": m.get("yes_bid"),
                "no_ask": m.get("no_ask"),
                "no_bid": m.get("no_bid"),
                "volume": m.get("volume", 0),
                "open_interest": m.get("open_interest", 0),
                "time_to_expiry": tte,
            })
        except Exception as e:
            continue

    return markets


def compute_additional_indicators(candles: List[Dict]) -> Dict[str, float]:
    """Compute additional indicators for ML training."""
    if not candles or len(candles) < 20:
        return {}

    closes = [c["close"] for c in candles]
    highs = [c["high"] for c in candles]
    lows = [c["low"] for c in candles]
    volumes = [c["volume"] for c in candles]

    # Volatility (std of returns)
    returns_1h = [(closes[i] - closes[i-1]) / closes[i-1] for i in range(-60, 0) if i > -len(closes)]
    returns_24h = [(closes[i] - closes[i-1]) / closes[i-1] for i in range(-min(1440, len(closes)), 0)]

    vol_1h = (sum(r**2 for r in returns_1h[-60:]) / len(returns_1h[-60:])) ** 0.5 if len(returns_1h) >= 60 else 0
    vol_24h = (sum(r**2 for r in returns_24h) / len(returns_24h)) ** 0.5 if returns_24h else 0

    # Trend (price change)
    trend_1h = (closes[-1] - closes[-60]) / closes[-60] if len(closes) >= 60 else 0
    trend_4h = (closes[-1] - closes[-min(240, len(closes))]) / closes[-min(240, len(closes))] if len(closes) >= 2 else 0

    # RSI
    gains = [max(0, closes[i] - closes[i-1]) for i in range(-14, 0)]
    losses = [max(0, closes[i-1] - closes[i]) for i in range(-14, 0)]
    avg_gain = sum(gains) / len(gains) if gains else 0
    avg_loss = sum(losses) / len(losses) if losses else 0.0001
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))

    # Bollinger Band position
    sma_20 = sum(closes[-20:]) / 20 if len(closes) >= 20 else closes[-1]
    std_20 = (sum((c - sma_20)**2 for c in closes[-20:]) / 20) ** 0.5 if len(closes) >= 20 else 1
    bb_upper = sma_20 + 2 * std_20
    bb_lower = sma_20 - 2 * std_20
    bb_position = (closes[-1] - bb_lower) / (bb_upper - bb_lower) if bb_upper != bb_lower else 0.5

    # VWAP distance
    if volumes and sum(volumes[-60:]) > 0:
        vwap = sum(c * v for c, v in zip(closes[-60:], volumes[-60:])) / sum(volumes[-60:])
        vwap_distance = (closes[-1] - vwap) / vwap
    else:
        vwap_distance = 0

    return {
        "volatility_1h": vol_1h,
        "volatility_24h": vol_24h,
        "trend_1h": trend_1h,
        "trend_4h": trend_4h,
        "rsi_14": rsi,
        "bb_position": bb_position,
        "vwap_distance": vwap_distance,
    }


def make_prediction(
    btc_price: float,
    strike_price: float,
    strike_type: str,
    time_to_expiry: float,
    features: Dict[str, float],
    indicators: Dict[str, float],
    range_lower: Optional[float] = None,
    range_upper: Optional[float] = None,
) -> tuple[str, float]:
    """Make a prediction for range or directional markets.

    For RANGE markets (KXBTC hourly):
    - YES wins only if BTC settles WITHIN the range [range_lower, range_upper]
    - These are typically ~$250 wide buckets
    - Probability depends on how close BTC is to the range and volatility

    Returns (prediction, probability_of_yes_winning).
    """
    # RANGE MARKET LOGIC
    if strike_type == "range" and range_lower is not None and range_upper is not None:
        range_width = range_upper - range_lower
        range_midpoint = (range_lower + range_upper) / 2

        # Distance from current price to range
        if btc_price < range_lower:
            distance_to_range = range_lower - btc_price
            position = "below"
        elif btc_price > range_upper:
            distance_to_range = btc_price - range_upper
            position = "above"
        else:
            distance_to_range = 0
            position = "inside"

        distance_pct = distance_to_range / btc_price if btc_price > 0 else 0

        # Get volatility for probability estimation
        volatility = indicators.get("volatility_1h", 0.001)  # Default 0.1%
        if volatility <= 0:
            volatility = 0.001

        # Expected movement based on volatility and time
        # Roughly: expected_move = volatility * sqrt(time_in_hours) * price
        time_hours = time_to_expiry / 3600
        expected_move = volatility * (time_hours ** 0.5) * btc_price

        # Probability estimation using simple model
        if position == "inside":
            # Already in range - probability depends on staying in range
            # Higher prob if range is wide relative to expected movement
            stay_prob = min(0.90, 0.5 + (range_width / (expected_move * 4)))
            prob_yes = stay_prob
        else:
            # Outside range - need to move INTO the range
            # Lower probability the further away we are
            if expected_move > 0:
                # Simplified: probability decays with distance relative to expected move
                moves_needed = distance_to_range / expected_move
                # Probability of reaching range decreases exponentially with distance
                reach_prob = max(0.01, 0.5 * (0.5 ** moves_needed))

                # Also consider: even if we reach range, we need to END there
                # Reduce further if far from range
                end_in_range_factor = min(1.0, range_width / (expected_move * 2))
                prob_yes = reach_prob * end_in_range_factor
            else:
                prob_yes = 0.01  # Very unlikely without volatility

        # Trend adjustment - if trending toward range, increase probability
        trend = indicators.get("trend_1h", 0)
        if position == "below" and trend > 0:  # Price below range, trending up
            prob_yes = min(0.90, prob_yes * (1 + trend * 5))
        elif position == "above" and trend < 0:  # Price above range, trending down
            prob_yes = min(0.90, prob_yes * (1 + abs(trend) * 5))
        elif position == "inside":
            # If inside and trending away from range center, reduce prob
            if (btc_price < range_midpoint and trend < 0) or (btc_price > range_midpoint and trend > 0):
                prob_yes = prob_yes * 0.9

        # Clamp probability
        prob_yes = max(0.01, min(0.95, prob_yes))

        # For range markets, we recommend YES if prob > market price, NO otherwise
        # But the prediction field indicates whether we think YES will win
        prediction = "in_range" if prob_yes > 0.5 else "out_of_range"

        return prediction, prob_yes

    # LEGACY DIRECTIONAL MARKET LOGIC (above/below)
    distance = btc_price - strike_price
    distance_pct = distance / strike_price

    # Time factor - more uncertainty with more time
    time_factor = min(1.0, time_to_expiry / 3600)  # normalize to 1 hour

    # Trend influence
    trend = indicators.get("trend_1h", 0)

    # Simple probability estimate
    if distance > 0:  # Currently above strike
        base_prob = 0.5 + min(0.4, distance_pct * 10)  # max 90%
        trend_adj = trend * 0.1
        prob_above = min(0.95, max(0.05, base_prob + trend_adj))
    else:  # Currently below strike
        base_prob = 0.5 - min(0.4, abs(distance_pct) * 10)
        trend_adj = trend * 0.1
        prob_above = min(0.95, max(0.05, base_prob + trend_adj))

    # Adjust for time - more reversion to 50% with more time
    prob_above = prob_above * (1 - time_factor * 0.3) + 0.5 * (time_factor * 0.3)

    if strike_type == "above":
        prediction = "above" if prob_above > 0.5 else "below"
        prob = prob_above
    else:  # strike_type == "below"
        prediction = "below" if prob_above < 0.5 else "above"
        prob = 1 - prob_above

    return prediction, prob


def log_prediction(
    db_path: Path,
    market: Dict,
    btc_price: float,
    prediction: str,
    model_prob: float,
    features: Dict,
    indicators: Dict,
) -> int:
    """Log a prediction to the database."""
    conn = sqlite3.connect(db_path)
    c = conn.cursor()

    # Add range columns if they don't exist (migration)
    try:
        c.execute("ALTER TABLE hourly_predictions ADD COLUMN range_lower REAL")
    except sqlite3.OperationalError:
        pass  # Column already exists
    try:
        c.execute("ALTER TABLE hourly_predictions ADD COLUMN range_upper REAL")
    except sqlite3.OperationalError:
        pass  # Column already exists

    is_range_market = market.get("market_type") == "range"

    if is_range_market:
        range_lower = market.get("range_lower", 0)
        range_upper = market.get("range_upper", 0)
        distance_to_range = market.get("distance_to_range", 0)
        distance_pct = distance_to_range / btc_price if btc_price > 0 else 0
    else:
        range_lower = None
        range_upper = None
        distance_to_range = btc_price - market["strike_price"]
        distance_pct = distance_to_range / market["strike_price"] if market["strike_price"] > 0 else 0

    # Calculate implied prob and edge
    yes_ask = market.get("yes_ask", 50) / 100
    no_ask = market.get("no_ask", 50) / 100
    yes_bid = market.get("yes_bid", 50) / 100

    if is_range_market:
        # For range markets, implied_prob is the yes_ask (probability of landing in range)
        implied_prob = yes_ask
        # Edge calculation with 7% fee on winnings
        fee_drag = KALSHI_FEE_RATE * (1 - yes_ask) * model_prob
        edge = model_prob - yes_ask - fee_drag
    elif market["strike_type"] == "above":
        implied_prob = yes_ask
        entry_price = yes_ask if prediction == "above" else no_ask
        fee_drag = KALSHI_FEE_RATE * (1 - entry_price) * model_prob
        edge = model_prob - entry_price - fee_drag
    else:
        implied_prob = no_ask
        entry_price = no_ask if prediction == "below" else yes_ask
        fee_drag = KALSHI_FEE_RATE * (1 - entry_price) * model_prob
        edge = model_prob - entry_price - fee_drag

    spread = yes_ask - (yes_bid or 0)

    c.execute("""
        INSERT INTO hourly_predictions (
            market_ticker, expiry_time, strike_price, strike_type,
            range_lower, range_upper,
            btc_price_at_prediction, distance_to_strike, distance_pct,
            time_to_expiry_sec, yes_ask, yes_bid, no_ask, no_bid,
            implied_prob, spread, volume, open_interest,
            prediction, model_prob, edge, features_json,
            volatility_1h, volatility_24h, trend_1h, trend_4h,
            rsi_14, bb_position, vwap_distance
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        market["ticker"],
        market["expiry_time"].isoformat(),
        market["strike_price"],
        market["strike_type"],
        range_lower,
        range_upper,
        btc_price,
        distance_to_range,
        distance_pct,
        int(market["time_to_expiry"]),
        market.get("yes_ask"),
        market.get("yes_bid"),
        market.get("no_ask"),
        market.get("no_bid"),
        implied_prob,
        spread,
        market.get("volume", 0),
        market.get("open_interest", 0),
        prediction,
        model_prob,
        edge,
        json.dumps(features),
        indicators.get("volatility_1h"),
        indicators.get("volatility_24h"),
        indicators.get("trend_1h"),
        indicators.get("trend_4h"),
        indicators.get("rsi_14"),
        indicators.get("bb_position"),
        indicators.get("vwap_distance"),
    ))

    pred_id = c.lastrowid
    conn.commit()
    conn.close()

    return pred_id


def log_paper_trade(
    db_path: Path,
    prediction_id: int,
    market: Dict,
    side: str,
    entry_price: float,
    model_prob: float,
    edge: float,
) -> int:
    """Log a paper trade."""
    conn = sqlite3.connect(db_path)
    c = conn.cursor()

    c.execute("""
        INSERT INTO hourly_paper_trades (
            prediction_id, market_ticker, side, entry_price,
            model_prob, edge, expiry_time, strike_price
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        prediction_id,
        market["ticker"],
        side,
        entry_price,
        model_prob,
        edge,
        market["expiry_time"].isoformat(),
        market["strike_price"],
    ))

    trade_id = c.lastrowid
    conn.commit()
    conn.close()

    return trade_id


def log_real_trade(
    db_path: Path,
    prediction_id: int,
    market: Dict,
    order_id: str,
    side: str,
    entry_price: float,
    contracts: int,
    model_prob: float,
    edge: float,
    btc_price: float,
) -> int:
    """Log a real trade."""
    conn = sqlite3.connect(db_path)
    c = conn.cursor()

    c.execute("""
        INSERT INTO hourly_real_trades (
            prediction_id, market_ticker, order_id, side, entry_price,
            contracts, model_prob, edge, expiry_time, strike_price, btc_price_at_trade
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        prediction_id,
        market["ticker"],
        order_id,
        side,
        entry_price,
        contracts,
        model_prob,
        edge,
        market["expiry_time"].isoformat(),
        market["strike_price"],
        btc_price,
    ))

    trade_id = c.lastrowid
    conn.commit()
    conn.close()

    return trade_id


def check_settlements(db_path: Path, current_price: float) -> List[Dict]:
    """Check and settle expired predictions and trades."""
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    c = conn.cursor()

    now = datetime.now(timezone.utc)
    settled = []

    # Get unsettled predictions past expiry
    c.execute("""
        SELECT * FROM hourly_predictions
        WHERE settled = 0 AND expiry_time < ?
    """, (now.isoformat(),))

    for pred in c.fetchall():
        strike = pred["strike_price"]
        strike_type = pred["strike_type"]

        # Determine actual outcome based on market type
        if strike_type == "range":
            # For range markets, check if price is within the range
            # Try to get range bounds from database
            range_lower = pred["range_lower"] if "range_lower" in pred.keys() else None
            range_upper = pred["range_upper"] if "range_upper" in pred.keys() else None

            # Fallback: reconstruct from strike_price (midpoint) if bounds not stored
            if range_lower is None or range_upper is None:
                range_width = 250
                range_lower = strike - range_width / 2
                range_upper = strike + range_width / 2

            if range_lower <= current_price <= range_upper:
                actual = "in_range"  # YES wins
            else:
                actual = "out_of_range"  # NO wins

            # Check if our prediction was correct
            # prediction field stores "in_range" or "out_of_range"
            correct = 1 if pred["prediction"] == actual else 0
        else:
            # Legacy directional markets
            if current_price > strike:
                actual = "above"
            else:
                actual = "below"
            correct = 1 if pred["prediction"] == actual else 0

        # Calculate P&L if we had traded
        entry_price = pred["yes_ask"] / 100 if pred["yes_ask"] else 0.5
        if pred["prediction"] == actual:
            pnl = (1 - entry_price) * 0.93  # 7% fee
        else:
            pnl = -entry_price

        # Update prediction
        c.execute("""
            UPDATE hourly_predictions SET
                settled = 1,
                final_btc_price = ?,
                actual_outcome = ?,
                prediction_correct = ?,
                pnl_if_traded = ?,
                settled_at = ?
            WHERE id = ?
        """, (current_price, actual, correct, pnl, now.isoformat(), pred["id"]))

        settled.append({
            "id": pred["id"],
            "ticker": pred["market_ticker"],
            "prediction": pred["prediction"],
            "actual": actual,
            "correct": correct,
            "pnl": pnl,
        })

    # Settle paper trades
    c.execute("""
        SELECT * FROM hourly_paper_trades
        WHERE settled = 0 AND expiry_time < ?
    """, (now.isoformat(),))

    for trade in c.fetchall():
        # Get the prediction to know outcome
        c.execute("SELECT actual_outcome, strike_type FROM hourly_predictions WHERE id = ?", (trade["prediction_id"],))
        pred_row = c.fetchone()
        if not pred_row:
            continue

        actual = pred_row["actual_outcome"]
        strike_type = pred_row["strike_type"] if len(pred_row) > 1 else "above"
        side = trade["side"]
        entry = trade["entry_price"]

        # Determine win/loss based on market type
        if strike_type == "range":
            # Range market: YES wins if in_range, NO wins if out_of_range
            if (side == "yes" and actual == "in_range") or (side == "no" and actual == "out_of_range"):
                outcome = "WIN"
                pnl = (1 - entry) * 0.93
            else:
                outcome = "LOSS"
                pnl = -entry
        else:
            # Legacy directional market
            if (side == "yes" and actual == "above") or (side == "no" and actual == "below"):
                outcome = "WIN"
                pnl = (1 - entry) * 0.93
            else:
                outcome = "LOSS"
                pnl = -entry

        c.execute("""
            UPDATE hourly_paper_trades SET
                settled = 1,
                outcome = ?,
                pnl = ?,
                settled_at = ?
            WHERE id = ?
        """, (outcome, pnl, now.isoformat(), trade["id"]))

    # Settle real trades
    c.execute("""
        SELECT * FROM hourly_real_trades
        WHERE settled = 0 AND expiry_time < ?
    """, (now.isoformat(),))

    for trade in c.fetchall():
        c.execute("SELECT actual_outcome, strike_type FROM hourly_predictions WHERE id = ?", (trade["prediction_id"],))
        pred_row = c.fetchone()
        if not pred_row:
            continue

        actual = pred_row["actual_outcome"]
        strike_type = pred_row["strike_type"] if len(pred_row) > 1 else "above"
        side = trade["side"]
        entry = trade["entry_price"]
        contracts = trade["contracts"]

        # Determine win/loss based on market type
        if strike_type == "range":
            # Range market: YES wins if in_range, NO wins if out_of_range
            if (side == "yes" and actual == "in_range") or (side == "no" and actual == "out_of_range"):
                outcome = "WIN"
                pnl = (1 - entry) * contracts * (1 - KALSHI_FEE_RATE)
            else:
                outcome = "LOSS"
                pnl = -entry * contracts
        else:
            # Legacy directional market
            if (side == "yes" and actual == "above") or (side == "no" and actual == "below"):
                outcome = "WIN"
                pnl = (1 - entry) * contracts * (1 - KALSHI_FEE_RATE)
            else:
                outcome = "LOSS"
                pnl = -entry * contracts

        c.execute("""
            UPDATE hourly_real_trades SET
                settled = 1,
                outcome = ?,
                pnl = ?,
                settled_at = ?
            WHERE id = ?
        """, (outcome, pnl, now.isoformat(), trade["id"]))

        settled.append({
            "id": trade["id"],
            "ticker": trade["market_ticker"],
            "side": side,
            "contracts": contracts,
            "outcome": outcome,
            "pnl": pnl,
            "is_real": True,
        })

        # Get cumulative stats for notification
        c.execute("SELECT SUM(pnl), COUNT(*), SUM(CASE WHEN outcome='WIN' THEN 1 ELSE 0 END) FROM hourly_real_trades WHERE settled=1")
        stats_row = c.fetchone()
        total_pnl = (stats_row[0] or 0) + pnl  # Include this trade
        total_trades = (stats_row[1] or 0) + 1
        total_wins = (stats_row[2] or 0) + (1 if outcome == "WIN" else 0)

        # Send improved Discord notification for real trade settlement
        # Get range bounds from prediction
        c.execute("SELECT range_lower, range_upper, model_prob FROM hourly_predictions WHERE id = ?", (trade["prediction_id"],))
        pred_info = c.fetchone()
        range_lower = pred_info[0] if pred_info and pred_info[0] else trade.get("strike_price", 0) - 125
        range_upper = pred_info[1] if pred_info and pred_info[1] else trade.get("strike_price", 0) + 125
        trade_model_prob = pred_info[2] if pred_info and pred_info[2] else 0.5

        send_hourly_settlement_alert(
            range_lower=range_lower,
            range_upper=range_upper,
            settlement_price=current_price,
            side=side,
            outcome=outcome,
            pnl=pnl,
            entry_price=entry,
            contracts=contracts,
            model_prob=trade_model_prob,
            cumulative_pnl=total_pnl,
            total_wins=total_wins,
            total_losses=total_trades - total_wins,
            webhook_url=DISCORD_WEBHOOK,
        )

    conn.commit()
    conn.close()

    return settled


def get_hourly_stats(db_path: Path) -> Dict:
    """Get statistics on hourly predictions."""
    conn = sqlite3.connect(db_path)
    c = conn.cursor()

    c.execute("SELECT COUNT(*) FROM hourly_predictions")
    total = c.fetchone()[0]

    c.execute("SELECT COUNT(*) FROM hourly_predictions WHERE settled = 1")
    settled = c.fetchone()[0]

    c.execute("SELECT COUNT(*) FROM hourly_predictions WHERE prediction_correct = 1")
    correct = c.fetchone()[0]

    c.execute("SELECT SUM(pnl_if_traded) FROM hourly_predictions WHERE settled = 1")
    total_pnl = c.fetchone()[0] or 0

    c.execute("SELECT COUNT(*) FROM hourly_paper_trades WHERE settled = 1 AND outcome = 'WIN'")
    wins = c.fetchone()[0]

    c.execute("SELECT COUNT(*) FROM hourly_paper_trades WHERE settled = 1 AND outcome = 'LOSS'")
    losses = c.fetchone()[0]

    c.execute("SELECT SUM(pnl) FROM hourly_paper_trades WHERE settled = 1")
    trade_pnl = c.fetchone()[0] or 0

    # Real trade stats
    c.execute("SELECT COUNT(*) FROM hourly_real_trades WHERE settled = 1 AND outcome = 'WIN'")
    real_wins = c.fetchone()[0]

    c.execute("SELECT COUNT(*) FROM hourly_real_trades WHERE settled = 1 AND outcome = 'LOSS'")
    real_losses = c.fetchone()[0]

    c.execute("SELECT SUM(pnl) FROM hourly_real_trades WHERE settled = 1")
    real_pnl = c.fetchone()[0] or 0

    c.execute("SELECT COUNT(*) FROM hourly_real_trades WHERE settled = 0")
    real_pending = c.fetchone()[0]

    conn.close()

    accuracy = correct / settled if settled > 0 else 0

    return {
        "total_predictions": total,
        "settled": settled,
        "pending": total - settled,
        "correct": correct,
        "accuracy": accuracy,
        "theoretical_pnl": total_pnl,
        "paper_wins": wins,
        "paper_losses": losses,
        "paper_pnl": trade_pnl,
        "real_wins": real_wins,
        "real_losses": real_losses,
        "real_pnl": real_pnl,
        "real_pending": real_pending,
    }


def send_hourly_alert(
    market: Dict,
    btc_price: float,
    prediction: str,
    model_prob: float,
    edge: float,
    traded: bool = False,
):
    """Send Discord alert for hourly prediction."""
    if not DISCORD_WEBHOOK:
        return

    title = "📊 [HOURLY] " + ("Paper Trade" if traded else "Prediction")
    color = 0x3498DB  # Blue

    tte_min = int(market["time_to_expiry"] / 60)
    distance = btc_price - market["strike_price"]

    fields = [
        {"name": "BTC Price", "value": f"${btc_price:,.2f}", "inline": True},
        {"name": "Strike", "value": f"${market['strike_price']:,.2f}", "inline": True},
        {"name": "Distance", "value": f"${distance:+,.2f}", "inline": True},
        {"name": "Prediction", "value": prediction.upper(), "inline": True},
        {"name": "Confidence", "value": f"{model_prob*100:.1f}%", "inline": True},
        {"name": "Edge", "value": f"{edge*100:+.1f}%", "inline": True},
        {"name": "Expires In", "value": f"{tte_min}m", "inline": True},
    ]

    send_discord_alert(
        title=title,
        description=f"Market: {market['ticker']}",
        color=color,
        fields=fields,
        webhook_url=DISCORD_WEBHOOK,
    )


def run_hourly_scan(
    client: KalshiAuthClient,
    db_path: Path,
    min_edge: float = 0.05,
    min_edge_paper: float = 0.02,  # Lower threshold for paper trades to collect more data
    logged_tickers: Set[str] = None,
    traded_tickers: Set[str] = None,
    live_mode: bool = False,
    max_contracts: int = 5,
    last_live_trade_hour: Optional[int] = None,
) -> Optional[int]:
    """Run a single scan cycle for hourly markets.

    Returns the hour of the last live trade (for tracking 1 trade per hour).
    """
    if logged_tickers is None:
        logged_tickers = set()
    if traded_tickers is None:
        traded_tickers = set()

    current_hour = datetime.now(timezone.utc).hour

    print(f"\n{'='*60}")
    print(f"[HOURLY] {datetime.now().strftime('%H:%M:%S')}")

    # Get current BTC price
    try:
        btc_price = fetch_current_price()
        chainlink_price = get_chainlink_btc_price()
        settlement_price = chainlink_price or btc_price
        print(f"[HOURLY] BTC: ${btc_price:,.2f}")
    except Exception as e:
        print(f"[HOURLY] Price error: {e}")
        return

    # Check settlements first
    settled = check_settlements(db_path, settlement_price)
    if settled:
        for s in settled:
            result = "✓" if s["correct"] else "✗"
            print(f"[SETTLE] {s['ticker']}: {result} (pred={s['prediction']}, actual={s['actual']}, pnl=${s['pnl']:+.2f})")

    # Fetch candles for features
    try:
        candles = fetch_klines(interval="1m", limit=100)
        candle_dicts = [
            {"open": c.open, "high": c.high, "low": c.low, "close": c.close, "volume": c.volume}
            for c in candles
        ]
        current_hour = datetime.now(timezone.utc).hour
        features = compute_all_features(candle_dicts, current_hour)
        indicators = compute_additional_indicators(candle_dicts)
    except Exception as e:
        print(f"[HOURLY] Feature error: {e}")
        features = {}
        indicators = {}

    # Fetch hourly markets
    markets = fetch_hourly_markets(client)
    print(f"[HOURLY] Found {len(markets)} hourly markets")

    if not markets:
        return

    # Filter to markets where BTC could realistically land in the range
    near_markets = []
    for m in markets:
        if m.get("market_type") == "range":
            range_lower = m.get("range_lower", 0)
            range_upper = m.get("range_upper", 0)
            range_mid = (range_lower + range_upper) / 2

            # Distance from current price to range midpoint
            distance_pct = abs(btc_price - range_mid) / btc_price if btc_price > 0 else 1

            # Only consider ranges within ~3% of current price
            # (BTC could realistically move into this range)
            if distance_pct < 0.03:
                # Calculate how far from range we are
                if btc_price < range_lower:
                    m["distance_to_range"] = range_lower - btc_price
                    m["position_vs_range"] = "below"
                elif btc_price > range_upper:
                    m["distance_to_range"] = btc_price - range_upper
                    m["position_vs_range"] = "above"
                else:
                    m["distance_to_range"] = 0
                    m["position_vs_range"] = "inside"

                near_markets.append(m)
        else:
            # Legacy above/below markets
            distance_pct = abs(btc_price - m["strike_price"]) / m["strike_price"]
            if distance_pct < 0.02:
                near_markets.append(m)

    print(f"[HOURLY] {len(near_markets)} markets within range of current price")

    # Collect all tradeable opportunities first
    trade_opportunities = []

    for market in near_markets[:10]:  # Limit to 10 markets per scan
        if market["ticker"] in logged_tickers:
            continue

        # Make prediction
        prediction, model_prob = make_prediction(
            btc_price=btc_price,
            strike_price=market["strike_price"],
            strike_type=market["strike_type"],
            time_to_expiry=market["time_to_expiry"],
            features=features,
            indicators=indicators,
            range_lower=market.get("range_lower"),
            range_upper=market.get("range_upper"),
        )

        # Calculate entry price and edge for range markets
        yes_ask = (market.get("yes_ask") or 50) / 100
        no_ask = (market.get("no_ask") or 50) / 100

        is_range_market = market.get("market_type") == "range"

        if is_range_market:
            # Edge calculation: expected profit minus cost, accounting for 7% fee on winnings
            # If we buy YES at price P and win, we get (1-P)*0.93, so breakeven prob = P / (P + (1-P)*0.93)
            # Simplified: edge = model_prob * (1-P) * 0.93 - (1-model_prob) * P  ... but we use simpler approximation
            # Edge = model_prob - price - fee_drag, where fee_drag ≈ 0.07 * (1 - price) * model_prob
            yes_fee_drag = KALSHI_FEE_RATE * (1 - yes_ask) * model_prob
            no_fee_drag = KALSHI_FEE_RATE * (1 - no_ask) * (1 - model_prob)
            yes_edge = model_prob - yes_ask - yes_fee_drag
            no_edge = (1 - model_prob) - no_ask - no_fee_drag

            if yes_edge > no_edge and yes_edge > 0:
                side_decision = "yes"
                entry_price = yes_ask
                edge = yes_edge
            elif no_edge > 0:
                side_decision = "no"
                entry_price = no_ask
                edge = no_edge
            else:
                side_decision = None
                entry_price = yes_ask
                edge = max(yes_edge, no_edge)
        else:
            if prediction == "above":
                entry_price = yes_ask
                side_decision = "yes"
                fee_drag = KALSHI_FEE_RATE * (1 - entry_price) * model_prob
            else:
                entry_price = no_ask
                side_decision = "no"
                fee_drag = KALSHI_FEE_RATE * (1 - entry_price) * (1 - model_prob)
            edge = model_prob - entry_price - fee_drag

        # Log prediction
        pred_id = log_prediction(
            db_path=db_path,
            market=market,
            btc_price=btc_price,
            prediction=prediction,
            model_prob=model_prob,
            features=features,
            indicators=indicators,
        )

        logged_tickers.add(market["ticker"])

        tte_min = int(market["time_to_expiry"] / 60)

        if is_range_market:
            range_lower = market.get("range_lower", 0)
            range_upper = market.get("range_upper", 0)
            position = market.get("position_vs_range", "unknown")
            dist_to_range = market.get("distance_to_range", 0)
            print(f"[RANGE] {market['ticker']}: ${range_lower:,.0f}-${range_upper:,.0f}")
            print(f"        BTC=${btc_price:,.0f} ({position}, dist=${dist_to_range:,.0f}), prob_in_range={model_prob*100:.1f}%, exp={tte_min}m")
            print(f"        YES@{yes_ask*100:.0f}¢ edge={yes_edge*100:+.1f}%, NO@{no_ask*100:.0f}¢ edge={no_edge*100:+.1f}%")
        else:
            distance = btc_price - market["strike_price"]
            print(f"[PRED] {market['ticker']}: {prediction.upper()} @ {model_prob*100:.1f}% (strike=${market['strike_price']:,.0f}, dist=${distance:+,.0f}, exp={tte_min}m)")

        # Collect tradeable opportunities (use lower paper threshold to collect more data)
        if edge >= min_edge_paper and side_decision is not None:
            # Calculate EV: edge * potential profit per dollar risked
            potential_profit = (1 - entry_price) * (1 - KALSHI_FEE_RATE)
            ev = edge * potential_profit / entry_price if entry_price > 0 else 0

            trade_opportunities.append({
                "market": market,
                "pred_id": pred_id,
                "side": side_decision,
                "entry_price": entry_price,
                "edge": edge,
                "ev": ev,
                "model_prob": model_prob,
                "is_range_market": is_range_market,
                "meets_live_threshold": edge >= min_edge,  # Track if meets higher live threshold
            })

    # Sort by EV (highest first)
    trade_opportunities.sort(key=lambda x: x["ev"], reverse=True)

    # IMPORTANT: Portfolio constraint for range markets
    # Multiple ranges for same expiry are mutually exclusive (only one can win)
    # Rule: total cost across all positions for same expiry must be < $1
    # Example: YES@35¢ + YES@35¢ = 70¢ total → can profit 30¢ if either wins ✓
    # Example: YES@55¢ + YES@55¢ = $1.10 total → guaranteed loss ✗
    expiry_costs: Dict[str, float] = {}
    for opp in trade_opportunities:
        expiry = opp["market"].get("expiry_time", opp["market"]["ticker"])
        current_cost = expiry_costs.get(expiry, 0)
        new_cost = current_cost + opp["entry_price"]

        if new_cost < 0.93:  # Max payout after 7% fee is ~93¢, so total cost must be below this
            expiry_costs[expiry] = new_cost
            opp["skipped_portfolio"] = False
            opp["portfolio_cost"] = new_cost
        else:
            # Would exceed $1 total - skip this trade
            opp["skipped_portfolio"] = True
            opp["portfolio_cost"] = new_cost

    # Determine if we can do a live trade this hour
    can_live_trade = live_mode and (last_live_trade_hour != current_hour)
    live_trade_done = False

    # Process trades: highest EV gets live trade (if allowed), rest are paper
    # Track which expiries we've traded to prevent conflicting positions
    for i, opp in enumerate(trade_opportunities):
        market = opp["market"]
        side = opp["side"]
        entry_price = opp["entry_price"]
        edge = opp["edge"]
        ev = opp["ev"]
        pred_id = opp["pred_id"]
        model_prob = opp["model_prob"]

        # Always log paper trade for data collection
        paper_id = log_paper_trade(
            db_path=db_path,
            prediction_id=pred_id,
            market=market,
            side=side,
            entry_price=entry_price,
            model_prob=model_prob,
            edge=edge,
        )

        # Live trade: only the first (highest EV) opportunity, once per hour
        # Price must be between 21-95 cents for live trades (paper trade outside this range)
        # Edge must meet higher live threshold (not just paper threshold)
        # Must not be skipped due to portfolio constraint (multiple ranges same expiry)
        price_in_range = 0.21 <= entry_price <= 0.95
        meets_live_threshold = opp.get("meets_live_threshold", edge >= min_edge)
        skipped_portfolio = opp.get("skipped_portfolio", False)
        if can_live_trade and not live_trade_done and market["ticker"] not in traded_tickers and price_in_range and meets_live_threshold and not skipped_portfolio:
            print(f"[LIVE] Best EV trade: {market['ticker']} (EV={ev:.3f}, edge={edge*100:+.1f}%)")

            # Get balance for Kelly sizing
            try:
                balance_resp = client.get_balance()
                bankroll = balance_resp.get('balance', 0) / 100.0
            except:
                bankroll = 0

            # Calculate position size
            kelly_contracts = calculate_half_kelly_contracts(
                bankroll=bankroll,
                edge=edge,
                entry_price=entry_price,
                min_contracts=1,
            )
            contracts = min(kelly_contracts, max_contracts)

            print(f"[LIVE] Placing order: {contracts}x {side.upper()} @ {int(entry_price*100)}c (half-Kelly: {kelly_contracts}, bankroll: ${bankroll:.2f})")

            filled, order_id, fill_price = place_hourly_order(
                client=client,
                ticker=market["ticker"],
                side=side,
                limit_price_cents=int(entry_price * 100),
                contracts=contracts,
            )

            if filled:
                traded_tickers.add(market["ticker"])
                actual_entry = fill_price / 100.0
                live_trade_done = True
                last_live_trade_hour = current_hour

                log_real_trade(
                    db_path=db_path,
                    prediction_id=pred_id,
                    market=market,
                    order_id=order_id,
                    side=side,
                    entry_price=actual_entry,
                    contracts=contracts,
                    model_prob=model_prob,
                    edge=edge,
                    btc_price=btc_price,
                )

                print(f"[LIVE] ✓ Filled {contracts}x {side.upper()} @ {fill_price}c on {market['ticker']}")

                # Get cumulative stats for notification
                conn = sqlite3.connect(db_path)
                c = conn.cursor()
                c.execute("SELECT SUM(pnl), COUNT(*) FROM hourly_real_trades WHERE settled=1")
                stats_row = c.fetchone()
                cum_pnl = stats_row[0] or 0
                total_trades_count = stats_row[1] or 0
                c.execute("SELECT SUM(CASE WHEN outcome='WIN' THEN 1 ELSE 0 END) FROM hourly_real_trades WHERE settled=1")
                wins_row = c.fetchone()
                total_wins_count = wins_row[0] or 0
                conn.close()

                # Discord alert with full context
                send_hourly_trade_alert(
                    range_lower=market.get("range_lower", 0),
                    range_upper=market.get("range_upper", 0),
                    btc_price=btc_price,
                    side=side,
                    entry_price=actual_entry,
                    contracts=contracts,
                    edge=edge,
                    ev_score=ev,
                    model_prob=model_prob,
                    time_to_expiry_min=int(market["time_to_expiry"] / 60),
                    position_vs_range=market.get("position_vs_range", "unknown"),
                    total_opportunities=len(trade_opportunities),
                    cumulative_pnl=cum_pnl,
                    total_record=f"{total_wins_count}W/{total_trades_count - total_wins_count}L",
                    bankroll=bankroll,
                    webhook_url=DISCORD_WEBHOOK,
                )
            else:
                print(f"[LIVE] ✗ Order not filled")
        else:
            # Paper trade (either no live mode, already traded this hour, price out of range, low edge, portfolio constraint, or not highest EV)
            if not live_mode:
                reason = "paper only"
            elif live_trade_done:
                reason = "already traded this hour"
            elif skipped_portfolio:
                portfolio_cost = opp.get("portfolio_cost", 0)
                reason = f"would exceed $1 total cost ({portfolio_cost*100:.0f}¢)"
            elif not price_in_range:
                reason = f"price {entry_price*100:.0f}¢ outside 21-95¢ range"
            elif not meets_live_threshold:
                reason = f"edge {edge*100:.1f}% below live threshold"
            else:
                reason = "not highest EV"
            print(f"[PAPER] Trade #{paper_id}: {side.upper()} @ {entry_price*100:.0f}¢ (edge={edge*100:+.1f}%, EV={ev:.3f}) [{reason}]")

    # Summary of opportunities this scan
    if trade_opportunities:
        live_eligible = sum(1 for o in trade_opportunities if o.get('meets_live_threshold', True))
        print(f"\n[SCAN] Found {len(trade_opportunities)} trade opportunities ({live_eligible} meet live threshold):")
        for i, opp in enumerate(trade_opportunities[:5]):
            opp_price = opp['entry_price']
            opp_in_range = 0.21 <= opp_price <= 0.95
            opp_meets_live = opp.get('meets_live_threshold', True)
            if i == 0 and live_trade_done:
                marker = "→ LIVE"
            elif can_live_trade and not opp_in_range:
                marker = "  skip (price)"
            elif can_live_trade and not opp_meets_live:
                marker = "  skip (edge)"
            else:
                marker = "  paper"
            print(f"  {marker} #{i+1}: {opp['market']['ticker']} EV={opp['ev']:.3f} edge={opp['edge']*100:+.1f}% @{opp_price*100:.0f}¢")

    # Print stats
    stats = get_hourly_stats(db_path)
    print(f"[STATS] Predictions: {stats['total_predictions']} ({stats['pending']} pending)")
    if stats['settled'] > 0:
        print(f"[STATS] Accuracy: {stats['accuracy']*100:.1f}% ({stats['correct']}/{stats['settled']})")
        print(f"[STATS] Paper P&L: ${stats['paper_pnl']:+.2f} ({stats['paper_wins']}W/{stats['paper_losses']}L)")
    if stats['real_wins'] + stats['real_losses'] > 0 or stats['real_pending'] > 0:
        print(f"[STATS] Real P&L: ${stats['real_pnl']:+.2f} ({stats['real_wins']}W/{stats['real_losses']}L, {stats['real_pending']} pending)")

    # Return the hour we made a live trade (for tracking)
    return current_hour if live_trade_done else last_live_trade_hour


def main():
    parser = argparse.ArgumentParser(description="BTC 1-Hour Trading Scanner")
    parser.add_argument("--interval", type=int, default=60, help="Scan interval in seconds")
    parser.add_argument("--min-edge", type=float, default=0.05, help="Minimum edge for LIVE trading")
    parser.add_argument("--min-edge-paper", type=float, default=0.02, help="Minimum edge for paper trades (lower to collect more data)")
    parser.add_argument("--max-contracts", type=int, default=5, help="Max contracts per trade")
    parser.add_argument("--live", action="store_true", help="Enable live trading (real money)")
    parser.add_argument("--db", type=str, default=str(HOURLY_DB_PATH))
    args = parser.parse_args()

    db_path = Path(args.db)
    ensure_hourly_schema(db_path)

    mode = "LIVE TRADING" if args.live else "PAPER TRADING"
    print("=" * 60)
    print(f"BTC 1-HOUR {mode} SCANNER")
    print("=" * 60)
    print(f"Mode: {'🔴 LIVE' if args.live else '📝 PAPER'}")
    print(f"Scan interval: {args.interval}s")
    print(f"Min edge: {args.min_edge*100:.0f}% (live), {args.min_edge_paper*100:.0f}% (paper)")
    print(f"Max contracts: {args.max_contracts}")
    print(f"Database: {db_path}")
    print("=" * 60)

    # Initialize Kalshi client
    client = get_kalshi_client()
    if not client:
        print("[ERROR] Failed to initialize Kalshi client")
        sys.exit(1)

    try:
        balance = client.get_balance()
        print(f"Kalshi balance: ${balance.get('balance', 0) / 100:.2f}")
    except Exception as e:
        print(f"Balance check error: {e}")

    # Track logged and traded tickers
    logged_tickers: Set[str] = set()
    traded_tickers: Set[str] = set()
    last_live_trade_hour: Optional[int] = None  # Track which hour we last traded

    if args.live:
        print("\n⚠️  LIVE TRADING ENABLED - 1 real trade per hour (highest EV)")
        print("    All other opportunities will be paper traded for data collection")

    # Check initial calibration state
    cal_state = load_calibration_state()
    print(f"\nLast calibration: {cal_state.get('last_calibration_count', 0)} samples")
    print("ML recalibrates automatically when settled trades increase by 25%")
    print("\nStarting hourly scan (Ctrl+C to stop)...")

    try:
        while True:
            try:
                last_live_trade_hour = run_hourly_scan(
                    client=client,
                    db_path=db_path,
                    min_edge=args.min_edge,
                    min_edge_paper=args.min_edge_paper,
                    logged_tickers=logged_tickers,
                    traded_tickers=traded_tickers,
                    live_mode=args.live,
                    max_contracts=args.max_contracts,
                    last_live_trade_hour=last_live_trade_hour,
                )

                # Check if recalibration is needed (25% more data than last calibration)
                should_recal, current_count, last_count = check_recalibration_needed(db_path)
                if should_recal:
                    print(f"\n[CALIBRATE] Triggering recalibration: {current_count} samples (+{((current_count/last_count - 1)*100) if last_count > 0 else 100:.0f}% from last)")
                    run_recalibration(db_path)

            except Exception as e:
                print(f"[ERROR] Scan error: {e}")

            time.sleep(args.interval)

            # Clear logged tickers every hour to allow re-logging
            if len(logged_tickers) > 500:
                logged_tickers.clear()

    except KeyboardInterrupt:
        print("\n\nScanner stopped by user")

        # Print final stats
        stats = get_hourly_stats(db_path)
        print(f"\n{'='*60}")
        print("SESSION SUMMARY")
        print(f"{'='*60}")
        print(f"Total predictions: {stats['total_predictions']}")
        print(f"Settled: {stats['settled']}")
        print(f"Accuracy: {stats['accuracy']*100:.1f}%")
        print(f"Paper trades: {stats['paper_wins'] + stats['paper_losses']}")
        print(f"Paper P&L: ${stats['paper_pnl']:+.2f}")
        if stats['real_wins'] + stats['real_losses'] > 0:
            print(f"Real trades: {stats['real_wins'] + stats['real_losses']} ({stats['real_wins']}W/{stats['real_losses']}L)")
            print(f"Real P&L: ${stats['real_pnl']:+.2f}")


if __name__ == "__main__":
    main()
