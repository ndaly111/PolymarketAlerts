#!/usr/bin/env python3
"""
Backtest simulator using historical 15-min BTC data.

Simulates thousands of trades to find what actually predicts the right side.
"""

import requests
import time
from datetime import datetime, timedelta
from typing import List, Dict
import statistics


def fetch_15min_candles(days: int = 7) -> List[Dict]:
    """Fetch historical 15-min candles from Binance."""
    url = "https://api.binance.us/api/v3/klines"

    candles = []
    end_time = int(datetime.now().timestamp() * 1000)
    start_time = int((datetime.now() - timedelta(days=days)).timestamp() * 1000)

    print(f"Fetching {days} days of 15-min candles...")

    current = start_time
    while current < end_time:
        params = {
            "symbol": "BTCUSDT",
            "interval": "15m",
            "startTime": current,
            "limit": 1000,
        }

        try:
            resp = requests.get(url, params=params, timeout=10)
            resp.raise_for_status()
            data = resp.json()

            if not data:
                break

            for k in data:
                candles.append({
                    "open_time": k[0],
                    "open": float(k[1]),
                    "high": float(k[2]),
                    "low": float(k[3]),
                    "close": float(k[4]),
                    "volume": float(k[5]),
                })

            current = data[-1][0] + 1  # Next candle
            time.sleep(0.1)  # Rate limit

        except Exception as e:
            print(f"Error: {e}")
            break

    print(f"Fetched {len(candles)} candles")
    return candles


def simulate_trades(candles: List[Dict]) -> List[Dict]:
    """
    Simulate trades for each 15-min candle.

    For each candle, we simulate multiple strike prices and test outcomes.
    """
    trades = []

    for i, candle in enumerate(candles[:-1]):  # Skip last (incomplete)
        open_price = candle["open"]
        close_price = candle["close"]
        high = candle["high"]
        low = candle["low"]

        # Price movement
        price_change = close_price - open_price
        price_change_pct = (price_change / open_price) * 100

        # Volatility within candle
        candle_range = high - low
        candle_volatility = (candle_range / open_price) * 100

        # Previous candle momentum (if available)
        prev_momentum = 0
        prev_velocity = 0
        if i > 0:
            prev = candles[i-1]
            prev_momentum = prev["close"] - prev["open"]
            prev_velocity = (prev_momentum / prev["open"]) * 100

        # 5-candle momentum
        momentum_5 = 0
        if i >= 5:
            momentum_5 = (open_price - candles[i-5]["open"]) / candles[i-5]["open"] * 100

        # Simulate different strike prices around open
        for strike_offset_pct in [-0.5, -0.3, -0.2, -0.1, 0, 0.1, 0.2, 0.3, 0.5]:
            strike = open_price * (1 + strike_offset_pct / 100)

            # Distance from open to strike
            distance_pct = ((open_price - strike) / open_price) * 100

            # Position: is price above or below strike at start?
            is_above_strike = open_price > strike

            # Actual outcome: did price end above or below strike?
            ended_above = close_price > strike

            trades.append({
                "timestamp": candle["open_time"],
                "open": open_price,
                "close": close_price,
                "strike": strike,
                "distance_pct": distance_pct,
                "is_above_strike": is_above_strike,
                "ended_above": ended_above,
                "price_change_pct": price_change_pct,
                "candle_volatility": candle_volatility,
                "prev_velocity": prev_velocity,
                "momentum_5": momentum_5,
            })

    return trades


def test_strategies(trades: List[Dict]) -> Dict:
    """Test different prediction strategies."""

    strategies = {
        "always_up": {"correct": 0, "total": 0, "description": "Always predict UP"},
        "always_down": {"correct": 0, "total": 0, "description": "Always predict DOWN"},
        "follow_position": {"correct": 0, "total": 0, "description": "UP if above strike, DOWN if below"},
        "momentum": {"correct": 0, "total": 0, "description": "Follow prev candle direction"},
        "momentum_5": {"correct": 0, "total": 0, "description": "Follow 5-candle momentum"},
        "aligned_close": {"correct": 0, "total": 0, "description": "Position aligned + close (<0.2%)"},
        "aligned_far": {"correct": 0, "total": 0, "description": "Position aligned + far (>0.2%)"},
        "contrarian": {"correct": 0, "total": 0, "description": "Opposite of position"},
        "high_vol_follow": {"correct": 0, "total": 0, "description": "Follow position when vol > 0.3%"},
        "low_vol_stay": {"correct": 0, "total": 0, "description": "Stay in position when vol < 0.2%"},
    }

    for trade in trades:
        is_above = trade["is_above_strike"]
        ended_above = trade["ended_above"]
        distance = abs(trade["distance_pct"])
        prev_vel = trade["prev_velocity"]
        mom_5 = trade["momentum_5"]
        vol = trade["candle_volatility"]

        # Always UP
        strategies["always_up"]["total"] += 1
        if ended_above:
            strategies["always_up"]["correct"] += 1

        # Always DOWN
        strategies["always_down"]["total"] += 1
        if not ended_above:
            strategies["always_down"]["correct"] += 1

        # Follow position (UP if above, DOWN if below)
        strategies["follow_position"]["total"] += 1
        if is_above == ended_above:
            strategies["follow_position"]["correct"] += 1

        # Contrarian (opposite of position)
        strategies["contrarian"]["total"] += 1
        if is_above != ended_above:
            strategies["contrarian"]["correct"] += 1

        # Follow momentum
        strategies["momentum"]["total"] += 1
        pred_up = prev_vel > 0
        if pred_up == ended_above:
            strategies["momentum"]["correct"] += 1

        # Follow 5-candle momentum
        strategies["momentum_5"]["total"] += 1
        pred_up = mom_5 > 0
        if pred_up == ended_above:
            strategies["momentum_5"]["correct"] += 1

        # Aligned + close
        if distance < 0.2:
            strategies["aligned_close"]["total"] += 1
            if is_above == ended_above:
                strategies["aligned_close"]["correct"] += 1

        # Aligned + far
        if distance >= 0.2:
            strategies["aligned_far"]["total"] += 1
            if is_above == ended_above:
                strategies["aligned_far"]["correct"] += 1

        # High volatility - follow position
        if vol > 0.3:
            strategies["high_vol_follow"]["total"] += 1
            if is_above == ended_above:
                strategies["high_vol_follow"]["correct"] += 1

        # Low volatility - stay in position
        if vol < 0.2:
            strategies["low_vol_stay"]["total"] += 1
            if is_above == ended_above:
                strategies["low_vol_stay"]["correct"] += 1

    return strategies


def analyze_by_distance(trades: List[Dict]):
    """Analyze win rate by distance from strike."""

    buckets = {}

    for trade in trades:
        distance = abs(trade["distance_pct"])
        is_above = trade["is_above_strike"]
        ended_above = trade["ended_above"]

        # Bucket by distance
        if distance < 0.1:
            bucket = "< 0.1%"
        elif distance < 0.2:
            bucket = "0.1-0.2%"
        elif distance < 0.3:
            bucket = "0.2-0.3%"
        elif distance < 0.5:
            bucket = "0.3-0.5%"
        else:
            bucket = "> 0.5%"

        if bucket not in buckets:
            buckets[bucket] = {"aligned_wins": 0, "aligned_total": 0}

        buckets[bucket]["aligned_total"] += 1
        if is_above == ended_above:  # Aligned prediction correct
            buckets[bucket]["aligned_wins"] += 1

    return buckets


def analyze_by_momentum(trades: List[Dict]):
    """Analyze when momentum helps prediction."""

    results = {
        "momentum_aligned": {"wins": 0, "total": 0},
        "momentum_against": {"wins": 0, "total": 0},
        "strong_momentum_up": {"wins": 0, "total": 0},
        "strong_momentum_down": {"wins": 0, "total": 0},
    }

    for trade in trades:
        prev_vel = trade["prev_velocity"]
        is_above = trade["is_above_strike"]
        ended_above = trade["ended_above"]

        # Momentum aligned with position
        momentum_up = prev_vel > 0
        if momentum_up == is_above:  # Momentum confirms position
            results["momentum_aligned"]["total"] += 1
            if is_above == ended_above:
                results["momentum_aligned"]["wins"] += 1
        else:  # Momentum against position
            results["momentum_against"]["total"] += 1
            if is_above == ended_above:
                results["momentum_against"]["wins"] += 1

        # Strong momentum (> 0.2%)
        if prev_vel > 0.2:
            results["strong_momentum_up"]["total"] += 1
            if ended_above:
                results["strong_momentum_up"]["wins"] += 1
        elif prev_vel < -0.2:
            results["strong_momentum_down"]["total"] += 1
            if not ended_above:
                results["strong_momentum_down"]["wins"] += 1

    return results


def find_best_conditions(trades: List[Dict]):
    """Find conditions with highest win rates."""

    conditions = []

    for trade in trades:
        distance = abs(trade["distance_pct"])
        is_above = trade["is_above_strike"]
        ended_above = trade["ended_above"]
        prev_vel = trade["prev_velocity"]
        mom_5 = trade["momentum_5"]
        vol = trade["candle_volatility"]

        # Create condition tuple
        cond = (
            "above" if is_above else "below",
            "close" if distance < 0.2 else "mid" if distance < 0.4 else "far",
            "up_mom" if prev_vel > 0.1 else "down_mom" if prev_vel < -0.1 else "neutral",
        )

        conditions.append({
            "condition": cond,
            "aligned_correct": is_above == ended_above,
        })

    # Aggregate by condition
    cond_stats = {}
    for c in conditions:
        key = str(c["condition"])
        if key not in cond_stats:
            cond_stats[key] = {"wins": 0, "total": 0}
        cond_stats[key]["total"] += 1
        if c["aligned_correct"]:
            cond_stats[key]["wins"] += 1

    # Calculate win rates
    for key in cond_stats:
        stats = cond_stats[key]
        stats["win_rate"] = stats["wins"] / stats["total"] * 100 if stats["total"] > 0 else 0

    return cond_stats


if __name__ == "__main__":
    print("=" * 70)
    print("BTC 15-MIN BACKTEST SIMULATOR")
    print("=" * 70)

    # Fetch historical data
    candles = fetch_15min_candles(days=14)  # 2 weeks of data

    if len(candles) < 100:
        print("Not enough data!")
        exit(1)

    # Simulate trades
    print("\nSimulating trades...")
    trades = simulate_trades(candles)
    print(f"Generated {len(trades)} simulated trades")

    # Test strategies
    print("\n" + "=" * 70)
    print("STRATEGY COMPARISON")
    print("=" * 70)

    strategies = test_strategies(trades)

    # Sort by accuracy
    sorted_strats = sorted(
        strategies.items(),
        key=lambda x: x[1]["correct"] / x[1]["total"] if x[1]["total"] > 0 else 0,
        reverse=True
    )

    print(f"\n{'Strategy':<25} | {'Accuracy':>8} | {'Trades':>7} | Description")
    print("-" * 70)
    for name, stats in sorted_strats:
        if stats["total"] > 0:
            acc = stats["correct"] / stats["total"] * 100
            print(f"{name:<25} | {acc:>7.1f}% | {stats['total']:>7} | {stats['description']}")

    # Analyze by distance
    print("\n" + "=" * 70)
    print("WIN RATE BY DISTANCE (following position)")
    print("=" * 70)

    distance_stats = analyze_by_distance(trades)
    print(f"\n{'Distance':<12} | {'Win Rate':>8} | {'Trades':>7}")
    print("-" * 35)
    for bucket in sorted(distance_stats.keys()):
        stats = distance_stats[bucket]
        wr = stats["aligned_wins"] / stats["aligned_total"] * 100 if stats["aligned_total"] > 0 else 0
        print(f"{bucket:<12} | {wr:>7.1f}% | {stats['aligned_total']:>7}")

    # Analyze momentum
    print("\n" + "=" * 70)
    print("MOMENTUM ANALYSIS")
    print("=" * 70)

    mom_stats = analyze_by_momentum(trades)
    print(f"\n{'Condition':<25} | {'Win Rate':>8} | {'Trades':>7}")
    print("-" * 45)
    for name, stats in mom_stats.items():
        if stats["total"] > 0:
            wr = stats["wins"] / stats["total"] * 100
            print(f"{name:<25} | {wr:>7.1f}% | {stats['total']:>7}")

    # Find best conditions
    print("\n" + "=" * 70)
    print("BEST CONDITIONS (position, distance, momentum)")
    print("=" * 70)

    cond_stats = find_best_conditions(trades)

    # Sort by win rate
    sorted_conds = sorted(
        cond_stats.items(),
        key=lambda x: (x[1]["win_rate"], x[1]["total"]),
        reverse=True
    )

    print(f"\n{'Condition':<45} | {'Win%':>6} | {'Trades':>6}")
    print("-" * 65)
    for cond, stats in sorted_conds[:15]:  # Top 15
        if stats["total"] >= 50:  # Min sample size
            print(f"{cond:<45} | {stats['win_rate']:>5.1f}% | {stats['total']:>6}")

    print("\n" + "=" * 70)
    print("WORST CONDITIONS")
    print("-" * 65)
    for cond, stats in sorted_conds[-10:]:  # Bottom 10
        if stats["total"] >= 50:
            print(f"{cond:<45} | {stats['win_rate']:>5.1f}% | {stats['total']:>6}")
