#!/usr/bin/env python3
"""
Generate performance report for weather trades.

Shows:
- Overall win/loss record
- P/L by EV bucket (to validate that higher EV = higher win rate)
- Model calibration (predicted prob vs actual win rate)

Usage:
    python scripts/weather_performance_report.py
"""

from __future__ import annotations

import sqlite3
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WEATHER_DB = ROOT / "weather_trades.db"


def main() -> int:
    if not WEATHER_DB.exists():
        print(f"No trades database found at {WEATHER_DB}")
        return 1

    conn = sqlite3.connect(str(WEATHER_DB))
    conn.row_factory = sqlite3.Row

    # Get all trades
    trades = conn.execute("""
        SELECT
            trade_date,
            city_key,
            event_display,
            side,
            limit_price_cents,
            fill_price_cents,
            fair_q,
            ev,
            status,
            settled,
            won,
            payout_cents
        FROM weather_trades
        ORDER BY trade_date DESC
    """).fetchall()

    if not trades:
        print("No trades found")
        return 0

    print("=" * 70)
    print("WEATHER TRADING PERFORMANCE REPORT")
    print(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print("=" * 70)

    # Overall stats
    total = len(trades)
    settled = [t for t in trades if t["settled"]]
    pending = [t for t in trades if not t["settled"]]
    dry_runs = [t for t in trades if t["status"] == "DRY_RUN"]
    live = [t for t in trades if t["status"] not in ("DRY_RUN", None)]

    wins = [t for t in settled if t["won"]]
    losses = [t for t in settled if not t["won"]]

    print(f"\nTOTAL TRADES: {total}")
    print(f"  Live trades: {len(live)}")
    print(f"  Paper trades (DRY_RUN): {len(dry_runs)}")
    print(f"  Settled: {len(settled)} ({len(wins)}W - {len(losses)}L)")
    print(f"  Pending: {len(pending)}")

    if settled:
        win_rate = len(wins) / len(settled) * 100
        total_cost = sum(t["fill_price_cents"] or t["limit_price_cents"] or 0 for t in settled)
        total_payout = sum(t["payout_cents"] or 0 for t in settled)
        pnl = total_payout - total_cost
        roi = (pnl / total_cost * 100) if total_cost > 0 else 0

        print(f"\nOVERALL RESULTS:")
        print(f"  Win rate: {win_rate:.1f}%")
        print(f"  Total cost: {total_cost}¢ (${total_cost/100:.2f})")
        print(f"  Total payout: {total_payout}¢ (${total_payout/100:.2f})")
        print(f"  Net P/L: {pnl:+d}¢ (${pnl/100:+.2f})")
        print(f"  ROI: {roi:+.1f}%")

    # Results by EV bucket
    print("\n" + "-" * 70)
    print("RESULTS BY EDGE (EV) BUCKET")
    print("-" * 70)

    buckets = {
        "0-5%": (0, 0.05),
        "5-10%": (0.05, 0.10),
        "10-15%": (0.10, 0.15),
        "15-20%": (0.15, 0.20),
        "20%+": (0.20, 1.0),
    }

    print(f"{'Bucket':<12} | {'Trades':>7} | {'W-L':>8} | {'Win%':>6} | {'Avg Cost':>9} | {'ROI':>8}")
    print("-" * 70)

    for bucket_name, (low, high) in buckets.items():
        bucket_trades = [t for t in settled if low <= (t["ev"] or 0) < high]
        if not bucket_trades:
            print(f"{bucket_name:<12} | {'--':>7} | {'--':>8} | {'--':>6} | {'--':>9} | {'--':>8}")
            continue

        b_wins = len([t for t in bucket_trades if t["won"]])
        b_losses = len(bucket_trades) - b_wins
        b_win_rate = b_wins / len(bucket_trades) * 100

        b_cost = sum(t["fill_price_cents"] or t["limit_price_cents"] or 0 for t in bucket_trades)
        b_payout = sum(t["payout_cents"] or 0 for t in bucket_trades)
        b_avg_cost = b_cost / len(bucket_trades)
        b_pnl = b_payout - b_cost
        b_roi = (b_pnl / b_cost * 100) if b_cost > 0 else 0

        print(f"{bucket_name:<12} | {len(bucket_trades):>7} | {b_wins:>3}-{b_losses:<4} | {b_win_rate:>5.1f}% | {b_avg_cost:>8.1f}¢ | {b_roi:>+7.1f}%")

    # Recent trades
    print("\n" + "-" * 70)
    print("RECENT TRADES (Last 10)")
    print("-" * 70)

    recent = trades[:10]
    for t in recent:
        date = t["trade_date"]
        city = t["city_key"]
        event = t["event_display"] or "?"
        side = t["side"]
        price = t["fill_price_cents"] or t["limit_price_cents"] or 0
        ev = (t["ev"] or 0) * 100
        status = t["status"]

        if t["settled"]:
            result = "WIN" if t["won"] else "LOSS"
            pnl = (t["payout_cents"] or 0) - price
            result_str = f"{result} ({pnl:+d}¢)"
        else:
            result_str = f"[{status}]"

        print(f"  {date} | {city:<8} | {event:<20} | {side:<3} @ {price:>2}¢ | EV:{ev:+.0f}¢ | {result_str}")

    # Calibration check
    print("\n" + "-" * 70)
    print("MODEL CALIBRATION (Predicted Prob vs Actual Win Rate)")
    print("-" * 70)

    prob_buckets = {
        "10-30%": (0.10, 0.30),
        "30-50%": (0.30, 0.50),
        "50-70%": (0.50, 0.70),
        "70-90%": (0.70, 0.90),
    }

    print(f"{'Predicted':>12} | {'Trades':>7} | {'Wins':>6} | {'Actual Win%':>12} | {'Calibration':>12}")
    print("-" * 70)

    for bucket_name, (low, high) in prob_buckets.items():
        # fair_q is the model's probability
        bucket_trades = [t for t in settled if t["fair_q"] and low <= t["fair_q"] < high]
        if not bucket_trades:
            continue

        b_wins = len([t for t in bucket_trades if t["won"]])
        actual_win_rate = b_wins / len(bucket_trades)
        expected_win_rate = (low + high) / 2
        calibration = actual_win_rate - expected_win_rate

        cal_str = "Good" if abs(calibration) < 0.10 else ("Over" if calibration > 0 else "Under")
        print(f"{bucket_name:>12} | {len(bucket_trades):>7} | {b_wins:>6} | {actual_win_rate*100:>11.1f}% | {cal_str} ({calibration:+.1%})")

    conn.close()
    print("\n" + "=" * 70)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
