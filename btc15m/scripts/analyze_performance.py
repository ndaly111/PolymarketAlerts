#!/usr/bin/env python3
"""
Performance analysis tool for BTC trading.

Analyzes historical data to identify patterns:
- Best/worst trading hours
- Feature importance
- Confidence calibration
- Win rate by various factors

Usage:
    python analyze_performance.py [--days N]
"""

import argparse
import sqlite3
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from collections import defaultdict
import sys

# Setup paths
SCRIPT_DIR = Path(__file__).resolve().parent
BTC15M_DIR = SCRIPT_DIR.parent

sys.path.insert(0, str(BTC15M_DIR))

DB_15MIN = BTC15M_DIR / "data" / "btc15m.db"
DB_HOURLY = BTC15M_DIR / "hourly_trades.db"
FEATURE_IMPORTANCE_FILE = BTC15M_DIR / "models" / "feature_importance.json"


def analyze_by_hour(db_path: Path, days: int = 30):
    """Analyze performance by hour of day."""
    if not db_path.exists():
        print("Database not found")
        return

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    c = conn.cursor()

    cutoff = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()

    c.execute("""
        SELECT timestamp, market_ask, was_correct, model_prob
        FROM prediction_samples
        WHERE was_correct IS NOT NULL AND timestamp > ?
    """, (cutoff,))

    hours = defaultdict(lambda: {"trades": 0, "wins": 0, "pnl": 0, "edge_sum": 0})

    for row in c.fetchall():
        try:
            ts = datetime.fromisoformat(row["timestamp"].replace("Z", "+00:00"))
            hour = ts.hour
        except:
            continue

        entry = row["market_ask"] or 0.5
        correct = row["was_correct"]
        prob = row["model_prob"] or 0.5

        hours[hour]["trades"] += 1
        hours[hour]["edge_sum"] += prob - entry

        if correct:
            hours[hour]["wins"] += 1
            hours[hour]["pnl"] += (1 - entry) * 0.93
        else:
            hours[hour]["pnl"] -= entry

    conn.close()

    print(f"\n{'='*60}")
    print(f"PERFORMANCE BY HOUR (last {days} days)")
    print(f"{'='*60}")
    print(f"{'Hour':<10} {'Trades':<8} {'Win Rate':<10} {'P&L':<12} {'Avg Edge':<10}")
    print("-" * 60)

    sorted_hours = sorted(hours.items(), key=lambda x: x[1]["pnl"], reverse=True)

    for hour, data in sorted_hours:
        if data["trades"] > 0:
            win_rate = data["wins"] / data["trades"] * 100
            avg_edge = data["edge_sum"] / data["trades"] * 100
            pnl_emoji = "✅" if data["pnl"] > 0 else "❌"
            print(f"{hour:02d}:00 UTC  {data['trades']:<8} {win_rate:>6.1f}%    ${data['pnl']:>+8.2f}   {avg_edge:>+6.1f}%  {pnl_emoji}")

    # Recommendations
    print("\n" + "="*60)
    print("RECOMMENDATIONS")
    print("="*60)

    profitable_hours = [h for h, d in sorted_hours if d["pnl"] > 0 and d["trades"] >= 3]
    losing_hours = [h for h, d in sorted_hours if d["pnl"] < -0.5 and d["trades"] >= 3]

    if profitable_hours:
        print(f"✅ Best hours to trade: {', '.join(f'{h:02d}:00' for h in profitable_hours[:5])}")
    if losing_hours:
        print(f"⚠️  Consider avoiding: {', '.join(f'{h:02d}:00' for h in losing_hours[:3])}")


def analyze_by_confidence(db_path: Path):
    """Analyze performance by confidence bucket."""
    if not db_path.exists():
        print("Database not found")
        return

    conn = sqlite3.connect(db_path)
    c = conn.cursor()

    c.execute("""
        SELECT model_prob, market_ask, was_correct
        FROM prediction_samples
        WHERE was_correct IS NOT NULL AND model_prob IS NOT NULL
    """)

    buckets = defaultdict(lambda: {"trades": 0, "wins": 0, "pnl": 0})

    for row in c.fetchall():
        prob = row[0]
        entry = row[1] or 0.5
        correct = row[2]

        # 5% buckets
        bucket = int(prob * 20) * 5  # 50, 55, 60, etc.

        buckets[bucket]["trades"] += 1
        if correct:
            buckets[bucket]["wins"] += 1
            buckets[bucket]["pnl"] += (1 - entry) * 0.93
        else:
            buckets[bucket]["pnl"] -= entry

    conn.close()

    print(f"\n{'='*60}")
    print("PERFORMANCE BY CONFIDENCE LEVEL")
    print(f"{'='*60}")
    print(f"{'Confidence':<12} {'Trades':<8} {'Win Rate':<10} {'Expected':<10} {'P&L':<10} {'Calibration'}")
    print("-" * 70)

    for bucket in sorted(buckets.keys()):
        data = buckets[bucket]
        if data["trades"] >= 2:
            win_rate = data["wins"] / data["trades"] * 100
            expected = bucket + 2.5  # Middle of bucket
            diff = win_rate - expected

            if abs(diff) < 5:
                cal = "✅ Well calibrated"
            elif diff > 0:
                cal = f"📈 Underconfident ({diff:+.0f}%)"
            else:
                cal = f"📉 Overconfident ({diff:+.0f}%)"

            print(f"{bucket:>3}-{bucket+5}%     {data['trades']:<8} {win_rate:>6.1f}%    {expected:>6.1f}%    ${data['pnl']:>+6.2f}    {cal}")


def analyze_features():
    """Show top features from the trained model."""
    if not FEATURE_IMPORTANCE_FILE.exists():
        print("Feature importance file not found. Run train_profit_model.py first.")
        return

    with open(FEATURE_IMPORTANCE_FILE) as f:
        data = json.load(f)

    print(f"\n{'='*60}")
    print(f"TOP PREDICTIVE FEATURES ({data.get('model', 'Unknown')} model)")
    print(f"{'='*60}")
    print(f"Model accuracy: {data.get('accuracy', 0)*100:.1f}%")
    print(f"Last updated: {data.get('timestamp', 'Unknown')}")
    print()

    features = data.get("features", [])
    for i, f in enumerate(features[:20], 1):
        bar_len = int(f["importance"] * 200)
        bar = "█" * bar_len
        print(f"{i:2}. {f['name']:<30} {f['importance']:.4f} {bar}")

    # Group by category
    print(f"\n{'='*60}")
    print("FEATURE CATEGORIES")
    print(f"{'='*60}")

    categories = defaultdict(float)
    for f in features:
        name = f["name"]
        imp = f["importance"]

        if name.startswith("comp_"):
            categories["Composite indicators"] += imp
        elif name in ["market_ask", "edge", "model_prob"]:
            categories["Market prices"] += imp
        elif "rsi" in name or "macd" in name or "adx" in name:
            categories["Momentum indicators"] += imp
        elif "bb_" in name or "atr" in name or "volatility" in name:
            categories["Volatility indicators"] += imp
        elif "volume" in name or "obv" in name or "cmf" in name:
            categories["Volume indicators"] += imp
        elif "hour" in name or "session" in name:
            categories["Time-based"] += imp
        else:
            categories["Other"] += imp

    for cat, imp in sorted(categories.items(), key=lambda x: x[1], reverse=True):
        pct = imp * 100
        bar = "█" * int(pct * 2)
        print(f"  {cat:<25} {pct:>5.1f}% {bar}")


def analyze_by_price_level(db_path: Path):
    """Analyze performance by entry price level."""
    if not db_path.exists():
        return

    conn = sqlite3.connect(db_path)
    c = conn.cursor()

    c.execute("""
        SELECT market_ask, was_correct
        FROM prediction_samples
        WHERE was_correct IS NOT NULL AND market_ask IS NOT NULL
    """)

    buckets = defaultdict(lambda: {"trades": 0, "wins": 0, "pnl": 0})

    for row in c.fetchall():
        entry = row[0]
        correct = row[1]

        # 10 cent buckets
        bucket = int(entry * 10) * 10  # 10, 20, 30, etc.

        buckets[bucket]["trades"] += 1
        if correct:
            buckets[bucket]["wins"] += 1
            buckets[bucket]["pnl"] += (1 - entry) * 0.93
        else:
            buckets[bucket]["pnl"] -= entry

    conn.close()

    print(f"\n{'='*60}")
    print("PERFORMANCE BY ENTRY PRICE")
    print(f"{'='*60}")
    print(f"{'Price':<12} {'Trades':<8} {'Win Rate':<10} {'P&L':<12} {'Avg P&L/Trade'}")
    print("-" * 60)

    for bucket in sorted(buckets.keys()):
        data = buckets[bucket]
        if data["trades"] >= 2:
            win_rate = data["wins"] / data["trades"] * 100
            avg_pnl = data["pnl"] / data["trades"]
            emoji = "✅" if data["pnl"] > 0 else "❌"
            print(f"{bucket:>2}-{bucket+10}¢      {data['trades']:<8} {win_rate:>6.1f}%    ${data['pnl']:>+8.2f}   ${avg_pnl:>+.3f}  {emoji}")


def main():
    parser = argparse.ArgumentParser(description="Analyze trading performance")
    parser.add_argument("--days", type=int, default=30, help="Days of history to analyze")
    parser.add_argument("--hours", action="store_true", help="Analyze by hour of day")
    parser.add_argument("--confidence", action="store_true", help="Analyze by confidence level")
    parser.add_argument("--features", action="store_true", help="Show feature importance")
    parser.add_argument("--price", action="store_true", help="Analyze by entry price")
    parser.add_argument("--all", action="store_true", help="Run all analyses")
    args = parser.parse_args()

    if args.all or (not args.hours and not args.confidence and not args.features and not args.price):
        args.hours = args.confidence = args.features = args.price = True

    print("="*60)
    print("BTC TRADING PERFORMANCE ANALYSIS")
    print("="*60)

    if args.hours:
        analyze_by_hour(DB_15MIN, args.days)

    if args.confidence:
        analyze_by_confidence(DB_15MIN)

    if args.features:
        analyze_features()

    if args.price:
        analyze_by_price_level(DB_15MIN)

    print("\n" + "="*60)
    print("Analysis complete")
    print("="*60)


if __name__ == "__main__":
    main()
