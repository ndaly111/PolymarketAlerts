#!/usr/bin/env python3
"""
Daily summary generator for BTC trading.

Collects stats from both 15-min and hourly markets,
analyzes performance, and sends a Discord notification.

Run at midnight or on-demand: python daily_summary.py [--date YYYY-MM-DD]
"""

import argparse
import sqlite3
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Setup paths
SCRIPT_DIR = Path(__file__).resolve().parent
BTC15M_DIR = SCRIPT_DIR.parent
REPO_DIR = BTC15M_DIR.parent

sys.path.insert(0, str(REPO_DIR))
sys.path.insert(0, str(BTC15M_DIR))

# Load env
try:
    from dotenv import load_dotenv
    env_path = BTC15M_DIR / ".env"
    if not env_path.exists():
        env_path = REPO_DIR / ".env"
    if env_path.exists():
        load_dotenv(env_path)
except ImportError:
    pass

from lib.discord import send_daily_summary, send_calibration_alert, get_webhook_url

# Database paths
DB_15MIN = BTC15M_DIR / "data" / "btc15m.db"
DB_HOURLY = BTC15M_DIR / "hourly_trades.db"


def get_15min_stats(db_path: Path, date_str: str) -> Dict:
    """Get 15-min market stats for a specific date."""
    if not db_path.exists():
        return {"pnl": 0, "trades": 0, "wins": 0, "win_rate": 0, "best": None, "worst": None}

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    c = conn.cursor()

    # Get trades settled on this date
    c.execute("""
        SELECT * FROM prediction_samples
        WHERE settled_at LIKE ? AND was_correct IS NOT NULL
    """, (f"{date_str}%",))

    trades = c.fetchall()
    conn.close()

    if not trades:
        return {"pnl": 0, "trades": 0, "wins": 0, "win_rate": 0, "best": None, "worst": None}

    total_pnl = 0
    wins = 0
    best_trade = None
    worst_trade = None
    best_pnl = float('-inf')
    worst_pnl = float('inf')

    for t in trades:
        entry = t["market_ask"] or 0.5
        was_correct = t["was_correct"]

        if was_correct:
            pnl = (1 - entry) * 0.93
            wins += 1
        else:
            pnl = -entry

        total_pnl += pnl

        if pnl > best_pnl:
            best_pnl = pnl
            best_trade = {
                "pnl": pnl,
                "market": t["market_ticker"],
                "side": t["prediction"],
                "entry": entry,
            }

        if pnl < worst_pnl:
            worst_pnl = pnl
            worst_trade = {
                "pnl": pnl,
                "market": t["market_ticker"],
                "side": t["prediction"],
                "entry": entry,
            }

    win_rate = (wins / len(trades) * 100) if trades else 0

    return {
        "pnl": total_pnl,
        "trades": len(trades),
        "wins": wins,
        "win_rate": win_rate,
        "best": best_trade,
        "worst": worst_trade,
    }


def get_hourly_stats(db_path: Path, date_str: str) -> Dict:
    """Get hourly market stats for a specific date."""
    if not db_path.exists():
        return {"pnl": 0, "trades": 0, "wins": 0, "win_rate": 0, "best": None, "worst": None}

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    c = conn.cursor()

    # Get real trades settled on this date
    c.execute("""
        SELECT * FROM hourly_real_trades
        WHERE settled_at LIKE ? AND settled = 1
    """, (f"{date_str}%",))

    trades = c.fetchall()
    conn.close()

    if not trades:
        return {"pnl": 0, "trades": 0, "wins": 0, "win_rate": 0, "best": None, "worst": None}

    total_pnl = 0
    wins = 0
    best_trade = None
    worst_trade = None
    best_pnl = float('-inf')
    worst_pnl = float('inf')

    for t in trades:
        pnl = t["pnl"] or 0
        total_pnl += pnl

        if t["outcome"] == "WIN":
            wins += 1

        if pnl > best_pnl:
            best_pnl = pnl
            best_trade = {
                "pnl": pnl,
                "market": t["market_ticker"],
                "side": t["side"],
                "entry": t["entry_price"],
            }

        if pnl < worst_pnl:
            worst_pnl = pnl
            worst_trade = {
                "pnl": pnl,
                "market": t["market_ticker"],
                "side": t["side"],
                "entry": t["entry_price"],
            }

    win_rate = (wins / len(trades) * 100) if trades else 0

    return {
        "pnl": total_pnl,
        "trades": len(trades),
        "wins": wins,
        "win_rate": win_rate,
        "best": best_trade,
        "worst": worst_trade,
    }


def get_calibration_data(db_path: Path) -> List[Dict]:
    """Analyze confidence calibration - does 80% confident mean 80% win rate?"""
    if not db_path.exists():
        return []

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    c = conn.cursor()

    c.execute("""
        SELECT model_prob, was_correct FROM prediction_samples
        WHERE was_correct IS NOT NULL AND model_prob IS NOT NULL
    """)

    trades = c.fetchall()
    conn.close()

    if not trades:
        return []

    # Bucket by confidence level
    buckets = {
        "50-60%": {"min": 0.50, "max": 0.60, "correct": 0, "total": 0},
        "60-70%": {"min": 0.60, "max": 0.70, "correct": 0, "total": 0},
        "70-80%": {"min": 0.70, "max": 0.80, "correct": 0, "total": 0},
        "80-90%": {"min": 0.80, "max": 0.90, "correct": 0, "total": 0},
        "90%+": {"min": 0.90, "max": 1.01, "correct": 0, "total": 0},
    }

    for t in trades:
        prob = t["model_prob"]
        correct = t["was_correct"]

        for name, bucket in buckets.items():
            if bucket["min"] <= prob < bucket["max"]:
                bucket["total"] += 1
                if correct:
                    bucket["correct"] += 1
                break

    # Calculate actual win rates
    results = []
    for name, bucket in buckets.items():
        if bucket["total"] >= 3:  # Need at least 3 samples
            expected = (bucket["min"] + bucket["max"]) / 2 * 100
            actual = (bucket["correct"] / bucket["total"]) * 100
            results.append({
                "range": name,
                "expected": expected,
                "actual": actual,
                "trades": bucket["total"],
            })

    return results


def get_time_analysis(db_path: Path) -> Tuple[Optional[Dict], Optional[Dict]]:
    """Analyze performance by hour of day."""
    if not db_path.exists():
        return None, None

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    c = conn.cursor()

    c.execute("""
        SELECT timestamp, market_ask, was_correct FROM prediction_samples
        WHERE was_correct IS NOT NULL
    """)

    trades = c.fetchall()
    conn.close()

    if not trades:
        return None, None

    # Bucket by hour
    hours = {}
    for t in trades:
        try:
            ts = datetime.fromisoformat(t["timestamp"].replace("Z", "+00:00"))
            hour = ts.hour
        except:
            continue

        if hour not in hours:
            hours[hour] = {"pnl": 0, "trades": 0}

        entry = t["market_ask"] or 0.5
        if t["was_correct"]:
            pnl = (1 - entry) * 0.93
        else:
            pnl = -entry

        hours[hour]["pnl"] += pnl
        hours[hour]["trades"] += 1

    if not hours:
        return None, None

    # Find best and worst hours
    best_hour = None
    worst_hour = None
    best_pnl = float('-inf')
    worst_pnl = float('inf')

    for hour, data in hours.items():
        if data["trades"] >= 2:  # Need at least 2 trades
            if data["pnl"] > best_pnl:
                best_pnl = data["pnl"]
                best_hour = {
                    "hour": f"{hour:02d}:00 UTC",
                    "pnl": data["pnl"],
                    "trades": data["trades"],
                }
            if data["pnl"] < worst_pnl:
                worst_pnl = data["pnl"]
                worst_hour = {
                    "hour": f"{hour:02d}:00 UTC",
                    "pnl": data["pnl"],
                    "trades": data["trades"],
                }

    return best_hour, worst_hour


def analyze_calibration(calibration_data: List[Dict]) -> Tuple[bool, bool, str]:
    """Analyze calibration data and determine if model is mis-calibrated.

    Returns: (has_issue, is_overconfident, suggestion)
    """
    if not calibration_data or len(calibration_data) < 2:
        return False, False, ""

    total_expected = 0
    total_actual = 0
    total_trades = 0
    overconfident_buckets = 0
    underconfident_buckets = 0

    for bucket in calibration_data:
        expected = bucket.get('expected', 0)
        actual = bucket.get('actual', 0)
        trades = bucket.get('trades', 0)

        total_expected += expected * trades
        total_actual += actual * trades
        total_trades += trades

        diff = actual - expected
        if diff < -10:  # Actual win rate 10%+ below expected
            overconfident_buckets += 1
        elif diff > 10:  # Actual win rate 10%+ above expected
            underconfident_buckets += 1

    if total_trades < 10:
        return False, False, "Not enough data for calibration analysis"

    avg_expected = total_expected / total_trades
    avg_actual = total_actual / total_trades
    overall_diff = avg_actual - avg_expected

    if overall_diff < -15:
        # Overconfident: model says 75% but actually winning 60%
        suggestion = f"Model is overconfident by ~{abs(overall_diff):.0f}%. Consider raising the minimum confidence threshold to be more selective."
        return True, True, suggestion
    elif overall_diff > 15:
        # Underconfident: model says 60% but actually winning 75%
        suggestion = f"Model is underconfident by ~{overall_diff:.0f}%. You may be missing profitable trades - consider lowering the threshold slightly."
        return True, False, suggestion

    return False, False, "Calibration looks good"


def get_all_time_stats(db_15min: Path, db_hourly: Path) -> Tuple[float, int]:
    """Get all-time P&L and trade count."""
    total_pnl = 0
    total_trades = 0

    # 15-min trades
    if db_15min.exists():
        conn = sqlite3.connect(db_15min)
        c = conn.cursor()
        c.execute("""
            SELECT market_ask, was_correct FROM prediction_samples
            WHERE was_correct IS NOT NULL
        """)
        for row in c.fetchall():
            entry = row[0] or 0.5
            if row[1]:
                total_pnl += (1 - entry) * 0.93
            else:
                total_pnl -= entry
            total_trades += 1
        conn.close()

    # Hourly trades
    if db_hourly.exists():
        conn = sqlite3.connect(db_hourly)
        c = conn.cursor()
        c.execute("SELECT SUM(pnl), COUNT(*) FROM hourly_real_trades WHERE settled = 1")
        row = c.fetchone()
        if row:
            total_pnl += row[0] or 0
            total_trades += row[1] or 0
        conn.close()

    return total_pnl, total_trades


def get_ml_accuracy(db_path: Path) -> Optional[float]:
    """Get overall ML direction accuracy."""
    if not db_path.exists():
        return None

    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    c.execute("""
        SELECT COUNT(*), SUM(was_correct) FROM prediction_samples
        WHERE was_correct IS NOT NULL
    """)
    row = c.fetchone()
    conn.close()

    if row and row[0] > 0:
        return (row[1] / row[0]) * 100
    return None


def main():
    parser = argparse.ArgumentParser(description="Generate daily trading summary")
    parser.add_argument("--date", type=str, help="Date to summarize (YYYY-MM-DD), defaults to yesterday")
    parser.add_argument("--test", action="store_true", help="Print summary without sending to Discord")
    args = parser.parse_args()

    # Determine which date to summarize
    if args.date:
        date_str = args.date
    else:
        # Default to yesterday
        yesterday = datetime.now(timezone.utc) - timedelta(days=1)
        date_str = yesterday.strftime("%Y-%m-%d")

    print(f"Generating daily summary for {date_str}...")

    # Gather all stats
    stats_15m = get_15min_stats(DB_15MIN, date_str)
    stats_hourly = get_hourly_stats(DB_HOURLY, date_str)
    calibration = get_calibration_data(DB_15MIN)
    best_hour, worst_hour = get_time_analysis(DB_15MIN)
    all_time_pnl, all_time_trades = get_all_time_stats(DB_15MIN, DB_HOURLY)
    ml_accuracy = get_ml_accuracy(DB_15MIN)

    # Determine best/worst trades across both markets
    best_trade = None
    worst_trade = None

    if stats_15m["best"] and stats_hourly["best"]:
        best_trade = stats_15m["best"] if stats_15m["best"]["pnl"] > stats_hourly["best"]["pnl"] else stats_hourly["best"]
    else:
        best_trade = stats_15m["best"] or stats_hourly["best"]

    if stats_15m["worst"] and stats_hourly["worst"]:
        worst_trade = stats_15m["worst"] if stats_15m["worst"]["pnl"] < stats_hourly["worst"]["pnl"] else stats_hourly["worst"]
    else:
        worst_trade = stats_15m["worst"] or stats_hourly["worst"]

    # Print summary
    print(f"\n{'='*50}")
    print(f"DAILY SUMMARY: {date_str}")
    print(f"{'='*50}")
    print(f"15-Min: ${stats_15m['pnl']:+.2f} ({stats_15m['trades']} trades, {stats_15m['win_rate']:.0f}% WR)")
    print(f"Hourly: ${stats_hourly['pnl']:+.2f} ({stats_hourly['trades']} trades, {stats_hourly['win_rate']:.0f}% WR)")
    print(f"Total:  ${stats_15m['pnl'] + stats_hourly['pnl']:+.2f}")
    print(f"\nAll-time: ${all_time_pnl:+.2f} ({all_time_trades} trades)")

    if calibration:
        print(f"\nCalibration:")
        for c in calibration:
            print(f"  {c['range']}: {c['actual']:.0f}% actual (expected {c['expected']:.0f}%)")

    if best_hour:
        print(f"\nBest hour: {best_hour['hour']} (${best_hour['pnl']:+.2f})")
    if worst_hour:
        print(f"Worst hour: {worst_hour['hour']} (${worst_hour['pnl']:+.2f})")

    # Analyze calibration
    has_cal_issue, is_overconfident, cal_suggestion = analyze_calibration(calibration)
    if has_cal_issue:
        print(f"\n⚠️  CALIBRATION ISSUE: {'Overconfident' if is_overconfident else 'Underconfident'}")
        print(f"   {cal_suggestion}")

    # Send to Discord
    if not args.test:
        webhook = get_webhook_url()
        if webhook:
            success = send_daily_summary(
                date_str=date_str,
                fifteen_min_pnl=stats_15m["pnl"],
                fifteen_min_trades=stats_15m["trades"],
                fifteen_min_wins=stats_15m["wins"],
                fifteen_min_win_rate=stats_15m["win_rate"],
                hourly_pnl=stats_hourly["pnl"],
                hourly_trades=stats_hourly["trades"],
                hourly_wins=stats_hourly["wins"],
                hourly_win_rate=stats_hourly["win_rate"],
                best_trade=best_trade,
                worst_trade=worst_trade,
                ml_accuracy=ml_accuracy,
                calibration_data=calibration,
                best_hour=best_hour,
                worst_hour=worst_hour,
                all_time_pnl=all_time_pnl,
                all_time_trades=all_time_trades,
                webhook_url=webhook,
            )
            print(f"\nDaily summary: {'sent' if success else 'failed'}")

            # Send calibration alert if there's an issue
            if has_cal_issue:
                cal_success = send_calibration_alert(
                    calibration_data=calibration,
                    is_overconfident=is_overconfident,
                    adjustment_suggestion=cal_suggestion,
                    current_threshold=0.75,  # TODO: get from config
                    webhook_url=webhook,
                )
                print(f"Calibration alert: {'sent' if cal_success else 'failed'}")
        else:
            print("\nNo Discord webhook configured")
    else:
        print("\n[TEST MODE] Would send to Discord")


if __name__ == "__main__":
    main()
