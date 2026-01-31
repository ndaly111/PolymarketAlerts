#!/usr/bin/env python3
"""
Generate ML vs NBM comparison data for dashboard.

Compares:
- ML model trades (weather_trades table)
- NBM simple strategy trades (nbm_strategy_trades table)

Output: docs/data/ml_comparison.json
"""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[1]
WEATHER_DB = ROOT / "weather_trades.db"
OUTPUT_PATH = ROOT / "docs" / "data" / "ml_comparison.json"


def get_ml_trades(conn: sqlite3.Connection) -> List[Dict[str, Any]]:
    """Get ML model trades from weather_trades table."""
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()
    cur.execute("""
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
            payout_cents,
            created_at
        FROM weather_trades
        ORDER BY trade_date DESC, created_at DESC
    """)
    return [dict(r) for r in cur.fetchall()]


def get_nbm_trades(conn: sqlite3.Connection) -> List[Dict[str, Any]]:
    """Get NBM strategy trades from nbm_strategy_trades table."""
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()

    # Check if table exists
    table_exists = cur.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='nbm_strategy_trades'"
    ).fetchone()

    if not table_exists:
        return []

    cur.execute("""
        SELECT
            trade_date,
            city_key,
            market_ticker,
            strike,
            side,
            ask_cents,
            forecast_high,
            distance_from_forecast,
            reason,
            ev_estimate,
            status,
            settled,
            won,
            payout_cents,
            created_at
        FROM nbm_strategy_trades
        ORDER BY trade_date DESC, created_at DESC
    """)
    return [dict(r) for r in cur.fetchall()]


def compute_stats(trades: List[Dict[str, Any]], cost_field: str, ev_field: str) -> Dict[str, Any]:
    """Compute aggregate stats for a set of trades."""
    total = len(trades)
    settled = [t for t in trades if t.get("settled")]
    pending = [t for t in trades if not t.get("settled")]
    wins = [t for t in settled if t.get("won")]
    losses = [t for t in settled if not t.get("won")]

    settled_cost = sum(t.get(cost_field) or 0 for t in settled)
    pending_cost = sum(t.get(cost_field) or 0 for t in pending)
    total_payout = sum(t.get("payout_cents") or 0 for t in settled)

    pnl = total_payout - settled_cost
    roi = (pnl / settled_cost * 100) if settled_cost > 0 else 0
    win_rate = (len(wins) / len(settled) * 100) if settled else 0
    avg_ev = sum(t.get(ev_field) or 0 for t in trades) / total * 100 if total > 0 else 0

    return {
        "total_trades": total,
        "wins": len(wins),
        "losses": len(losses),
        "pending": len(pending),
        "win_rate": round(win_rate, 1),
        "settled_cost_cents": settled_cost,
        "pending_cost_cents": pending_cost,
        "total_payout_cents": total_payout,
        "pnl_cents": pnl,
        "roi": round(roi, 1),
        "avg_ev": round(avg_ev, 1),
    }


def compute_by_date(trades: List[Dict[str, Any]], cost_field: str) -> List[Dict[str, Any]]:
    """Compute cumulative P/L by date."""
    # Group by date
    by_date = {}
    for t in trades:
        date = t.get("trade_date")
        if not date or not t.get("settled"):
            continue
        if date not in by_date:
            by_date[date] = {"cost": 0, "payout": 0, "wins": 0, "losses": 0}
        by_date[date]["cost"] += t.get(cost_field) or 0
        by_date[date]["payout"] += t.get("payout_cents") or 0
        if t.get("won"):
            by_date[date]["wins"] += 1
        else:
            by_date[date]["losses"] += 1

    # Build cumulative series
    result = []
    cum_pnl = 0
    cum_wins = 0
    cum_losses = 0
    for date in sorted(by_date.keys()):
        d = by_date[date]
        daily_pnl = d["payout"] - d["cost"]
        cum_pnl += daily_pnl
        cum_wins += d["wins"]
        cum_losses += d["losses"]
        result.append({
            "date": date,
            "daily_pnl_cents": daily_pnl,
            "cumulative_pnl_cents": cum_pnl,
            "cumulative_wins": cum_wins,
            "cumulative_losses": cum_losses,
        })

    return result


def compute_by_city(trades: List[Dict[str, Any]], cost_field: str, ev_field: str) -> Dict[str, Dict[str, Any]]:
    """Compute stats by city."""
    by_city = {}
    for t in trades:
        city = t.get("city_key", "Unknown")
        if city not in by_city:
            by_city[city] = {"trades": [], "settled": [], "wins": 0, "losses": 0}
        by_city[city]["trades"].append(t)
        if t.get("settled"):
            by_city[city]["settled"].append(t)
            if t.get("won"):
                by_city[city]["wins"] += 1
            else:
                by_city[city]["losses"] += 1

    result = {}
    for city, data in by_city.items():
        settled_cost = sum(t.get(cost_field) or 0 for t in data["settled"])
        payout = sum(t.get("payout_cents") or 0 for t in data["settled"])
        total_settled = data["wins"] + data["losses"]
        result[city] = {
            "count": len(data["trades"]),
            "wins": data["wins"],
            "losses": data["losses"],
            "win_rate": round(data["wins"] / total_settled * 100, 1) if total_settled > 0 else 0,
            "pnl_cents": payout - settled_cost,
            "roi": round((payout - settled_cost) / settled_cost * 100, 1) if settled_cost > 0 else 0,
        }

    return result


def find_overlapping_trades(ml_trades: List[Dict], nbm_trades: List[Dict]) -> List[Dict[str, Any]]:
    """Find trades where both strategies had signals on the same market."""
    # Create lookup by (date, city) for NBM
    nbm_lookup = {}
    for t in nbm_trades:
        key = (t.get("trade_date"), t.get("city_key"))
        if key not in nbm_lookup:
            nbm_lookup[key] = []
        nbm_lookup[key].append(t)

    overlaps = []
    for ml_t in ml_trades:
        key = (ml_t.get("trade_date"), ml_t.get("city_key"))
        if key in nbm_lookup:
            for nbm_t in nbm_lookup[key]:
                overlaps.append({
                    "date": ml_t.get("trade_date"),
                    "city": ml_t.get("city_key"),
                    "ml_side": ml_t.get("side"),
                    "ml_price": ml_t.get("fill_price_cents") or ml_t.get("limit_price_cents"),
                    "ml_ev": round((ml_t.get("ev") or 0) * 100, 1),
                    "ml_won": ml_t.get("won"),
                    "ml_settled": ml_t.get("settled"),
                    "nbm_side": nbm_t.get("side"),
                    "nbm_price": nbm_t.get("ask_cents"),
                    "nbm_ev": round((nbm_t.get("ev_estimate") or 0) * 100, 1),
                    "nbm_won": nbm_t.get("won"),
                    "nbm_settled": nbm_t.get("settled"),
                    "nbm_strike": nbm_t.get("strike"),
                })

    return overlaps


def main() -> int:
    if not WEATHER_DB.exists():
        print(f"No trades database found at {WEATHER_DB}")
        # Create empty output
        OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(OUTPUT_PATH, "w") as f:
            json.dump({
                "generated_at": datetime.utcnow().isoformat() + "Z",
                "ml": {"total_trades": 0},
                "nbm": {"total_trades": 0},
            }, f, indent=2)
        return 0

    conn = sqlite3.connect(str(WEATHER_DB))

    # Get trades from both strategies
    ml_trades = get_ml_trades(conn)
    nbm_trades = get_nbm_trades(conn)

    print(f"ML trades: {len(ml_trades)}")
    print(f"NBM trades: {len(nbm_trades)}")

    # Compute stats for each
    ml_stats = compute_stats(ml_trades, "fill_price_cents", "ev")
    nbm_stats = compute_stats(nbm_trades, "ask_cents", "ev_estimate")

    # Compute time series
    ml_by_date = compute_by_date(ml_trades, "fill_price_cents")
    nbm_by_date = compute_by_date(nbm_trades, "ask_cents")

    # Compute by city
    ml_by_city = compute_by_city(ml_trades, "fill_price_cents", "ev")
    nbm_by_city = compute_by_city(nbm_trades, "ask_cents", "ev_estimate")

    # Find overlapping trades
    overlaps = find_overlapping_trades(ml_trades, nbm_trades)

    # Format recent trades for each strategy
    ml_recent = []
    for t in ml_trades[:20]:
        ml_recent.append({
            "date": t.get("trade_date"),
            "city": t.get("city_key"),
            "event": t.get("event_display"),
            "side": t.get("side"),
            "price": t.get("fill_price_cents") or t.get("limit_price_cents"),
            "ev": round((t.get("ev") or 0) * 100, 1),
            "settled": t.get("settled"),
            "won": t.get("won"),
        })

    nbm_recent = []
    for t in nbm_trades[:20]:
        nbm_recent.append({
            "date": t.get("trade_date"),
            "city": t.get("city_key"),
            "strike": t.get("strike"),
            "side": t.get("side"),
            "price": t.get("ask_cents"),
            "ev": round((t.get("ev_estimate") or 0) * 100, 1),
            "settled": t.get("settled"),
            "won": t.get("won"),
            "forecast": t.get("forecast_high"),
        })

    # Build output
    output = {
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "ml": {
            **ml_stats,
            "by_date": ml_by_date,
            "by_city": ml_by_city,
            "recent_trades": ml_recent,
        },
        "nbm": {
            **nbm_stats,
            "by_date": nbm_by_date,
            "by_city": nbm_by_city,
            "recent_trades": nbm_recent,
        },
        "overlapping_trades": overlaps[:50],  # Limit to 50 most recent
    }

    # Write output
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump(output, f, indent=2)

    print(f"Wrote comparison data to {OUTPUT_PATH}")

    # Print summary
    print("\n" + "=" * 60)
    print("ML vs NBM COMPARISON")
    print("=" * 60)
    print(f"\n{'Metric':<20} {'ML Model':>15} {'NBM Simple':>15}")
    print("-" * 60)
    print(f"{'Total Trades':<20} {ml_stats['total_trades']:>15} {nbm_stats['total_trades']:>15}")
    ml_record = f"{ml_stats['wins']}W-{ml_stats['losses']}L"
    nbm_record = f"{nbm_stats['wins']}W-{nbm_stats['losses']}L"
    print(f"{'Record':<20} {ml_record:>15} {nbm_record:>15}")
    print(f"{'Win Rate':<20} {ml_stats['win_rate']:>14}% {nbm_stats['win_rate']:>14}%")
    print(f"{'ROI':<20} {ml_stats['roi']:>14}% {nbm_stats['roi']:>14}%")
    ml_pnl = f"${ml_stats['pnl_cents']/100:.2f}"
    nbm_pnl = f"${nbm_stats['pnl_cents']/100:.2f}"
    print(f"{'P/L':<20} {ml_pnl:>15} {nbm_pnl:>15}")
    print(f"{'Avg EV':<20} {ml_stats['avg_ev']:>14}% {nbm_stats['avg_ev']:>14}%")
    print(f"{'Pending':<20} {ml_stats['pending']:>15} {nbm_stats['pending']:>15}")

    conn.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
