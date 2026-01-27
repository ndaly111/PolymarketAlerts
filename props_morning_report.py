#!/usr/bin/env python3
"""
Morning report for props trading.

Analyzes yesterday's scanned opportunities and placed trades:
- Fetches settlement results from Kalshi
- Calculates implied vs actual win rates
- Computes P&L and ROI
- Posts summary to Discord
"""

from __future__ import annotations

import os
import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import requests

from kalshi_auth_client import KalshiAuthClient


# Database
DB_PATH = Path(os.getenv("PROPS_DB_PATH", "props_trades.db"))

# Discord
DISCORD_WEBHOOK = os.getenv("DISCORD_PROPS_WEBHOOK", os.getenv("DISCORD_WEBHOOK_URL", ""))


def get_yesterday_date() -> str:
    """Get yesterday's date in YYYY-MM-DD format."""
    yesterday = datetime.now(timezone.utc) - timedelta(days=1)
    return yesterday.strftime("%Y-%m-%d")


def fetch_scanned_opportunities(db_path: Path, scan_date: str) -> List[Dict[str, Any]]:
    """Fetch all scanned opportunities for a given date."""
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()

    cur.execute("""
        SELECT * FROM scanned_opportunities
        WHERE scan_date = ?
        ORDER BY edge DESC
    """, (scan_date,))

    rows = [dict(row) for row in cur.fetchall()]
    conn.close()
    return rows


def fetch_placed_trades(db_path: Path, trade_date: str) -> List[Dict[str, Any]]:
    """Fetch all placed trades for a given date."""
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()

    cur.execute("""
        SELECT * FROM trades
        WHERE trade_date = ?
        ORDER BY created_at
    """, (trade_date,))

    rows = [dict(row) for row in cur.fetchall()]
    conn.close()
    return rows


def get_market_result(client: KalshiAuthClient, ticker: str) -> Optional[str]:
    """
    Get the settlement result for a market.

    Returns: 'yes', 'no', or None if not settled
    """
    try:
        market = client.get_market(ticker)
        result = market.get("result")
        if result:
            return result.lower()

        # Check status
        status = market.get("status", "").lower()
        if status == "settled":
            # Try to infer from other fields
            if market.get("yes_sub_title") == "Won":
                return "yes"
            elif market.get("no_sub_title") == "Won":
                return "no"

        return None
    except Exception as e:
        print(f"  [warn] Could not fetch result for {ticker}: {e}")
        return None


def update_opportunity_result(db_path: Path, ticker: str, scan_date: str, result: str) -> None:
    """Update the result for a scanned opportunity."""
    now = datetime.now(timezone.utc).isoformat()
    conn = sqlite3.connect(str(db_path))
    cur = conn.cursor()

    cur.execute("""
        UPDATE scanned_opportunities
        SET result = ?, settled_at = ?
        WHERE ticker = ? AND scan_date = ?
    """, (result, now, ticker, scan_date))

    conn.commit()
    conn.close()


def update_trade_result(db_path: Path, ticker: str, trade_date: str, result: str) -> None:
    """Update the result for a placed trade."""
    conn = sqlite3.connect(str(db_path))
    cur = conn.cursor()

    # Add result column if it doesn't exist
    try:
        cur.execute("ALTER TABLE trades ADD COLUMN result TEXT")
    except sqlite3.OperationalError:
        pass  # Column already exists

    cur.execute("""
        UPDATE trades
        SET result = ?
        WHERE ticker = ? AND trade_date = ?
    """, (result, ticker, trade_date))

    conn.commit()
    conn.close()


def calculate_stats(records: List[Dict[str, Any]], is_trades: bool = False) -> Dict[str, Any]:
    """
    Calculate performance statistics.

    Returns dict with:
    - total_bets: number of bets
    - wins: number of wins
    - losses: number of losses
    - unsettled: number not yet settled
    - avg_implied_prob: average implied win probability (price paid)
    - actual_win_rate: observed win rate
    - calibration: actual - implied (positive = we're beating expectations)
    - net_pnl: net profit/loss in cents (assuming $1 per bet)
    - roi: return on investment percentage
    """
    total = len(records)
    if total == 0:
        return {
            "total_bets": 0,
            "wins": 0,
            "losses": 0,
            "unsettled": 0,
            "avg_implied_prob": 0,
            "actual_win_rate": 0,
            "calibration": 0,
            "net_pnl_cents": 0,
            "roi_pct": 0,
            "avg_edge": 0,
        }

    wins = 0
    losses = 0
    unsettled = 0
    total_implied_prob = 0
    total_edge = 0
    net_pnl_cents = 0
    settled_count = 0

    for r in records:
        result = r.get("result")
        side = r.get("side", "").upper()

        # Get the price we would pay (or did pay)
        if is_trades:
            price_cents = r.get("limit_price_cents", 50)
            implied_prob = price_cents / 100.0
            edge = r.get("edge", 0)
        else:
            # For scanned opportunities, use kalshi_prob
            implied_prob = r.get("kalshi_prob", 0.5)
            price_cents = int(implied_prob * 100)
            edge = r.get("edge", 0)

        total_implied_prob += implied_prob
        total_edge += edge

        if result is None:
            unsettled += 1
            continue

        settled_count += 1

        # Determine if we won
        # OVER bets win if result is 'yes', UNDER bets win if result is 'no'
        won = False
        if side == "OVER" and result == "yes":
            won = True
        elif side == "UNDER" and result == "no":
            won = True

        if won:
            wins += 1
            # Win: we paid price_cents, we get 100 cents back
            # Profit = 100 - price_cents
            net_pnl_cents += (100 - price_cents)
        else:
            losses += 1
            # Loss: we lose what we paid
            net_pnl_cents -= price_cents

    avg_implied_prob = total_implied_prob / total if total > 0 else 0
    avg_edge = total_edge / total if total > 0 else 0
    actual_win_rate = wins / settled_count if settled_count > 0 else 0
    calibration = actual_win_rate - avg_implied_prob

    # ROI = net_pnl / total_risked
    # Total risked = sum of prices paid (assuming 1 bet each)
    total_risked_cents = sum(
        (r.get("limit_price_cents", 50) if is_trades else int(r.get("kalshi_prob", 0.5) * 100))
        for r in records if r.get("result") is not None
    )
    roi_pct = (net_pnl_cents / total_risked_cents * 100) if total_risked_cents > 0 else 0

    return {
        "total_bets": total,
        "wins": wins,
        "losses": losses,
        "unsettled": unsettled,
        "settled": settled_count,
        "avg_implied_prob": avg_implied_prob,
        "actual_win_rate": actual_win_rate,
        "calibration": calibration,
        "net_pnl_cents": net_pnl_cents,
        "roi_pct": roi_pct,
        "avg_edge": avg_edge,
    }


def format_report(
    scan_date: str,
    scanned_stats: Dict[str, Any],
    trade_stats: Dict[str, Any],
) -> str:
    """Format the morning report for Discord."""
    lines = [
        f"**Props Report: {scan_date}**",
        "",
    ]

    # Scanned opportunities section
    lines.append("**All Scanned Opportunities (10%+ EV)**")
    if scanned_stats["total_bets"] == 0:
        lines.append("No opportunities scanned")
    else:
        lines.append(f"Total: {scanned_stats['total_bets']} | Settled: {scanned_stats['settled']} | Unsettled: {scanned_stats['unsettled']}")
        if scanned_stats["settled"] > 0:
            lines.append(f"Record: {scanned_stats['wins']}W - {scanned_stats['losses']}L")
            lines.append(f"Avg Implied: {scanned_stats['avg_implied_prob']:.1%} | Actual: {scanned_stats['actual_win_rate']:.1%}")
            lines.append(f"Avg Edge: {scanned_stats['avg_edge']:.1%} | Calibration: {scanned_stats['calibration']:+.1%}")
            lines.append(f"Net P&L: {scanned_stats['net_pnl_cents']:+.0f}¢ | ROI: {scanned_stats['roi_pct']:+.1f}%")

    lines.append("")

    # Placed trades section
    lines.append("**Placed Trades**")
    if trade_stats["total_bets"] == 0:
        lines.append("No trades placed")
    else:
        lines.append(f"Total: {trade_stats['total_bets']} | Settled: {trade_stats['settled']} | Unsettled: {trade_stats['unsettled']}")
        if trade_stats["settled"] > 0:
            lines.append(f"Record: {trade_stats['wins']}W - {trade_stats['losses']}L")
            lines.append(f"Avg Implied: {trade_stats['avg_implied_prob']:.1%} | Actual: {trade_stats['actual_win_rate']:.1%}")
            lines.append(f"Avg Edge: {trade_stats['avg_edge']:.1%} | Calibration: {trade_stats['calibration']:+.1%}")
            lines.append(f"Net P&L: {trade_stats['net_pnl_cents']:+.0f}¢ | ROI: {trade_stats['roi_pct']:+.1f}%")

    return "\n".join(lines)


def post_discord(message: str) -> None:
    """Post to Discord webhook."""
    if not DISCORD_WEBHOOK:
        print("DISCORD_WEBHOOK not set, skipping Discord post")
        return

    try:
        payload = {"content": message[:1990]}
        r = requests.post(DISCORD_WEBHOOK, json=payload, timeout=10)
        if r.status_code >= 400:
            print(f"Discord post failed: {r.status_code}")
    except Exception as e:
        print(f"Discord post failed: {e}")


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--date", default="", help="Date to analyze (YYYY-MM-DD), default: yesterday")
    parser.add_argument("--skip-fetch", action="store_true", help="Skip fetching results from Kalshi")
    args = parser.parse_args()

    target_date = args.date.strip() or get_yesterday_date()

    print("=" * 60)
    print(f"PROPS MORNING REPORT - {target_date}")
    print("=" * 60)

    # Fetch data
    print("\nFetching scanned opportunities...")
    opportunities = fetch_scanned_opportunities(DB_PATH, target_date)
    print(f"  Found {len(opportunities)} opportunities")

    print("Fetching placed trades...")
    trades = fetch_placed_trades(DB_PATH, target_date)
    print(f"  Found {len(trades)} trades")

    if not opportunities and not trades:
        print("\nNo data for this date.")
        return 0

    # Fetch results from Kalshi
    if not args.skip_fetch:
        print("\nFetching settlement results from Kalshi...")
        try:
            client = KalshiAuthClient.from_env()

            # Get unique tickers
            all_tickers = set()
            for opp in opportunities:
                if not opp.get("result"):
                    all_tickers.add(opp["ticker"])
            for trade in trades:
                if not trade.get("result"):
                    all_tickers.add(trade["ticker"])

            print(f"  Checking {len(all_tickers)} unsettled markets...")

            for ticker in all_tickers:
                result = get_market_result(client, ticker)
                if result:
                    print(f"    {ticker}: {result}")
                    update_opportunity_result(DB_PATH, ticker, target_date, result)
                    update_trade_result(DB_PATH, ticker, target_date, result)

            # Re-fetch with updated results
            opportunities = fetch_scanned_opportunities(DB_PATH, target_date)
            trades = fetch_placed_trades(DB_PATH, target_date)

        except Exception as e:
            print(f"  [warn] Could not fetch Kalshi results: {e}")
            print("  Continuing with cached results...")

    # Calculate stats
    print("\nCalculating statistics...")
    scanned_stats = calculate_stats(opportunities, is_trades=False)
    trade_stats = calculate_stats(trades, is_trades=True)

    # Print report
    report = format_report(target_date, scanned_stats, trade_stats)
    print("\n" + report)

    # Post to Discord
    post_discord(report)
    print("\nPosted to Discord.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
