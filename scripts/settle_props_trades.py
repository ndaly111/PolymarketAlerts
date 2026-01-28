#!/usr/bin/env python3
"""
Check Kalshi for settled props trades and update the database.
Run after games complete to record win/loss results.
"""

from __future__ import annotations

import os
import sqlite3
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
from zoneinfo import ZoneInfo

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from kalshi_auth_client import KalshiAuthClient

# Configuration
ROOT = Path(__file__).resolve().parents[1]
DB_PATH = ROOT / "props_trades.db"


def ensure_settlement_columns(db_path: Path) -> None:
    """Add settlement columns if they don't exist (migration)."""
    if not db_path.exists():
        return

    conn = sqlite3.connect(str(db_path))
    cur = conn.cursor()

    columns = {row[1] for row in cur.execute("PRAGMA table_info(trades);").fetchall()}
    if "settled" not in columns:
        cur.execute("ALTER TABLE trades ADD COLUMN settled INTEGER DEFAULT 0;")
    if "won" not in columns:
        cur.execute("ALTER TABLE trades ADD COLUMN won INTEGER;")
    if "payout_cents" not in columns:
        cur.execute("ALTER TABLE trades ADD COLUMN payout_cents INTEGER;")
    if "settled_at" not in columns:
        cur.execute("ALTER TABLE trades ADD COLUMN settled_at TEXT;")

    conn.commit()
    conn.close()
DISCORD_WEBHOOK = os.getenv("DISCORD_PROPS_WEBHOOK", os.getenv("DISCORD_WEBHOOK_URL", ""))
ET = ZoneInfo("America/New_York")


def post_discord(message: str) -> None:
    """Post to Discord webhook."""
    if not DISCORD_WEBHOOK:
        return

    try:
        import requests
        payload = {"content": message[:1990]}
        requests.post(DISCORD_WEBHOOK, json=payload, timeout=10)
    except Exception as e:
        print(f"  [warn] Discord post failed: {e}")


def get_unsettled_trades(db_path: Path) -> List[Dict[str, Any]]:
    """Query trades that are filled but not yet settled."""
    if not db_path.exists():
        return []

    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()

    cur.execute("""
        SELECT
            id, trade_date, ticker, player_name, stat_type, line, side,
            quantity, limit_price_cents, fill_price_cents, fair_prob, edge,
            order_id, status
        FROM trades
        WHERE status IN ('filled', 'FILLED')
          AND (settled IS NULL OR settled = 0)
        ORDER BY trade_date ASC
    """)

    trades = [dict(row) for row in cur.fetchall()]
    conn.close()
    return trades


def check_market_settlement(client: KalshiAuthClient, ticker: str) -> Optional[Dict[str, Any]]:
    """
    Check if a Kalshi market has settled and get the result.

    Returns:
        dict with 'settled', 'result' ('yes' or 'no'), 'status', or None if error
    """
    try:
        market = client.get_market(ticker)
        status = market.get("status", "").lower()
        result = market.get("result", "").lower()

        if status in ("settled", "finalized", "closed"):
            return {
                "settled": True,
                "result": result,  # 'yes' or 'no'
                "status": status,
            }
        return {
            "settled": False,
            "status": status,
        }
    except Exception as e:
        print(f"  [error] Failed to check market {ticker}: {e}")
        return None


def mark_trade_settled(
    db_path: Path,
    trade_id: int,
    won: bool,
    payout_cents: int,
) -> None:
    """Update a trade record with settlement result."""
    now = datetime.now(ET).isoformat()

    conn = sqlite3.connect(str(db_path))
    cur = conn.cursor()

    cur.execute("""
        UPDATE trades
        SET settled = 1,
            won = ?,
            payout_cents = ?,
            settled_at = ?,
            updated_at = ?
        WHERE id = ?
    """, (1 if won else 0, payout_cents, now, now, trade_id))

    conn.commit()
    conn.close()


def main() -> int:
    print("=" * 60)
    print("PROPS SETTLEMENT CHECKER")
    print(f"Database: {DB_PATH}")
    print("=" * 60)

    # Ensure settlement columns exist (migration)
    ensure_settlement_columns(DB_PATH)

    # Get unsettled trades
    trades = get_unsettled_trades(DB_PATH)
    print(f"\nFound {len(trades)} unsettled trades")

    if not trades:
        print("No trades to check. Exiting.")
        return 0

    # Initialize Kalshi client
    print("\nInitializing Kalshi client...")
    try:
        client = KalshiAuthClient.from_env()
    except Exception as e:
        print(f"Failed to initialize Kalshi client: {e}")
        return 1

    # Track results for summary
    settled_count = 0
    wins = 0
    losses = 0
    total_pnl_cents = 0
    settled_trades = []

    # Check each trade
    for trade in trades:
        ticker = trade["ticker"]
        player = trade["player_name"]
        stat_type = trade["stat_type"]
        line = trade["line"]
        side = trade["side"]
        cost_cents = trade["fill_price_cents"] or trade["limit_price_cents"]

        print(f"\nChecking: {player} {stat_type} {line}+ {side} ({ticker})")

        result = check_market_settlement(client, ticker)

        if result is None:
            continue

        if not result["settled"]:
            print(f"  Not settled yet (status: {result['status']})")
            continue

        # Determine if we won
        # For OVER bets: we win if result is "yes"
        # For UNDER bets: we win if result is "no"
        market_result = result["result"]
        if side.upper() == "OVER":
            won = (market_result == "yes")
        else:
            won = (market_result == "no")

        # Calculate payout (100 cents if won, 0 if lost)
        payout_cents = 100 if won else 0
        pnl_cents = payout_cents - cost_cents

        print(f"  Settled: {'WON' if won else 'LOST'} | Cost: {cost_cents}¢ | Payout: {payout_cents}¢ | P/L: {pnl_cents:+d}¢")

        # Update database
        mark_trade_settled(DB_PATH, trade["id"], won, payout_cents)

        # Track for summary
        settled_count += 1
        if won:
            wins += 1
        else:
            losses += 1
        total_pnl_cents += pnl_cents
        settled_trades.append({
            "player": player,
            "stat_type": stat_type,
            "line": line,
            "side": side,
            "won": won,
            "pnl_cents": pnl_cents,
        })

    # Print summary
    print("\n" + "=" * 60)
    print(f"Settlement complete: {settled_count} trades settled")
    if settled_count > 0:
        print(f"  Record: {wins}W - {losses}L")
        print(f"  Net P/L: {total_pnl_cents:+d}¢ (${total_pnl_cents / 100:+.2f})")
    print("=" * 60)

    # Post to Discord
    if settled_count > 0:
        timestamp = datetime.now(ET).strftime("%I:%M %p ET")
        lines = [
            f"**Props Settlement Update** ({wins}W-{losses}L)",
            "",
        ]
        for t in settled_trades:
            result_emoji = "W" if t["won"] else "L"
            lines.append(f"  {result_emoji} | {t['player']} {t['stat_type']} {t['line']}+ {t['side']} | {t['pnl_cents']:+d}¢")

        lines.append("")
        lines.append(f"**Net P/L: {total_pnl_cents:+d}¢ (${total_pnl_cents / 100:+.2f})**")
        lines.append(f"_{timestamp}_")

        post_discord("\n".join(lines))

    # Update dashboard JSON
    try:
        from scripts.generate_dashboard_data import update_dashboard
        update_dashboard(quiet=True)
        print("Dashboard data updated")
    except Exception as e:
        print(f"  [warn] Failed to update dashboard: {e}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
