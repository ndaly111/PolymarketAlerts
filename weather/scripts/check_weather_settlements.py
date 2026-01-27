#!/usr/bin/env python3
"""
Check Kalshi for settled weather markets and update trade results.

Queries unsettled trades from weather_trades.db, checks Kalshi API for
settlement status, and updates win/loss/payout.
"""

from __future__ import annotations

import os
import sqlite3
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from kalshi_auth_client import KalshiAuthClient

TRADES_DB_PATH = Path(os.getenv("WEATHER_TRADES_DB_PATH", str(ROOT / "weather_trades.db")))
DISCORD_WEBHOOK = os.getenv("DISCORD_WEATHER_TRADES_WEBHOOK", os.getenv("DISCORD_WEATHER_ALERTS", os.getenv("DISCORD_WEBHOOK_URL", "")))


def get_unsettled_trades(db_path: Path) -> List[Dict[str, Any]]:
    """Get all trades that haven't been settled yet."""
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()
    cur.execute("""
        SELECT
            id, trade_date, city_key, market_ticker, event_display, side,
            quantity, limit_price_cents, fill_price_cents, fair_q, ev,
            order_id, status
        FROM weather_trades
        WHERE (settled = 0 OR settled IS NULL)
          AND status IN ('filled', 'FILLED', 'PLACED')
    """)
    rows = [dict(r) for r in cur.fetchall()]
    conn.close()
    return rows


def mark_trade_settled(
    db_path: Path,
    trade_id: int,
    won: bool,
    payout_cents: int,
) -> None:
    """Mark a trade as settled."""
    now = datetime.now(ZoneInfo("UTC")).isoformat()
    conn = sqlite3.connect(str(db_path))
    cur = conn.cursor()
    cur.execute("""
        UPDATE weather_trades
        SET settled = 1, won = ?, payout_cents = ?, settled_at_utc = ?, updated_at = ?
        WHERE id = ?
    """, (1 if won else 0, payout_cents, now, now, trade_id))
    conn.commit()
    conn.close()


def check_market_settlement(client: KalshiAuthClient, ticker: str) -> Optional[Dict[str, Any]]:
    """
    Check if a market has settled and get the result.

    Returns dict with 'settled', 'result' ('yes'/'no'), or None if error.
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
        print(f"  [error] Failed to check {ticker}: {e}")
        return None


def post_discord(message: str) -> None:
    """Post to Discord webhook."""
    if not DISCORD_WEBHOOK:
        return
    try:
        import requests
        requests.post(DISCORD_WEBHOOK, json={"content": message[:1990]}, timeout=10)
    except Exception as e:
        print(f"  [warn] Discord post failed: {e}")


def main() -> int:
    print("=" * 50)
    print("WEATHER SETTLEMENT CHECKER")
    print("=" * 50)

    # Get unsettled trades
    trades = get_unsettled_trades(TRADES_DB_PATH)
    print(f"\nFound {len(trades)} unsettled trades")

    if not trades:
        print("No unsettled trades to check.")
        return 0

    # Initialize Kalshi client
    try:
        client = KalshiAuthClient.from_env()
    except Exception as e:
        print(f"Failed to init Kalshi client: {e}")
        return 1

    settled_count = 0
    wins = 0
    losses = 0
    total_payout = 0
    total_cost = 0
    results_summary = []

    for trade in trades:
        ticker = trade["market_ticker"]
        side = trade["side"].upper()
        cost = trade["fill_price_cents"] or trade["limit_price_cents"] or 0
        ev = trade["ev"] or 0
        city = trade["city_key"]
        event = trade["event_display"]

        print(f"\nChecking {ticker} ({side})...")

        result = check_market_settlement(client, ticker)
        if result is None:
            continue

        if not result["settled"]:
            print(f"  Not settled yet (status: {result['status']})")
            continue

        # Determine win/loss
        market_result = result["result"]  # 'yes' or 'no'

        # We win if our side matches the result
        won = (side == "YES" and market_result == "yes") or \
              (side == "NO" and market_result == "no")

        # Payout: $1 (100 cents) if won, $0 if lost
        payout = 100 if won else 0

        mark_trade_settled(TRADES_DB_PATH, trade["id"], won, payout)
        settled_count += 1

        if won:
            wins += 1
            profit = 100 - cost
            print(f"  WIN! Paid {cost}¢, received 100¢, profit: {profit}¢")
        else:
            losses += 1
            print(f"  LOSS. Paid {cost}¢, received 0¢, loss: -{cost}¢")

        total_payout += payout
        total_cost += cost

        results_summary.append({
            "city": city,
            "event": event,
            "side": side,
            "cost": cost,
            "won": won,
            "ev": ev,
        })

    # Summary
    print(f"\n{'=' * 50}")
    print(f"Settled: {settled_count} trades")
    print(f"Record: {wins}W - {losses}L")
    if total_cost > 0:
        roi = ((total_payout - total_cost) / total_cost) * 100
        print(f"ROI: {roi:+.1f}%")
        print(f"P/L: {total_payout - total_cost:+}¢")

    # Post to Discord if any settled
    if settled_count > 0:
        lines = [f"**Weather Settlements** ({settled_count} trades)"]
        lines.append(f"Record: {wins}W - {losses}L")
        if total_cost > 0:
            roi = ((total_payout - total_cost) / total_cost) * 100
            lines.append(f"ROI: {roi:+.1f}% | P/L: {total_payout - total_cost:+}¢")
        lines.append("")
        for r in results_summary[:10]:
            icon = "✅" if r["won"] else "❌"
            lines.append(f"{icon} {r['city']} {r['event']} {r['side']} @ {r['cost']}¢ (EV: {r['ev']:.0%})")

        post_discord("\n".join(lines))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
