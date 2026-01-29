#!/usr/bin/env python3
"""
Unified settlement checker for all trade types.

Checks Kalshi for settled markets across:
- Weather trades (weather_trades.db)
- Props trades (props_trades.db)
- Sports trades (sports_trades.db)

Run hourly via GitHub Actions to keep results current.
"""

from __future__ import annotations

import os
import sqlite3
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from kalshi_auth_client import KalshiAuthClient

ET = ZoneInfo("America/New_York")
UTC = ZoneInfo("UTC")

# Database paths
WEATHER_DB = Path(os.getenv("WEATHER_TRADES_DB_PATH", str(ROOT / "weather_trades.db")))
PROPS_DB = ROOT / "props_trades.db"
SPORTS_DB = ROOT / "sports_trades.db"

# Discord webhooks (category-specific or fallback)
DISCORD_WEATHER = os.getenv("DISCORD_WEATHER_ALERTS", os.getenv("DISCORD_WEBHOOK_URL", ""))
DISCORD_PROPS = os.getenv("DISCORD_PROPS_WEBHOOK", os.getenv("DISCORD_WEBHOOK_URL", ""))
DISCORD_SPORTS = os.getenv("DISCORD_SPORTS_WEBHOOK", os.getenv("DISCORD_WEBHOOK_URL", ""))


@dataclass
class SettlementResult:
    """Result of settling trades for a category."""
    category: str
    settled_count: int
    wins: int
    losses: int
    total_cost_cents: int
    total_payout_cents: int
    trades: List[Dict[str, Any]]

    @property
    def pnl_cents(self) -> int:
        return self.total_payout_cents - self.total_cost_cents

    @property
    def roi_pct(self) -> float:
        if self.total_cost_cents == 0:
            return 0.0
        return (self.pnl_cents / self.total_cost_cents) * 100


def post_discord(webhook: str, message: str) -> None:
    """Post to Discord webhook."""
    if not webhook:
        return
    try:
        import requests
        requests.post(webhook, json={"content": message[:1990]}, timeout=10)
    except Exception as e:
        print(f"  [warn] Discord post failed: {e}")


def check_market_settlement(client: KalshiAuthClient, ticker: str) -> Optional[Dict[str, Any]]:
    """Check if a Kalshi market has settled and get the result."""
    try:
        market = client.get_market(ticker)
        status = market.get("status", "").lower()
        result = market.get("result", "").lower()

        if status in ("settled", "finalized", "closed"):
            return {"settled": True, "result": result, "status": status}
        return {"settled": False, "status": status}
    except Exception as e:
        print(f"  [error] Failed to check {ticker}: {e}")
        return None


def ensure_settlement_columns(db_path: Path, table: str, settled_at_col: str = "settled_at") -> None:
    """Add settlement columns if they don't exist (migration)."""
    if not db_path.exists():
        return

    conn = sqlite3.connect(str(db_path))
    cur = conn.cursor()

    columns = {row[1] for row in cur.execute(f"PRAGMA table_info({table});").fetchall()}
    if "settled" not in columns:
        cur.execute(f"ALTER TABLE {table} ADD COLUMN settled INTEGER DEFAULT 0;")
    if "won" not in columns:
        cur.execute(f"ALTER TABLE {table} ADD COLUMN won INTEGER;")
    if "payout_cents" not in columns:
        cur.execute(f"ALTER TABLE {table} ADD COLUMN payout_cents INTEGER;")
    if settled_at_col not in columns:
        cur.execute(f"ALTER TABLE {table} ADD COLUMN {settled_at_col} TEXT;")

    conn.commit()
    conn.close()


# =============================================================================
# Weather Trades
# =============================================================================

def get_unsettled_weather_trades(db_path: Path) -> List[Dict[str, Any]]:
    """Get unsettled weather trades."""
    if not db_path.exists():
        return []

    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()
    cur.execute("""
        SELECT id, trade_date, city_key, market_ticker, event_display, side,
               quantity, limit_price_cents, fill_price_cents, fair_q, ev, status
        FROM weather_trades
        WHERE (settled = 0 OR settled IS NULL)
          AND status IN ('filled', 'FILLED', 'PLACED')
    """)
    rows = [dict(r) for r in cur.fetchall()]
    conn.close()
    return rows


def mark_weather_settled(db_path: Path, trade_id: int, won: bool, payout_cents: int) -> None:
    """Mark a weather trade as settled."""
    now = datetime.now(UTC).isoformat()
    conn = sqlite3.connect(str(db_path))
    cur = conn.cursor()
    cur.execute("""
        UPDATE weather_trades
        SET settled = 1, won = ?, payout_cents = ?, settled_at_utc = ?, updated_at = ?
        WHERE id = ?
    """, (1 if won else 0, payout_cents, now, now, trade_id))
    conn.commit()
    conn.close()


def settle_weather_trades(client: KalshiAuthClient) -> SettlementResult:
    """Settle weather trades."""
    print("\n" + "=" * 50)
    print("WEATHER TRADES")
    print("=" * 50)

    trades = get_unsettled_weather_trades(WEATHER_DB)
    print(f"Found {len(trades)} unsettled trades")

    result = SettlementResult("weather", 0, 0, 0, 0, 0, [])

    for trade in trades:
        ticker = trade["market_ticker"]
        side = trade["side"].upper()
        cost = trade["fill_price_cents"] or trade["limit_price_cents"] or 0

        print(f"  Checking {ticker} ({side})...", end=" ")
        settlement = check_market_settlement(client, ticker)

        if settlement is None or not settlement["settled"]:
            print(f"not settled ({settlement['status'] if settlement else 'error'})")
            continue

        market_result = settlement["result"]
        won = (side == "YES" and market_result == "yes") or \
              (side == "NO" and market_result == "no")
        payout = 100 if won else 0

        mark_weather_settled(WEATHER_DB, trade["id"], won, payout)
        print(f"{'WON' if won else 'LOST'} | P/L: {payout - cost:+d}¢")

        result.settled_count += 1
        if won:
            result.wins += 1
        else:
            result.losses += 1
        result.total_cost_cents += cost
        result.total_payout_cents += payout
        result.trades.append({
            "display": f"{trade['city_key']} {trade['event_display']}",
            "side": side,
            "cost": cost,
            "won": won,
            "ev": trade.get("ev", 0),
        })

    return result


# =============================================================================
# Props Trades
# =============================================================================

def get_unsettled_props_trades(db_path: Path) -> List[Dict[str, Any]]:
    """Get unsettled props trades."""
    if not db_path.exists():
        return []

    ensure_settlement_columns(db_path, "trades")

    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()
    cur.execute("""
        SELECT id, trade_date, ticker, player_name, stat_type, line, side,
               quantity, limit_price_cents, fill_price_cents, fair_prob, edge, status
        FROM trades
        WHERE (settled = 0 OR settled IS NULL)
          AND status IN ('filled', 'FILLED')
    """)
    rows = [dict(r) for r in cur.fetchall()]
    conn.close()
    return rows


def mark_props_settled(db_path: Path, trade_id: int, won: bool, payout_cents: int) -> None:
    """Mark a props trade as settled."""
    now = datetime.now(ET).isoformat()
    conn = sqlite3.connect(str(db_path))
    cur = conn.cursor()
    cur.execute("""
        UPDATE trades
        SET settled = 1, won = ?, payout_cents = ?, settled_at = ?, updated_at = ?
        WHERE id = ?
    """, (1 if won else 0, payout_cents, now, now, trade_id))
    conn.commit()
    conn.close()


def settle_props_trades(client: KalshiAuthClient) -> SettlementResult:
    """Settle props trades."""
    print("\n" + "=" * 50)
    print("PROPS TRADES")
    print("=" * 50)

    trades = get_unsettled_props_trades(PROPS_DB)
    print(f"Found {len(trades)} unsettled trades")

    result = SettlementResult("props", 0, 0, 0, 0, 0, [])

    for trade in trades:
        ticker = trade["ticker"]
        side = trade["side"].upper()
        cost = trade["fill_price_cents"] or trade["limit_price_cents"] or 0
        player = trade["player_name"]
        stat_type = trade["stat_type"]
        line = trade["line"]

        print(f"  Checking {player} {stat_type} {line}+ ({side})...", end=" ")
        settlement = check_market_settlement(client, ticker)

        if settlement is None or not settlement["settled"]:
            print(f"not settled ({settlement['status'] if settlement else 'error'})")
            continue

        market_result = settlement["result"]
        # For props: OVER wins if result is "yes", UNDER wins if result is "no"
        if side == "OVER":
            won = (market_result == "yes")
        else:
            won = (market_result == "no")

        payout = 100 if won else 0
        mark_props_settled(PROPS_DB, trade["id"], won, payout)
        print(f"{'WON' if won else 'LOST'} | P/L: {payout - cost:+d}¢")

        result.settled_count += 1
        if won:
            result.wins += 1
        else:
            result.losses += 1
        result.total_cost_cents += cost
        result.total_payout_cents += payout
        result.trades.append({
            "display": f"{player} {stat_type} {line}+",
            "side": side,
            "cost": cost,
            "won": won,
            "edge": trade.get("edge", 0),
        })

    return result


# =============================================================================
# Sports Trades
# =============================================================================

def get_unsettled_sports_trades(db_path: Path) -> List[Dict[str, Any]]:
    """Get unsettled sports trades."""
    if not db_path.exists():
        return []

    ensure_settlement_columns(db_path, "trades")

    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()
    cur.execute("""
        SELECT id, trade_date, ticker, event_title, line_type, line_label, side,
               quantity, limit_price_cents, fill_price_cents, fair_prob, edge, status
        FROM trades
        WHERE (settled = 0 OR settled IS NULL)
          AND status IN ('filled', 'FILLED')
    """)
    rows = [dict(r) for r in cur.fetchall()]
    conn.close()
    return rows


def mark_sports_settled(db_path: Path, trade_id: int, won: bool, payout_cents: int) -> None:
    """Mark a sports trade as settled."""
    now = datetime.now(ET).isoformat()
    conn = sqlite3.connect(str(db_path))
    cur = conn.cursor()
    cur.execute("""
        UPDATE trades
        SET settled = 1, won = ?, payout_cents = ?, settled_at = ?, updated_at = ?
        WHERE id = ?
    """, (1 if won else 0, payout_cents, now, now, trade_id))
    conn.commit()
    conn.close()


def settle_sports_trades(client: KalshiAuthClient) -> SettlementResult:
    """Settle sports trades."""
    print("\n" + "=" * 50)
    print("SPORTS TRADES")
    print("=" * 50)

    trades = get_unsettled_sports_trades(SPORTS_DB)
    print(f"Found {len(trades)} unsettled trades")

    result = SettlementResult("sports", 0, 0, 0, 0, 0, [])

    for trade in trades:
        ticker = trade["ticker"]
        side = trade["side"].upper()
        cost = trade["fill_price_cents"] or trade["limit_price_cents"] or 0
        event = trade["event_title"]
        line_label = trade["line_label"]

        print(f"  Checking {event} {line_label} ({side})...", end=" ")
        settlement = check_market_settlement(client, ticker)

        if settlement is None or not settlement["settled"]:
            print(f"not settled ({settlement['status'] if settlement else 'error'})")
            continue

        market_result = settlement["result"]
        # We win if our side matches the result
        won = (side == "YES" and market_result == "yes") or \
              (side == "NO" and market_result == "no")

        payout = 100 if won else 0
        mark_sports_settled(SPORTS_DB, trade["id"], won, payout)
        print(f"{'WON' if won else 'LOST'} | P/L: {payout - cost:+d}¢")

        result.settled_count += 1
        if won:
            result.wins += 1
        else:
            result.losses += 1
        result.total_cost_cents += cost
        result.total_payout_cents += payout
        result.trades.append({
            "display": f"{event} {line_label}",
            "side": side,
            "cost": cost,
            "won": won,
            "edge": trade.get("edge", 0),
        })

    return result


# =============================================================================
# Main
# =============================================================================

def post_settlement_discord(result: SettlementResult, webhook: str) -> None:
    """Post settlement summary to Discord."""
    if result.settled_count == 0 or not webhook:
        return

    lines = [
        f"**{result.category.title()} Settlements** ({result.wins}W-{result.losses}L)",
        f"P/L: {result.pnl_cents:+d}¢ | ROI: {result.roi_pct:+.1f}%",
        "",
    ]
    for t in result.trades[:10]:
        icon = "W" if t["won"] else "L"
        lines.append(f"  {icon} | {t['display']} {t['side']} @ {t['cost']}¢")

    if len(result.trades) > 10:
        lines.append(f"  ...and {len(result.trades) - 10} more")

    post_discord(webhook, "\n".join(lines))


def main() -> int:
    print("=" * 60)
    print("UNIFIED SETTLEMENT CHECKER")
    print(f"Time: {datetime.now(ET).strftime('%Y-%m-%d %I:%M %p ET')}")
    print("=" * 60)

    # Initialize Kalshi client
    try:
        client = KalshiAuthClient.from_env()
    except Exception as e:
        print(f"Failed to init Kalshi client: {e}")
        return 1

    # Settle all categories
    weather_result = settle_weather_trades(client)
    props_result = settle_props_trades(client)
    sports_result = settle_sports_trades(client)

    # Summary
    total_settled = weather_result.settled_count + props_result.settled_count + sports_result.settled_count
    total_wins = weather_result.wins + props_result.wins + sports_result.wins
    total_losses = weather_result.losses + props_result.losses + sports_result.losses
    total_pnl = weather_result.pnl_cents + props_result.pnl_cents + sports_result.pnl_cents

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Total settled: {total_settled} trades")
    print(f"Record: {total_wins}W - {total_losses}L")
    print(f"Net P/L: {total_pnl:+d}¢ (${total_pnl / 100:+.2f})")
    print("=" * 60)

    # Post to Discord (category-specific webhooks)
    post_settlement_discord(weather_result, DISCORD_WEATHER)
    post_settlement_discord(props_result, DISCORD_PROPS)
    post_settlement_discord(sports_result, DISCORD_SPORTS)

    # Update dashboard
    if total_settled > 0:
        try:
            from scripts.generate_dashboard_data import update_dashboard
            update_dashboard(quiet=True)
            print("\nDashboard updated")
        except Exception as e:
            print(f"\n[warn] Failed to update dashboard: {e}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
