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

import re

from kalshi_auth_client import KalshiAuthClient

ET = ZoneInfo("America/New_York")
UTC = ZoneInfo("UTC")

# Kalshi ticker patterns for validation
KALSHI_TICKER_PATTERNS = [
    r"^KX[A-Z]+-.+",  # Standard Kalshi format: KXNBAREB-26JAN28-...
    r"^KXHIGH.+",     # Weather high temp markets
]


def is_valid_kalshi_ticker(ticker: str) -> bool:
    """Check if a ticker matches known Kalshi ticker patterns."""
    if not ticker:
        return False
    return any(re.match(pattern, ticker) for pattern in KALSHI_TICKER_PATTERNS)

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
          AND status IN ('filled', 'FILLED', 'PLACED', 'DRY_RUN')
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
# Weather Scanned Opportunities (model accuracy tracking)
# =============================================================================

def get_unsettled_scanned_opportunities(db_path: Path) -> List[Dict[str, Any]]:
    """Get unsettled scanned opportunities for model accuracy tracking."""
    if not db_path.exists():
        return []

    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row

    # Check if table exists
    table_exists = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='weather_scanned_opportunities'"
    ).fetchone()
    if not table_exists:
        conn.close()
        return []

    # Ensure result/settled_at columns exist
    columns = {row[1] for row in conn.execute("PRAGMA table_info(weather_scanned_opportunities);").fetchall()}
    if "result" not in columns:
        conn.execute("ALTER TABLE weather_scanned_opportunities ADD COLUMN result TEXT;")
        conn.commit()
    if "settled_at" not in columns:
        conn.execute("ALTER TABLE weather_scanned_opportunities ADD COLUMN settled_at TEXT;")
        conn.commit()

    cur = conn.cursor()
    cur.execute("""
        SELECT id, scan_date, city_key, market_ticker, event_display, side,
               ask_cents, fair_q, ev
        FROM weather_scanned_opportunities
        WHERE result IS NULL
    """)
    rows = [dict(r) for r in cur.fetchall()]
    conn.close()
    return rows


def settle_scanned_opportunities(client: KalshiAuthClient) -> int:
    """Settle scanned opportunities to track model accuracy."""
    print("\n" + "=" * 50)
    print("WEATHER SCANNED OPPORTUNITIES (model accuracy)")
    print("=" * 50)

    opps = get_unsettled_scanned_opportunities(WEATHER_DB)
    print(f"Found {len(opps)} unsettled scanned opportunities")

    settled_count = 0
    conn = sqlite3.connect(str(WEATHER_DB))
    cur = conn.cursor()

    for opp in opps:
        ticker = opp["market_ticker"]
        side = opp["side"].upper()

        settlement = check_market_settlement(client, ticker)
        if settlement is None or not settlement["settled"]:
            continue

        market_result = settlement["result"]
        won = (side == "YES" and market_result == "yes") or \
              (side == "NO" and market_result == "no")
        result_str = "won" if won else "lost"
        now = datetime.now(UTC).isoformat()

        cur.execute("""
            UPDATE weather_scanned_opportunities
            SET result = ?, settled_at = ?
            WHERE id = ?
        """, (result_str, now, opp["id"]))

        settled_count += 1

    conn.commit()
    conn.close()
    print(f"Settled {settled_count} scanned opportunities")
    return settled_count


# =============================================================================
# NBM Strategy Trades (paper trades for comparison)
# =============================================================================

def get_unsettled_nbm_trades(db_path: Path) -> List[Dict[str, Any]]:
    """Get unsettled NBM strategy trades."""
    if not db_path.exists():
        return []

    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row

    # Check if table exists
    table_exists = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='nbm_strategy_trades'"
    ).fetchone()

    if not table_exists:
        conn.close()
        return []

    cur = conn.cursor()
    cur.execute("""
        SELECT id, trade_date, city_key, market_ticker, strike, side,
               ask_cents, forecast_high, ev_estimate, status
        FROM nbm_strategy_trades
        WHERE (settled = 0 OR settled IS NULL)
    """)
    rows = [dict(r) for r in cur.fetchall()]
    conn.close()
    return rows


def mark_nbm_settled(db_path: Path, trade_id: int, won: bool, payout_cents: int) -> None:
    """Mark an NBM strategy trade as settled."""
    now = datetime.now(UTC).isoformat()
    conn = sqlite3.connect(str(db_path))
    cur = conn.cursor()
    cur.execute("""
        UPDATE nbm_strategy_trades
        SET settled = 1, won = ?, payout_cents = ?, settled_at = ?
        WHERE id = ?
    """, (1 if won else 0, payout_cents, now, trade_id))
    conn.commit()
    conn.close()


def settle_nbm_trades(client: KalshiAuthClient) -> SettlementResult:
    """Settle NBM strategy trades."""
    print("\n" + "=" * 50)
    print("NBM STRATEGY TRADES")
    print("=" * 50)

    trades = get_unsettled_nbm_trades(WEATHER_DB)
    print(f"Found {len(trades)} unsettled trades")

    result = SettlementResult("nbm_strategy", 0, 0, 0, 0, 0, [])

    for trade in trades:
        ticker = trade["market_ticker"]
        side = trade["side"].upper()
        cost = trade["ask_cents"] or 0

        print(f"  Checking {ticker} ({side})...", end=" ")
        settlement = check_market_settlement(client, ticker)

        if settlement is None or not settlement["settled"]:
            print(f"not settled ({settlement['status'] if settlement else 'error'})")
            continue

        market_result = settlement["result"]
        won = (side == "YES" and market_result == "yes") or \
              (side == "NO" and market_result == "no")
        payout = 100 if won else 0

        mark_nbm_settled(WEATHER_DB, trade["id"], won, payout)
        print(f"{'WON' if won else 'LOST'} | P/L: {payout - cost:+d}¢")

        result.settled_count += 1
        if won:
            result.wins += 1
        else:
            result.losses += 1
        result.total_cost_cents += cost
        result.total_payout_cents += payout
        result.trades.append({
            "display": f"{trade['city_key']} {trade['strike']}°F+",
            "side": side,
            "cost": cost,
            "won": won,
            "ev": trade.get("ev_estimate", 0),
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

        # Validate ticker format
        if not is_valid_kalshi_ticker(ticker):
            print(f"INVALID TICKER '{ticker}' - needs manual settlement")
            continue

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

        # Validate ticker format
        if not is_valid_kalshi_ticker(ticker):
            print(f"INVALID TICKER '{ticker}' - needs manual settlement")
            continue

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

def generate_weather_calibration_report(db_path: Path) -> Optional[str]:
    """
    Generate a calibration report comparing model predicted probs vs actual outcomes.

    Uses both weather_trades and weather_scanned_opportunities to get a fuller picture.
    Returns a Discord-friendly string, or None if insufficient data.
    """
    if not db_path.exists():
        return None

    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row

    # Get settled trades with model probabilities
    rows = []

    # From actual trades
    try:
        trades = conn.execute("""
            SELECT fair_q, ev, won, fill_price_cents, limit_price_cents, payout_cents, side
            FROM weather_trades
            WHERE settled = 1 AND fair_q IS NOT NULL AND fair_q > 0
        """).fetchall()
        for t in trades:
            rows.append({"prob": t["fair_q"], "won": bool(t["won"]), "source": "trade"})
    except Exception:
        pass

    # From scanned opportunities
    try:
        opps = conn.execute("""
            SELECT fair_q, ev, result, side
            FROM weather_scanned_opportunities
            WHERE result IS NOT NULL AND fair_q IS NOT NULL AND fair_q > 0
        """).fetchall()
        for o in opps:
            rows.append({"prob": o["fair_q"], "won": o["result"] == "won", "source": "scan"})
    except Exception:
        pass

    conn.close()

    if len(rows) < 10:
        return None

    # Bucket by predicted probability
    prob_buckets = [
        ("5-20%", 0.05, 0.20),
        ("20-40%", 0.20, 0.40),
        ("40-60%", 0.40, 0.60),
        ("60-80%", 0.60, 0.80),
        ("80%+", 0.80, 1.01),
    ]

    lines = [
        "**Weather Model Calibration Report**",
        f"Based on {len(rows)} settled predictions",
        "",
        "```",
        f"{'Predicted':>12} | {'Count':>6} | {'Wins':>5} | {'Actual':>8} | {'Status':>10}",
        "-" * 55,
    ]

    total_brier = 0.0
    for name, low, high in prob_buckets:
        bucket = [r for r in rows if low <= r["prob"] < high]
        if not bucket:
            continue

        wins = sum(1 for r in bucket if r["won"])
        actual_rate = wins / len(bucket)
        expected_rate = (low + min(high, 1.0)) / 2
        diff = actual_rate - expected_rate

        # Calibration assessment
        if abs(diff) < 0.10:
            status = "Good"
        elif diff > 0:
            status = "Overperform"
        else:
            status = "Underperform"

        lines.append(
            f"{name:>12} | {len(bucket):>6} | {wins:>5} | {actual_rate:>7.0%} | {status:>10}"
        )

        # Brier score contribution
        for r in bucket:
            total_brier += (r["prob"] - (1.0 if r["won"] else 0.0)) ** 2

    brier = total_brier / len(rows) if rows else 0
    total_wins = sum(1 for r in rows if r["won"])
    overall_rate = total_wins / len(rows)

    lines.append("-" * 55)
    lines.append(f"{'Overall':>12} | {len(rows):>6} | {total_wins:>5} | {overall_rate:>7.0%} |")
    lines.append(f"Brier Score: {brier:.3f} (lower is better, 0.25 = random)")
    lines.append("```")

    return "\n".join(lines)


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
    nbm_result = settle_nbm_trades(client)
    props_result = settle_props_trades(client)
    sports_result = settle_sports_trades(client)

    # Settle scanned opportunities (model accuracy tracking)
    scanned_settled = settle_scanned_opportunities(client)

    # Summary
    total_settled = weather_result.settled_count + nbm_result.settled_count + props_result.settled_count + sports_result.settled_count
    total_wins = weather_result.wins + nbm_result.wins + props_result.wins + sports_result.wins
    total_losses = weather_result.losses + nbm_result.losses + props_result.losses + sports_result.losses
    total_pnl = weather_result.pnl_cents + nbm_result.pnl_cents + props_result.pnl_cents + sports_result.pnl_cents

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

    # Post calibration report once per day (6pm ET check)
    now_et = datetime.now(ET)
    if now_et.hour == 18:
        try:
            cal_report = generate_weather_calibration_report(WEATHER_DB)
            if cal_report:
                post_discord(DISCORD_WEATHER, cal_report)
                print("\nCalibration report posted to Discord")
        except Exception as e:
            print(f"\n[warn] Failed to generate calibration report: {e}")

    # Update dashboard
    if total_settled > 0 or scanned_settled > 0:
        try:
            from scripts.generate_dashboard_data import update_dashboard
            update_dashboard(quiet=True)
            print("\nDashboard updated")
        except Exception as e:
            print(f"\n[warn] Failed to update dashboard: {e}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
