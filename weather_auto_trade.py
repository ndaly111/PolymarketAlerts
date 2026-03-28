#!/usr/bin/env python3
"""
Automated weather trading script.

Workflow:
1. Compute fair prices from NWS forecasts + error models
2. Collect fresh Kalshi weather market quotes
3. Compute edges (fee-adjusted EV)
4. Filter for opportunities: EV >= 10%, q in [5%, 95%], ask <= 85¢
5. Re-fetch fresh Kalshi price before trading
6. Place limit order at ask for N contracts (no fill polling)
7. Post one consolidated Discord summary
8. Max 10 trades/day across all weather markets
"""

from __future__ import annotations

import json
import os
import sqlite3
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
from zoneinfo import ZoneInfo

# Ensure the repo root is in the path
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from kalshi_auth_client import KalshiAuthClient


# --- Configuration ---
MIN_EV = float(
    os.getenv("WEATHER_AUTOTRADE_MIN_EV", os.getenv("WEATHER_MIN_EV", "0.10"))
)  # 10% min EV (matches edge computation)
MIN_Q = float(os.getenv("WEATHER_MIN_Q", "0.05"))  # 5% min probability
MAX_KALSHI_ASK_CENTS = int(os.getenv("WEATHER_MAX_ASK", "85"))  # ≤ 85¢
MAX_TRADES_PER_DAY = int(os.getenv("WEATHER_MAX_TRADES_PER_DAY", "10"))
CONTRACTS_PER_TRADE = int(os.getenv("WEATHER_CONTRACTS_PER_TRADE", "1"))

# Database paths
WEATHER_DB_PATH = Path(os.getenv("WEATHER_DB_PATH", str(ROOT / "weather" / "data" / "weather_forecast_accuracy.db")))
TRADES_DB_PATH = Path(os.getenv("WEATHER_TRADES_DB_PATH", "weather_trades.db"))

# Discord webhook for trade notifications
DISCORD_WEBHOOK = os.getenv("DISCORD_WEATHER_TRADES_WEBHOOK", os.getenv("DISCORD_WEATHER_ALERTS", os.getenv("DISCORD_WEBHOOK_URL", "")))

ENV_DRY_RUN = os.getenv("WEATHER_DRY_RUN", "0") == "1"
ENV_AUTOTRADE_ENABLED = os.getenv("WEATHER_AUTOTRADE_ENABLED", "0") == "1"
# Use environment variables to control trading mode
# Set WEATHER_DRY_RUN=1 for paper trading (logs but no real orders)
# Set WEATHER_AUTOTRADE_ENABLED=1 to enable live trading

# SAFETY LOCK: Hardcoded paper-trading-only mode.
# The Ubuntu server sources .env (not .env.weather) and may have AUTOTRADE_ENABLED=1
# from when the bot was live in January. This override prevents any live orders
# from being placed regardless of env vars until explicitly re-enabled here.
# To re-enable live trading: set _PAPER_ONLY = False AND update .env / GitHub secrets.
_PAPER_ONLY = True

DRY_RUN = _PAPER_ONLY or ENV_DRY_RUN or not ENV_AUTOTRADE_ENABLED
AUTOTRADE_ENABLED = ENV_AUTOTRADE_ENABLED and not _PAPER_ONLY

# Paths for edge artifacts
EDGES_BASE = ROOT / "weather" / "outputs" / "edges"
EDGES_LOW_BASE = ROOT / "weather" / "outputs" / "edges_low"
FAIR_BASE = ROOT / "weather" / "outputs" / "fair_prices"


def ensure_db_schema(db_path: Path) -> None:
    """Create database tables if they don't exist."""
    conn = sqlite3.connect(str(db_path))
    cur = conn.cursor()

    cur.execute("""
        CREATE TABLE IF NOT EXISTS weather_trades (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            trade_date TEXT NOT NULL,
            city_key TEXT NOT NULL,
            market_ticker TEXT NOT NULL,
            event_display TEXT NOT NULL,
            side TEXT NOT NULL,
            quantity INTEGER NOT NULL,
            limit_price_cents INTEGER NOT NULL,
            fair_q REAL NOT NULL,
            ev REAL NOT NULL,
            forecast_high_f INTEGER,
            order_id TEXT,
            status TEXT NOT NULL,
            fill_price_cents INTEGER,
            placed_at_utc TEXT,
            last_checked_at_utc TEXT,
            filled_at_utc TEXT,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            UNIQUE(trade_date, market_ticker)
        )
    """)

    columns = {row[1] for row in cur.execute("PRAGMA table_info(weather_trades);").fetchall()}
    if "placed_at_utc" not in columns:
        cur.execute("ALTER TABLE weather_trades ADD COLUMN placed_at_utc TEXT;")
    if "last_checked_at_utc" not in columns:
        cur.execute("ALTER TABLE weather_trades ADD COLUMN last_checked_at_utc TEXT;")
    if "filled_at_utc" not in columns:
        cur.execute("ALTER TABLE weather_trades ADD COLUMN filled_at_utc TEXT;")
    # Settlement tracking columns
    if "settled" not in columns:
        cur.execute("ALTER TABLE weather_trades ADD COLUMN settled INTEGER DEFAULT 0;")
    if "won" not in columns:
        cur.execute("ALTER TABLE weather_trades ADD COLUMN won INTEGER;")
    if "payout_cents" not in columns:
        cur.execute("ALTER TABLE weather_trades ADD COLUMN payout_cents INTEGER;")
    if "settled_at_utc" not in columns:
        cur.execute("ALTER TABLE weather_trades ADD COLUMN settled_at_utc TEXT;")
    if "metric" not in columns:
        cur.execute("ALTER TABLE weather_trades ADD COLUMN metric TEXT DEFAULT 'high';")

    cur.execute("""
        CREATE TABLE IF NOT EXISTS weather_daily_stats (
            date TEXT PRIMARY KEY,
            trades_attempted INTEGER DEFAULT 0,
            trades_filled INTEGER DEFAULT 0,
            trades_cancelled INTEGER DEFAULT 0,
            total_ev_cents REAL DEFAULT 0
        )
    """)

    cur.execute("""
        CREATE TABLE IF NOT EXISTS weather_scanned_opportunities (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            scan_date TEXT NOT NULL,
            scan_time TEXT NOT NULL,
            city_key TEXT NOT NULL,
            market_ticker TEXT NOT NULL,
            event_display TEXT NOT NULL,
            side TEXT NOT NULL,
            ask_cents INTEGER,
            fair_q REAL NOT NULL,
            ev REAL NOT NULL,
            forecast_high_f INTEGER,
            result TEXT,
            settled_at TEXT,
            UNIQUE(scan_date, market_ticker, side)
        )
    """)

    scanned_cols = {row[1] for row in cur.execute("PRAGMA table_info(weather_scanned_opportunities);").fetchall()}
    if "metric" not in scanned_cols:
        cur.execute("ALTER TABLE weather_scanned_opportunities ADD COLUMN metric TEXT DEFAULT 'high';")

    # Early exit tracking — logs every reassessment for backtesting
    cur.execute("""
        CREATE TABLE IF NOT EXISTS weather_exit_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ts_utc TEXT NOT NULL,
            trade_id INTEGER NOT NULL,
            market_ticker TEXT NOT NULL,
            city_key TEXT NOT NULL,
            side TEXT NOT NULL,
            entry_price_cents INTEGER NOT NULL,
            entry_fair_q REAL NOT NULL,
            current_fair_q REAL,
            current_bid_cents INTEGER,
            fair_value_cents INTEGER,
            ev_shift REAL,
            time_bucket TEXT,
            exit_threshold REAL,
            action TEXT NOT NULL,
            sell_price_cents INTEGER,
            FOREIGN KEY (trade_id) REFERENCES weather_trades(id)
        )
    """)

    # Track consecutive negative scans per position
    cur.execute("""
        CREATE TABLE IF NOT EXISTS weather_exit_streak (
            trade_id INTEGER PRIMARY KEY,
            consecutive_negative INTEGER DEFAULT 0,
            last_checked_utc TEXT
        )
    """)

    conn.commit()
    conn.close()


def get_ev_bucket(ev: float) -> str:
    """Classify EV into bucket."""
    if ev < 0.05:
        return "0-5%"
    elif ev < 0.10:
        return "5-10%"
    elif ev < 0.15:
        return "10-15%"
    elif ev < 0.20:
        return "15-20%"
    else:
        return "20%+"


def get_results_by_ev_bucket(db_path: Path) -> Dict[str, Dict[str, Any]]:
    """
    Get trading results grouped by EV bucket.

    Returns dict with keys like "0-5%", "5-10%", etc.
    Each value contains: trades, wins, losses, pending, total_cost, total_payout, roi
    """
    conn = sqlite3.connect(str(db_path))
    cur = conn.cursor()

    # Get all filled trades with their outcomes
    cur.execute("""
        SELECT
            ev,
            limit_price_cents,
            fill_price_cents,
            settled,
            won,
            payout_cents,
            side,
            fair_q
        FROM weather_trades
        WHERE status IN ('filled', 'FILLED')
    """)
    rows = cur.fetchall()
    conn.close()

    buckets = {
        "0-5%": {"trades": 0, "wins": 0, "losses": 0, "pending": 0, "total_cost_cents": 0, "total_payout_cents": 0, "kalshi_odds": []},
        "5-10%": {"trades": 0, "wins": 0, "losses": 0, "pending": 0, "total_cost_cents": 0, "total_payout_cents": 0, "kalshi_odds": []},
        "10-15%": {"trades": 0, "wins": 0, "losses": 0, "pending": 0, "total_cost_cents": 0, "total_payout_cents": 0, "kalshi_odds": []},
        "15-20%": {"trades": 0, "wins": 0, "losses": 0, "pending": 0, "total_cost_cents": 0, "total_payout_cents": 0, "kalshi_odds": []},
        "20%+": {"trades": 0, "wins": 0, "losses": 0, "pending": 0, "total_cost_cents": 0, "total_payout_cents": 0, "kalshi_odds": []},
    }

    for ev, limit_price, fill_price, settled, won, payout, side, fair_q in rows:
        bucket = get_ev_bucket(ev or 0)
        if bucket not in buckets:
            continue

        b = buckets[bucket]
        b["trades"] += 1

        cost = fill_price if fill_price else limit_price
        b["total_cost_cents"] += cost or 0
        b["kalshi_odds"].append(cost or 0)

        if settled:
            if won:
                b["wins"] += 1
                b["total_payout_cents"] += payout or 100  # Win pays $1 = 100 cents
            else:
                b["losses"] += 1
                b["total_payout_cents"] += 0
        else:
            b["pending"] += 1

    # Calculate ROI for each bucket
    for bucket, data in buckets.items():
        if data["total_cost_cents"] > 0:
            profit = data["total_payout_cents"] - data["total_cost_cents"]
            data["roi_pct"] = (profit / data["total_cost_cents"]) * 100
        else:
            data["roi_pct"] = 0.0

        # Average Kalshi odds
        if data["kalshi_odds"]:
            data["avg_kalshi_cents"] = sum(data["kalshi_odds"]) / len(data["kalshi_odds"])
        else:
            data["avg_kalshi_cents"] = 0
        del data["kalshi_odds"]  # Remove raw list from output

    return buckets


def mark_trade_settled(
    db_path: Path,
    market_ticker: str,
    won: bool,
    payout_cents: int = 0,
) -> None:
    """Mark a trade as settled with win/loss result."""
    now = datetime.now(ZoneInfo("UTC")).isoformat()
    conn = sqlite3.connect(str(db_path))
    cur = conn.cursor()
    cur.execute("""
        UPDATE weather_trades
        SET settled = 1, won = ?, payout_cents = ?, settled_at_utc = ?, updated_at = ?
        WHERE market_ticker = ? AND settled = 0
    """, (1 if won else 0, payout_cents, now, now, market_ticker))
    conn.commit()
    conn.close()


def settle_open_trades(db_path: Path, client: KalshiAuthClient) -> int:
    """Check all unsettled live trades against Kalshi and mark settled ones."""
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    open_trades = conn.execute("""
        SELECT id, market_ticker, side, quantity, trade_date, city_key, fill_price_cents
        FROM weather_trades
        WHERE settled = 0
        AND status NOT IN ('paper_low', 'dry_run', 'paper',
                           'SKIPPED_HAS_POSITION', 'SKIPPED_MIN_EV', 'FAILED_PLACE_ORDER')
    """).fetchall()
    conn.close()

    if not open_trades:
        return 0

    print(f"\n--- Settling open trades: {len(open_trades)} to check ---")
    settled_rows: List[Dict[str, Any]] = []
    now = datetime.now(ZoneInfo("UTC")).isoformat()
    conn = sqlite3.connect(str(db_path))
    try:
        for t in open_trades:
            ticker = t["market_ticker"]
            try:
                mkt = client.get_market(ticker)
                mkt_status = mkt.get("status", "")
                result = mkt.get("result", "")
                if mkt_status in ("settled", "finalized") or result in ("yes", "no"):
                    side = t["side"].upper()
                    won = (side == "YES" and result == "yes") or (side == "NO" and result == "no")
                    payout = 100 * (t["quantity"] or 1) if won else 0
                    conn.execute("""
                        UPDATE weather_trades
                        SET settled=1, won=?, payout_cents=?, settled_at_utc=?, updated_at=?
                        WHERE id=?
                    """, (1 if won else 0, payout, now, now, t["id"]))
                    print(f"  settled {ticker}: {'WIN' if won else 'LOSS'} ({payout}¢)")
                    settled_rows.append({
                        "ticker": ticker,
                        "city": t["city_key"],
                        "date": t["trade_date"],
                        "side": side,
                        "fill": t["fill_price_cents"],
                        "won": won,
                        "payout": payout,
                    })
            except Exception as e:
                print(f"  [warn] Could not check {ticker}: {e}")
        conn.commit()
    finally:
        conn.close()

    n = len(settled_rows)
    print(f"  {n} trade(s) settled.")

    if settled_rows:
        wins = sum(1 for r in settled_rows if r["won"])
        total_cost = sum((r["fill"] or 0) for r in settled_rows)
        total_payout = sum(r["payout"] for r in settled_rows)
        roi = (total_payout - total_cost) / total_cost * 100 if total_cost > 0 else 0

        lines = [f"**Settlement Update — {n} trade(s) resolved**",
                 f"Results: {wins}W / {n - wins}L | Cost: {total_cost}¢ | Payout: {total_payout}¢ | ROI: {roi:+.1f}%", ""]
        for r in settled_rows:
            icon = "✅" if r["won"] else "❌"
            lines.append(f"{icon} {r['ticker']} {r['side']} @ {r['fill']}¢ -> {r['payout']}¢")

        msg = "\n".join(lines)
        if len(msg) > 1990:
            msg = "\n".join(lines[:3]) + f"\n... and {n} more trades settled ({wins}W/{n-wins}L)"
        post_discord(msg)

    return n


def _get_time_bucket_and_threshold(now_et: datetime) -> tuple:
    """Return (time_bucket_name, min_ev_shift_to_trigger_exit) based on time of day.

    Early in the day forecasts are noisy — require large shifts.
    Later in the day forecasts are locked in — smaller shifts trigger exit.
    """
    h = now_et.hour
    if h < 10:
        return "morning", 0.25    # Before 10 AM: require 25pp shift
    elif h < 14:
        return "midday", 0.15     # 10 AM - 2 PM: require 15pp shift
    else:
        return "afternoon", 0.10  # After 2 PM: require 10pp shift


def reassess_positions(
    db_path: Path,
    client: Optional[Any],
    forecast_source: str,
    target_date: str,
) -> List[Dict[str, Any]]:
    """Re-evaluate open positions against fresh fair prices.

    For each held position:
      1. Load the latest fair-price PMF for that city
      2. Recompute fair_q for the held market's event
      3. Compare to entry fair_q — if shifted beyond threshold, consider exit
      4. Check market bid — only sell if bid > current fair value
      5. Require 2+ consecutive negative scans before acting

    All checks are logged to weather_exit_log for later analysis.
    Returns list of exit actions taken.
    """
    import re
    sys.path.insert(0, str(ROOT))
    from weather.lib.kalshi_weather import EventSpec, parse_event_spec_from_ticker, prob_event

    now_utc = datetime.now(ZoneInfo("UTC"))
    now_et = datetime.now(ZoneInfo("America/New_York"))
    time_bucket, exit_threshold = _get_time_bucket_and_threshold(now_et)

    print(f"\n--- Reassessing open positions (time={now_et.strftime('%H:%M')} ET, bucket={time_bucket}, threshold={exit_threshold:.0%}) ---")

    # Load today's open (unsettled, non-paper) positions
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    open_trades = conn.execute("""
        SELECT id, market_ticker, city_key, side, limit_price_cents, fair_q, ev,
               trade_date, event_display, status, metric
        FROM weather_trades
        WHERE settled = 0
        AND trade_date = ?
        AND status IN ('PLACED', 'dry_run', 'DRY_RUN', 'filled', 'FILLED')
    """, (target_date,)).fetchall()
    conn.close()

    if not open_trades:
        print("  No open positions to reassess.")
        return []

    print(f"  {len(open_trades)} open positions to check")

    # Load fresh fair price PMFs per city
    pmf_cache: Dict[str, Dict[int, float]] = {}
    src_slug = forecast_source.replace("/", "_")
    fair_dir = FAIR_BASE / src_slug / target_date

    if fair_dir.exists():
        for json_file in fair_dir.glob("*.json"):
            try:
                data = json.loads(json_file.read_text(encoding="utf-8"))
                city_key = data.get("city_key", json_file.stem)
                metric_key = "pmf_high_f"  # TODO: support low metric
                pmf_raw = data.get(metric_key, {})
                pmf = {int(k): float(v) for k, v in pmf_raw.items() if float(v) > 0}
                if pmf:
                    pmf_cache[city_key] = pmf
            except Exception:
                pass
    print(f"  Loaded PMFs for {len(pmf_cache)} cities")

    exit_actions = []
    conn = sqlite3.connect(str(db_path))

    for trade in open_trades:
        trade_id = trade["id"]
        ticker = trade["market_ticker"]
        city_key = trade["city_key"]
        side = trade["side"]
        entry_price = trade["limit_price_cents"]
        entry_fair_q = trade["fair_q"]

        # Parse event spec from ticker
        spec = parse_event_spec_from_ticker(ticker)
        if spec is None:
            continue

        # Get fresh PMF for this city
        pmf = pmf_cache.get(city_key)
        if not pmf:
            continue

        # Compute current fair probability
        current_fair_q = float(prob_event(pmf, spec))

        # Compute fair value in cents for our side
        if side == "YES":
            entry_fair_cents = int(entry_fair_q * 100)
            current_fair_cents = int(current_fair_q * 100)
            ev_shift = entry_fair_q - current_fair_q  # positive = fair value dropped
        else:
            entry_fair_cents = int((1 - entry_fair_q) * 100)
            current_fair_cents = int((1 - current_fair_q) * 100)
            ev_shift = (1 - current_fair_q) - (1 - entry_fair_q)
            ev_shift = -ev_shift  # positive = our NO position lost value

        # For YES: ev_shift > 0 means fair_q dropped (bad for us)
        # For NO:  ev_shift > 0 means fair_q rose (bad for us)
        # Normalize: positive ev_shift = position losing value
        if side == "NO":
            ev_shift = current_fair_q - entry_fair_q

        # Determine action
        should_exit = ev_shift > exit_threshold
        action = "HOLD"

        # Check consecutive negative streak
        streak_row = conn.execute(
            "SELECT consecutive_negative FROM weather_exit_streak WHERE trade_id = ?",
            (trade_id,)
        ).fetchone()
        prev_streak = streak_row[0] if streak_row else 0

        if should_exit:
            new_streak = prev_streak + 1
        else:
            new_streak = 0

        # Update streak
        conn.execute("""
            INSERT INTO weather_exit_streak (trade_id, consecutive_negative, last_checked_utc)
            VALUES (?, ?, ?)
            ON CONFLICT(trade_id) DO UPDATE SET
                consecutive_negative = ?,
                last_checked_utc = ?
        """, (trade_id, new_streak, now_utc.isoformat(), new_streak, now_utc.isoformat()))

        # Fetch current market bid
        current_bid = None
        sell_price = None
        if client is not None and should_exit and new_streak >= 2:
            try:
                fresh = fetch_fresh_kalshi_price(client, ticker)
                if fresh:
                    if side == "YES":
                        current_bid = fresh["yes_bid"]
                    else:
                        current_bid = fresh["no_bid"]
            except Exception:
                pass

        # Exit conditions:
        # 1. EV shift exceeds time-based threshold
        # 2. Consecutive negative scans >= 2
        # 3. Market bid > current fair value (don't sell below fair)
        # 4. Bid >= 8¢ (worth the effort)
        if should_exit and new_streak >= 2:
            if current_bid is not None and current_bid > current_fair_cents and current_bid >= 8:
                action = "EXIT"
                sell_price = current_bid
            elif current_bid is not None and current_bid < 8:
                action = "HOLD_LOW_BID"
            elif current_bid is not None and current_bid <= current_fair_cents:
                action = "HOLD_BID_BELOW_FAIR"
            else:
                action = "EXIT_CANDIDATE"  # Would exit but no bid data (dry run)

        # For dry run positions, mark as EXIT_CANDIDATE for paper tracking
        if action == "EXIT_CANDIDATE" and trade["status"] in ("dry_run", "DRY_RUN"):
            action = "PAPER_EXIT"
            sell_price = current_fair_cents  # estimate

        status_sym = {"HOLD": ".", "EXIT": "!!", "PAPER_EXIT": "(P)", "EXIT_CANDIDATE": "?",
                      "HOLD_LOW_BID": "~", "HOLD_BID_BELOW_FAIR": "~"}.get(action, "?")
        print(f"  {status_sym} {ticker} {side} | entry_q={entry_fair_q:.1%} now_q={current_fair_q:.1%} "
              f"shift={ev_shift:+.1%} streak={new_streak} bid={current_bid or '?'}¢ -> {action}")

        # Log to exit_log for analysis
        conn.execute("""
            INSERT INTO weather_exit_log
                (ts_utc, trade_id, market_ticker, city_key, side, entry_price_cents,
                 entry_fair_q, current_fair_q, current_bid_cents, fair_value_cents,
                 ev_shift, time_bucket, exit_threshold, action, sell_price_cents)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            now_utc.isoformat(), trade_id, ticker, city_key, side, entry_price,
            entry_fair_q, current_fair_q, current_bid, current_fair_cents,
            ev_shift, time_bucket, exit_threshold, action, sell_price,
        ))

        if action == "EXIT" and client is not None:
            # Place sell order
            try:
                kalshi_side = "yes" if side == "YES" else "no"
                order = client.sell_position(
                    ticker=ticker,
                    side=kalshi_side,
                    quantity=1,
                    limit_price=sell_price,
                )
                order_id = order.get("order_id", order.get("id", ""))
                print(f"    SOLD {ticker} at {sell_price}¢ (order: {order_id})")
                # Mark as settled with early exit
                conn.execute("""
                    UPDATE weather_trades
                    SET settled = 1, won = 0, payout_cents = ?, settled_at_utc = ?, updated_at = ?
                    WHERE id = ?
                """, (sell_price, now_utc.isoformat(), now_utc.isoformat(), trade_id))
            except Exception as e:
                print(f"    [error] Failed to sell {ticker}: {e}")
                action = "EXIT_FAILED"

        if action == "PAPER_EXIT":
            # Mark paper position as early-exited for tracking
            conn.execute("""
                UPDATE weather_trades
                SET settled = 1, won = 0, payout_cents = ?, settled_at_utc = ?, updated_at = ?
                WHERE id = ?
            """, (sell_price, now_utc.isoformat(), now_utc.isoformat(), trade_id))

        if action in ("EXIT", "PAPER_EXIT"):
            exit_actions.append({
                "ticker": ticker,
                "city": city_key,
                "side": side,
                "entry_price": entry_price,
                "entry_fair_q": entry_fair_q,
                "current_fair_q": current_fair_q,
                "sell_price": sell_price,
                "action": action,
                "ev_shift": ev_shift,
                "pnl": (sell_price or 0) - entry_price,
            })

    conn.commit()
    conn.close()

    if exit_actions:
        total_pnl = sum(e["pnl"] for e in exit_actions)
        lines = [f"**Early Exit Alert — {len(exit_actions)} position(s)**"]
        for e in exit_actions:
            tag = "[PAPER]" if e["action"] == "PAPER_EXIT" else "[LIVE]"
            lines.append(
                f"{tag} {e['ticker']} {e['side']} | entry {e['entry_price']}¢ -> exit ~{e['sell_price']}¢ "
                f"| P&L {e['pnl']:+d}¢ | q shifted {e['entry_fair_q']:.0%}->{e['current_fair_q']:.0%}"
            )
        lines.append(f"Total early exit P&L: {total_pnl:+d}¢")
        post_discord("\n".join(lines)[:1990])

    return exit_actions


def format_ev_bucket_report(buckets: Dict[str, Dict[str, Any]]) -> str:
    """Format EV bucket results as a Discord-friendly string."""
    lines = ["**Weather Trades by EV Bucket**", ""]
    lines.append("| EV Bucket | Trades | W-L-P | Avg Ask | ROI |")
    lines.append("|-----------|--------|-------|---------|-----|")

    total_trades = 0
    total_wins = 0
    total_losses = 0
    total_pending = 0
    total_cost = 0
    total_payout = 0

    for bucket in ["0-5%", "5-10%", "10-15%", "15-20%", "20%+"]:
        data = buckets.get(bucket, {})
        trades = data.get("trades", 0)
        wins = data.get("wins", 0)
        losses = data.get("losses", 0)
        pending = data.get("pending", 0)
        avg_ask = data.get("avg_kalshi_cents", 0)
        roi = data.get("roi_pct", 0)

        total_trades += trades
        total_wins += wins
        total_losses += losses
        total_pending += pending
        total_cost += data.get("total_cost_cents", 0)
        total_payout += data.get("total_payout_cents", 0)

        if trades > 0:
            lines.append(f"| {bucket} | {trades} | {wins}-{losses}-{pending} | {avg_ask:.0f}¢ | {roi:+.1f}% |")

    # Total row
    if total_cost > 0:
        total_roi = ((total_payout - total_cost) / total_cost) * 100
    else:
        total_roi = 0
    lines.append(f"| **Total** | {total_trades} | {total_wins}-{total_losses}-{total_pending} | - | {total_roi:+.1f}% |")

    return "\n".join(lines)


def get_trades_today(db_path: Path) -> int:
    """Get number of trades attempted today."""
    today = datetime.now(ZoneInfo("America/New_York")).strftime("%Y-%m-%d")
    conn = sqlite3.connect(str(db_path))
    cur = conn.cursor()
    cur.execute("SELECT trades_attempted FROM weather_daily_stats WHERE date = ?", (today,))
    row = cur.fetchone()
    conn.close()
    return row[0] if row else 0


def increment_trades_today(db_path: Path) -> None:
    """Increment trade counter for today."""
    today = datetime.now(ZoneInfo("America/New_York")).strftime("%Y-%m-%d")
    conn = sqlite3.connect(str(db_path))
    cur = conn.cursor()
    cur.execute("""
        INSERT INTO weather_daily_stats (date, trades_attempted)
        VALUES (?, 1)
        ON CONFLICT(date) DO UPDATE SET trades_attempted = trades_attempted + 1
    """, (today,))
    conn.commit()
    conn.close()


def record_trade(
    db_path: Path,
    city_key: str,
    market_ticker: str,
    event_display: str,
    side: str,
    quantity: int,
    limit_price_cents: int,
    fair_q: float,
    ev: float,
    forecast_high_f: Optional[int],
    order_id: Optional[str],
    status: str,
    fill_price_cents: Optional[int] = None,
    placed_at_utc: Optional[str] = None,
    last_checked_at_utc: Optional[str] = None,
    filled_at_utc: Optional[str] = None,
    metric: str = "high",
) -> None:
    """Record a trade in the database."""
    now = datetime.now(ZoneInfo("UTC")).isoformat()
    today = datetime.now(ZoneInfo("America/New_York")).strftime("%Y-%m-%d")
    if placed_at_utc is None and status in {"PLACED", "dry_run"}:
        placed_at_utc = now

    conn = sqlite3.connect(str(db_path))
    cur = conn.cursor()

    cur.execute("""
        INSERT OR REPLACE INTO weather_trades (
            trade_date, city_key, market_ticker, event_display, side,
            quantity, limit_price_cents, fair_q, ev, forecast_high_f,
            order_id, status, fill_price_cents, placed_at_utc,
            last_checked_at_utc, filled_at_utc, created_at, updated_at, metric
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        today, city_key, market_ticker, event_display, side,
        quantity, limit_price_cents, fair_q, ev, forecast_high_f,
        order_id, status, fill_price_cents, placed_at_utc,
        last_checked_at_utc, filled_at_utc, now, now, metric
    ))

    conn.commit()
    conn.close()


def log_scanned_opportunity(
    db_path: Path,
    city_key: str,
    market_ticker: str,
    event_display: str,
    side: str,
    ask_cents: int,
    fair_q: float,
    ev: float,
    forecast_high_f: Optional[int],
) -> None:
    """Log a scanned opportunity for later analysis."""
    now = datetime.now(ZoneInfo("UTC"))
    scan_date = now.strftime("%Y-%m-%d")
    scan_time = now.isoformat()

    conn = sqlite3.connect(str(db_path))
    cur = conn.cursor()

    cur.execute("""
        INSERT OR REPLACE INTO weather_scanned_opportunities (
            scan_date, scan_time, city_key, market_ticker, event_display, side,
            ask_cents, fair_q, ev, forecast_high_f
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        scan_date, scan_time, city_key, market_ticker, event_display, side,
        ask_cents, fair_q, ev, forecast_high_f
    ))

    conn.commit()
    conn.close()


def already_traded_today(db_path: Path, market_ticker: str) -> bool:
    """Check if we already traded this ticker today."""
    today = datetime.now(ZoneInfo("America/New_York")).strftime("%Y-%m-%d")
    conn = sqlite3.connect(str(db_path))
    cur = conn.cursor()
    cur.execute(
        "SELECT 1 FROM weather_trades WHERE trade_date = ? AND market_ticker = ?",
        (today, market_ticker)
    )
    row = cur.fetchone()
    conn.close()
    return row is not None


def _kalshi_cents(market: dict, key: str, default: int) -> int:
    """Read a bid/ask field from Kalshi, falling back to _dollars string variant."""
    v = market.get(key)
    if v is not None:
        return int(v)
    # Kalshi removed integer fields ~Mar 2026; fall back to dollar-string variants
    dollars_key = f"{key}_dollars"
    dv = market.get(dollars_key)
    if dv is not None:
        try:
            return int(round(float(dv) * 100))
        except (ValueError, TypeError):
            pass
    return default


def fetch_fresh_kalshi_price(client: KalshiAuthClient, ticker: str) -> Optional[Dict[str, int]]:
    """Fetch fresh bid/ask from Kalshi for a specific ticker."""
    try:
        market = client.get_market(ticker)
        return {
            "yes_bid": _kalshi_cents(market, "yes_bid", 0),
            "yes_ask": _kalshi_cents(market, "yes_ask", 100),
            "no_bid": _kalshi_cents(market, "no_bid", 0),
            "no_ask": _kalshi_cents(market, "no_ask", 100),
        }
    except Exception as e:
        print(f"  [error] Failed to fetch Kalshi price for {ticker}: {e}")
        return None


def load_edge_opportunities(forecast_source: str, target_date: str) -> List[Dict[str, Any]]:
    """Load pre-computed edge opportunities from JSON artifacts."""
    opportunities = []
    src_slug = forecast_source.replace("/", "_")
    edges_dir = EDGES_BASE / src_slug / target_date

    if not edges_dir.exists():
        print(f"[warn] No edges directory for {target_date}: {edges_dir}")
        return []

    for json_file in edges_dir.glob("*.json"):
        if json_file.name == "SUMMARY.md":
            continue
        try:
            data = json.loads(json_file.read_text(encoding="utf-8"))
            city_key = data.get("city_key", json_file.stem)
            forecast_high_f = data.get("forecast_high_f")

            for candidate in data.get("candidates", []):
                decision = candidate.get("decision", {})
                ev = decision.get("ev", 0)
                side = decision.get("side_to_buy", "")
                buy_price = decision.get("buy_price", 0)
                q = candidate.get("model_q", 0)

                if ev < MIN_EV:
                    continue
                if q < MIN_Q or q > (1 - MIN_Q):
                    continue

                # Convert price to cents
                ask_cents = int(buy_price * 100) if buy_price else 100

                if ask_cents > MAX_KALSHI_ASK_CENTS:
                    continue

                event = candidate.get("event", {})
                event_display = event.get("desc", "")
                if not event_display:
                    kind = event.get("kind", "")
                    a = event.get("a", 0)
                    b = event.get("b")
                    if kind == "between" and b:
                        event_display = f"{a}-{b}"
                    elif kind in ("ge", "gte"):
                        event_display = f">={a}"
                    else:
                        event_display = str(a)

                opportunities.append({
                    "city_key": city_key,
                    "market_ticker": candidate.get("market_ticker", ""),
                    "title": candidate.get("title", ""),
                    "event_display": event_display,
                    "side": side,
                    "ask_cents": ask_cents,
                    "fair_q": q,
                    "ev": ev,
                    "forecast_high_f": forecast_high_f,
                })
        except Exception as e:
            print(f"  [warn] Error loading {json_file}: {e}")

    # Sort by EV descending
    opportunities.sort(key=lambda x: x.get("ev", 0), reverse=True)
    return opportunities


def load_basket_opportunities(forecast_source: str, target_date: str) -> List[Dict[str, Any]]:
    """Load pre-computed basket opportunities from JSON edge artifacts."""
    baskets = []
    src_slug = forecast_source.replace("/", "_")
    edges_dir = EDGES_BASE / src_slug / target_date

    if not edges_dir.exists():
        return []

    for json_file in edges_dir.glob("*.json"):
        try:
            data = json.loads(json_file.read_text(encoding="utf-8"))
            city_key = data.get("city_key", json_file.stem)
            forecast_high_f = data.get("forecast_high_f")

            for basket in data.get("baskets", []):
                ev = (basket.get("decision") or {}).get("ev", 0)
                if ev < MIN_EV:
                    continue
                baskets.append({
                    "city_key": city_key,
                    "basket": basket,
                    "forecast_high_f": forecast_high_f,
                    "ev": ev,
                })
        except Exception as e:
            print(f"  [warn] Error loading baskets from {json_file}: {e}")

    baskets.sort(key=lambda x: x.get("ev", 0), reverse=True)
    return baskets


def load_low_edge_opportunities(forecast_source: str, target_date: str) -> List[Dict[str, Any]]:
    """Load pre-computed low-temp edge opportunities from JSON artifacts (for paper trading)."""
    opportunities = []
    src_slug = forecast_source.replace("/", "_")
    edges_dir = EDGES_LOW_BASE / src_slug / target_date

    if not edges_dir.exists():
        return []

    for json_file in edges_dir.glob("*.json"):
        try:
            data = json.loads(json_file.read_text(encoding="utf-8"))
            city_key = data.get("city_key", json_file.stem)
            forecast_low_f = data.get("forecast_low_f")

            for candidate in data.get("candidates", []):
                decision = candidate.get("decision", {})
                ev = decision.get("ev", 0)
                side = decision.get("side_to_buy", "")
                buy_price = decision.get("buy_price", 0)
                q = candidate.get("model_q", 0)

                if ev < MIN_EV:
                    continue
                if q < MIN_Q or q > (1 - MIN_Q):
                    continue

                ask_cents = int(buy_price * 100) if buy_price else 100
                if ask_cents > MAX_KALSHI_ASK_CENTS:
                    continue

                event = candidate.get("event", {})
                event_display = event.get("desc", "")
                if not event_display:
                    kind = event.get("kind", "")
                    a = event.get("a", 0)
                    b = event.get("b")
                    if kind == "between" and b:
                        event_display = f"{a}-{b}"
                    elif kind in ("ge", "gte"):
                        event_display = f">={a}"
                    else:
                        event_display = str(a)

                opportunities.append({
                    "city_key": city_key,
                    "market_ticker": candidate.get("market_ticker", ""),
                    "title": candidate.get("title", ""),
                    "event_display": event_display,
                    "side": side,
                    "ask_cents": ask_cents,
                    "fair_q": q,
                    "ev": ev,
                    "forecast_high_f": forecast_low_f,
                    "metric": "low",
                })
        except Exception as e:
            print(f"  [warn] Error loading low edge {json_file}: {e}")

    opportunities.sort(key=lambda x: x.get("ev", 0), reverse=True)
    return opportunities


def execute_paper_trade(
    db_path: Path,
    opportunity: Dict[str, Any],
) -> Dict[str, Any]:
    """Log a paper trade for a low-temp opportunity. Never places a real order."""
    ticker = opportunity["market_ticker"]
    city_key = opportunity["city_key"]
    event_display = opportunity["event_display"]
    side = opportunity["side"]
    fair_q = opportunity["fair_q"]
    ev = opportunity["ev"]
    ask_cents = opportunity.get("ask_cents", 0)
    forecast_low_f = opportunity.get("forecast_high_f")

    if already_traded_today(db_path, ticker):
        return {
            "ticker": ticker,
            "city": city_key,
            "event": event_display,
            "side": side,
            "limit": ask_cents,
            "fair_q": fair_q,
            "ev": ev,
            "order_id": None,
            "status": "SKIPPED_ALREADY_TRADED",
        }

    record_trade(
        db_path, city_key, ticker, event_display, side,
        CONTRACTS_PER_TRADE, ask_cents, fair_q, ev, forecast_low_f,
        "PAPER", "paper_low",
        metric="low",
    )
    return {
        "ticker": ticker,
        "city": city_key,
        "event": event_display,
        "side": side,
        "limit": ask_cents,
        "fair_q": fair_q,
        "ev": ev,
        "order_id": "PAPER",
        "status": "PAPER_LOW",
    }


def execute_basket(
    client: KalshiAuthClient,
    db_path: Path,
    basket_opp: Dict[str, Any],
    trade_enabled: bool,
) -> List[Dict[str, Any]]:
    """Place one YES order per qualifying leg in the basket.

    Each leg counts as 1 toward the daily trade limit.
    Skips legs with fresh ask < 5¢ (junk/already-resolved markets).
    Returns a result dict per leg processed.
    """
    basket = basket_opp["basket"]
    city_key = basket_opp["city_key"]
    forecast_high_f = basket_opp.get("forecast_high_f")
    basket_ticker = basket.get("market_ticker", "")
    basket_desc = (basket.get("event") or {}).get("desc", basket_ticker)
    basket_ev = basket_opp.get("ev", 0)

    print(f"\n--- Processing basket: {city_key} | {basket_desc} ---")
    print(f"  {len(basket.get('legs', []))} legs | basket EV: {basket_ev:.1%}")

    results = []
    for leg in basket.get("legs", []):
        ticker = leg["market_ticker"]
        ask_cents = leg["ask_cents"]
        leg_ev = leg.get("ev", 0)
        fair_q = leg.get("q", 0)

        # Skip junk/already-resolved markets
        if ask_cents < 5:
            print(f"  [skip leg] {ticker} ask {ask_cents}¢ < 5¢ (junk market)")
            continue

        if already_traded_today(db_path, ticker):
            print(f"  [skip leg] Already traded {ticker} today")
            results.append({
                "ticker": ticker,
                "city": city_key,
                "event": basket_desc,
                "side": "YES",
                "limit": ask_cents,
                "fair_q": fair_q,
                "ev": leg_ev,
                "order_id": None,
                "status": "SKIPPED_ALREADY_TRADED",
            })
            continue

        # Re-fetch fresh price if client available; apply min-ask guard
        if client is not None:
            fresh = fetch_fresh_kalshi_price(client, ticker)
            if fresh:
                fresh_ask = fresh["yes_ask"]
                if fresh_ask < 5:
                    print(f"  [skip leg] {ticker} fresh ask {fresh_ask}¢ < 5¢")
                    continue
                ask_cents = fresh_ask
        print(f"  Leg: {ticker} YES at {ask_cents}¢ | leg EV {leg_ev:.1%}")

        if not trade_enabled:
            record_trade(
                db_path, city_key, ticker, basket_desc, "YES",
                CONTRACTS_PER_TRADE, ask_cents, fair_q, leg_ev, forecast_high_f,
                None, "SKIPPED_AUTOTRADE_DISABLED",
            )
            results.append({
                "ticker": ticker,
                "city": city_key,
                "event": basket_desc,
                "side": "YES",
                "limit": ask_cents,
                "fair_q": fair_q,
                "ev": leg_ev,
                "order_id": None,
                "status": "SKIPPED_AUTOTRADE_DISABLED",
            })
            continue

        if DRY_RUN:
            print(f"  [DRY_RUN] Would place {ticker} YES at {ask_cents}¢")
            record_trade(
                db_path, city_key, ticker, basket_desc, "YES",
                CONTRACTS_PER_TRADE, ask_cents, fair_q, leg_ev, forecast_high_f,
                "DRY_RUN", "dry_run",
            )
            results.append({
                "ticker": ticker,
                "city": city_key,
                "event": basket_desc,
                "side": "YES",
                "limit": ask_cents,
                "fair_q": fair_q,
                "ev": leg_ev,
                "order_id": "DRY_RUN",
                "status": "DRY_RUN",
            })
            continue

        try:
            order = client.place_order(
                ticker=ticker,
                side="yes",
                quantity=CONTRACTS_PER_TRADE,
                limit_price=ask_cents,
            )
            order_id = order.get("order_id", order.get("id", ""))
            print(f"    Order placed: {order_id}")
            record_trade(
                db_path, city_key, ticker, basket_desc, "YES",
                CONTRACTS_PER_TRADE, ask_cents, fair_q, leg_ev, forecast_high_f,
                order_id, "PLACED",
            )
            results.append({
                "ticker": ticker,
                "city": city_key,
                "event": basket_desc,
                "side": "YES",
                "limit": ask_cents,
                "fair_q": fair_q,
                "ev": leg_ev,
                "order_id": order_id,
                "status": "PLACED",
            })
        except Exception as e:
            print(f"  [error] Failed to place leg {ticker}: {e}")
            results.append({
                "ticker": ticker,
                "city": city_key,
                "event": basket_desc,
                "side": "YES",
                "limit": ask_cents,
                "fair_q": fair_q,
                "ev": leg_ev,
                "order_id": None,
                "status": "FAILED_PLACE_ORDER",
                "error": str(e)[:120],
            })

    return results


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


def format_summary_message(
    results: List[Dict[str, Any]],
    target_date: str,
    forecast_source: str,
) -> str:
    placed = [r for r in results if r["status"] in {"PLACED", "DRY_RUN"}]
    skipped = [r for r in results if r["status"].startswith("SKIPPED")]
    failed = [r for r in results if r["status"].startswith("FAILED")]

    header_lines = [
        f"**Weather Auto-Trade Summary — {target_date}**",
        f"Forecast source: {forecast_source}",
        f"Min EV: {MIN_EV:.0%} | Max ask: {MAX_KALSHI_ASK_CENTS}¢ | Live limit: {MAX_TRADES_PER_DAY}/day",
        f"Placed: {len(placed)} | Skipped: {len(skipped)} | Failed: {len(failed)}",
    ]
    if DRY_RUN:
        header_lines.append("Mode: DRY_RUN (no orders submitted)")
    elif not AUTOTRADE_ENABLED:
        header_lines.append("Mode: AUTOTRADE DISABLED (no orders submitted)")

    sections: List[str] = ["\n".join(header_lines)]

    def add_section(title: str, rows: List[str]) -> None:
        if not rows:
            return
        sections.append(f"\n{title}\n" + "\n".join(rows))

    def format_ev(value: Optional[float]) -> str:
        if value is None:
            return "n/a"
        return f"{value:.1%}"

    placed_rows = [
        (
            f"- {r['ticker']} {r['side']} | {r['limit']}¢ | EV {format_ev(r.get('ev'))} | "
            f"Order {r['order_id']}"
        )
        for r in placed
    ]
    skipped_rows = [
        (
            f"- {r['ticker']} {r['side']} | "
            f"{r['status']} ({format_ev(r.get('ev'))})"
        )
        for r in skipped
    ]
    failed_rows = [
        (
            f"- {r['ticker']} {r['side']} | "
            f"{r['status']} ({r.get('error', 'unknown error')})"
        )
        for r in failed
    ]

    add_section("✅ PLACED", placed_rows)
    add_section("⏭️ SKIPPED", skipped_rows)
    add_section("❌ FAILED", failed_rows)

    message = "\n".join(sections)
    if len(message) <= 1990:
        return message

    trimmed_sections: List[str] = [sections[0]]
    for title, rows in [("✅ PLACED", placed_rows), ("⏭️ SKIPPED", skipped_rows), ("❌ FAILED", failed_rows)]:
        if not rows:
            continue
        preview = rows[:5]
        remaining = len(rows) - len(preview)
        block = f"\n{title}\n" + "\n".join(preview)
        if remaining > 0:
            block += f"\n- ... and {remaining} more"
        trimmed_sections.append(block)

    trimmed_message = "\n".join(trimmed_sections)
    return trimmed_message[:1990]


def execute_trade(
    client: KalshiAuthClient,
    db_path: Path,
    opportunity: Dict[str, Any],
    trade_enabled: bool,
) -> Dict[str, Any]:
    """
    Execute the trade workflow for a single opportunity.

    Returns a result dict summarizing the outcome.
    """
    ticker = opportunity["market_ticker"]
    city_key = opportunity["city_key"]
    event_display = opportunity["event_display"]
    side = opportunity["side"]
    fair_q = opportunity["fair_q"]
    pre_ev = opportunity["ev"]
    forecast_high_f = opportunity.get("forecast_high_f")

    print(f"\n--- Processing: {city_key} | {event_display} ({side}) ---")

    # Check if already traded today
    if already_traded_today(db_path, ticker):
        print(f"  [skip] Already traded {ticker} today")
        return {
            "ticker": ticker,
            "city": city_key,
            "event": event_display,
            "side": side,
            "limit": None,
            "fair_q": fair_q,
            "ev": pre_ev,
            "order_id": None,
            "status": "SKIPPED_ALREADY_TRADED",
        }

    # Check if we have existing position
    try:
        if client.has_position(ticker):
            print(f"  [skip] Already have position in {ticker}")
            record_trade(
                db_path, city_key, ticker, event_display, side,
                0, 0, fair_q, pre_ev, forecast_high_f,
                None, "SKIPPED_HAS_POSITION"
            )
            return {
                "ticker": ticker,
                "city": city_key,
                "event": event_display,
                "side": side,
                "limit": None,
                "fair_q": fair_q,
                "ev": pre_ev,
                "order_id": None,
                "status": "SKIPPED_HAS_POSITION",
            }
    except Exception as e:
        print(f"  [warn] Could not check position: {e}")

    # Re-fetch fresh Kalshi orderbook
    print("  Fetching fresh Kalshi price...")
    fresh_kalshi = fetch_fresh_kalshi_price(client, ticker)
    if not fresh_kalshi:
        print(f"  [skip] Could not get fresh Kalshi price")
        record_trade(
            db_path, city_key, ticker, event_display, side,
            0, 0, fair_q, pre_ev, forecast_high_f,
            None, "SKIPPED_NO_PRICE"
        )
        return {
            "ticker": ticker,
            "city": city_key,
            "event": event_display,
            "side": side,
            "limit": None,
            "fair_q": fair_q,
            "ev": pre_ev,
            "order_id": None,
            "status": "SKIPPED_NO_PRICE",
        }

    # Determine which price to use based on side
    if side == "YES":
        kalshi_ask = fresh_kalshi["yes_ask"]
        kalshi_side = "yes"
    else:
        kalshi_ask = fresh_kalshi["no_ask"]
        kalshi_side = "no"

    print(f"  Fresh Kalshi ask: {kalshi_ask}¢")

    # YES price floor — cheap YES longshots (<15c) have ~3% historical win rate
    MIN_YES_PRICE_CENTS = 15
    if side == "YES" and kalshi_ask < MIN_YES_PRICE_CENTS:
        print(f"  [skip] YES ask {kalshi_ask}¢ < {MIN_YES_PRICE_CENTS}¢ floor (cheap longshots ~3% WR)")
        record_trade(
            db_path, city_key, ticker, event_display, side,
            0, kalshi_ask, fair_q, pre_ev, forecast_high_f,
            None, "SKIPPED_YES_FLOOR"
        )
        return {
            "ticker": ticker,
            "city": city_key,
            "event": event_display,
            "side": side,
            "limit": kalshi_ask,
            "fair_q": fair_q,
            "ev": pre_ev,
            "order_id": None,
            "status": "SKIPPED_YES_FLOOR",
        }

    # Side-specific max ask: NO trades capped at 65c (expensive NOs lose money
    # despite high WR), YES uses global MAX_KALSHI_ASK_CENTS
    MAX_NO_ASK_CENTS = 65
    effective_max_ask = MAX_NO_ASK_CENTS if side == "NO" else MAX_KALSHI_ASK_CENTS

    # Check price hasn't moved too much
    if kalshi_ask > effective_max_ask:
        print(f"  [skip] Kalshi ask {kalshi_ask}¢ > {effective_max_ask}¢ limit ({side})")
        record_trade(
            db_path, city_key, ticker, event_display, side,
            0, kalshi_ask, fair_q, pre_ev, forecast_high_f,
            None, "SKIPPED_MAX_ASK"
        )
        return {
            "ticker": ticker,
            "city": city_key,
            "event": event_display,
            "side": side,
            "limit": kalshi_ask,
            "fair_q": fair_q,
            "ev": pre_ev,
            "order_id": None,
            "status": "SKIPPED_MAX_ASK",
        }

    # Recalculate EV with fresh price
    kalshi_prob = kalshi_ask / 100.0
    if side == "YES":
        # EV = q * (1 - price) - (1-q) * price - fee
        ev = fair_q * (1.0 - kalshi_prob) - (1 - fair_q) * kalshi_prob - 0.02  # 2¢ fee
    else:
        # EV for NO = (1-q) * (1 - price) - q * price - fee
        ev = (1 - fair_q) * (1.0 - kalshi_prob) - fair_q * kalshi_prob - 0.02

    print(f"  Fair q: {fair_q:.1%}, Fresh price: {kalshi_prob:.1%}, EV: {ev:.1%}")

    if ev < MIN_EV:
        print(f"  [skip] Fresh EV {ev:.1%} < {MIN_EV:.1%} threshold")
        record_trade(
            db_path, city_key, ticker, event_display, side,
            0, kalshi_ask, fair_q, ev, forecast_high_f,
            None, "SKIPPED_MIN_EV"
        )
        return {
            "ticker": ticker,
            "city": city_key,
            "event": event_display,
            "side": side,
            "limit": kalshi_ask,
            "fair_q": fair_q,
            "ev": ev,
            "order_id": None,
            "status": "SKIPPED_MIN_EV",
        }

    # All checks passed - place order
    print(f"  Placing limit order: {CONTRACTS_PER_TRADE} contract(s) at {kalshi_ask}¢")

    if not trade_enabled:
        print("  [skip] Auto-trade disabled")
        record_trade(
            db_path, city_key, ticker, event_display, side,
            CONTRACTS_PER_TRADE, kalshi_ask, fair_q, ev, forecast_high_f,
            None, "SKIPPED_AUTOTRADE_DISABLED"
        )
        return {
            "ticker": ticker,
            "city": city_key,
            "event": event_display,
            "side": side,
            "limit": kalshi_ask,
            "fair_q": fair_q,
            "ev": ev,
            "order_id": None,
            "status": "SKIPPED_AUTOTRADE_DISABLED",
        }

    if DRY_RUN:
        print("  [DRY_RUN] Would place order")
        record_trade(
            db_path, city_key, ticker, event_display, side,
            CONTRACTS_PER_TRADE, kalshi_ask, fair_q, ev, forecast_high_f,
            "DRY_RUN", "dry_run"
        )
        return {
            "ticker": ticker,
            "city": city_key,
            "event": event_display,
            "side": side,
            "limit": kalshi_ask,
            "fair_q": fair_q,
            "ev": ev,
            "order_id": "DRY_RUN",
            "status": "DRY_RUN",
        }

    try:
        order = client.place_order(
            ticker=ticker,
            side=kalshi_side,
            quantity=CONTRACTS_PER_TRADE,
            limit_price=kalshi_ask,
        )
        order_id = order.get("order_id", order.get("id", ""))
        print(f"  Order placed: {order_id}")
        record_trade(
            db_path, city_key, ticker, event_display, side,
            CONTRACTS_PER_TRADE, kalshi_ask, fair_q, ev, forecast_high_f,
            order_id, "PLACED"
        )
        return {
            "ticker": ticker,
            "city": city_key,
            "event": event_display,
            "side": side,
            "limit": kalshi_ask,
            "fair_q": fair_q,
            "ev": ev,
            "order_id": order_id,
            "status": "PLACED",
        }

    except Exception as e:
        print(f"  [error] Failed to place order: {e}")
        record_trade(
            db_path, city_key, ticker, event_display, side,
            CONTRACTS_PER_TRADE, kalshi_ask, fair_q, ev, forecast_high_f,
            None, "FAILED_PLACE_ORDER"
        )
        return {
            "ticker": ticker,
            "city": city_key,
            "event": event_display,
            "side": side,
            "limit": kalshi_ask,
            "fair_q": fair_q,
            "ev": ev,
            "order_id": None,
            "status": "FAILED_PLACE_ORDER",
            "error": str(e)[:120],
        }


def run_data_pipeline(target_date: str, forecast_source: str) -> bool:
    """
    Run the data collection and fair price computation pipeline.
    Returns True if successful.
    """
    import subprocess

    print("\n--- Running data pipeline ---")

    # 1. Collect fresh NWS forecast snapshot
    print("Collecting fresh NWS forecast...")
    result = subprocess.run([
        sys.executable, "-m", "weather.scripts.collect_forecast_snapshot",
        "--no-gate", "--use-local-hour",
        "--source", forecast_source,
        "--target-date", target_date,
        "--db", str(WEATHER_DB_PATH),
    ], capture_output=True, text=True, cwd=str(ROOT), env={**os.environ, "PYTHONPATH": str(ROOT)})
    if result.returncode != 0:
        print(f"  [warn] Forecast collection: {result.stderr[:200]}")
    else:
        print("  Forecast collection OK")

    # 2. Collect intraday observations
    print("Collecting intraday observations...")
    result = subprocess.run([
        sys.executable, "-m", "weather.scripts.collect_intraday_observations",
        "--no-gate",
        "--db", str(WEATHER_DB_PATH),
    ], capture_output=True, text=True, cwd=str(ROOT), env={**os.environ, "PYTHONPATH": str(ROOT)})
    if result.returncode != 0:
        print(f"  [warn] Intraday collection: {result.stderr[:200]}")
    else:
        print("  Intraday collection OK")

    # 3. Compute fair prices
    print("Computing fair prices...")
    result = subprocess.run([
        sys.executable, "-m", "weather.scripts.compute_fair_prices",
        "--forecast-source", forecast_source,
        "--use-latest",
        "--db", str(WEATHER_DB_PATH),
    ], capture_output=True, text=True, cwd=str(ROOT), env={
        **os.environ,
        "PYTHONPATH": str(ROOT),
        "WEATHER_ERROR_MODEL_SOURCE": "mos_gfs_18z_archive",
        "WEATHER_ERROR_MODEL_HOUR": "12",
        "WEATHER_ENABLE_INTRADAY_ADJUSTMENT": "true",
    })
    if result.returncode != 0:
        print(f"  [error] Fair price computation failed: {result.stderr[:300]}")
        return False
    print("  Fair prices computed OK")

    # 4. Collect Kalshi weather markets
    print("Collecting Kalshi weather markets...")
    result = subprocess.run([
        sys.executable, "-m", "weather.scripts.collect_kalshi_weather_markets",
        "--require-city-match",
        "--db", str(WEATHER_DB_PATH),
    ], capture_output=True, text=True, cwd=str(ROOT), env={**os.environ, "PYTHONPATH": str(ROOT)})
    if result.returncode != 0:
        print(f"  [error] Kalshi collection failed: {result.stderr[:300]}")
        return False

    # Parse output to check if markets were found
    try:
        output = json.loads(result.stdout)
        rows_written = output.get("rows_written", 0)
        print(f"  Kalshi markets collected: {rows_written} rows")
        if rows_written == 0:
            print("  [warn] No Kalshi markets found")
            return False
    except Exception:
        print(f"  [warn] Could not parse Kalshi output: {result.stdout[:200]}")

    # 5. Compute edges
    print("Computing edges...")
    result = subprocess.run([
        sys.executable, "-m", "weather.scripts.compute_edges",
        "--forecast-source", forecast_source,
        "--date", target_date,
        "--require-ask",
        "--db", str(WEATHER_DB_PATH),
    ], capture_output=True, text=True, cwd=str(ROOT), env={
        **os.environ,
        "PYTHONPATH": str(ROOT),
        "WEATHER_MIN_EV": str(MIN_EV),
        "WEATHER_MIN_Q": str(MIN_Q),
        "WEATHER_BUY_FEE_CENTS": "2",
    })
    if result.returncode != 0:
        print(f"  [error] Edge computation failed: {result.stderr[:300]}")
        return False
    print("  Edges computed OK")

    # 6. Compute low fair prices (for paper trading)
    print("Computing low fair prices...")
    result = subprocess.run([
        sys.executable, "-m", "weather.scripts.compute_fair_prices",
        "--forecast-source", forecast_source,
        "--use-latest",
        "--metric", "low",
        "--db", str(WEATHER_DB_PATH),
    ], capture_output=True, text=True, cwd=str(ROOT), env={
        **os.environ,
        "PYTHONPATH": str(ROOT),
    })
    if result.returncode != 0:
        print(f"  [warn] Low fair price computation: {result.stderr[:200]}")
    else:
        print("  Low fair prices computed OK")

    # 7. Compute low edges (for paper trading)
    print("Computing low edges...")
    result = subprocess.run([
        sys.executable, "-m", "weather.scripts.compute_edges",
        "--forecast-source", forecast_source,
        "--date", target_date,
        "--metric", "low",
        "--db", str(WEATHER_DB_PATH),
    ], capture_output=True, text=True, cwd=str(ROOT), env={
        **os.environ,
        "PYTHONPATH": str(ROOT),
        "WEATHER_MIN_EV": str(MIN_EV),
        "WEATHER_MIN_Q": str(MIN_Q),
        "WEATHER_BUY_FEE_CENTS": "2",
    })
    if result.returncode != 0:
        print(f"  [warn] Low edge computation: {result.stderr[:200]}")
    else:
        print("  Low edges computed OK")

    return True


def main() -> int:
    print("=" * 60)
    print("WEATHER AUTO TRADER")
    print(f"Min EV threshold: {MIN_EV:.0%}")
    print(f"Min q: {MIN_Q:.1%}")
    print(f"Max Kalshi ask: {MAX_KALSHI_ASK_CENTS}¢")
    print(f"Contracts per trade: {CONTRACTS_PER_TRADE}")
    print(f"Max live trades/day: {MAX_TRADES_PER_DAY}")
    print(f"Dry run: {DRY_RUN}")
    print(f"Autotrade enabled: {AUTOTRADE_ENABLED}")
    print("=" * 60)

    # Initialize
    ensure_db_schema(TRADES_DB_PATH)

    # Check daily limit
    trades_today = get_trades_today(TRADES_DB_PATH)
    print(f"\nLive trades today: {trades_today}/{MAX_TRADES_PER_DAY}")
    remaining_trades = max(0, MAX_TRADES_PER_DAY - trades_today)

    # Get target date (today in ET)
    now_et = datetime.now(ZoneInfo("America/New_York"))
    target_date = now_et.strftime("%Y-%m-%d")
    # Default to open_meteo for better accuracy (can override with WEATHER_FORECAST_SOURCE)
    forecast_source = os.getenv("WEATHER_FORECAST_SOURCE", "open_meteo")

    print(f"\nTarget date: {target_date}")
    print(f"Forecast source: {forecast_source}")

    client = None
    if AUTOTRADE_ENABLED or DRY_RUN:
        # Initialize Kalshi client
        print("\nInitializing Kalshi client...")
        try:
            client = KalshiAuthClient.from_env()
            balance = client.get_balance()
            available = balance.get("available_balance", balance.get("balance", 0)) / 100
            print(f"Balance: ${available:.2f}")
        except Exception as e:
            print(f"Failed to initialize Kalshi client: {e}")
            return 1

    # Settle any open trades from previous days
    if client:
        settle_open_trades(TRADES_DB_PATH, client)

    # Run data pipeline to get fresh prices and edges
    if not run_data_pipeline(target_date, forecast_source):
        print("\n[error] Data pipeline failed. Exiting.")
        return 1

    # Reassess existing positions with fresh fair prices (early exit logic)
    exit_actions = reassess_positions(
        db_path=TRADES_DB_PATH,
        client=client,
        forecast_source=forecast_source,
        target_date=target_date,
    )
    if exit_actions:
        print(f"\n  Early exits: {len(exit_actions)}")

    # Load opportunities from edge artifacts
    print("\n--- Loading opportunities ---")
    opportunities = load_edge_opportunities(forecast_source, target_date)
    print(f"Found {len(opportunities)} opportunities meeting criteria")

    if not opportunities:
        print("\nNo opportunities found. Exiting.")
        return 0

    # Log all opportunities for analysis
    print(f"\nLogging {len(opportunities)} opportunities to database...")
    for opp in opportunities:
        try:
            log_scanned_opportunity(
                db_path=TRADES_DB_PATH,
                city_key=opp.get("city_key", ""),
                market_ticker=opp.get("market_ticker", ""),
                event_display=opp.get("event_display", ""),
                side=opp.get("side", ""),
                ask_cents=opp.get("ask_cents", 0),
                fair_q=opp.get("fair_q", 0.5),
                ev=opp.get("ev", 0),
                forecast_high_f=opp.get("forecast_high_f"),
            )
        except Exception as e:
            print(f"  [warn] Failed to log opportunity: {e}")

    # Process opportunities — live up to limit, paper the rest
    print(f"\n--- Processing {len(opportunities)} opportunities (live slots: {remaining_trades}) ---")

    trades_placed = 0
    results: List[Dict[str, Any]] = []
    for opp in opportunities:
        live = (AUTOTRADE_ENABLED or DRY_RUN) and trades_placed < remaining_trades
        if live:
            result = execute_trade(client, TRADES_DB_PATH, opp, trade_enabled=AUTOTRADE_ENABLED or DRY_RUN)
            results.append(result)
            if result["status"] in {"PLACED", "DRY_RUN"}:
                trades_placed += 1
                increment_trades_today(TRADES_DB_PATH)
        else:
            result = execute_paper_trade(TRADES_DB_PATH, opp)
            results.append(result)

    # Load and process basket opportunities
    basket_opportunities = load_basket_opportunities(forecast_source, target_date)
    print(f"\nFound {len(basket_opportunities)} basket opportunities")

    for basket_opp in basket_opportunities:
        n_legs = len(basket_opp["basket"].get("legs", []))
        if trades_placed + n_legs <= remaining_trades and (AUTOTRADE_ENABLED or DRY_RUN):
            leg_results = execute_basket(client, TRADES_DB_PATH, basket_opp, trade_enabled=AUTOTRADE_ENABLED or DRY_RUN)
            placed_in_basket = sum(1 for r in leg_results if r["status"] in {"PLACED", "DRY_RUN"})
            trades_placed += placed_in_basket
            for _ in range(placed_in_basket):
                increment_trades_today(TRADES_DB_PATH)
        else:
            leg_results = [execute_paper_trade(TRADES_DB_PATH, {"market_ticker": l["market_ticker"], "city_key": basket_opp["city_key"], "event_display": (basket_opp["basket"].get("event") or {}).get("desc", ""), "side": "YES", "ask_cents": l["ask_cents"], "fair_q": l.get("q", 0), "ev": l.get("ev", 0), "forecast_high_f": basket_opp.get("forecast_high_f")}) for l in basket_opp["basket"].get("legs", [])]
            leg_results = [r for r in leg_results]  # keep all
        results.extend(leg_results)

    # --- Paper trade low temperature opportunities (no live orders, no trade limit) ---
    low_opportunities = load_low_edge_opportunities(forecast_source, target_date)
    print(f"\n--- Low temp paper trading: {len(low_opportunities)} opportunities ---")

    paper_results: List[Dict[str, Any]] = []
    for opp in low_opportunities:
        result = execute_paper_trade(TRADES_DB_PATH, opp)
        paper_results.append(result)

    if paper_results:
        fmt_ev = lambda v: f"{v:.1%}" if v is not None else "n/a"
        low_lines = [
            f"- {r['ticker']} {r['side']} | {r.get('limit', 'n/a')}¢ | EV {fmt_ev(r.get('ev'))}"
            for r in paper_results if r["status"] == "PAPER_LOW"
        ]
        if low_lines:
            low_msg = f"**[PAPER] Low Temp Opportunities — {target_date}**\n" + "\n".join(low_lines[:20])
            if len(low_lines) > 20:
                low_msg += f"\n... and {len(low_lines) - 20} more"
            post_discord(low_msg)

    # Post only when an edge/opportunity was found.
    if opportunities or basket_opportunities:
        summary_message = format_summary_message(results, target_date, forecast_source)
        post_discord(summary_message)

    # Post EV bucket performance report (once per day at 8pm ET or later)
    now_et = datetime.now(ZoneInfo("America/New_York"))
    if now_et.hour >= 20:
        try:
            buckets = get_results_by_ev_bucket(TRADES_DB_PATH)
            total_trades = sum(b.get("trades", 0) for b in buckets.values())
            if total_trades > 0:
                report = format_ev_bucket_report(buckets)
                post_discord(report)
        except Exception as e:
            print(f"  [warn] Failed to generate EV bucket report: {e}")

    print(f"\n{'=' * 60}")
    print(f"Session complete: {trades_placed} orders placed")
    print("=" * 60)

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
