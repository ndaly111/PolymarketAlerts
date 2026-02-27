#!/usr/bin/env python3
"""
Automated weather trading on Kalshi.

Expects the data pipeline (forecast collection, fair prices, Kalshi markets,
edge computation) to have already run — either via the GitHub Actions workflow
steps or manually. This script just:

1. Loads pre-computed edge opportunities from JSON artifacts
2. Filters for tradeable opportunities (EV, price, probability thresholds)
3. Re-fetches fresh Kalshi prices before each trade
4. Places limit orders (or paper-trades in DRY_RUN mode)
5. Logs everything to SQLite and posts a Discord summary
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
)
MIN_Q = float(os.getenv("WEATHER_MIN_Q", "0.05"))
MAX_KALSHI_ASK_CENTS = int(os.getenv("WEATHER_MAX_ASK", "85"))
MIN_KALSHI_ASK_CENTS = int(os.getenv("WEATHER_MIN_ASK", "20"))
MAX_TRADES_PER_DAY = int(os.getenv("WEATHER_MAX_TRADES_PER_DAY", "10"))
CONTRACTS_PER_TRADE = int(os.getenv("WEATHER_CONTRACTS_PER_TRADE", "1"))
KALSHI_FEE_RATE = 0.07  # 7% on winning profits only

# Database paths
WEATHER_DB_PATH = Path(os.getenv("WEATHER_DB_PATH", str(ROOT / "weather" / "data" / "weather_forecast_accuracy.db")))
TRADES_DB_PATH = Path(os.getenv("WEATHER_TRADES_DB_PATH", "weather_trades.db"))

# Discord webhook for trade notifications
DISCORD_WEBHOOK = os.getenv("DISCORD_WEATHER_TRADES_WEBHOOK", os.getenv("DISCORD_WEATHER_ALERTS", os.getenv("DISCORD_WEBHOOK_URL", "")))

# ──── TRADING MODE SAFETY ────
# Two independent flags must BOTH agree before real money goes out:
#   1. WEATHER_DRY_RUN must be explicitly "0" (default is "1" = paper trade)
#   2. WEATHER_AUTOTRADE_ENABLED must be "1" (default is "0" = disabled)
# If either flag is missing or wrong, we paper trade.
ENV_DRY_RUN = os.getenv("WEATHER_DRY_RUN", "1") != "0"  # Default ON (safe)
ENV_AUTOTRADE_ENABLED = os.getenv("WEATHER_AUTOTRADE_ENABLED", "0") == "1"
DRY_RUN = ENV_DRY_RUN or not ENV_AUTOTRADE_ENABLED
AUTOTRADE_ENABLED = ENV_AUTOTRADE_ENABLED and not ENV_DRY_RUN

# Paths for edge artifacts
EDGES_BASE = ROOT / "weather" / "outputs" / "edges"


# ──── Database ────

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
            settled INTEGER DEFAULT 0,
            won INTEGER,
            payout_cents INTEGER,
            settled_at_utc TEXT,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            UNIQUE(trade_date, market_ticker)
        )
    """)

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

    conn.commit()
    conn.close()


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
) -> None:
    """Record a trade in the database."""
    now = datetime.now(ZoneInfo("UTC")).isoformat()
    today = datetime.now(ZoneInfo("America/New_York")).strftime("%Y-%m-%d")
    placed_at_utc = now if status in {"PLACED", "dry_run"} else None

    conn = sqlite3.connect(str(db_path))
    cur = conn.cursor()

    cur.execute("""
        INSERT OR REPLACE INTO weather_trades (
            trade_date, city_key, market_ticker, event_display, side,
            quantity, limit_price_cents, fair_q, ev, forecast_high_f,
            order_id, status, placed_at_utc, created_at, updated_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        today, city_key, market_ticker, event_display, side,
        quantity, limit_price_cents, fair_q, ev, forecast_high_f,
        order_id, status, placed_at_utc, now, now
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


# ──── Kalshi ────

def fetch_fresh_kalshi_price(client: KalshiAuthClient, ticker: str) -> Optional[Dict[str, int]]:
    """Fetch fresh bid/ask from Kalshi for a specific ticker."""
    try:
        market = client.get_market(ticker)
        return {
            "yes_bid": market.get("yes_bid", 0),
            "yes_ask": market.get("yes_ask", 100),
            "no_bid": market.get("no_bid", 0),
            "no_ask": market.get("no_ask", 100),
        }
    except Exception as e:
        print(f"  [error] Failed to fetch Kalshi price for {ticker}: {e}")
        return None


# ──── Edge loading ────

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

                ask_cents = int(buy_price * 100) if buy_price else 100

                if ask_cents > MAX_KALSHI_ASK_CENTS:
                    continue
                if ask_cents < MIN_KALSHI_ASK_CENTS:
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

    opportunities.sort(key=lambda x: x.get("ev", 0), reverse=True)
    return opportunities


# ──── Trade execution ────

def _make_result(
    ticker: str,
    city_key: str,
    event_display: str,
    side: str,
    fair_q: float,
    ev: float,
    status: str,
    limit: Optional[int] = None,
    order_id: Optional[str] = None,
    error: Optional[str] = None,
) -> Dict[str, Any]:
    """Build a standardized trade result dict."""
    result: Dict[str, Any] = {
        "ticker": ticker,
        "city": city_key,
        "event": event_display,
        "side": side,
        "limit": limit,
        "fair_q": fair_q,
        "ev": ev,
        "order_id": order_id,
        "status": status,
    }
    if error:
        result["error"] = error
    return result


def compute_fresh_ev(fair_q: float, kalshi_ask: int, side: str) -> float:
    """Recalculate EV with a fresh Kalshi price, accounting for the 7% fee on profits."""
    kalshi_prob = kalshi_ask / 100.0
    profit_if_win = 1.0 - kalshi_prob
    net_profit_if_win = profit_if_win * (1 - KALSHI_FEE_RATE)

    if side == "YES":
        return fair_q * net_profit_if_win - (1 - fair_q) * kalshi_prob
    else:
        return (1 - fair_q) * net_profit_if_win - fair_q * kalshi_prob


def execute_trade(
    client: KalshiAuthClient,
    db_path: Path,
    opportunity: Dict[str, Any],
    trade_enabled: bool,
) -> Dict[str, Any]:
    """Execute the trade workflow for a single opportunity."""
    ticker = opportunity["market_ticker"]
    city_key = opportunity["city_key"]
    event_display = opportunity["event_display"]
    side = opportunity["side"]
    fair_q = opportunity["fair_q"]
    pre_ev = opportunity["ev"]
    forecast_high_f = opportunity.get("forecast_high_f")

    print(f"\n--- Processing: {city_key} | {event_display} ({side}) ---")

    # Already traded this ticker today?
    if already_traded_today(db_path, ticker):
        print(f"  [skip] Already traded {ticker} today")
        return _make_result(ticker, city_key, event_display, side, fair_q, pre_ev, "SKIPPED_ALREADY_TRADED")

    # Already have a position?
    try:
        if client.has_position(ticker):
            print(f"  [skip] Already have position in {ticker}")
            record_trade(db_path, city_key, ticker, event_display, side, 0, 0, fair_q, pre_ev, forecast_high_f, None, "SKIPPED_HAS_POSITION")
            return _make_result(ticker, city_key, event_display, side, fair_q, pre_ev, "SKIPPED_HAS_POSITION")
    except Exception as e:
        print(f"  [warn] Could not check position: {e}")

    # Re-fetch fresh Kalshi orderbook
    print("  Fetching fresh Kalshi price...")
    fresh_kalshi = fetch_fresh_kalshi_price(client, ticker)
    if not fresh_kalshi:
        print(f"  [skip] Could not get fresh Kalshi price")
        record_trade(db_path, city_key, ticker, event_display, side, 0, 0, fair_q, pre_ev, forecast_high_f, None, "SKIPPED_NO_PRICE")
        return _make_result(ticker, city_key, event_display, side, fair_q, pre_ev, "SKIPPED_NO_PRICE")

    kalshi_ask = fresh_kalshi["yes_ask"] if side == "YES" else fresh_kalshi["no_ask"]
    kalshi_side = "yes" if side == "YES" else "no"
    print(f"  Fresh Kalshi ask: {kalshi_ask}¢")

    # Price range checks
    if kalshi_ask < MIN_KALSHI_ASK_CENTS:
        print(f"  [skip] Kalshi ask {kalshi_ask}¢ < {MIN_KALSHI_ASK_CENTS}¢ floor")
        record_trade(db_path, city_key, ticker, event_display, side, 0, kalshi_ask, fair_q, pre_ev, forecast_high_f, None, "SKIPPED_MIN_ASK")
        return _make_result(ticker, city_key, event_display, side, fair_q, pre_ev, "SKIPPED_MIN_ASK", limit=kalshi_ask)

    if kalshi_ask > MAX_KALSHI_ASK_CENTS:
        print(f"  [skip] Kalshi ask {kalshi_ask}¢ > {MAX_KALSHI_ASK_CENTS}¢ limit")
        record_trade(db_path, city_key, ticker, event_display, side, 0, kalshi_ask, fair_q, pre_ev, forecast_high_f, None, "SKIPPED_MAX_ASK")
        return _make_result(ticker, city_key, event_display, side, fair_q, pre_ev, "SKIPPED_MAX_ASK", limit=kalshi_ask)

    # Recalculate EV with fresh price
    ev = compute_fresh_ev(fair_q, kalshi_ask, side)
    print(f"  Fair q: {fair_q:.1%}, Fresh price: {kalshi_ask/100:.1%}, EV: {ev:.1%}")

    if ev < MIN_EV:
        print(f"  [skip] Fresh EV {ev:.1%} < {MIN_EV:.1%} threshold")
        record_trade(db_path, city_key, ticker, event_display, side, 0, kalshi_ask, fair_q, ev, forecast_high_f, None, "SKIPPED_MIN_EV")
        return _make_result(ticker, city_key, event_display, side, fair_q, ev, "SKIPPED_MIN_EV", limit=kalshi_ask)

    # All checks passed
    print(f"  Placing limit order: {CONTRACTS_PER_TRADE} contract(s) at {kalshi_ask}¢")

    if not trade_enabled:
        print("  [skip] Auto-trade disabled")
        record_trade(db_path, city_key, ticker, event_display, side, CONTRACTS_PER_TRADE, kalshi_ask, fair_q, ev, forecast_high_f, None, "SKIPPED_AUTOTRADE_DISABLED")
        return _make_result(ticker, city_key, event_display, side, fair_q, ev, "SKIPPED_AUTOTRADE_DISABLED", limit=kalshi_ask)

    if DRY_RUN:
        print("  [DRY_RUN] Would place order")
        record_trade(db_path, city_key, ticker, event_display, side, CONTRACTS_PER_TRADE, kalshi_ask, fair_q, ev, forecast_high_f, "DRY_RUN", "dry_run")
        return _make_result(ticker, city_key, event_display, side, fair_q, ev, "DRY_RUN", limit=kalshi_ask, order_id="DRY_RUN")

    try:
        order = client.place_order(
            ticker=ticker,
            side=kalshi_side,
            quantity=CONTRACTS_PER_TRADE,
            limit_price=kalshi_ask,
        )
        order_id = order.get("order_id", order.get("id", ""))
        print(f"  Order placed: {order_id}")
        record_trade(db_path, city_key, ticker, event_display, side, CONTRACTS_PER_TRADE, kalshi_ask, fair_q, ev, forecast_high_f, order_id, "PLACED")
        return _make_result(ticker, city_key, event_display, side, fair_q, ev, "PLACED", limit=kalshi_ask, order_id=order_id)

    except Exception as e:
        print(f"  [error] Failed to place order: {e}")
        record_trade(db_path, city_key, ticker, event_display, side, CONTRACTS_PER_TRADE, kalshi_ask, fair_q, ev, forecast_high_f, None, "FAILED_PLACE_ORDER")
        return _make_result(ticker, city_key, event_display, side, fair_q, ev, "FAILED_PLACE_ORDER", limit=kalshi_ask, error=str(e)[:120])


# ──── Discord ────

def post_discord(message: str) -> None:
    """Post to Discord webhook."""
    if not DISCORD_WEBHOOK:
        return
    try:
        import requests
        requests.post(DISCORD_WEBHOOK, json={"content": message[:1990]}, timeout=10)
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
        f"Min EV: {MIN_EV:.0%} | Ask range: {MIN_KALSHI_ASK_CENTS}-{MAX_KALSHI_ASK_CENTS}¢ | Max trades/day: {MAX_TRADES_PER_DAY}",
        f"Placed: {len(placed)} | Skipped: {len(skipped)} | Failed: {len(failed)}",
    ]
    if DRY_RUN:
        header_lines.append("Mode: DRY_RUN (no orders submitted)")
    elif not AUTOTRADE_ENABLED:
        header_lines.append("Mode: AUTOTRADE DISABLED (no orders submitted)")

    sections: List[str] = ["\n".join(header_lines)]

    def fmt_ev(value: Optional[float]) -> str:
        return f"{value:.1%}" if value is not None else "n/a"

    def add_section(title: str, rows: List[str]) -> None:
        if rows:
            sections.append(f"\n{title}\n" + "\n".join(rows))

    placed_rows = [
        f"- {r['ticker']} {r['side']} | {r['limit']}¢ | EV {fmt_ev(r.get('ev'))} | Order {r['order_id']}"
        for r in placed
    ]
    skipped_rows = [
        f"- {r['ticker']} {r['side']} | {r['status']} ({fmt_ev(r.get('ev'))})"
        for r in skipped
    ]
    failed_rows = [
        f"- {r['ticker']} {r['side']} | {r['status']} ({r.get('error', 'unknown error')})"
        for r in failed
    ]

    add_section("✅ PLACED", placed_rows)
    add_section("⏭️ SKIPPED", skipped_rows)
    add_section("❌ FAILED", failed_rows)

    message = "\n".join(sections)
    if len(message) <= 1990:
        return message

    # Trim to fit Discord's limit
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

    return "\n".join(trimmed_sections)[:1990]


# ──── Main ────

def main() -> int:
    print("=" * 60)
    print("WEATHER AUTO TRADER")
    if DRY_RUN:
        print(">>> MODE: PAPER TRADE (DRY RUN) — no real orders <<<")
    else:
        print(">>> MODE: LIVE TRADING — real money at risk <<<")
    print(f"  WEATHER_DRY_RUN={os.getenv('WEATHER_DRY_RUN', '(unset)')}")
    print(f"  WEATHER_AUTOTRADE_ENABLED={os.getenv('WEATHER_AUTOTRADE_ENABLED', '(unset)')}")
    print(f"  → DRY_RUN={DRY_RUN}, AUTOTRADE_ENABLED={AUTOTRADE_ENABLED}")
    print(f"Min EV threshold: {MIN_EV:.0%}")
    print(f"Min q: {MIN_Q:.1%}")
    print(f"Kalshi ask range: {MIN_KALSHI_ASK_CENTS}-{MAX_KALSHI_ASK_CENTS}¢")
    print(f"Max trades/day: {MAX_TRADES_PER_DAY}")
    print(f"Contracts per trade: {CONTRACTS_PER_TRADE}")
    print("=" * 60)

    ensure_db_schema(TRADES_DB_PATH)

    if AUTOTRADE_ENABLED and MAX_TRADES_PER_DAY <= 0:
        print("ERROR: WEATHER_MAX_TRADES_PER_DAY must be > 0 when auto-trade is enabled.")
        return 1

    # Check daily limit
    trades_today = get_trades_today(TRADES_DB_PATH)
    print(f"\nTrades today: {trades_today}/{MAX_TRADES_PER_DAY}")

    if trades_today >= MAX_TRADES_PER_DAY:
        print("Daily trade limit reached. Exiting.")
        return 0

    remaining_trades = MAX_TRADES_PER_DAY - trades_today

    # Target date and forecast source
    now_et = datetime.now(ZoneInfo("America/New_York"))
    target_date = now_et.strftime("%Y-%m-%d")
    forecast_source = os.getenv("WEATHER_FORECAST_SOURCE", "open_meteo")

    print(f"\nTarget date: {target_date}")
    print(f"Forecast source: {forecast_source}")

    # Initialize Kalshi client
    client = None
    if AUTOTRADE_ENABLED or DRY_RUN:
        print("\nInitializing Kalshi client...")
        try:
            client = KalshiAuthClient.from_env()
            balance = client.get_balance()
            available = balance.get("available_balance", balance.get("balance", 0)) / 100
            print(f"Balance: ${available:.2f}")
        except Exception as e:
            print(f"Failed to initialize Kalshi client: {e}")
            return 1

    # Load pre-computed edge opportunities
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

    # Process opportunities
    print(f"\n--- Processing top {min(len(opportunities), remaining_trades)} opportunities ---")

    trades_placed = 0
    results: List[Dict[str, Any]] = []
    for opp in opportunities:
        if trades_placed >= remaining_trades:
            print(f"\nReached trade limit ({MAX_TRADES_PER_DAY}/day)")
            break

        result = execute_trade(
            client,
            TRADES_DB_PATH,
            opp,
            trade_enabled=AUTOTRADE_ENABLED or DRY_RUN,
        )
        results.append(result)
        if result["status"] in {"PLACED", "DRY_RUN"}:
            trades_placed += 1
            increment_trades_today(TRADES_DB_PATH)

    # Post Discord summary when opportunities were found
    if opportunities:
        summary_message = format_summary_message(results, target_date, forecast_source)
        post_discord(summary_message)

    print(f"\n{'=' * 60}")
    print(f"Session complete: {trades_placed} orders placed")
    print("=" * 60)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
