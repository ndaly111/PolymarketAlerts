#!/usr/bin/env python3
"""
Automated weather trading script.

Workflow:
1. Compute fair prices from NWS forecasts + error models
2. Collect fresh Kalshi weather market quotes
3. Compute edges (fee-adjusted EV)
4. Filter for opportunities: EV >= 2%, q in [5%, 95%], ask <= 85¢
5. Re-fetch fresh Kalshi price before trading
6. Place limit order at ask for N contracts
7. Wait 30 sec, cancel if not filled
8. Max 10 trades/day across all weather markets
"""

from __future__ import annotations

import json
import os
import sqlite3
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
from zoneinfo import ZoneInfo

# Ensure the repo root is in the path
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from kalshi_auth_client import KalshiAuthClient


# --- Configuration ---
MIN_EV = float(os.getenv("WEATHER_MIN_EV", "0.02"))  # 2% min EV (matches edge computation)
MIN_Q = float(os.getenv("WEATHER_MIN_Q", "0.05"))  # 5% min probability
MAX_KALSHI_ASK_CENTS = int(os.getenv("WEATHER_MAX_ASK", "85"))  # ≤ 85¢
MAX_TRADES_PER_DAY = int(os.getenv("WEATHER_MAX_TRADES_PER_DAY", "10"))
ORDER_TIMEOUT_SECONDS = int(os.getenv("WEATHER_ORDER_TIMEOUT", "30"))
CONTRACTS_PER_TRADE = int(os.getenv("WEATHER_CONTRACTS_PER_TRADE", "1"))

# Database paths
WEATHER_DB_PATH = Path(os.getenv("WEATHER_DB_PATH", str(ROOT / "weather" / "data" / "weather_forecast_accuracy.db")))
TRADES_DB_PATH = Path(os.getenv("WEATHER_TRADES_DB_PATH", "weather_trades.db"))

# Discord webhook for trade notifications
DISCORD_WEBHOOK = os.getenv("DISCORD_WEATHER_TRADES_WEBHOOK", os.getenv("DISCORD_WEATHER_ALERTS", os.getenv("DISCORD_WEBHOOK_URL", "")))

DRY_RUN = os.getenv("WEATHER_DRY_RUN", "0") == "1"

# Paths for edge artifacts
EDGES_BASE = ROOT / "weather" / "outputs" / "edges"
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
    today = datetime.now(ZoneInfo("UTC")).strftime("%Y-%m-%d")
    conn = sqlite3.connect(str(db_path))
    cur = conn.cursor()
    cur.execute("SELECT trades_attempted FROM weather_daily_stats WHERE date = ?", (today,))
    row = cur.fetchone()
    conn.close()
    return row[0] if row else 0


def increment_trades_today(db_path: Path) -> None:
    """Increment trade counter for today."""
    today = datetime.now(ZoneInfo("UTC")).strftime("%Y-%m-%d")
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
) -> None:
    """Record a trade in the database."""
    now = datetime.now(ZoneInfo("UTC")).isoformat()
    today = datetime.now(ZoneInfo("UTC")).strftime("%Y-%m-%d")

    conn = sqlite3.connect(str(db_path))
    cur = conn.cursor()

    cur.execute("""
        INSERT OR REPLACE INTO weather_trades (
            trade_date, city_key, market_ticker, event_display, side,
            quantity, limit_price_cents, fair_q, ev, forecast_high_f,
            order_id, status, fill_price_cents, created_at, updated_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        today, city_key, market_ticker, event_display, side,
        quantity, limit_price_cents, fair_q, ev, forecast_high_f,
        order_id, status, fill_price_cents, now, now
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
    today = datetime.now(ZoneInfo("UTC")).strftime("%Y-%m-%d")
    conn = sqlite3.connect(str(db_path))
    cur = conn.cursor()
    cur.execute(
        "SELECT 1 FROM weather_trades WHERE trade_date = ? AND market_ticker = ?",
        (today, market_ticker)
    )
    row = cur.fetchone()
    conn.close()
    return row is not None


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


def execute_trade(
    client: KalshiAuthClient,
    db_path: Path,
    opportunity: Dict[str, Any],
) -> bool:
    """
    Execute the full trade workflow for a single opportunity.

    Returns True if trade was placed (filled or pending), False otherwise.
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
        return False

    # Check if we have existing position
    try:
        if client.has_position(ticker):
            print(f"  [skip] Already have position in {ticker}")
            return False
    except Exception as e:
        print(f"  [warn] Could not check position: {e}")

    # Re-fetch fresh Kalshi orderbook
    print("  Fetching fresh Kalshi price...")
    fresh_kalshi = fetch_fresh_kalshi_price(client, ticker)
    if not fresh_kalshi:
        print(f"  [skip] Could not get fresh Kalshi price")
        return False

    # Determine which price to use based on side
    if side == "YES":
        kalshi_ask = fresh_kalshi["yes_ask"]
        kalshi_side = "yes"
    else:
        kalshi_ask = fresh_kalshi["no_ask"]
        kalshi_side = "no"

    print(f"  Fresh Kalshi ask: {kalshi_ask}¢")

    # Check price hasn't moved too much
    if kalshi_ask > MAX_KALSHI_ASK_CENTS:
        print(f"  [skip] Kalshi ask {kalshi_ask}¢ > {MAX_KALSHI_ASK_CENTS}¢ limit")
        return False

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
        return False

    # All checks passed - place order
    print(f"  Placing limit order: {CONTRACTS_PER_TRADE} contract(s) at {kalshi_ask}¢")

    if DRY_RUN:
        print("  [DRY_RUN] Would place order")
        record_trade(
            db_path, city_key, ticker, event_display, side,
            CONTRACTS_PER_TRADE, kalshi_ask, fair_q, ev, forecast_high_f,
            "DRY_RUN", "dry_run"
        )
        return True

    try:
        order = client.place_order(
            ticker=ticker,
            side=kalshi_side,
            quantity=CONTRACTS_PER_TRADE,
            limit_price=kalshi_ask,
        )
        order_id = order.get("order_id", order.get("id", ""))
        print(f"  Order placed: {order_id}")

        # Discord notification - order placed
        ev_cents = ev * 100
        place_msg = (
            f"**Weather Order** | {city_key} {event_display} {side}\n"
            f"Limit: {kalshi_ask}¢ | Fair q: {fair_q:.1%} | EV: {ev:.1%} ({ev_cents:.1f}¢)\n"
            f"Forecast high: {forecast_high_f}°F | Waiting {ORDER_TIMEOUT_SECONDS}s..."
        )
        post_discord(place_msg)

    except Exception as e:
        print(f"  [error] Failed to place order: {e}")
        record_trade(
            db_path, city_key, ticker, event_display, side,
            CONTRACTS_PER_TRADE, kalshi_ask, fair_q, ev, forecast_high_f,
            None, "error"
        )
        post_discord(f"**Weather Order Failed** | {city_key} {event_display} {side}\nError: {str(e)[:100]}")
        return False

    # Wait and check fill status
    print(f"  Waiting {ORDER_TIMEOUT_SECONDS}s for fill...")
    time.sleep(ORDER_TIMEOUT_SECONDS)

    try:
        order_status = client.get_order(order_id)
        status = order_status.get("status", "").lower()
        filled_count = order_status.get("filled_count", 0)
        remaining = order_status.get("remaining_count", CONTRACTS_PER_TRADE)

        if filled_count > 0:
            fill_price = order_status.get("avg_fill_price", kalshi_ask)
            print(f"  FILLED: {filled_count} contract(s) at {fill_price}¢")
            record_trade(
                db_path, city_key, ticker, event_display, side,
                filled_count, kalshi_ask, fair_q, ev, forecast_high_f,
                order_id, "filled", fill_price
            )

            # Recalc EV at fill price
            fill_prob = fill_price / 100.0
            if side == "YES":
                actual_ev = fair_q * (1.0 - fill_prob) - (1 - fair_q) * fill_prob - 0.02
            else:
                actual_ev = (1 - fair_q) * (1.0 - fill_prob) - fair_q * fill_prob - 0.02
            ev_cents = actual_ev * 100

            msg = (
                f"**FILLED** | {city_key} {event_display} {side}\n"
                f"Fill: {fill_price}¢ | Fair q: {fair_q:.1%} | EV: {actual_ev:.1%} ({ev_cents:.1f}¢)\n"
                f"Forecast high: {forecast_high_f}°F"
            )
            post_discord(msg)
            return True

        elif remaining > 0:
            # Cancel unfilled order
            print(f"  Not filled, cancelling...")
            client.cancel_order(order_id)
            record_trade(
                db_path, city_key, ticker, event_display, side,
                CONTRACTS_PER_TRADE, kalshi_ask, fair_q, ev, forecast_high_f,
                order_id, "cancelled"
            )
            print(f"  Order cancelled")
            post_discord(f"**Cancelled** | {city_key} {event_display} {side} | Not filled after {ORDER_TIMEOUT_SECONDS}s")
            return False

    except Exception as e:
        print(f"  [error] Failed to check/cancel order: {e}")
        record_trade(
            db_path, city_key, ticker, event_display, side,
            CONTRACTS_PER_TRADE, kalshi_ask, fair_q, ev, forecast_high_f,
            order_id, "unknown"
        )
        return False

    return False


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

    return True


def main() -> int:
    print("=" * 60)
    print("WEATHER AUTO TRADER")
    print(f"Min EV threshold: {MIN_EV:.0%}")
    print(f"Min q: {MIN_Q:.1%}")
    print(f"Max Kalshi ask: {MAX_KALSHI_ASK_CENTS}¢")
    print(f"Max trades/day: {MAX_TRADES_PER_DAY}")
    print(f"Order timeout: {ORDER_TIMEOUT_SECONDS}s")
    print(f"Contracts per trade: {CONTRACTS_PER_TRADE}")
    print(f"Dry run: {DRY_RUN}")
    print("=" * 60)

    # Initialize
    ensure_db_schema(TRADES_DB_PATH)

    # Check daily limit
    trades_today = get_trades_today(TRADES_DB_PATH)
    print(f"\nTrades today: {trades_today}/{MAX_TRADES_PER_DAY}")

    if trades_today >= MAX_TRADES_PER_DAY:
        print("Daily trade limit reached. Exiting.")
        return 0

    remaining_trades = MAX_TRADES_PER_DAY - trades_today

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

    # Get target date (today in ET)
    now_et = datetime.now(ZoneInfo("America/New_York"))
    target_date = now_et.strftime("%Y-%m-%d")
    forecast_source = os.getenv("WEATHER_FORECAST_SOURCE", "nws_hourly_max")

    print(f"\nTarget date: {target_date}")
    print(f"Forecast source: {forecast_source}")

    # Run data pipeline to get fresh prices and edges
    if not run_data_pipeline(target_date, forecast_source):
        print("\n[error] Data pipeline failed. Exiting.")
        return 1

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

    # Process opportunities
    print(f"\n--- Processing top {min(len(opportunities), remaining_trades)} opportunities ---")

    trades_placed = 0
    for opp in opportunities:
        if trades_placed >= remaining_trades:
            print(f"\nReached trade limit ({MAX_TRADES_PER_DAY}/day)")
            break

        success = execute_trade(client, TRADES_DB_PATH, opp)
        if success:
            trades_placed += 1
            increment_trades_today(TRADES_DB_PATH)

        # Rate limit between trades
        time.sleep(1)

    print(f"\n{'=' * 60}")
    print(f"Session complete: {trades_placed} trades placed")
    print("=" * 60)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
