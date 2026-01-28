#!/usr/bin/env python3
"""
Automated sports betting script for moneylines, spreads, and totals.

Workflow:
1. Fetch Kalshi sports markets (moneylines, spreads, totals)
2. Fetch sportsbook odds from Odds API (with caching)
3. Calculate edges (fair prob vs Kalshi price)
4. Filter for opportunities: EV >= threshold, min books
5. Place limit orders on Kalshi
6. Track in database with settlement support

Similar to props_auto_trade.py but for game lines instead of player props.
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

from kalshi_auth_client import KalshiAuthClient

# --- Configuration ---
MIN_EDGE = float(os.getenv("SPORTS_MIN_EDGE", "0.05"))  # 5% min edge
MIN_BOOKS = int(os.getenv("SPORTS_MIN_BOOKS", "2"))  # Require 2+ books
MAX_KALSHI_ASK_CENTS = int(os.getenv("SPORTS_MAX_ASK", "75"))  # ≤ 75¢
MAX_TRADES_PER_DAY = int(os.getenv("SPORTS_MAX_TRADES_PER_DAY", "10"))
ORDER_TIMEOUT_SECONDS = int(os.getenv("SPORTS_ORDER_TIMEOUT", "30"))
CONTRACTS_PER_TRADE = int(os.getenv("SPORTS_CONTRACTS_PER_TRADE", "1"))

# Database for tracking trades
DB_PATH = Path(os.getenv("SPORTS_DB_PATH", "sports_trades.db"))

# Discord webhook for trade notifications
DISCORD_WEBHOOK = os.getenv("DISCORD_SPORTS_WEBHOOK", os.getenv("DISCORD_WEBHOOK_URL", ""))

DRY_RUN = os.getenv("SPORTS_DRY_RUN", "0") == "1"

# Timezone for daily boundaries (Eastern Time)
ET = ZoneInfo("America/New_York")

# Cache settings for Odds API efficiency
CACHE_DIR = Path(os.getenv("SPORTS_CACHE_DIR", "/tmp/sports_odds_cache"))
CACHE_TTL_SECONDS = int(os.getenv("SPORTS_CACHE_TTL", "1800"))  # 30 min default

# Sports to include
SPORT_KEYS = os.getenv("SPORTS_SPORT_KEYS", "basketball_nba,basketball_ncaab,americanfootball_nfl").split(",")


def ensure_db_schema(db_path: Path) -> None:
    """Create database tables if they don't exist."""
    conn = sqlite3.connect(str(db_path))
    cur = conn.cursor()

    cur.execute("""
        CREATE TABLE IF NOT EXISTS trades (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            trade_date TEXT NOT NULL,
            ticker TEXT NOT NULL,
            event_title TEXT NOT NULL,
            line_type TEXT NOT NULL,
            line_label TEXT NOT NULL,
            side TEXT NOT NULL,
            quantity INTEGER NOT NULL,
            limit_price_cents INTEGER NOT NULL,
            fair_prob REAL NOT NULL,
            edge REAL NOT NULL,
            books_count INTEGER NOT NULL,
            order_id TEXT,
            status TEXT NOT NULL,
            fill_price_cents INTEGER,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            settled INTEGER DEFAULT 0,
            won INTEGER,
            payout_cents INTEGER,
            settled_at TEXT,
            UNIQUE(trade_date, ticker)
        )
    """)

    cur.execute("""
        CREATE TABLE IF NOT EXISTS daily_stats (
            date TEXT PRIMARY KEY,
            trades_attempted INTEGER DEFAULT 0,
            trades_filled INTEGER DEFAULT 0,
            trades_cancelled INTEGER DEFAULT 0,
            total_edge_cents REAL DEFAULT 0
        )
    """)

    cur.execute("""
        CREATE TABLE IF NOT EXISTS scanned_opportunities (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            scan_date TEXT NOT NULL,
            scan_time TEXT NOT NULL,
            ticker TEXT NOT NULL,
            event_title TEXT NOT NULL,
            line_type TEXT NOT NULL,
            line_label TEXT NOT NULL,
            side TEXT NOT NULL,
            kalshi_bid INTEGER,
            kalshi_ask INTEGER,
            fair_prob REAL NOT NULL,
            edge REAL NOT NULL,
            books_count INTEGER NOT NULL,
            commence_time TEXT,
            UNIQUE(scan_date, ticker, side)
        )
    """)

    # Migration for existing tables
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


def get_trades_today(db_path: Path) -> int:
    """Get number of trades attempted today (ET timezone)."""
    today = datetime.now(ET).strftime("%Y-%m-%d")
    conn = sqlite3.connect(str(db_path))
    cur = conn.cursor()
    cur.execute("SELECT trades_attempted FROM daily_stats WHERE date = ?", (today,))
    row = cur.fetchone()
    conn.close()
    return row[0] if row else 0


def increment_trades_today(db_path: Path) -> None:
    """Increment trade counter for today (ET timezone)."""
    today = datetime.now(ET).strftime("%Y-%m-%d")
    conn = sqlite3.connect(str(db_path))
    cur = conn.cursor()
    cur.execute("""
        INSERT INTO daily_stats (date, trades_attempted)
        VALUES (?, 1)
        ON CONFLICT(date) DO UPDATE SET trades_attempted = trades_attempted + 1
    """, (today,))
    conn.commit()
    conn.close()


def record_trade(
    db_path: Path,
    ticker: str,
    event_title: str,
    line_type: str,
    line_label: str,
    side: str,
    quantity: int,
    limit_price_cents: int,
    fair_prob: float,
    edge: float,
    books_count: int,
    order_id: Optional[str],
    status: str,
    fill_price_cents: Optional[int] = None,
) -> None:
    """Record a trade in the database (ET timezone)."""
    now = datetime.now(ET).isoformat()
    today = datetime.now(ET).strftime("%Y-%m-%d")

    conn = sqlite3.connect(str(db_path))
    cur = conn.cursor()

    cur.execute("""
        INSERT OR REPLACE INTO trades (
            trade_date, ticker, event_title, line_type, line_label, side,
            quantity, limit_price_cents, fair_prob, edge, books_count,
            order_id, status, fill_price_cents, created_at, updated_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        today, ticker, event_title, line_type, line_label, side,
        quantity, limit_price_cents, fair_prob, edge, books_count,
        order_id, status, fill_price_cents, now, now
    ))

    conn.commit()
    conn.close()


def log_scanned_opportunity(
    db_path: Path,
    ticker: str,
    event_title: str,
    line_type: str,
    line_label: str,
    side: str,
    kalshi_bid: int,
    kalshi_ask: int,
    fair_prob: float,
    edge: float,
    books_count: int,
    commence_time: str = "",
) -> None:
    """Log a scanned opportunity for later analysis (ET timezone)."""
    now = datetime.now(ET)
    scan_date = now.strftime("%Y-%m-%d")
    scan_time = now.isoformat()

    conn = sqlite3.connect(str(db_path))
    cur = conn.cursor()

    cur.execute("""
        INSERT OR REPLACE INTO scanned_opportunities (
            scan_date, scan_time, ticker, event_title, line_type, line_label, side,
            kalshi_bid, kalshi_ask, fair_prob, edge, books_count, commence_time
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        scan_date, scan_time, ticker, event_title, line_type, line_label, side,
        kalshi_bid, kalshi_ask, fair_prob, edge, books_count, commence_time
    ))

    conn.commit()
    conn.close()


def already_traded_today(db_path: Path, ticker: str) -> bool:
    """Check if we already traded this ticker today (ET timezone)."""
    today = datetime.now(ET).strftime("%Y-%m-%d")
    conn = sqlite3.connect(str(db_path))
    cur = conn.cursor()
    cur.execute(
        "SELECT 1 FROM trades WHERE trade_date = ? AND ticker = ?",
        (today, ticker)
    )
    row = cur.fetchone()
    conn.close()
    return row is not None


# --- Caching for Odds API ---

def _get_cache_path(cache_key: str) -> Path:
    """Get cache file path for a given key."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    safe_key = cache_key.replace("/", "_").replace(":", "_")
    return CACHE_DIR / f"{safe_key}.json"


def _read_cache(cache_key: str) -> Optional[Dict[str, Any]]:
    """Read from cache if exists and not expired."""
    cache_path = _get_cache_path(cache_key)
    if not cache_path.exists():
        return None
    try:
        data = json.loads(cache_path.read_text())
        cached_at = data.get("_cached_at", 0)
        if time.time() - cached_at > CACHE_TTL_SECONDS:
            return None
        return data.get("events", [])
    except Exception:
        return None


def _write_cache(cache_key: str, events: List[Dict[str, Any]]) -> None:
    """Write events to cache."""
    cache_path = _get_cache_path(cache_key)
    try:
        cache_path.write_text(json.dumps({
            "_cached_at": time.time(),
            "events": events,
        }))
    except Exception:
        pass


# --- Discord ---

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


# --- Main Trading Logic ---

def fetch_kalshi_sports_markets(client: KalshiAuthClient) -> List[Dict[str, Any]]:
    """Fetch sports markets from Kalshi (moneylines, spreads, totals)."""
    markets = []

    # Series patterns for game lines
    series_patterns = [
        "KXNBA",      # NBA games
        "KXNCAAB",    # College basketball
        "KXNFL",      # NFL games
        "KXNCAAF",    # College football
        "KXNHL",      # NHL games
        "KXMLB",      # MLB games
    ]

    try:
        # Get all markets and filter for sports
        all_markets = client.get_markets(limit=1000)

        for market in all_markets:
            ticker = market.get("ticker", "")
            title = market.get("title", "").upper()

            # Check if it's a game line market (not props)
            is_game_line = any(p in ticker for p in series_patterns)
            is_prop = "PROP" in title or "PLAYER" in title or "AST" in ticker or "REB" in ticker or "PTS" in ticker

            if is_game_line and not is_prop:
                # Determine line type
                if "SPREAD" in title or "SPREAD" in ticker:
                    line_type = "spread"
                elif "TOTAL" in title or "OVER" in title or "UNDER" in title:
                    line_type = "total"
                else:
                    line_type = "moneyline"

                markets.append({
                    "ticker": ticker,
                    "title": market.get("title", ""),
                    "line_type": line_type,
                    "yes_bid": market.get("yes_bid", 0),
                    "yes_ask": market.get("yes_ask", 100),
                    "no_bid": market.get("no_bid", 0),
                    "no_ask": market.get("no_ask", 100),
                    "volume": market.get("volume", 0),
                    "close_time": market.get("close_time", ""),
                })

    except Exception as e:
        print(f"  [error] Failed to fetch Kalshi markets: {e}")

    return markets


def fetch_odds_events(sport_keys: List[str], use_cache: bool = True) -> List[Dict[str, Any]]:
    """Fetch events from Odds API with caching."""
    from oddsapi_lines import fetch_all_sports_events

    all_events = []

    for sport_key in sport_keys:
        cache_key = f"sports_lines_{sport_key}"

        if use_cache:
            cached = _read_cache(cache_key)
            if cached:
                print(f"  [cache] {sport_key}: {len(cached)} events from cache")
                all_events.extend(cached)
                continue

        try:
            events = fetch_all_sports_events([sport_key], markets="h2h,spreads,totals", regions="us")
            if events:
                _write_cache(cache_key, events)
                all_events.extend(events)
                print(f"  [api] {sport_key}: {len(events)} events fetched")
        except Exception as e:
            print(f"  [error] {sport_key}: {e}")

    return all_events


def main() -> int:
    print("=" * 60)
    print("SPORTS AUTO TRADER")
    print(f"Min edge: {MIN_EDGE:.0%}")
    print(f"Min books: {MIN_BOOKS}")
    print(f"Max Kalshi ask: {MAX_KALSHI_ASK_CENTS}¢")
    print(f"Max trades/day: {MAX_TRADES_PER_DAY}")
    print(f"Sports: {', '.join(SPORT_KEYS)}")
    print(f"Dry run: {DRY_RUN}")
    print("=" * 60)

    # Initialize
    ensure_db_schema(DB_PATH)

    # Check daily limit
    trades_today = get_trades_today(DB_PATH)
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

    # Fetch data
    print("\n--- Fetching Data ---")

    print("Fetching Kalshi sports markets...")
    kalshi_markets = fetch_kalshi_sports_markets(client)
    print(f"  Found {len(kalshi_markets)} Kalshi game line markets")

    if not kalshi_markets:
        print("No Kalshi sports markets available. Exiting.")
        return 0

    print("Fetching sportsbook odds (cached 30min)...")
    odds_events = fetch_odds_events(SPORT_KEYS, use_cache=True)
    print(f"  Found {len(odds_events)} sportsbook events")

    # TODO: Match Kalshi markets to Odds API events and calculate edges
    # This requires implementing the matching logic from kalshi_sports_value.py

    print("\n--- Matching and Edge Calculation ---")
    print("  [TODO] Implement market matching and edge calculation")
    print("  This will use the logic from kalshi_sports_value.py")

    # For now, just log that we're set up
    print(f"\n{'=' * 60}")
    print(f"Session complete: Infrastructure ready")
    print(f"Next step: Implement matching logic from kalshi_sports_value.py")
    print("=" * 60)

    # Update dashboard
    try:
        from scripts.generate_dashboard_data import update_dashboard
        update_dashboard(quiet=True)
        print("Dashboard data updated")
    except Exception as e:
        print(f"  [warn] Failed to update dashboard: {e}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
