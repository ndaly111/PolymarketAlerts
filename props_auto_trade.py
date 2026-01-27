#!/usr/bin/env python3
"""
Automated props trading script.

Workflow:
1. Initial scan: find 5%+ EV opportunities
2. Check if already have position → skip if yes
3. Re-fetch Kalshi orderbook (get actual ask)
4. Re-fetch Odds API (min 2 books)
5. Filters: Kalshi ask ≤ 67¢, Books fair prob ≥ 33%
6. Recalculate edge with fresh data
7. If edge ≥ 5%, place limit order at ask for 1 contract
8. Wait 30 sec, cancel if not filled
9. Max 10 trades/day
"""

from __future__ import annotations

import json
import os
import sqlite3
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from kalshi_auth_client import KalshiAuthClient, prob_to_american, american_to_prob
from kalshi_props import fetch_kalshi_props, KALSHI_TO_ODDSAPI
from oddsapi_props import fetch_all_props, normalize_player_name
from props_value import match_props, filter_and_rank


# --- Configuration ---
MIN_INITIAL_EV = float(os.getenv("MIN_INITIAL_EV", "0.05"))  # 5% for initial scan
MIN_FINAL_EV = float(os.getenv("MIN_FINAL_EV", "0.05"))  # 5% after re-verification
MIN_BOOKS = int(os.getenv("MIN_BOOKS", "2"))  # Require 2+ books
MAX_KALSHI_ASK_CENTS = int(os.getenv("MAX_KALSHI_ASK", "67"))  # ≤ 67¢ (better than -200)
MIN_FAIR_PROB = float(os.getenv("MIN_FAIR_PROB", "0.333"))  # ≥ 33% (better than +200)
MAX_TRADES_PER_DAY = int(os.getenv("MAX_TRADES_PER_DAY", "10"))
ORDER_TIMEOUT_SECONDS = int(os.getenv("ORDER_TIMEOUT", "30"))
CONTRACTS_PER_TRADE = int(os.getenv("CONTRACTS_PER_TRADE", "1"))

# Database for tracking trades
DB_PATH = Path(os.getenv("PROPS_DB_PATH", "props_trades.db"))

# Discord webhook for trade notifications
DISCORD_WEBHOOK = os.getenv("DISCORD_PROPS_WEBHOOK", os.getenv("DISCORD_WEBHOOK_URL", ""))

DRY_RUN = os.getenv("DRY_RUN", "0") == "1"


def ensure_db_schema(db_path: Path) -> None:
    """Create database tables if they don't exist."""
    conn = sqlite3.connect(str(db_path))
    cur = conn.cursor()

    cur.execute("""
        CREATE TABLE IF NOT EXISTS trades (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            trade_date TEXT NOT NULL,
            ticker TEXT NOT NULL,
            player_name TEXT NOT NULL,
            stat_type TEXT NOT NULL,
            line REAL NOT NULL,
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

    # Log ALL scanned opportunities (for backtesting/analysis)
    cur.execute("""
        CREATE TABLE IF NOT EXISTS scanned_opportunities (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            scan_date TEXT NOT NULL,
            scan_time TEXT NOT NULL,
            ticker TEXT NOT NULL,
            player_name TEXT NOT NULL,
            stat_type TEXT NOT NULL,
            line REAL NOT NULL,
            side TEXT NOT NULL,
            kalshi_bid INTEGER,
            kalshi_ask INTEGER,
            kalshi_prob REAL NOT NULL,
            fair_prob REAL NOT NULL,
            edge REAL NOT NULL,
            books_count INTEGER NOT NULL,
            volume INTEGER,
            event_time TEXT,
            result TEXT,
            settled_at TEXT,
            UNIQUE(scan_date, ticker, side)
        )
    """)

    conn.commit()
    conn.close()


def get_trades_today(db_path: Path) -> int:
    """Get number of trades attempted today."""
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    conn = sqlite3.connect(str(db_path))
    cur = conn.cursor()
    cur.execute("SELECT trades_attempted FROM daily_stats WHERE date = ?", (today,))
    row = cur.fetchone()
    conn.close()
    return row[0] if row else 0


def increment_trades_today(db_path: Path) -> None:
    """Increment trade counter for today."""
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
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
    player_name: str,
    stat_type: str,
    line: float,
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
    """Record a trade in the database."""
    now = datetime.now(timezone.utc).isoformat()
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    conn = sqlite3.connect(str(db_path))
    cur = conn.cursor()

    cur.execute("""
        INSERT OR REPLACE INTO trades (
            trade_date, ticker, player_name, stat_type, line, side,
            quantity, limit_price_cents, fair_prob, edge, books_count,
            order_id, status, fill_price_cents, created_at, updated_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        today, ticker, player_name, stat_type, line, side,
        quantity, limit_price_cents, fair_prob, edge, books_count,
        order_id, status, fill_price_cents, now, now
    ))

    conn.commit()
    conn.close()


def log_scanned_opportunity(
    db_path: Path,
    ticker: str,
    player_name: str,
    stat_type: str,
    line: float,
    side: str,
    kalshi_bid: int,
    kalshi_ask: int,
    kalshi_prob: float,
    fair_prob: float,
    edge: float,
    books_count: int,
    volume: int = 0,
    event_time: str = "",
) -> None:
    """Log a scanned opportunity for later analysis."""
    now = datetime.now(timezone.utc)
    scan_date = now.strftime("%Y-%m-%d")
    scan_time = now.isoformat()

    conn = sqlite3.connect(str(db_path))
    cur = conn.cursor()

    cur.execute("""
        INSERT OR REPLACE INTO scanned_opportunities (
            scan_date, scan_time, ticker, player_name, stat_type, line, side,
            kalshi_bid, kalshi_ask, kalshi_prob, fair_prob, edge, books_count,
            volume, event_time
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        scan_date, scan_time, ticker, player_name, stat_type, line, side,
        kalshi_bid, kalshi_ask, kalshi_prob, fair_prob, edge, books_count,
        volume, event_time
    ))

    conn.commit()
    conn.close()


def already_traded_today(db_path: Path, ticker: str) -> bool:
    """Check if we already traded this ticker today."""
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    conn = sqlite3.connect(str(db_path))
    cur = conn.cursor()
    cur.execute(
        "SELECT 1 FROM trades WHERE trade_date = ? AND ticker = ?",
        (today, ticker)
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


def find_odds_in_cache(
    cached_odds: List[Dict[str, Any]],
    player_name_norm: str,
    oddsapi_type: str,
    target_line: float,
    min_books: int = 2,
) -> Optional[Dict[str, Any]]:
    """
    Find matching odds in cached data (no API call).

    Returns the matching prop with fair probabilities, or None if not found.
    """
    for prop in cached_odds:
        if prop.get("books_used", 0) < min_books:
            continue
        if prop.get("player_name_norm") != player_name_norm:
            continue
        if prop.get("prop_type") != oddsapi_type:
            continue
        # Check line (Kalshi "8+" means > 7.5, so target is line - 0.5)
        odds_line = prop.get("line", 0)
        if abs(odds_line - target_line) <= 0.5:
            return prop

    return None


def fetch_fresh_odds_for_prop(
    player_name_norm: str,
    oddsapi_type: str,
    target_line: float,
    event_id: str = "",
    sport_key: str = "",
    min_books: int = 2,
) -> Optional[Dict[str, Any]]:
    """
    Fetch FRESH odds from Odds API for a specific prop (bypasses cache).

    Used for final verification before placing a trade.
    Only fetches the specific event if event_id is provided.
    Updates cache with fresh data to avoid wasting the API call.

    Returns the matching prop with fair probabilities, or None if not found.
    """
    from oddsapi_props import (
        fetch_event_props, extract_props_from_event, SPORT_PROP_MARKETS,
        _update_cache_with_prop
    )

    # Determine sport if not provided
    if not sport_key:
        if "pass" in oddsapi_type or "rush" in oddsapi_type or "reception" in oddsapi_type or "td" in oddsapi_type:
            sport_key = "americanfootball_nfl"
        elif "shots" in oddsapi_type or "goals" in oddsapi_type:
            sport_key = "icehockey_nhl"
        else:
            sport_key = "basketball_nba"

    markets = SPORT_PROP_MARKETS.get(sport_key, [])
    if not markets:
        return None

    try:
        # If we have event_id, fetch just that event (1 API call)
        if event_id:
            event_data = fetch_event_props(sport_key, event_id, markets)
            props = extract_props_from_event(event_data)
        else:
            # Fallback: fetch all events for sport (more expensive)
            from oddsapi_props import fetch_all_props
            props = fetch_all_props(sport_keys=[sport_key], min_books=min_books, use_cache=False, force_refresh=True)

        # Update cache with ALL fresh props from this event (don't waste the call)
        cache_key = f"props_{sport_key}_us"
        for prop in props:
            _update_cache_with_prop(cache_key, prop)

        # Find matching prop
        for prop in props:
            if prop.get("books_used", 0) < min_books:
                continue
            if prop.get("player_name_norm") != player_name_norm:
                continue
            if prop.get("prop_type") != oddsapi_type:
                continue
            odds_line = prop.get("line", 0)
            if abs(odds_line - target_line) <= 0.5:
                return prop

        return None

    except Exception as e:
        print(f"  [error] Failed to fetch fresh odds: {e}")
        return None


def get_active_sports_from_kalshi(kalshi_props: List[Dict[str, Any]]) -> List[str]:
    """
    Determine which sports have active Kalshi markets.

    Returns list of Odds API sport keys that have matching Kalshi props.
    """
    active_sports = set()

    for prop in kalshi_props:
        stat_type = prop.get("stat_type", "")
        oddsapi_type = KALSHI_TO_ODDSAPI.get(stat_type, "")

        if not oddsapi_type:
            continue

        # Map oddsapi_type to sport
        if "pass" in oddsapi_type or "rush" in oddsapi_type or "reception" in oddsapi_type or "td" in oddsapi_type:
            active_sports.add("americanfootball_nfl")
        elif "shots" in oddsapi_type or "goals" in oddsapi_type:
            active_sports.add("icehockey_nhl")
        else:
            # Basketball props - NBA only (no NCAAB props on Kalshi)
            active_sports.add("basketball_nba")

    return list(active_sports)


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
    cached_odds: List[Dict[str, Any]],
) -> bool:
    """
    Execute the full trade workflow for a single opportunity.

    Uses cached_odds for verification (no new Odds API calls).
    Re-fetches Kalshi prices (free).

    Returns True if trade was placed (filled or pending), False otherwise.
    """
    ticker = opportunity["kalshi_ticker"]
    player = opportunity["player_name"]
    stat_type = opportunity["stat_type"]
    line = opportunity["kalshi_line"]
    side = opportunity["best_side"]
    player_norm = opportunity["player_name_norm"]
    oddsapi_type = opportunity.get("oddsapi_type", KALSHI_TO_ODDSAPI.get(stat_type, ""))

    print(f"\n--- Processing: {player} {stat_type} {line}+ ({side}) ---")

    # Step 2: Check if already traded today
    if already_traded_today(db_path, ticker):
        print(f"  [skip] Already traded {ticker} today")
        return False

    # Step 2b: Check if we have existing position
    try:
        if client.has_position(ticker):
            print(f"  [skip] Already have position in {ticker}")
            return False
    except Exception as e:
        print(f"  [warn] Could not check position: {e}")

    # Step 3: Re-fetch Kalshi orderbook
    print("  Fetching fresh Kalshi price...")
    fresh_kalshi = fetch_fresh_kalshi_price(client, ticker)
    if not fresh_kalshi:
        print(f"  [skip] Could not get fresh Kalshi price")
        return False

    # Determine which price to use based on side
    if side == "OVER":
        kalshi_ask = fresh_kalshi["yes_ask"]
        kalshi_side = "yes"
    else:
        kalshi_ask = fresh_kalshi["no_ask"]
        kalshi_side = "no"

    print(f"  Kalshi ask: {kalshi_ask}¢")

    # Step 5a: Check Kalshi odds filter (better than -200 = ≤67¢)
    if kalshi_ask > MAX_KALSHI_ASK_CENTS:
        print(f"  [skip] Kalshi ask {kalshi_ask}¢ > {MAX_KALSHI_ASK_CENTS}¢ (worse than -200)")
        return False

    # Step 4a: Quick check with cached odds
    target_line = line - 0.5  # Kalshi "8+" = over 7.5
    cached_prop = find_odds_in_cache(
        cached_odds, player_norm, oddsapi_type, target_line, min_books=MIN_BOOKS
    )

    if not cached_prop:
        print(f"  [skip] Could not find prop in cached odds with {MIN_BOOKS}+ books")
        return False

    # Get event info for targeted API call
    event_id = cached_prop.get("event_id", "")
    sport_key = cached_prop.get("sport_key", "")

    # Step 4b: Fetch FRESH odds before placing trade (1 API call)
    print(f"  Fetching fresh odds for verification...")
    fresh_prop = fetch_fresh_odds_for_prop(
        player_norm, oddsapi_type, target_line,
        event_id=event_id, sport_key=sport_key, min_books=MIN_BOOKS
    )

    if not fresh_prop:
        print(f"  [skip] Could not verify with fresh odds ({MIN_BOOKS}+ books)")
        return False

    books_count = fresh_prop.get("books_used", 0)
    print(f"  Verified with fresh odds from {books_count} books")

    # Get fair probability for our side
    if side == "OVER":
        fair_prob = fresh_prop.get("fair_over_prob", 0.5)
    else:
        fair_prob = fresh_prop.get("fair_under_prob", 0.5)

    # Step 5b: Check books odds filter (better than +200 = ≥33%)
    if fair_prob < MIN_FAIR_PROB:
        fair_american = prob_to_american(fair_prob) if 0 < fair_prob < 1 else 0
        print(f"  [skip] Fair prob {fair_prob:.1%} ({fair_american:+d}) < {MIN_FAIR_PROB:.1%} (worse than +200)")
        return False

    # Step 6: Recalculate edge
    kalshi_prob = kalshi_ask / 100.0
    edge = fair_prob - kalshi_prob

    print(f"  Fair prob: {fair_prob:.1%}, Kalshi: {kalshi_prob:.1%}, Edge: {edge:.1%}")

    # Step 7: Check minimum edge
    if edge < MIN_FINAL_EV:
        print(f"  [skip] Edge {edge:.1%} < {MIN_FINAL_EV:.1%} threshold")
        return False

    # All checks passed - place order
    print(f"  Placing limit order: {CONTRACTS_PER_TRADE} contract(s) at {kalshi_ask}¢")

    if DRY_RUN:
        print("  [DRY_RUN] Would place order")
        record_trade(
            db_path, ticker, player, stat_type, line, side,
            CONTRACTS_PER_TRADE, kalshi_ask, fair_prob, edge, books_count,
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
        ev_cents = edge * 100
        place_msg = (
            f"**Order Placed** | {player} {stat_type} {line:.0f}+ {side}\n"
            f"Limit: {kalshi_ask}¢ | Fair: {fair_prob:.1%} | Edge: {edge:.1%} | EV: {ev_cents:.1f}¢\n"
            f"Books: {books_count} | Waiting {ORDER_TIMEOUT_SECONDS}s for fill..."
        )
        post_discord(place_msg)

    except Exception as e:
        print(f"  [error] Failed to place order: {e}")
        record_trade(
            db_path, ticker, player, stat_type, line, side,
            CONTRACTS_PER_TRADE, kalshi_ask, fair_prob, edge, books_count,
            None, "error"
        )
        # Discord notification - error
        post_discord(f"**Order Failed** | {player} {stat_type} {line:.0f}+ {side}\nError: {str(e)[:100]}")
        return False

    # Step 8: Wait and check fill status
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
                db_path, ticker, player, stat_type, line, side,
                filled_count, kalshi_ask, fair_prob, edge, books_count,
                order_id, "filled", fill_price
            )

            # Discord notification - filled
            actual_edge = fair_prob - (fill_price / 100.0)
            ev_cents = actual_edge * 100
            msg = (
                f"**FILLED** | {player} {stat_type} {line:.0f}+ {side}\n"
                f"Fill: {fill_price}¢ | Fair: {fair_prob:.1%} | Edge: {actual_edge:.1%} | EV: {ev_cents:.1f}¢\n"
                f"Books: {books_count}"
            )
            post_discord(msg)
            return True

        elif remaining > 0:
            # Cancel unfilled order
            print(f"  Not filled, cancelling...")
            client.cancel_order(order_id)
            record_trade(
                db_path, ticker, player, stat_type, line, side,
                CONTRACTS_PER_TRADE, kalshi_ask, fair_prob, edge, books_count,
                order_id, "cancelled"
            )
            print(f"  Order cancelled")

            # Discord notification - cancelled
            post_discord(f"**Cancelled** | {player} {stat_type} {line:.0f}+ {side} | Not filled after {ORDER_TIMEOUT_SECONDS}s")
            return False

    except Exception as e:
        print(f"  [error] Failed to check/cancel order: {e}")
        record_trade(
            db_path, ticker, player, stat_type, line, side,
            CONTRACTS_PER_TRADE, kalshi_ask, fair_prob, edge, books_count,
            order_id, "unknown"
        )
        return False

    return False


def main() -> int:
    print("=" * 60)
    print("PROPS AUTO TRADER")
    print(f"Initial EV threshold: {MIN_INITIAL_EV:.0%}")
    print(f"Final EV threshold: {MIN_FINAL_EV:.0%}")
    print(f"Min books: {MIN_BOOKS}")
    print(f"Max Kalshi ask: {MAX_KALSHI_ASK_CENTS}¢")
    print(f"Min fair prob: {MIN_FAIR_PROB:.1%}")
    print(f"Max trades/day: {MAX_TRADES_PER_DAY}")
    print(f"Order timeout: {ORDER_TIMEOUT_SECONDS}s")
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

    # Step 1: Initial scan for 5%+ EV opportunities
    print("\n--- Step 1: Initial Scan ---")

    # Fetch Kalshi props first (free, unlimited)
    print("Fetching Kalshi props...")
    kalshi_props = fetch_kalshi_props()
    print(f"  Found {len(kalshi_props)} Kalshi props")

    if not kalshi_props:
        print("No Kalshi props available. Exiting.")
        return 0

    # Determine which sports have active Kalshi markets
    active_sports = get_active_sports_from_kalshi(kalshi_props)
    print(f"  Active sports: {', '.join(active_sports) or 'none'}")

    if not active_sports:
        print("No active sports with Kalshi markets. Exiting.")
        return 0

    # Only fetch Odds API for sports with active Kalshi markets (uses cache)
    print(f"Fetching sportsbook odds (only {len(active_sports)} sports, cached 30min)...")
    odds_props = fetch_all_props(sport_keys=active_sports, min_books=2, use_cache=True)
    print(f"  Found {len(odds_props)} sportsbook props")

    print("Matching and finding edges...")
    matches = match_props(odds_props, kalshi_props)
    opportunities = filter_and_rank(matches, min_edge=MIN_INITIAL_EV, top_n=50)
    print(f"  Found {len(opportunities)} opportunities with {MIN_INITIAL_EV:.0%}+ edge")

    if not opportunities:
        print("\nNo opportunities found. Exiting.")
        return 0

    # Log ALL scanned opportunities for later analysis
    print(f"\nLogging {len(opportunities)} opportunities to database...")
    for opp in opportunities:
        try:
            log_scanned_opportunity(
                db_path=DB_PATH,
                ticker=opp.get("kalshi_ticker", ""),
                player_name=opp.get("player_name", ""),
                stat_type=opp.get("stat_type", ""),
                line=opp.get("kalshi_line", 0),
                side=opp.get("best_side", ""),
                kalshi_bid=opp.get("kalshi_yes_bid", 0),
                kalshi_ask=opp.get("kalshi_yes_ask", 100),
                kalshi_prob=opp.get("bet_price", 0.5),
                fair_prob=opp.get("odds_over_prob", 0.5) if opp.get("best_side") == "OVER" else opp.get("odds_under_prob", 0.5),
                edge=opp.get("best_edge", 0),
                books_count=opp.get("odds_books_used", 0),
                volume=opp.get("kalshi_volume", 0),
                event_time=opp.get("commence_time", ""),
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

        # Pass cached odds to avoid new Odds API calls
        success = execute_trade(client, DB_PATH, opp, cached_odds=odds_props)
        if success:
            trades_placed += 1
            increment_trades_today(DB_PATH)

        # Rate limit between trades
        time.sleep(1)

    print(f"\n{'=' * 60}")
    print(f"Session complete: {trades_placed} trades placed")
    print("=" * 60)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
