#!/usr/bin/env python3
"""
Maker-style props trading script.

Instead of taking the ask (taker), this script places limit bids at the
fair value price and waits for fills. This provides liquidity and
captures edge when the market moves to us.

Workflow:
1. Scan for props where sportsbook fair value differs from Kalshi mid
2. Calculate fair bid price (where we'd have edge if filled)
3. Place resting limit orders at fair price
4. Orders stay open until filled or cancelled
5. Check fills periodically
"""

from __future__ import annotations

import os
import sqlite3
import sys
import time
from datetime import datetime, timezone
from zoneinfo import ZoneInfo
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

from kalshi_auth_client import KalshiAuthClient, prob_to_american
from kalshi_props import fetch_kalshi_props, KALSHI_TO_ODDSAPI
from oddsapi_props import fetch_all_props
from props_value import match_props

from lib.edge import calculate_edge_with_fee, DEFAULT_FEE_CENTS
from lib.validation import validate_price

# --- Configuration ---
# Lower threshold for maker orders - we're providing liquidity
MIN_EDGE_FOR_BID = float(os.getenv("MIN_EDGE_FOR_BID", "0.02"))  # 2% edge minimum
MIN_BOOKS = int(os.getenv("MIN_BOOKS", "2"))

# Price guardrails
MIN_PRICE_CENTS = int(os.getenv("MIN_PRICE_CENTS", "10"))  # Don't bid below 10¢
MAX_PRICE_CENTS = int(os.getenv("MAX_PRICE_CENTS", "90"))  # Don't bid above 90¢

# Fee
FEE_CENTS = int(os.getenv("FEE_CENTS", str(DEFAULT_FEE_CENTS)))

# How many maker orders to place per run
MAX_MAKER_ORDERS = int(os.getenv("MAX_MAKER_ORDERS", "5"))

# Contracts per order
CONTRACTS_PER_ORDER = int(os.getenv("CONTRACTS_PER_ORDER", "1"))

# Database
DB_PATH = Path(os.getenv("PROPS_DB_PATH", "props_trades.db"))

# Discord
DISCORD_WEBHOOK = os.getenv("DISCORD_PROPS_WEBHOOK", os.getenv("DISCORD_WEBHOOK_URL", ""))

DRY_RUN = os.getenv("DRY_RUN", "0") == "1"

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


def ensure_maker_table(db_path: Path) -> None:
    """Create maker orders table if not exists."""
    conn = sqlite3.connect(str(db_path))
    cur = conn.cursor()

    cur.execute("""
        CREATE TABLE IF NOT EXISTS maker_orders (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            order_id TEXT UNIQUE NOT NULL,
            ticker TEXT NOT NULL,
            player_name TEXT NOT NULL,
            stat_type TEXT NOT NULL,
            line REAL NOT NULL,
            side TEXT NOT NULL,
            kalshi_side TEXT NOT NULL,
            quantity INTEGER NOT NULL,
            bid_price_cents INTEGER NOT NULL,
            fair_prob REAL NOT NULL,
            edge_at_placement REAL NOT NULL,
            books_count INTEGER NOT NULL,
            status TEXT NOT NULL DEFAULT 'open',
            fill_price_cents INTEGER,
            filled_quantity INTEGER DEFAULT 0,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            cancelled_at TEXT,
            filled_at TEXT
        )
    """)

    conn.commit()
    conn.close()


def get_open_orders(db_path: Path) -> List[Dict[str, Any]]:
    """Get all open maker orders from database."""
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()
    cur.execute("SELECT * FROM maker_orders WHERE status = 'open'")
    rows = cur.fetchall()
    conn.close()
    return [dict(row) for row in rows]


def record_maker_order(
    db_path: Path,
    order_id: str,
    ticker: str,
    player_name: str,
    stat_type: str,
    line: float,
    side: str,
    kalshi_side: str,
    quantity: int,
    bid_price_cents: int,
    fair_prob: float,
    edge: float,
    books_count: int,
) -> None:
    """Record a new maker order."""
    now = datetime.now(ET).isoformat()
    conn = sqlite3.connect(str(db_path))
    cur = conn.cursor()

    cur.execute("""
        INSERT INTO maker_orders (
            order_id, ticker, player_name, stat_type, line, side, kalshi_side,
            quantity, bid_price_cents, fair_prob, edge_at_placement, books_count,
            status, created_at, updated_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'open', ?, ?)
    """, (
        order_id, ticker, player_name, stat_type, line, side, kalshi_side,
        quantity, bid_price_cents, fair_prob, edge, books_count,
        now, now
    ))

    conn.commit()
    conn.close()


def update_order_status(
    db_path: Path,
    order_id: str,
    status: str,
    fill_price: Optional[int] = None,
    filled_qty: Optional[int] = None,
) -> None:
    """Update order status in database."""
    now = datetime.now(ET).isoformat()
    conn = sqlite3.connect(str(db_path))
    cur = conn.cursor()

    if status == "filled":
        cur.execute("""
            UPDATE maker_orders
            SET status = ?, fill_price_cents = ?, filled_quantity = ?,
                filled_at = ?, updated_at = ?
            WHERE order_id = ?
        """, (status, fill_price, filled_qty, now, now, order_id))
    elif status == "cancelled":
        cur.execute("""
            UPDATE maker_orders
            SET status = ?, cancelled_at = ?, updated_at = ?
            WHERE order_id = ?
        """, (status, now, now, order_id))
    else:
        cur.execute("""
            UPDATE maker_orders SET status = ?, updated_at = ?
            WHERE order_id = ?
        """, (status, now, order_id))

    conn.commit()
    conn.close()


def already_has_order(db_path: Path, ticker: str) -> bool:
    """Check if we already have an open order on this ticker."""
    conn = sqlite3.connect(str(db_path))
    cur = conn.cursor()
    cur.execute(
        "SELECT 1 FROM maker_orders WHERE ticker = ? AND status = 'open'",
        (ticker,)
    )
    row = cur.fetchone()
    conn.close()
    return row is not None


def calculate_bid_price(fair_prob: float, min_edge: float, fee_cents: int) -> int:
    """
    Calculate the maximum price we'd pay to have min_edge after fees.

    If fair_prob = 50% and we want 2% edge after 2¢ fee:
    We need: fair_prob - bid_price/100 - fee_impact >= min_edge

    Simplified: bid_price <= (fair_prob - min_edge) * 100 - fee_adjustment
    """
    # Maximum price where we still have min_edge
    # edge = fair_prob - price/100 - fee/100
    # price = (fair_prob - min_edge - fee/100) * 100
    max_price = int((fair_prob - min_edge) * 100 - fee_cents)

    # Round down to be conservative
    return max(1, max_price)


def check_and_update_open_orders(client: KalshiAuthClient, db_path: Path) -> int:
    """
    Check status of all open orders and update database.
    Returns count of newly filled orders.
    """
    open_orders = get_open_orders(db_path)
    if not open_orders:
        return 0

    print(f"\nChecking {len(open_orders)} open maker orders...")
    filled_count = 0

    for order in open_orders:
        order_id = order["order_id"]
        try:
            status = client.get_order(order_id)
            kalshi_status = status.get("status", "").lower()
            filled = status.get("filled_count", 0)
            remaining = status.get("remaining_count", order["quantity"])

            if filled > 0 or kalshi_status in {"filled", "executed"}:
                fill_price = status.get("avg_fill_price", order["bid_price_cents"])
                update_order_status(db_path, order_id, "filled", fill_price, filled or order["quantity"])
                filled_count += 1

                # Calculate actual edge
                edge_info = calculate_edge_with_fee(order["fair_prob"], fill_price, FEE_CENTS)

                print(f"  FILLED: {order['player_name']} {order['stat_type']} {order['line']}+ {order['side']}")
                print(f"    Fill: {fill_price}¢ | Edge: {edge_info['net_edge']:.1%}")

                # Discord notification
                timestamp = datetime.now(ET).strftime("%I:%M %p ET")
                msg = (
                    f"**MAKER FILLED** | {order['player_name']} {order['stat_type']} {order['line']:.0f}+ {order['side']}\n"
                    f"Fill: {fill_price}¢ | Fair: {order['fair_prob']:.1%} | Edge: {edge_info['net_edge']:.1%}\n"
                    f"_{timestamp}_"
                )
                post_discord(msg)

            elif kalshi_status in {"cancelled", "canceled", "expired"}:
                update_order_status(db_path, order_id, "cancelled")
                print(f"  Cancelled: {order['player_name']} {order['stat_type']} - {kalshi_status}")

        except Exception as e:
            print(f"  [error] Failed to check order {order_id}: {e}")

    return filled_count


def find_maker_opportunities(
    kalshi_props: List[Dict[str, Any]],
    odds_props: List[Dict[str, Any]],
    min_edge: float,
) -> List[Dict[str, Any]]:
    """
    Find opportunities for maker orders.

    Returns props where we could place a bid at fair value and have edge.
    """
    matches = match_props(odds_props, kalshi_props)

    opportunities = []
    for m in matches:
        # Get the recommended side
        side = m.get("best_side", "")
        if side == "OVER":
            fair_prob = m.get("odds_over_prob", 0.5)
            kalshi_prob = m.get("kalshi_over_prob", 0.5)
        else:
            fair_prob = m.get("odds_under_prob", 0.5)
            kalshi_prob = m.get("kalshi_under_prob", 0.5)

        # Calculate what bid price would give us min_edge
        bid_price = calculate_bid_price(fair_prob, min_edge, FEE_CENTS)

        # Check price guardrails
        if bid_price < MIN_PRICE_CENTS or bid_price > MAX_PRICE_CENTS:
            continue

        # Check if bid is better than current market
        # We want to bid BELOW current ask
        if side == "OVER":
            current_ask = m.get("kalshi_yes_ask", 100)
        else:
            current_ask = 100 - m.get("kalshi_yes_bid", 0)  # No ask = 100 - yes bid

        if bid_price >= current_ask:
            # Our bid would take the ask - this is a taker order, skip
            continue

        # Calculate edge if filled at our bid
        edge_info = calculate_edge_with_fee(fair_prob, bid_price, FEE_CENTS)
        if edge_info["net_edge"] < min_edge:
            continue

        # Only need 2+ books
        if m.get("odds_books_used", 0) < MIN_BOOKS:
            continue

        m["bid_price"] = bid_price
        m["bid_edge"] = edge_info["net_edge"]
        m["fair_prob"] = fair_prob
        opportunities.append(m)

    # Sort by edge (best first)
    opportunities.sort(key=lambda x: x.get("bid_edge", 0), reverse=True)

    return opportunities


def place_maker_order(
    client: KalshiAuthClient,
    db_path: Path,
    opp: Dict[str, Any],
) -> Optional[str]:
    """
    Place a maker (resting limit) order.
    Returns order_id if successful, None otherwise.
    """
    ticker = opp["kalshi_ticker"]
    player = opp["player_name"]
    stat_type = opp["stat_type"]
    line = opp["kalshi_line"]
    side = opp["best_side"]
    bid_price = opp["bid_price"]
    fair_prob = opp["fair_prob"]
    edge = opp["bid_edge"]
    books_count = opp.get("odds_books_used", 0)

    # Determine Kalshi side
    kalshi_side = "yes" if side == "OVER" else "no"

    print(f"\n  Placing maker bid: {player} {stat_type} {line}+ {side}")
    print(f"    Bid: {bid_price}¢ | Fair: {fair_prob:.1%} | Edge if filled: {edge:.1%}")

    if DRY_RUN:
        print("    [DRY_RUN] Would place order")
        return "DRY_RUN"

    try:
        order = client.place_order(
            ticker=ticker,
            side=kalshi_side,
            quantity=CONTRACTS_PER_ORDER,
            limit_price=bid_price,
        )
        order_id = order.get("order_id", order.get("id", ""))
        print(f"    Order placed: {order_id}")

        # Record in database
        record_maker_order(
            db_path, order_id, ticker, player, stat_type, line, side, kalshi_side,
            CONTRACTS_PER_ORDER, bid_price, fair_prob, edge, books_count
        )

        # Discord notification
        timestamp = datetime.now(ET).strftime("%I:%M %p ET")
        msg = (
            f"**MAKER BID** | {player} {stat_type} {line:.0f}+ {side}\n"
            f"Bid: {bid_price}¢ | Fair: {fair_prob:.1%} | Edge if filled: {edge:.1%}\n"
            f"Books: {books_count} | Waiting for fill...\n"
            f"_{timestamp}_"
        )
        post_discord(msg)

        return order_id

    except Exception as e:
        print(f"    [error] Failed to place order: {e}")
        return None


def main() -> int:
    print("=" * 60)
    print("PROPS MAKER TRADER")
    print(f"Min edge for bid: {MIN_EDGE_FOR_BID:.0%}")
    print(f"Min books: {MIN_BOOKS}")
    print(f"Max orders per run: {MAX_MAKER_ORDERS}")
    print(f"Contracts per order: {CONTRACTS_PER_ORDER}")
    print(f"Dry run: {DRY_RUN}")
    print("=" * 60)

    ensure_maker_table(DB_PATH)

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

    # First, check status of existing open orders
    filled = check_and_update_open_orders(client, DB_PATH)
    if filled:
        print(f"\n{filled} maker order(s) filled!")

    # Count current open orders
    open_orders = get_open_orders(DB_PATH)
    open_count = len(open_orders)
    print(f"\nOpen maker orders: {open_count}")

    if open_count >= MAX_MAKER_ORDERS:
        print(f"Already at max open orders ({MAX_MAKER_ORDERS}). Exiting.")
        return 0

    slots_available = MAX_MAKER_ORDERS - open_count

    # Fetch data
    print("\n--- Finding Maker Opportunities ---")

    print("Fetching Kalshi props...")
    kalshi_props = fetch_kalshi_props()
    print(f"  Found {len(kalshi_props)} Kalshi props")

    if not kalshi_props:
        print("No Kalshi props. Exiting.")
        return 0

    print("Fetching sportsbook odds...")
    odds_props = fetch_all_props(min_books=MIN_BOOKS)
    print(f"  Found {len(odds_props)} sportsbook props")

    if not odds_props:
        print("No sportsbook props. Exiting.")
        return 0

    # Find opportunities
    opportunities = find_maker_opportunities(kalshi_props, odds_props, MIN_EDGE_FOR_BID)
    print(f"\nFound {len(opportunities)} maker opportunities with {MIN_EDGE_FOR_BID:.0%}+ edge")

    if not opportunities:
        print("No opportunities. Exiting.")
        return 0

    # Show top opportunities
    print("\nTop opportunities:")
    for opp in opportunities[:10]:
        print(f"  {opp['player_name'][:15]:<15} {opp['stat_type'][:6]:<6} {opp['kalshi_line']:>4.0f}+ {opp['best_side']:<5} "
              f"| bid={opp['bid_price']}¢ edge={opp['bid_edge']:+.1%} books={opp.get('odds_books_used', 0)}")

    # Place orders for top opportunities
    print(f"\n--- Placing up to {slots_available} maker orders ---")

    orders_placed = 0
    for opp in opportunities:
        if orders_placed >= slots_available:
            break

        ticker = opp["kalshi_ticker"]

        # Skip if we already have order on this ticker
        if already_has_order(DB_PATH, ticker):
            print(f"  [skip] Already have order on {ticker}")
            continue

        order_id = place_maker_order(client, DB_PATH, opp)
        if order_id:
            orders_placed += 1

        time.sleep(0.5)  # Rate limit

    print(f"\n{'=' * 60}")
    print(f"Placed {orders_placed} new maker orders")
    print(f"Total open orders: {open_count + orders_placed}")
    print("=" * 60)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
