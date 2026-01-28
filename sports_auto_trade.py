#!/usr/bin/env python3
"""
Automated sports betting script for moneylines, spreads, and totals.

Workflow:
1. Fetch Kalshi sports markets (moneylines, spreads, totals)
2. Fetch sportsbook odds from Odds API (with caching)
3. Match Kalshi markets to sportsbook events by team
4. Calculate edges (fair prob vs Kalshi price)
5. Filter for opportunities: EV >= threshold, min books
6. Place limit orders on Kalshi
7. Track in database with settlement support

Similar to props_auto_trade.py but for game lines instead of player props.
"""

from __future__ import annotations

import json
import os
import re
import sqlite3
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from zoneinfo import ZoneInfo

from kalshi_auth_client import KalshiAuthClient
from oddsapi_lines import (
    fair_prob_h2h,
    fair_prob_spread,
    fair_prob_total,
    normalize_team_name,
)

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
# Only refresh at scheduled times to conserve API credits (650/day limit)
CACHE_DIR = Path(os.getenv("SPORTS_CACHE_DIR", "/tmp/sports_odds_cache"))
CACHE_TTL_SECONDS = int(os.getenv("SPORTS_CACHE_TTL", "14400"))  # 4 hours - but we use scheduled refresh

# Scheduled refresh times (ET) - only fetch fresh data at these times
SCHEDULED_REFRESH_HOURS = [6, 8, 12, 16, 18, 20]  # 6am, 8am, 12pm, 4pm, 6pm, 8pm ET

# Sports to include
SPORT_KEYS = os.getenv("SPORTS_SPORT_KEYS", "basketball_nba,basketball_ncaab,americanfootball_nfl").split(",")


# --- Parsing helpers from kalshi_sports_value.py ---

TOTAL_RE = re.compile(r"\b(over|under)\s*([0-9]+(?:\.[0-9]+)?)\b", re.IGNORECASE)
SPREAD_RE = re.compile(
    r"^(?P<team>.+?)\s*(?P<spread>[+-]?\d+(?:\.\d+)?)\s*(?:pts?|points?)?\s*$",
    re.IGNORECASE,
)


def parse_matchup(title: str) -> Optional[Tuple[str, str]]:
    """Extract team A and team B from a title like 'Chicago at Brooklyn' or 'A vs B'."""
    t = str(title or "")
    if not t:
        return None

    def _clean_team_fragment(raw: str) -> str:
        s = re.sub(r"[\(\[].*?[\)\]]", " ", raw)
        s = re.sub(
            r"(?i)\b(moneyline|money line|ml|spread|total|over|under|h2h|line|game|match|odds)\b",
            " ",
            s,
        )
        s = re.sub(r"(?i)\b(wins?|to win|beat|defeat)\b", " ", s)
        s = re.sub(r"[|/]+", " ", s)
        s = re.sub(r"\s+", " ", s).strip(" -–—:|")
        return s.strip()

    t = re.sub(r"[\(\[].*?[\)\]]", " ", t)
    t = t.replace("@", " at ").replace(" vs. ", " vs ").replace(" v. ", " vs ").replace(" v ", " vs ")
    t = " ".join(t.split()).strip()
    if ":" in t:
        t_tail = t.split(":")[-1].strip()
        if len(t_tail) >= 5:
            t = t_tail

    patterns = [
        r"(?i)\bwill\s+(.+?)\s+(?:beat|defeat|win(?:\s+against|\s+over|\s+vs)?)\s+(.+?)(?:\?|$)",
        r"(?i)\b(.+?)\s+(?:beat|defeat|win(?:\s+against|\s+over|\s+vs)?)\s+(.+?)(?:\?|$)",
        r"(?i)\b(.+?)\s+(at|vs|versus)\s+(.+?)(?:\s*(?:\||-|–|—|:).*)?$",
    ]

    for pat in patterns:
        m = re.search(pat, t)
        if not m:
            continue
        if len(m.groups()) >= 3 and m.group(2).lower() in ("at", "vs", "versus"):
            a_raw, b_raw = m.group(1), m.group(3)
        else:
            a_raw, b_raw = m.group(1), m.group(2)
        a = _clean_team_fragment(str(a_raw or ""))
        b = _clean_team_fragment(str(b_raw or ""))
        if not a or not b:
            continue
        if a.lower() == b.lower():
            continue
        return a, b

    return None


def parse_line_from_subtitle(line_type: str, subtitle: str) -> Optional[Dict[str, Any]]:
    """Parse spread/total/h2h info from subtitle."""
    s = " ".join(str(subtitle or "").split())
    if not s:
        return None

    if line_type == "total":
        m = TOTAL_RE.search(s)
        if not m:
            return None
        side = m.group(1).lower()
        pts = float(m.group(2))
        return {"side": side, "points": pts, "label": f"{side.title()} {pts:g}"}

    if line_type == "spread":
        m = SPREAD_RE.match(s)
        if not m:
            return None
        team = m.group("team").strip()
        spread = float(m.group("spread"))
        return {"team": team, "spread": spread, "label": f"{team} {spread:+g}"}

    if line_type == "h2h":
        team = s
        team = re.sub(r"\b(wins|win|to win)\b", "", team, flags=re.IGNORECASE).strip()
        if not team:
            return None
        return {"team": team, "label": team}

    return None


def name_similarity(a: str, b: str) -> float:
    """Deterministic similarity score in [0,1] for team name matching."""
    na = normalize_team_name(a)
    nb = normalize_team_name(b)
    if not na or not nb:
        return 0.0
    if na == nb:
        return 1.0
    if na in nb or nb in na:
        short, long_ = (na, nb) if len(na) <= len(nb) else (nb, na)
        if len(short) >= 4:
            return 0.90
        if len(short) == 3 and long_.startswith(short):
            return 0.85

    stop = {
        "pro", "professional", "football", "basketball", "baseball", "hockey",
        "game", "match", "season", "playoffs", "postseason",
        "nfl", "nba", "mlb", "nhl", "ncaaf", "ncaab",
    }

    ta = [t for t in na.split() if t not in stop]
    tb = [t for t in nb.split() if t not in stop]
    if not ta or not tb:
        ta = na.split()
        tb = nb.split()

    sa = set(ta)
    sb = set(tb)
    inter = len(sa & sb)
    union = len(sa | sb)
    if union == 0:
        return 0.0
    return inter / union


def best_event_match(
    events: List[Dict[str, Any]], team_a: str, team_b: str
) -> Optional[Dict[str, Any]]:
    """Find the best odds event matching the two team strings."""
    best: Optional[Dict[str, Any]] = None
    best_score = 0.0

    for e in events:
        home = e.get("home_team")
        away = e.get("away_team")
        if not home or not away:
            continue
        s1 = name_similarity(team_a, str(home)) + name_similarity(team_b, str(away))
        s2 = name_similarity(team_a, str(away)) + name_similarity(team_b, str(home))
        s = max(s1, s2)
        if s > best_score:
            best_score = s
            best = e

    # Require both teams to be at least somewhat close
    if best is None or best_score < 1.30:
        return None
    return best


def map_team_to_event_team(
    team_guess: str, odds_event: Dict[str, Any], min_score: float = 0.72
) -> Tuple[Optional[str], float]:
    """Map a Kalshi team name to the corresponding Odds API team name."""
    home = str(odds_event.get("home_team") or "")
    away = str(odds_event.get("away_team") or "")
    if not team_guess or not home or not away:
        return None, 0.0

    sh = name_similarity(team_guess, home)
    sa = name_similarity(team_guess, away)

    best_team = home if sh >= sa else away
    best_score = max(sh, sa)

    if best_score < float(min_score):
        return None, best_score
    return best_team, best_score


def get_ev_bucket(edge: float) -> str:
    """Classify edge into bucket for analytics."""
    if edge < 0.05:
        return "0-5%"
    elif edge < 0.10:
        return "5-10%"
    elif edge < 0.15:
        return "10-15%"
    elif edge < 0.20:
        return "15-20%"
    else:
        return "20%+"


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


def is_scheduled_refresh_time() -> bool:
    """Check if current time is a scheduled refresh time (within 30 min window)."""
    now_et = datetime.now(ET)
    return now_et.hour in SCHEDULED_REFRESH_HOURS and now_et.minute <= 30


def fetch_odds_events_draftkings(sport_keys: List[str]) -> List[Dict[str, Any]]:
    """
    Fetch events from DraftKings API (FREE, no API key needed).

    This is the preferred source for sports lines to conserve Odds API credits.
    """
    from draftkings_lines import fetch_all_draftkings_events

    cache_key = "sports_lines_draftkings"

    # Check cache first (unless at scheduled refresh time)
    if not is_scheduled_refresh_time():
        cached = _read_cache(cache_key)
        if cached:
            print(f"  [cache] DraftKings: {len(cached)} events from cache")
            return cached

    try:
        events = fetch_all_draftkings_events(sport_keys)
        if events:
            _write_cache(cache_key, events)
            print(f"  [DraftKings] Fetched {len(events)} events (FREE - no API credits used)")
        return events or []
    except Exception as e:
        print(f"  [error] DraftKings fetch failed: {e}")
        # Try cache as fallback
        cached = _read_cache(cache_key)
        if cached:
            print(f"  [cache fallback] {len(cached)} events")
            return cached
        return []


def fetch_odds_events_oddsapi(sport_keys: List[str]) -> List[Dict[str, Any]]:
    """
    Fetch events from Odds API (uses API credits).

    Only use this if DraftKings doesn't have the data you need.
    """
    from oddsapi_lines import fetch_all_sports_events

    cache_key = "sports_lines_oddsapi"

    # Check cache first (unless at scheduled refresh time)
    if not is_scheduled_refresh_time():
        cached = _read_cache(cache_key)
        if cached:
            print(f"  [cache] Odds API: {len(cached)} events from cache")
            return cached

    try:
        events = fetch_all_sports_events(sport_keys, markets="h2h,spreads,totals", regions="us")
        if events:
            _write_cache(cache_key, events)
            print(f"  [Odds API] Fetched {len(events)} events (uses API credits)")
        return events or []
    except Exception as e:
        print(f"  [error] Odds API fetch failed: {e}")
        cached = _read_cache(cache_key)
        if cached:
            print(f"  [cache fallback] {len(cached)} events")
            return cached
        return []


def try_execute_trade(
    client: KalshiAuthClient,
    db_path: Path,
    ticker: str,
    event_title: str,
    line_type: str,
    line_label: str,
    side: str,
    kalshi_ask: int,
    fair_prob: float,
    edge: float,
    books_count: int,
) -> bool:
    """
    Execute a trade on Kalshi for a sports line opportunity.

    Returns True if trade was filled, False otherwise.
    """
    kalshi_side = "yes" if side.upper() in ("YES", "OVER") else "no"

    # Check if already traded this ticker today
    if already_traded_today(db_path, ticker):
        print(f"    [skip] Already traded today: {ticker}")
        return False

    print(f"    Placing limit order: {CONTRACTS_PER_TRADE} contract(s) at {kalshi_ask}¢")

    if DRY_RUN:
        print("    [DRY_RUN] Would place order")
        record_trade(
            db_path, ticker, event_title, line_type, line_label, side,
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
        print(f"    Order placed: {order_id}")

        # Discord notification - order placed
        ev_cents = edge * 100
        timestamp = datetime.now(ET).strftime("%I:%M %p ET")
        place_msg = (
            f"**Sports Order Placed** | {line_type.upper()} {line_label}\n"
            f"Event: {event_title}\n"
            f"Order ID: {order_id}\n"
            f"Side: {side} | Limit: {kalshi_ask}¢ | Fair: {fair_prob:.1%} | Edge: {edge:.1%} | EV: {ev_cents:.1f}¢\n"
            f"Books: {books_count} | Waiting {ORDER_TIMEOUT_SECONDS}s for fill...\n"
            f"_{timestamp}_"
        )
        post_discord(place_msg)

    except Exception as e:
        print(f"    [error] Failed to place order: {e}")
        record_trade(
            db_path, ticker, event_title, line_type, line_label, side,
            CONTRACTS_PER_TRADE, kalshi_ask, fair_prob, edge, books_count,
            None, "error"
        )
        timestamp = datetime.now(ET).strftime("%I:%M %p ET")
        post_discord(
            f"**Sports Order Failed** | {line_type.upper()} {line_label}\n"
            f"Event: {event_title}\nError: {str(e)[:100]}\n_{timestamp}_"
        )
        return False

    # Wait and check fill status
    print(f"    Waiting {ORDER_TIMEOUT_SECONDS}s for fill...")
    time.sleep(ORDER_TIMEOUT_SECONDS)

    try:
        order_status = client.get_order(order_id)
        status = order_status.get("status", "").lower()
        filled_count = order_status.get("filled_count", 0)
        remaining = order_status.get("remaining_count", CONTRACTS_PER_TRADE)

        if filled_count > 0 or status in {"filled", "executed"}:
            if filled_count == 0:
                filled_count = CONTRACTS_PER_TRADE
            fill_price = order_status.get("avg_fill_price", kalshi_ask)
            print(f"    FILLED: {filled_count} contract(s) at {fill_price}¢")
            record_trade(
                db_path, ticker, event_title, line_type, line_label, side,
                filled_count, kalshi_ask, fair_prob, edge, books_count,
                order_id, "filled", fill_price
            )

            # Discord notification - filled
            actual_edge = fair_prob - (fill_price / 100.0)
            ev_cents = actual_edge * 100
            timestamp = datetime.now(ET).strftime("%I:%M %p ET")
            msg = (
                f"**Sports FILLED** | {line_type.upper()} {line_label}\n"
                f"Event: {event_title}\n"
                f"Order ID: {order_id}\n"
                f"Fill: {fill_price}¢ | Fair: {fair_prob:.1%} | Edge: {actual_edge:.1%} | EV: {ev_cents:.1f}¢\n"
                f"Books: {books_count}\n"
                f"_{timestamp}_"
            )
            post_discord(msg)
            return True

        elif remaining > 0:
            # Cancel unfilled order
            print(f"    Not filled, cancelling...")
            client.cancel_order(order_id)
            record_trade(
                db_path, ticker, event_title, line_type, line_label, side,
                CONTRACTS_PER_TRADE, kalshi_ask, fair_prob, edge, books_count,
                order_id, "cancelled"
            )
            timestamp = datetime.now(ET).strftime("%I:%M %p ET")
            post_discord(
                f"**Sports Cancelled** | {line_type.upper()} {line_label}\n"
                f"Event: {event_title}\n"
                f"Order ID: {order_id} | Limit: {kalshi_ask}¢ | Not filled after {ORDER_TIMEOUT_SECONDS}s\n"
                f"_{timestamp}_"
            )
            return False

    except Exception as e:
        print(f"    [error] Failed to check order status: {e}")
        record_trade(
            db_path, ticker, event_title, line_type, line_label, side,
            CONTRACTS_PER_TRADE, kalshi_ask, fair_prob, edge, books_count,
            order_id, "unknown"
        )

    return False


def calculate_edge_for_market(
    market: Dict[str, Any],
    odds_events: List[Dict[str, Any]],
) -> Optional[Dict[str, Any]]:
    """
    Match a Kalshi market to Odds API event and calculate edge.

    Returns opportunity dict if edge >= MIN_EDGE, else None.
    """
    ticker = market.get("ticker", "")
    title = market.get("title", "")
    line_type = market.get("line_type", "")
    yes_ask = market.get("yes_ask", 100)
    no_ask = market.get("no_ask", 100)

    # Parse matchup from title
    matchup = parse_matchup(title)
    if not matchup:
        return None

    team_a, team_b = matchup

    # Find matching odds event
    odds_event = best_event_match(odds_events, team_a, team_b)
    if not odds_event:
        return None

    home_team = odds_event.get("home_team", "")
    away_team = odds_event.get("away_team", "")
    commence_time = odds_event.get("commence_time", "")

    opportunities = []

    if line_type == "moneyline":
        # For moneyline, check both teams
        for team_guess in [team_a, team_b]:
            mapped_team, score = map_team_to_event_team(team_guess, odds_event)
            if not mapped_team:
                continue

            fair = fair_prob_h2h(odds_event, mapped_team, min_books=MIN_BOOKS)
            if not fair:
                continue

            fair_prob = fair.get("fair_prob", 0.5)
            books_count = fair.get("books_used", 0)

            # Check YES side (betting on this team to win)
            yes_prob = yes_ask / 100.0
            yes_edge = fair_prob - yes_prob

            if yes_edge >= MIN_EDGE and yes_ask <= MAX_KALSHI_ASK_CENTS:
                opportunities.append({
                    "ticker": ticker,
                    "event_title": f"{away_team} at {home_team}",
                    "line_type": line_type,
                    "line_label": f"{mapped_team} ML",
                    "side": "YES",
                    "kalshi_ask": yes_ask,
                    "fair_prob": fair_prob,
                    "edge": yes_edge,
                    "books_count": books_count,
                    "commence_time": commence_time,
                })

            # Check NO side (betting against this team)
            no_prob = no_ask / 100.0
            no_fair = 1.0 - fair_prob
            no_edge = no_fair - no_prob

            if no_edge >= MIN_EDGE and no_ask <= MAX_KALSHI_ASK_CENTS:
                other_team = home_team if mapped_team == away_team else away_team
                opportunities.append({
                    "ticker": ticker,
                    "event_title": f"{away_team} at {home_team}",
                    "line_type": line_type,
                    "line_label": f"{other_team} ML",
                    "side": "NO",
                    "kalshi_ask": no_ask,
                    "fair_prob": no_fair,
                    "edge": no_edge,
                    "books_count": books_count,
                    "commence_time": commence_time,
                })

    elif line_type == "spread":
        # Parse spread from subtitle or title
        yes_sub = market.get("yes_sub_title", "")
        no_sub = market.get("no_sub_title", "")

        yes_line = parse_line_from_subtitle("spread", yes_sub)
        no_line = parse_line_from_subtitle("spread", no_sub)

        for line_info, sub, kalshi_ask_val, side in [
            (yes_line, yes_sub, yes_ask, "YES"),
            (no_line, no_sub, no_ask, "NO"),
        ]:
            if not line_info:
                continue

            team_guess = line_info.get("team", "")
            spread = line_info.get("spread", 0)

            mapped_team, score = map_team_to_event_team(team_guess, odds_event)
            if not mapped_team:
                continue

            fair = fair_prob_spread(odds_event, mapped_team, spread, min_books=MIN_BOOKS)
            if not fair:
                continue

            fair_prob = fair.get("fair_prob", 0.5)
            books_count = fair.get("books_used", 0)

            kalshi_prob = kalshi_ask_val / 100.0
            edge = fair_prob - kalshi_prob

            if edge >= MIN_EDGE and kalshi_ask_val <= MAX_KALSHI_ASK_CENTS:
                opportunities.append({
                    "ticker": ticker,
                    "event_title": f"{away_team} at {home_team}",
                    "line_type": line_type,
                    "line_label": f"{mapped_team} {spread:+g}",
                    "side": side,
                    "kalshi_ask": kalshi_ask_val,
                    "fair_prob": fair_prob,
                    "edge": edge,
                    "books_count": books_count,
                    "commence_time": commence_time,
                })

    elif line_type == "total":
        # Parse total from title or subtitle
        m = TOTAL_RE.search(title)
        if not m:
            return None

        total_side = m.group(1).lower()  # "over" or "under"
        total_points = float(m.group(2))

        fair = fair_prob_total(odds_event, total_side, total_points, min_books=MIN_BOOKS)
        if not fair:
            return None

        fair_prob = fair.get("fair_prob", 0.5)
        books_count = fair.get("books_used", 0)

        # YES means the total line hits
        yes_prob = yes_ask / 100.0
        yes_edge = fair_prob - yes_prob

        if yes_edge >= MIN_EDGE and yes_ask <= MAX_KALSHI_ASK_CENTS:
            opportunities.append({
                "ticker": ticker,
                "event_title": f"{away_team} at {home_team}",
                "line_type": line_type,
                "line_label": f"{total_side.title()} {total_points:g}",
                "side": "YES",
                "kalshi_ask": yes_ask,
                "fair_prob": fair_prob,
                "edge": yes_edge,
                "books_count": books_count,
                "commence_time": commence_time,
            })

        # NO means opposite side
        no_prob = no_ask / 100.0
        no_fair = 1.0 - fair_prob
        no_edge = no_fair - no_prob
        opposite_side = "under" if total_side == "over" else "over"

        if no_edge >= MIN_EDGE and no_ask <= MAX_KALSHI_ASK_CENTS:
            opportunities.append({
                "ticker": ticker,
                "event_title": f"{away_team} at {home_team}",
                "line_type": line_type,
                "line_label": f"{opposite_side.title()} {total_points:g}",
                "side": "NO",
                "kalshi_ask": no_ask,
                "fair_prob": no_fair,
                "edge": no_edge,
                "books_count": books_count,
                "commence_time": commence_time,
            })

    # Return best opportunity if any
    if opportunities:
        return max(opportunities, key=lambda x: x["edge"])
    return None


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

    # Fetch odds - only at scheduled refresh times to conserve API credits
    # At ~54 credits/day (6 refreshes × 9 sports), well under 650 limit
    print("Fetching sportsbook odds...")
    if is_scheduled_refresh_time():
        print("  [scheduled refresh] Fetching fresh data from Odds API")
    else:
        print("  [off-schedule] Using cached data only (conserving API credits)")
    odds_events = fetch_odds_events_oddsapi(SPORT_KEYS)
    print(f"  Found {len(odds_events)} sportsbook events")

    if not odds_events:
        print("No sportsbook odds available. Exiting.")
        return 0

    # Match and find opportunities
    print("\n--- Matching and Edge Calculation ---")

    opportunities: List[Dict[str, Any]] = []
    matched_count = 0
    unmatched_count = 0

    for market in kalshi_markets:
        opp = calculate_edge_for_market(market, odds_events)
        if opp:
            opportunities.append(opp)
            matched_count += 1
        else:
            unmatched_count += 1

    print(f"  Matched: {matched_count}, Unmatched: {unmatched_count}")
    print(f"  Opportunities with edge >= {MIN_EDGE:.0%}: {len(opportunities)}")

    # Sort by edge
    opportunities.sort(key=lambda x: x["edge"], reverse=True)

    # Log all opportunities
    for opp in opportunities[:20]:
        bucket = get_ev_bucket(opp["edge"])
        print(
            f"  {opp['edge']*100:5.1f}% | {opp['line_type']:>10} | {opp['line_label']} | "
            f"ask={opp['kalshi_ask']}¢ fair={opp['fair_prob']:.1%} | books={opp['books_count']} | {bucket}"
        )

        # Log to database for analysis
        log_scanned_opportunity(
            DB_PATH,
            opp["ticker"],
            opp["event_title"],
            opp["line_type"],
            opp["line_label"],
            opp["side"],
            0,  # bid not tracked
            opp["kalshi_ask"],
            opp["fair_prob"],
            opp["edge"],
            opp["books_count"],
            opp.get("commence_time", ""),
        )

    # Execute trades
    print("\n--- Executing Trades ---")
    trades_executed = 0

    for opp in opportunities:
        if trades_executed >= remaining_trades:
            print(f"Reached daily trade limit ({MAX_TRADES_PER_DAY})")
            break

        print(f"\n  Processing: {opp['line_type']} {opp['line_label']} @ {opp['edge']*100:.1f}% edge")

        success = try_execute_trade(
            client=client,
            db_path=DB_PATH,
            ticker=opp["ticker"],
            event_title=opp["event_title"],
            line_type=opp["line_type"],
            line_label=opp["line_label"],
            side=opp["side"],
            kalshi_ask=opp["kalshi_ask"],
            fair_prob=opp["fair_prob"],
            edge=opp["edge"],
            books_count=opp["books_count"],
        )

        if success:
            trades_executed += 1
            increment_trades_today(DB_PATH)

    # Session summary
    print(f"\n{'=' * 60}")
    print(f"Session complete: {trades_executed} trades executed")
    if opportunities:
        avg_edge = sum(o["edge"] for o in opportunities) / len(opportunities)
        print(f"Opportunities: {len(opportunities)}, Avg edge: {avg_edge:.1%}")
    print("=" * 60)

    # Post summary to Discord
    if trades_executed > 0 or len(opportunities) > 0:
        timestamp = datetime.now(ET).strftime("%I:%M %p ET")
        summary_lines = [f"**Sports Scan Summary** | {timestamp}"]
        summary_lines.append(f"Markets scanned: {len(kalshi_markets)}")
        summary_lines.append(f"Opportunities: {len(opportunities)}")
        summary_lines.append(f"Trades executed: {trades_executed}")

        if opportunities:
            summary_lines.append("\nTop opportunities:")
            for opp in opportunities[:5]:
                summary_lines.append(
                    f"- {opp['line_type']} {opp['line_label']}: "
                    f"{opp['edge']*100:.1f}% edge @ {opp['kalshi_ask']}¢"
                )

        post_discord("\n".join(summary_lines))

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
