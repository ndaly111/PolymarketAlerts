#!/usr/bin/env python3
"""
Player props value scanner.

Compares Kalshi player prop markets to sportsbook consensus odds
from The Odds API to find mispriced props.
"""

import os
import json
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from oddsapi_props import fetch_all_props, normalize_player_name
from kalshi_props import fetch_kalshi_props, kalshi_props_by_player


def make_session() -> requests.Session:
    """Create a requests session with retry logic."""
    s = requests.Session()
    retries = Retry(
        total=6,
        backoff_factor=1.0,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=frozenset(["GET", "POST"]),
        raise_on_status=False,
    )
    s.mount("https://", HTTPAdapter(max_retries=retries))
    s.headers.update({"Accept": "application/json", "User-Agent": "props-value/1.0"})
    return s


SESSION = make_session()

# Environment config
def env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None or not str(value).strip():
        return default
    return int(value)

def env_float(name: str, default: float) -> float:
    value = os.getenv(name)
    if value is None or not str(value).strip():
        return default
    return float(value)

MIN_EDGE = env_float("MIN_EDGE", 0.05)  # 5% edge threshold
MIN_BOOKS = env_int("MIN_BOOKS", 2)  # Require at least 2 books for reliable consensus
TOP_N = env_int("TOP_N", 20)
DEBUG_MODE = env_int("DEBUG_MODE", 0)
DRY_RUN = env_int("DRY_RUN", 0)

# Kelly criterion config
KELLY_FRACTION = env_float("KELLY_FRACTION", 0.5)  # Half Kelly by default

# Discord config
DISCORD_WEBHOOK_URL = os.getenv("DISCORD_WEBHOOK_URL", "").strip()
DISCORD_MENTION = os.getenv("DISCORD_MENTION", "").strip()
DISCORD_TOP_N = env_int("DISCORD_TOP_N", 8)


LINE_TOLERANCE = env_float("LINE_TOLERANCE", 0.5)  # Max line difference to match


def match_props(
    odds_props: List[Dict[str, Any]],
    kalshi_props: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """
    Match Kalshi props to Odds API props by player name, stat type, and line.

    Returns list of matched pairs with edge calculations.
    """
    # Index Odds API props by (player_norm, prop_type)
    # Then we can search for closest line
    odds_by_player_type: Dict[Tuple[str, str], List[Dict[str, Any]]] = {}
    for op in odds_props:
        key = (
            op.get("player_name_norm", ""),
            op.get("prop_type", ""),
        )
        if key not in odds_by_player_type:
            odds_by_player_type[key] = []
        odds_by_player_type[key].append(op)

    matches = []

    for kp in kalshi_props:
        player_norm = kp.get("player_name_norm", "")
        oddsapi_type = kp.get("oddsapi_type", "")
        kalshi_line = kp.get("line", 0)

        if not player_norm or not oddsapi_type:
            continue

        key = (player_norm, oddsapi_type)
        odds_list = odds_by_player_type.get(key, [])

        if not odds_list:
            continue

        # Kalshi "8+" means > 7.5, so target odds line is kalshi_line - 0.5
        target_line = kalshi_line - 0.5

        # Find closest line within tolerance
        best_op = None
        best_diff = float('inf')
        for op in odds_list:
            odds_line = op.get("line", 0)
            diff = abs(odds_line - target_line)
            if diff <= LINE_TOLERANCE and diff < best_diff:
                best_diff = diff
                best_op = op

        if not best_op:
            continue

        op = best_op
        odds_line = op.get("line", 0)

        # Calculate edges
        # Kalshi: yes_mid is probability of over
        kalshi_over_prob = kp.get("kalshi_prob")
        if kalshi_over_prob is None or kalshi_over_prob == 0.5:
            # Skip illiquid markets with no real price
            yes_bid = kp.get("yes_bid", 0)
            yes_ask = kp.get("yes_ask", 100)
            if yes_bid == 0 or yes_ask >= 99:
                continue
            kalshi_over_prob = (yes_bid + yes_ask) / 2 / 100

        # Odds API fair prob
        odds_over_prob = op.get("fair_over_prob", 0.5)
        odds_under_prob = op.get("fair_under_prob", 0.5)

        # Edge = fair prob - kalshi prob
        # Positive edge on Over = books think over is more likely than Kalshi
        over_edge = odds_over_prob - kalshi_over_prob
        under_edge = odds_under_prob - (1 - kalshi_over_prob)

        # Determine best side and calculate EV/Kelly for that side
        if over_edge > under_edge:
            best_side = "OVER"
            best_edge = over_edge
            fair_prob = odds_over_prob
            kalshi_price = kalshi_over_prob  # Price to buy YES (over)
        else:
            best_side = "UNDER"
            best_edge = under_edge
            fair_prob = odds_under_prob
            kalshi_price = 1 - kalshi_over_prob  # Price to buy NO (under)

        # EV per contract: Expected profit per $1 notional
        # Buy at kalshi_price, win $1 with prob fair_prob
        # EV = fair_prob * (1 - kalshi_price) - (1 - fair_prob) * kalshi_price
        #    = fair_prob - kalshi_price = edge
        ev_per_contract = best_edge  # In dollars (contracts pay $1)

        # ROI: Return on capital risked
        # ROI = EV / cost = edge / kalshi_price
        roi = best_edge / kalshi_price if kalshi_price > 0 else 0

        # Kelly criterion: optimal fraction of bankroll to bet
        # For binary bet: kelly = (p * b - q) / b where p=win prob, q=lose prob, b=odds
        # For Kalshi: b = (1 - price) / price (profit per dollar risked)
        # kelly = (fair_prob * (1-price)/price - (1-fair_prob)) / ((1-price)/price)
        #       = (fair_prob * (1-price) - (1-fair_prob) * price) / (1-price)
        #       = (fair_prob - price) / (1 - price)
        #       = edge / (1 - kalshi_price)
        full_kelly = best_edge / (1 - kalshi_price) if kalshi_price < 1 else 0
        kelly_bet = full_kelly * KELLY_FRACTION

        match = {
            # Player info
            "player_name": kp.get("player_name", ""),
            "player_name_norm": player_norm,
            # Prop info
            "stat_type": kp.get("stat_type", ""),
            "kalshi_line": kalshi_line,
            "odds_line": odds_line,
            # Kalshi data
            "kalshi_over_prob": kalshi_over_prob,
            "kalshi_under_prob": 1 - kalshi_over_prob,
            "kalshi_yes_bid": kp.get("yes_bid"),
            "kalshi_yes_ask": kp.get("yes_ask"),
            "kalshi_ticker": kp.get("ticker", ""),
            "kalshi_volume": kp.get("volume", 0),
            "kalshi_liquidity": kp.get("liquidity", 0),
            # Odds API data
            "odds_over_prob": odds_over_prob,
            "odds_under_prob": odds_under_prob,
            "odds_books_used": op.get("books_used", 0),
            "odds_over_odds": op.get("over_odds"),
            "odds_under_odds": op.get("under_odds"),
            # Event info
            "home_team": op.get("home_team", ""),
            "away_team": op.get("away_team", ""),
            "commence_time": op.get("commence_time", ""),
            "sport_key": op.get("sport_key", ""),
            # Edge calculations
            "over_edge": over_edge,
            "under_edge": under_edge,
            "best_edge": best_edge,
            "best_side": best_side,
            # EV and Kelly
            "ev_per_contract": ev_per_contract,
            "roi": roi,
            "full_kelly": full_kelly,
            "kelly_bet": kelly_bet,
            "bet_price": kalshi_price,  # The price you'd pay for the recommended side
        }
        matches.append(match)

    return matches


def filter_and_rank(
    matches: List[Dict[str, Any]],
    min_edge: float = MIN_EDGE,
    top_n: int = TOP_N,
) -> List[Dict[str, Any]]:
    """Filter by minimum edge and rank by best edge."""
    filtered = [m for m in matches if m.get("best_edge", 0) >= min_edge]
    filtered.sort(key=lambda x: x.get("best_edge", 0), reverse=True)
    return filtered[:top_n]


def format_edge_report(matches: List[Dict[str, Any]]) -> str:
    """Format matches as a readable report."""
    if not matches:
        return "No edges found above threshold."

    lines = [
        f"{'Player':<18} {'Stat':<7} {'Line':>5} {'Side':<5} {'Edge':>6} "
        f"{'Kalshi':>7} {'Books':>7} {'EV':>5} {'½K':>5} {'#Bk':>3} {'Matchup':<20}",
        "-" * 105,
    ]

    for m in matches:
        player = m.get("player_name", "")[:17]
        stat = m.get("stat_type", "")[:6]
        k_line = m.get("kalshi_line", 0)
        side = m.get("best_side", "")

        # Get probs for recommended side
        if side == "OVER":
            kalshi_prob = m.get("kalshi_over_prob", 0)
            books_prob = m.get("odds_over_prob", 0)
        else:
            kalshi_prob = m.get("kalshi_under_prob", 0)
            books_prob = m.get("odds_under_prob", 0)

        edge = m.get("best_edge", 0)
        ev_cents = m.get("ev_per_contract", 0) * 100
        kelly = round_kelly(m.get("kelly_bet", 0))
        books = m.get("odds_books_used", 0)

        away = m.get("away_team", "")[:9]
        home = m.get("home_team", "")[:9]
        matchup = f"{away}@{home}" if away or home else ""

        kalshi_str = prob_to_american(kalshi_prob)
        books_str = prob_to_american(books_prob)
        kelly_str = f"{kelly:.0%}" if kelly > 0 else "—"

        lines.append(
            f"{player:<18} {stat:<7} {k_line:>4.0f}+ {side:<5} {edge:>+5.1%} "
            f"{kalshi_str:>7} {books_str:>7} {ev_cents:>+4.0f}¢ {kelly_str:>5} {books:>3} {matchup:<20}"
        )

    lines.append("")
    lines.append(f"Kelly fraction: {KELLY_FRACTION:.0%} | Line = Kalshi threshold (8+ = 8 or more)")

    return "\n".join(lines)


def prob_to_american(prob: float) -> str:
    """Convert probability to American odds string."""
    if prob <= 0 or prob >= 1:
        return "N/A"
    if prob >= 0.5:
        odds = -100 * prob / (1 - prob)
        return f"{int(odds):+d}"
    else:
        odds = 100 * (1 - prob) / prob
        return f"+{int(odds)}"


def round_kelly(kelly: float) -> float:
    """Round Kelly down to nearest 5% increment, with 2.5% as an option."""
    if kelly <= 0:
        return 0
    if kelly < 0.025:
        return 0
    if kelly < 0.05:
        return 0.025
    # Round down to nearest 5%
    return (int(kelly * 20) / 20)


def format_matchup(home_team: str, away_team: str) -> str:
    """Format matchup as 'AWAY @ HOME'."""
    if not home_team and not away_team:
        return ""
    away = away_team[:10] if len(away_team) > 10 else away_team
    home = home_team[:10] if len(home_team) > 10 else home_team
    return f"{away} @ {home}"


def format_discord_report(matches: List[Dict[str, Any]]) -> str:
    """Format matches for Discord (readable multi-line format)."""
    if not matches:
        return "No prop edges found above threshold."

    now_str = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    lines = [f"**Kalshi Props Value** ({now_str})\n"]

    for m in matches:
        player = m.get("player_name", "")
        stat = m.get("stat_type", "")
        k_line = m.get("kalshi_line", 0)
        o_line = m.get("odds_line", 0)
        side = m.get("best_side", "")

        # Get the right probabilities for the recommended side
        if side == "OVER":
            kalshi_prob = m.get("kalshi_over_prob", 0)
            books_prob = m.get("odds_over_prob", 0)
        else:
            kalshi_prob = m.get("kalshi_under_prob", 0)
            books_prob = m.get("odds_under_prob", 0)

        kalshi_american = prob_to_american(kalshi_prob)
        books_american = prob_to_american(books_prob)

        ev_cents = m.get("ev_per_contract", 0) * 100  # Convert to cents
        kelly = round_kelly(m.get("kelly_bet", 0))
        matchup = format_matchup(m.get("home_team", ""), m.get("away_team", ""))
        books_count = m.get("odds_books_used", 0)

        # Format: Bold player - stat
        # Line: K 8+ vs B 7.5 | OVER
        # Odds: Kalshi -150 vs Books -180 (3 books)
        # EV: +8¢ | Kelly: 5%
        lines.append(f"**{player}** - {stat}")
        lines.append(f"Line: K {k_line:.0f}+ vs B {o_line:.1f}")
        lines.append(f"{side} | Kalshi {kalshi_american} vs Books {books_american} ({books_count}bk)")
        kelly_str = f"{kelly:.1%}" if kelly > 0 else "—"
        lines.append(f"EV: {ev_cents:+.0f}¢ | ½K: {kelly_str}")
        if matchup:
            lines.append(f"_{matchup}_")
        lines.append("")  # Blank line between entries

    return "\n".join(lines)


def post_discord(text: str) -> None:
    """Post a message to Discord via webhook."""
    if not DISCORD_WEBHOOK_URL:
        print("DISCORD_WEBHOOK_URL not set, skipping Discord post.")
        return

    content = f"{DISCORD_MENTION}\n{text}" if DISCORD_MENTION else text

    if DRY_RUN:
        print("[DRY_RUN] Would post to Discord:")
        print(content)
        return

    # Discord limit is 2000 chars, keep some margin
    if len(content) > 1900:
        content = content[:1890] + "\n…(trimmed)"

    payload = {"content": content, "allowed_mentions": {"parse": ["users", "roles", "everyone"]}}
    r = SESSION.post(DISCORD_WEBHOOK_URL, json=payload, timeout=20)
    if r.status_code >= 400:
        raise RuntimeError(f"Discord webhook HTTP {r.status_code}: {r.text[:200]}")


def main():
    print("=" * 60)
    print("PROPS VALUE SCANNER")
    print(f"Min edge: {MIN_EDGE:.0%} | Min books: {MIN_BOOKS}")
    print("=" * 60)
    print()

    # Fetch Odds API props
    print("Fetching sportsbook props...")
    odds_props = fetch_all_props(min_books=MIN_BOOKS)
    print(f"  Found {len(odds_props)} props from Odds API")

    # Fetch Kalshi props
    print("Fetching Kalshi props...")
    kalshi_props = fetch_kalshi_props()
    print(f"  Found {len(kalshi_props)} props from Kalshi")
    print()

    # Match and find edges
    print("Matching props...")
    matches = match_props(odds_props, kalshi_props)
    print(f"  Found {len(matches)} matched props")

    # Filter and rank
    ranked = filter_and_rank(matches, min_edge=MIN_EDGE, top_n=TOP_N)
    print(f"  {len(ranked)} props with edge >= {MIN_EDGE:.0%}")
    print()

    # Print report
    print(format_edge_report(ranked))
    print()

    # Post to Discord if we have edges
    if ranked:
        discord_matches = ranked[:DISCORD_TOP_N]
        discord_text = format_discord_report(discord_matches)
        post_discord(discord_text)
        print(f"Posted {len(discord_matches)} edges to Discord.")
    else:
        print("No edges to post to Discord.")

    # Debug: show some unmatched if requested
    if DEBUG_MODE and matches:
        print("\n--- Sample matched props (all) ---")
        for m in matches[:10]:
            print(f"  {m['player_name']}: {m['stat_type']} {m['kalshi_line']}+ "
                  f"| edge={m['best_edge']:+.1%} ({m['best_side']})")


if __name__ == "__main__":
    main()
