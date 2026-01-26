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

from oddsapi_props import fetch_all_props, normalize_player_name
from kalshi_props import fetch_kalshi_props, kalshi_props_by_player

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
MIN_BOOKS = env_int("MIN_BOOKS", 1)
TOP_N = env_int("TOP_N", 20)
DEBUG_MODE = env_int("DEBUG_MODE", 0)


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
            "best_edge": max(over_edge, under_edge),
            "best_side": "OVER" if over_edge > under_edge else "UNDER",
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
        f"{'Player':<20} {'Stat':<10} {'K-Line':>6} {'O-Line':>6} {'Side':<5} {'Edge':>6} "
        f"{'Kalshi':>6} {'Books':>6} {'#Bk':>3}",
        "-" * 80,
    ]

    for m in matches:
        player = m.get("player_name", "")[:19]
        stat = m.get("stat_type", "")[:9]
        k_line = m.get("kalshi_line", 0)
        o_line = m.get("odds_line", 0)
        side = m.get("best_side", "")[:4]
        edge = m.get("best_edge", 0)
        kalshi_prob = m.get("kalshi_over_prob", 0) if m.get("best_side") == "OVER" else m.get("kalshi_under_prob", 0)
        odds_prob = m.get("odds_over_prob", 0) if m.get("best_side") == "OVER" else m.get("odds_under_prob", 0)
        books = m.get("odds_books_used", 0)

        lines.append(
            f"{player:<20} {stat:<10} {k_line:>6.1f} {o_line:>6.1f} {side:<5} {edge:>+5.1%} "
            f"{kalshi_prob:>5.1%} {odds_prob:>5.1%} {books:>3}"
        )

    return "\n".join(lines)


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

    # Debug: show some unmatched if requested
    if DEBUG_MODE and matches:
        print("\n--- Sample matched props (all) ---")
        for m in matches[:10]:
            print(f"  {m['player_name']}: {m['stat_type']} {m['kalshi_line']}+ "
                  f"| edge={m['best_edge']:+.1%} ({m['best_side']})")


if __name__ == "__main__":
    main()
