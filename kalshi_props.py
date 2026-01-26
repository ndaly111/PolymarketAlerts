"""
Kalshi player props fetcher.

Fetches player prop markets from Kalshi and structures them
for comparison with sportsbook odds.
"""

import os
import re
import time
from typing import Any, Dict, List, Optional, Tuple

import requests

KALSHI_API_BASE = "https://api.elections.kalshi.com/trade-api/v2"

# Kalshi prop series tickers and their corresponding stat types
KALSHI_PROP_SERIES = {
    # NBA
    "KXNBAREB": "rebounds",
    "KXNBAAST": "assists",
    "KXNBAPTS": "points",
    "KXNBASTL": "steals",
    "KXNBABLK": "blocks",
    "KXNBA3PM": "threes",
    # College Basketball (NCAAB)
    "KXNCAABREB": "rebounds",
    "KXNCAABAST": "assists",
    "KXNCAABPTS": "points",
    "KXNCAAB3PM": "threes",
    # NFL
    "KXNFLRSHYDS": "rushing_yards",
    "KXNFLPASSYDS": "passing_yards",
    "KXNFLRECYDS": "receiving_yards",
    "KXNFLANYTD": "anytime_td",
    "KXNFL1TD": "first_td",
    "KXNFL2TD": "two_plus_td",
    "KXNFLREC": "receptions",
    # NHL
    "KXNHLPTS": "points",
    "KXNHLAST": "assists",
    "KXNHLGOALS": "goals",
    "KXNHLSOG": "shots_on_goal",
}

# Map Kalshi stat types to Odds API prop types
KALSHI_TO_ODDSAPI = {
    "rebounds": "player_rebounds",
    "assists": "player_assists",
    "points": "player_points",
    "steals": "player_steals",
    "blocks": "player_blocks",
    "threes": "player_threes",
    "rushing_yards": "player_rush_yds",
    "passing_yards": "player_pass_yds",
    "receiving_yards": "player_reception_yds",
    "anytime_td": "player_anytime_td",
    "first_td": "player_first_td",
    "receptions": "player_receptions",
    "shots_on_goal": "player_shots_on_goal",
}


def normalize_player_name(s: str) -> str:
    """Normalize player name for matching."""
    if not s:
        return ""
    return (
        s.lower()
        .replace(".", "")
        .replace(",", "")
        .replace("'", "")
        .replace("-", " ")
        .replace("  ", " ")
        .strip()
    )


def extract_player_from_title(title: str) -> Optional[str]:
    """Extract player name from Kalshi title like 'LaMelo Ball: 8+ rebounds'."""
    if not title:
        return None
    # Pattern: "Player Name: X+ stat"
    match = re.match(r"^([^:]+):\s*\d+", title)
    if match:
        return match.group(1).strip()
    return None


def extract_line_from_title(title: str) -> Optional[float]:
    """Extract the line from title like 'LaMelo Ball: 8+ rebounds' -> 8.0."""
    if not title:
        return None
    match = re.search(r":\s*(\d+(?:\.\d+)?)\+", title)
    if match:
        return float(match.group(1))
    return None


def fetch_kalshi_props(
    series_tickers: Optional[List[str]] = None,
    status: str = "open",
) -> List[Dict[str, Any]]:
    """
    Fetch player prop markets from Kalshi.

    Returns list of structured prop dicts.
    """
    series_tickers = series_tickers or list(KALSHI_PROP_SERIES.keys())
    all_props: List[Dict[str, Any]] = []

    for series_ticker in series_tickers:
        stat_type = KALSHI_PROP_SERIES.get(series_ticker, "unknown")
        oddsapi_type = KALSHI_TO_ODDSAPI.get(stat_type, "")

        try:
            url = f"{KALSHI_API_BASE}/markets"
            params = {
                "series_ticker": series_ticker,
                "status": status,
                "limit": 200,
            }
            r = requests.get(url, params=params, timeout=30)
            r.raise_for_status()
            data = r.json()
            markets = data.get("markets", [])

            for m in markets:
                title = m.get("title", "")
                player_name = extract_player_from_title(title)
                line = extract_line_from_title(title)

                if not player_name or line is None:
                    continue

                # Get prices (in cents, 0-100)
                yes_bid = m.get("yes_bid", 0)
                yes_ask = m.get("yes_ask", 100)
                no_bid = m.get("no_bid", 0)
                no_ask = m.get("no_ask", 100)

                # Convert to probabilities
                # Mid price is best estimate
                yes_mid = (yes_bid + yes_ask) / 2 / 100 if yes_bid and yes_ask else None
                no_mid = (no_bid + no_ask) / 2 / 100 if no_bid and no_ask else None

                # Use yes_ask as conservative "buy yes" price
                yes_prob = yes_ask / 100 if yes_ask else None

                prop = {
                    "player_name": player_name,
                    "player_name_norm": normalize_player_name(player_name),
                    "stat_type": stat_type,
                    "oddsapi_type": oddsapi_type,
                    "line": line,
                    "yes_bid": yes_bid,
                    "yes_ask": yes_ask,
                    "no_bid": no_bid,
                    "no_ask": no_ask,
                    "yes_mid": yes_mid,
                    "no_mid": no_mid,
                    "kalshi_prob": yes_mid,  # Probability of hitting the over
                    "ticker": m.get("ticker", ""),
                    "series_ticker": series_ticker,
                    "event_ticker": m.get("event_ticker", ""),
                    "title": title,
                    "volume": m.get("volume", 0),
                    "liquidity": m.get("liquidity", 0),
                    "close_time": m.get("close_time", ""),
                    "expiration_time": m.get("expected_expiration_time", m.get("expiration_time", "")),
                }
                all_props.append(prop)

            time.sleep(0.2)  # Rate limit

        except Exception as e:
            if os.getenv("DEBUG_MODE"):
                print(f"Error fetching {series_ticker}: {e}")
            continue

    return all_props


def kalshi_props_by_player(props: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    """Group Kalshi props by normalized player name."""
    by_player: Dict[str, List[Dict[str, Any]]] = {}
    for p in props:
        name = p.get("player_name_norm", "")
        if name not in by_player:
            by_player[name] = []
        by_player[name].append(p)
    return by_player


if __name__ == "__main__":
    print("Fetching Kalshi NBA props...")
    # Just NBA for testing
    nba_series = ["KXNBAREB", "KXNBAAST", "KXNBAPTS"]
    props = fetch_kalshi_props(series_tickers=nba_series)
    print(f"Found {len(props)} Kalshi props\n")

    for p in props[:15]:
        prob = p.get('kalshi_prob')
        prob_str = f"{prob:.1%}" if prob else "N/A"
        print(f"{p['player_name']:25} | {p['stat_type']:10} | {p['line']:>5}+ | "
              f"prob:{prob_str:>6} | bid:{p['yes_bid']:>2} ask:{p['yes_ask']:>2}")
