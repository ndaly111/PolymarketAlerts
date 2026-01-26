"""
Odds API player props fetcher.

Fetches player props (points, rebounds, assists, etc.) from The Odds API
and structures them for comparison with prediction markets like Kalshi.
"""

import os
import re
import time
from statistics import median
from typing import Any, Dict, List, Optional, Tuple

import requests

ODDS_API_BASE = "https://api.the-odds-api.com/v4/sports"

# Prop market keys supported by The Odds API
# See: https://the-odds-api.com/sports-odds-data/betting-markets.html
NBA_PROP_MARKETS = [
    "player_points",
    "player_rebounds",
    "player_assists",
    "player_threes",
    "player_blocks",
    "player_steals",
    "player_points_rebounds_assists",
    "player_points_rebounds",
    "player_points_assists",
    "player_rebounds_assists",
    "player_double_double",
]

NFL_PROP_MARKETS = [
    "player_pass_yds",
    "player_pass_tds",
    "player_pass_completions",
    "player_pass_attempts",
    "player_pass_interceptions",
    "player_rush_yds",
    "player_rush_attempts",
    "player_receptions",
    "player_reception_yds",
    "player_anytime_td",
    "player_first_td",
]

MLB_PROP_MARKETS = [
    "batter_home_runs",
    "batter_hits",
    "batter_total_bases",
    "batter_rbis",
    "batter_runs_scored",
    "batter_walks",
    "batter_strikeouts",
    "pitcher_strikeouts",
    "pitcher_outs",
]

NHL_PROP_MARKETS = [
    "player_points",
    "player_assists",
    "player_shots_on_goal",
    "player_blocked_shots",
]

# Map sport keys to their prop markets
SPORT_PROP_MARKETS = {
    "basketball_nba": NBA_PROP_MARKETS,
    "basketball_ncaab": NBA_PROP_MARKETS,
    "americanfootball_nfl": NFL_PROP_MARKETS,
    "americanfootball_ncaaf": NFL_PROP_MARKETS,
    "baseball_mlb": MLB_PROP_MARKETS,
    "icehockey_nhl": NHL_PROP_MARKETS,
}

# Default sport keys to fetch props for
DEFAULT_SPORT_KEYS = [
    "basketball_nba",
    "basketball_ncaab",
    "americanfootball_nfl",
    "baseball_mlb",
    "icehockey_nhl",
]


def _get_odds_api_key() -> str:
    k = os.getenv("THE_ODDS_API")
    if not k:
        raise RuntimeError("Missing THE_ODDS_API env var (Odds API key).")
    return k


def _env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None:
        return default
    text = str(value).strip()
    if not text:
        return default
    try:
        return int(text)
    except Exception:
        return default


def american_to_prob(odds: int) -> float:
    """Convert American odds to implied probability."""
    if odds == 0:
        raise ValueError("American odds cannot be 0")
    if odds < 0:
        return (-odds) / ((-odds) + 100.0)
    return 100.0 / (odds + 100.0)


def decimal_to_prob(odds: float) -> float:
    """Convert decimal odds to implied probability."""
    if odds <= 1.0:
        raise ValueError(f"Decimal odds must be > 1.0, got {odds}")
    return 1.0 / odds


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


def fetch_events(
    sport_key: str,
    sleep_s: float = 0.2,
) -> List[Dict[str, Any]]:
    """Fetch upcoming events for a sport."""
    params = {
        "apiKey": _get_odds_api_key(),
        "dateFormat": "iso",
    }
    url = f"{ODDS_API_BASE}/{sport_key}/events"
    r = requests.get(url, params=params, timeout=30)
    r.raise_for_status()
    time.sleep(sleep_s)
    return r.json()


def fetch_event_props(
    sport_key: str,
    event_id: str,
    markets: List[str],
    regions: str = "us",
    sleep_s: float = 1.0,  # Increased to avoid rate limits
) -> Dict[str, Any]:
    """Fetch player props for a specific event."""
    params = {
        "apiKey": _get_odds_api_key(),
        "regions": regions,
        "markets": ",".join(markets),
        "oddsFormat": "american",
        "dateFormat": "iso",
    }
    url = f"{ODDS_API_BASE}/{sport_key}/events/{event_id}/odds"
    r = requests.get(url, params=params, timeout=30)
    r.raise_for_status()
    time.sleep(sleep_s)
    return r.json()


def extract_props_from_event(event_data: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Extract structured props from event odds data.

    Returns list of dicts with:
        - player_name: str
        - player_name_norm: str
        - prop_type: str (e.g., 'player_points')
        - line: float (e.g., 23.5)
        - over_odds: int (American)
        - under_odds: int (American)
        - over_prob: float (implied, with vig)
        - under_prob: float (implied, with vig)
        - fair_over_prob: float (no-vig)
        - fair_under_prob: float (no-vig)
        - books: List[str]
        - home_team: str
        - away_team: str
        - commence_time: str
        - event_id: str
        - sport_key: str
    """
    props: Dict[Tuple[str, str, float], Dict[str, Any]] = {}

    home_team = event_data.get("home_team", "")
    away_team = event_data.get("away_team", "")
    commence_time = event_data.get("commence_time", "")
    event_id = event_data.get("id", "")
    sport_key = event_data.get("sport_key", "")

    for book in event_data.get("bookmakers", []):
        book_title = book.get("title", book.get("key", ""))

        for market in book.get("markets", []):
            prop_type = market.get("key", "")
            outcomes = market.get("outcomes", [])

            # Group outcomes by player + line
            player_lines: Dict[Tuple[str, float], Dict[str, int]] = {}
            for o in outcomes:
                player = o.get("description", "")
                name = o.get("name", "").lower()  # "over" or "under"
                point = o.get("point")
                price = o.get("price")

                if not player or point is None or price is None:
                    continue

                try:
                    line = float(point)
                    odds = int(price)  # American odds
                except (ValueError, TypeError):
                    continue

                key = (player, line)
                if key not in player_lines:
                    player_lines[key] = {}

                if "over" in name:
                    player_lines[key]["over"] = odds
                elif "under" in name:
                    player_lines[key]["under"] = odds

            # Build prop records
            for (player, line), odds_dict in player_lines.items():
                over_odds = odds_dict.get("over")
                under_odds = odds_dict.get("under")

                if over_odds is None or under_odds is None:
                    continue

                prop_key = (player, prop_type, line)

                if prop_key not in props:
                    # Calculate probabilities (American odds)
                    try:
                        over_prob = american_to_prob(over_odds)
                        under_prob = american_to_prob(under_odds)
                        total = over_prob + under_prob
                        fair_over = over_prob / total if total > 0 else 0.5
                        fair_under = under_prob / total if total > 0 else 0.5
                    except Exception:
                        continue

                    props[prop_key] = {
                        "player_name": player,
                        "player_name_norm": normalize_player_name(player),
                        "prop_type": prop_type,
                        "line": line,
                        "over_odds_list": [over_odds],
                        "under_odds_list": [under_odds],
                        "over_prob_list": [over_prob],
                        "under_prob_list": [under_prob],
                        "books": [book_title],
                        "home_team": home_team,
                        "away_team": away_team,
                        "commence_time": commence_time,
                        "event_id": event_id,
                        "sport_key": sport_key,
                    }
                else:
                    # Add to existing prop
                    try:
                        over_prob = american_to_prob(over_odds)
                        under_prob = american_to_prob(under_odds)
                    except Exception:
                        continue

                    props[prop_key]["over_odds_list"].append(over_odds)
                    props[prop_key]["under_odds_list"].append(under_odds)
                    props[prop_key]["over_prob_list"].append(over_prob)
                    props[prop_key]["under_prob_list"].append(under_prob)
                    props[prop_key]["books"].append(book_title)

    # Compute consensus fair probs
    result = []
    for prop in props.values():
        over_probs = prop.pop("over_prob_list")
        under_probs = prop.pop("under_prob_list")
        over_odds_list = prop.pop("over_odds_list")
        under_odds_list = prop.pop("under_odds_list")

        # No-vig fair probs per book, then median
        fair_overs = []
        fair_unders = []
        for op, up in zip(over_probs, under_probs):
            total = op + up
            if total > 0:
                fair_overs.append(op / total)
                fair_unders.append(up / total)

        if fair_overs:
            prop["fair_over_prob"] = float(median(fair_overs))
            prop["fair_under_prob"] = float(median(fair_unders))
        else:
            prop["fair_over_prob"] = 0.5
            prop["fair_under_prob"] = 0.5

        # Keep raw odds from first book for reference
        prop["over_odds"] = over_odds_list[0] if over_odds_list else None
        prop["under_odds"] = under_odds_list[0] if under_odds_list else None
        prop["books_used"] = len(prop["books"])

        result.append(prop)

    return result


def fetch_all_props(
    sport_keys: Optional[List[str]] = None,
    regions: str = "us",
    min_books: int = 1,
) -> List[Dict[str, Any]]:
    """
    Fetch all player props for given sports.

    Returns list of prop dicts (see extract_props_from_event).
    """
    sport_keys = sport_keys or DEFAULT_SPORT_KEYS
    all_props: List[Dict[str, Any]] = []
    errors: List[Dict[str, Any]] = []

    for sport_key in sport_keys:
        markets = SPORT_PROP_MARKETS.get(sport_key, [])
        if not markets:
            continue

        try:
            events = fetch_events(sport_key)
        except Exception as e:
            errors.append({"sport_key": sport_key, "stage": "events", "error": str(e)})
            continue

        for event in events:
            event_id = event.get("id")
            if not event_id:
                continue

            try:
                event_data = fetch_event_props(sport_key, event_id, markets, regions=regions)
                props = extract_props_from_event(event_data)

                # Filter by min books
                props = [p for p in props if p.get("books_used", 0) >= min_books]
                all_props.extend(props)

            except Exception as e:
                errors.append({
                    "sport_key": sport_key,
                    "event_id": event_id,
                    "stage": "props",
                    "error": str(e)[:200],
                })
                continue

    if errors and os.getenv("DEBUG_MODE"):
        print(f"Props fetch errors: {len(errors)}")
        for err in errors[:5]:
            print(f"  {err}")

    return all_props


def props_by_player(props: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    """Group props by normalized player name."""
    by_player: Dict[str, List[Dict[str, Any]]] = {}
    for p in props:
        name = p.get("player_name_norm", "")
        if name not in by_player:
            by_player[name] = []
        by_player[name].append(p)
    return by_player


if __name__ == "__main__":
    # Quick test
    import json

    print("Fetching NBA props...")
    props = fetch_all_props(sport_keys=["basketball_nba"])
    print(f"Found {len(props)} props")

    for p in props[:10]:
        print(f"{p['player_name']:25} | {p['prop_type']:20} | {p['line']:>6} | "
              f"O:{p['fair_over_prob']:.1%} U:{p['fair_under_prob']:.1%} | "
              f"books:{p['books_used']}")
