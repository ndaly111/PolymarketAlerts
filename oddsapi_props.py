"""
Odds API player props fetcher.

Fetches player props (points, rebounds, assists, etc.) from The Odds API
and structures them for comparison with prediction markets like Kalshi.

Includes caching to minimize API calls.
"""

import json
import os
import re
import time
from pathlib import Path
from statistics import median
from typing import Any, Dict, List, Optional, Tuple

import requests

# Cache configuration
CACHE_DIR = Path(os.getenv("ODDS_CACHE_DIR", "/tmp/odds_api_cache"))
CACHE_TTL_SECONDS = int(os.getenv("ODDS_CACHE_TTL", "14400"))  # 4 hours default (refresh at scheduled times only)

# Scheduled refresh times (EST hours) - only fetch fresh odds at these times
# 6am, 8am, 12pm, 4pm, 6pm EST
SCHEDULED_REFRESH_HOURS_EST = [6, 8, 12, 16, 18]

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
    "americanfootball_nfl",
    "baseball_mlb",
    "icehockey_nhl",
]

# Tennis sport keys for moneylines (H2H)
# Full list from Odds API - includes all major ATP and WTA tournaments
TENNIS_SPORT_KEYS = [
    # Grand Slams
    "tennis_atp_australian_open",
    "tennis_wta_australian_open",
    "tennis_atp_french_open",
    "tennis_wta_french_open",
    "tennis_atp_wimbledon",
    "tennis_wta_wimbledon",
    "tennis_atp_us_open",
    "tennis_wta_us_open",
    # ATP Tour
    "tennis_atp_adelaide",
    "tennis_atp_auckland",
    "tennis_atp_barcelona",
    "tennis_atp_beijing",
    "tennis_atp_brisbane",
    "tennis_atp_buenos_aires",
    "tennis_atp_canadian_open",
    "tennis_atp_chile",
    "tennis_atp_china_open",
    "tennis_atp_cincinnati_open",
    "tennis_atp_davis_cup",
    "tennis_atp_doha",
    "tennis_atp_dubai",
    "tennis_atp_eastbourne",
    "tennis_atp_estoril",
    "tennis_atp_geneva",
    "tennis_atp_halle",
    "tennis_atp_hamburg",
    "tennis_atp_houston",
    "tennis_atp_indian_wells",
    "tennis_atp_lyon",
    "tennis_atp_madrid_open",
    "tennis_atp_marseille",
    "tennis_atp_miami_open",
    "tennis_atp_monte_carlo",
    "tennis_atp_munich",
    "tennis_atp_newport",
    "tennis_atp_paris",
    "tennis_atp_queens_club",
    "tennis_atp_rio",
    "tennis_atp_rome",
    "tennis_atp_rotterdam",
    "tennis_atp_shanghai",
    "tennis_atp_stockholm",
    "tennis_atp_stuttgart",
    "tennis_atp_tokyo",
    "tennis_atp_umag",
    "tennis_atp_vienna",
    "tennis_atp_washington",
    "tennis_atp_winston_salem",
    # WTA Tour
    "tennis_wta_adelaide",
    "tennis_wta_auckland",
    "tennis_wta_beijing",
    "tennis_wta_birmingham",
    "tennis_wta_brisbane",
    "tennis_wta_canadian_open",
    "tennis_wta_charleston",
    "tennis_wta_china_open",
    "tennis_wta_cincinnati",
    "tennis_wta_doha",
    "tennis_wta_dubai",
    "tennis_wta_eastbourne",
    "tennis_wta_guadalajara",
    "tennis_wta_hobart",
    "tennis_wta_indian_wells",
    "tennis_wta_madrid_open",
    "tennis_wta_miami_open",
    "tennis_wta_monterrey",
    "tennis_wta_nottingham",
    "tennis_wta_pan_pacific",
    "tennis_wta_rome",
    "tennis_wta_san_diego",
    "tennis_wta_strasbourg",
    "tennis_wta_stuttgart",
    "tennis_wta_tokyo",
    "tennis_wta_wuhan",
]


def _get_odds_api_key() -> str:
    k = os.getenv("THE_ODDS_API")
    if not k:
        raise RuntimeError("Missing THE_ODDS_API env var (Odds API key).")
    return k


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
            return None  # Expired
        return data.get("_data")
    except Exception:
        return None


def _read_cache_any_age(cache_key: str) -> Optional[Dict[str, Any]]:
    """Read from cache regardless of age (for non-scheduled refresh times)."""
    cache_path = _get_cache_path(cache_key)
    if not cache_path.exists():
        return None

    try:
        data = json.loads(cache_path.read_text())
        return data.get("_data")
    except Exception:
        return None


def _write_cache(cache_key: str, data: Any) -> None:
    """Write data to cache."""
    cache_path = _get_cache_path(cache_key)
    try:
        cache_path.write_text(json.dumps({
            "_cached_at": time.time(),
            "_data": data,
        }))
    except Exception:
        pass  # Cache write failure is not critical


def _update_cache_with_prop(cache_key: str, updated_prop: Dict[str, Any]) -> None:
    """
    Update a single prop in the cache (used after verification fetch).
    This avoids wasting API calls by caching fresh data.
    """
    cache_path = _get_cache_path(cache_key)
    if not cache_path.exists():
        return

    try:
        data = json.loads(cache_path.read_text())
        props = data.get("_data", [])

        # Find and update the matching prop
        player_norm = updated_prop.get("player_name_norm", "")
        prop_type = updated_prop.get("prop_type", "")
        line = updated_prop.get("line", 0)

        for i, prop in enumerate(props):
            if (prop.get("player_name_norm") == player_norm and
                prop.get("prop_type") == prop_type and
                abs(prop.get("line", 0) - line) <= 0.5):
                props[i] = updated_prop
                break
        else:
            # Prop not found, add it
            props.append(updated_prop)

        data["_data"] = props
        cache_path.write_text(json.dumps(data))
    except Exception:
        pass


def is_scheduled_refresh_time() -> bool:
    """Check if current time is a scheduled refresh time (within 15 min window)."""
    from datetime import datetime
    from zoneinfo import ZoneInfo

    now_est = datetime.now(ZoneInfo("America/New_York"))
    current_hour = now_est.hour
    current_minute = now_est.minute

    # Check if we're within 15 minutes of a scheduled hour
    for scheduled_hour in SCHEDULED_REFRESH_HOURS_EST:
        if current_hour == scheduled_hour and current_minute < 15:
            return True

    return False


def force_cache_refresh() -> None:
    """Clear cache to force refresh on next fetch."""
    try:
        for cache_file in CACHE_DIR.glob("*.json"):
            cache_file.unlink()
    except Exception:
        pass


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
    use_cache: bool = True,
    force_refresh: bool = False,
) -> List[Dict[str, Any]]:
    """
    Fetch all player props for given sports.

    Uses caching to minimize API calls. Only refreshes at scheduled times
    (6am, 8am, 12pm, 4pm, 6pm EST) unless force_refresh=True.

    Returns list of prop dicts (see extract_props_from_event).
    """
    sport_keys = sport_keys or DEFAULT_SPORT_KEYS
    all_props: List[Dict[str, Any]] = []
    errors: List[Dict[str, Any]] = []
    api_calls = 0

    # Check if this is a scheduled refresh time
    is_refresh_time = is_scheduled_refresh_time()
    if is_refresh_time:
        print("  [scheduled refresh] Fetching fresh odds (scheduled time)")
    elif force_refresh:
        print("  [force refresh] Fetching fresh odds")

    for sport_key in sport_keys:
        markets = SPORT_PROP_MARKETS.get(sport_key, [])
        if not markets:
            continue

        # Check cache for this sport
        cache_key = f"props_{sport_key}_{regions}"
        if use_cache and not force_refresh:
            # At non-scheduled times, use cache even if "expired"
            cached = _read_cache_any_age(cache_key) if not is_refresh_time else _read_cache(cache_key)
            if cached is not None:
                # Filter by min_books and add to results
                props = [p for p in cached if p.get("books_used", 0) >= min_books]
                all_props.extend(props)
                print(f"  [cache] {sport_key}: {len(props)} props from cache")
                continue

        # Not in cache or scheduled refresh, fetch from API
        sport_props: List[Dict[str, Any]] = []

        try:
            events = fetch_events(sport_key)
            api_calls += 1
        except Exception as e:
            errors.append({"sport_key": sport_key, "stage": "events", "error": str(e)})
            continue

        for event in events:
            event_id = event.get("id")
            if not event_id:
                continue

            try:
                event_data = fetch_event_props(sport_key, event_id, markets, regions=regions)
                api_calls += 1
                props = extract_props_from_event(event_data)
                sport_props.extend(props)

            except Exception as e:
                errors.append({
                    "sport_key": sport_key,
                    "event_id": event_id,
                    "stage": "props",
                    "error": str(e)[:200],
                })
                continue

        # Cache all props for this sport (before min_books filter)
        if use_cache and sport_props:
            _write_cache(cache_key, sport_props)

        # Filter by min_books and add to results
        filtered = [p for p in sport_props if p.get("books_used", 0) >= min_books]
        all_props.extend(filtered)
        print(f"  [api] {sport_key}: {len(filtered)} props ({api_calls} API calls so far)")

    if errors and os.getenv("DEBUG_MODE"):
        print(f"Props fetch errors: {len(errors)}")
        for err in errors[:5]:
            print(f"  {err}")

    print(f"  Total: {len(all_props)} props, {api_calls} API calls")
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


def fetch_tennis_moneylines(
    regions: str = "us",
    min_books: int = 2,
    use_cache: bool = True,
) -> List[Dict[str, Any]]:
    """
    Fetch tennis moneylines (H2H) from Odds API.

    Returns list of match dicts with fair probabilities for each player.
    """
    all_matches: List[Dict[str, Any]] = []
    api_calls = 0

    # Check if scheduled refresh time
    is_refresh_time = is_scheduled_refresh_time()

    for sport_key in TENNIS_SPORT_KEYS:
        cache_key = f"tennis_{sport_key}_{regions}"

        # Check cache
        if use_cache:
            cached = _read_cache_any_age(cache_key) if not is_refresh_time else _read_cache(cache_key)
            if cached is not None:
                matches = [m for m in cached if m.get("books_used", 0) >= min_books]
                all_matches.extend(matches)
                continue

        try:
            # Fetch events with H2H odds
            params = {
                "apiKey": _get_odds_api_key(),
                "regions": regions,
                "markets": "h2h",
                "oddsFormat": "american",
                "dateFormat": "iso",
            }
            url = f"{ODDS_API_BASE}/{sport_key}/odds"
            r = requests.get(url, params=params, timeout=30)
            api_calls += 1

            if r.status_code == 404:
                # No events for this tournament
                continue
            r.raise_for_status()

            events = r.json()
            sport_matches: List[Dict[str, Any]] = []

            for event in events:
                home = event.get("home_team", "")
                away = event.get("away_team", "")
                commence_time = event.get("commence_time", "")
                event_id = event.get("id", "")

                # Collect odds from all bookmakers
                home_odds_list = []
                away_odds_list = []
                books = []

                for book in event.get("bookmakers", []):
                    book_title = book.get("title", "")
                    for market in book.get("markets", []):
                        if market.get("key") != "h2h":
                            continue
                        for outcome in market.get("outcomes", []):
                            name = outcome.get("name", "")
                            price = outcome.get("price")
                            if price is None:
                                continue
                            if name == home:
                                home_odds_list.append(price)
                            elif name == away:
                                away_odds_list.append(price)
                    if home_odds_list and away_odds_list:
                        books.append(book_title)

                if len(books) < min_books:
                    continue

                # Calculate fair probabilities (no-vig)
                fair_home_probs = []
                fair_away_probs = []
                for ho, ao in zip(home_odds_list, away_odds_list):
                    try:
                        hp = american_to_prob(ho)
                        ap = american_to_prob(ao)
                        total = hp + ap
                        if total > 0:
                            fair_home_probs.append(hp / total)
                            fair_away_probs.append(ap / total)
                    except Exception:
                        continue

                if not fair_home_probs:
                    continue

                fair_home = float(median(fair_home_probs))
                fair_away = float(median(fair_away_probs))

                match = {
                    "sport_key": sport_key,
                    "event_id": event_id,
                    "home_team": home,
                    "away_team": away,
                    "home_team_norm": normalize_player_name(home),
                    "away_team_norm": normalize_player_name(away),
                    "commence_time": commence_time,
                    "fair_home_prob": fair_home,
                    "fair_away_prob": fair_away,
                    "home_odds": home_odds_list[0] if home_odds_list else None,
                    "away_odds": away_odds_list[0] if away_odds_list else None,
                    "books_used": len(books),
                    "books": books,
                    "market_type": "moneyline",
                }
                sport_matches.append(match)

            # Cache results
            if use_cache and sport_matches:
                _write_cache(cache_key, sport_matches)

            filtered = [m for m in sport_matches if m.get("books_used", 0) >= min_books]
            all_matches.extend(filtered)

            time.sleep(0.5)  # Rate limit

        except Exception as e:
            if os.getenv("DEBUG_MODE"):
                print(f"Tennis fetch error for {sport_key}: {e}")
            continue

    print(f"  Tennis: {len(all_matches)} matches, {api_calls} API calls")
    return all_matches


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
