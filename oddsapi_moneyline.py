import os
import time
import requests
from statistics import median
from typing import Any, Dict, List, Optional

ODDS_API_BASE = "https://api.the-odds-api.com/v4/sports"

# Major leagues + college
DEFAULT_SPORT_KEYS = [
    "americanfootball_nfl",
    "basketball_nba",
    "baseball_mlb",
    "icehockey_nhl",
    "americanfootball_ncaaf",
    "basketball_ncaab",
]

# ---- Odds math helpers ----

def american_to_prob(odds: int) -> float:
    """Convert American odds to implied probability with vig included."""
    if odds == 0:
        raise ValueError("American odds cannot be 0")
    if odds < 0:
        return (-odds) / ((-odds) + 100.0)
    return 100.0 / (odds + 100.0)


def prob_to_american(p: float) -> int:
    """Convert probability to fair American odds."""
    if p <= 0 or p >= 1:
        raise ValueError(f"Probability out of range: {p}")
    if p >= 0.5:
        return int(round(-100.0 * p / (1.0 - p)))
    return int(round(100.0 * (1.0 - p) / p))


def normalize_team_name(s: str) -> str:
    """Best-effort normalization for matching team names across feeds."""
    return (
        s.lower()
        .replace("&", "and")
        .replace(".", "")
        .replace(",", "")
        .replace("  ", " ")
        .strip()
    )

# ---- Odds API fetch ----

def _get_odds_api_key() -> str:
    # IMPORTANT: use repo secret THE_ODDS_API
    k = os.getenv("THE_ODDS_API")
    if not k:
        raise RuntimeError("Missing THE_ODDS_API env var (Odds API key).")
    return k


def fetch_h2h_events(
    sport_key: str,
    regions: str = "us",
    odds_format: str = "american",
    date_format: str = "iso",
    sleep_s: float = 0.25,
) -> List[Dict[str, Any]]:
    """
    Fetch moneyline (h2h) odds for a sport_key from The Odds API.
    Returns list of event dicts with bookmakers->markets->outcomes.
    """
    params = {
        "apiKey": _get_odds_api_key(),
        "regions": regions,
        "markets": "h2h",           # moneyline
        "oddsFormat": odds_format,  # american
        "dateFormat": date_format,
    }

    url = f"{ODDS_API_BASE}/{sport_key}/odds"
    r = requests.get(url, params=params, timeout=30)
    r.raise_for_status()

    # Light throttle so we don’t hammer API across multiple sports
    time.sleep(sleep_s)
    return r.json()

# ---- Consensus + range extraction ----

def _extract_book_prices_for_event(event: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    For a single event, extract per-book American odds for home/away.
    Returns None if insufficient data.
    Output includes:
      - home_team, away_team, commence_time, event_id
      - per_book: list of {book_key, book_title, home_odds, away_odds}
    """
    home = event.get("home_team")
    away = event.get("away_team")
    if not home or not away:
        return None

    per_book = []
    for book in event.get("bookmakers", []):
        book_key = book.get("key") or ""
        book_title = book.get("title") or book_key
        for market in book.get("markets", []):
            if market.get("key") != "h2h":
                continue
            outcomes = market.get("outcomes", [])
            if len(outcomes) < 2:
                continue

            prices = {o.get("name"): o.get("price") for o in outcomes}
            if home not in prices or away not in prices:
                continue

            try:
                home_odds = int(prices[home])
                away_odds = int(prices[away])
            except Exception:
                continue

            # Guard against bogus 0
            if home_odds == 0 or away_odds == 0:
                continue

            per_book.append(
                {
                    "book_key": book_key,
                    "book_title": book_title,
                    "home_odds": home_odds,
                    "away_odds": away_odds,
                }
            )
            break  # one h2h market per bookmaker is enough

    if len(per_book) < 3:
        # require at least 3 books for “consensus”
        return None

    return {
        "event_id": event.get("id"),
        "commence_time": event.get("commence_time"),
        "home_team": home,
        "away_team": away,
        "per_book": per_book,
    }


def _range_for_side(per_book: List[Dict[str, Any]], side: str) -> Dict[str, Any]:
    """
    Range for a side ('home' or 'away') in *no-vig probability* space.
    This matches the consensus calculation (remove vig per book first).
    """
    if side not in ("home", "away"):
        return {}

    vals = []
    for b in per_book:
        try:
            home_odds = int(b["home_odds"])
            away_odds = int(b["away_odds"])
        except Exception:
            continue

        try:
            p_home_raw = american_to_prob(home_odds)
            p_away_raw = american_to_prob(away_odds)
        except Exception:
            continue

        s = p_home_raw + p_away_raw
        if s <= 0:
            continue

        p_home = p_home_raw / s
        p_away = p_away_raw / s

        if side == "home":
            p_side = p_home
            odds_side = home_odds
        else:
            p_side = p_away
            odds_side = away_odds

        vals.append((float(p_side), int(odds_side), str(b.get("book_title", ""))))

    if not vals:
        return {}

    min_p, min_odds, min_book = min(vals, key=lambda t: t[0])
    max_p, max_odds, max_book = max(vals, key=lambda t: t[0])

    return {
        "min_prob": float(min_p),
        "min_odds": int(min_odds),
        "min_book": str(min_book),
        "max_prob": float(max_p),
        "max_odds": int(max_odds),
        "max_book": str(max_book),
    }


def compute_no_vig_consensus(per_book: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Compute no-vig consensus probabilities (median across books after removing vig per book).
    Output includes:
      - home_prob_fair, away_prob_fair
      - home_fair_american, away_fair_american
      - books_used
      - home_range, away_range (raw american odds ranges across books)
    """
    home_ps = []
    away_ps = []

    for b in per_book:
        try:
            p_home_raw = american_to_prob(int(b["home_odds"]))
            p_away_raw = american_to_prob(int(b["away_odds"]))
        except Exception:
            continue

        s = p_home_raw + p_away_raw
        if s <= 0:
            continue

        # remove vig for that book
        home_ps.append(p_home_raw / s)
        away_ps.append(p_away_raw / s)

    if len(home_ps) < 3:
        raise RuntimeError("Not enough valid books to compute consensus.")

    p_home = float(median(home_ps))
    p_away = 1.0 - p_home

    return {
        "home_prob_fair": p_home,
        "away_prob_fair": p_away,
        "home_fair_american": prob_to_american(p_home),
        "away_fair_american": prob_to_american(p_away),
        "books_used": len(home_ps),
        "home_range": _range_for_side(per_book, "home"),
        "away_range": _range_for_side(per_book, "away"),
    }


def build_moneyline_board(
    sport_keys: Optional[List[str]] = None,
    regions: str = "us",
) -> List[Dict[str, Any]]:
    """
    Returns a list of events with:
      - teams, commence_time, sport_key
      - consensus fair probs/odds
      - raw range across books for both sides
      - per_book list
    """
    sport_keys = sport_keys or DEFAULT_SPORT_KEYS
    board: List[Dict[str, Any]] = []

    for sk in sport_keys:
        events = fetch_h2h_events(sk, regions=regions)

        for ev in events:
            extracted = _extract_book_prices_for_event(ev)
            if not extracted:
                continue

            consensus = compute_no_vig_consensus(extracted["per_book"])
            board.append(
                {
                    "sport_key": sk,
                    "event_id": extracted["event_id"],
                    "commence_time": extracted["commence_time"],
                    "home_team": extracted["home_team"],
                    "away_team": extracted["away_team"],
                    "consensus": consensus,
                    "per_book": extracted["per_book"],
                    # Include normalized names to help matching downstream
                    "home_team_norm": normalize_team_name(extracted["home_team"]),
                    "away_team_norm": normalize_team_name(extracted["away_team"]),
                }
            )

    return board
