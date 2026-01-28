import os
import re
import time
from statistics import median
from typing import Any, Dict, List, Optional, Tuple

import requests

ODDS_API_BASE = "https://api.the-odds-api.com/v4/sports"

# Major leagues + college + soccer
BASE_SPORT_KEYS = [
    "americanfootball_nfl",
    "basketball_nba",
    "baseball_mlb",
    "icehockey_nhl",
    "americanfootball_ncaaf",
    "basketball_ncaab",
    # Soccer
    "soccer_epl",
    "soccer_usa_mls",
    "soccer_spain_la_liga",
    "soccer_germany_bundesliga",
    "soccer_portugal_primeira_liga",
    "soccer_mexico_ligamx",
]

# All tennis tournaments (ATP + WTA)
TENNIS_SPORT_KEYS = [
    # ATP - Grand Slams
    "tennis_atp_aus_open_singles",
    "tennis_atp_french_open",
    "tennis_atp_wimbledon",
    "tennis_atp_us_open",
    # ATP - Masters 1000 & other majors
    "tennis_atp_indian_wells",
    "tennis_atp_miami_open",
    "tennis_atp_monte_carlo_masters",
    "tennis_atp_madrid_open",
    "tennis_atp_italian_open",
    "tennis_atp_canadian_open",
    "tennis_atp_cincinnati_open",
    "tennis_atp_shanghai_masters",
    "tennis_atp_paris_masters",
    "tennis_atp_dubai",
    "tennis_atp_qatar_open",
    "tennis_atp_china_open",
    # WTA - Grand Slams
    "tennis_wta_aus_open_singles",
    "tennis_wta_french_open",
    "tennis_wta_wimbledon",
    "tennis_wta_us_open",
    # WTA - Premier events
    "tennis_wta_indian_wells",
    "tennis_wta_miami_open",
    "tennis_wta_madrid_open",
    "tennis_wta_italian_open",
    "tennis_wta_canadian_open",
    "tennis_wta_cincinnati_open",
    "tennis_wta_dubai",
    "tennis_wta_qatar_open",
    "tennis_wta_china_open",
    "tennis_wta_wuhan_open",
]

# Environment toggle for tennis (default ON)
INCLUDE_TENNIS = os.getenv("INCLUDE_TENNIS", "1").strip().lower() in ("1", "true", "yes")

# Build DEFAULT_SPORT_KEYS based on config
DEFAULT_SPORT_KEYS = BASE_SPORT_KEYS + (TENNIS_SPORT_KEYS if INCLUDE_TENNIS else [])


def american_to_prob(odds: int) -> float:
    """Convert American odds to implied probability with vig included."""
    if odds == 0:
        raise ValueError("American odds cannot be 0")
    if odds < 0:
        return (-odds) / ((-odds) + 100.0)
    return 100.0 / (odds + 100.0)


def normalize_team_name(s: Optional[str]) -> str:
    if not s:
        return ""
    return (
        s.lower()
        .replace("&", "and")
        .replace(".", "")
        .replace(",", "")
        .replace("  ", " ")
        .strip()
    )


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


def fetch_events(
    sport_key: str,
    markets: str,
    regions: str = "us",
    odds_format: str = "american",
    date_format: str = "iso",
    sleep_s: float = 0.20,
) -> List[Dict[str, Any]]:
    """Fetch odds events for a sport_key with specified markets (comma-separated)."""
    params = {
        "apiKey": _get_odds_api_key(),
        "regions": regions,
        "markets": markets,
        "oddsFormat": odds_format,
        "dateFormat": date_format,
    }

    url = f"{ODDS_API_BASE}/{sport_key}/odds"
    r = requests.get(url, params=params, timeout=30)
    r.raise_for_status()

    # light throttle to avoid hammering API across multiple sports
    time.sleep(sleep_s)
    data = r.json()
    # Defensive: downstream matching expects sport_key on each event.
    if isinstance(data, list):
        for e in data:
            if isinstance(e, dict) and "sport_key" not in e:
                e["sport_key"] = sport_key
    return data


def _extract_market(book: Dict[str, Any], market_key: str) -> Optional[Dict[str, Any]]:
    for m in book.get("markets", []) or []:
        if m.get("key") == market_key:
            return m
    return None


def _extract_book_h2h(event: Dict[str, Any]) -> List[Tuple[str, int, int]]:
    """Return list of (book_title, home_odds, away_odds) for h2h."""
    home = event.get("home_team")
    away = event.get("away_team")
    out: List[Tuple[str, int, int]] = []

    for book in event.get("bookmakers", []) or []:
        m = _extract_market(book, "h2h")
        if not m:
            continue
        # Be tolerant of naming differences (esp. DraftKings abbreviations) by matching normalized names.
        prices: Dict[str, Any] = {}
        for o in (m.get("outcomes") or []):
            n = o.get("name")
            if n is None:
                continue
            prices[normalize_team_name(str(n))] = o.get("price")
        hn = normalize_team_name(str(home))
        an = normalize_team_name(str(away))
        if hn not in prices or an not in prices:
            continue
        try:
            ho = int(prices[hn])
            ao = int(prices[an])
        except Exception:
            continue
        if ho == 0 or ao == 0:
            continue
        out.append((str(book.get("title") or book.get("key") or ""), ho, ao))

    return out


def fair_prob_h2h(event: Dict[str, Any], team_name: str, min_books: int = 3) -> Optional[Dict[str, Any]]:
    """No-vig consensus probability for team_name winning (moneyline) for this event."""
    home = event.get("home_team")
    away = event.get("away_team")
    if not home or not away:
        return None

    books = _extract_book_h2h(event)
    if len(books) < max(1, int(min_books)):
        return None

    target_norm = normalize_team_name(team_name)
    home_norm = normalize_team_name(home)
    away_norm = normalize_team_name(away)

    if target_norm not in (home_norm, away_norm):
        return None

    probs: List[float] = []
    for _, ho, ao in books:
        try:
            p_home_raw = american_to_prob(ho)
            p_away_raw = american_to_prob(ao)
        except Exception:
            continue
        s = p_home_raw + p_away_raw
        if s <= 0:
            continue
        p_home = p_home_raw / s
        p_away = p_away_raw / s
        probs.append(p_home if target_norm == home_norm else p_away)

    if len(probs) < max(1, int(min_books)):
        return None

    return {
        "fair_prob": float(median(probs)),
        "books_used": len(probs),
        "home_team": home,
        "away_team": away,
    }


def _extract_book_spread_at_point(event: Dict[str, Any], point: float) -> List[Tuple[str, int, int]]:
    """Return list of (book_title, home_odds, away_odds) for spreads at exact point."""
    home = event.get("home_team")
    away = event.get("away_team")
    out: List[Tuple[str, int, int]] = []

    for book in event.get("bookmakers", []) or []:
        title = str(book.get("title") or book.get("key") or "")
        found_for_book = False

        for m in (book.get("markets") or []):
            if m.get("key") != "spreads":
                continue
            outcomes = m.get("outcomes") or []
            if len(outcomes) < 2:
                continue

            # We expect both teams present; points can be home/away with opposite sign.
            prices: Dict[str, Tuple[float, int]] = {}
            for o in outcomes:
                name = o.get("name")
                p = o.get("point")
                price = o.get("price")
                if name is None or p is None or price is None:
                    continue
                try:
                    p_f = float(p)
                    pr_i = int(price)
                except Exception:
                    continue
                prices[normalize_team_name(str(name))] = (p_f, pr_i)

            hn = normalize_team_name(str(home))
            an = normalize_team_name(str(away))
            if hn not in prices or an not in prices:
                continue

            home_point, home_odds = prices[hn]
            away_point, away_odds = prices[an]

            # exact line match: require the specific point for the relevant side
            if float(home_point) != float(point) and float(away_point) != float(point):
                continue

            # sanity: they should be opposite signs (allow 0)
            if float(home_point) != -float(away_point) and not (float(home_point) == 0 and float(away_point) == 0):
                continue

            if home_odds == 0 or away_odds == 0:
                continue

            out.append((title, home_odds, away_odds))
            found_for_book = True
            break

        if found_for_book:
            continue

    return out


def fair_prob_spread(event: Dict[str, Any], team_name: str, point: float, min_books: int = 3) -> Optional[Dict[str, Any]]:
    """No-vig consensus probability that team_name covers spread at exact point."""
    home = event.get("home_team")
    away = event.get("away_team")
    if not home or not away:
        return None

    books = _extract_book_spread_at_point(event, point)
    if len(books) < max(1, int(min_books)):
        return None

    target_norm = normalize_team_name(team_name)
    home_norm = normalize_team_name(home)
    away_norm = normalize_team_name(away)
    if target_norm not in (home_norm, away_norm):
        return None

    probs: List[float] = []
    for _, ho, ao in books:
        try:
            p_home_raw = american_to_prob(ho)
            p_away_raw = american_to_prob(ao)
        except Exception:
            continue
        s = p_home_raw + p_away_raw
        if s <= 0:
            continue
        p_home = p_home_raw / s
        p_away = p_away_raw / s
        probs.append(p_home if target_norm == home_norm else p_away)

    if len(probs) < max(1, int(min_books)):
        return None

    return {
        "fair_prob": float(median(probs)),
        "books_used": len(probs),
        "home_team": home,
        "away_team": away,
    }


def _extract_book_total_at_point(event: Dict[str, Any], point: float) -> List[Tuple[str, int, int]]:
    """Return list of (book_title, over_odds, under_odds) for totals at exact point."""
    out: List[Tuple[str, int, int]] = []

    for book in event.get("bookmakers", []) or []:
        title = str(book.get("title") or book.get("key") or "")
        found_for_book = False

        for m in (book.get("markets") or []):
            if m.get("key") != "totals":
                continue
            outcomes = m.get("outcomes") or []
            if len(outcomes) < 2:
                continue

            over = None
            under = None
            for o in outcomes:
                name = str(o.get("name") or "").lower()
                p = o.get("point")
                price = o.get("price")
                if p is None or price is None:
                    continue
                try:
                    p_f = float(p)
                    pr_i = int(price)
                except Exception:
                    continue
                if p_f != float(point):
                    continue
                if name.startswith("over"):
                    over = pr_i
                elif name.startswith("under"):
                    under = pr_i

            if over is None or under is None:
                continue
            if over == 0 or under == 0:
                continue

            out.append((title, int(over), int(under)))
            found_for_book = True
            break

        if found_for_book:
            continue

    return out


def fair_prob_total(event: Dict[str, Any], side: str, point: float, min_books: int = 3) -> Optional[Dict[str, Any]]:
    """No-vig consensus probability for Over/Under at exact total point."""
    side_l = str(side).strip().lower()
    if side_l not in ("over", "under"):
        return None

    books = _extract_book_total_at_point(event, point)
    if len(books) < max(1, int(min_books)):
        return None

    probs: List[float] = []
    for _, oo, uo in books:
        try:
            p_over_raw = american_to_prob(oo)
            p_under_raw = american_to_prob(uo)
        except Exception:
            continue
        s = p_over_raw + p_under_raw
        if s <= 0:
            continue
        p_over = p_over_raw / s
        p_under = p_under_raw / s
        probs.append(p_over if side_l == "over" else p_under)

    if len(probs) < max(1, int(min_books)):
        return None

    return {
        "fair_prob": float(median(probs)),
        "books_used": len(probs),
    }


def build_event_index(events: List[Dict[str, Any]]) -> Dict[Tuple[str, str], Dict[str, Any]]:
    """Index odds events by (home_norm, away_norm) and (away_norm, home_norm)."""
    idx: Dict[Tuple[str, str], Dict[str, Any]] = {}
    for e in events:
        home = e.get("home_team")
        away = e.get("away_team")
        if not home or not away:
            continue
        h = normalize_team_name(home)
        a = normalize_team_name(away)
        idx[(h, a)] = e
        idx[(a, h)] = e
    return idx


def _redact_oddsapi_key(text: str) -> str:
    # requests exceptions can include the full URL (including apiKey=...).
    return re.sub(r"(apiKey=)[^&\s]+", r"\1***", str(text))


def fetch_all_sports_events(
    sport_keys: List[str],
    markets: str = "h2h,spreads,totals",
    regions: str = "us",
) -> List[Dict[str, Any]]:
    all_events: List[Dict[str, Any]] = []
    errors: List[Dict[str, Any]] = []

    debug_enabled = (
        os.getenv("ODDSAPI_DEBUG")
        or os.getenv("DEBUG_MODE")
        or os.getenv("MASTER_DEBUG")
    )
    for sk in sport_keys:
        try:
            all_events.extend(fetch_events(sk, markets=markets, regions=regions))
        except Exception as e:
            # Keep it short (Actions logs), but avoid leaking api keys in exception strings.
            err: Dict[str, Any] = {"sport_key": sk, "error_type": type(e).__name__}
            msg = str(e)
            if msg:
                err["message"] = _redact_oddsapi_key(msg)[:200]

            # requests will often wrap HTTP errors with response details
            if debug_enabled:
                try:
                    if hasattr(e, "response") and getattr(e, "response") is not None:
                        r = getattr(e, "response")
                        err["status"] = getattr(r, "status_code", None)
                        err["body_head"] = str(getattr(r, "text", ""))[:200]
                except Exception:
                    pass

            errors.append(err)
            # keep going; one sport failing shouldn't kill the run
            continue

    if not all_events and errors:
        try:
            print(f"Odds API: 0 events; sample error: {errors[0]}")
        except Exception:
            pass
    return all_events


def env_sport_keys() -> List[str]:
    raw = os.getenv("SPORT_KEYS")
    if raw and raw.strip():
        return [s.strip() for s in raw.split(",") if s.strip()]
    return list(DEFAULT_SPORT_KEYS)


def env_min_books() -> int:
    return _env_int("MIN_BOOKS", 1)
