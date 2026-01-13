#!/usr/bin/env python3
"""
Polymarket sports moneyline value scanner.

- Pulls active SPORTS moneyline markets from the Polymarket Gamma API.
- Converts Polymarket implied probabilities -> American moneyline odds.
- Loads sportsbook moneylines from The Odds API (moneyline consensus + range).
- Matches games and ranks the biggest mispricings under configured filters.
- Posts ONE Discord message (top-N) via webhook.

This script does NOT place trades.
"""

from __future__ import annotations

import csv
import json
import os
import re
from zoneinfo import ZoneInfo
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from oddsapi_moneyline import build_moneyline_board

GAMMA_API_BASE = "https://gamma-api.polymarket.com"

# Discord
DISCORD_WEBHOOK_URL = os.getenv("DISCORD_WEBHOOK_URL", "").strip()
DISCORD_MENTION = os.getenv("DISCORD_MENTION", "").strip()

# Output
OUTPUT_DIR = Path(os.getenv("OUTPUT_DIR", "reports"))

# Filters / ranking
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


TOP_N = env_int("TOP_N", 20)
DISCORD_TOP_N = env_int("DISCORD_TOP_N", 8)
MIN_EDGE = env_float("MIN_EDGE", 0.03)  # prob gap threshold (e.g., 0.03 = 3%)
GAME_BETS_TAG_ID = env_int("GAME_BETS_TAG_ID", 100639)  # Polymarket "game bets" tag
SPORTSBOOK_MAX_UNDERDOG = env_int("SPORTSBOOK_MAX_UNDERDOG", 200)
POLYMARKET_MAX_FAVORITE = env_int("POLYMARKET_MAX_FAVORITE", -300)
TIME_WINDOW_HOURS = env_int("TIME_WINDOW_HOURS", 36)
INCLUDE_DEBUG_SUMMARY = env_int("INCLUDE_DEBUG_SUMMARY", 1)  # 1=yes, 0=no
DEBUG_SUMMARY_LINES = env_int("DEBUG_SUMMARY_LINES", 8)
DRY_RUN = env_int("DRY_RUN", 0)
REQUIRE_OUTSIDE_RANGE = env_int("REQUIRE_OUTSIDE_RANGE", 1)

# Moneyline types override (comma-separated). If empty, we autodetect using /sports/market-types.
MONEYLINE_TYPES = os.getenv("MONEYLINE_TYPES", "").strip()

# Odds API sport keys override (comma-separated). If empty, use defaults.
ODDS_SPORT_KEYS = [s.strip() for s in os.getenv("ODDS_SPORT_KEYS", "").split(",") if s.strip()]
ODDS_REGIONS = (os.getenv("ODDS_REGIONS", "us") or "us").strip() or "us"

# Polymarket fetch behavior
POLY_FALLBACK = env_int("POLY_FALLBACK", 1)  # 1=yes (retry with looser filters), 0=no

REQUEST_TIMEOUT = env_int("REQUEST_TIMEOUT", 25)

EXCLUDED_MARKET_KEYWORDS = {
    "spread",
    "total",
    "over",
    "under",
    "o/u",
    "handicap",
    "puck line",
    "run line",
}
SIGNED_NUMBER_RE = re.compile(r"[+-]\d+(\.\d+)?")

def make_session() -> requests.Session:
    s = requests.Session()
    retries = Retry(
        total=6,
        backoff_factor=1.0,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=frozenset(["GET", "POST"]),
        raise_on_status=False,
    )
    s.mount("https://", HTTPAdapter(max_retries=retries))
    s.headers.update({"Accept": "application/json", "User-Agent": "polymarket-sports-value/1.0"})
    return s


SESSION = make_session()


def http_get_json(url: str, params: Optional[Dict[str, Any]] = None) -> Any:
    r = SESSION.get(url, params=params or {}, timeout=REQUEST_TIMEOUT)
    if r.status_code >= 400:
        raise RuntimeError(f"HTTP {r.status_code} GET {url}: {r.text[:300]}")
    return r.json()


def gamma_get(path: str, params: Optional[Dict[str, Any]] = None) -> Any:
    return http_get_json(f"{GAMMA_API_BASE}{path}", params=params)


def parse_json_list(value: Any) -> List[Any]:
    """
    Gamma returns some list fields as JSON-encoded strings.
    Examples: outcomes, outcomePrices, clobTokenIds, etc.
    """
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, str):
        v = value.strip()
        if not v:
            return []
        try:
            return json.loads(v)
        except json.JSONDecodeError:
            # fallback: treat as comma-separated
            v = v.strip("[]")
            return [x.strip().strip('"').strip("'") for x in v.split(",") if x.strip()]
    return []


def iso_to_dt(s: Any) -> Optional[datetime]:
    if not s or not isinstance(s, str):
        return None
    try:
        # handle trailing "Z"
        if s.endswith("Z"):
            s = s[:-1] + "+00:00"
        return datetime.fromisoformat(s).astimezone(timezone.utc)
    except Exception:
        return None


ET_TZ = ZoneInfo("America/New_York")


def fmt_start_time(dt: Optional[datetime]) -> str:
    """
    Return an Eastern Time string (EST/EDT) that’s easy to read in Discord/logs.
    Example: 2026-01-12 03:15 PM EST
    """
    if not dt:
        return ""
    try:
        local = dt.astimezone(ET_TZ)
        return local.strftime("%Y-%m-%d %I:%M %p %Z")
    except Exception:
        # fallback: show UTC ISO if something weird happens
        return dt.isoformat().replace("+00:00", "Z")


def normalize_league(league: str) -> str:
    """
    Make sportsbook league strings consistent for display + matching context.
    This is intentionally simple: it improves stability across different feeds
    without requiring team alias maintenance.
    """
    s = (league or "").strip().lower()
    if not s:
        return ""
    # common sportsbook identifiers
    if "nfl" in s:
        return "NFL"
    if "ncaaf" in s or ("college" in s and "football" in s):
        return "NCAAF"
    if "ncaab" in s or ("college" in s and ("basketball" in s or "hoops" in s)):
        return "NCAAB"
    if "nba" in s:
        return "NBA"
    if "nhl" in s:
        return "NHL"
    if "mlb" in s:
        return "MLB"
    return (league or "").strip().upper()


SPORT_KEY_TO_LEAGUE = {
    "americanfootball_nfl": "NFL",
    "basketball_nba": "NBA",
    "baseball_mlb": "MLB",
    "icehockey_nhl": "NHL",
    "americanfootball_ncaaf": "NCAAF",
    "basketball_ncaab": "NCAAB",
}


def clean_team_text(name: str) -> str:
    text = (name or "").strip()
    if not text:
        return ""

    text = text.replace("&", "and")
    text = re.sub(r"^\s*(#\d+|\(\d+\)|No\.\s*\d+)\s+", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\(\d+\s*-\s*\d+[^)]*\)", "", text)
    text = re.sub(r"\b(university|college|univ|the)\b", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def canon(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", (s or "").lower())


GENERIC_OUTCOMES = {"yes", "no", "home", "away", "draw", "tie"}


def team_tokens(name: str) -> set[str]:
    """
    Lightweight tokenization for matching ONLY.
    Display always uses Polymarket's raw team strings.
    """
    cleaned = clean_team_text(name).lower()
    if not cleaned:
        return set()
    toks = [t for t in re.split(r"\s+", cleaned) if t]
    toks = [re.sub(r"[^a-z0-9]+", "", t) for t in toks]
    toks = [t for t in toks if t]
    return set(toks)


def overlap_coeff(a: set[str], b: set[str]) -> float:
    """
    Intersection / min(|A|, |B|). Works well for cases like:
      {"texans"} vs {"houston","texans"} => 1.0
      {"ohio","state"} vs {"ohio","state","buckeyes"} => 1.0
    """
    if not a or not b:
        return 0.0
    inter = len(a & b)
    denom = min(len(a), len(b))
    return inter / denom if denom else 0.0


MATCH_MIN_OVERLAP = env_float("MATCH_MIN_OVERLAP", 0.80)


def moneyline_to_prob(ml: int) -> float:
    """
    Convert American odds to implied probability (WITH vig).
    """
    if ml < 0:
        return (-ml) / ((-ml) + 100.0)
    return 100.0 / (ml + 100.0)


def prob_to_moneyline(p: float) -> Optional[int]:
    """
    Convert probability to American odds.
    """
    if p <= 0 or p >= 1:
        return None
    if p >= 0.5:
        ml = -100.0 * p / (1.0 - p)
    else:
        ml = 100.0 * (1.0 - p) / p
    return int(round(ml))


def is_outside_range(poly_odds: int, range_data: Dict[str, Any]) -> bool:
    """
    Determine if Polymarket moneyline is outside the sportsbook range.
    Must be done in probability space (American odds are non-linear).
    """
    try:
        min_p = float(range_data["min_prob"])
        max_p = float(range_data["max_prob"])
    except Exception:
        return False

    poly_p = moneyline_to_prob(poly_odds)
    lo = min(min_p, max_p)
    hi = max(min_p, max_p)
    return poly_p < lo or poly_p > hi


@dataclass(frozen=True)
class PolyMoneylineMarket:
    slug: str
    url: str
    start_time: Optional[datetime]
    team1: str
    team2: str
    prob1: float
    prob2: float
    ml1: Optional[int]
    ml2: Optional[int]


@dataclass(frozen=True)
class SportsbookGame:
    start_time: Optional[datetime]
    home_team: str
    away_team: str
    home_fair_prob: float
    away_fair_prob: float
    home_fair_ml: int
    away_fair_ml: int
    home_range: Dict[str, Any]
    away_range: Dict[str, Any]
    books_used: int
    league: str
    sport_key: str
    per_book: List[Dict[str, Any]]


def get_moneyline_types() -> List[str]:
    """
    Determine which Gamma sports_market_types correspond to moneyline markets.
    If MONEYLINE_TYPES env is set, use that (comma-separated).
    Otherwise, call /sports/market-types and keep anything containing MONEY+LINE.
    """
    if MONEYLINE_TYPES:
        return [t.strip() for t in MONEYLINE_TYPES.split(",") if t.strip()]

    try:
        types = gamma_get("/sports/market-types")
        items: List[Any] = []
        if isinstance(types, list):
            items = types
        elif isinstance(types, dict):
            for key in ("marketTypes", "market_types", "types", "data", "results"):
                value = types.get(key)
                if isinstance(value, list):
                    items = value
                    break

        found: List[str] = []
        for item in items:
            if isinstance(item, str):
                name = item
            elif isinstance(item, dict):
                name = str(
                    item.get("name")
                    or item.get("type")
                    or item.get("sportsMarketType")
                    or item.get("label")
                    or ""
                )
            else:
                name = str(item)

            up = name.upper()
            if "MONEY" in up and "LINE" in up:
                found.append(name)

        seen = set()
        output: List[str] = []
        for name in found:
            if name in seen:
                continue
            output.append(name)
            seen.add(name)

        return output or ["MONEYLINE"]
    except Exception:
        return ["MONEYLINE"]


def _parse_polymarket_markets(raw_markets: List[Dict[str, Any]], debug: Dict[str, Any]) -> List[PolyMoneylineMarket]:
    markets: List[PolyMoneylineMarket] = []
    for m in raw_markets:
        outcomes = parse_json_list(m.get("outcomes"))
        prices = parse_json_list(m.get("outcomePrices"))

        if len(outcomes) != 2:
            debug["parse_outcomes_failed"] += 1
            continue

        if len(prices) != 2:
            debug["parse_prices_failed"] += 1
            continue

        o1, o2 = str(outcomes[0]), str(outcomes[1])
        if canon(o1) in GENERIC_OUTCOMES or canon(o2) in GENERIC_OUTCOMES:
            continue

        slug = str(m.get("slug") or "").strip()
        title = str(m.get("title") or "").strip()
        question = str(m.get("question") or "").strip()
        text = " ".join([slug, title, question, o1, o2]).lower()
        if any(keyword in text for keyword in EXCLUDED_MARKET_KEYWORDS):
            debug["parse_excluded_non_moneyline"] += 1
            continue
        if SIGNED_NUMBER_RE.search(text):
            debug["parse_excluded_non_moneyline"] += 1
            continue

        try:
            p1 = float(prices[0])
            p2 = float(prices[1])
        except Exception:
            debug["parse_prices_failed"] += 1
            continue

        if not (0.0 < p1 < 1.0 and 0.0 < p2 < 1.0):
            debug["parse_prices_failed"] += 1
            continue

        url = f"https://polymarket.com/market/{slug}" if slug else ""

        start_time = iso_to_dt(
            m.get("eventStartTime")
            or m.get("gameStartTime")
            or m.get("startDateIso")
            or m.get("startDate")
        )

        markets.append(
            PolyMoneylineMarket(
                slug=slug,
                url=url,
                start_time=start_time,
                team1=o1,
                team2=o2,
                prob1=p1,
                prob2=p2,
                ml1=prob_to_moneyline(p1),
                ml2=prob_to_moneyline(p2),
            )
        )

    return markets


def fetch_polymarket_moneylines_with_debug() -> Tuple[List[PolyMoneylineMarket], Dict[str, Any]]:
    """
    Fetch active, unclosed sports moneyline markets with fallback filters if enabled.
    """
    ml_types = get_moneyline_types()
    debug: Dict[str, Any] = {
        "tag_id": GAME_BETS_TAG_ID,
        "market_types": ml_types,
        "closed": False,
        "selected_attempt": None,
        "attempts": [],
        "raw_markets": 0,
        "parsed_markets": 0,
        "parse_outcomes_failed": 0,
        "parse_prices_failed": 0,
        "parse_excluded_non_moneyline": 0,
    }

    def _pull_raw(params_base: Dict[str, Any]) -> List[Dict[str, Any]]:
        raw: List[Dict[str, Any]] = []
        offset = 0
        limit = 500
        while True:
            params = dict(params_base)
            params["limit"] = limit
            params["offset"] = offset
            batch = gamma_get("/markets", params=params)
            if not isinstance(batch, list) or not batch:
                break
            raw.extend(batch)
            if len(batch) < limit:
                break
            offset += limit
        return raw

    attempts: List[Tuple[str, Dict[str, Any]]] = [
        (
            "tag+types",
            {
                "active": True,
                "closed": False,
                "tag_id": GAME_BETS_TAG_ID,
                "sports_market_types": ml_types,
            },
        ),
        ("tag_only", {"active": True, "closed": False, "tag_id": GAME_BETS_TAG_ID}),
        ("no_tag", {"active": True, "closed": False}),
    ]

    raw_markets: List[Dict[str, Any]] = []
    for name, params in attempts:
        if raw_markets:
            break
        if name != "tag+types" and not POLY_FALLBACK:
            continue
        try:
            batch = _pull_raw(params)
            debug["attempts"].append({"name": name, "params": params, "count": len(batch)})
            if batch:
                raw_markets = batch
                debug["selected_attempt"] = name
        except Exception as exc:
            debug["attempts"].append(
                {"name": name, "params": params, "count": 0, "error": str(exc)}
            )

    debug["raw_markets"] = len(raw_markets)
    markets = _parse_polymarket_markets(raw_markets, debug)
    debug["parsed_markets"] = len(markets)
    return markets, debug


def fetch_polymarket_moneylines() -> List[PolyMoneylineMarket]:
    markets, _debug = fetch_polymarket_moneylines_with_debug()
    return markets


def write_polymarket_snapshot(markets: List[PolyMoneylineMarket]) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # CSV snapshot
    csv_path = OUTPUT_DIR / "polymarket_moneylines.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "start_time_utc",
                "team1",
                "team2",
                "prob1",
                "prob2",
                "ml1",
                "ml2",
                "url",
                "slug",
            ]
        )
        for m in sorted(
            markets,
            key=lambda x: (x.start_time or datetime.max.replace(tzinfo=timezone.utc), x.slug),
        ):
            w.writerow(
                [
                    (m.start_time.isoformat().replace("+00:00", "Z") if m.start_time else ""),
                    m.team1,
                    m.team2,
                    f"{m.prob1:.6f}",
                    f"{m.prob2:.6f}",
                    m.ml1 if m.ml1 is not None else "",
                    m.ml2 if m.ml2 is not None else "",
                    m.url,
                    m.slug,
                ]
            )

    # JSON snapshot
    json_path = OUTPUT_DIR / "polymarket_moneylines.json"
    json_path.write_text(
        json.dumps(
            [
                {
                    "start_time_utc": (m.start_time.isoformat().replace("+00:00", "Z") if m.start_time else None),
                    "team1": m.team1,
                    "team2": m.team2,
                    "prob1": m.prob1,
                    "prob2": m.prob2,
                    "ml1": m.ml1,
                    "ml2": m.ml2,
                    "url": m.url,
                    "slug": m.slug,
                }
                for m in markets
            ],
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )


def load_sportsbook_games() -> List[SportsbookGame]:
    games, _debug = load_sportsbook_games_with_debug()
    return games


def load_sportsbook_games_with_debug() -> Tuple[List[SportsbookGame], Dict[str, Any]]:
    board, odds_debug = build_moneyline_board(
        sport_keys=ODDS_SPORT_KEYS or None,
        regions=ODDS_REGIONS,
        return_debug=True,
    )

    out: List[SportsbookGame] = []
    for event in board:
        consensus = event.get("consensus") or {}
        start = iso_to_dt(event.get("commence_time"))
        home = str(event.get("home_team") or "").strip()
        away = str(event.get("away_team") or "").strip()
        if not home or not away:
            continue

        try:
            home_prob_fair = float(consensus["home_prob_fair"])
            away_prob_fair = float(consensus["away_prob_fair"])
            home_fair_ml = int(consensus["home_fair_american"])
            away_fair_ml = int(consensus["away_fair_american"])
        except Exception:
            continue

        home_range = consensus.get("home_range") or {}
        away_range = consensus.get("away_range") or {}
        books_used = int(consensus.get("books_used") or 0)
        sport_key = str(event.get("sport_key") or "")
        league = SPORT_KEY_TO_LEAGUE.get(sport_key, normalize_league(sport_key))

        out.append(
            SportsbookGame(
                start_time=start,
                home_team=home,
                away_team=away,
                home_fair_prob=home_prob_fair,
                away_fair_prob=away_prob_fair,
                home_fair_ml=home_fair_ml,
                away_fair_ml=away_fair_ml,
                home_range=home_range,
                away_range=away_range,
                books_used=books_used,
                league=league,
                sport_key=sport_key,
                per_book=event.get("per_book", []),
            )
        )

    return out, odds_debug


def match_sportsbook_game(
    poly: PolyMoneylineMarket, books: List[SportsbookGame]
) -> Optional[Tuple[int, SportsbookGame]]:
    """
    Match by lightweight token overlap + time proximity.
    No aliases/uniforming; this is only to align sportsbook odds with Polymarket outcomes.
    """
    best_match: Optional[Tuple[int, SportsbookGame]] = None
    best_score = 0.0
    best_delta = float("inf")

    for idx, sb in enumerate(books):
        # If both have times, require proximity (avoids accidental collisions)
        if poly.start_time and sb.start_time:
            delta = abs((poly.start_time - sb.start_time).total_seconds())
            if delta > TIME_WINDOW_HOURS * 3600:
                continue
        else:
            delta = float("inf")

        p1 = team_tokens(poly.team1)
        p2 = team_tokens(poly.team2)
        a = team_tokens(sb.home_team)
        b = team_tokens(sb.away_team)

        # direct mapping score and swapped mapping score
        direct = overlap_coeff(p1, a) + overlap_coeff(p2, b)
        swapped = overlap_coeff(p1, b) + overlap_coeff(p2, a)

        score = max(direct, swapped)
        if score < (2 * MATCH_MIN_OVERLAP):
            continue

        if score > best_score or (score == best_score and delta < best_delta):
            best_match = (idx, sb)
            best_score = score
            best_delta = delta

    return best_match


def post_discord(text: str) -> None:
    if not DISCORD_WEBHOOK_URL:
        raise RuntimeError("DISCORD_WEBHOOK_URL not set (workflow should map secrets.DISCORD_SPORTS_ALERT).")

    content = f"{DISCORD_MENTION}\n{text}" if DISCORD_MENTION else text

    if DRY_RUN:
        print(content)
        return

    # Keep it to one alert; if we overflow, hard-trim (still one message).
    if len(content) > 1900:
        content = content[:1890] + "\n…(trimmed)"

    payload = {"content": content, "allowed_mentions": {"parse": ["users", "roles", "everyone"]}}
    r = SESSION.post(DISCORD_WEBHOOK_URL, json=payload, timeout=20)
    if r.status_code >= 400:
        raise RuntimeError(f"Discord webhook HTTP {r.status_code}: {r.text[:200]}")


def compute_value_rankings(
    polys: List[PolyMoneylineMarket], books: List[SportsbookGame]
) -> Tuple[List[Dict[str, Any]], List[PolyMoneylineMarket], List[SportsbookGame], List[str]]:
    """
    For each matched game:
      - compute de-vig fair probs from sportsbook moneylines
      - evaluate both sides with filtering
      - compute edge = p_fair - p_poly
      - keep if edge >= MIN_EDGE and value_rel criteria
    """
    rows: List[Dict[str, Any]] = []
    skipped_reasons: List[str] = []
    matched_books: set[int] = set()
    unmatched_polys: List[PolyMoneylineMarket] = []

    for pm in polys:
        matched = match_sportsbook_game(pm, books)
        if not matched:
            unmatched_polys.append(pm)
            continue
        sb_index, sb = matched
        matched_books.add(sb_index)

        sb_league = normalize_league(sb.league or "")
        # Determine which sportsbook side maps to which Polymarket outcome using the same token overlap approach.
        p1 = team_tokens(pm.team1)
        p2 = team_tokens(pm.team2)
        a = team_tokens(sb.home_team)
        b = team_tokens(sb.away_team)
        direct = overlap_coeff(p1, a) + overlap_coeff(p2, b)
        swapped = overlap_coeff(p1, b) + overlap_coeff(p2, a)
        if max(direct, swapped) < (2 * MATCH_MIN_OVERLAP):
            skipped_reasons.append(
                f"Low match confidence: {sb.home_team} vs {sb.away_team} <-> {pm.team1} vs {pm.team2}"
            )
            continue

        # Map sportsbook odds/probs to Polymarket outcomes (team1/team2) for legible output.
        if direct >= swapped:
            sb_ml_1, sb_p_1 = sb.home_fair_ml, sb.home_fair_prob
            sb_ml_2, sb_p_2 = sb.away_fair_ml, sb.away_fair_prob
            range_1, range_2 = sb.home_range, sb.away_range
        else:
            sb_ml_1, sb_p_1 = sb.away_fair_ml, sb.away_fair_prob
            sb_ml_2, sb_p_2 = sb.home_fair_ml, sb.home_fair_prob
            range_1, range_2 = sb.away_range, sb.home_range

        start = pm.start_time or sb.start_time
        start_s = fmt_start_time(start)
        # Always show Polymarket's team strings exactly (what you asked for).
        matchup = f"{pm.team1} vs {pm.team2}"

        candidates = [
            {
                "team": pm.team1,
                "sportsbook_ml": sb_ml_1,
                "sportsbook_fair_prob": sb_p_1,
                "polymarket_prob": pm.prob1,
                "polymarket_ml": pm.ml1,
                "range": range_1,
                "range_side": "home" if direct >= swapped else "away",
            },
            {
                "team": pm.team2,
                "sportsbook_ml": sb_ml_2,
                "sportsbook_fair_prob": sb_p_2,
                "polymarket_prob": pm.prob2,
                "polymarket_ml": pm.ml2,
                "range": range_2,
                "range_side": "away" if direct >= swapped else "home",
            },
        ]

        for candidate in candidates:
            sportsbook_ml = candidate["sportsbook_ml"]
            # "less than +200" means exclude +200 exactly (and beyond)
            if sportsbook_ml >= SPORTSBOOK_MAX_UNDERDOG:
                continue

            polymarket_ml = candidate["polymarket_ml"]
            # "greater than -300" means exclude -300 exactly (and more negative)
            if polymarket_ml is None or polymarket_ml <= POLYMARKET_MAX_FAVORITE:
                continue

            range_data = candidate["range"]
            outside_range = is_outside_range(polymarket_ml, range_data)
            if REQUIRE_OUTSIDE_RANGE and not outside_range:
                continue

            sportsbook_fair_prob = candidate["sportsbook_fair_prob"]
            if sportsbook_fair_prob <= 0:
                continue

            polymarket_prob = candidate["polymarket_prob"]
            edge_abs = sportsbook_fair_prob - polymarket_prob
            if edge_abs < MIN_EDGE:
                continue

            value_rel = edge_abs / sportsbook_fair_prob

            hr = sb.home_range or {}
            ar = sb.away_range or {}

            rows.append(
                {
                    "league": sb_league,
                    "matchup": matchup,
                    "recommended_side": candidate["team"],
                    "range_side": candidate["range_side"],
                    "start_time_et": start_s,
                    "sportsbook_ml": sportsbook_ml,
                    "polymarket_ml": polymarket_ml,
                    "sportsbook_fair_prob": sportsbook_fair_prob,
                    "polymarket_prob": polymarket_prob,
                    "edge_abs": edge_abs,
                    "value_rel": value_rel,
                    "polymarket_url": pm.url,
                    "home_team": sb.home_team,
                    "away_team": sb.away_team,
                    "home_fair_ml": sb.home_fair_ml,
                    "away_fair_ml": sb.away_fair_ml,
                    "home_prob_fair": sb.home_fair_prob,
                    "away_prob_fair": sb.away_fair_prob,
                    "home_min_prob": hr.get("min_prob"),
                    "home_max_prob": hr.get("max_prob"),
                    "home_min_odds": hr.get("min_odds"),
                    "home_max_odds": hr.get("max_odds"),
                    "home_min_book": hr.get("min_book"),
                    "home_max_book": hr.get("max_book"),
                    "away_min_prob": ar.get("min_prob"),
                    "away_max_prob": ar.get("max_prob"),
                    "away_min_odds": ar.get("min_odds"),
                    "away_max_odds": ar.get("max_odds"),
                    "away_min_book": ar.get("min_book"),
                    "away_max_book": ar.get("max_book"),
                    "books_used": sb.books_used,
                    "polymarket_home_ml": pm.ml1,
                    "polymarket_away_ml": pm.ml2,
                    "home_outside": is_outside_range(pm.ml1, sb.home_range) if pm.ml1 is not None else False,
                    "away_outside": is_outside_range(pm.ml2, sb.away_range) if pm.ml2 is not None else False,
                }
            )

    unmatched_books = [sb for idx, sb in enumerate(books) if idx not in matched_books]
    # stable-ish: value first, then absolute edge
    rows.sort(key=lambda r: (r["value_rel"], r["edge_abs"]), reverse=True)
    return rows, unmatched_polys, unmatched_books, skipped_reasons


def write_value_reports(rows: List[Dict[str, Any]]) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Full ranking CSV
    csv_path = OUTPUT_DIR / "value_rankings.csv"
    cols = [
        "league",
        "matchup",
        "recommended_side",
        "range_side",
        "start_time_et",
        "sportsbook_ml",
        "polymarket_ml",
        "home_fair_ml",
        "away_fair_ml",
        "home_prob_fair",
        "away_prob_fair",
        "home_min_prob",
        "home_max_prob",
        "home_min_odds",
        "home_max_odds",
        "home_min_book",
        "home_max_book",
        "away_min_prob",
        "away_max_prob",
        "away_min_odds",
        "away_max_odds",
        "away_min_book",
        "away_max_book",
        "books_used",
        "home_outside",
        "away_outside",
        "sportsbook_fair_prob",
        "polymarket_prob",
        "edge_abs",
        "value_rel",
        "polymarket_url",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow(r)

    # Top-N text file (human readable)
    top = rows[:TOP_N]
    txt_path = OUTPUT_DIR / "value_topN.txt"
    lines: List[str] = []
    lines.append(f"Top {TOP_N} Polymarket value")
    lines.append(
        "Filters: "
        f"sportsbook_ml < {SPORTSBOOK_MAX_UNDERDOG}, "
        f"polymarket_ml > {POLYMARKET_MAX_FAVORITE}, "
        f"min edge >= {MIN_EDGE:.1%}, "
        "outside sportsbook range only"
    )
    lines.append("")
    for i, r in enumerate(top, 1):
        pm_ml_s = str(r["polymarket_ml"]) if r["polymarket_ml"] is not None else "n/a"
        league_prefix = f"{r['league']} - " if r.get("league") else ""
        lines.append(
            f"{i:>2}. {league_prefix}{r['matchup']} @ {r['start_time_et']}\n"
            f"    Side: {r['recommended_side']} | "
            f"Consensus: {r['sportsbook_ml']}  |  PM: {pm_ml_s}  |  "
            f"value: {r['value_rel']:+.1%}  |  edge: {r['edge_abs']:+.1%}  |  {r['polymarket_url']}"
        )
    txt_path.write_text("\n".join(lines), encoding="utf-8")


def format_discord_message(rows: List[Dict[str, Any]], polymarket_count: int, sportsbook_count: int) -> str:
    top = rows[: min(TOP_N, DISCORD_TOP_N)]

    header = f"**Polymarket vs Sportsbooks — Top {len(top)} (edge ≥ {MIN_EDGE:.1%}, outside SB range)**\n"

    if not top:
        return header + "\nNo qualifying edges found today."

    lines: List[str] = [header]
    for i, r in enumerate(top, 1):
        pm_ml_s = str(r["polymarket_ml"]) if r["polymarket_ml"] is not None else "n/a"
        league_prefix = f"{r['league']} - " if r.get("league") else ""
        if r.get("range_side") == "home":
            min_odds = r.get("home_min_odds")
            max_odds = r.get("home_max_odds")
            min_book = r.get("home_min_book")
            max_book = r.get("home_max_book")
        else:
            min_odds = r.get("away_min_odds")
            max_odds = r.get("away_max_odds")
            min_book = r.get("away_min_book")
            max_book = r.get("away_max_book")

        def _fmt_range(odds: Optional[int], book: Optional[str]) -> str:
            if odds is None:
                return "?"
            label = (book or "").strip().replace(" ", "")[:10]
            return f"{int(odds)}({label})" if label else f"{int(odds)}"

        sb_range = f"{_fmt_range(min_odds, min_book)}→{_fmt_range(max_odds, max_book)}"
        lines.append(
            f"{i}) {league_prefix}{r['matchup']} — {r['recommended_side']} | "
            f"PM {pm_ml_s} vs Fair {r['sportsbook_ml']} | "
            f"Edge {r['edge_abs']:+.1%} | "
            f"SB {sb_range} | "
            f"{r['polymarket_url']}"
        )

    # Optional lightweight debug footer (helps you confirm matching volume without digging into artifacts)
    if INCLUDE_DEBUG_SUMMARY:
        lines.append("")
        lines.append(f"_Debug: scored {len(rows)} edges; see artifact reports/ for full CSVs._")

    return "\n".join(lines)


def write_unmatched_reports(
    unmatched_polys: List[PolyMoneylineMarket], unmatched_books: List[SportsbookGame]
) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    poly_path = OUTPUT_DIR / "unmatched_polymarket.csv"
    with poly_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["start_time_et", "team1", "team2", "url", "slug"])
        for pm in unmatched_polys:
            start_s = fmt_start_time(pm.start_time)
            w.writerow([start_s, pm.team1, pm.team2, pm.url, pm.slug])

    book_path = OUTPUT_DIR / "unmatched_sportsbook.csv"
    with book_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["league", "start_time_et", "home_team", "away_team", "sport_key"])
        for sb in unmatched_books:
            start_s = fmt_start_time(sb.start_time)
            w.writerow([sb.league, start_s, sb.home_team, sb.away_team, sb.sport_key])


def write_diagnostics(diagnostics: Dict[str, Any]) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUTPUT_DIR / "diagnostics.json"
    path.write_text(json.dumps(diagnostics, indent=2, sort_keys=True), encoding="utf-8")


def main() -> None:
    diagnostics: Dict[str, Any] = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "config": {
            "min_edge": MIN_EDGE,
            "top_n": TOP_N,
            "discord_top_n": DISCORD_TOP_N,
            "time_window_hours": TIME_WINDOW_HOURS,
            "odds_sport_keys": ODDS_SPORT_KEYS or None,
            "odds_regions": ODDS_REGIONS,
            "min_books": env_int("MIN_BOOKS", 3),
            "poly_tag_id": GAME_BETS_TAG_ID,
            "poly_fallback": POLY_FALLBACK,
        },
    }
    write_diagnostics(diagnostics)

    polys, poly_debug = fetch_polymarket_moneylines_with_debug()
    diagnostics["polymarket"] = poly_debug
    write_diagnostics(diagnostics)
    write_polymarket_snapshot(polys)

    books, odds_debug = load_sportsbook_games_with_debug()
    diagnostics["odds_api"] = odds_debug
    write_diagnostics(diagnostics)
    rows, unmatched_polys, unmatched_books, _skips = compute_value_rankings(polys, books)
    diagnostics["results"] = {
        "polymarket_count": len(polys),
        "sportsbook_count": len(books),
        "unmatched_polymarket": len(unmatched_polys),
        "unmatched_sportsbook": len(unmatched_books),
        "qualifying_edges": len(rows),
    }
    write_diagnostics(diagnostics)
    write_value_reports(rows)
    write_unmatched_reports(unmatched_polys, unmatched_books)

    msg = format_discord_message(rows, polymarket_count=len(polys), sportsbook_count=len(books))
    post_discord(msg)


if __name__ == "__main__":
    main()
