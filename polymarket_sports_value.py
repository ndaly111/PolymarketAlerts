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
import textwrap
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
DEBUG_DISCORD_TOP_N = env_int("DEBUG_DISCORD_TOP_N", 20)  # how many lines to print in DEBUG_MODE
MIN_EDGE = env_float("MIN_EDGE", 0.03)  # prob gap threshold (e.g., 0.03 = 3%)
GAME_BETS_TAG_ID = env_int("GAME_BETS_TAG_ID", 100639)  # Polymarket "game bets" tag
SPORTSBOOK_MAX_UNDERDOG = env_int("SPORTSBOOK_MAX_UNDERDOG", 200)
POLYMARKET_MAX_FAVORITE = env_int("POLYMARKET_MAX_FAVORITE", -300)
TIME_WINDOW_HOURS = env_int("TIME_WINDOW_HOURS", 36)
INCLUDE_DEBUG_SUMMARY = env_int("INCLUDE_DEBUG_SUMMARY", 1)  # 1=yes, 0=no
DEBUG_SUMMARY_LINES = env_int("DEBUG_SUMMARY_LINES", 8)
DRY_RUN = env_int("DRY_RUN", 0)
REQUIRE_OUTSIDE_RANGE = env_int("REQUIRE_OUTSIDE_RANGE", 1)
DEBUG_MODE = env_int("DEBUG_MODE", 0)  # 1 = print Discord debug preview
DEBUG_POLY_SLUG = (os.getenv("DEBUG_POLY_SLUG", "") or "").strip()
DEBUG_POLY_MAX = env_int("DEBUG_POLY_MAX", 10)  # max markets to dump when debug is on
CLOB_BASE = "https://clob.polymarket.com"

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
    "o/u",
    "handicap",
    "puck line",
    "run line",
}
# Use word boundaries so we don't falsely match team names like "Rovers" or "Governors".
EXCLUDED_MARKET_RE = re.compile(
    r"\b(spread|handicap|puck line|run line)\b"
    r"|\b(total|o/u|over/under)\b"
    r"|\b(over|under)\s*\d",
    flags=re.IGNORECASE,
)
# Detect signed point/goal totals like -3.5, +7, etc.
# IMPORTANT: do not match dates in slugs like 2025-11-03 (the "-03" part).
SIGNED_NUMBER_RE = re.compile(r"(?<!\d)[+-]\s*\d+(?:\.\d+)?(?!\d)")

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


_CLOB_PRICE_CACHE: Dict[Tuple[str, str], Optional[float]] = {}


def clob_get_json(path: str, params: Optional[Dict[str, Any]] = None) -> Any:
    url = f"{CLOB_BASE}{path}"
    r = SESSION.get(url, params=params or {}, timeout=REQUEST_TIMEOUT)
    if r.status_code >= 400:
        raise RuntimeError(f"HTTP {r.status_code} GET {url}: {r.text[:300]}")
    return r.json()


def clob_get_price(token_id: str, side: str = "buy") -> Optional[float]:
    token_id = str(token_id or "").strip()
    if not token_id:
        return None
    key = (token_id, side)
    if key in _CLOB_PRICE_CACHE:
        return _CLOB_PRICE_CACHE[key]

    try:
        data = clob_get_json("/price", params={"token_id": token_id, "side": side})
        price = data.get("price", None)
        if price is None:
            _CLOB_PRICE_CACHE[key] = None
            return None
        p = float(price)
        if not (0.0 < p < 1.0):
            _CLOB_PRICE_CACHE[key] = None
            return None
        _CLOB_PRICE_CACHE[key] = p
        return p
    except Exception as exc:
        _CLOB_PRICE_CACHE[key] = None
        _ = exc
        return None


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


def _market_debug_block(m: Dict[str, Any]) -> List[str]:
    lines: List[str] = []

    market_id = str(m.get("id") or m.get("marketId") or "").strip()
    slug = str(m.get("slug") or "").strip()
    title = str(m.get("title") or "").strip().replace("\n", " ")
    question = str(m.get("question") or "").strip().replace("\n", " ")
    smt = (
        m.get("sportsMarketType")
        or m.get("sports_market_type")
        or m.get("sports_market_types")
        or ""
    )

    outcomes = parse_json_list(m.get("outcomes"))
    prices = parse_json_list(m.get("outcomePrices"))
    token_ids = parse_json_list(m.get("clobTokenIds"))

    lines.append("----- MARKET DEBUG -----")
    lines.append(f"id={market_id}")
    lines.append(f"slug={slug}")
    if smt:
        lines.append(f"sportsMarketType={smt}")
    if title:
        lines.append(f"title={textwrap.shorten(title, width=160, placeholder='…')}")
    if question and question != title:
        lines.append(f"question={textwrap.shorten(question, width=160, placeholder='…')}")

    lines.append(f"outcomes={outcomes}")
    lines.append(f"outcomePrices(Gamma)={prices}")
    lines.append(f"clobTokenIds={token_ids}")

    if len(outcomes) == 2 and len(prices) == 2:
        try:
            g1 = float(prices[0])
            g2 = float(prices[1])
        except Exception:
            g1 = None
            g2 = None

        if g1 and 0.0 < g1 < 1.0:
            lines.append(f"gamma_ml1={prob_to_moneyline(g1)}")
        if g2 and 0.0 < g2 < 1.0:
            lines.append(f"gamma_ml2={prob_to_moneyline(g2)}")

    if len(token_ids) == 2:
        t1 = str(token_ids[0])
        t2 = str(token_ids[1])
        p1 = clob_get_price(t1, side="buy")
        p2 = clob_get_price(t2, side="buy")
        lines.append(f"clob_buy_price1={p1}")
        lines.append(f"clob_buy_price2={p2}")
        if p1 and 0.0 < p1 < 1.0:
            lines.append(f"clob_ml1={prob_to_moneyline(p1)}")
        if p2 and 0.0 < p2 < 1.0:
            lines.append(f"clob_ml2={prob_to_moneyline(p2)}")

    lines.append("------------------------")
    return lines


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


def _clean_matchup_text(text: str) -> str:
    cleaned = text.replace("\n", " ").strip()
    cleaned = re.sub(r"^[A-Za-z]{2,12}\s*:\s*", "", cleaned).strip()
    for sep in (" — ", " - ", " | ", ": "):
        if sep in cleaned:
            cleaned = cleaned.split(sep, 1)[0].strip()
    return cleaned


def _strip_slug_date_tail(text: str) -> str:
    """
    Slugs often end with YYYY MM DD (after replacing '-' with spaces).
    Strip that suffix so team names don't include date tokens.
    """
    return re.sub(r"\b20\d{2}\s+[01]?\d\s+[0-3]?\d\b.*$", "", text).strip()


def _clean_team_name(name: str) -> str:
    return re.sub(r"[?!.]+$", "", name).strip()


def _extract_matchup(text: str) -> Optional[Tuple[str, str]]:
    if not text:
        return None
    cleaned = _clean_matchup_text(text)
    patterns = [
        r"^will (?P<a>.+?) (?:beat|defeat|win(?: vs| against)?|win over|over) (?P<b>.+?)\??$",
        r"^(?P<a>.+?) vs\.? (?P<b>.+)$",
        r"^(?P<a>.+?) v\.? (?P<b>.+)$",
        r"^(?P<a>.+?) @ (?P<b>.+)$",
        r"^(?P<a>.+?) at (?P<b>.+)$",
    ]
    for pattern in patterns:
        match = re.match(pattern, cleaned, flags=re.IGNORECASE)
        if not match:
            continue
        team_a = _clean_team_name(match.group("a"))
        team_b = _clean_team_name(match.group("b"))
        if team_a and team_b:
            return team_a, team_b
    return None


def _extract_matchup_from_market(market: Dict[str, Any]) -> Optional[Tuple[str, str]]:
    for key in ("title", "question"):
        value = str(market.get(key) or "").strip()
        matchup = _extract_matchup(value)
        if matchup:
            return matchup
    slug = str(market.get("slug") or "").strip()
    if slug:
        slug_text = _strip_slug_date_tail(slug.replace("-", " "))
        matchup = _extract_matchup(slug_text)
        if matchup:
            return matchup
        if " vs " in slug_text:
            parts = slug_text.split(" vs ", 1)
            if len(parts) == 2:
                team_a = _clean_team_name(_strip_slug_date_tail(parts[0]))
                team_b = _clean_team_name(_strip_slug_date_tail(parts[1]))
                if team_a and team_b:
                    return team_a, team_b
    return None


def _normalize_generic_outcomes(
    market: Dict[str, Any],
    outcomes: List[Any],
    prices: List[Any],
    debug: Dict[str, Any],
) -> Optional[Tuple[str, str, float, float]]:
    o1, o2 = str(outcomes[0]), str(outcomes[1])
    canon_o1 = canon(o1)
    canon_o2 = canon(o2)
    yes_no = {canon_o1, canon_o2} <= {"yes", "no"}
    if not yes_no:
        debug["parse_skipped_generic_outcomes"] += 1
        sample = debug.setdefault("skipped_generic_samples", [])
        if len(sample) < 20:
            sample.append(str(market.get("title") or market.get("question") or market.get("slug") or ""))
        return None

    matchup = _extract_matchup_from_market(market)
    if not matchup:
        debug["parse_generic_title_failed"] += 1
        return None

    try:
        p1 = float(prices[0])
        p2 = float(prices[1])
    except Exception:
        debug["parse_generic_yesno_missing"] += 1
        return None

    yes_prob = p1 if canon_o1 == "yes" else p2
    no_prob = p2 if canon_o1 == "yes" else p1
    if not (0.0 < yes_prob < 1.0 and 0.0 < no_prob < 1.0):
        debug["parse_generic_yesno_missing"] += 1
        return None

    team_a, team_b = matchup
    return team_a, team_b, yes_prob, no_prob


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

        slug = str(m.get("slug") or "").strip()
        title = str(m.get("title") or "").strip()
        question = str(m.get("question") or "").strip()
        # IMPORTANT: do NOT include slug here because slugs often include dates (e.g., 2025-11-03)
        # which can trigger signed-number detection and incorrectly exclude valid markets.
        text = " ".join([title, question, o1, o2]).lower()
        if EXCLUDED_MARKET_RE.search(text):
            debug["parse_excluded_non_moneyline"] += 1
            continue
        m_signed = SIGNED_NUMBER_RE.search(text)
        if m_signed:
            s = m_signed.group(0).replace(" ", "")
            try:
                val = float(s)
            except Exception:
                val = 0.0
            if abs(val) <= 50:
                debug["parse_excluded_non_moneyline"] += 1
                continue

        if canon(o1) in GENERIC_OUTCOMES or canon(o2) in GENERIC_OUTCOMES:
            normalized = _normalize_generic_outcomes(m, outcomes, prices, debug)
            if not normalized:
                continue
            team1, team2, p1, p2 = normalized
        else:
            team1, team2 = o1, o2
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
                team1=team1,
                team2=team2,
                prob1=p1,
                prob2=p2,
                ml1=prob_to_moneyline(p1),
                ml2=prob_to_moneyline(p2),
            )
        )

    return markets


def write_polymarket_parse_debug(raw_markets: List[Dict[str, Any]], debug: Dict[str, Any]) -> None:
    """
    Writes a lightweight text dump so we can see why markets are being excluded.
    This is intentionally human-readable (not huge JSON).
    """
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUTPUT_DIR / "polymarket_parse_debug.txt"

    lines: List[str] = []
    lines.append("POLYMARKET PARSE DEBUG")
    lines.append("")
    lines.append("COUNTS")
    for k in [
        "raw_markets",
        "parsed_markets",
        "parse_outcomes_failed",
        "parse_prices_failed",
        "parse_excluded_non_moneyline",
        "parse_skipped_generic_outcomes",
        "parse_generic_title_failed",
        "parse_generic_yesno_missing",
        "selected_attempt",
    ]:
        if k in debug:
            lines.append(f"- {k}: {debug.get(k)}")
    lines.append("")

    skipped_samples = debug.get("skipped_generic_samples") or []
    if skipped_samples:
        lines.append("SAMPLE SKIPPED GENERIC TITLES (first 20)")
        for title in skipped_samples[:20]:
            lines.append(f"- {textwrap.shorten(str(title), width=140, placeholder='…')}")
        lines.append("")

    if DEBUG_MODE and DEBUG_POLY_SLUG:
        lines.append(f"FOCUSED DEBUG (slug={DEBUG_POLY_SLUG})")
        found = 0
        for m in raw_markets:
            slug = str(m.get("slug") or "").strip()
            if slug != DEBUG_POLY_SLUG:
                continue
            lines.extend(_market_debug_block(m))
            found += 1
            if found >= DEBUG_POLY_MAX:
                break
        if found == 0:
            lines.append("No raw markets matched DEBUG_POLY_SLUG.")
        lines.append("")

    lines.append("SAMPLE RAW MARKETS (first 60)")
    sample = raw_markets[:60]
    for i, m in enumerate(sample, 1):
        slug = (m.get("slug") or "").strip()
        title = (m.get("title") or m.get("question") or "").strip().replace("\n", " ")
        outcomes = m.get("outcomes")
        prices = m.get("outcomePrices") or m.get("outcome_prices")
        start = m.get("eventStartTime") or m.get("gameStartTime") or m.get("startDateIso") or m.get("startDate")
        title = textwrap.shorten(title, width=140, placeholder="…")
        lines.append(f"{i:02d}) slug={slug}")
        lines.append(f"    title={title}")
        lines.append(f"    start={start}")
        lines.append(f"    outcomes={outcomes}")
        lines.append(f"    prices={prices}")
        lines.append("")

    path.write_text("\n".join(lines), encoding="utf-8")


def _gamma_param(v: Any) -> str:
    """
    Gamma query params are picky. Legacy script uses lowercase 'true'/'false' strings.
    requests will serialize Python bools as 'True'/'False' (capitalized), which can yield 0 results.
    """
    if isinstance(v, bool):
        return "true" if v else "false"
    if v is None:
        return ""
    if isinstance(v, (int, float)):
        # ints should be strings for query params
        # floats are unlikely here but keep consistent
        return str(v)
    if isinstance(v, list):
        # safest for query params: comma-separated
        return ",".join(str(x) for x in v if str(x).strip())
    return str(v)


def _normalize_gamma_params(params: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for k, v in (params or {}).items():
        s = _gamma_param(v)
        if s != "":
            out[k] = s
    return out


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
        "parse_skipped_generic_outcomes": 0,
        "parse_generic_title_failed": 0,
        "parse_generic_yesno_missing": 0,
        "skipped_generic_samples": [],
    }

    def _pull_raw(params_base: Dict[str, Any]) -> List[Dict[str, Any]]:
        raw: List[Dict[str, Any]] = []
        offset = 0
        limit = 500
        while True:
            params = dict(params_base)
            params["limit"] = str(limit)
            params["offset"] = str(offset)
            params = _normalize_gamma_params(params)
            batch = gamma_get("/markets", params=params)
            # IMPORTANT: pagination must be based on the unfiltered batch, otherwise we can
            # stop early before reaching the page where DEBUG_POLY_SLUG appears.
            if not isinstance(batch, list) or not batch:
                break

            if DEBUG_MODE and DEBUG_POLY_SLUG:
                filtered = [
                    x for x in batch if str(x.get("slug") or "").strip() == DEBUG_POLY_SLUG
                ]
            else:
                filtered = batch

            raw.extend(filtered)
            # Use the unfiltered batch length to decide if we reached the end.
            if len(batch) < limit:
                break
            offset += limit
        return raw

    attempts: List[Tuple[str, Dict[str, Any]]] = [
        (
            "tag+types",
            {"active": "true", "closed": "false", "tag_id": str(GAME_BETS_TAG_ID)},
        ),
        ("tag_only", {"active": "true", "closed": "false", "tag_id": str(GAME_BETS_TAG_ID)}),
        ("no_tag", {"active": "true", "closed": "false"}),
    ]

    raw_markets: List[Dict[str, Any]] = []
    for name, params in attempts:
        if raw_markets:
            break
        if name != "tag+types" and not POLY_FALLBACK:
            continue
        try:
            if name == "tag+types":
                if not ml_types:
                    batch = _pull_raw(params)
                    debug["attempts"].append({"name": name, "params": params, "count": len(batch)})
                    if batch:
                        raw_markets = batch
                        debug["selected_attempt"] = name
                else:
                    combined: List[Dict[str, Any]] = []
                    seen_ids: set = set()
                    for market_type in ml_types:
                        typed_params = dict(params)
                        typed_params["sports_market_types"] = market_type
                        batch = _pull_raw(typed_params)
                        debug["attempts"].append(
                            {"name": name, "params": typed_params, "count": len(batch)}
                        )
                        for item in batch:
                            market_id = str(
                                item.get("id") or item.get("marketId") or item.get("slug") or ""
                            )
                            if market_id in seen_ids:
                                continue
                            seen_ids.add(market_id)
                            combined.append(item)
                    if combined:
                        raw_markets = combined
                        debug["selected_attempt"] = name
            else:
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
    write_polymarket_parse_debug(raw_markets, debug)
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
) -> Tuple[
    List[Dict[str, Any]],
    List[PolyMoneylineMarket],
    List[SportsbookGame],
    List[str],
    Optional[List[Dict[str, Any]]],
    Optional[Dict[str, int]],
]:
    """
    For each matched game:
      - compute de-vig fair probs from sportsbook moneylines
      - evaluate both sides with filtering
      - compute edge = p_fair - p_poly
      - keep if edge >= MIN_EDGE and value_rel criteria
    """
    rows: List[Dict[str, Any]] = []
    debug_rows_all: List[Dict[str, Any]] = []
    debug_counts: Dict[str, int] = {
        "matched_games": 0,
        "candidates_total": 0,
        "cut_sportsbook_underdog": 0,
        "cut_polymarket_favorite": 0,
        "cut_inside_range": 0,
        "cut_edge": 0,
        "cut_bad_prob": 0,
        "kept": 0,
    }
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
        debug_counts["matched_games"] += 1

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
            debug_counts["candidates_total"] += 1
            sportsbook_ml = candidate["sportsbook_ml"]
            polymarket_ml = candidate["polymarket_ml"]
            range_data = candidate["range"]
            sportsbook_fair_prob = candidate["sportsbook_fair_prob"]
            polymarket_prob = candidate["polymarket_prob"]
            edge_abs = sportsbook_fair_prob - polymarket_prob
            outside_range = False
            if polymarket_ml is not None:
                outside_range = is_outside_range(polymarket_ml, range_data)

            pass_sportsbook = sportsbook_ml < SPORTSBOOK_MAX_UNDERDOG
            pass_polymarket = (polymarket_ml is not None) and (polymarket_ml > POLYMARKET_MAX_FAVORITE)
            pass_outside = (not REQUIRE_OUTSIDE_RANGE) or outside_range
            pass_prob = sportsbook_fair_prob > 0
            pass_edge = edge_abs >= MIN_EDGE

            if DEBUG_MODE:
                debug_rows_all.append(
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
                        "outside_range": outside_range,
                        "books_used": sb.books_used,
                        "pass_sportsbook_ml": pass_sportsbook,
                        "pass_polymarket_ml": pass_polymarket,
                        "pass_outside_range": pass_outside,
                        "pass_edge": pass_edge,
                        "pass_prob": pass_prob,
                        "polymarket_url": pm.url,
                    }
                )

            if not pass_sportsbook:
                debug_counts["cut_sportsbook_underdog"] += 1
                continue
            if not pass_polymarket:
                debug_counts["cut_polymarket_favorite"] += 1
                continue
            if not pass_outside:
                debug_counts["cut_inside_range"] += 1
                continue
            if not pass_prob:
                debug_counts["cut_bad_prob"] += 1
                continue
            if not pass_edge:
                debug_counts["cut_edge"] += 1
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
            debug_counts["kept"] += 1

    unmatched_books = [sb for idx, sb in enumerate(books) if idx not in matched_books]
    # stable-ish: value first, then absolute edge
    rows.sort(key=lambda r: (r["value_rel"], r["edge_abs"]), reverse=True)
    if DEBUG_MODE:
        debug_rows_all.sort(key=lambda r: (r["edge_abs"], r["sportsbook_fair_prob"]), reverse=True)
        return rows, unmatched_polys, unmatched_books, skipped_reasons, debug_rows_all, debug_counts
    return rows, unmatched_polys, unmatched_books, skipped_reasons, None, None


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
        "home_team",
        "away_team",
        "polymarket_home_ml",
        "polymarket_away_ml",
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
    # If row dicts gain new fields over time, include them automatically so the workflow
    # doesn't crash with:
    #   ValueError: dict contains fields not in fieldnames
    extra_cols = sorted({k for r in rows for k in r.keys()} - set(cols))
    cols = cols + extra_cols

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
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


def format_discord_message(
    rows: List[Dict[str, Any]],
    polymarket_count: int,
    sportsbook_count: int,
    poly_debug: Optional[Dict[str, Any]] = None,
    debug_rows_all: Optional[List[Dict[str, Any]]] = None,
    debug_counts: Optional[Dict[str, int]] = None,
) -> str:
    if DEBUG_MODE:
        all_rows = debug_rows_all or []
        top_dbg = all_rows[: min(DEBUG_DISCORD_TOP_N, len(all_rows))]

        header = (
            f"**Polymarket vs Sportsbooks — DEBUG (top {len(top_dbg)} by edge, unfiltered)**\n"
            f"_Normal filters still apply to the non-debug ranking (min edge {MIN_EDGE:.1%}, "
            f"outside_range={bool(REQUIRE_OUTSIDE_RANGE)}, "
            f"SB<+{SPORTSBOOK_MAX_UNDERDOG}, PM>{POLYMARKET_MAX_FAVORITE})._"
        )

        lines: List[str] = [header]
        if poly_debug:
            attempts = poly_debug.get("attempts") or []
            if attempts:
                attempts_summary = ", ".join(
                    f"{attempt.get('name', '?')}={attempt.get('count', 0)}" for attempt in attempts
                )
                lines.append(f"Poly attempts: {attempts_summary}")
        if not top_dbg:
            lines.append("\nNo candidates available to display (debug_rows_all empty).")
            lines.append(f"Polymarket markets: {polymarket_count}, Sportsbook events: {sportsbook_count}")
            return "\n".join(lines)

        lines.append("")
        if debug_counts:
            lines.append(
                f"Counts: matched_games={debug_counts.get('matched_games', 0)}, "
                f"candidates={debug_counts.get('candidates_total', 0)}, "
                f"kept={debug_counts.get('kept', 0)}, "
                f"cut_edge={debug_counts.get('cut_edge', 0)}, "
                f"cut_inside_range={debug_counts.get('cut_inside_range', 0)}, "
                f"cut_SB_underdog={debug_counts.get('cut_sportsbook_underdog', 0)}, "
                f"cut_PM_fav={debug_counts.get('cut_polymarket_favorite', 0)}"
            )
            lines.append("")

        for i, row in enumerate(top_dbg, 1):
            league_prefix = f"{row['league']} - " if row.get("league") else ""
            pm_ml_s = str(row["polymarket_ml"]) if row["polymarket_ml"] is not None else "n/a"
            sb_ml = row.get("sportsbook_ml")
            edge = row.get("edge_abs", 0.0)
            sbp = row.get("sportsbook_fair_prob", 0.0)
            pmp = row.get("polymarket_prob", 0.0)
            books_used = row.get("books_used")

            fail_tags = []
            if not row.get("pass_outside_range", True) and REQUIRE_OUTSIDE_RANGE:
                fail_tags.append("inRange")
            if not row.get("pass_edge", True):
                fail_tags.append("lowEdge")
            if not row.get("pass_sportsbook_ml", True):
                fail_tags.append("SB>cap")
            if not row.get("pass_polymarket_ml", True):
                fail_tags.append("PM<cap")
            if not row.get("pass_prob", True):
                fail_tags.append("badProb")
            tag_s = f" ({', '.join(fail_tags)})" if fail_tags else ""

            lines.append(
                f"{i}) {league_prefix}{row['matchup']} @ {row.get('start_time_et','?')} — "
                f"{row['recommended_side']} | PM {pm_ml_s} ({pmp:.1%}) vs Fair {sb_ml} ({sbp:.1%}) | "
                f"Edge {edge:+.1%} | books={books_used}{tag_s}"
            )

        return "\n".join(lines)

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
            "debug_mode": DEBUG_MODE,
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
    rows, unmatched_polys, unmatched_books, _skips, debug_rows_all, debug_counts = compute_value_rankings(polys, books)
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

    msg = format_discord_message(
        rows,
        polymarket_count=len(polys),
        sportsbook_count=len(books),
        poly_debug=poly_debug,
        debug_rows_all=debug_rows_all,
        debug_counts=debug_counts,
    )
    post_discord(msg)


if __name__ == "__main__":
    main()
