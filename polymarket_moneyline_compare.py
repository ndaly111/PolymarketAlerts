#!/usr/bin/env python3
"""
Polymarket vs Sportsbook Moneyline Comparator
- Fetch Polymarket sports "winner" markets
- Fetch sportsbook H2H (moneyline) odds
- Match events by teams + start time tolerance
- Compute implied-prob edge and post ranked list to Discord

Designed for GitHub Actions daily run + workflow_dispatch manual run.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from zoneinfo import ZoneInfo
from typing import Any, List, Optional, Tuple

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from dateutil import parser as dtparser


# -----------------------------
# Config
# -----------------------------

GAMMA_BASE = "https://gamma-api.polymarket.com"
USER_AGENT = "polymarket-moneyline-compare/1.0 (+github actions)"

DEFAULT_TOP_N = int(os.getenv("COMPARE_TOP_N_DEFAULT", "25"))
DEFAULT_MIN_EDGE_BPS = int(os.getenv("COMPARE_MIN_EDGE_BPS", "0"))
GAME_BETS_TAG_ID = int(os.getenv("GAME_BETS_TAG_ID", "100639"))

SPORT_KEYS = {
    "americanfootball_nfl": "NFL",
    "basketball_nba": "NBA",
    "baseball_mlb": "MLB",
    "icehockey_nhl": "NHL",
    "americanfootball_ncaaf": "NCAAF",
    "basketball_ncaab": "NCAAB",
    "soccer_epl": "EPL",
    "soccer_uefa_champs_league": "UCL",
}

DEFAULT_SPORT_KEYS = [
    "americanfootball_nfl",
    "basketball_nba",
    "baseball_mlb",
    "icehockey_nhl",
    "americanfootball_ncaaf",
    "basketball_ncaab",
]

TEAM_STRIP_TOKENS = [
    "fc",
    "sc",
    "cf",
    "club",
    "the",
]

DISCORD_MAX_CHARS = 1800
MATCH_MIN_SCORE = float(os.getenv("MATCH_MIN_SCORE", "1.6"))


# -----------------------------
# Models
# -----------------------------

@dataclass
class PolyMarket:
    market_id: str
    start_time: Optional[datetime]
    team_a: str
    team_b: str
    prob_a: float
    prob_b: float
    url: Optional[str]


@dataclass
class BookLine:
    book: str
    odds_american: int


@dataclass
class BookEvent:
    sport_key: str
    league: str
    start_time: datetime
    home: str
    away: str
    best_home: Optional[BookLine]
    best_away: Optional[BookLine]


@dataclass
class MatchResult:
    league: str
    start_time: datetime
    team: str
    opponent: str
    poly_team: str
    poly_opponent: str
    poly_prob: float
    poly_american: int
    book: str
    book_american: int
    book_prob: float
    edge_bps: int
    polymarket_url: Optional[str]


# -----------------------------
# Helpers
# -----------------------------


def make_session() -> requests.Session:
    session = requests.Session()
    retries = Retry(
        total=3,
        backoff_factor=0.6,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=frozenset(["GET"]),
        raise_on_status=False,
    )
    session.mount("https://", HTTPAdapter(max_retries=retries))
    session.headers.update({"User-Agent": USER_AGENT})
    return session


SESSION = make_session()


def http_get(url: str, params: Optional[dict] = None, timeout: int = 30) -> Any:
    response = SESSION.get(
        url,
        params=params,
        timeout=timeout,
    )
    if response.status_code >= 400:
        raise RuntimeError(
            f"HTTP {response.status_code} GET {url} params={params} timeout={timeout}: "
            f"{response.text[:200]}"
        )
    return response.json()


def implied_prob_from_american(odds: int) -> float:
    if odds < 0:
        return (-odds) / ((-odds) + 100.0)
    return 100.0 / (odds + 100.0)


def american_from_prob(probability: float) -> int:
    probability = max(1e-6, min(1.0 - 1e-6, probability))
    if probability >= 0.5:
        return int(round(-100.0 * probability / (1.0 - probability)))
    return int(round(100.0 * (1.0 - probability) / probability))


def to_dt(value: Optional[str]) -> Optional[datetime]:
    if not value:
        return None
    parsed = dtparser.parse(value)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def normalize_team(name: str) -> str:
    cleaned = name.lower().strip()
    cleaned = re.sub(r"[^a-z0-9\s]", " ", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    tokens = [token for token in cleaned.split(" ") if token not in TEAM_STRIP_TOKENS]
    return " ".join(tokens)


def is_probable_team_name(name: str) -> bool:
    cleaned = normalize_team(name)
    if not cleaned:
        return False
    if re.search(r"\b[+-]?\d+(?:\.\d+)?\b", cleaned):
        return False
    if any(token in cleaned for token in ("spread", "total", "points", "pts", "over", "under")):
        return False
    banned = {
        "yes",
        "no",
        "over",
        "under",
        "home",
        "away",
        "draw",
        "tie",
        "true",
        "false",
    }
    tokens = set(cleaned.split())
    return not tokens.issubset(banned)


def discord_post(webhook_url: str, content: str) -> None:
    payload = {"content": content}
    response = SESSION.post(webhook_url, json=payload, timeout=30)
    if response.status_code == 429:
        retry_after = response.headers.get("Retry-After")
        delay = None
        if retry_after:
            try:
                delay = float(retry_after)
            except ValueError:
                delay = None
        if delay is None:
            try:
                data = response.json()
                delay = float(data.get("retry_after")) if isinstance(data, dict) else None
            except (ValueError, TypeError):
                delay = None
        if delay is not None:
            delay = max(0.0, min(delay, 30.0))
            time.sleep(delay)
            response = SESSION.post(webhook_url, json=payload, timeout=30)
    response.raise_for_status()


def fit_discord_message(lines: List[str]) -> str:
    message = ""
    for line in lines:
        if len(message) + len(line) + 1 > DISCORD_MAX_CHARS:
            message = message.rstrip()
            if not message.endswith("_Output truncated to fit Discord limits._"):
                message = f"{message}\n_Output truncated to fit Discord limits._"
            return message
        message += line + "\n"
    return message.rstrip()


def parse_json_list(value: Optional[object]) -> List[object]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, str):
        trimmed = value.strip()
        if not trimmed:
            return []
        try:
            data = json.loads(trimmed)
            if isinstance(data, list):
                return data
        except Exception:
            return []
    return []


def parse_sport_keys(raw: Optional[str]) -> List[str]:
    if not raw:
        return DEFAULT_SPORT_KEYS
    cleaned = raw.strip().lower()
    if not cleaned:
        return DEFAULT_SPORT_KEYS
    if cleaned in ("all", "*"):
        return DEFAULT_SPORT_KEYS
    return [item.strip() for item in cleaned.split(",") if item.strip()]


# -----------------------------
# Polymarket fetch + parse
# -----------------------------


def fetch_polymarket_markets(active_only: bool = True, limit: int = 200) -> List[dict]:
    markets: List[dict] = []
    offset = 0
    while True:
        params = {
            "active": "true" if active_only else "false",
            "closed": "false",
            "limit": str(limit),
            "offset": str(offset),
        }
        if GAME_BETS_TAG_ID:
            params["tag_id"] = str(GAME_BETS_TAG_ID)
        batch = http_get(f"{GAMMA_BASE}/markets", params=params)
        if not isinstance(batch, list) or not batch:
            break
        markets.extend(batch)
        if len(batch) < limit:
            break
        offset += limit
        time.sleep(0.2)
    return markets


def is_candidate_moneyline_market(market: dict) -> bool:
    outcomes = parse_json_list(market.get("outcomes"))
    prices = parse_json_list(market.get("outcomePrices"))
    if len(outcomes) != 2 or len(prices) != 2:
        return False
    if not all(is_probable_team_name(str(outcome)) for outcome in outcomes):
        return False

    return True


def build_polymarket_candidates(markets: List[dict]) -> List[PolyMarket]:
    candidates: List[PolyMarket] = []
    for market in markets:
        if not is_candidate_moneyline_market(market):
            continue

        outcomes = parse_json_list(market.get("outcomes"))
        prices = parse_json_list(market.get("outcomePrices"))
        if len(outcomes) != 2 or len(prices) != 2:
            continue

        start_time = to_dt(
            market.get("eventStartTime")
            or market.get("gameStartTime")
            or market.get("startDateIso")
            or market.get("startDate")
            or market.get("startTime")
        )

        market_id = str(market.get("id") or market.get("marketId") or "")
        if not market_id:
            continue

        try:
            prob_a = float(prices[0])
            prob_b = float(prices[1])
        except (TypeError, ValueError):
            continue
        if not (0 < prob_a < 1 and 0 < prob_b < 1):
            continue
        if not 0.95 <= (prob_a + prob_b) <= 1.05:
            continue

        url = None
        slug = market.get("slug")
        if slug:
            url = f"https://polymarket.com/market/{slug}"

        candidates.append(
            PolyMarket(
                market_id=market_id,
                start_time=start_time,
                team_a=str(outcomes[0]),
                team_b=str(outcomes[1]),
                prob_a=prob_a,
                prob_b=prob_b,
                url=url,
            )
        )
    return candidates


# -----------------------------
# Sportsbook odds fetch
# -----------------------------


def fetch_sportsbook_events(sport_key: str) -> List[BookEvent]:
    api_key = os.getenv("ODDS_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("Missing ODDS_API_KEY secret/env var.")

    base = "https://api.the-odds-api.com/v4/sports"
    params = {
        "apiKey": api_key,
        "regions": "us",
        "markets": "h2h",
        "oddsFormat": "american",
        "dateFormat": "iso",
    }

    data = http_get(f"{base}/{sport_key}/odds", params=params)
    if not isinstance(data, list):
        return []

    events: List[BookEvent] = []
    for event in data:
        commence = to_dt(event.get("commence_time"))
        if not commence:
            continue

        home = event.get("home_team") or ""
        away = event.get("away_team") or ""
        if not home or not away:
            continue

        best_home: Optional[BookLine] = None
        best_away: Optional[BookLine] = None

        for bookmaker in event.get("bookmakers", []) or []:
            book_name = bookmaker.get("title") or bookmaker.get("key") or "book"
            for market in bookmaker.get("markets", []) or []:
                if market.get("key") != "h2h":
                    continue
                outcomes = market.get("outcomes", []) or []
                for outcome in outcomes:
                    name = outcome.get("name")
                    price = outcome.get("price")
                    if name is None or price is None:
                        continue
                    try:
                        price = int(price)
                    except (TypeError, ValueError):
                        continue

                    if name == home:
                        if best_home is None or is_better_american(price, best_home.odds_american):
                            best_home = BookLine(book=book_name, odds_american=price)
                    elif name == away:
                        if best_away is None or is_better_american(price, best_away.odds_american):
                            best_away = BookLine(book=book_name, odds_american=price)

        events.append(
            BookEvent(
                sport_key=sport_key,
                league=SPORT_KEYS.get(sport_key, sport_key),
                start_time=commence,
                home=home,
                away=away,
                best_home=best_home,
                best_away=best_away,
            )
        )
    return events


def is_better_american(new: int, old: int) -> bool:
    return implied_prob_from_american(new) < implied_prob_from_american(old)


# -----------------------------
# Matching + scoring
# -----------------------------


def token_overlap_score(a: str, b: str) -> float:
    tokens_a = {token for token in normalize_team(a).split() if token}
    tokens_b = {token for token in normalize_team(b).split() if token}
    if not tokens_a or not tokens_b:
        return 0.0
    overlap = tokens_a & tokens_b
    return len(overlap) / min(len(tokens_a), len(tokens_b))


def mapping_scores(poly: PolyMarket, event: BookEvent) -> Tuple[float, float]:
    direct = token_overlap_score(poly.team_a, event.home) + token_overlap_score(poly.team_b, event.away)
    swapped = token_overlap_score(poly.team_a, event.away) + token_overlap_score(poly.team_b, event.home)
    return direct, swapped


def best_match_event(poly: PolyMarket, books: List[BookEvent], min_score: float) -> Optional[BookEvent]:
    poly_dt = poly.start_time
    best_event: Optional[BookEvent] = None
    best_score = 0.0
    best_time_delta = float("inf")

    for event in books:
        direct_score, swapped_score = mapping_scores(poly, event)
        score = max(direct_score, swapped_score)
        if score < min_score:
            continue
        time_delta = 0.0
        if poly_dt:
            time_delta = abs((poly_dt - event.start_time).total_seconds())
            if time_delta > 6 * 3600:
                continue
        if score > best_score or (score == best_score and time_delta < best_time_delta):
            best_score = score
            best_time_delta = time_delta
            best_event = event

    return best_event


def match_polymarket_to_books(poly: PolyMarket, books: List[BookEvent]) -> Optional[BookEvent]:
    return best_match_event(poly, books, min_score=MATCH_MIN_SCORE)


def compute_match_results(
    polys: List[PolyMarket],
    book_events: List[BookEvent],
    min_edge_bps: int,
) -> List[MatchResult]:
    results: List[MatchResult] = []

    for market in polys:
        event = match_polymarket_to_books(market, book_events)
        if not event:
            continue

        direct_score, swapped_score = mapping_scores(market, event)
        if direct_score >= swapped_score:
            poly_prob_team = market.prob_a
            team = event.home
            opponent = event.away
            book_line = event.best_home
            poly_team = market.team_a
            poly_opponent = market.team_b
        else:
            poly_prob_team = market.prob_b
            team = event.away
            opponent = event.home
            book_line = event.best_away
            poly_team = market.team_b
            poly_opponent = market.team_a

        if not book_line:
            continue

        book_prob_raw = implied_prob_from_american(book_line.odds_american)

        if event.best_home and event.best_away:
            prob_home = implied_prob_from_american(event.best_home.odds_american)
            prob_away = implied_prob_from_american(event.best_away.odds_american)
            denom = prob_home + prob_away
            if denom > 0:
                book_prob = prob_home / denom if team == event.home else prob_away / denom
            else:
                book_prob = book_prob_raw
        else:
            book_prob = book_prob_raw

        edge = book_prob - poly_prob_team
        edge_bps = int(round(edge * 10000))

        if edge_bps < min_edge_bps:
            continue

        results.append(
            MatchResult(
                league=event.league,
                start_time=event.start_time,
                team=team,
                opponent=opponent,
                poly_team=poly_team,
                poly_opponent=poly_opponent,
                poly_prob=poly_prob_team,
                poly_american=american_from_prob(poly_prob_team),
                book=book_line.book,
                book_american=book_line.odds_american,
                book_prob=book_prob,
                edge_bps=edge_bps,
                polymarket_url=market.url,
            )
        )

    results.sort(key=lambda result: result.edge_bps, reverse=True)
    return results


def format_results_for_discord(results: List[MatchResult], top_n: int) -> List[str]:
    lines: List[str] = []
    now = datetime.now(timezone.utc).astimezone(ZoneInfo("America/New_York")).strftime("%Y-%m-%d %I:%M %p ET")
    lines.append(f"**Polymarket vs Sportsbook (Moneyline) — Top {top_n} — {now}**")
    lines.append("Sorted by largest **book - Polymarket** implied-prob edge.")
    lines.append("Showing only edges where **book > poly**.")
    lines.append("")

    if not results:
        lines.append("_No matched markets found (or no edges above threshold)._")
        return lines

    for idx, result in enumerate(results[:top_n], start=1):
        edge_pct = result.edge_bps / 100.0
        sign = "+" if edge_pct >= 0 else ""
        start_local = result.start_time.astimezone(ZoneInfo("America/New_York")).strftime("%b %d %I:%M %p ET")

        poly_pct = result.poly_prob * 100.0
        book_pct = result.book_prob * 100.0

        url_part = f" — {result.polymarket_url}" if result.polymarket_url else ""

        lines.append(f"**{idx}. {result.league} — {result.poly_team} vs {result.poly_opponent}** ({start_local})")
        lines.append(
            f"- Poly: {poly_pct:.1f}% ({result.poly_american:+d})"
            f" | Book ({result.book}): {book_pct:.1f}% ({result.book_american:+d})"
            f" | **Edge (book - poly):** {sign}{edge_pct:.2f}%{url_part}"
        )
        lines.append("")

    return lines


# -----------------------------
# Main
# -----------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sport", default=None, help="Sportsbook sport key or comma list")
    parser.add_argument("--top", type=int, default=DEFAULT_TOP_N, help="How many results to post")
    parser.add_argument(
        "--min-edge-bps",
        type=int,
        default=DEFAULT_MIN_EDGE_BPS,
        help="Filter edges below this value (bps)",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print output only; do not post to Discord")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    raw_sports = args.sport or os.getenv("SPORT_KEYS")
    sport_keys = parse_sport_keys(raw_sports)

    discord_webhook = os.getenv("DISCORD_WEBHOOK_URL", "").strip()
    if not discord_webhook and not args.dry_run:
        print("ERROR: Missing DISCORD_WEBHOOK_URL secret/env var.", file=sys.stderr)
        return 2

    book_events: List[BookEvent] = []
    for sport_key in sport_keys:
        book_events.extend(fetch_sportsbook_events(sport_key))

    raw_markets = fetch_polymarket_markets(active_only=True, limit=200)
    candidates = build_polymarket_candidates(raw_markets)

    results = compute_match_results(candidates, book_events, args.min_edge_bps)
    print(
        f"book_events={len(book_events)} poly_candidates={len(candidates)} "
        f"matched_edges={len(results)}",
        file=sys.stderr,
    )

    lines = format_results_for_discord(results, args.top)
    message = fit_discord_message(lines)

    if args.dry_run:
        print(message)
        return 0

    discord_post(discord_webhook, message)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
