"""Kalshi Sports Value Scan

Compares Kalshi *lines* (moneyline, spread, total) vs sportsbook consensus from The Odds API.

Key behavior:
- We consider both opening sides: BUY YES or BUY NO.
- Opening a position on Kalshi has a flat per-contract fee (default +3 cents).
  We incorporate that by adding +fee_cents to the Kalshi buy price before converting to probability.

This scanner is intentionally separate from the Polymarket workflows.
"""

from __future__ import annotations

import base64
import csv
import json
import os
import re
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple
from urllib.parse import urlparse

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding

from oddsapi_lines import (
    build_event_index,
    env_min_books,
    env_sport_keys,
    fair_prob_h2h,
    fair_prob_spread,
    fair_prob_total,
    fetch_all_sports_events,
    normalize_team_name,
)


# -----------------------------
# Env helpers
# -----------------------------

def env_int(name: str, default: int) -> int:
    v = os.getenv(name)
    if v is None or not str(v).strip():
        return default
    try:
        return int(str(v).strip())
    except Exception:
        return default


def env_float(name: str, default: float) -> float:
    v = os.getenv(name)
    if v is None or not str(v).strip():
        return default
    try:
        return float(str(v).strip())
    except Exception:
        return default


def env_str(name: str, default: str) -> str:
    v = os.getenv(name)
    if v is None:
        return default
    s = str(v).strip()
    return s if s else default


# -----------------------------
# Kalshi client
# -----------------------------

# Default to documented trade-api/v2 base; allow override via env for safety.
# If you set only the host (no /trade-api/v2), we will append it automatically.
KALSHI_BASE = env_str("KALSHI_BASE", "https://api.elections.kalshi.com/trade-api/v2")
USER_AGENT = env_str(
    "KALSHI_USER_AGENT",
    "kalshi-sports-value/0.1 (+https://github.com/ndaly111/PolymarketAlerts)",
)


def _session() -> requests.Session:
    s = requests.Session()
    retries = Retry(
        total=4,
        backoff_factor=0.6,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=("GET",),
    )
    s.mount("https://", HTTPAdapter(max_retries=retries))
    s.headers.update({"User-Agent": USER_AGENT})
    return s


def _kalshi_base_url() -> str:
    base = str(KALSHI_BASE or "").strip().rstrip("/")
    if not base:
        base = "https://api.elections.kalshi.com/trade-api/v2"
    # If user provides only host, normalize to include /trade-api/v2.
    if "/trade-api/" not in base:
        base = base + "/trade-api/v2"
    return base


@lru_cache(maxsize=1)
def _kalshi_private_key():
    pem = os.getenv("KALSHI_PRIVATE_KEY") or ""
    pem = pem.strip()
    if not pem:
        raise RuntimeError("Missing KALSHI_PRIVATE_KEY (PEM).")
    return serialization.load_pem_private_key(pem.encode("utf-8"), password=None)


def _kalshi_signed_headers(method: str, full_path: str) -> Dict[str, str]:
    """
    Kalshi requires signed requests:
      message = timestamp_ms + method + path
      headers: KALSHI-ACCESS-KEY, KALSHI-ACCESS-SIGNATURE, KALSHI-ACCESS-TIMESTAMP
    """
    key_id = (os.getenv("KALSHI_KEY_ID") or os.getenv("KALSHI_API_KEY_ID") or "").strip()
    if not key_id:
        raise RuntimeError("Missing KALSHI_KEY_ID (or KALSHI_API_KEY_ID).")

    # Kalshi uses millisecond timestamps in signing examples.
    ts = str(int(time.time() * 1000))
    m = method.upper().strip()
    path = str(full_path).strip()
    if not path.startswith("/"):
        path = "/" + path

    msg = (ts + m + path).encode("utf-8")
    priv = _kalshi_private_key()
    sig = priv.sign(
        msg,
        padding.PSS(
            mgf=padding.MGF1(hashes.SHA256()),
            salt_length=padding.PSS.DIGEST_LENGTH,
        ),
        hashes.SHA256(),
    )
    return {
        "KALSHI-ACCESS-KEY": key_id,
        "KALSHI-ACCESS-SIGNATURE": base64.b64encode(sig).decode("ascii"),
        "KALSHI-ACCESS-TIMESTAMP": ts,
    }


def kalshi_get(session: requests.Session, path: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    base_url = _kalshi_base_url()
    url = f"{base_url}{path}"

    # Sign the *path* (no querystring), including the /trade-api/v2 prefix.
    prefix = urlparse(base_url).path.rstrip("/")
    full_path = prefix + (path if str(path).startswith("/") else f"/{path}")
    headers = _kalshi_signed_headers("GET", full_path)

    r = session.get(url, params=params or {}, headers=headers, timeout=30)
    r.raise_for_status()
    return r.json()


def list_series(
    session: requests.Session,
    category: Optional[str] = None,
    include_volume: bool = False,
    include_product_metadata: bool = False,
) -> List[Dict[str, Any]]:
    """
    Fetch series from /series.

    IMPORTANT: Kalshi Trade API v2 /series does not reliably support a `category` query param.
    Passing `category=` can yield an empty response even though sports series exist. The
    correct approach is to fetch normally and filter by the returned `series[i].category`
    field locally.
    """
    # Keep paginated and bounded so it remains quick/reliable in Actions.
    max_pages = 10
    cursor: Optional[str] = None
    out: List[Dict[str, Any]] = []

    want_cat = (category or "").strip()
    want_cat_l = want_cat.lower()
    use_category_param = bool(want_cat)

    def _next_cursor(d: Dict[str, Any]) -> Optional[str]:
        for k in ("cursor", "next_cursor", "nextCursor", "next_page_token", "nextPageToken"):
            v = d.get(k)
            if isinstance(v, str) and v.strip():
                return v.strip()
        pag = d.get("pagination")
        if isinstance(pag, dict):
            for k in ("cursor", "next_cursor", "nextCursor", "next_page_token", "nextPageToken"):
                v = pag.get(k)
                if isinstance(v, str) and v.strip():
                    return v.strip()
        return None

    for _ in range(max_pages):
        params: Dict[str, Any] = {"limit": 200}
        if cursor:
            params["cursor"] = cursor

        # Prefer server-side filtering (documented), but fall back if it returns empty on page 1.
        params["include_volume"] = str(include_volume).lower()
        params["include_product_metadata"] = str(include_product_metadata).lower()
        if use_category_param:
            params["category"] = want_cat

        data = kalshi_get(session, "/series", params)
        items = data.get("series") or []
        if not isinstance(items, list) or not items:
            if use_category_param and not cursor:
                use_category_param = False
                cursor = None
                continue
            break

        for s in items:
            if not isinstance(s, dict):
                continue
            if want_cat and str(s.get("category") or "").strip().lower() != want_cat_l:
                continue
            out.append(s)

        cursor = _next_cursor(data)
        if not cursor:
            break
        time.sleep(0.10)

    return out


def list_markets_for_series(
    session: requests.Session,
    series_ticker: str,
    status: str = "open",
    max_close_ts: Optional[int] = None,
    limit: int = 200,
    mve_filter: str = "exclude",
) -> List[Dict[str, Any]]:
    params: Dict[str, Any] = {"series_ticker": series_ticker, "limit": limit, "mve_filter": mve_filter}
    # Some Kalshi filters can be picky about mixing status with close-ts bounds.
    # If we're bounding close time, omit status in the API call and filter locally.
    if status and max_close_ts is None:
        params["status"] = status
    if max_close_ts is not None:
        params["max_close_ts"] = int(max_close_ts)

    markets: List[Dict[str, Any]] = []
    cursor = None
    for _ in range(50):
        if cursor:
            params["cursor"] = cursor
        data = kalshi_get(session, "/markets", params=params)
        markets.extend(data.get("markets", []) or [])
        cursor = data.get("cursor")
        if not cursor:
            break
        time.sleep(0.12)
    return markets


def get_event(session: requests.Session, event_ticker: str) -> Optional[Dict[str, Any]]:
    try:
        return kalshi_get(session, f"/events/{event_ticker}")
    except Exception:
        return None


# -----------------------------
# Parsing helpers
# -----------------------------

TOTAL_RE = re.compile(r"\b(over|under)\s*([0-9]+(?:\.[0-9]+)?)\b", re.IGNORECASE)
SPREAD_RE = re.compile(
    r"^(?P<team>.+?)\s*(?P<spread>[+-]?\d+(?:\.\d+)?)\s*(?:pts?|points?)?\s*$",
    re.IGNORECASE,
)


def parse_matchup(title: str) -> Optional[Tuple[str, str]]:
    """Extract team A and team B from a title like 'Chicago at Brooklyn' or 'A vs B'."""
    t = " ".join(str(title).replace("@", " at ").split())
    m = re.search(r"^(.*?)\s+(?:at|vs\.?|v\.?|versus)\s+(.*?)(?:\s*:\s*.*)?$", t, re.IGNORECASE)
    if not m:
        return None
    a = m.group(1).strip()
    b = m.group(2).strip()
    if not a or not b:
        return None
    return a, b


def parse_line_from_subtitle(line_type: str, subtitle: str) -> Optional[Dict[str, Any]]:
    s = " ".join(str(subtitle or "").split())
    if not s:
        return None

    lt = line_type
    if lt == "total":
        m = TOTAL_RE.search(s)
        if not m:
            return None
        side = m.group(1).lower()
        pts = float(m.group(2))
        return {"side": side, "points": pts, "label": f"{side.title()} {pts:g}"}

    if lt == "spread":
        # Common forms: "Brooklyn -1", "BKN -1.5"
        m = SPREAD_RE.match(s)
        if not m:
            return None
        team = m.group("team").strip()
        spread = float(m.group("spread"))
        return {"team": team, "spread": spread, "label": f"{team} {spread:+g}"}

    if lt == "h2h":
        # Common forms: "Brooklyn", "Brooklyn wins"
        team = s
        team = re.sub(r"\b(wins|win|to win)\b", "", team, flags=re.IGNORECASE).strip()
        if not team:
            return None
        return {"team": team, "label": team}

    return None


def infer_line_type(series_ticker: str, market_ticker: str, market_title: str) -> Optional[str]:
    hay = f"{series_ticker} {market_ticker} {market_title}".upper()
    if "TOTAL" in hay:
        return "total"
    if "SPREAD" in hay:
        return "spread"
    # Many moneyline series use GAME
    if "GAME" in hay or "ML" in hay or "MONEY" in hay:
        return "h2h"
    return None


def cents_to_prob(cents: Optional[Any]) -> Optional[float]:
    if cents is None:
        return None
    try:
        c = float(cents)
    except Exception:
        return None
    if c < 0:
        return None
    # Kalshi prices are in cents; clamp to [0, 1] to avoid weirdness if API ever returns >100.
    return max(0.0, min(1.0, c / 100.0))


def best_buy_probs(market: Dict[str, Any]) -> Tuple[Optional[float], Optional[float]]:
    """Return (yes_buy_prob, no_buy_prob) based on ask prices.

    Prefer yes_ask/no_ask if present. If missing, derive from opposing bids:
      yes_ask ~= 1 - no_bid
      no_ask  ~= 1 - yes_bid
    """
    yes_ask = cents_to_prob(market.get("yes_ask"))
    no_ask = cents_to_prob(market.get("no_ask"))

    if yes_ask is None:
        no_bid = cents_to_prob(market.get("no_bid"))
        if no_bid is not None:
            yes_ask = max(0.0, 1.0 - no_bid)

    if no_ask is None:
        yes_bid = cents_to_prob(market.get("yes_bid"))
        if yes_bid is not None:
            no_ask = max(0.0, 1.0 - yes_bid)

    return yes_ask, no_ask


def name_similarity(a: str, b: str) -> float:
    """Deterministic similarity score in [0,1].

    We strip common sport/league words so titles like 'Jacksonville Pro Football' still
    match 'Jacksonville Jaguars' reasonably well.
    """

    na = normalize_team_name(a)
    nb = normalize_team_name(b)
    if not na or not nb:
        return 0.0
    if na == nb:
        return 1.0
    if na in nb or nb in na:
        return 0.90

    stop = {
        "pro",
        "professional",
        "football",
        "basketball",
        "baseball",
        "hockey",
        "game",
        "match",
        "season",
        "playoffs",
        "postseason",
        "nfl",
        "nba",
        "mlb",
        "nhl",
        "ncaaf",
        "ncaab",
    }

    ta = [t for t in na.split() if t not in stop]
    tb = [t for t in nb.split() if t not in stop]
    if not ta or not tb:
        ta = na.split()
        tb = nb.split()

    sa = set(ta)
    sb = set(tb)
    inter = len(sa & sb)
    union = len(sa | sb)
    if union == 0:
        return 0.0
    return inter / union


def rank_event_matches(
    events: List[Dict[str, Any]],
    team_a: str,
    team_b: str,
    top_k: int = 5,
) -> List[Tuple[float, Dict[str, Any]]]:
    """
    Return top_k candidate Odds API events sorted by descending match score.
    Score is in [0,2] (sum of two team similarity scores).
    """
    scored: List[Tuple[float, Dict[str, Any]]] = []
    for e in events:
        home = e.get("home_team")
        away = e.get("away_team")
        if not home or not away:
            continue
        s1 = name_similarity(team_a, str(home)) + name_similarity(team_b, str(away))
        s2 = name_similarity(team_a, str(away)) + name_similarity(team_b, str(home))
        s = max(s1, s2)
        scored.append((float(s), e))
    scored.sort(key=lambda x: x[0], reverse=True)
    return scored[: max(1, int(top_k))]


def best_event_match(events: List[Dict[str, Any]], team_a: str, team_b: str) -> Optional[Dict[str, Any]]:
    """Find the best odds event matching the two team strings."""
    ranked = rank_event_matches(events, team_a, team_b, top_k=1)
    if not ranked:
        return None
    best_score, best = ranked[0]
    # Require both teams to be at least kinda close.
    if best is None or float(best_score) < 1.30:
        return None
    return best


def kalshi_market_url(market_ticker: str) -> str:
    return f"https://kalshi.com/markets/{str(market_ticker).lower()}"


# -----------------------------
# Output + Discord
# -----------------------------


def post_discord(webhook_url: str, text: str) -> None:
    if not webhook_url:
        return
    r = requests.post(webhook_url, json={"content": text[:1900]}, timeout=20)
    r.raise_for_status()


@dataclass
class Row:
    fee_cents: int
    edge: float
    side_to_buy: str
    all_in_buy_prob: float
    fair_prob_side: float
    fair_prob_yes: float
    yes_buy_prob: Optional[float]
    no_buy_prob: Optional[float]
    market_ticker: str
    event_title: str
    market_title: str
    line_type: str
    line_label: str
    books_used: int
    url: str


def to_dict(r: Row) -> Dict[str, Any]:
    return {
        "fee_cents": r.fee_cents,
        "edge": r.edge,
        "side_to_buy": r.side_to_buy,
        "all_in_buy_prob": r.all_in_buy_prob,
        "fair_prob_side": r.fair_prob_side,
        "fair_prob_yes": r.fair_prob_yes,
        "yes_buy_prob": r.yes_buy_prob,
        "no_buy_prob": r.no_buy_prob,
        "market_ticker": r.market_ticker,
        "event_title": r.event_title,
        "market_title": r.market_title,
        "line_type": r.line_type,
        "line_label": r.line_label,
        "books_used": r.books_used,
        "url": r.url,
    }


FIELDNAMES = [
    "fee_cents",
    "edge",
    "side_to_buy",
    "all_in_buy_prob",
    "fair_prob_side",
    "fair_prob_yes",
    "yes_buy_prob",
    "no_buy_prob",
    "market_ticker",
    "event_title",
    "market_title",
    "line_type",
    "line_label",
    "books_used",
    "url",
]


# -----------------------------
# Main scan
# -----------------------------

def _write_json(path: str, obj: Any) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


def _write_jsonl(path: str, items: List[Dict[str, Any]]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for it in items:
            f.write(json.dumps(it, ensure_ascii=False))
            f.write("\n")


def _copy_text(src: str, dst: str) -> None:
    try:
        Path(dst).write_text(Path(src).read_text(encoding="utf-8"), encoding="utf-8")
    except Exception:
        # best-effort; don't break run on debug copy issues
        pass


def scan() -> int:
    # Output directory (create immediately so artifacts always have *something* if we can write it)
    outdir = env_str("OUTDIR", "out")
    os.makedirs(outdir, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%SZ")

    debug_max = env_int("KALSHI_DEBUG_MAX", 50)  # keep artifacts small; bump if needed

    fee_cents = env_int("KALSHI_BUY_FEE_CENTS", 3)
    fee_prob = fee_cents / 100.0

    lookahead_hours = env_int("KALSHI_LOOKAHEAD_HOURS", 72)
    max_close_ts = int(time.time()) + lookahead_hours * 3600

    min_edge = env_float("MIN_EDGE", 0.03)
    top_n = env_int("TOP_N", 25)

    odds_regions = env_str("ODDS_REGIONS", "us")
    min_books = env_min_books()

    run_meta: Dict[str, Any] = {
        "stamp_utc": stamp,
        "fee_cents": fee_cents,
        "lookahead_hours": lookahead_hours,
        "min_edge": min_edge,
        "top_n": top_n,
        "min_books": min_books,
        "odds_regions": odds_regions,
        "sport_keys": env_sport_keys(),
        "kalshi_base": _kalshi_base_url(),
        "status": "starting",
    }

    meta_path = os.path.join(outdir, f"kalshi_run_meta_{stamp}.json")
    meta_latest = os.path.join(outdir, "kalshi_run_meta_latest.json")

    series_override = env_str("KALSHI_SERIES_TICKERS", "").strip()
    series_category = env_str("KALSHI_SERIES_CATEGORY", "Sports")
    max_series = env_int("KALSHI_MAX_SERIES_TICKERS", 60)

    include_pat = env_str(
        "KALSHI_SPORTS_SERIES_INCLUDE_REGEX",
        r"(GAME|MATCH|MONEYLINE|SPREAD|TOTAL|OVER|UNDER|SET|DISTANCE)",
    )
    exclude_pat = env_str(
        "KALSHI_SPORTS_SERIES_EXCLUDE_REGEX",
        r"(PROP|PLAYER|MVP|CHAMP|TROPHY|AWARD|SEASON|FUTURE|FUTURES|WINS|EXACT|COMBO|PARLAY|PACK|SALE|COACH|TOP\s*\d|MVE)",
    )
    include_re = re.compile(include_pat, re.IGNORECASE)
    exclude_re = re.compile(exclude_pat, re.IGNORECASE)

    run_meta["series_category"] = series_category
    run_meta["series_include_regex"] = include_pat
    run_meta["series_exclude_regex"] = exclude_pat
    run_meta["max_series_tickers"] = max_series

    # Odds events (all at once)
    sport_keys = env_sport_keys()
    odds_events = fetch_all_sports_events(sport_keys, markets="h2h,spreads,totals", regions=odds_regions)

    if not odds_events:
        run_meta["status"] = "error_no_odds_events"
        _write_json(meta_path, run_meta)
        _copy_text(meta_path, meta_latest)
        print("No sportsbook events returned from The Odds API (check THE_ODDS_API / quota).")
        return 2

    # Fail fast if Kalshi auth isn't present (trade-api/v2 requires signed requests).
    has_key_id = bool((os.getenv("KALSHI_KEY_ID") or os.getenv("KALSHI_API_KEY_ID") or "").strip())
    if not (has_key_id and os.getenv("KALSHI_PRIVATE_KEY")):
        run_meta["status"] = "error_missing_kalshi_auth"
        _write_json(meta_path, run_meta)
        _copy_text(meta_path, meta_latest)
        print("Missing Kalshi auth env vars: KALSHI_KEY_ID (or KALSHI_API_KEY_ID) and/or KALSHI_PRIVATE_KEY")
        return 2

    # Kalshi markets
    session = _session()
    series: List[str] = []
    discovered_series: List[Dict[str, Any]] = []

    if series_override:
        series = [s.strip() for s in series_override.split(",") if s.strip()]
    else:
        # Discover sports series tickers and keep those that match the include regex.
        discovered_series = list_series(session, category=series_category)
        if not discovered_series and series_category:
            discovered_series = list_series(session, category=None)
            if discovered_series:
                run_meta["series_category_fallback"] = True

        categories: Dict[str, int] = {}
        for s in discovered_series:
            c = str(s.get("category") or "").strip() or "(missing)"
            categories[c] = categories.get(c, 0) + 1
        run_meta["discovered_series_count"] = len(discovered_series)
        run_meta["discovered_category_counts"] = categories
        for s in discovered_series:
            ticker = str(s.get("ticker") or "").strip()
            title = str(s.get("title") or "")
            if not ticker:
                continue
            if exclude_re.search(ticker) or exclude_re.search(title):
                continue
            if include_re.search(ticker) or include_re.search(title):
                series.append(ticker)

        if series:
            run_meta["series_discovery_mode"] = "include_regex"
        else:
            # If include regex was too strict, fall back to anything not excluded.
            for s in discovered_series:
                ticker = str(s.get("ticker") or "").strip()
                title = str(s.get("title") or "")
                if not ticker:
                    continue
                if exclude_re.search(ticker) or exclude_re.search(title):
                    continue
                series.append(ticker)
            if series:
                run_meta["series_discovery_mode"] = "fallback_not_excluded"

    series = sorted(set(series))
    if (not series_override) and len(series) > max_series:
        series = series[:max_series]
        run_meta["series_truncated_to"] = max_series
    if not series:
        run_meta["status"] = "error_no_series"

        debug = {
            "stamp_utc": stamp,
            "base": _kalshi_base_url(),
            "requested_series_category": series_category,
            "series_include_regex": include_re.pattern,
            "series_exclude_regex": exclude_re.pattern,
            "discovered_series_count": len(discovered_series),
            "discovered_category_counts": run_meta.get("discovered_category_counts", {}),
            "discovered_series_sample": [
                {
                    "ticker": s.get("ticker"),
                    "title": s.get("title"),
                    "category": s.get("category"),
                    "sub_category": s.get("sub_category") or s.get("subcategory"),
                    "active": s.get("active"),
                }
                for s in discovered_series[:50]
            ],
        }
        debug_path = os.path.join(outdir, f"kalshi_series_discovery_debug_{stamp}.json")
        debug_latest = os.path.join(outdir, "kalshi_series_discovery_debug_latest.json")
        _write_json(debug_path, debug)
        _write_json(debug_latest, debug)

        _write_json(meta_path, run_meta)
        _copy_text(meta_path, meta_latest)
        print(
            "No Kalshi series tickers selected. See "
            f"{debug_latest} for discovery details. "
            "Try KALSHI_SERIES_CATEGORY=Sports, or set KALSHI_SERIES_TICKERS=... to override."
        )
        return 2

    run_meta["series_count"] = len(series)
    run_meta["status"] = "running"
    print(f"Using {len(series)} Kalshi series tickers. Lookahead: {lookahead_hours}h. Fee: +{fee_cents}c")

    # Cache event titles by ticker (avoid repeated calls)
    event_title_cache: Dict[str, str] = {}

    rows: List[Row] = []

    debug_matched: List[Dict[str, Any]] = []
    debug_unmatched: List[Dict[str, Any]] = []
    series_errors = 0

    for st in series:
        try:
            markets = list_markets_for_series(session, st, status="open", max_close_ts=max_close_ts)
        except Exception as exc:
            series_errors += 1
            if series_errors <= 3:
                print(f"[WARN] list_markets_for_series failed for {st}: {type(exc).__name__}: {exc}")
            continue

        # If the API didn't enforce open-only (or doesn't like combining filters),
        # enforce it here without dropping markets that omit 'status'.
        filtered: List[Dict[str, Any]] = []
        for m in markets:
            s = str(m.get("status") or "").strip().lower()
            if s and s != "open":
                continue
            filtered.append(m)
        markets = filtered

        for m in markets:
            market_ticker = str(m.get("ticker") or "")
            if not market_ticker:
                continue

            event_ticker = str(m.get("event_ticker") or "")
            market_title = str(m.get("title") or "")
            yes_sub = str(m.get("yes_sub_title") or "")
            no_sub = str(m.get("no_sub_title") or "")

            lt = infer_line_type(st, market_ticker, market_title)
            if lt is None:
                continue

            # Parse the YES proposition from subtitle (preferred) else title.
            line = parse_line_from_subtitle(lt, yes_sub) or parse_line_from_subtitle(lt, market_title)
            if not line:
                continue

            # Resolve event title (to get matchup) - prefer nested event in response
            event_title = ""
            if event_ticker:
                event_title = event_title_cache.get(event_ticker, "")
                if not event_title:
                    ev = get_event(session, event_ticker)
                    if ev and isinstance(ev, dict):
                        eobj = ev.get("event") if "event" in ev else ev
                        event_title = str((eobj or {}).get("title") or "")
                    event_title_cache[event_ticker] = event_title

            if not event_title:
                # Some responses include nested event dict
                event_title = str((m.get("event") or {}).get("title") or "")

            matchup = parse_matchup(event_title) or parse_matchup(market_title)
            if not matchup:
                continue

            team_a, team_b = matchup
            ranked = rank_event_matches(odds_events, team_a, team_b, top_k=5)
            odds_event = ranked[0][1] if ranked and ranked[0][0] >= 1.30 else None
            if not odds_event:
                if len(debug_unmatched) < debug_max:
                    debug_unmatched.append(
                        {
                            "series_ticker": st,
                            "event_title": event_title or "",
                            "market_title": market_title,
                            "market_ticker": market_ticker,
                            "yes_sub_title": yes_sub,
                            "no_sub_title": no_sub,
                            "parsed_line": line,
                            "matchup": {"a": team_a, "b": team_b},
                            "top_candidates": [
                                {
                                    "score": float(s),
                                    "sport_key": e.get("sport_key"),
                                    "commence_time": e.get("commence_time"),
                                    "home_team": e.get("home_team"),
                                    "away_team": e.get("away_team"),
                                    "id": e.get("id"),
                                }
                                for (s, e) in ranked
                            ],
                            "kalshi_market": m,
                        }
                    )
                continue

            fair_yes: Optional[Dict[str, Any]] = None

            if lt == "h2h":
                fair_yes = fair_prob_h2h(odds_event, team_name=line["team"], min_books=min_books)
            elif lt == "spread":
                fair_yes = fair_prob_spread(
                    odds_event,
                    team_name=line["team"],
                    point=float(line["spread"]),
                    min_books=min_books,
                )
            elif lt == "total":
                fair_yes = fair_prob_total(
                    odds_event,
                    side=line["side"],
                    point=float(line["points"]),
                    min_books=min_books,
                )

            if not fair_yes:
                continue

            fair_prob_yes = float(fair_yes["fair_prob"])
            books_used = int(fair_yes.get("books_used") or 0)
            if books_used < min_books:
                continue

            # Kalshi buy prices (asks)
            yes_buy, no_buy = best_buy_probs(m)
            if yes_buy is None and no_buy is None:
                continue

            # apply fee to opening cost
            all_in_yes = None if yes_buy is None else min(1.0, yes_buy + fee_prob)
            all_in_no = None if no_buy is None else min(1.0, no_buy + fee_prob)

            fair_no = 1.0 - fair_prob_yes

            # Evaluate both sides (if available)
            best_side = None
            best_edge = -1e9
            best_all_in = None
            best_fair_side = None

            if all_in_yes is not None:
                e = fair_prob_yes - all_in_yes
                if e > best_edge:
                    best_edge = e
                    best_side = "YES"
                    best_all_in = all_in_yes
                    best_fair_side = fair_prob_yes

            if all_in_no is not None:
                e = fair_no - all_in_no
                if e > best_edge:
                    best_edge = e
                    best_side = "NO"
                    best_all_in = all_in_no
                    best_fair_side = fair_no

            if best_side is None or best_all_in is None or best_fair_side is None:
                continue

            if best_edge < min_edge:
                continue

            if len(debug_matched) < debug_max:
                best_score = float(ranked[0][0]) if ranked else None
                debug_matched.append(
                    {
                        "series_ticker": st,
                        "match_score": best_score,
                        "matchup": {"a": team_a, "b": team_b},
                        "kalshi_market": m,
                        "kalshi_event_title": event_title or market_title,
                        "parsed_line": line,
                        "odds_event": odds_event,
                        "fair": {
                            "fair_prob_yes": float(fair_prob_yes),
                            "fair_prob_no": float(1.0 - fair_prob_yes),
                            "books_used": books_used,
                            "min_books": min_books,
                        },
                        "kalshi_buy": {
                            "yes_buy_prob": yes_buy,
                            "no_buy_prob": no_buy,
                            "fee_cents": fee_cents,
                            "all_in_yes": all_in_yes,
                            "all_in_no": all_in_no,
                        },
                        "decision": {
                            "side_to_buy": best_side,
                            "edge": float(best_edge),
                            "min_edge": min_edge,
                        },
                    }
                )

            rows.append(
                Row(
                    fee_cents=fee_cents,
                    edge=float(best_edge),
                    side_to_buy=best_side,
                    all_in_buy_prob=float(best_all_in),
                    fair_prob_side=float(best_fair_side),
                    fair_prob_yes=float(fair_prob_yes),
                    yes_buy_prob=yes_buy,
                    no_buy_prob=no_buy,
                    market_ticker=market_ticker,
                    event_title=event_title or market_title,
                    market_title=market_title,
                    line_type=lt,
                    line_label=str(line.get("label") or ""),
                    books_used=books_used,
                    url=kalshi_market_url(market_ticker),
                )
            )

    rows.sort(key=lambda r: r.edge, reverse=True)
    rows = rows[: max(1, top_n)]

    csv_path = os.path.join(outdir, f"kalshi_value_{stamp}.csv")
    json_path = os.path.join(outdir, f"kalshi_value_{stamp}.json")
    csv_latest = os.path.join(outdir, "kalshi_value_latest.csv")
    json_latest = os.path.join(outdir, "kalshi_value_latest.json")

    matched_path = os.path.join(outdir, f"kalshi_debug_matched_{stamp}.jsonl")
    unmatched_path = os.path.join(outdir, f"kalshi_debug_unmatched_{stamp}.jsonl")
    matched_latest = os.path.join(outdir, "kalshi_debug_matched_latest.jsonl")
    unmatched_latest = os.path.join(outdir, "kalshi_debug_unmatched_latest.jsonl")

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDNAMES)
        w.writeheader()
        for r in rows:
            w.writerow(to_dict(r))

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump([to_dict(r) for r in rows], f, indent=2)

    # Stable "latest" copies (makes it easy to grab from artifacts without timestamps)
    _copy_text(csv_path, csv_latest)
    _copy_text(json_path, json_latest)

    # Debug bundles (these are the “files I’m looking for”)
    _write_jsonl(matched_path, debug_matched)
    _write_jsonl(unmatched_path, debug_unmatched)
    _copy_text(matched_path, matched_latest)
    _copy_text(unmatched_path, unmatched_latest)

    run_meta["status"] = "ok"
    run_meta["rows_written"] = len(rows)
    run_meta["debug_matched_rows"] = len(debug_matched)
    run_meta["debug_unmatched_rows"] = len(debug_unmatched)
    _write_json(meta_path, run_meta)
    _copy_text(meta_path, meta_latest)

    print(f"Wrote {len(rows)} rows to {csv_path} and {json_path}")

    webhook = env_str("DISCORD_WEBHOOK_URL", "")

    if webhook and rows:
        lines: List[str] = []
        lines.append(f"**Kalshi value scan (fee +{fee_cents}¢ on open)**")
        for i, r in enumerate(rows[: min(8, len(rows))], start=1):
            all_in_c = int(round(r.all_in_buy_prob * 100))
            fair_c = r.fair_prob_side * 100.0
            edge_pp = r.edge * 100.0
            lines.append(
                f"{i}. `{r.line_type.upper()}` **BUY {r.side_to_buy}** @ **{all_in_c}¢** (all-in) | fair {fair_c:.1f}% | edge {edge_pp:.1f}pp | books {r.books_used} | {r.event_title} | {r.line_label} | {r.url}"
            )

        post_discord(webhook, "\n".join(lines))

    # Print top results for logs
    for r in rows[: min(10, len(rows))]:
        print(
            f"{r.edge*100:6.2f}pp | BUY {r.side_to_buy:>3} | all-in {r.all_in_buy_prob*100:5.1f}c | fair {r.fair_prob_side*100:5.1f}% | {r.line_type:>6} | {r.event_title} | {r.line_label}"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(scan())
