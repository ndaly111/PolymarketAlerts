"""Kalshi Sports Value Scan

Compares Kalshi *lines* (moneyline, spread, total) vs sportsbook consensus from The Odds API.

Key behavior:
- We consider both opening sides: BUY YES or BUY NO.
- Opening a position on Kalshi has a flat per-contract fee (default +2 cents).
  We incorporate that by adding +fee_cents to the Kalshi buy price before converting to probability.

This scanner is intentionally separate from the Polymarket workflows.
"""

from __future__ import annotations

import base64
import csv
import itertools
import json
import os
import re
import time
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from zoneinfo import ZoneInfo
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple
from urllib.parse import parse_qsl, urlencode, urlparse, urlunparse

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
from draftkings_lines import fetch_all_draftkings_events
from team_ids import (
    kalshi_team_id,
    odds_event_team_ids,
    sport_prefix_from_odds_key,
)


# -----------------------------
# Env helpers
# -----------------------------

def env_int(name: str, default: int) -> int:
    v = os.getenv(name)
    if v is None or not str(v).strip():
        return default
    try:
        return int(str(v).strip().replace(",", "").replace("_", ""))
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


def env_bool(name: str, default: bool = False) -> bool:
    """Parse common truthy/falsey env values."""
    v = os.getenv(name)
    if v is None:
        return default
    s = str(v).strip().lower()
    if s in ("1", "true", "t", "yes", "y", "on"):
        return True
    if s in ("0", "false", "f", "no", "n", "off", ""):
        return False
    return default


def _parse_iso(value: str) -> Optional[datetime]:
    if not value:
        return None
    s = str(value).strip()
    # Odds API commonly uses trailing 'Z' to mean UTC
    if s.endswith("Z"):
        s = s[:-1] + "+00:00"
    try:
        dt = datetime.fromisoformat(s)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt
    except Exception:
        return None


def iso_to_ts(value: str) -> Optional[int]:
    dt = _parse_iso(value)
    return int(dt.timestamp()) if dt else None


def _normalize_epoch_seconds(value: Any) -> Optional[int]:
    """Normalize epoch timestamps that may be seconds, ms, us, or ns into seconds."""
    if value is None:
        return None
    try:
        if isinstance(value, str) and ("T" in value or "Z" in value or "+" in value):
            return iso_to_ts(value)
        t = int(float(value))
    except Exception:
        return None
    if t < 0:
        return None
    while t > 10_000_000_000:
        t //= 1000
    return t


def market_close_ts(market: Dict[str, Any]) -> Optional[int]:
    """Return the *effective* market timestamp (unix seconds) when available.

    IMPORTANT:
      For Kalshi sports markets, ``close_time`` / ``expiration_time`` can be a
      *settlement/expiration* timestamp that may be days or weeks after the game.
      The field ``expected_expiration_time`` is usually the best proxy for the
      underlying game start.

    We therefore prefer (in order):
      1) expected_expiration_ts / expected_expiration_time
      2) expiration_ts / expiration_time
      3) close_ts / close_time

    The rest of the pipeline uses this value for "within lookahead" filtering
    and Odds API time-skew checks, so choosing the right field is critical.
    """

    # Prefer the game-time proxy when present.
    for k in ("expected_expiration_ts", "expected_expiration_time", "expectedExpirationTime"):
        try:
            if k in market and market[k] is not None:
                ts = _normalize_epoch_seconds(market[k])
                if ts is not None:
                    return ts
        except Exception:
            pass

    for k in ("expiration_ts", "expiration_time", "expirationTime"):
        try:
            if k in market and market[k] is not None:
                ts = _normalize_epoch_seconds(market[k])
                if ts is not None:
                    return ts
        except Exception:
            pass

    # Fall back to close time.
    try:
        if "close_ts" in market and market["close_ts"] is not None:
            return _normalize_epoch_seconds(market["close_ts"])
    except Exception:
        pass
    try:
        if "close_time" in market and market["close_time"]:
            return _normalize_epoch_seconds(market["close_time"])
    except Exception:
        pass
    return None


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
    min_close_ts: Optional[int] = None,
    max_close_ts: Optional[int] = None,
    limit: int = 200,
    mve_filter: str = "all",
) -> List[Dict[str, Any]]:
    params: Dict[str, Any] = {"series_ticker": series_ticker, "limit": limit}
    mf = str(mve_filter or "").strip().lower()
    if mf in ("exclude", "only"):
        params["mve_filter"] = mf
    # Some Kalshi filters can be picky about mixing status with close-ts bounds.
    # If we're bounding close time, omit status in the API call and filter locally.
    if status and max_close_ts is None and min_close_ts is None:
        params["status"] = status
    if min_close_ts is not None:
        params["min_close_ts"] = int(min_close_ts)
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

WIN_BY_RE = re.compile(
    r"^(?P<team>.+?)\s+wins?\s+by\s+(?P<side>over|under)\s*(?P<points>[0-9]+(?:\.[0-9]+)?)\s*(?:pts?|points?)?\s*$",
    re.IGNORECASE,
)


def parse_matchup(title: str) -> Optional[Tuple[str, str]]:
    """Extract team A and team B from a title like 'Chicago at Brooklyn' or 'A vs B'.

    Kalshi event titles sometimes contain colons for descriptors, e.g.
      - "NCAAB: Xavier at Creighton"
      - "Xavier at Creighton: Spread"
    A naive "substring after the last colon" breaks the second form, so we try
    a few candidates and return the first that parses.
    """
    raw = str(title or "")
    raw = re.sub(r"[\(\[].*?[\)\]]", " ", raw)
    raw = raw.replace("@", " at ").replace(" vs. ", " vs ").replace(" v ", " vs ").strip()
    raw = " ".join(raw.split())

    cands = [raw]
    if ":" in raw:
        # Common patterns: "LEAGUE: A at B" or "A at B: Spread"
        after_first = raw.split(":", 1)[1].strip()
        before_last = raw.rsplit(":", 1)[0].strip()
        for t in (after_first, before_last):
            if t and t not in cands:
                cands.append(t)

    rx = re.compile(r"(?i)(.+?)\s+(at|vs|versus)\s+(.+?)(?:\s*(?:\||-|–|—|:).*)?$")
    for t in cands:
        m = rx.search(t)
        if not m:
            continue
        a = str(m.group(1)).strip()
        b = str(m.group(3)).strip()
        if a and b and a.lower() != b.lower():
            return a, b
    return None


def derive_matchup_from_subtitles(line_type: str, yes_sub: str, no_sub: str) -> Optional[Tuple[str, str]]:
    """
    Fallback when Kalshi titles don't contain 'at'/'vs'.
    For H2H, subtitles are usually team/team.
    For spreads, subtitles are often 'TEAM -3.5' vs 'TEAM +3.5'.
    """
    y = str(yes_sub or "").strip()
    n = str(no_sub or "").strip()
    if not y or not n:
        return None
    if y.lower() in ("yes", "y") or n.lower() in ("no", "n"):
        return None

    if line_type == "h2h":
        y2 = re.sub(r"(?i)\b(wins?|to\s*win)\b", "", y).strip()
        n2 = re.sub(r"(?i)\b(wins?|to\s*win)\b", "", n).strip()
        if y2 and n2 and y2.lower() != n2.lower():
            return y2, n2

    if line_type == "spread":
        ly = parse_line_from_subtitle("spread", y)
        ln = parse_line_from_subtitle("spread", n)
        ty = str((ly or {}).get("team") or "").strip()
        tn = str((ln or {}).get("team") or "").strip()
        if ty and tn and ty.lower() != tn.lower():
            return ty, tn

    return None


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
        if m:
            team = m.group("team").strip()
            spread = float(m.group("spread"))
            return {"team": team, "spread": spread, "label": f"{team} {spread:+g}"}

        # Kalshi alt-spread phrasing: "TEAM wins by over 9.5 Points"
        wm = WIN_BY_RE.match(s)
        if wm:
            team = wm.group("team").strip()
            side = wm.group("side").lower()
            pts = float(wm.group("points"))
            # "wins by over X" aligns best with covering approximately -X.
            if side == "over":
                spread = -pts
                return {"team": team, "spread": spread, "label": f"{team} {spread:+g}"}
            # If Kalshi ever emits "wins by under X", we can't safely map it to a standard spread.
            return None

        return None

    if lt == "h2h":
        # Common forms: "Brooklyn", "Brooklyn wins"
        team = re.sub(r"(?i)\b(wins|win|to win)\b", "", s).strip()
        if not team:
            return None
        return {"team": team, "label": team}

    return None


def infer_line_type(series_ticker: str, market_ticker: str, market_title: str) -> Optional[str]:
    raw = f"{series_ticker} {market_ticker} {market_title}".lower()
    st_u = str(series_ticker or "").upper()

    # Spread signals must come before totals because Kalshi spread props often contain "over 9.5".
    if "spread" in raw or "wins by" in raw or "win by" in raw or "margin" in raw or "SPREAD" in st_u:
        return "spread"

    # Totals: require an explicit keyword or a strong ticker hint.
    if "total" in raw or "TOTAL" in st_u or "OVERUNDER" in st_u or "OU" in st_u:
        if "total" in raw or TOTAL_RE.search(raw):
            return "total"

    # Many moneyline series use GAME; also allow "h2h".
    if "game" in raw or "moneyline" in raw or "money line" in raw or "h2h" in raw:
        return "h2h"
    return None


def infer_line_type_from_subtitles(yes_sub: str, no_sub: str) -> Optional[str]:
    """Fallback inference when ticker/title don't include obvious keywords."""
    ys = (yes_sub or "").strip()
    ns = (no_sub or "").strip()
    if not ys and not ns:
        return None

    hay = f"{ys} {ns}".upper()

    # Kalshi spread markets often look like: "TEAM wins by over 9.5 Points"
    if "WINS BY" in hay or "WIN BY" in hay or "MARGIN" in hay:
        return "spread"

    # Totals usually look like: "Over 42.5" / "Under 42.5"
    if ("OVER" in hay or "UNDER" in hay) and re.search(r"\d", hay):
        return "total"

    # Spreads usually contain +/- with a number: "-4.5", "+3", etc.
    if re.search(r"(^|\s)[+-]\s*\d", hay):
        return "spread"

    # H2H usually looks like two team names (no digits) and not literal YES/NO
    if ys and ns:
        ys_l = ys.lower()
        ns_l = ns.lower()
        if ys_l not in ("yes", "no") and ns_l not in ("yes", "no"):
            if not re.search(r"\d", hay):
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


def _short_market_view(m: Dict[str, Any], close_ts: Optional[int] = None) -> Dict[str, Any]:
    """Small, stable view of a market for pipeline artifacts."""
    return {
        "ticker": m.get("ticker") or m.get("market_ticker"),
        "status": m.get("status"),
        "close_time": m.get("close_time"),
        "close_ts": close_ts if close_ts is not None else m.get("close_ts"),
        "expected_expiration_time": m.get("expected_expiration_time") or m.get("expectedExpirationTime"),
        "expiration_time": m.get("expiration_time") or m.get("expirationTime"),
        "event_title": (m.get("event_title") or ""),
        "title": (m.get("title") or m.get("market_title") or ""),
        "yes_sub_title": m.get("yes_sub_title"),
        "no_sub_title": m.get("no_sub_title"),
        "volume": m.get("volume"),
        "yes_ask": m.get("yes_ask"),
        "no_ask": m.get("no_ask"),
        "yes_bid": m.get("yes_bid"),
        "no_bid": m.get("no_bid"),
    }


def _has_real_bid(p: Optional[float]) -> bool:
    """Return True only if a bid is meaningfully present.

    We treat 0 as *no bid*; deriving an ask from a 0 bid yields a phantom 100¢ ask.
    """
    try:
        return p is not None and float(p) > 0.0
    except Exception:
        return False


def best_buy_probs(market: Dict[str, Any]) -> Tuple[Optional[float], Optional[float]]:
    """Return (yes_buy_prob, no_buy_prob) based on ask prices.

    Prefer yes_ask/no_ask if present. If missing, derive from opposing bids:
      yes_ask ~= 1 - no_bid
      no_ask  ~= 1 - yes_bid
    """
    yes_ask = cents_to_prob(market.get("yes_ask"))
    no_ask = cents_to_prob(market.get("no_ask"))
    # Newer API responses may provide *_dollars instead of cent fields.
    if yes_ask is None:
        yes_ask = _price_to_prob(market.get("yes_ask_dollars"))
    if no_ask is None:
        no_ask = _price_to_prob(market.get("no_ask_dollars"))

    if yes_ask is None:
        no_bid = cents_to_prob(market.get("no_bid"))
        if no_bid is None:
            no_bid = _price_to_prob(market.get("no_bid_dollars"))
        if _has_real_bid(no_bid):
            yes_ask = max(0.0, 1.0 - no_bid)

    if no_ask is None:
        yes_bid = cents_to_prob(market.get("yes_bid"))
        if yes_bid is None:
            yes_bid = _price_to_prob(market.get("yes_bid_dollars"))
        if _has_real_bid(yes_bid):
            no_ask = max(0.0, 1.0 - yes_bid)

    return yes_ask, no_ask


def _price_to_prob(x: Any) -> Optional[float]:
    if x is None:
        return None
    try:
        # Kalshi orderbook_fp uses dollars like "0.1500" for 15c (15%).
        if isinstance(x, str):
            v = float(x)
        else:
            v = float(x)
        if v > 1.0:
            # Sometimes APIs return cents; treat >1 as cents.
            v = v / 100.0
        return max(0.0, min(1.0, v))
    except Exception:
        return None


def best_buy_probs_from_orderbook(session: requests.Session, market_ticker: str) -> Tuple[Optional[float], Optional[float]]:
    """
    Derive best-buy YES and NO probabilities from the CLOB (orderbook).
    On Kalshi, the orderbook is bids for YES and bids for NO. Best-buy price for:
      - YES is 1 - (best NO bid)
      - NO  is 1 - (best YES bid)
    """
    ob = kalshi_get(session, f"/markets/{market_ticker}/orderbook", params={"depth": 1})

    # Prefer floating-point orderbook if present.
    fp = (ob or {}).get("orderbook_fp") or {}
    yes_levels = fp.get("yes_dollars") or []
    no_levels = fp.get("no_dollars") or []

    # Fallback to raw orderbook.
    if not yes_levels and not no_levels:
        raw = (ob or {}).get("orderbook") or {}
        # Some responses include *_dollars inside `orderbook`
        yes_levels = raw.get("yes_dollars") or raw.get("yes") or []
        no_levels = raw.get("no_dollars") or raw.get("no") or []

    def best_bid(levels: Any) -> Optional[float]:
        """
        Robust: docs show price levels are ordered, but we don't assume direction.
        We take the max price level as the best bid.
        """
        best: Optional[float] = None
        if not levels:
            return None
        for lvl in levels:
            p = None
            if isinstance(lvl, dict):
                p = _price_to_prob(lvl.get("price") or lvl.get("p"))
            elif isinstance(lvl, (list, tuple)) and len(lvl) >= 1:
                p = _price_to_prob(lvl[0])
            if p is None or p <= 0.0:
                continue
            if best is None or p > best:
                best = p
        return best

    yes_bid = best_bid(yes_levels)
    no_bid = best_bid(no_levels)

    # Only derive an ask if the opposing bid is real; otherwise treat as unknown.
    yes_ask = (max(0.0, min(1.0, 1.0 - no_bid))) if _has_real_bid(no_bid) else None
    no_ask = (max(0.0, min(1.0, 1.0 - yes_bid))) if _has_real_bid(yes_bid) else None
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
        short, long_ = (na, nb) if len(na) <= len(nb) else (nb, na)
        if len(short) >= 4:
            return 0.90
        if len(short) == 3 and long_.startswith(short):
            return 0.85

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


def map_team_to_event_team(
    team_guess: str,
    odds_event: Dict[str, Any],
    min_score: float = 0.72,
) -> Tuple[Optional[str], float]:
    home = str(odds_event.get("home_team") or "")
    away = str(odds_event.get("away_team") or "")
    if not team_guess or not home or not away:
        return None, 0.0

    sh = name_similarity(team_guess, home)
    sa = name_similarity(team_guess, away)

    best_team = home if sh >= sa else away
    best_score = max(sh, sa)

    if best_score < float(min_score):
        return None, best_score
    return best_team, best_score


def kalshi_market_url(market_ticker: str) -> str:
    return f"https://kalshi.com/markets/{str(market_ticker).lower()}"


# -----------------------------
# Team-ID matching helpers
# -----------------------------

# These functions use the repo-checked JSON mapping files (kalshi_team_cache.json
# and theoddsapi_team_cache-2.json) to match Kalshi markets to Odds API events
# without relying on fuzzy string similarity. We still keep the old fuzzy path as
# a fallback when mapping data is missing/ambiguous.


def build_odds_event_id_index(
    events: List[Dict[str, Any]],
) -> Dict[Tuple[str, frozenset], List[Dict[str, Any]]]:
    """Index Odds API events by (sport_prefix, {home_id, away_id})."""
    idx: Dict[Tuple[str, frozenset], List[Dict[str, Any]]] = {}
    for e in events:
        sport = sport_prefix_from_odds_key(str(e.get("sport_key") or ""))
        if not sport:
            continue
        h_id, a_id = odds_event_team_ids(e)
        if not h_id or not a_id:
            continue
        key = (sport, frozenset({h_id, a_id}))
        idx.setdefault(key, []).append(e)
    return idx


def _ts_from_odds_event(event: Dict[str, Any]) -> Optional[float]:
    try:
        ct = event.get("commence_time")
        if not ct:
            return None
        dt = datetime.fromisoformat(str(ct).replace("Z", "+00:00"))
        return dt.timestamp()
    except Exception:
        return None


def match_odds_event_by_team_ids(
    odds_id_index: Dict[Tuple[str, frozenset], List[Dict[str, Any]]],
    candidate_sports: List[str],
    team_a: str,
    team_b: str,
    close_ts: Optional[float] = None,
) -> Tuple[Optional[Dict[str, Any]], Optional[str], bool]:
    """Try to match a Kalshi matchup to an Odds API event via team ids.

    Returns (event, sport_prefix_used, skew_rejected).
    """
    best: Optional[Dict[str, Any]] = None
    best_sport: Optional[str] = None
    best_dt: Optional[float] = None

    for sport in candidate_sports:
        a_id = kalshi_team_id(sport, team_a)
        b_id = kalshi_team_id(sport, team_b)
        if not a_id or not b_id:
            continue
        key = (sport, frozenset({a_id, b_id}))
        events = odds_id_index.get(key) or []
        if not events:
            continue

        for e in events:
            et = _ts_from_odds_event(e)
            if close_ts is not None and et is not None:
                dt = abs(float(et) - float(close_ts))
            elif et is not None:
                dt = float(et)
            else:
                dt = 0.0

            if best_dt is None or dt < best_dt:
                best = e
                best_sport = sport
                best_dt = dt

    # Safety: if we had a Kalshi close time, don't accept an ID-match that's wildly far away.
    if close_ts is not None and best is not None and best_dt is not None:
        max_skew_hours = env_float("KALSHI_ID_MATCH_MAX_SKEW_HOURS", 12.0)
        max_skew_s = float(max_skew_hours) * 3600.0
        if max_skew_s > 0 and float(best_dt) > max_skew_s:
            return None, None, True

    return best, best_sport, False


def infer_sport_prefixes_from_kalshi(
    series_ticker: str,
    series_title: str,
    market_ticker: str,
    event_title: str,
    market_title: str,
) -> List[str]:
    hay = f"{series_ticker} {series_title} {market_ticker} {event_title} {market_title}".lower()
    out: List[str] = []
    if "mlb" in hay:
        out.append("mlb")
    if "nba" in hay:
        out.append("nba")
    if "nfl" in hay:
        out.append("nfl")
    if "nhl" in hay:
        out.append("nhl")
    if "ncaab" in hay or "cbb" in hay:
        out.append("ncaab")
    if "ncaaf" in hay or "cfb" in hay:
        out.append("ncaaf")
    return out


def _league_tokens_from_sport_keys(sport_keys: List[str]) -> List[str]:
    """Best-effort mapping from Odds API sport keys -> league keywords likely to appear in Kalshi series."""
    toks: List[str] = []
    for k in (sport_keys or []):
        kl = str(k).lower()
        if "_nfl" in kl:
            toks += ["NFL"]
        if "_nba" in kl:
            toks += ["NBA"]
        if "_mlb" in kl:
            toks += ["MLB"]
        if "_nhl" in kl:
            toks += ["NHL"]
        if "_ncaaf" in kl:
            toks += ["NCAAF", "CFB", "NCAA", "NCAA FOOTBALL", "COLLEGE FOOTBALL"]
        if "_ncaab" in kl:
            toks += ["NCAAB", "CBB", "NCAA", "NCAA BASKETBALL", "COLLEGE BASKETBALL"]

    out: List[str] = []
    seen: set = set()
    for t in toks:
        tt = str(t).strip()
        if not tt or tt in seen:
            continue
        seen.add(tt)
        out.append(tt)
    return out


def _sport_words_from_sport_keys(sport_keys: List[str]) -> List[str]:
    """Fallback sport-level tokens to help pick the right Kalshi series."""
    sk = set((k or "").strip().lower() for k in sport_keys or [])
    toks: List[str] = []
    if any(k.startswith("americanfootball_") for k in sk):
        toks.append("FOOTBALL")
    if any(k.startswith("basketball_") for k in sk):
        toks.append("BASKETBALL")
    if any(k.startswith("baseball_") for k in sk):
        toks.append("BASEBALL")
    if any(k.startswith("icehockey_") for k in sk):
        toks.append("HOCKEY")
    if any("ncaaf" in k or "ncaab" in k for k in sk):
        toks.append("COLLEGE")

    out: List[str] = []
    seen: set[str] = set()
    for t in toks:
        tt = str(t).upper().strip()
        if not tt or tt in seen:
            continue
        seen.add(tt)
        out.append(tt)
    return out


def _team_tokens_from_odds_events(events: List[Dict[str, Any]], max_tokens: int = 350) -> List[str]:
    """Extract a bounded set of team tokens from Odds API events for series scoring."""
    toks: set[str] = set()
    for ev in events or []:
        for k in ("home_team", "away_team"):
            raw = str(ev.get(k) or "").strip()
            if not raw:
                continue
            norm = normalize_team_name(raw)
            if not norm:
                continue
            toks.add(norm.upper())
            for part in norm.split():
                if len(part) >= 4:
                    toks.add(part.upper())

    out = sorted(toks, key=lambda s: (-len(s), s))
    return out[: max(1, int(max_tokens))]


def _score_series_candidate(
    ticker: str,
    title: str,
    league_tokens: List[str],
    sport_words: List[str],
    team_tokens: List[str],
) -> int:
    """Higher score = more likely the series is relevant to the requested sports + line types."""
    txt = f"{ticker} {title}".upper()
    weights = {
        "NFL": 12, "NBA": 12, "MLB": 12, "NHL": 12,
        "NCAAF": 10, "NCAAB": 10,
        "COLLEGE FOOTBALL": 8, "COLLEGE BASKETBALL": 8,
        "NCAA FOOTBALL": 7, "NCAA BASKETBALL": 7,
        "CFB": 6, "CBB": 6,
        "NCAA": 4,
    }
    score = 0
    for tok in league_tokens or []:
        tt = str(tok).upper().strip()
        if tt and tt in txt:
            score += int(weights.get(tt, 3))

    for w in sport_words or []:
        ww = str(w).upper().strip()
        if ww and ww in txt:
            score += 2

    for tok in team_tokens or []:
        tt = str(tok).upper().strip()
        if tt and tt in txt:
            score += 8
            break

    if "MONEYLINE" in txt or "H2H" in txt:
        score += 3
    if "SPREAD" in txt:
        score += 3
    if "TOTAL" in txt or "OVER" in txt or "UNDER" in txt or "O/U" in txt:
        score += 3
    if "GAME" in txt or "MATCH" in txt:
        score += 1
    return int(score)


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
    commence_time_iso: str
    kalshi_volume: int
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
        "commence_time_iso": r.commence_time_iso,
        "kalshi_volume": r.kalshi_volume,
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
    "commence_time_iso",
    "kalshi_volume",
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


def _redact_headers(headers: Dict[str, Any]) -> Dict[str, Any]:
    """Remove/replace sensitive headers before writing artifacts."""
    if not headers:
        return {}
    redacted_keys = {
        "authorization",
        "cookie",
        "set-cookie",
        "x-api-key",
        "apikey",
        "api-key",
        "signature",
        "x-signature",
        "kalshi-key-id",
        "kalshi-timestamp",
        "kalshi-signature",
    }
    out: Dict[str, Any] = {}
    for k, v in headers.items():
        lk = str(k).lower()
        if lk in redacted_keys:
            out[k] = "REDACTED"
        else:
            out[k] = v
    return out


def _redact_url(url: str) -> str:
    """Redact sensitive query params (e.g., apiKey) while keeping the URL useful."""
    try:
        parsed = urlparse(url)
        if not parsed.query:
            return url
        qs = parse_qsl(parsed.query, keep_blank_values=True)
        safe_qs = []
        for k, v in qs:
            lk = str(k).lower()
            if lk in ("apikey", "api_key", "key", "token", "signature"):
                safe_qs.append((k, "REDACTED"))
            else:
                safe_qs.append((k, v))
        new_query = urlencode(safe_qs, doseq=True)
        return urlunparse(
            (parsed.scheme, parsed.netloc, parsed.path, parsed.params, new_query, parsed.fragment)
        )
    except Exception:
        # If anything goes wrong, fallback to dropping the query string entirely
        try:
            parsed = urlparse(url)
            return urlunparse((parsed.scheme, parsed.netloc, parsed.path, parsed.params, "", parsed.fragment))
        except Exception:
            return url


def _safe_stem_from_url(url: str) -> str:
    """Create a short, filesystem-safe stem from a URL path."""
    try:
        parsed = urlparse(url)
        path = parsed.path or ""
        # last 2 segments are usually enough signal
        segs = [s for s in path.split("/") if s]
        tail = "_".join(segs[-2:]) if segs else "request"
        tail = re.sub(r"[^A-Za-z0-9_.-]+", "_", tail)[:80]
        return tail or "request"
    except Exception:
        return "request"


def _attach_master_debug(
    session: requests.Session,
    outdir: str,
    stamp: str,
    *,
    max_requests: int = 250,
    max_kb: int = 512,
) -> Dict[str, Any]:
    """
    Attach a response hook that writes per-request artifacts:
      - <N>_<stem>_meta.json   (request + response metadata, secrets redacted)
      - <N>_<stem>_raw.txt     (raw response text, truncated)
      - <N>_<stem>_json.json   (parsed JSON pretty-printed, when possible)
      - index.jsonl            (one line per saved response)

    Returns a small dict with counters you can include in run_meta.
    """
    os.makedirs(outdir, exist_ok=True)
    index_path = os.path.join(outdir, "index.jsonl")
    counter = itertools.count(1)
    stats = {
        "saved": 0,
        "skipped_over_max": 0,
        "parse_json_failed": 0,
        "json_skipped_too_large": 0,
    }

    def hook(resp: requests.Response, *args: Any, **kwargs: Any) -> requests.Response:
        n = next(counter)
        if n > max_requests:
            stats["skipped_over_max"] += 1
            return resp

        try:
            req = resp.request
            safe_url = _redact_url(getattr(req, "url", "") or "")
            stem = _safe_stem_from_url(safe_url)
            base = f"{n:04d}_{stem}"

            meta = {
                "stamp_utc": stamp,
                "request": {
                    "method": getattr(req, "method", None),
                    "url": safe_url,
                    "headers": _redact_headers(dict(getattr(req, "headers", {}) or {})),
                    # NOTE: body can be bytes; avoid dumping huge payloads
                    "body_preview": None,
                },
                "response": {
                    "status_code": getattr(resp, "status_code", None),
                    "reason": getattr(resp, "reason", None),
                    "elapsed_seconds": getattr(
                        getattr(resp, "elapsed", None), "total_seconds", lambda: None
                    )(),
                    "headers": _redact_headers(dict(getattr(resp, "headers", {}) or {})),
                },
            }

            body = getattr(req, "body", None)
            if body is not None:
                try:
                    if isinstance(body, (bytes, bytearray)):
                        meta["request"]["body_preview"] = body[:1024].decode("utf-8", errors="replace")
                    else:
                        body_text = str(body)
                        meta["request"]["body_preview"] = body_text[:1024]
                except Exception:
                    meta["request"]["body_preview"] = "<unavailable>"

            # Raw response (truncated)
            raw_path = os.path.join(outdir, f"{base}_raw.txt")
            content = b""
            try:
                content = resp.content or b""
                limit = max(1, int(max_kb)) * 1024
                snippet = content[:limit]
                Path(raw_path).write_bytes(snippet)
            except Exception:
                pass

            # Parsed JSON (when possible)
            json_path = os.path.join(outdir, f"{base}_json.json")
            parsed_ok = False
            try:
                limit = max(1, int(max_kb)) * 1024
                # Avoid double-heavy work: only pretty-print JSON if the response is small enough.
                clen = len(content) if content is not None else 0
                if clen == 0:
                    # no body; nothing to parse
                    pass
                elif clen <= limit:
                    parsed = resp.json()
                    _write_json(json_path, parsed)
                    parsed_ok = True
                else:
                    stats["json_skipped_too_large"] += 1
            except Exception:
                stats["parse_json_failed"] += 1

            meta_path = os.path.join(outdir, f"{base}_meta.json")
            _write_json(meta_path, meta)

            # Index line
            try:
                idx = {
                    "n": n,
                    "method": meta["request"]["method"],
                    "url": safe_url,
                    "status": meta["response"]["status_code"],
                    "stem": stem,
                    "meta_file": os.path.basename(meta_path),
                    "raw_file": os.path.basename(raw_path),
                    "json_file": os.path.basename(json_path) if parsed_ok else None,
                }
                with open(index_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(idx, ensure_ascii=False))
                    f.write("\n")
            except Exception:
                pass

            stats["saved"] += 1
        except Exception:
            # Never allow debug logging to break the run
            pass

        return resp

    session.hooks.setdefault("response", []).append(hook)
    return stats


def _stage_inc(stages: Dict[str, Any], name: str, n: int = 1) -> None:
    s = stages.setdefault(name, {"count": 0})
    try:
        s["count"] = int(s.get("count") or 0) + int(n)
    except Exception:
        s["count"] = 0


def _stage_sample(stages: Dict[str, Any], name: str, item: Any, limit: int = 20) -> None:
    s = stages.setdefault(name, {"count": 0, "samples": []})
    if "count" not in s:
        s["count"] = 0
    samples = s.setdefault("samples", [])
    if isinstance(samples, list) and len(samples) < limit:
        samples.append(item)


def scan() -> int:
    # Output directory (create immediately so artifacts always have *something* if we can write it)
    outdir = env_str("OUTDIR", "out")
    os.makedirs(outdir, exist_ok=True)
    now_dt_utc = datetime.now(timezone.utc)
    stamp = now_dt_utc.strftime("%Y%m%d_%H%M%SZ")

    debug_max = env_int("KALSHI_DEBUG_MAX", 50)  # keep artifacts small; bump if needed

    pipeline_report_enabled = env_bool("KALSHI_PIPELINE_REPORT", False)
    pipeline_stages: Dict[str, Any] = {}
    api_close_filter = env_bool("KALSHI_API_CLOSE_FILTER", False)

    fee_cents = env_int("KALSHI_BUY_FEE_CENTS", 2)
    fee_prob = fee_cents / 100.0

    lookahead_hours = env_int("KALSHI_LOOKAHEAD_HOURS", 72)
    game_lookahead_hours = env_int("KALSHI_GAME_LOOKAHEAD_HOURS", 24)
    # IMPORTANT: Default min volume to 0. The GitHub workflow can (and should) set
    # a higher value via repo Variables once the pipeline is producing valid matches.
    min_volume = env_int("KALSHI_MIN_VOLUME", 0)
    volume_missing_policy = env_str("KALSHI_VOLUME_MISSING_POLICY", "include").strip().lower()
    if volume_missing_policy not in ("include", "exclude"):
        # Defensive: treat unknown values as include
        volume_missing_policy = "include"
    master_debug = env_bool("MASTER_DEBUG", False) or env_bool("KALSHI_MASTER_DEBUG", False)
    master_debug_max = env_int("MASTER_DEBUG_MAX", 250)
    master_debug_max_kb = env_int("MASTER_DEBUG_MAX_KB", 512)
    use_orderbook = env_bool("KALSHI_USE_ORDERBOOK", True)
    orderbook_max_calls = env_int("KALSHI_ORDERBOOK_MAX_CALLS", 300)
    orderbook_calls = 0
    now_ts = int(now_dt_utc.timestamp())
    time_time_ts = int(time.time())
    clock_skew_seconds = time_time_ts - now_ts
    max_close_ts = now_ts + lookahead_hours * 3600
    min_close_buffer_hours = env_int("KALSHI_MIN_CLOSE_BUFFER_HOURS", 0)
    min_close_ts = now_ts - max(0, min_close_buffer_hours) * 3600
    mve_filter = env_str("KALSHI_MVE_FILTER", "all")

    min_edge = env_float("MIN_EDGE", 0.03)
    top_n = env_int("TOP_N", 25)
    no_filters = env_bool("KALSHI_NO_FILTERS", False) or env_bool("NO_FILTERS", False)
    debug_mode = env_bool("DEBUG_MODE", False) or env_bool("KALSHI_DEBUG_MODE", False)
    debug_wide_scan = env_bool("KALSHI_DEBUG_WIDE_SCAN", False)
    # Debug artifacts = extra JSONL context, but do NOT implicitly weaken matching.
    debug_artifacts = debug_mode or env_bool("KALSHI_DEBUG_ARTIFACTS", False) or no_filters
    # Only loosen matching if explicitly requested.
    debug_loose = env_bool("KALSHI_DEBUG_LOOSE_MATCH", False)
    debug_log = debug_mode or debug_artifacts
    debug_force_orderbook = env_bool("KALSHI_DEBUG_FORCE_ORDERBOOK", False)
    # In DEBUG_MODE we want to *see* non-edge matches to validate matching/filtering.
    debug_show_all = debug_mode or env_bool("KALSHI_DEBUG_SHOW_ALL", False)
    match_max_time_skew_hours = env_int("KALSHI_MATCH_MAX_TIME_SKEW_HOURS", 12)

    # If debug/no-filter, automatically enable pipeline report unless explicitly disabled.
    pipeline_report_enabled = pipeline_report_enabled or debug_mode or no_filters

    odds_regions = env_str("ODDS_REGIONS", "us")
    min_books = env_min_books()
    include_unknown_volume = True if no_filters else (volume_missing_policy != "exclude")
    min_books_effective = max(1, min_books)
    # DEBUG_MODE should still respect volume/time filters; it should only show non-edge matches.
    min_volume_effective = 0 if no_filters else min_volume
    if no_filters or debug_wide_scan or lookahead_hours <= 0:
        max_close_ts_effective = None
        min_close_ts_effective = None
    else:
        max_close_ts_effective = max_close_ts
        min_close_ts_effective = min_close_ts
    close_window_enabled = (min_close_ts_effective is not None) or (max_close_ts_effective is not None)

    run_meta: Dict[str, Any] = {
        "stamp_utc": stamp,
        "now_ts": now_ts,
        "time_time_ts": time_time_ts,
        "clock_skew_seconds": clock_skew_seconds,
        "fee_cents": fee_cents,
        "lookahead_hours": lookahead_hours,
        "game_lookahead_hours": game_lookahead_hours,
        "min_volume": min_volume,
        "volume_missing_policy": volume_missing_policy,
        "include_unknown_volume": include_unknown_volume,
        "master_debug": master_debug,
        "master_debug_max": master_debug_max,
        "master_debug_max_kb": master_debug_max_kb,
        "min_volume_effective": min_volume_effective,
        "use_orderbook": use_orderbook,
        "min_edge": min_edge,
        "top_n": top_n,
        "debug_mode": debug_mode,
        "debug_wide_scan": debug_wide_scan,
        "no_filters": no_filters,
        "debug_artifacts": debug_artifacts,
        "debug_loose": debug_loose,
        "debug_force_orderbook": debug_force_orderbook,
        "debug_show_all": debug_show_all,
        "match_max_time_skew_hours": match_max_time_skew_hours,
        "debug_log": debug_log,
        "min_close_buffer_hours": min_close_buffer_hours,
        "min_close_ts": min_close_ts_effective,
        "max_close_ts": max_close_ts_effective,
        "close_window_enabled": close_window_enabled,
        "min_books": min_books,
        "min_books_effective": min_books_effective,
        "odds_regions": odds_regions,
        "sport_keys": env_sport_keys(),
        "kalshi_base": _kalshi_base_url(),
        "mve_filter": mve_filter,
        "orderbook_max_calls": orderbook_max_calls,
        "status": "starting",
    }

    if pipeline_report_enabled:
        pipeline_stages["config"] = {
            "now_ts": now_ts,
            "min_close_ts": min_close_ts_effective,
            "max_close_ts": max_close_ts_effective,
            "lookahead_hours": lookahead_hours,
            "game_lookahead_hours": game_lookahead_hours,
        }

    meta_path = os.path.join(outdir, f"kalshi_run_meta_{stamp}.json")
    meta_latest = os.path.join(outdir, "kalshi_run_meta_latest.json")

    series_override = env_str("KALSHI_SERIES_TICKERS", "").strip()
    series_category = env_str("KALSHI_SERIES_CATEGORY", "Sports")
    max_series = env_int("KALSHI_MAX_SERIES_TICKERS", 60)

    include_pat = env_str(
        "KALSHI_SPORTS_SERIES_INCLUDE_REGEX",
        r"(GAME|MATCH|MONEYLINE|H2H|SPREAD|TOTAL|OVER|UNDER)",
    )
    exclude_pat = env_str(
        "KALSHI_SPORTS_SERIES_EXCLUDE_REGEX",
        r"(PROP|PLAYER|MVP|CHAMP|TROPHY|AWARD|SEASON|FUTURE|FUTURES|WINS|EXACT|COMBO|PARLAY|PACK|SALE|COACH|TOP\s*\d)",
    )
    if no_filters:
        include_pat = r".*"
        exclude_pat = r"^$"
    include_re = re.compile(include_pat, re.IGNORECASE)
    exclude_re = re.compile(exclude_pat, re.IGNORECASE)

    run_meta["series_category"] = series_category
    run_meta["series_include_regex"] = include_pat
    run_meta["series_exclude_regex"] = exclude_pat
    run_meta["max_series_tickers"] = max_series

    # Odds events (all at once)
    sport_keys = env_sport_keys()
    sportsbook_source = "theoddsapi"
    odds_events = fetch_all_sports_events(sport_keys, markets="h2h,spreads,totals", regions=odds_regions)

    if not odds_events:
        print("Odds API returned no events — falling back to DraftKings API.")
        sportsbook_source = "draftkings"
        odds_events = fetch_all_draftkings_events(sport_keys)

    run_meta["sportsbook_source"] = sportsbook_source

    if pipeline_report_enabled:
        _stage_inc(pipeline_stages, "odds_ingest", len(odds_events or []))
        for e in (odds_events or [])[:20]:
            _stage_sample(
                pipeline_stages,
                "odds_ingest",
                {
                    "id": e.get("id"),
                    "sport_key": e.get("sport_key"),
                    "commence_time": e.get("commence_time"),
                    "home_team": e.get("home_team"),
                    "away_team": e.get("away_team"),
                },
                limit=20,
            )

    if not odds_events:
        # Default behavior: skip (exit 0) if sportsbook odds are unavailable.
        # Set FAIL_ON_NO_ODDS_EVENTS=1 to preserve the old hard-fail behavior.
        fail_on_no_odds = env_bool("FAIL_ON_NO_ODDS_EVENTS", False)
        run_meta["status"] = (
            "error_no_odds_events" if fail_on_no_odds else "skipped_no_odds_events"
        )
        _write_json(meta_path, run_meta)
        _copy_text(meta_path, meta_latest)
        print("No sportsbook events returned from The Odds API or DraftKings fallback.")
        print("Likely causes: quota exhausted, rate limits, invalid API key, or network failure.")

        # Optional heads-up in Discord so you still know the run happened.
        try:
            webhook = os.getenv("DISCORD_WEBHOOK_URL", "").strip()
            if master_debug or debug_mode:
                webhook = ""
            if webhook:
                post_discord(
                    webhook,
                    "⚠️ Kalshi value scan skipped — sportsbook odds unavailable.\n"
                    "See workflow logs for upstream errors (Odds API quota / DK 403, etc.).\n"
                    "Fix: add Odds API credits, reduce SPORT_KEYS/markets, or use a self-hosted "
                    "runner for DK fallback.",
                    "",
                )
        except Exception:
            pass

        return 2 if fail_on_no_odds else 0

    max_game_ts: Optional[int]
    if no_filters or game_lookahead_hours <= 0:
        max_game_ts = None
    else:
        max_game_ts = now_ts + int(game_lookahead_hours * 3600)
    filtered_events = []
    for ev in odds_events:
        commence_time = str(ev.get("commence_time") or "")
        ct = iso_to_ts(commence_time)
        if ct is None:
            continue
        if ct <= now_ts:
            continue
        if max_game_ts is not None and ct > max_game_ts:
            continue
        filtered_events.append(ev)
    odds_events = filtered_events

    if not odds_events:
        fail_on_no_odds = env_bool("FAIL_ON_NO_ODDS_EVENTS", False)
        run_meta["status"] = (
            "error_no_odds_events_in_window"
            if fail_on_no_odds
            else "skipped_no_odds_events_in_window"
        )
        _write_json(meta_path, run_meta)
        _copy_text(meta_path, meta_latest)
        print("No sportsbook events within the lookahead window after filtering.")
        webhook = env_str("DISCORD_WEBHOOK_URL", "")
        always_notify = env_bool("ALWAYS_NOTIFY", False)
        if master_debug or debug_mode:
            webhook = ""
        if webhook and always_notify:
            if max_game_ts is None:
                msg = "**Kalshi value**: no sportsbook games starting in the future."
            else:
                msg = f"**Kalshi value**: no sportsbook games starting within {game_lookahead_hours}h."
            post_discord(webhook, msg)
        return 2 if fail_on_no_odds else 0

    # Build a fast lookup index for ID-based matching (using the repo JSON mapping files).
    candidate_sports = sorted({s for s in (sport_prefix_from_odds_key(k) for k in sport_keys) if s})
    odds_event_id_index = build_odds_event_id_index(odds_events) if candidate_sports else {}
    odds_events_total = len(odds_events)
    odds_events_with_ids = sum(1 for ev in odds_events if all(odds_event_team_ids(ev)))
    run_meta["odds_events_total"] = odds_events_total
    run_meta["odds_events_with_team_ids"] = odds_events_with_ids
    run_meta["odds_event_id_index_pairs"] = len(odds_event_id_index)
    print(
        "[ID_MATCH] Odds events with team IDs: "
        f"{odds_events_with_ids}/{odds_events_total} "
        f"(pairs={len(odds_event_id_index)})"
    )

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
    master_debug_dir = None
    master_debug_stats: Optional[Dict[str, Any]] = None
    if master_debug:
        master_debug_dir = os.path.join(outdir, f"kalshi_master_debug_{stamp}")
        master_debug_stats = _attach_master_debug(
            session,
            master_debug_dir,
            stamp,
            max_requests=master_debug_max,
            max_kb=master_debug_max_kb,
        )
        run_meta["master_debug_dir"] = master_debug_dir
        run_meta["master_debug_stats"] = master_debug_stats
        try:
            latest_debug_path = os.path.join(outdir, "kalshi_master_debug_latest.txt")
            Path(latest_debug_path).write_text(master_debug_dir + "\n", encoding="utf-8")
        except Exception:
            pass
        print(
            "[MASTER_DEBUG] Writing per-request artifacts to: "
            f"{master_debug_dir} (max {master_debug_max} requests)"
        )
    series: List[str] = []
    discovered_series: List[Dict[str, Any]] = []
    title_by_ticker: Dict[str, str] = {}

    if series_override:
        series = [s.strip() for s in series_override.split(",") if s.strip()]
    else:
        min_series_for_include = env_int("KALSHI_MIN_SERIES_FOR_INCLUDE", 8)

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
        run_meta["min_series_for_include"] = min_series_for_include
        candidates: List[str] = []
        for s in discovered_series:
            ticker = str(s.get("ticker") or "").strip()
            title = str(s.get("title") or "")
            if not ticker:
                continue
            if exclude_re.search(ticker) or exclude_re.search(title):
                continue
            candidates.append(ticker)
            if include_re.search(ticker) or include_re.search(title):
                series.append(ticker)

        if len(series) >= min_series_for_include:
            run_meta["series_discovery_mode"] = "include_regex"
        else:
            # If include regex was too strict, fall back to anything not excluded.
            series = candidates
            if series:
                run_meta["series_discovery_mode"] = "fallback_not_excluded_min_series"

    if series_override:
        seen: set[str] = set()
        deduped: List[str] = []
        for t in series:
            if not t or t in seen:
                continue
            seen.add(t)
            deduped.append(t)
        series = deduped
    else:
        for s in discovered_series:
            t = str(s.get("ticker") or "").strip()
            if not t:
                continue
            title_by_ticker.setdefault(t, str(s.get("title") or ""))

        sport_keys = env_sport_keys()
        league_tokens = _league_tokens_from_sport_keys(sport_keys)
        sport_words = _sport_words_from_sport_keys(sport_keys)
        team_tokens = _team_tokens_from_odds_events(odds_events)
        series_min_score = env_int("KALSHI_SERIES_MIN_SCORE", 8)
        if no_filters:
            series_min_score = -999
        run_meta["series_league_tokens"] = league_tokens
        run_meta["series_sport_words"] = sport_words
        run_meta["series_team_tokens_count"] = len(team_tokens)
        run_meta["series_min_score"] = series_min_score

        scored: List[Tuple[int, str, str]] = []
        seen: set[str] = set()
        for t in series:
            if not t or t in seen:
                continue
            seen.add(t)
            title = title_by_ticker.get(t, "")
            score = _score_series_candidate(t, title, league_tokens, sport_words, team_tokens)
            scored.append((int(score), t, title))

        scored.sort(key=lambda x: (-x[0], x[1]))
        scored_keep = [x for x in scored if int(x[0]) >= series_min_score]
        if scored_keep:
            scored = scored_keep
        else:
            scored_pos = [x for x in scored if int(x[0]) > 0]
            if scored_pos:
                scored = scored_pos
            run_meta["series_score_filter_fallback"] = True
        if len(scored) > max_series:
            scored = scored[:max_series]
            run_meta["series_truncated_to"] = max_series

        series = [t for _, t, _ in scored]
        run_meta["series_scored_top"] = [
            {"ticker": t, "title": title, "score": int(score)}
            for (score, t, title) in scored[: min(25, len(scored))]
        ]

    if pipeline_report_enabled:
        _stage_inc(pipeline_stages, "kalshi_series_selected", len(series or []))
        for st in (series or [])[:20]:
            _stage_sample(
                pipeline_stages,
                "kalshi_series_selected",
                {"series_ticker": st},
                limit=20,
            )
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
    debug_candidates: List[Dict[str, Any]] = []
    debug_series_counts: List[Dict[str, Any]] = []
    drops = Counter()
    markets_fetched_total = 0
    markets_seen_total = 0
    series_errors = 0
    match_id_ok = 0
    match_id_skew_reject = 0
    match_fuzzy_ok = 0

    for st in series:
        try:
            print(f"Fetching markets for series={st} (close_window_enabled={close_window_enabled})")
            use_api_bounds = api_close_filter and not (no_filters or debug_wide_scan or lookahead_hours <= 0)
            api_min = min_close_ts_effective if use_api_bounds else None
            api_max = max_close_ts_effective if use_api_bounds else None
            markets = list_markets_for_series(
                session,
                st,
                status="",
                min_close_ts=api_min,
                max_close_ts=api_max,
                mve_filter=mve_filter,
            )
        except Exception as exc:
            series_errors += 1
            if series_errors <= 3:
                print(f"[WARN] list_markets_for_series failed for {st}: {type(exc).__name__}: {exc}")
            continue

        raw_count = len(markets)
        markets_fetched_total += raw_count
        if pipeline_report_enabled:
            _stage_inc(pipeline_stages, "kalshi_markets_ingest", raw_count)
            for m in markets[:5]:
                _stage_sample(
                    pipeline_stages,
                    "kalshi_markets_ingest",
                    _short_market_view(m),
                    limit=30,
                )
        # If the API didn't enforce open-only (or doesn't like combining filters),
        # enforce it here without dropping markets that omit 'status'.
        filtered: List[Dict[str, Any]] = []
        for m in markets:
            markets_seen_total += 1
            if pipeline_report_enabled:
                _stage_inc(pipeline_stages, "kalshi_markets_seen", 1)
            if pipeline_report_enabled and markets_seen_total <= 30:
                _stage_sample(
                    pipeline_stages,
                    "kalshi_markets_seen",
                    _short_market_view(m),
                    limit=30,
                )
            s = str(m.get("status") or "").strip().lower()
            if not (debug_mode or no_filters):
                if s in ("closed", "settled"):
                    continue
                if s and s not in ("open", "paused", "active"):
                    continue
            filtered.append(m)
        markets = filtered
        status_filtered_count = len(markets)
        if pipeline_report_enabled:
            _stage_inc(pipeline_stages, "status_pass", status_filtered_count)

        if close_window_enabled:
            close_filtered: List[Dict[str, Any]] = []
            for m in markets:
                cts = market_close_ts(m)
                if cts is None:
                    drops["close_time_missing"] += 1
                    if pipeline_report_enabled:
                        _stage_inc(pipeline_stages, "close_time_missing", 1)
                    close_filtered.append(m)
                    continue
                if min_close_ts_effective is not None and cts < min_close_ts_effective:
                    drops["close_time_before_min"] += 1
                    if pipeline_report_enabled:
                        _stage_inc(pipeline_stages, "drop_close_time_before_min", 1)
                    continue
                if max_close_ts_effective is not None and cts > max_close_ts_effective:
                    drops["close_time_after_max"] += 1
                    if pipeline_report_enabled:
                        _stage_inc(pipeline_stages, "drop_close_time_after_max", 1)
                        _stage_sample(
                            pipeline_stages,
                            "drop_close_time_after_max",
                            {
                                "market": _short_market_view(m, close_ts=cts),
                                "close_ts": cts,
                                "now_ts": now_ts,
                                "max_close_ts": max_close_ts_effective,
                                "delta_hours": (cts - now_ts) / 3600.0,
                            },
                            limit=20,
                        )
                    continue
                if pipeline_report_enabled:
                    _stage_inc(pipeline_stages, "pass_close_window", 1)
                close_filtered.append(m)
            markets = close_filtered
        close_window_count = len(markets)
        if pipeline_report_enabled:
            _stage_inc(pipeline_stages, "close_window_pass", close_window_count)

        if len(debug_series_counts) < debug_max:
            debug_series_counts.append(
                {
                    "series_ticker": st,
                    "raw_count": raw_count,
                    "status_filtered_count": status_filtered_count,
                    "close_window_count": close_window_count,
                }
            )

        for m in markets:
            market_ticker = str(m.get("ticker") or "")
            if not market_ticker:
                continue

            event_ticker = str(m.get("event_ticker") or "")
            market_title = str(m.get("title") or "")
            yes_sub = str(
                m.get("yes_sub_title")
                or m.get("yes_subtitle")
                or m.get("yes_sub_title_text")
                or ""
            )
            no_sub = str(
                m.get("no_sub_title")
                or m.get("no_subtitle")
                or m.get("no_sub_title_text")
                or ""
            )
            subtitle = str(m.get("subtitle") or "").strip()
            raw_vol = m.get("volume_24h")
            if raw_vol is None:
                raw_vol = m.get("volume")
            vol: Optional[int] = None
            if raw_vol is not None:
                try:
                    vol = int(float(str(raw_vol).replace(",", "")))
                except Exception:
                    vol = None
            if vol is None:
                drops["volume_missing"] += 1
                if (min_volume_effective > 0) and (not include_unknown_volume):
                    drops["volume_missing_dropped"] += 1
                    continue
            if min_volume_effective > 0 and vol is not None and vol < min_volume_effective:
                drops["volume_below_min"] += 1
                continue

            lt = infer_line_type(st, market_ticker, market_title) or infer_line_type_from_subtitles(yes_sub, no_sub)
            if lt is None:
                drops["line_type_unknown"] += 1
                if debug_log and len(debug_unmatched) < debug_max:
                    debug_unmatched.append(
                        {
                            "reason": "unknown_line_type",
                            "series_ticker": st,
                            "event_title": "",
                            "market_title": market_title,
                            "market_ticker": market_ticker,
                            "yes_sub_title": yes_sub,
                            "no_sub_title": no_sub,
                            "kalshi_market": m,
                        }
                    )
                continue

            # Parse the YES proposition from subtitle (preferred) else title.
            line = parse_line_from_subtitle(lt, yes_sub) or parse_line_from_subtitle(lt, market_title)
            if not line:
                drops["line_parse_failed"] += 1
                if debug_log and len(debug_unmatched) < debug_max:
                    debug_unmatched.append(
                        {
                            "reason": "line_parse_failed",
                            "line_type": lt,
                            "series_ticker": st,
                            "event_title": "",
                            "market_title": market_title,
                            "market_ticker": market_ticker,
                            "yes_sub_title": yes_sub,
                            "no_sub_title": no_sub,
                            "kalshi_market": m,
                        }
                    )
                continue

            # Resolve event title (to get matchup).
            # Only trust subtitle if it parses as a matchup; otherwise fetch the event title.
            event_title = ""
            if subtitle and parse_matchup(subtitle):
                event_title = subtitle
            if event_ticker:
                cached_title = event_title_cache.get(event_ticker, "")
                if not event_title and cached_title:
                    event_title = cached_title
                if not event_title:
                    ev = get_event(session, event_ticker)
                    fetched_title = ""
                    if ev and isinstance(ev, dict):
                        eobj = ev.get("event") if "event" in ev else ev
                        fetched_title = str((eobj or {}).get("title") or "")
                    if fetched_title:
                        event_title_cache[event_ticker] = fetched_title
                        event_title = fetched_title
                elif event_title and not cached_title:
                    event_title_cache[event_ticker] = event_title

            if not event_title:
                # Some responses include nested event dict
                event_title = str((m.get("event") or {}).get("title") or "")

            matchup = parse_matchup(event_title) or parse_matchup(subtitle) or parse_matchup(market_title)
            if not matchup:
                matchup = derive_matchup_from_subtitles(lt, yes_sub, no_sub)
            if not matchup:
                drops["matchup_parse_failed"] += 1
                if debug_log and len(debug_unmatched) < debug_max:
                    debug_unmatched.append(
                        {
                            "reason": "matchup_parse_failed",
                            "line_type": lt,
                            "series_ticker": st,
                            "event_title": event_title,
                            "market_title": market_title,
                            "market_ticker": market_ticker,
                            "subtitle": subtitle,
                            "yes_sub_title": yes_sub,
                            "no_sub_title": no_sub,
                            "kalshi_market": m,
                        }
                    )
                continue

            team_a, team_b = matchup
            close_ts = market_close_ts(m)
            series_title = title_by_ticker.get(st, "")
            sport_hint = infer_sport_prefixes_from_kalshi(
                st,
                series_title,
                market_ticker,
                event_title,
                market_title,
            )
            sport_candidates = [p for p in sport_hint if p in candidate_sports] or candidate_sports

            odds_event: Optional[Dict[str, Any]] = None
            ranked = []
            sport_prefix_used: Optional[str] = None

            id_match_rejected = False
            if odds_event_id_index and sport_candidates:
                odds_event, sport_prefix_used, id_match_rejected = match_odds_event_by_team_ids(
                    odds_event_id_index,
                    sport_candidates,
                    team_a,
                    team_b,
                    close_ts=close_ts,
                )
                if odds_event:
                    match_id_ok += 1
                elif id_match_rejected:
                    match_id_skew_reject += 1

            if not odds_event:
                # Constrain fuzzy matching to inferred sport candidates to avoid cross-sport/date matches.
                scoped_events = odds_events
                if sport_candidates:
                    scoped_events = [
                        e
                        for e in odds_events
                        if sport_prefix_from_odds_key(str(e.get("sport_key") or "")) in sport_candidates
                    ]
                ranked = rank_event_matches(scoped_events, team_a, team_b, top_k=5)
                match_threshold = 0.0 if debug_loose else 1.30
                if not debug_loose and odds_event_id_index and sport_candidates:
                    has_id_pair = any(
                        kalshi_team_id(sport, team_a) and kalshi_team_id(sport, team_b)
                        for sport in sport_candidates
                    )
                    if has_id_pair:
                        match_threshold = 1.60
                odds_event = ranked[0][1] if ranked and ranked[0][0] >= match_threshold else None
                if odds_event:
                    # Reject fuzzy matches whose game time is wildly different than the Kalshi close time.
                    # This prevents e.g. a "2025NOV" Kalshi market matching to a "today" Odds event.
                    if close_ts is not None and match_max_time_skew_hours > 0:
                        commence_ts = iso_to_ts(str(odds_event.get("commence_time") or ""))
                        if commence_ts is not None:
                            skew = abs(int(commence_ts) - int(close_ts))
                            if skew > int(match_max_time_skew_hours * 3600):
                                drops["odds_match_time_skew"] += 1
                                if pipeline_report_enabled:
                                    _stage_inc(pipeline_stages, "drop_odds_match_time_skew", 1)
                                    _stage_sample(
                                        pipeline_stages,
                                        "drop_odds_match_time_skew",
                                        {
                                            "market": _short_market_view(m),
                                            "matchup": {"a": team_a, "b": team_b},
                                            "close_ts": close_ts,
                                            "odds_commence_time": odds_event.get("commence_time"),
                                            "skew_seconds": skew,
                                        },
                                        limit=10,
                                    )
                                if debug_log and len(debug_unmatched) < debug_max:
                                    debug_unmatched.append(
                                        {
                                            "reason": "odds_event_time_skew_reject",
                                            "series_ticker": st,
                                            "market_ticker": market_ticker,
                                            "kalshi_event_title": event_title or market_title,
                                            "matchup": {"a": team_a, "b": team_b},
                                            "close_ts": close_ts,
                                            "odds_commence_time": odds_event.get("commence_time"),
                                            "odds_sport_key": odds_event.get("sport_key"),
                                            "odds_id": odds_event.get("id"),
                                            "skew_seconds": skew,
                                            "skew_hours": skew / 3600.0,
                                            "max_skew_hours": match_max_time_skew_hours,
                                        }
                                    )
                                odds_event = None
                    if odds_event:
                        sport_prefix_used = sport_prefix_from_odds_key(str(odds_event.get("sport_key") or ""))
                        match_fuzzy_ok += 1
            if not odds_event:
                drops["odds_match_failed"] += 1
                if pipeline_report_enabled:
                    _stage_inc(pipeline_stages, "drop_odds_match_failed", 1)
                    _stage_sample(
                        pipeline_stages,
                        "drop_odds_match_failed",
                        {
                            "market": _short_market_view(m),
                            "matchup": {"a": team_a, "b": team_b},
                            "line_type": lt,
                        },
                        limit=10,
                    )
                if debug_log and len(debug_unmatched) < debug_max:
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
                if debug_log and len(debug_candidates) < debug_max:
                    debug_candidates.append(
                        {
                            "reason": "odds_event_match_failed",
                            "series_ticker": st,
                            "event_title": event_title or "",
                            "market_title": market_title,
                            "market_ticker": market_ticker,
                            "line_type": lt,
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
                        }
                    )
                continue

            fair_yes: Optional[Dict[str, Any]] = None
            team_for_books = str(line.get("team") or "")
            team_map_score = None
            if lt in ("h2h", "spread"):
                mapped_by_id = False
                if sport_prefix_used:
                    line_team_id = kalshi_team_id(sport_prefix_used, team_for_books)
                    home_id, away_id = odds_event_team_ids(odds_event)
                    if line_team_id and home_id and away_id:
                        if line_team_id == home_id:
                            team_for_books = str(odds_event.get("home_team") or "")
                            mapped_by_id = True
                        elif line_team_id == away_id:
                            team_for_books = str(odds_event.get("away_team") or "")
                            mapped_by_id = True

                if not mapped_by_id:
                    min_map = 0.55 if debug_loose else 0.72
                    team_for_books, team_map_score = map_team_to_event_team(
                        team_for_books,
                        odds_event,
                        min_score=min_map,
                    )
            if not team_for_books:
                drops["team_map_failed"] += 1
                if debug_log and len(debug_candidates) < debug_max:
                    debug_candidates.append(
                        {
                            "reason": "team_map_failed",
                            "series_ticker": st,
                            "event_title": event_title or "",
                            "market_title": market_title,
                            "market_ticker": market_ticker,
                            "line_type": lt,
                            "parsed_line": line,
                            "matchup": {"a": team_a, "b": team_b},
                            "team_guess": line.get("team"),
                            "team_map_score": team_map_score,
                            "odds_event": {
                                "home_team": odds_event.get("home_team"),
                                "away_team": odds_event.get("away_team"),
                                "sport_key": odds_event.get("sport_key"),
                                "commence_time": odds_event.get("commence_time"),
                                "id": odds_event.get("id"),
                            },
                        }
                    )
                    continue

            if lt == "h2h":
                fair_yes = fair_prob_h2h(odds_event, team_name=team_for_books, min_books=min_books_effective)
            elif lt == "spread":
                fair_yes = fair_prob_spread(
                    odds_event,
                    team_name=team_for_books,
                    point=float(line["spread"]),
                    min_books=min_books_effective,
                )
            elif lt == "total":
                fair_yes = fair_prob_total(
                    odds_event,
                    side=line["side"],
                    point=float(line["points"]),
                    min_books=min_books_effective,
                )

            if not fair_yes:
                drops["fair_prob_unavailable"] += 1
                if pipeline_report_enabled:
                    _stage_inc(pipeline_stages, "drop_fair_prob_unavailable", 1)
                    _stage_sample(
                        pipeline_stages,
                        "drop_fair_prob_unavailable",
                        {
                            "market": _short_market_view(m),
                            "line_type": lt,
                            "matchup": {"a": team_a, "b": team_b},
                        },
                        limit=10,
                    )
                if debug_log and len(debug_candidates) < debug_max:
                    debug_candidates.append(
                        {
                            "reason": "fair_prob_unavailable",
                            "series_ticker": st,
                            "event_title": event_title or "",
                            "market_title": market_title,
                            "market_ticker": market_ticker,
                            "line_type": lt,
                            "parsed_line": line,
                            "team_guess": line.get("team"),
                            "team_mapped": team_for_books,
                            "team_map_score": team_map_score,
                            "odds_event": {
                                "home_team": odds_event.get("home_team"),
                                "away_team": odds_event.get("away_team"),
                                "sport_key": odds_event.get("sport_key"),
                                "commence_time": odds_event.get("commence_time"),
                                "id": odds_event.get("id"),
                            },
                        }
                    )
                continue

            fair_prob_yes = float(fair_yes["fair_prob"])
            books_used = int(fair_yes.get("books_used") or 0)
            if books_used < min_books_effective:
                drops["books_below_min"] += 1
                if pipeline_report_enabled:
                    _stage_inc(pipeline_stages, "drop_books_below_min", 1)
                    _stage_sample(
                        pipeline_stages,
                        "drop_books_below_min",
                        {
                            "market": _short_market_view(m),
                            "books_used": books_used,
                            "min_books": min_books_effective,
                        },
                        limit=10,
                    )
                continue

            # Kalshi buy prices (asks)
            yes_buy, no_buy = best_buy_probs(m)
            # Pricing:
            # - In normal mode, use orderbook only as a fallback when snapshot asks are missing.
            # - In DEBUG_MODE, do NOT automatically explode runtime by fetching every orderbook.
            #   Enable KALSHI_DEBUG_FORCE_ORDERBOOK=1 when you explicitly want that behavior.
            want_orderbook = False
            if use_orderbook and market_ticker:
                want_orderbook = (yes_buy is None or no_buy is None) or (debug_force_orderbook and debug_mode)

            if want_orderbook and orderbook_calls < orderbook_max_calls:
                try:
                    orderbook_calls += 1
                    ob_yes_buy, ob_no_buy = best_buy_probs_from_orderbook(session, market_ticker)
                    if ob_yes_buy is not None:
                        yes_buy = ob_yes_buy
                    if ob_no_buy is not None:
                        no_buy = ob_no_buy
                except Exception as exc:
                    if debug_log and len(debug_unmatched) < debug_max:
                        debug_unmatched.append(
                            {
                                "reason": "orderbook_error",
                                "series_ticker": st,
                                "event_title": event_title or "",
                                "market_title": market_title,
                                "market_ticker": market_ticker,
                                "error": str(exc),
                            }
                        )
            if yes_buy is None and no_buy is None:
                drops["no_prices"] += 1
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

            # In DEBUG_MODE we want to keep *matched* games even if the edge is below MIN_EDGE,
            # so we can validate matching + filtering correctness.
            if (not debug_show_all) and (not no_filters) and best_edge < min_edge:
                drops["edge_below_min"] += 1
                if pipeline_report_enabled:
                    _stage_inc(pipeline_stages, "drop_edge_below_min", 1)
                    _stage_sample(
                        pipeline_stages,
                        "drop_edge_below_min",
                        {
                            "market": _short_market_view(m),
                            "edge": float(best_edge),
                            "min_edge": min_edge,
                        },
                        limit=10,
                    )
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
                            "min_books": min_books_effective,
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
                    commence_time_iso=str(odds_event.get("commence_time") or ""),
                    kalshi_volume=int(vol),
                    books_used=books_used,
                    url=kalshi_market_url(market_ticker),
                )
            )

    rows.sort(key=lambda r: r.edge, reverse=True)
    rows_all = list(rows)
    rows = rows[: max(1, top_n)]

    csv_path = os.path.join(outdir, f"kalshi_value_{stamp}.csv")
    json_path = os.path.join(outdir, f"kalshi_value_{stamp}.json")
    csv_latest = os.path.join(outdir, "kalshi_value_latest.csv")
    json_latest = os.path.join(outdir, "kalshi_value_latest.json")

    # In debug/no-filter modes, also write untruncated outputs to validate matching/filters.
    csv_all_path = os.path.join(outdir, f"kalshi_value_all_{stamp}.csv")
    json_all_path = os.path.join(outdir, f"kalshi_value_all_{stamp}.json")
    csv_all_latest = os.path.join(outdir, "kalshi_value_all_latest.csv")
    json_all_latest = os.path.join(outdir, "kalshi_value_all_latest.json")

    matched_path = os.path.join(outdir, f"kalshi_debug_matched_{stamp}.jsonl")
    unmatched_path = os.path.join(outdir, f"kalshi_debug_unmatched_{stamp}.jsonl")
    candidates_path = os.path.join(outdir, f"kalshi_debug_candidates_{stamp}.jsonl")
    series_counts_path = os.path.join(outdir, f"kalshi_debug_series_market_counts_{stamp}.json")
    matched_latest = os.path.join(outdir, "kalshi_debug_matched_latest.jsonl")
    unmatched_latest = os.path.join(outdir, "kalshi_debug_unmatched_latest.jsonl")
    candidates_latest = os.path.join(outdir, "kalshi_debug_candidates_latest.jsonl")
    series_counts_latest = os.path.join(outdir, "kalshi_debug_series_market_counts_latest.json")

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

    if debug_show_all or no_filters:
        with open(csv_all_path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=FIELDNAMES)
            w.writeheader()
            for r in rows_all:
                w.writerow(to_dict(r))
        with open(json_all_path, "w", encoding="utf-8") as f:
            json.dump([to_dict(r) for r in rows_all], f, indent=2)
        _copy_text(csv_all_path, csv_all_latest)
        _copy_text(json_all_path, json_all_latest)

    # Debug bundles (these are the “files I’m looking for”)
    _write_jsonl(matched_path, debug_matched)
    _write_jsonl(unmatched_path, debug_unmatched)
    _copy_text(matched_path, matched_latest)
    _copy_text(unmatched_path, unmatched_latest)
    if debug_artifacts:
        _write_jsonl(candidates_path, debug_candidates)
        _copy_text(candidates_path, candidates_latest)
    _write_json(series_counts_path, debug_series_counts)
    _copy_text(series_counts_path, series_counts_latest)

    run_meta["series_errors"] = series_errors
    run_meta["markets_fetched_total"] = markets_fetched_total
    run_meta["markets_seen_total"] = markets_seen_total
    run_meta["drop_counts"] = dict(drops)
    run_meta["match_id_ok"] = match_id_ok
    run_meta["match_id_skew_reject"] = match_id_skew_reject
    run_meta["match_fuzzy_ok"] = match_fuzzy_ok

    run_meta["status"] = "ok"
    run_meta["rows_written"] = len(rows)
    run_meta["rows_written_all"] = len(rows_all)
    run_meta["orderbook_calls"] = orderbook_calls
    run_meta["debug_matched_rows"] = len(debug_matched)
    run_meta["debug_unmatched_rows"] = len(debug_unmatched)
    run_meta["debug_series_count_rows"] = len(debug_series_counts)
    if debug_artifacts:
        run_meta["debug_candidates_rows"] = len(debug_candidates)
    _write_json(meta_path, run_meta)
    _copy_text(meta_path, meta_latest)

    if pipeline_report_enabled:
        pipe_path = os.path.join(outdir, f"kalshi_pipeline_report_{stamp}.json")
        pipe_latest = os.path.join(outdir, "kalshi_pipeline_report_latest.json")
        payload = {
            "stamp_utc": stamp,
            "run_meta_key_counts": {
                "odds_events_total": run_meta.get("odds_events_total"),
                "markets_fetched_total": run_meta.get("markets_fetched_total"),
                "markets_seen_total": run_meta.get("markets_seen_total"),
                "rows_written": run_meta.get("rows_written"),
            },
            "drop_counts": run_meta.get("drop_counts") or {},
            "stages": pipeline_stages,
        }
        _write_json(pipe_path, payload)
        _copy_text(pipe_path, pipe_latest)

    print(f"Wrote {len(rows)} rows to {csv_path} and {json_path}")

    webhook = env_str("DISCORD_WEBHOOK_URL", "")
    always_notify = env_bool("ALWAYS_NOTIFY", False)

    def _diag_summary() -> str:
        # Keep this short; Discord truncates. The full meta JSON has everything.
        dc = run_meta.get("drop_counts") or {}

        def g(k: str) -> int:
            try:
                return int(dc.get(k) or 0)
            except Exception:
                return 0

        parts = [
            f"events={run_meta.get('odds_events_total', '?')}",
            f"markets_seen={run_meta.get('markets_seen_total', '?')}",
            f"id_ok={run_meta.get('match_id_ok', 0)}",
            f"id_skew_rej={run_meta.get('match_id_skew_reject', 0)}",
            f"fuzzy_ok={run_meta.get('match_fuzzy_ok', 0)}",
            f"drop_odds_match={g('odds_match_failed')}",
            f"drop_time_skew={g('odds_match_time_skew')}",
            f"drop_fair={g('fair_prob_unavailable')}",
            f"drop_books={g('books_below_min')}",
            f"drop_edge={g('edge_below_min')}",
        ]
        return " | ".join(parts)

    # Discord notifications:
    # - Never post in DEBUG_MODE (validation runs)
    # - Never post in NO_FILTERS / WIDE_SCAN modes (can include junk)
    if debug_mode or no_filters or debug_wide_scan:
        print("Debug/no-filter run: wrote outputs but skipping Discord post.")
    elif webhook and (rows or always_notify):
        header_line = (
            f"**Kalshi value scanner**\n"
            f"Source: `{sportsbook_source}` | min_books={min_books_effective}\n"
            "Books shown per play reflect available lines\n"
        )
        if not rows:
            vol_note = f"vol≥{min_volume_effective}"
            if min_volume_effective > 0 and include_unknown_volume:
                vol_note = f"{vol_note} (unknown vol included)"
            elif min_volume_effective > 0 and not include_unknown_volume:
                vol_note = f"{vol_note} (unknown vol excluded)"
            post_discord(
                webhook,
                (
                    f"{header_line}\n"
                    f"**Kalshi value**: no plays found (min {min_edge*100:.1f}pp, "
                    f"fee +{fee_cents}¢, games ≤{game_lookahead_hours}h, {vol_note})."
                    f"\n_{_diag_summary()}_"
                ),
            )
            return 0

        def fmt_start(iso: str) -> str:
            dt = _parse_iso(iso)
            if not dt:
                return ""
            et = dt.astimezone(ZoneInfo("America/New_York"))
            return et.strftime("%b %d %I:%M %p ET")

        grouped: Dict[Tuple[str, str], List[Row]] = {}
        for r in rows:
            grouped.setdefault((r.event_title, r.commence_time_iso), []).append(r)

        lines: List[str] = []
        lines.append(header_line)
        lines.append(
            (
                f"**Kalshi value**: {len(rows)} plays (min {min_edge*100:.1f}pp, fee +{fee_cents}¢, "
                f"games ≤{game_lookahead_hours}h, vol≥{min_volume_effective})."
            )
        )

        game_items = sorted(
            grouped.items(),
            key=lambda kv: max(x.edge for x in kv[1]),
            reverse=True,
        )
        # Keep Discord payload readable and non-truncating
        max_len = 1850
        cur_len = sum(len(x) + 1 for x in lines)
        for (title, iso), items in game_items:
            items.sort(key=lambda x: x.edge, reverse=True)
            start = fmt_start(iso)
            header = f"\n**{title}** — {start}" if start else f"\n**{title}**"
            if cur_len + len(header) + 1 > max_len:
                break
            lines.append(header)
            cur_len += len(header) + 1
            for it in items:
                buy_cents = int(round(it.all_in_buy_prob * 100))
                fair_cents = int(round(it.fair_prob_side * 100))
                line1 = (
                    f"- {it.line_type.upper()} {it.line_label}: buy {it.side_to_buy} @{buy_cents}¢ "
                    f"vs fair {fair_cents}¢ → **{it.edge*100:+.1f}pp** | vol {it.kalshi_volume} | "
                    f"books {it.books_used}"
                )
                line2 = f"  {it.url}"
                if cur_len + len(line1) + len(line2) + 2 > max_len:
                    break
                lines.append(line1)
                cur_len += len(line1) + 1
                lines.append(line2)
                cur_len += len(line2) + 1

        if cur_len <= max_len - 80:
            lines.append("\n*(If this looks cut short, see workflow artifacts for the full CSV/JSON.)*")

        post_discord(webhook, "\n".join(lines))
    elif webhook and not rows:
        print("No edges above MIN_EDGE; skipping Discord post.")

    # Print top results for logs
    for r in rows[: min(10, len(rows))]:
        print(
            f"{r.edge*100:6.2f}pp | BUY {r.side_to_buy:>3} | all-in {r.all_in_buy_prob*100:5.1f}c | fair {r.fair_prob_side*100:5.1f}% | {r.line_type:>6} | {r.event_title} | {r.line_label}"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(scan())
