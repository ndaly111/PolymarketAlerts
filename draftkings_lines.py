"""DraftKings sportsbook lines (public endpoints).

This module is intentionally shaped to look like The Odds API payload:

  {
    "id": "dk_<eventId>",
    "sport_key": "basketball_nba",
    "commence_time": "2026-01-20T00:00:00Z",
    "home_team": "...",
    "away_team": "...",
    "bookmakers": [
      {
        "key": "draftkings",
        "title": "DraftKings",
        "markets": [
          {"key": "h2h", "outcomes": [{"name": "...", "price": -110}, ...]},
          {"key": "spreads", "outcomes": [{"name": "...", "price": -110, "point": -2.5}, ...]},
          {"key": "totals", "outcomes": [{"name": "Over", "price": -110, "point": 221.5}, ...]},
        ],
      }
    ],
  }

Key detail: the v5 "eventgroups/{gid}?format=json" response typically includes offer category
metadata under eventGroup.offerCategories, but the actual offers for a category are returned by:

  /eventgroups/{gid}/categories/{categoryId}?format=json

So we:
  1) fetch the eventgroup to discover the "Game Lines" category id
  2) fetch that category payload and parse Moneyline/Spread/Total offers by eventId

When enabled, we write a small structural debug summary to out/dk_debug/.

Notes:
  • DraftKings sometimes blocks vanilla requests (403). If curl_cffi is installed, we use it
    with browser impersonation as a best-effort workaround.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import requests

try:
    # Optional: better success rate vs DK anti-bot (TLS fingerprinting)
    from curl_cffi import requests as curl_requests  # type: ignore
except Exception:  # pragma: no cover
    curl_requests = None


DK_EVENTGROUPS: Dict[str, int] = {
    "americanfootball_nfl": 88808,
    "americanfootball_ncaaf": 87637,
    "basketball_nba": 42648,
    "basketball_ncaab": 92483,
    "baseball_mlb": 84240,
    "icehockey_nhl": 42133,
}

DK_EVENTGROUP_URL = "https://sportsbook.draftkings.com/sites/US-SB/api/v5/eventgroups/{gid}?format=json"
DK_CATEGORY_URL = (
    "https://sportsbook.draftkings.com/sites/US-SB/api/v5/eventgroups/{gid}/categories/{cid}?format=json"
)


def _env_bool(name: str, default: bool = False) -> bool:
    v = os.getenv(name)
    if v is None:
        return default
    s = str(v).strip().lower()
    return s in ("1", "true", "t", "yes", "y", "on")


def _parse_american(val: Any) -> int:
    if val is None:
        return 0
    s = str(val).strip()
    if not s:
        return 0
    if s.startswith("+"):
        s = s[1:]
    try:
        return int(s)
    except Exception:
        return 0


def _iso_z(ts: Any) -> str:
    if not ts:
        return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    # DK usually gives ms epoch, but some payloads may include ISO strings.
    try:
        if isinstance(ts, str) and "T" in ts:
            dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")
    except Exception:
        pass
    # Epoch (ms)
    try:
        dt = datetime.fromtimestamp(int(ts) / 1000.0, tz=timezone.utc)
        return dt.isoformat().replace("+00:00", "Z")
    except Exception:
        return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _pick_team_names(ev: Dict[str, Any]) -> Tuple[str, str]:
    # DK uses teamName1/teamName2; treat as home/away for matching purposes.
    home = ev.get("teamName1") or ev.get("teamShortName1") or ""
    away = ev.get("teamName2") or ev.get("teamShortName2") or ""
    return str(home), str(away)


def _dk_headers() -> Dict[str, str]:
    # Keep this light; curl_cffi provides the important browser-like TLS fingerprint.
    return {
        "User-Agent": (
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0.0.0 Safari/537.36"
        ),
        "Accept": "application/json,text/plain,*/*",
        "Accept-Language": "en-US,en;q=0.9",
        "Referer": "https://sportsbook.draftkings.com/",
    }


def _http_get_json(url: str, timeout: int = 20) -> Tuple[Dict[str, Any], int]:
    headers = _dk_headers()

    # Prefer curl_cffi if available (higher success rate vs DK).
    if curl_requests is not None and not _env_bool("DK_NO_CURL_CFFI", False):
        r = curl_requests.get(url, headers=headers, timeout=timeout, impersonate="chrome120")
        status = getattr(r, "status_code", 0) or 0
        if status >= 400:
            # curl_cffi response has .text too
            raise requests.HTTPError(f"HTTP {status} for {url}: {str(getattr(r,'text',''))[:200]}")
        return r.json(), status

    # Fallback: vanilla requests
    r2 = requests.get(url, headers=headers, timeout=timeout)
    status2 = r2.status_code
    if status2 >= 400:
        raise requests.HTTPError(f"HTTP {status2} for {url}: {r2.text[:200]}")
    return r2.json(), status2


def _summarize_eventgroup(data: Dict[str, Any]) -> Dict[str, Any]:
    eg = data.get("eventGroup") or {}
    oc = eg.get("offerCategories") or data.get("offerCategories") or []
    events = data.get("events") or eg.get("events") or []
    cats = []
    for c in oc[:40]:
        cats.append(
            {
                "name": c.get("name"),
                "offerCategoryId": c.get("offerCategoryId") or c.get("id") or c.get("categoryId"),
            }
        )
    return {
        "top_keys": sorted(list(data.keys()))[:60],
        "eventGroup_keys": sorted(list(eg.keys()))[:60] if isinstance(eg, dict) else [],
        "events_len": len(events) if isinstance(events, list) else 0,
        "offerCategories_len": len(oc) if isinstance(oc, list) else 0,
        "offerCategories_sample": cats,
    }


def _summarize_category(data: Dict[str, Any]) -> Dict[str, Any]:
    eg = data.get("eventGroup") or {}
    events = data.get("events") or eg.get("events") or []
    oc = eg.get("offerCategories") or data.get("offerCategories") or []
    first = None
    for c in oc:
        if isinstance(c, dict) and (c.get("offerSubcategoryDescriptors") or c.get("name")):
            first = c
            break
    sub_desc = (first or {}).get("offerSubcategoryDescriptors") or []
    sample_offer = None
    for sc in sub_desc[:3]:
        sub = (sc or {}).get("offerSubcategory") or {}
        offers = sub.get("offers") or []
        if offers and offers[0] and offers[0][0]:
            sample_offer = offers[0][0]
            break
    return {
        "top_keys": sorted(list(data.keys()))[:60],
        "eventGroup_keys": sorted(list(eg.keys()))[:60] if isinstance(eg, dict) else [],
        "events_len": len(events) if isinstance(events, list) else 0,
        "offerCategories_len": len(oc) if isinstance(oc, list) else 0,
        "first_offerCategory_name": (first or {}).get("name"),
        "first_offerCategory_id": (first or {}).get("offerCategoryId") or (first or {}).get("id"),
        "offerSubcategoryDescriptors_len": len(sub_desc) if isinstance(sub_desc, list) else 0,
        "sample_offer_keys": sorted(list(sample_offer.keys()))[:80] if isinstance(sample_offer, dict) else None,
        "sample_offer_label": sample_offer.get("label") if isinstance(sample_offer, dict) else None,
        "sample_offer_eventId": sample_offer.get("eventId") if isinstance(sample_offer, dict) else None,
        "sample_outcome_keys": (
            sorted(list((sample_offer.get("outcomes") or [{}])[0].keys()))[:80]
            if isinstance(sample_offer, dict) and sample_offer.get("outcomes")
            else None
        ),
    }


def _ensure_dir(path: str) -> None:
    try:
        os.makedirs(path, exist_ok=True)
    except Exception:
        pass


def _write_json(path: str, payload: Dict[str, Any]) -> None:
    try:
        _ensure_dir(os.path.dirname(path))
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
    except Exception:
        pass


def _find_game_lines_category_id(eventgroup_data: Dict[str, Any]) -> Optional[int]:
    eg = eventgroup_data.get("eventGroup") or {}
    offer_categories = eg.get("offerCategories") or eventgroup_data.get("offerCategories") or []
    if not isinstance(offer_categories, list):
        return None
    for cat in offer_categories:
        if not isinstance(cat, dict):
            continue
        name = str(cat.get("name", "")).strip().lower()
        if "game lines" in name:
            cid = cat.get("offerCategoryId") or cat.get("id") or cat.get("categoryId")
            try:
                return int(cid)
            except Exception:
                return None
    return None


def _market_key_from_label(label: str) -> Optional[str]:
    s = (label or "").strip().lower()
    if not s:
        return None
    if "moneyline" in s or s in ("money line", "ml"):
        return "h2h"
    if any(x in s for x in ("spread", "point spread", "run line", "puck line")):
        return "spreads"
    if any(
        x in s for x in ("total", "game total", "total points", "game total points", "total runs", "total goals")
    ):
        return "totals"
    return None


def fetch_draftkings_events_for_sport(sport_key: str) -> List[Dict[str, Any]]:
    gid = DK_EVENTGROUPS.get(sport_key)
    if not gid:
        return []

    debug_enabled = _env_bool("DK_DEBUG", False) or _env_bool("MASTER_DEBUG", False) or _env_bool("DEBUG_MODE", False)
    outdir = os.getenv("OUTDIR", "out")
    dbg_dir = os.path.join(outdir, "dk_debug")

    # 1) fetch eventgroup metadata and discover Game Lines category id
    eg_url = DK_EVENTGROUP_URL.format(gid=gid)
    eg_data, eg_status = _http_get_json(eg_url, timeout=25)

    cid = _find_game_lines_category_id(eg_data)
    if debug_enabled:
        _write_json(
            os.path.join(dbg_dir, f"dk_summary_eventgroup_{sport_key}_{gid}.json"),
            {"url": eg_url, "status": eg_status, "summary": _summarize_eventgroup(eg_data), "game_lines_cid": cid},
        )

    if cid is None:
        return []

    # 2) fetch category payload (contains offers)
    cat_url = DK_CATEGORY_URL.format(gid=gid, cid=cid)
    cat_data, cat_status = _http_get_json(cat_url, timeout=25)
    if debug_enabled:
        _write_json(
            os.path.join(dbg_dir, f"dk_summary_category_{sport_key}_{gid}_{cid}.json"),
            {"url": cat_url, "status": cat_status, "summary": _summarize_category(cat_data)},
        )

    # Build map eventId -> event basics
    events = cat_data.get("events") or (cat_data.get("eventGroup") or {}).get("events") or []
    ev_map: Dict[int, Dict[str, Any]] = {}
    if isinstance(events, list):
        for ev in events:
            if not isinstance(ev, dict):
                continue
            try:
                eid = int(ev.get("eventId"))
            except Exception:
                continue
            # Keep only OPEN events when status is present (avoid CLOSED/SUSPENDED noise).
            st = str(ev.get("eventStatus") or "").strip().upper()
            if st and st != "OPEN":
                continue
            home, away = _pick_team_names(ev)
            ev_map[eid] = {
                "id": f"dk_{eid}",
                "sport_key": sport_key,
                "commence_time": _iso_z(ev.get("startDate")),
                "home_team": home,
                "away_team": away,
                "bookmakers": [],
            }

    if not ev_map:
        return []

    # Find "Game Lines" offer category in category response (usually the only one)
    eg2 = cat_data.get("eventGroup") or {}
    offer_categories2 = eg2.get("offerCategories") or cat_data.get("offerCategories") or []
    game_lines_cat = None
    if isinstance(offer_categories2, list):
        for c in offer_categories2:
            if not isinstance(c, dict):
                continue
            name = str(c.get("name", "")).strip().lower()
            if "game lines" in name:
                game_lines_cat = c
                break
    if game_lines_cat is None and isinstance(offer_categories2, list) and offer_categories2:
        # Sometimes the category response only includes the category without a stable name match;
        # in that case, take the first and parse offers anyway.
        if isinstance(offer_categories2[0], dict):
            game_lines_cat = offer_categories2[0]

    if not isinstance(game_lines_cat, dict):
        return []

    subcats = game_lines_cat.get("offerSubcategoryDescriptors") or []
    if not isinstance(subcats, list):
        return []

    markets_by_event: Dict[int, List[Dict[str, Any]]] = {eid: [] for eid in ev_map.keys()}

    # Parse offers into markets per event
    for sc in subcats:
        if not isinstance(sc, dict):
            continue
        sub = sc.get("offerSubcategory") or {}
        if not isinstance(sub, dict):
            continue
        offers = sub.get("offers") or []
        if not isinstance(offers, list):
            continue

        for group in offers:
            if not group:
                continue
            for offer in group or []:
                if not isinstance(offer, dict):
                    continue

                mkey = _market_key_from_label(str(offer.get("label", "")))
                if mkey is None:
                    continue

                try:
                    eid = int(offer.get("eventId"))
                except Exception:
                    continue
                if eid not in markets_by_event:
                    continue

                outcomes = []
                for o in offer.get("outcomes", []) or []:
                    if not isinstance(o, dict):
                        continue
                    raw_name = o.get("label") or o.get("participant") or ""
                    name = str(raw_name).strip()
                    # Totals sometimes come back as "Over 221.5" / "Under 221.5" — normalize for downstream.
                    if mkey == "totals":
                        nl = name.lower()
                        if nl.startswith("over"):
                            name = "Over"
                        elif nl.startswith("under"):
                            name = "Under"
                    oc: Dict[str, Any] = {
                        "name": name,
                        "price": _parse_american(o.get("oddsAmerican")),
                    }
                    if mkey in ("spreads", "totals") and o.get("line") is not None:
                        try:
                            oc["point"] = float(o.get("line"))
                        except Exception:
                            pass
                    outcomes.append(oc)

                if outcomes:
                    markets_by_event[eid].append({"key": mkey, "outcomes": outcomes})

    # Attach DK as a single bookmaker per event
    out: List[Dict[str, Any]] = []
    for eid, ev in ev_map.items():
        mkts = markets_by_event.get(eid, [])
        if not mkts:
            continue
        ev["bookmakers"] = [{"key": "draftkings", "title": "DraftKings", "markets": mkts}]
        out.append(ev)
    return out


def fetch_all_draftkings_events(sport_keys: List[str]) -> List[Dict[str, Any]]:
    events_out: List[Dict[str, Any]] = []
    errors: List[Dict[str, Any]] = []

    for sk in sport_keys:
        try:
            events_out.extend(fetch_draftkings_events_for_sport(sk))
        except Exception as e:
            errors.append({"sport_key": sk, "error": repr(e)})

    # If we got nothing, emit a small hint so Actions logs aren’t silent.
    if not events_out and errors:
        try:
            msg = f"DraftKings fallback: {len(errors)} errors, first: {errors[0]}"
            print(msg)
        except Exception:
            pass

    return events_out
