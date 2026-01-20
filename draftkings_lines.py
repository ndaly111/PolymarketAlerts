"""
DraftKings sportsbook fallback using sportsbook-nash sportscontent API.

This is the SAME backend used by the DraftKings frontend and by the
public DraftKings API Explorer repositories.

No API key required.
No cookies required.
No account required.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, List

import requests

try:
    from curl_cffi import requests as curl_requests
except Exception:
    curl_requests = None


SPORT_CONTENT_BASE = "https://sportsbook-nash.draftkings.com/api/sportscontent/dkusoh/v1"

DK_LEAGUES = {
    "basketball_nba": "nba",
    "basketball_ncaab": "cbk",
    "americanfootball_nfl": "nfl",
    "americanfootball_ncaaf": "cfb",
    "baseball_mlb": "mlb",
    "icehockey_nhl": "nhl",
}


def _headers() -> Dict[str, str]:
    return {
        "User-Agent": (
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0.0.0 Safari/537.36"
        ),
        "Accept": "application/json,text/plain,*/*",
        "Referer": "https://sportsbook.draftkings.com/",
    }


def _http_json(url: str) -> Dict[str, Any]:
    if curl_requests:
        r = curl_requests.get(url, headers=_headers(), impersonate="chrome120", timeout=20)
        r.raise_for_status()
        return r.json()

    r = requests.get(url, headers=_headers(), timeout=20)
    r.raise_for_status()
    return r.json()


def _iso(ts: Any) -> str:
    try:
        if isinstance(ts, str):
            dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
        else:
            dt = datetime.fromtimestamp(int(ts) / 1000, tz=timezone.utc)
        return dt.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")
    except Exception:
        return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _parse_american(val: Any) -> int:
    if val is None:
        return 0
    s = str(val).strip()
    s = s.replace("−", "-")
    if s.startswith("+"):
        s = s[1:]
    try:
        return int(s)
    except Exception:
        return 0


def _market_key(label: str) -> str | None:
    s = label.lower()
    if "moneyline" in s:
        return "h2h"
    if "spread" in s or "run line" in s or "puck line" in s:
        return "spreads"
    if "total" in s:
        return "totals"
    return None


def fetch_all_draftkings_events(sport_keys: List[str]) -> List[Dict[str, Any]]:
    """
    Returns events shaped exactly like The Odds API:

    {
      id
      sport_key
      commence_time
      home_team
      away_team
      bookmakers: [
        {
          key: draftkings
          markets: [h2h, spreads, totals]
        }
      ]
    }
    """

    out: List[Dict[str, Any]] = []

    for sport_key in sport_keys:
        league = DK_LEAGUES.get(sport_key)
        if not league:
            continue

        url = f"{SPORT_CONTENT_BASE}/leagues/{league}/categories/100"

        try:
            data = _http_json(url)
        except Exception:
            continue

        events = data.get("events", [])

        for ev in events:
            eid = ev.get("eventId")
            if not eid:
                continue

            home = None
            away = None

            for p in ev.get("participants", []):
                if p.get("venueRole") == "Home":
                    home = p.get("name")
                elif p.get("venueRole") == "Away":
                    away = p.get("name")

            if not home or not away:
                continue

            markets = []

            markets_src = ev.get("markets")
            if not markets_src:
                for group in ev.get("displayGroups", []) or []:
                    markets_src = group.get("markets")
                    if markets_src:
                        break

            for m in markets_src or []:
                mkey = _market_key(m.get("label", ""))
                if not mkey:
                    continue

                outcomes = []

                for sel in m.get("selections", []):
                    odds = _parse_american(sel.get("displayOdds", {}).get("american"))
                    if odds == 0:
                        continue

                    name = sel.get("label", "")

                    if mkey == "totals":
                        nl = name.lower()
                        if nl.startswith("over"):
                            name = "Over"
                        elif nl.startswith("under"):
                            name = "Under"

                    o = {"name": name, "price": odds}

                    if mkey in ("spreads", "totals"):
                        try:
                            o["point"] = float(sel.get("points"))
                        except Exception:
                            pass

                    outcomes.append(o)

                if outcomes:
                    markets.append({"key": mkey, "outcomes": outcomes})

            if not markets:
                continue

            out.append(
                {
                    "id": f"dk_{eid}",
                    "sport_key": sport_key,
                    "commence_time": _iso(ev.get("startEventDate")),
                    "home_team": home,
                    "away_team": away,
                    "bookmakers": [
                        {
                            "key": "draftkings",
                            "title": "DraftKings",
                            "markets": markets,
                        }
                    ],
                }
            )

    return out
