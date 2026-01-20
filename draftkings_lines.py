from datetime import datetime, timezone
from typing import Any, Dict, List

import requests

DK_EVENTGROUPS = {
    "americanfootball_nfl": 88808,
    "americanfootball_ncaaf": 87637,
    "basketball_nba": 42648,
    "basketball_ncaab": 92483,
    "baseball_mlb": 84240,
    "icehockey_nhl": 42133,
}

DK_URL = "https://sportsbook.draftkings.com/sites/US-SB/api/v5/eventgroups/{gid}?format=json"


def _parse_american(value: Any) -> int:
    if value is None:
        raise ValueError("missing american odds")
    text = str(value).strip()
    if not text:
        raise ValueError("empty american odds")
    if text.startswith("+"):
        text = text[1:]
    return int(text)


def _american_from_decimal(dec: float) -> int:
    if dec <= 1.0:
        raise ValueError("decimal odds must be > 1.0")
    if dec >= 2.0:
        return int(round((dec - 1.0) * 100.0))
    return int(round(-100.0 / (dec - 1.0)))


def _outcome_price_american(outcome: Dict[str, Any]) -> int:
    if "oddsAmerican" in outcome and outcome.get("oddsAmerican") is not None:
        return _parse_american(outcome.get("oddsAmerican"))
    dec = outcome.get("oddsDecimal")
    if dec is None:
        raise ValueError("no oddsDecimal/oddsAmerican")
    return _american_from_decimal(float(dec))


def _normalize_start(value: str) -> str:
    if not value:
        return value
    try:
        dt = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except Exception:
        return value
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def fetch_all_draftkings_events(sport_keys: List[str]) -> List[Dict[str, Any]]:
    events_out: List[Dict[str, Any]] = []

    for sport_key in sport_keys:
        gid = DK_EVENTGROUPS.get(sport_key)
        if not gid:
            continue

        try:
            url = DK_URL.format(gid=gid)
            response = requests.get(url, timeout=15)
            response.raise_for_status()
            data = response.json()
        except Exception:
            continue

        events = data.get("events", []) or []
        offer_categories = data.get("offerCategories", []) or []

        for ev in events:
            if ev.get("eventStatus") != "OPEN":
                continue

            home = ev.get("teamName1")
            away = ev.get("teamName2")
            start = _normalize_start(ev.get("startDate") or "")

            markets: List[Dict[str, Any]] = []

            for cat in offer_categories:
                if cat.get("name") != "Game Lines":
                    continue

                for sub in cat.get("offerSubcategoryDescriptors", []) or []:
                    offers = (sub.get("offerSubcategory") or {}).get("offers", [])

                    for group in offers:
                        for offer in group:
                            label = str(offer.get("label") or "").lower()
                            label_clean = " ".join(label.split())
                            outcomes = []

                            for o in offer.get("outcomes", []) or []:
                                try:
                                    price_am = _outcome_price_american(o)
                                except Exception:
                                    continue

                                outcomes.append(
                                    {
                                        "name": o.get("label"),
                                        "price": price_am,
                                        "point": o.get("line"),
                                    }
                                )

                            if not outcomes:
                                continue

                            if label_clean in ("moneyline", "money line"):
                                markets.append({"key": "h2h", "outcomes": outcomes})
                            elif label_clean in ("spread", "point spread", "run line", "puck line"):
                                markets.append({"key": "spreads", "outcomes": outcomes})
                            elif label_clean in ("total", "total points", "game total", "game total points"):
                                markets.append({"key": "totals", "outcomes": outcomes})

            if not markets:
                continue

            events_out.append(
                {
                    "id": f"dk_{ev.get('eventId')}",
                    "sport_key": sport_key,
                    "commence_time": start,
                    "home_team": home,
                    "away_team": away,
                    "bookmakers": [{"key": "draftkings", "title": "DraftKings", "markets": markets}],
                }
            )

    return events_out
