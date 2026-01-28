"""
ESPN free odds API - returns DraftKings lines without needing direct DK access.

This is FREE, no API key required, and works reliably.
Returns data in the same format as Odds API for easy integration.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import requests


# ESPN sport endpoints
ESPN_SPORTS = {
    "basketball_nba": "basketball/nba",
    "basketball_ncaab": "basketball/mens-college-basketball",
    "americanfootball_nfl": "football/nfl",
    "americanfootball_ncaaf": "football/college-football",
    "icehockey_nhl": "hockey/nhl",
    "baseball_mlb": "baseball/mlb",
}


def _parse_american(odds_str: str) -> int:
    """Parse American odds string to int."""
    if not odds_str:
        return 0
    s = str(odds_str).strip().replace("−", "-")
    if s.startswith("+"):
        s = s[1:]
    try:
        return int(s)
    except ValueError:
        return 0


def fetch_espn_odds(sport_keys: List[str]) -> List[Dict[str, Any]]:
    """
    Fetch odds from ESPN's free API.

    Returns events in Odds API format for compatibility:
    {
        id, sport_key, commence_time, home_team, away_team,
        bookmakers: [{key, title, markets: [{key, outcomes}]}]
    }
    """
    events = []

    for sport_key in sport_keys:
        espn_path = ESPN_SPORTS.get(sport_key)
        if not espn_path:
            continue

        url = f"https://site.api.espn.com/apis/site/v2/sports/{espn_path}/scoreboard"

        try:
            r = requests.get(url, timeout=15)
            r.raise_for_status()
            data = r.json()
        except Exception as e:
            print(f"  [ESPN] Failed to fetch {sport_key}: {e}")
            continue

        for ev in data.get("events", []):
            comp = ev.get("competitions", [{}])[0]
            odds_list = comp.get("odds", [])

            if not odds_list:
                continue

            odds = odds_list[0]  # Usually DraftKings

            # Get teams
            competitors = comp.get("competitors", [])
            home_team = None
            away_team = None
            for c in competitors:
                if c.get("homeAway") == "home":
                    home_team = c.get("team", {}).get("displayName")
                elif c.get("homeAway") == "away":
                    away_team = c.get("team", {}).get("displayName")

            if not home_team or not away_team:
                continue

            # Build markets
            markets = []

            # Moneyline
            ml_data = odds.get("moneyline", {})
            if ml_data:
                home_ml = _parse_american(ml_data.get("home", {}).get("close", {}).get("odds", ""))
                away_ml = _parse_american(ml_data.get("away", {}).get("close", {}).get("odds", ""))
                if home_ml and away_ml:
                    markets.append({
                        "key": "h2h",
                        "outcomes": [
                            {"name": home_team, "price": home_ml},
                            {"name": away_team, "price": away_ml},
                        ]
                    })

            # Spread
            spread_data = odds.get("pointSpread", {})
            if spread_data:
                home_spread = spread_data.get("home", {}).get("close", {})
                away_spread = spread_data.get("away", {}).get("close", {})
                home_line = home_spread.get("line", "")
                away_line = away_spread.get("line", "")
                home_odds = _parse_american(home_spread.get("odds", ""))
                away_odds = _parse_american(away_spread.get("odds", ""))

                if home_line and away_line and home_odds and away_odds:
                    try:
                        markets.append({
                            "key": "spreads",
                            "outcomes": [
                                {"name": home_team, "price": home_odds, "point": float(home_line)},
                                {"name": away_team, "price": away_odds, "point": float(away_line)},
                            ]
                        })
                    except ValueError:
                        pass

            # Total
            total_data = odds.get("total", {})
            if total_data:
                over_data = total_data.get("over", {}).get("close", {})
                under_data = total_data.get("under", {}).get("close", {})
                over_line = over_data.get("line", "")
                under_line = under_data.get("line", "")
                over_odds = _parse_american(over_data.get("odds", ""))
                under_odds = _parse_american(under_data.get("odds", ""))

                # Extract numeric total from "o235.5" or "u235.5"
                try:
                    total_val = float(over_line.replace("o", "").replace("O", ""))
                    if over_odds and under_odds:
                        markets.append({
                            "key": "totals",
                            "outcomes": [
                                {"name": "Over", "price": over_odds, "point": total_val},
                                {"name": "Under", "price": under_odds, "point": total_val},
                            ]
                        })
                except (ValueError, AttributeError):
                    pass

            if not markets:
                continue

            # Get commence time
            commence_time = ev.get("date", "")

            events.append({
                "id": f"espn_{ev.get('id', '')}",
                "sport_key": sport_key,
                "commence_time": commence_time,
                "home_team": home_team,
                "away_team": away_team,
                "bookmakers": [{
                    "key": "draftkings",
                    "title": "DraftKings (via ESPN)",
                    "markets": markets,
                }]
            })

    return events


if __name__ == "__main__":
    # Test
    sports = ["basketball_nba", "icehockey_nhl", "americanfootball_nfl"]
    events = fetch_espn_odds(sports)
    print(f"Found {len(events)} events with odds")

    for ev in events[:5]:
        print(f"\n{ev['sport_key']}: {ev['away_team']} @ {ev['home_team']}")
        for bm in ev["bookmakers"]:
            for mkt in bm["markets"]:
                print(f"  {mkt['key']}: {mkt['outcomes']}")
