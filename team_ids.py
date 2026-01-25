from __future__ import annotations

import json
import os
import re
from functools import lru_cache
from pathlib import Path
from typing import Dict, Optional, Set, Tuple

ODDS_SPORT_KEY_TO_PREFIX: Dict[str, str] = {
    "americanfootball_nfl": "nfl",
    "basketball_nba": "nba",
    "baseball_mlb": "mlb",
    "icehockey_nhl": "nhl",
    "americanfootball_ncaaf": "ncaaf",
    "basketball_ncaab": "ncaab",
}

# Kalshi uses slightly different sport prefixes for some leagues.
KALSHI_SPORT_PREFIX_ALIASES: Dict[str, str] = {
    "ncaab": "ncaamb",
}

# Fix known ID mismatches between Odds and Kalshi caches
ODDS_TO_KALSHI_ID: Dict[str, Dict[str, str]] = {
    "mlb": {"ath": "a", "chw": "cws", "wsh": "was"},
    "nhl": {"lak": "la", "njd": "nj", "sjs": "sj", "tbl": "tb"},
}


def sport_prefix_from_odds_key(sport_key: Optional[str]) -> Optional[str]:
    if not sport_key:
        return None
    return ODDS_SPORT_KEY_TO_PREFIX.get(str(sport_key).strip())


def _default_map_dir() -> Path:
    p = os.getenv("TEAM_MAP_DIR")
    if p and str(p).strip():
        return Path(str(p).strip())
    return Path(__file__).resolve().parent


def _load_json(path: Path) -> Dict[str, str]:
    obj = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(obj, dict) and isinstance(obj.get("mappings"), dict):
        return {str(k): str(v) for k, v in obj["mappings"].items()}
    if isinstance(obj, dict):
        return {str(k): str(v) for k, v in obj.items()}
    raise ValueError(f"Unexpected JSON format in {path}")


@lru_cache(maxsize=1)
def load_team_maps() -> Tuple[Dict[str, str], Dict[str, str]]:
    base = _default_map_dir()
    kalshi_path = Path(os.getenv("KALSHI_TEAM_MAP_PATH") or base / "kalshi_team_cache.json")
    odds_path = Path(os.getenv("ODDS_TEAM_MAP_PATH") or base / "theoddsapi_team_cache-2.json")
    kalshi_map: Dict[str, str] = {}
    odds_map: Dict[str, str] = {}
    try:
        if kalshi_path.exists():
            kalshi_map = _load_json(kalshi_path)
    except Exception:
        kalshi_map = {}
    try:
        if odds_path.exists():
            odds_map = _load_json(odds_path)
    except Exception:
        odds_map = {}
    return kalshi_map, odds_map


_slug_re = re.compile(r"[^a-z0-9]+")


def slugify(text: Optional[str]) -> str:
    if not text:
        return ""
    # Normalize punctuation variants (mobile keyboards often use “smart quotes”)
    s = str(text).strip().lower().replace("&", "and")
    s = s.replace("\u2019", "").replace("\u2018", "")
    s = s.replace("'", "")
    s = _slug_re.sub("_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s


def _to_kalshi_id(sport_prefix: str, raw_id: str) -> str:
    overrides = ODDS_TO_KALSHI_ID.get(sport_prefix, {})
    return overrides.get(raw_id, raw_id)


def _kalshi_sport_prefix(sport_prefix: str) -> str:
    return KALSHI_SPORT_PREFIX_ALIASES.get(sport_prefix, sport_prefix)


def _kalshi_slug_lookup(sport_prefix: str, slug: str, kalshi_map: Dict[str, str]) -> Optional[str]:
    if not slug:
        return None
    pref = _kalshi_sport_prefix(sport_prefix)
    candidates = [slug, slug.replace("_", "")]
    parts = [p for p in slug.split("_") if p]
    if parts:
        candidates.append(parts[-1])
    if len(parts) >= 2:
        candidates.append("_".join(parts[-2:]))
    for c in candidates:
        v = kalshi_map.get(f"{pref}:{c}")
        if v:
            return v
    return None


def odds_team_id(sport_key: Optional[str], team_name: Optional[str]) -> Optional[str]:
    sport = sport_prefix_from_odds_key(sport_key)
    if not sport:
        return None
    kalshi_map, odds_map = load_team_maps()
    slug = slugify(team_name)
    if not slug:
        return None
    v = odds_map.get(f"{sport}:{slug}")
    if v:
        return _to_kalshi_id(sport, v)
    slug2 = slug.replace("st_", "saint_")
    if slug2 != slug:
        v = odds_map.get(f"{sport}:{slug2}")
        if v:
            return _to_kalshi_id(sport, v)
    slug3 = slug.replace("saint_", "st_")
    if slug3 != slug:
        v = odds_map.get(f"{sport}:{slug3}")
        if v:
            return _to_kalshi_id(sport, v)
    if kalshi_map:
        v = _kalshi_slug_lookup(sport, slug, kalshi_map)
        if v:
            return v
        if slug2 != slug:
            v = _kalshi_slug_lookup(sport, slug2, kalshi_map)
            if v:
                return v
        if slug3 != slug:
            v = _kalshi_slug_lookup(sport, slug3, kalshi_map)
            if v:
                return v
    return None


def _substring_odds_id(sport: str, frag_slug: str, odds_map: Dict[str, str]) -> Optional[str]:
    if not frag_slug or len(frag_slug) < 4:
        return None
    pref = f"{sport}:"
    found: Set[str] = set()
    for k, v in odds_map.items():
        if not k.startswith(pref):
            continue
        if frag_slug in k:
            found.add(_to_kalshi_id(sport, v))
            if len(found) > 1:
                return None
    return next(iter(found)) if found else None


def kalshi_team_id(sport_prefix: str, team_text: Optional[str]) -> Optional[str]:
    if not sport_prefix:
        return None
    kalshi_map, odds_map = load_team_maps()
    slug = slugify(team_text)
    if not slug:
        return None

    # Common edge-case nicknames/abbreviations used in some Kalshi titles.
    if sport_prefix == "mlb" and slug == "as":
        return "a"

    v = _kalshi_slug_lookup(sport_prefix, slug, kalshi_map)
    if v:
        return v
    if odds_map:
        inferred = _substring_odds_id(sport_prefix, slug, odds_map)
        if inferred:
            return inferred
    return None


def odds_event_team_ids(event: Dict[str, object]) -> Tuple[Optional[str], Optional[str]]:
    try:
        sport_key = event.get("sport_key")  # type: ignore[attr-defined]
        home = event.get("home_team")  # type: ignore[attr-defined]
        away = event.get("away_team")  # type: ignore[attr-defined]
    except Exception:
        return (None, None)
    return (
        odds_team_id(str(sport_key) if sport_key is not None else None, str(home) if home is not None else None),
        odds_team_id(str(sport_key) if sport_key is not None else None, str(away) if away is not None else None),
    )
