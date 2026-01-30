"""
Player and team name normalization utilities.

Consolidates duplicated normalization functions from:
- kalshi_props.py
- oddsapi_props.py
- oddsapi_lines.py
"""

from __future__ import annotations

import re
from typing import Optional


def normalize_player_name(s: str) -> str:
    """
    Normalize player name for matching.

    Handles common variations:
    - Case (LeBron James -> lebron james)
    - Punctuation (D.J. -> dj)
    - Apostrophes (O'Brien -> obrien)
    - Hyphens (Davis-Smith -> davis smith)
    - Extra spaces

    Args:
        s: Raw player name

    Returns:
        Normalized name for comparison

    Examples:
        >>> normalize_player_name("LeBron James")
        "lebron james"
        >>> normalize_player_name("D.J. Moore")
        "dj moore"
        >>> normalize_player_name("Ja'Marr Chase")
        "jamarr chase"
    """
    if not s:
        return ""
    return (
        s.lower()
        .replace(".", "")
        .replace(",", "")
        .replace("'", "")
        .replace("-", " ")
        .replace("  ", " ")
        .strip()
    )


def normalize_team_name(s: Optional[str]) -> str:
    """
    Normalize team name for matching.

    Handles common variations:
    - Case
    - Ampersands (Texas A&M -> texas and m)
    - Punctuation
    - Extra spaces

    Args:
        s: Raw team name

    Returns:
        Normalized name for comparison

    Examples:
        >>> normalize_team_name("Los Angeles Lakers")
        "los angeles lakers"
        >>> normalize_team_name("Texas A&M")
        "texas and m"
    """
    if not s:
        return ""
    return (
        s.lower()
        .replace("&", "and")
        .replace(".", "")
        .replace(",", "")
        .replace("  ", " ")
        .strip()
    )


def name_similarity(a: str, b: str) -> float:
    """
    Calculate similarity score between two names (0.0-1.0).

    Uses token-based comparison with stop word removal
    for robust team/player matching.

    Args:
        a: First name
        b: Second name

    Returns:
        Similarity score from 0.0 (no match) to 1.0 (exact match)

    Examples:
        >>> name_similarity("Los Angeles Lakers", "Lakers")
        0.9  # Substring match
        >>> name_similarity("Miami Heat", "Miami Heat")
        1.0  # Exact match
        >>> name_similarity("Bulls", "Bears")
        0.0  # No match
    """
    na = normalize_team_name(a)
    nb = normalize_team_name(b)

    if not na or not nb:
        return 0.0

    # Exact match
    if na == nb:
        return 1.0

    # Substring match
    if na in nb or nb in na:
        short, long_ = (na, nb) if len(na) <= len(nb) else (nb, na)
        if len(short) >= 4:
            return 0.90
        if len(short) == 3 and long_.startswith(short):
            return 0.85

    # Token-based matching with stop word removal
    stop_words = {
        "pro", "professional", "football", "basketball", "baseball", "hockey",
        "game", "match", "season", "playoffs", "postseason",
        "nfl", "nba", "mlb", "nhl", "ncaaf", "ncaab",
        "the", "a", "an",
    }

    ta = [t for t in na.split() if t not in stop_words]
    tb = [t for t in nb.split() if t not in stop_words]

    if not ta or not tb:
        ta = na.split()
        tb = nb.split()

    # Jaccard similarity
    sa = set(ta)
    sb = set(tb)
    inter = len(sa & sb)
    union = len(sa | sb)

    if union == 0:
        return 0.0

    return inter / union


def extract_player_from_title(title: str) -> Optional[str]:
    """
    Extract player name from Kalshi title.

    Handles format: "Player Name: X+ stat"

    Args:
        title: Kalshi market title

    Returns:
        Player name or None if not parseable

    Example:
        >>> extract_player_from_title("LaMelo Ball: 8+ rebounds")
        "LaMelo Ball"
    """
    if not title:
        return None

    # Pattern: "Player Name: X+ stat"
    match = re.match(r"^([^:]+):\s*\d+", title)
    if match:
        return match.group(1).strip()
    return None


def extract_line_from_title(title: str) -> Optional[float]:
    """
    Extract line/threshold from Kalshi title.

    Handles format: "Player Name: X+ stat"

    Args:
        title: Kalshi market title

    Returns:
        Line value or None if not parseable

    Example:
        >>> extract_line_from_title("LaMelo Ball: 8+ rebounds")
        8.0
    """
    if not title:
        return None

    match = re.search(r":\s*(\d+(?:\.\d+)?)\+", title)
    if match:
        return float(match.group(1))
    return None


def fuzzy_name_match(
    name1: str,
    name2: str,
    threshold: float = 0.70,
) -> bool:
    """
    Check if two names match using fuzzy comparison.

    Args:
        name1: First name
        name2: Second name
        threshold: Minimum similarity to consider a match

    Returns:
        True if names match above threshold
    """
    return name_similarity(name1, name2) >= threshold
