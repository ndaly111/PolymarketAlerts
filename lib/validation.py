"""
Trade validation utilities.

Price guardrails, edge sanity checks, and ticker/title validation.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional, List


@dataclass
class TradeValidation:
    """
    Result of a trade validation check.

    Attributes:
        is_valid: Whether the validation passed
        is_blocking: Whether this should block the trade (vs just flag)
        message: Human-readable explanation
        code: Machine-readable validation code
    """
    is_valid: bool
    is_blocking: bool
    message: str
    code: str

    @classmethod
    def valid(cls) -> "TradeValidation":
        """Create a passing validation result."""
        return cls(is_valid=True, is_blocking=False, message="", code="OK")

    @classmethod
    def invalid_blocking(cls, message: str, code: str) -> "TradeValidation":
        """Create a blocking validation failure."""
        return cls(is_valid=False, is_blocking=True, message=message, code=code)

    @classmethod
    def invalid_warning(cls, message: str, code: str) -> "TradeValidation":
        """Create a non-blocking validation warning (flag but continue)."""
        return cls(is_valid=False, is_blocking=False, message=message, code=code)


def validate_price(
    price_cents: int,
    min_cents: int = 5,
    max_cents: int = 95,
) -> TradeValidation:
    """
    Validate price is within acceptable range.

    Blocks trades at extreme prices which are:
    - < 5 cents: Too risky, often bad data or illiquid
    - > 95 cents: Too expensive, minimal upside

    Args:
        price_cents: Price in cents (1-99)
        min_cents: Minimum acceptable price (default 5)
        max_cents: Maximum acceptable price (default 95)

    Returns:
        TradeValidation result

    Example:
        >>> validate_price(3).is_valid
        False  # 3 cent price blocked
        >>> validate_price(50).is_valid
        True  # Normal price OK
    """
    if price_cents < min_cents:
        return TradeValidation.invalid_blocking(
            f"Price {price_cents}¢ below minimum {min_cents}¢ (too risky/illiquid)",
            "PRICE_TOO_LOW"
        )

    if price_cents > max_cents:
        return TradeValidation.invalid_blocking(
            f"Price {price_cents}¢ above maximum {max_cents}¢ (minimal upside)",
            "PRICE_TOO_HIGH"
        )

    return TradeValidation.valid()


def validate_edge(
    edge: float,
    min_edge: float = 0.05,
    suspicious_threshold: float = 0.30,
) -> TradeValidation:
    """
    Validate edge is reasonable.

    - Edge below minimum: Block (not profitable enough)
    - Edge above suspicious threshold: Flag but don't block (likely bad data)

    Args:
        edge: Calculated edge (0.0-1.0)
        min_edge: Minimum edge to trade (default 5%)
        suspicious_threshold: Flag edges above this (default 30%)

    Returns:
        TradeValidation result
    """
    if edge < min_edge:
        return TradeValidation.invalid_blocking(
            f"Edge {edge:.1%} below minimum {min_edge:.1%}",
            "EDGE_TOO_LOW"
        )

    if edge > suspicious_threshold:
        return TradeValidation.invalid_warning(
            f"Edge {edge:.1%} above suspicious threshold {suspicious_threshold:.1%} (may be bad data)",
            "EDGE_SUSPICIOUS"
        )

    return TradeValidation.valid()


def validate_ticker_matchup(
    ticker: str,
    event_title: str,
) -> TradeValidation:
    """
    Validate that ticker matches the event title.

    Prevents trading on wrong events when ticker contains team info.

    Examples:
        - KXNBAGAME-CHI-BKN should match "Chicago at Brooklyn"
        - KXNBASPREAD-LAL should match title with "Lakers"

    Args:
        ticker: Kalshi market ticker
        event_title: Event title from sportsbook

    Returns:
        TradeValidation result
    """
    if not ticker or not event_title:
        return TradeValidation.valid()  # Can't validate, allow

    ticker_upper = ticker.upper()
    title_upper = event_title.upper()

    # Extract team abbreviations from ticker
    # Pattern: KXSPORT-TEAM1-TEAM2 or KXSPORT-TEAM
    team_pattern = r"[A-Z]{2,4}(?=-|$)"

    # Skip common prefixes
    ticker_clean = re.sub(r"^KX[A-Z]+(?:GAME|SPREAD|TOTAL|ML)?-?", "", ticker_upper)

    if not ticker_clean:
        return TradeValidation.valid()  # No team info in ticker

    # Common NBA team abbreviations to full names
    NBA_ABBREV = {
        "ATL": ["ATLANTA", "HAWKS"],
        "BOS": ["BOSTON", "CELTICS"],
        "BKN": ["BROOKLYN", "NETS"],
        "CHA": ["CHARLOTTE", "HORNETS"],
        "CHI": ["CHICAGO", "BULLS"],
        "CLE": ["CLEVELAND", "CAVALIERS", "CAVS"],
        "DAL": ["DALLAS", "MAVERICKS", "MAVS"],
        "DEN": ["DENVER", "NUGGETS"],
        "DET": ["DETROIT", "PISTONS"],
        "GSW": ["GOLDEN STATE", "WARRIORS"],
        "HOU": ["HOUSTON", "ROCKETS"],
        "IND": ["INDIANA", "PACERS"],
        "LAC": ["LOS ANGELES", "CLIPPERS", "LA CLIPPERS"],
        "LAL": ["LOS ANGELES", "LAKERS", "LA LAKERS"],
        "MEM": ["MEMPHIS", "GRIZZLIES"],
        "MIA": ["MIAMI", "HEAT"],
        "MIL": ["MILWAUKEE", "BUCKS"],
        "MIN": ["MINNESOTA", "TIMBERWOLVES", "WOLVES"],
        "NOP": ["NEW ORLEANS", "PELICANS"],
        "NYK": ["NEW YORK", "KNICKS"],
        "OKC": ["OKLAHOMA CITY", "THUNDER"],
        "ORL": ["ORLANDO", "MAGIC"],
        "PHI": ["PHILADELPHIA", "76ERS", "SIXERS"],
        "PHX": ["PHOENIX", "SUNS"],
        "POR": ["PORTLAND", "TRAIL BLAZERS", "BLAZERS"],
        "SAC": ["SACRAMENTO", "KINGS"],
        "SAS": ["SAN ANTONIO", "SPURS"],
        "TOR": ["TORONTO", "RAPTORS"],
        "UTA": ["UTAH", "JAZZ"],
        "WAS": ["WASHINGTON", "WIZARDS"],
    }

    # NFL team abbreviations
    NFL_ABBREV = {
        "ARI": ["ARIZONA", "CARDINALS"],
        "ATL": ["ATLANTA", "FALCONS"],
        "BAL": ["BALTIMORE", "RAVENS"],
        "BUF": ["BUFFALO", "BILLS"],
        "CAR": ["CAROLINA", "PANTHERS"],
        "CHI": ["CHICAGO", "BEARS"],
        "CIN": ["CINCINNATI", "BENGALS"],
        "CLE": ["CLEVELAND", "BROWNS"],
        "DAL": ["DALLAS", "COWBOYS"],
        "DEN": ["DENVER", "BRONCOS"],
        "DET": ["DETROIT", "LIONS"],
        "GB": ["GREEN BAY", "PACKERS"],
        "HOU": ["HOUSTON", "TEXANS"],
        "IND": ["INDIANAPOLIS", "COLTS"],
        "JAX": ["JACKSONVILLE", "JAGUARS"],
        "KC": ["KANSAS CITY", "CHIEFS"],
        "LAC": ["LOS ANGELES", "CHARGERS", "LA CHARGERS"],
        "LAR": ["LOS ANGELES", "RAMS", "LA RAMS"],
        "LV": ["LAS VEGAS", "RAIDERS"],
        "MIA": ["MIAMI", "DOLPHINS"],
        "MIN": ["MINNESOTA", "VIKINGS"],
        "NE": ["NEW ENGLAND", "PATRIOTS"],
        "NO": ["NEW ORLEANS", "SAINTS"],
        "NYG": ["NEW YORK", "GIANTS"],
        "NYJ": ["NEW YORK", "JETS"],
        "PHI": ["PHILADELPHIA", "EAGLES"],
        "PIT": ["PITTSBURGH", "STEELERS"],
        "SEA": ["SEATTLE", "SEAHAWKS"],
        "SF": ["SAN FRANCISCO", "49ERS"],
        "TB": ["TAMPA BAY", "BUCCANEERS", "BUCS"],
        "TEN": ["TENNESSEE", "TITANS"],
        "WAS": ["WASHINGTON", "COMMANDERS"],
    }

    # Combine abbreviation maps
    all_abbrevs = {**NBA_ABBREV, **NFL_ABBREV}

    # Extract abbreviations from cleaned ticker
    ticker_abbrevs = re.findall(r"[A-Z]{2,4}", ticker_clean)

    if not ticker_abbrevs:
        return TradeValidation.valid()

    # Check if at least one abbreviation matches the title
    for abbrev in ticker_abbrevs:
        if abbrev in all_abbrevs:
            # Check if any of the team's identifiers appear in title
            for identifier in all_abbrevs[abbrev]:
                if identifier in title_upper:
                    return TradeValidation.valid()
        else:
            # Unknown abbreviation, check if it appears directly
            if abbrev in title_upper:
                return TradeValidation.valid()

    # No match found - this is suspicious
    return TradeValidation.invalid_blocking(
        f"Ticker teams {ticker_abbrevs} don't match event '{event_title}'",
        "TICKER_MISMATCH"
    )


def validate_books_count(
    books_count: int,
    min_books: int = 2,
) -> TradeValidation:
    """
    Validate sufficient sportsbook consensus.

    Args:
        books_count: Number of sportsbooks in consensus
        min_books: Minimum required books (default 2)

    Returns:
        TradeValidation result
    """
    if books_count < min_books:
        return TradeValidation.invalid_blocking(
            f"Only {books_count} books, need at least {min_books}",
            "INSUFFICIENT_BOOKS"
        )

    return TradeValidation.valid()


def run_all_validations(
    price_cents: int,
    edge: float,
    books_count: int,
    ticker: str = "",
    event_title: str = "",
    min_price_cents: int = 5,
    max_price_cents: int = 95,
    min_edge: float = 0.05,
    suspicious_edge: float = 0.30,
    min_books: int = 2,
) -> List[TradeValidation]:
    """
    Run all validations and return list of failures.

    Returns empty list if all validations pass.
    """
    failures = []

    price_check = validate_price(price_cents, min_price_cents, max_price_cents)
    if not price_check.is_valid:
        failures.append(price_check)

    edge_check = validate_edge(edge, min_edge, suspicious_edge)
    if not edge_check.is_valid:
        failures.append(edge_check)

    books_check = validate_books_count(books_count, min_books)
    if not books_check.is_valid:
        failures.append(books_check)

    if ticker and event_title:
        ticker_check = validate_ticker_matchup(ticker, event_title)
        if not ticker_check.is_valid:
            failures.append(ticker_check)

    return failures


def has_blocking_failure(validations: List[TradeValidation]) -> bool:
    """Check if any validation failures are blocking."""
    return any(v.is_blocking for v in validations)
