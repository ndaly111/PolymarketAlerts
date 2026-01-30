"""
Sportsbook consensus outlier detection.

Detects outlier books that may skew consensus and provides
robust fair probability calculation.
"""

from __future__ import annotations

from statistics import mean, stdev, median
from typing import Dict, List, Tuple, Any, Optional


def detect_outliers(
    probabilities: List[float],
    z_threshold: float = 2.0,
) -> Tuple[List[float], List[float]]:
    """
    Detect outlier probabilities using z-score method.

    Args:
        probabilities: List of fair probabilities from different books
        z_threshold: Z-score threshold for outlier detection (default 2.0)

    Returns:
        Tuple of (normal_probs, outlier_probs)

    Example:
        >>> probs = [0.55, 0.57, 0.54, 0.56, 0.80]  # One outlier
        >>> normal, outliers = detect_outliers(probs)
        >>> outliers
        [0.80]
    """
    if len(probabilities) < 3:
        # Not enough data for meaningful outlier detection
        return list(probabilities), []

    mu = mean(probabilities)
    try:
        sigma = stdev(probabilities)
    except Exception:
        # All same value
        return list(probabilities), []

    if sigma == 0:
        # All same value, no outliers
        return list(probabilities), []

    normal = []
    outliers = []

    for p in probabilities:
        z_score = abs(p - mu) / sigma
        if z_score > z_threshold:
            outliers.append(p)
        else:
            normal.append(p)

    return normal, outliers


def consensus_fair_prob(
    probabilities: List[float],
    book_names: Optional[List[str]] = None,
    z_threshold: float = 2.0,
    remove_outliers: bool = True,
    use_median: bool = True,
) -> Dict[str, Any]:
    """
    Calculate consensus fair probability with optional outlier removal.

    Args:
        probabilities: List of fair probabilities from different books
        book_names: Optional list of book names (same order as probabilities)
        z_threshold: Z-score threshold for outlier detection
        remove_outliers: Whether to remove outliers from consensus
        use_median: Use median (True) or mean (False) for consensus

    Returns:
        Dict with:
            - fair_prob: Consensus fair probability
            - books_used: Number of books in consensus
            - outliers_removed: Number of outliers removed
            - outlier_books: Names of outlier books (if book_names provided)
            - consensus_spread: Max - min of included probabilities
            - raw_fair_prob: Fair prob without outlier removal
    """
    if not probabilities:
        return {
            "fair_prob": 0.5,
            "books_used": 0,
            "outliers_removed": 0,
            "outlier_books": [],
            "consensus_spread": 0.0,
            "raw_fair_prob": 0.5,
        }

    # Calculate raw fair prob (before outlier removal)
    raw_fair_prob = median(probabilities) if use_median else mean(probabilities)

    if not remove_outliers or len(probabilities) < 3:
        return {
            "fair_prob": raw_fair_prob,
            "books_used": len(probabilities),
            "outliers_removed": 0,
            "outlier_books": [],
            "consensus_spread": max(probabilities) - min(probabilities) if len(probabilities) > 1 else 0.0,
            "raw_fair_prob": raw_fair_prob,
        }

    # Detect outliers
    normal_probs, outlier_probs = detect_outliers(probabilities, z_threshold)

    # Identify outlier book names
    outlier_books = []
    if book_names and len(book_names) == len(probabilities):
        for i, p in enumerate(probabilities):
            if p in outlier_probs:
                outlier_books.append(book_names[i])

    # Use normal probabilities for consensus
    if not normal_probs:
        # All were outliers, fall back to all
        normal_probs = list(probabilities)

    fair_prob = median(normal_probs) if use_median else mean(normal_probs)
    consensus_spread = max(normal_probs) - min(normal_probs) if len(normal_probs) > 1 else 0.0

    return {
        "fair_prob": fair_prob,
        "books_used": len(normal_probs),
        "outliers_removed": len(outlier_probs),
        "outlier_books": outlier_books,
        "consensus_spread": consensus_spread,
        "raw_fair_prob": raw_fair_prob,
    }


def calculate_no_vig_probs(
    over_prob_raw: float,
    under_prob_raw: float,
) -> Tuple[float, float]:
    """
    Remove vigorish to get fair probabilities.

    Args:
        over_prob_raw: Raw implied over probability (with vig)
        under_prob_raw: Raw implied under probability (with vig)

    Returns:
        Tuple of (fair_over_prob, fair_under_prob)

    Example:
        >>> # -110 odds = 52.4% implied
        >>> # Total overround = 104.8%
        >>> fair_over, fair_under = calculate_no_vig_probs(0.524, 0.524)
        >>> fair_over  # ~0.50 (true 50/50 market)
    """
    total = over_prob_raw + under_prob_raw

    if total <= 0:
        return 0.5, 0.5

    fair_over = over_prob_raw / total
    fair_under = under_prob_raw / total

    return fair_over, fair_under


def american_to_prob(odds: int) -> float:
    """
    Convert American odds to implied probability.

    Args:
        odds: American odds (e.g., -110, +150)

    Returns:
        Implied probability (0.0-1.0)
    """
    if odds == 0:
        raise ValueError("American odds cannot be 0")

    if odds < 0:
        return (-odds) / ((-odds) + 100.0)
    else:
        return 100.0 / (odds + 100.0)


def compute_fair_probs_from_odds(
    over_odds: int,
    under_odds: int,
) -> Tuple[float, float]:
    """
    Convert over/under American odds to fair probabilities.

    Args:
        over_odds: American odds for over
        under_odds: American odds for under

    Returns:
        Tuple of (fair_over_prob, fair_under_prob)
    """
    over_raw = american_to_prob(over_odds)
    under_raw = american_to_prob(under_odds)
    return calculate_no_vig_probs(over_raw, under_raw)
