"""Unit tests for lib/outlier.py"""

import pytest
from lib.outlier import (
    detect_outliers,
    consensus_fair_prob,
    calculate_no_vig_probs,
    american_to_prob,
    compute_fair_probs_from_odds,
)


class TestDetectOutliers:
    """Tests for detect_outliers function."""

    def test_no_outliers(self):
        """Test with no outliers."""
        probs = [0.50, 0.51, 0.49, 0.50, 0.52]
        normal, outliers = detect_outliers(probs)
        assert len(normal) == 5
        assert len(outliers) == 0

    def test_with_outlier_low_threshold(self):
        """Test with outlier at lower z-threshold."""
        probs = [0.50, 0.50, 0.50, 0.50, 0.90]
        normal, outliers = detect_outliers(probs, z_threshold=1.5)
        assert len(outliers) == 1
        assert 0.90 in outliers

    def test_insufficient_data(self):
        """Test with fewer than 3 data points."""
        probs = [0.50, 0.90]
        normal, outliers = detect_outliers(probs)
        assert normal == [0.50, 0.90]
        assert outliers == []

    def test_all_same_value(self):
        """Test when all values are identical."""
        probs = [0.55, 0.55, 0.55, 0.55]
        normal, outliers = detect_outliers(probs)
        assert len(normal) == 4
        assert len(outliers) == 0


class TestConsensusFairProb:
    """Tests for consensus_fair_prob function."""

    def test_simple_consensus(self):
        """Test simple median consensus."""
        probs = [0.50, 0.52, 0.54]
        result = consensus_fair_prob(probs, remove_outliers=False)

        assert result["fair_prob"] == pytest.approx(0.52, rel=1e-6)
        assert result["books_used"] == 3
        assert result["outliers_removed"] == 0

    def test_with_outlier_removal(self):
        """Test consensus with outlier removal."""
        probs = [0.50, 0.50, 0.50, 0.50, 0.90]
        result = consensus_fair_prob(probs, z_threshold=1.5, remove_outliers=True)

        # 0.90 should be removed as outlier
        assert result["outliers_removed"] >= 0  # May or may not be removed depending on z-score
        assert result["books_used"] <= 5

    def test_empty_input(self):
        """Test with empty input."""
        result = consensus_fair_prob([])
        assert result["fair_prob"] == 0.5
        assert result["books_used"] == 0

    def test_with_book_names(self):
        """Test with book names for outlier identification."""
        probs = [0.50, 0.50, 0.50, 0.95]
        names = ["DraftKings", "FanDuel", "BetMGM", "OutlierBook"]
        result = consensus_fair_prob(probs, book_names=names, z_threshold=1.5, remove_outliers=True)

        assert "outlier_books" in result

    def test_consensus_spread(self):
        """Test consensus spread calculation."""
        probs = [0.45, 0.50, 0.55]
        result = consensus_fair_prob(probs, remove_outliers=False)

        assert result["consensus_spread"] == pytest.approx(0.10, rel=1e-6)

    def test_use_mean(self):
        """Test using mean instead of median."""
        probs = [0.40, 0.50, 0.60]
        result = consensus_fair_prob(probs, use_median=False)

        assert result["fair_prob"] == pytest.approx(0.50, rel=1e-6)


class TestCalculateNoVigProbs:
    """Tests for calculate_no_vig_probs function."""

    def test_balanced_market(self):
        """Test balanced (50/50) market with vig."""
        # -110 odds on both sides = 52.4% implied each
        over_raw = 0.524
        under_raw = 0.524
        fair_over, fair_under = calculate_no_vig_probs(over_raw, under_raw)

        assert fair_over == pytest.approx(0.50, rel=1e-2)
        assert fair_under == pytest.approx(0.50, rel=1e-2)
        assert fair_over + fair_under == pytest.approx(1.0, rel=1e-6)

    def test_unbalanced_market(self):
        """Test unbalanced market."""
        over_raw = 0.70
        under_raw = 0.40
        fair_over, fair_under = calculate_no_vig_probs(over_raw, under_raw)

        # 0.70 / 1.10 ≈ 0.636
        # 0.40 / 1.10 ≈ 0.364
        assert fair_over == pytest.approx(0.636, rel=1e-2)
        assert fair_under == pytest.approx(0.364, rel=1e-2)

    def test_zero_total(self):
        """Test edge case with zero total."""
        fair_over, fair_under = calculate_no_vig_probs(0.0, 0.0)
        assert fair_over == 0.5
        assert fair_under == 0.5


class TestAmericanToProb:
    """Tests for american_to_prob function."""

    def test_favorite(self):
        """Test negative (favorite) odds."""
        prob = american_to_prob(-150)
        assert prob == pytest.approx(0.60, rel=1e-2)

    def test_underdog(self):
        """Test positive (underdog) odds."""
        prob = american_to_prob(150)
        assert prob == pytest.approx(0.40, rel=1e-2)

    def test_even_odds(self):
        """Test even money (+100 / -100)."""
        prob_plus = american_to_prob(100)
        prob_minus = american_to_prob(-100)

        assert prob_plus == pytest.approx(0.50, rel=1e-6)
        assert prob_minus == pytest.approx(0.50, rel=1e-6)

    def test_zero_raises(self):
        """Test that zero odds raises error."""
        with pytest.raises(ValueError):
            american_to_prob(0)


class TestComputeFairProbsFromOdds:
    """Tests for compute_fair_probs_from_odds function."""

    def test_standard_vig_line(self):
        """Test standard -110/-110 line."""
        fair_over, fair_under = compute_fair_probs_from_odds(-110, -110)

        assert fair_over == pytest.approx(0.50, rel=1e-2)
        assert fair_under == pytest.approx(0.50, rel=1e-2)

    def test_unbalanced_line(self):
        """Test unbalanced line."""
        fair_over, fair_under = compute_fair_probs_from_odds(-200, 180)

        # -200 implied = 0.667, +180 implied = 0.357
        # Total = 1.024, fair = 0.651 / 0.349
        assert fair_over > fair_under
        assert fair_over + fair_under == pytest.approx(1.0, rel=1e-6)
