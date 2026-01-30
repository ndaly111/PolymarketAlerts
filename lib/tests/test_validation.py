"""Unit tests for lib/validation.py"""

import pytest
from lib.validation import (
    TradeValidation,
    validate_price,
    validate_edge,
    validate_ticker_matchup,
    validate_books_count,
    run_all_validations,
    has_blocking_failure,
)


class TestTradeValidation:
    """Tests for TradeValidation dataclass."""

    def test_valid(self):
        """Test valid() factory method."""
        v = TradeValidation.valid()
        assert v.is_valid is True
        assert v.is_blocking is False
        assert v.code == "OK"

    def test_invalid_blocking(self):
        """Test invalid_blocking() factory method."""
        v = TradeValidation.invalid_blocking("Test message", "TEST_CODE")
        assert v.is_valid is False
        assert v.is_blocking is True
        assert v.message == "Test message"
        assert v.code == "TEST_CODE"

    def test_invalid_warning(self):
        """Test invalid_warning() factory method."""
        v = TradeValidation.invalid_warning("Warning message", "WARN_CODE")
        assert v.is_valid is False
        assert v.is_blocking is False
        assert v.message == "Warning message"
        assert v.code == "WARN_CODE"


class TestValidatePrice:
    """Tests for validate_price function."""

    def test_valid_price(self):
        """Test valid price in normal range."""
        result = validate_price(50)
        assert result.is_valid is True

    def test_price_at_minimum(self):
        """Test price at minimum threshold."""
        result = validate_price(5)
        assert result.is_valid is True

    def test_price_at_maximum(self):
        """Test price at maximum threshold."""
        result = validate_price(95)
        assert result.is_valid is True

    def test_price_below_minimum(self):
        """Test price below minimum blocks trade."""
        result = validate_price(4)
        assert result.is_valid is False
        assert result.is_blocking is True
        assert result.code == "PRICE_TOO_LOW"
        assert "below minimum" in result.message

    def test_price_above_maximum(self):
        """Test price above maximum blocks trade."""
        result = validate_price(96)
        assert result.is_valid is False
        assert result.is_blocking is True
        assert result.code == "PRICE_TOO_HIGH"
        assert "above maximum" in result.message

    def test_custom_thresholds(self):
        """Test custom price thresholds."""
        # 10 is below default but above custom
        result = validate_price(10, min_cents=10, max_cents=90)
        assert result.is_valid is True

        result = validate_price(9, min_cents=10, max_cents=90)
        assert result.is_valid is False


class TestValidateEdge:
    """Tests for validate_edge function."""

    def test_valid_edge(self):
        """Test valid edge in normal range."""
        result = validate_edge(0.10)
        assert result.is_valid is True

    def test_edge_at_minimum(self):
        """Test edge at minimum threshold."""
        result = validate_edge(0.05)
        assert result.is_valid is True

    def test_edge_below_minimum(self):
        """Test edge below minimum blocks trade."""
        result = validate_edge(0.04)
        assert result.is_valid is False
        assert result.is_blocking is True
        assert result.code == "EDGE_TOO_LOW"

    def test_suspicious_edge(self):
        """Test suspicious edge (above 30%) flags but doesn't block."""
        result = validate_edge(0.50)
        assert result.is_valid is False
        assert result.is_blocking is False  # Warning, not blocking
        assert result.code == "EDGE_SUSPICIOUS"
        assert "suspicious" in result.message.lower()

    def test_custom_thresholds(self):
        """Test custom edge thresholds."""
        result = validate_edge(0.08, min_edge=0.10, suspicious_threshold=0.25)
        assert result.is_valid is False
        assert result.is_blocking is True


class TestValidateTickerMatchup:
    """Tests for validate_ticker_matchup function."""

    def test_empty_inputs(self):
        """Test with empty inputs allows trade."""
        result = validate_ticker_matchup("", "")
        assert result.is_valid is True

    def test_matching_nba_ticker(self):
        """Test NBA ticker that matches event."""
        result = validate_ticker_matchup("KXNBAGAME-LAL-BOS", "Los Angeles Lakers at Boston Celtics")
        assert result.is_valid is True

    def test_mismatched_ticker(self):
        """Test ticker that doesn't match event."""
        result = validate_ticker_matchup("KXNBAGAME-MIA-NYK", "Los Angeles Lakers at Boston Celtics")
        assert result.is_valid is False
        assert result.is_blocking is True
        assert result.code == "TICKER_MISMATCH"

    def test_partial_match(self):
        """Test ticker with partial team match."""
        result = validate_ticker_matchup("KXNBASPREAD-LAL", "Los Angeles Lakers at Golden State Warriors")
        assert result.is_valid is True


class TestValidateBooksCount:
    """Tests for validate_books_count function."""

    def test_sufficient_books(self):
        """Test with sufficient number of books."""
        result = validate_books_count(3)
        assert result.is_valid is True

    def test_minimum_books(self):
        """Test with exactly minimum books."""
        result = validate_books_count(2)
        assert result.is_valid is True

    def test_insufficient_books(self):
        """Test with insufficient books."""
        result = validate_books_count(1)
        assert result.is_valid is False
        assert result.is_blocking is True
        assert result.code == "INSUFFICIENT_BOOKS"


class TestRunAllValidations:
    """Tests for run_all_validations function."""

    def test_all_pass(self):
        """Test when all validations pass."""
        failures = run_all_validations(
            price_cents=50,
            edge=0.10,
            books_count=3,
        )
        assert len(failures) == 0

    def test_price_failure(self):
        """Test with price validation failure."""
        failures = run_all_validations(
            price_cents=2,
            edge=0.10,
            books_count=3,
        )
        assert len(failures) == 1
        assert failures[0].code == "PRICE_TOO_LOW"

    def test_multiple_failures(self):
        """Test with multiple validation failures."""
        failures = run_all_validations(
            price_cents=2,
            edge=0.02,
            books_count=1,
        )
        assert len(failures) == 3

    def test_with_ticker_validation(self):
        """Test with ticker/title validation."""
        failures = run_all_validations(
            price_cents=50,
            edge=0.10,
            books_count=3,
            ticker="KXNBAGAME-MIA-NYK",
            event_title="Los Angeles Lakers at Boston Celtics",
        )
        assert len(failures) == 1
        assert failures[0].code == "TICKER_MISMATCH"


class TestHasBlockingFailure:
    """Tests for has_blocking_failure function."""

    def test_no_failures(self):
        """Test with no failures."""
        assert has_blocking_failure([]) is False

    def test_only_warnings(self):
        """Test with only warning failures."""
        warnings = [TradeValidation.invalid_warning("warn", "WARN")]
        assert has_blocking_failure(warnings) is False

    def test_with_blocking(self):
        """Test with blocking failure."""
        failures = [
            TradeValidation.invalid_warning("warn", "WARN"),
            TradeValidation.invalid_blocking("block", "BLOCK"),
        ]
        assert has_blocking_failure(failures) is True
