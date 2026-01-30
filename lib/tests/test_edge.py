"""Unit tests for lib/edge.py"""

import pytest
from lib.edge import (
    calculate_edge_with_fee,
    ev_yes,
    ev_no,
    kelly_fraction,
    round_kelly,
    FeeSchedule,
    DEFAULT_FEE_CENTS,
)


class TestCalculateEdgeWithFee:
    """Tests for calculate_edge_with_fee function."""

    def test_basic_calculation(self):
        """Test basic edge calculation with default fee."""
        result = calculate_edge_with_fee(0.60, 50)

        assert result["raw_edge"] == pytest.approx(0.10, rel=1e-6)
        assert result["net_edge"] == pytest.approx(0.08, rel=1e-6)
        assert result["ev_cents"] == pytest.approx(8.0, rel=1e-6)
        assert result["price_cents"] == 50
        assert result["fee_cents"] == DEFAULT_FEE_CENTS
        assert result["fair_prob"] == 0.60

    def test_no_edge(self):
        """Test when fair price equals market price."""
        result = calculate_edge_with_fee(0.50, 50)

        assert result["raw_edge"] == pytest.approx(0.0, rel=1e-6)
        assert result["net_edge"] == pytest.approx(-0.02, rel=1e-6)
        assert result["ev_cents"] == pytest.approx(-2.0, rel=1e-6)

    def test_negative_edge(self):
        """Test when market is overpriced."""
        result = calculate_edge_with_fee(0.40, 50)

        assert result["raw_edge"] == pytest.approx(-0.10, rel=1e-6)
        assert result["net_edge"] == pytest.approx(-0.12, rel=1e-6)

    def test_custom_fee(self):
        """Test with custom fee amount."""
        result = calculate_edge_with_fee(0.60, 50, fee_cents=5)

        assert result["raw_edge"] == pytest.approx(0.10, rel=1e-6)
        assert result["net_edge"] == pytest.approx(0.05, rel=1e-6)
        assert result["fee_cents"] == 5

    def test_zero_fee(self):
        """Test with no fee."""
        result = calculate_edge_with_fee(0.60, 50, fee_cents=0)

        assert result["raw_edge"] == pytest.approx(0.10, rel=1e-6)
        assert result["net_edge"] == pytest.approx(0.10, rel=1e-6)

    def test_extreme_prices(self):
        """Test with extreme price values."""
        # Very low price
        result_low = calculate_edge_with_fee(0.10, 5)
        assert result_low["raw_edge"] == pytest.approx(0.05, rel=1e-6)

        # Very high price
        result_high = calculate_edge_with_fee(0.95, 90)
        assert result_high["raw_edge"] == pytest.approx(0.05, rel=1e-6)


class TestEvFunctions:
    """Tests for ev_yes and ev_no functions."""

    def test_ev_yes(self):
        """Test EV for YES position."""
        # 60% fair prob, 50 cent price, 2 cent fee
        ev = ev_yes(0.60, 0.50, 0.02)
        assert ev == pytest.approx(0.08, rel=1e-6)

    def test_ev_no(self):
        """Test EV for NO position."""
        # 60% fair prob for YES means 40% for NO
        # NO price = 0.50, fee = 0.02
        ev = ev_no(0.60, 0.50, 0.02)
        assert ev == pytest.approx(-0.12, rel=1e-6)

    def test_ev_break_even(self):
        """Test EV at break-even point."""
        ev = ev_yes(0.52, 0.50, 0.02)
        assert ev == pytest.approx(0.0, rel=1e-6)


class TestKellyFraction:
    """Tests for kelly_fraction function."""

    def test_positive_edge(self):
        """Test Kelly with positive edge."""
        kelly = kelly_fraction(0.60, 50, fee_cents=2, multiplier=0.5)

        # Net edge = 0.08, profit multiplier = 0.50
        # Full Kelly = 0.08 / 0.50 = 0.16
        # Half Kelly = 0.08
        assert kelly == pytest.approx(0.08, rel=1e-2)

    def test_zero_edge(self):
        """Test Kelly with no edge returns 0."""
        kelly = kelly_fraction(0.52, 50, fee_cents=2, multiplier=0.5)
        assert kelly == pytest.approx(0.0, abs=1e-10)

    def test_negative_edge(self):
        """Test Kelly with negative edge returns 0."""
        kelly = kelly_fraction(0.40, 50, fee_cents=2, multiplier=0.5)
        assert kelly == 0.0

    def test_full_kelly(self):
        """Test full Kelly (multiplier=1.0)."""
        kelly = kelly_fraction(0.60, 50, fee_cents=2, multiplier=1.0)
        assert kelly == pytest.approx(0.16, rel=1e-2)


class TestRoundKelly:
    """Tests for round_kelly function."""

    def test_zero_kelly(self):
        """Test rounding zero."""
        assert round_kelly(0) == 0.0
        assert round_kelly(-0.1) == 0.0

    def test_tiny_kelly(self):
        """Test rounding very small Kelly."""
        assert round_kelly(0.01) == 0.0
        assert round_kelly(0.024) == 0.0

    def test_small_kelly(self):
        """Test rounding to 2.5%."""
        assert round_kelly(0.03) == 0.025
        assert round_kelly(0.049) == 0.025

    def test_standard_kelly(self):
        """Test rounding to 5% increments."""
        assert round_kelly(0.05) == 0.05
        assert round_kelly(0.07) == 0.05
        assert round_kelly(0.10) == 0.10
        assert round_kelly(0.15) == 0.15


class TestFeeSchedule:
    """Tests for FeeSchedule dataclass."""

    def test_default_fee(self):
        """Test default fee schedule."""
        fs = FeeSchedule()
        assert fs.open_fee_cents == 2
        assert fs.open_fee_dollars() == 0.02

    def test_custom_fee(self):
        """Test custom fee schedule."""
        fs = FeeSchedule(open_fee_cents=5)
        assert fs.open_fee_cents == 5
        assert fs.open_fee_dollars() == 0.05

    def test_frozen(self):
        """Test that FeeSchedule is immutable."""
        fs = FeeSchedule()
        with pytest.raises(Exception):
            fs.open_fee_cents = 10
