"""Tests for intraday PMF adjustment functions."""

import pytest

from weather.lib.fair import (
    adjust_pmf_with_progress,
    compute_progress,
    normalize_pmf,
    shrink_pmf_dispersion,
    truncate_pmf_below,
)


class TestTruncatePmfBelow:
    def test_simple_truncation(self):
        pmf = {68: 0.1, 69: 0.2, 70: 0.4, 71: 0.2, 72: 0.1}
        result = truncate_pmf_below(pmf, 70)
        assert 68 not in result
        assert 69 not in result
        assert 70 in result
        assert 71 in result
        assert 72 in result
        assert abs(sum(result.values()) - 1.0) < 1e-9

    def test_floor_above_all_support(self):
        pmf = {68: 0.5, 69: 0.5}
        result = truncate_pmf_below(pmf, 75)
        assert result == {75: 1.0}

    def test_floor_at_min(self):
        pmf = {70: 0.5, 71: 0.5}
        result = truncate_pmf_below(pmf, 70)
        # Should be unchanged (just renormalized)
        assert 70 in result
        assert 71 in result
        assert abs(sum(result.values()) - 1.0) < 1e-9

    def test_empty_pmf(self):
        result = truncate_pmf_below({}, 70)
        assert result == {70: 1.0}


class TestComputeProgress:
    def test_zero_progress(self):
        # Current temp equals baseline
        assert compute_progress(current_temp=60, baseline_temp=60, forecast_high=80) == 0.0

    def test_full_progress(self):
        # Current temp equals forecast high
        assert compute_progress(current_temp=80, baseline_temp=60, forecast_high=80) == 1.0

    def test_half_progress(self):
        # Halfway between baseline and forecast high
        assert compute_progress(current_temp=70, baseline_temp=60, forecast_high=80) == 0.5

    def test_quarter_progress(self):
        assert compute_progress(current_temp=65, baseline_temp=60, forecast_high=80) == 0.25

    def test_overshoot_clamps_to_one(self):
        # Current temp exceeds forecast high
        assert compute_progress(current_temp=85, baseline_temp=60, forecast_high=80) == 1.0

    def test_below_baseline_clamps_to_zero(self):
        # Current temp below baseline
        assert compute_progress(current_temp=55, baseline_temp=60, forecast_high=80) == 0.0

    def test_forecast_equals_baseline(self):
        # Edge case: forecast high equals baseline (shouldn't happen in practice)
        assert compute_progress(current_temp=60, baseline_temp=60, forecast_high=60) == 1.0

    def test_forecast_below_baseline(self):
        # Edge case: forecast lower than baseline
        result = compute_progress(current_temp=65, baseline_temp=60, forecast_high=55)
        assert result == 1.0  # Already exceeded forecast


class TestShrinkPmfDispersion:
    def test_no_shrink_at_zero_progress(self):
        pmf = {69: 0.25, 70: 0.5, 71: 0.25}
        result = shrink_pmf_dispersion(pmf, progress=0.0)
        # Should be unchanged
        assert result == pmf

    def test_full_shrink_collapses_to_mean(self):
        pmf = {68: 0.1, 69: 0.2, 70: 0.4, 71: 0.2, 72: 0.1}
        result = shrink_pmf_dispersion(pmf, progress=1.0)
        # Should collapse to point mass at mean (70)
        assert len(result) == 1
        assert 70 in result
        assert abs(result[70] - 1.0) < 1e-9

    def test_half_shrink_reduces_spread(self):
        pmf = {68: 0.1, 69: 0.2, 70: 0.4, 71: 0.2, 72: 0.1}
        result = shrink_pmf_dispersion(pmf, progress=0.5)
        # Should be narrower but not collapsed
        assert abs(sum(result.values()) - 1.0) < 1e-9
        # Mean should still be ~70
        mean = sum(k * v for k, v in result.items())
        assert abs(mean - 70) < 0.5

    def test_empty_pmf(self):
        result = shrink_pmf_dispersion({}, progress=0.5)
        assert result == {}

    def test_negative_progress_treated_as_zero(self):
        pmf = {69: 0.25, 70: 0.5, 71: 0.25}
        result = shrink_pmf_dispersion(pmf, progress=-0.5)
        assert result == pmf


class TestAdjustPmfWithProgress:
    def test_combined_adjustment(self):
        pmf = {68: 0.1, 69: 0.2, 70: 0.4, 71: 0.2, 72: 0.1}
        result, meta = adjust_pmf_with_progress(
            pmf,
            max_observed=69,
            progress=0.5,
        )
        # Should have truncated below 69 and shrunk
        assert 68 not in result
        assert abs(sum(result.values()) - 1.0) < 1e-9
        assert meta["max_observed_f"] == 69
        assert meta["progress"] == 0.5

    def test_no_adjustment_at_zero_progress(self):
        pmf = {68: 0.1, 69: 0.2, 70: 0.4, 71: 0.2, 72: 0.1}
        result, meta = adjust_pmf_with_progress(
            pmf,
            max_observed=68,  # Floor at min
            progress=0.0,     # No shrink
        )
        # Should be approximately unchanged
        assert 68 in result
        assert abs(sum(result.values()) - 1.0) < 1e-9

    def test_full_adjustment(self):
        pmf = {68: 0.1, 69: 0.2, 70: 0.4, 71: 0.2, 72: 0.1}
        result, meta = adjust_pmf_with_progress(
            pmf,
            max_observed=70,
            progress=1.0,
        )
        # Truncated below 70, then collapsed to mean of remaining
        assert 68 not in result
        assert 69 not in result
        # Should be narrow around 70-71 range
        assert len(result) <= 2

    def test_metadata_contains_expected_fields(self):
        pmf = {68: 0.1, 69: 0.2, 70: 0.4, 71: 0.2, 72: 0.1}
        _, meta = adjust_pmf_with_progress(pmf, max_observed=69, progress=0.5)
        assert "max_observed_f" in meta
        assert "progress" in meta
        assert "original_support" in meta
        assert "adjusted_support" in meta
        assert "original_mean" in meta
        assert "adjusted_mean" in meta


class TestIntegration:
    """Integration tests simulating real usage patterns."""

    def test_morning_scenario(self):
        """Early morning: low progress, wide distribution."""
        # 4am baseline: 55F, forecast high: 75F, current: 58F
        progress = compute_progress(58, 55, 75)
        assert progress == pytest.approx(0.15, abs=0.01)

        # PMF centered on 75F with +-5F spread
        pmf = normalize_pmf({70: 0.05, 71: 0.1, 72: 0.15, 73: 0.2, 74: 0.2,
                            75: 0.15, 76: 0.1, 77: 0.05})
        result, _ = adjust_pmf_with_progress(pmf, max_observed=58, progress=progress)

        # Still wide distribution, floor doesn't matter much (58 < all PMF support)
        assert len(result) >= 6

    def test_midday_scenario(self):
        """Midday: medium progress, narrower distribution."""
        # 4am baseline: 55F, forecast high: 75F, current: 68F
        progress = compute_progress(68, 55, 75)
        assert progress == pytest.approx(0.65, abs=0.01)

        pmf = normalize_pmf({70: 0.05, 71: 0.1, 72: 0.15, 73: 0.2, 74: 0.2,
                            75: 0.15, 76: 0.1, 77: 0.05})
        result, _ = adjust_pmf_with_progress(pmf, max_observed=71, progress=progress)

        # Truncated below 71, shrunk by 65%
        assert 70 not in result
        # Distribution should be noticeably narrower

    def test_afternoon_scenario(self):
        """Late afternoon: high progress, tight distribution."""
        # 4am baseline: 55F, forecast high: 75F, current: 74F
        progress = compute_progress(74, 55, 75)
        assert progress == pytest.approx(0.95, abs=0.01)

        pmf = normalize_pmf({70: 0.05, 71: 0.1, 72: 0.15, 73: 0.2, 74: 0.2,
                            75: 0.15, 76: 0.1, 77: 0.05})
        result, _ = adjust_pmf_with_progress(pmf, max_observed=74, progress=progress)

        # Truncated below 74, heavily shrunk
        assert 70 not in result
        assert 71 not in result
        assert 72 not in result
        assert 73 not in result
        # Should be very tight around 74-75
        assert len(result) <= 3
