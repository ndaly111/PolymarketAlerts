"""Tests for the ML-enhanced fair price pipeline.

Tests the integration of ML quantile models and ProbabilityLadder
into the compute_fair_prices pipeline.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Dict

import pytest

from weather.lib.probability_ladder import (
    ProbabilityLadder,
    blend_ladders,
    from_percentiles,
    from_pmf,
    from_point_forecast_and_error_pmf,
)
from weather.lib.fair import normalize_pmf, shift_pmf


class TestFromPercentiles:
    """Test ProbabilityLadder construction from ML-predicted percentiles."""

    def test_basic_construction(self):
        ladder = from_percentiles(p10=65, p25=68, p50=72, p75=76, p90=80)
        assert isinstance(ladder, ProbabilityLadder)
        assert ladder.pmf  # should have non-empty PMF
        total = sum(ladder.pmf.values())
        assert abs(total - 1.0) < 0.01

    def test_percentile_ordering_respected(self):
        ladder = from_percentiles(p10=60, p25=65, p50=70, p75=75, p90=80)
        # P(T >= 60) should be ~0.9 (since 60 is the p10)
        assert ladder.prob_above(60) > 0.85
        # P(T >= 80) should be ~0.1 (since 80 is the p90)
        assert ladder.prob_above(80) < 0.15
        # P(T >= 70) should be ~0.5 (since 70 is the p50)
        assert 0.40 < ladder.prob_above(70) < 0.60

    def test_narrow_distribution(self):
        """All percentiles close together → tight distribution."""
        ladder = from_percentiles(p10=70, p25=71, p50=72, p75=73, p90=74)
        assert ladder.prob_between(69, 75) > 0.90

    def test_wide_distribution(self):
        """Wide spread → wider distribution."""
        ladder = from_percentiles(p10=50, p25=60, p50=70, p75=80, p90=90)
        # Should have substantial probability across the range
        assert ladder.prob_between(50, 90) > 0.75

    def test_pmf_sums_to_one(self):
        ladder = from_percentiles(p10=65, p25=68, p50=72, p75=76, p90=80)
        total = sum(ladder.pmf.values())
        assert abs(total - 1.0) < 0.01


class TestFromPmf:
    """Test ProbabilityLadder construction from PMF dict."""

    def test_basic_construction(self):
        pmf = {68: 0.1, 69: 0.2, 70: 0.4, 71: 0.2, 72: 0.1}
        ladder = from_pmf(pmf)
        assert isinstance(ladder, ProbabilityLadder)
        assert abs(sum(ladder.pmf.values()) - 1.0) < 0.001

    def test_exceedance_derived(self):
        pmf = {70: 0.5, 71: 0.5}
        ladder = from_pmf(pmf)
        assert abs(ladder.prob_above(70) - 1.0) < 0.001
        assert abs(ladder.prob_above(71) - 0.5) < 0.001

    def test_unnormalized_pmf(self):
        """Should normalize an unnormalized PMF."""
        pmf = {70: 2.0, 71: 3.0}
        ladder = from_pmf(pmf)
        assert abs(sum(ladder.pmf.values()) - 1.0) < 0.001
        assert abs(ladder.pmf[70] - 0.4) < 0.001
        assert abs(ladder.pmf[71] - 0.6) < 0.001


class TestFromPointForecastAndErrorPmf:
    """Test construction from forecast + error distribution."""

    def test_shift_matches_legacy(self):
        error_pmf = {-2: 0.1, -1: 0.2, 0: 0.4, 1: 0.2, 2: 0.1}
        forecast = 70
        ladder = from_point_forecast_and_error_pmf(forecast, error_pmf)
        # Expected temp: 70 + 0 = 70 (error mean is 0)
        assert abs(ladder.mean() - 70.0) < 0.5
        # Temp range should be 68-72
        assert 68 in ladder.pmf
        assert 72 in ladder.pmf

    def test_bias_in_error(self):
        """Error with positive bias → higher temperatures."""
        error_pmf = {1: 0.3, 2: 0.4, 3: 0.3}
        forecast = 70
        ladder = from_point_forecast_and_error_pmf(forecast, error_pmf)
        # Mean should be forecast + error_mean = 70 + 2 = 72
        assert abs(ladder.mean() - 72.0) < 0.5


class TestBlendLadders:
    """Test blending multiple ladders."""

    def test_equal_weight_blend(self):
        ladder1 = from_pmf({70: 1.0})
        ladder2 = from_pmf({80: 1.0})
        blended = blend_ladders([(ladder1, 0.5), (ladder2, 0.5)])
        # Mean should be 75
        assert abs(blended.mean() - 75.0) < 0.5

    def test_weighted_blend(self):
        ladder1 = from_pmf({70: 1.0})
        ladder2 = from_pmf({80: 1.0})
        blended = blend_ladders([(ladder1, 0.7), (ladder2, 0.3)])
        # Mean should be 0.7*70 + 0.3*80 = 73
        assert abs(blended.mean() - 73.0) < 0.5

    def test_blend_preserves_normalization(self):
        ladder1 = from_percentiles(60, 65, 70, 75, 80)
        ladder2 = from_percentiles(62, 67, 72, 77, 82)
        blended = blend_ladders([(ladder1, 0.7), (ladder2, 0.3)])
        total = sum(blended.pmf.values())
        assert abs(total - 1.0) < 0.01


class TestPmfBackwardCompatibility:
    """Test that the ML path produces output compatible with compute_edges.py."""

    def test_ladder_pmf_has_integer_keys(self):
        """compute_edges.py expects pmf_high_f with integer keys."""
        ladder = from_percentiles(p10=65, p25=68, p50=72, p75=76, p90=80)
        for key in ladder.pmf:
            assert isinstance(key, int), f"PMF key {key} is not an integer"

    def test_ladder_pmf_values_are_float(self):
        ladder = from_percentiles(p10=65, p25=68, p50=72, p75=76, p90=80)
        for val in ladder.pmf.values():
            assert isinstance(val, float), f"PMF value {val} is not a float"
            assert 0.0 <= val <= 1.0

    def test_pmf_to_json_roundtrip(self):
        """Test that PMF survives JSON serialization (as compute_fair_prices does)."""
        ladder = from_percentiles(p10=65, p25=68, p50=72, p75=76, p90=80)
        # Mimic compute_fair_prices.py output format
        pmf_json = {str(k): v for k, v in sorted(ladder.pmf.items())}
        serialized = json.dumps(pmf_json)
        deserialized = json.loads(serialized)
        # Reconstruct PMF as compute_edges.py does
        pmf_back: Dict[int, float] = {}
        for k, v in deserialized.items():
            pmf_back[int(k)] = float(v)
        total = sum(pmf_back.values())
        assert abs(total - 1.0) < 0.01


class TestFallbackGaussianPmf:
    """Test the fallback Gaussian PMF builder."""

    def test_basic_properties(self):
        from weather.scripts.compute_fair_prices import _build_fallback_gaussian_pmf
        pmf = _build_fallback_gaussian_pmf(center=75, std=4.0)
        assert isinstance(pmf, dict)
        total = sum(pmf.values())
        assert abs(total - 1.0) < 0.001
        # Most probability should be near center
        assert pmf.get(75, 0.0) > pmf.get(70, 0.0)
        assert pmf.get(75, 0.0) > pmf.get(80, 0.0)

    def test_wider_std_wider_spread(self):
        from weather.scripts.compute_fair_prices import _build_fallback_gaussian_pmf
        narrow = _build_fallback_gaussian_pmf(center=70, std=2.0)
        wide = _build_fallback_gaussian_pmf(center=70, std=8.0)
        # Wide should have more keys (wider support)
        assert len(wide) > len(narrow)


class TestMultiModelSpread:
    """Test multi-model spread computation (requires DB)."""

    def test_spread_with_no_data(self):
        """When DB has no data, should return None."""
        from weather.scripts.compute_fair_prices import _get_multi_model_spread
        import tempfile
        import sqlite3

        # Create empty DB with schema
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = Path(f.name)
        db_lib_module = __import__("weather.lib.db", fromlist=["ensure_schema"])
        db_lib_module.ensure_schema(db_path)

        spread, details = _get_multi_model_spread(db_path, "NHIGH", "2025-06-15")
        assert spread is None
        assert details == []

        db_path.unlink(missing_ok=True)
