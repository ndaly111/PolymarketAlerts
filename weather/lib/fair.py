from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, Tuple


@dataclass(frozen=True)
class FairSummary:
    mean: float
    p10: int
    p50: int
    p90: int


def shift_pmf(pmf: Dict[int, float], shift: int) -> Dict[int, float]:
    """Shift PMF support by an integer amount."""
    return {int(k) + int(shift): float(v) for k, v in pmf.items()}


def normalize_pmf(pmf: Dict[int, float]) -> Dict[int, float]:
    s = float(sum(pmf.values()))
    if s <= 0:
        return {}
    return {int(k): float(v) / s for k, v in pmf.items()}


def summarize_pmf(pmf: Dict[int, float]) -> FairSummary:
    """
    Summary stats for an integer-support PMF.
    Quantiles are computed on the discrete CDF.
    """
    pmf = normalize_pmf(pmf)
    keys = sorted(pmf.keys())
    if not keys:
        return FairSummary(mean=float("nan"), p10=0, p50=0, p90=0)

    mean = sum(k * pmf[k] for k in keys)

    def q(p: float) -> int:
        c = 0.0
        for k in keys:
            c += pmf[k]
            if c >= p:
                return int(k)
        return int(keys[-1])

    return FairSummary(mean=float(mean), p10=q(0.10), p50=q(0.50), p90=q(0.90))


# ---------------------------------------------------------------------------
# Intraday PMF adjustment functions
# ---------------------------------------------------------------------------


def truncate_pmf_below(pmf: Dict[int, float], floor_temp: int) -> Dict[int, float]:
    """
    Remove all probability mass below floor_temp (hard floor).

    The daily high cannot be less than the maximum temperature already observed.

    Args:
        pmf: Original PMF {temp: probability}
        floor_temp: Minimum possible temperature (max observed so far)

    Returns:
        New PMF with mass below floor_temp removed and renormalized.
        If all mass is below floor_temp, returns point mass at floor_temp.
    """
    truncated = {k: v for k, v in pmf.items() if k >= floor_temp}

    if not truncated:
        # Edge case: max_observed exceeds all PMF support
        return {floor_temp: 1.0}

    return normalize_pmf(truncated)


def compute_progress(
    current_temp: int,
    baseline_temp: int,
    forecast_high: int,
) -> float:
    """
    Calculate progress toward daily high.

    Formula: progress = (current_temp - baseline_temp) / (forecast_high - baseline_temp)

    Args:
        current_temp: Current observed temperature
        baseline_temp: Morning baseline temperature (4am snapshot)
        forecast_high: Forecasted high temperature

    Returns:
        Progress value clamped to [0, 1]
    """
    if forecast_high <= baseline_temp:
        # Edge case: forecast high is at or below baseline
        return 1.0 if current_temp >= forecast_high else 0.0

    denominator = forecast_high - baseline_temp
    numerator = current_temp - baseline_temp

    progress = numerator / denominator
    return max(0.0, min(1.0, progress))


def shrink_pmf_dispersion(pmf: Dict[int, float], progress: float) -> Dict[int, float]:
    """
    Shrink PMF dispersion using a sqrt-based curve (less aggressive than linear).

    The shrink factor uses sqrt(1 - progress) so that:
    - At progress=0.0: shrink_factor=1.0 (no change)
    - At progress=0.25: shrink_factor=0.87 (13% narrower, was 25% with linear)
    - At progress=0.50: shrink_factor=0.71 (29% narrower, was 50% with linear)
    - At progress=0.75: shrink_factor=0.50 (50% narrower, was 75% with linear)
    - At progress=1.0: shrink_factor=0.0 (collapses to mean)

    This preserves more uncertainty in the early/mid day when the temperature
    hasn't peaked yet, only collapsing aggressively near end of day.

    Args:
        pmf: Original PMF {temp: probability}
        progress: Progress toward high [0, 1]

    Returns:
        PMF with reduced dispersion around the mean.
    """
    if not pmf or progress <= 0:
        return pmf

    # Clamp progress to [0, 1]
    progress = max(0.0, min(1.0, progress))

    # Calculate the mean (expected value)
    total = sum(pmf.values())
    if total <= 0:
        return pmf

    mean = sum(k * v for k, v in pmf.items()) / total

    # sqrt-based shrinkage: preserves more width early in the day
    shrink_factor = math.sqrt(1.0 - progress)

    # Shrink each temperature toward the mean
    shrunk: Dict[float, float] = {}
    for k, v in pmf.items():
        new_k = mean + (k - mean) * shrink_factor
        shrunk[new_k] = shrunk.get(new_k, 0.0) + v

    # Round to integers and renormalize
    result: Dict[int, float] = {}
    for k, v in shrunk.items():
        int_k = int(round(k))
        result[int_k] = result.get(int_k, 0.0) + v

    return normalize_pmf(result)


def adjust_pmf_with_progress(
    pmf: Dict[int, float],
    max_observed: int,
    progress: float,
) -> Tuple[Dict[int, float], Dict[str, Any]]:
    """
    Apply intraday adjustments to a PMF.

    1. Truncate below max_observed (hard floor - high can't be lower than observed)
    2. Shrink dispersion linearly with progress (1:1 ratio)

    Args:
        pmf: Original forecast error-shifted PMF
        max_observed: Maximum temperature observed so far today
        progress: Progress toward high [0, 1]

    Returns:
        (adjusted_pmf, metadata_dict) for logging/auditing
    """
    original_summary = summarize_pmf(pmf) if pmf else None

    # Step 1: Hard floor truncation
    truncated = truncate_pmf_below(pmf, max_observed)

    # Step 2: Shrink dispersion based on progress
    adjusted = shrink_pmf_dispersion(truncated, progress)

    adjusted_summary = summarize_pmf(adjusted) if adjusted else None

    metadata = {
        "max_observed_f": max_observed,
        "progress": round(progress, 4),
        "original_support": (min(pmf.keys()), max(pmf.keys())) if pmf else None,
        "adjusted_support": (min(adjusted.keys()), max(adjusted.keys())) if adjusted else None,
        "original_mean": round(original_summary.mean, 2) if original_summary else None,
        "adjusted_mean": round(adjusted_summary.mean, 2) if adjusted_summary else None,
    }

    return adjusted, metadata
