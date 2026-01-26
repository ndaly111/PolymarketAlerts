from __future__ import annotations

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
    Shrink PMF dispersion linearly with progress (1:1 ratio).

    As progress increases, the distribution shrinks toward its mean.
    At progress=1.0, collapses to a point mass at the mean.

    Formula: new_dispersion = original_dispersion * (1 - progress)

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

    # Shrink each temperature toward the mean
    # new_temp = mean + (old_temp - mean) * (1 - progress)
    shrunk: Dict[float, float] = {}
    for k, v in pmf.items():
        new_k = mean + (k - mean) * (1 - progress)
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
