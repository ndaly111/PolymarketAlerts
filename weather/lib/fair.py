from __future__ import annotations

from dataclasses import dataclass
from typing import Dict


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
