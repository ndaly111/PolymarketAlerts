"""
Probability Ladder: Core data structure for temperature probability distributions.

The probability ladder represents the full probability distribution of temperature
outcomes, enabling precise pricing of Kalshi weather contracts.

Key concepts:
- Exceedance probabilities: P(T >= threshold) for each temperature threshold
- PMF (Probability Mass Function): P(T = k) for each integer temperature
- Bucket probabilities: P(low <= T < high) for Kalshi "between" contracts

The ladder can be constructed from:
1. NBM percentiles (p10, p25, p50, p75, p90)
2. Historical error distributions
3. ML model quantile predictions
4. Combinations of the above via blending
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import math


@dataclass
class ProbabilityLadder:
    """
    Probability distribution for temperature outcomes.

    Attributes:
        thresholds: Temperature thresholds (integers, e.g., [68, 69, 70, ...])
        exceedance: Exceedance probabilities P(T >= threshold) for each threshold
        pmf: Probability mass function P(T = k) derived from exceedance
        source: Description of how this ladder was constructed
    """
    thresholds: List[int]
    exceedance: Dict[int, float]  # {threshold: P(T >= threshold)}
    pmf: Dict[int, float] = field(default_factory=dict)  # {temp: P(T = temp)}
    source: str = "unknown"

    def __post_init__(self):
        """Compute PMF from exceedance if not provided."""
        if not self.pmf and self.exceedance:
            self.pmf = self._exceedance_to_pmf()

    def _exceedance_to_pmf(self) -> Dict[int, float]:
        """
        Convert exceedance probabilities to PMF.

        P(T = k) = P(T >= k) - P(T >= k+1)
        """
        pmf = {}
        sorted_thresholds = sorted(self.exceedance.keys())

        for i, t in enumerate(sorted_thresholds):
            if i + 1 < len(sorted_thresholds):
                next_t = sorted_thresholds[i + 1]
                # P(T = t) = P(T >= t) - P(T >= next_t)
                prob = self.exceedance[t] - self.exceedance[next_t]
            else:
                # Last threshold: all remaining probability
                prob = self.exceedance[t]

            if prob > 0:
                pmf[t] = max(0.0, prob)

        # Normalize to sum to 1
        total = sum(pmf.values())
        if total > 0 and abs(total - 1.0) > 0.001:
            pmf = {k: v / total for k, v in pmf.items()}

        return pmf

    def prob_above(self, threshold: int) -> float:
        """
        P(T >= threshold) - probability temperature is at or above threshold.

        Used for Kalshi "above X" or "below X" contracts.

        Args:
            threshold: Temperature threshold (inclusive)

        Returns:
            Probability that temperature >= threshold
        """
        # Direct lookup if available
        if threshold in self.exceedance:
            return self.exceedance[threshold]

        # Interpolate from PMF
        return sum(p for t, p in self.pmf.items() if t >= threshold)

    def prob_below(self, threshold: int) -> float:
        """
        P(T < threshold) - probability temperature is strictly below threshold.

        Args:
            threshold: Temperature threshold (exclusive)

        Returns:
            Probability that temperature < threshold
        """
        return 1.0 - self.prob_above(threshold)

    def prob_between(self, low: int, high: int, inclusive: str = "both") -> float:
        """
        P(low <= T <= high) or P(low <= T < high) for Kalshi bucket contracts.

        Kalshi bucket contracts are typically "X to Y degrees" which means
        the actual high temp falls in that range.

        Args:
            low: Lower bound (inclusive by default)
            high: Upper bound
            inclusive: "both" (low <= T <= high), "left" (low <= T < high),
                      "right" (low < T <= high), "neither" (low < T < high)

        Returns:
            Probability that temperature falls in the range
        """
        prob = 0.0

        for t, p in self.pmf.items():
            include = False

            if inclusive == "both":
                include = low <= t <= high
            elif inclusive == "left":
                include = low <= t < high
            elif inclusive == "right":
                include = low < t <= high
            elif inclusive == "neither":
                include = low < t < high

            if include:
                prob += p

        return prob

    def prob_exact(self, temperature: int) -> float:
        """
        P(T = temperature) - probability of exact temperature.

        Args:
            temperature: Exact temperature

        Returns:
            Probability of this exact temperature
        """
        return self.pmf.get(temperature, 0.0)

    def mean(self) -> float:
        """Expected value of the temperature distribution."""
        if not self.pmf:
            return 0.0
        return sum(t * p for t, p in self.pmf.items())

    def variance(self) -> float:
        """Variance of the temperature distribution."""
        if not self.pmf:
            return 0.0
        mu = self.mean()
        return sum(p * (t - mu) ** 2 for t, p in self.pmf.items())

    def std_dev(self) -> float:
        """Standard deviation of the temperature distribution."""
        return math.sqrt(self.variance())

    def percentile(self, p: float) -> int:
        """
        Get the temperature at a given percentile.

        Args:
            p: Percentile (0-1, e.g., 0.5 for median)

        Returns:
            Temperature at that percentile (rounded to integer)
        """
        if not self.pmf:
            return 0

        sorted_temps = sorted(self.pmf.keys())
        cumulative = 0.0

        for t in sorted_temps:
            cumulative += self.pmf[t]
            if cumulative >= p:
                return t

        return sorted_temps[-1] if sorted_temps else 0

    @property
    def p10(self) -> int:
        """10th percentile temperature."""
        return self.percentile(0.10)

    @property
    def p25(self) -> int:
        """25th percentile temperature."""
        return self.percentile(0.25)

    @property
    def p50(self) -> int:
        """50th percentile (median) temperature."""
        return self.percentile(0.50)

    @property
    def p75(self) -> int:
        """75th percentile temperature."""
        return self.percentile(0.75)

    @property
    def p90(self) -> int:
        """90th percentile temperature."""
        return self.percentile(0.90)

    @property
    def spread(self) -> int:
        """Spread (p90 - p10) as a measure of uncertainty."""
        return self.p90 - self.p10

    def summary(self) -> Dict[str, float]:
        """Get summary statistics."""
        return {
            "mean": round(self.mean(), 1),
            "std_dev": round(self.std_dev(), 1),
            "p10": self.p10,
            "p25": self.p25,
            "p50": self.p50,
            "p75": self.p75,
            "p90": self.p90,
            "spread": self.spread,
        }


def from_percentiles(
    p10: int,
    p25: int,
    p50: int,
    p75: int,
    p90: int,
    min_temp: int = 0,
    max_temp: int = 130,
    source: str = "percentiles",
) -> ProbabilityLadder:
    """
    Construct a probability ladder from percentile values.

    Uses linear interpolation between percentile points to build
    the full exceedance curve.

    Args:
        p10-p90: Temperature percentiles
        min_temp: Minimum temperature for thresholds
        max_temp: Maximum temperature for thresholds
        source: Description of data source

    Returns:
        ProbabilityLadder with interpolated exceedance and PMF
    """
    # Known points: (temperature, exceedance_probability)
    # p10 means 10% of outcomes are below this temp, so 90% are >= p10
    points = [
        (p10, 0.90),
        (p25, 0.75),
        (p50, 0.50),
        (p75, 0.25),
        (p90, 0.10),
    ]

    # Sort by temperature
    points.sort(key=lambda x: x[0])

    # Generate thresholds
    thresholds = list(range(min_temp, max_temp + 1))

    # Interpolate exceedance probabilities
    exceedance = {}

    for thresh in thresholds:
        if thresh <= points[0][0]:
            # Below lowest point - extrapolate
            t1, p1 = points[0]
            t2, p2 = points[1]
            if t2 != t1:
                slope = (p2 - p1) / (t2 - t1)
                prob = p1 + slope * (thresh - t1)
            else:
                prob = p1
            prob = min(1.0, max(0.0, prob))
        elif thresh >= points[-1][0]:
            # Above highest point - extrapolate
            t1, p1 = points[-2]
            t2, p2 = points[-1]
            if t2 != t1:
                slope = (p2 - p1) / (t2 - t1)
                prob = p2 + slope * (thresh - t2)
            else:
                prob = p2
            prob = min(1.0, max(0.0, prob))
        else:
            # Interpolate between bracketing points
            for i in range(len(points) - 1):
                t1, p1 = points[i]
                t2, p2 = points[i + 1]
                if t1 <= thresh <= t2:
                    if t2 != t1:
                        frac = (thresh - t1) / (t2 - t1)
                        prob = p1 + frac * (p2 - p1)
                    else:
                        prob = p1
                    break
            else:
                prob = 0.5  # Fallback

        exceedance[thresh] = round(prob, 6)

    return ProbabilityLadder(
        thresholds=thresholds,
        exceedance=exceedance,
        source=source,
    )


def from_pmf(
    pmf: Dict[int, float],
    source: str = "pmf",
) -> ProbabilityLadder:
    """
    Construct a probability ladder from a PMF.

    Args:
        pmf: Probability mass function {temperature: probability}
        source: Description of data source

    Returns:
        ProbabilityLadder
    """
    # Normalize PMF
    total = sum(pmf.values())
    if total > 0:
        pmf = {k: v / total for k, v in pmf.items()}

    # Derive exceedance from PMF
    sorted_temps = sorted(pmf.keys())
    thresholds = sorted_temps

    exceedance = {}
    for t in sorted_temps:
        exceedance[t] = sum(p for temp, p in pmf.items() if temp >= t)

    return ProbabilityLadder(
        thresholds=thresholds,
        exceedance=exceedance,
        pmf=pmf,
        source=source,
    )


def from_point_forecast_and_error_pmf(
    forecast_high: int,
    error_pmf: Dict[int, float],
    source: str = "forecast+error",
) -> ProbabilityLadder:
    """
    Construct a probability ladder by convolving a point forecast with error distribution.

    This is the traditional approach: observed = forecast + error
    So P(observed = x) = sum over e of P(error = e) * I(forecast + e = x)

    In practice, this shifts the error PMF by the forecast amount.

    Args:
        forecast_high: Point forecast temperature
        error_pmf: Error distribution P(error = e) where error = observed - forecast
        source: Description of data source

    Returns:
        ProbabilityLadder for observed temperature distribution
    """
    # Shift error PMF by forecast to get observed PMF
    observed_pmf = {}
    for error, prob in error_pmf.items():
        observed_temp = forecast_high + error
        observed_pmf[observed_temp] = observed_pmf.get(observed_temp, 0.0) + prob

    return from_pmf(observed_pmf, source=source)


def blend_ladders(
    ladders: List[Tuple[ProbabilityLadder, float]],
    source: str = "blended",
) -> ProbabilityLadder:
    """
    Blend multiple probability ladders with specified weights.

    This is useful for combining:
    - NBM native percentiles with historical error calibration
    - Multiple model forecasts (ensemble approach)
    - Different forecast horizons

    Args:
        ladders: List of (ladder, weight) tuples
        source: Description of blending approach

    Returns:
        Blended ProbabilityLadder
    """
    # Normalize weights
    total_weight = sum(w for _, w in ladders)
    if total_weight <= 0:
        raise ValueError("Total weight must be positive")

    weights = [(ladder, w / total_weight) for ladder, w in ladders]

    # Blend PMFs
    blended_pmf: Dict[int, float] = {}

    for ladder, weight in weights:
        for temp, prob in ladder.pmf.items():
            blended_pmf[temp] = blended_pmf.get(temp, 0.0) + prob * weight

    return from_pmf(blended_pmf, source=source)


def apply_calibration(
    ladder: ProbabilityLadder,
    bias: float = 0.0,
    scale: float = 1.0,
    source: Optional[str] = None,
) -> ProbabilityLadder:
    """
    Apply calibration adjustments to a probability ladder.

    Args:
        ladder: Original ladder
        bias: Temperature bias to add (positive = warmer)
        scale: Spread scaling factor (>1 = wider distribution)
        source: New source description

    Returns:
        Calibrated ProbabilityLadder
    """
    if bias == 0.0 and scale == 1.0:
        return ladder

    mean = ladder.mean()
    new_pmf = {}

    for temp, prob in ladder.pmf.items():
        # Apply bias and scale: new_temp = mean + (temp - mean) * scale + bias
        new_temp = round(mean + (temp - mean) * scale + bias)
        new_pmf[new_temp] = new_pmf.get(new_temp, 0.0) + prob

    return from_pmf(
        new_pmf,
        source=source or f"{ladder.source}+calibrated(bias={bias},scale={scale})",
    )


def truncate_below(
    ladder: ProbabilityLadder,
    floor_temp: int,
    source: Optional[str] = None,
) -> ProbabilityLadder:
    """
    Truncate distribution below a floor temperature.

    Useful for intraday trading when current observed max
    establishes a floor for the final high.

    Args:
        ladder: Original ladder
        floor_temp: Minimum possible temperature (observed max so far)
        source: New source description

    Returns:
        Truncated ProbabilityLadder with floor applied
    """
    new_pmf = {}

    for temp, prob in ladder.pmf.items():
        if temp >= floor_temp:
            new_pmf[temp] = prob

    # If no probability remains above floor, put everything at floor
    if not new_pmf:
        new_pmf = {floor_temp: 1.0}
    else:
        # Normalize
        total = sum(new_pmf.values())
        if total > 0:
            new_pmf = {k: v / total for k, v in new_pmf.items()}

    return from_pmf(
        new_pmf,
        source=source or f"{ladder.source}+truncated(floor={floor_temp})",
    )


def shrink_dispersion(
    ladder: ProbabilityLadder,
    progress: float,
    source: Optional[str] = None,
) -> ProbabilityLadder:
    """
    Shrink distribution dispersion based on intraday progress.

    Uses sqrt-based curve for less aggressive shrinkage:
    - At progress=0.25: retains 87% of spread (vs 75% linear)
    - At progress=0.50: retains 71% of spread (vs 50% linear)
    - At progress=0.75: retains 50% of spread (vs 25% linear)

    Args:
        ladder: Original ladder
        progress: Progress through the day (0.0 = start, 1.0 = end)
        source: New source description

    Returns:
        ProbabilityLadder with reduced dispersion
    """
    if progress <= 0:
        return ladder
    if progress >= 1.0:
        # Collapse to point mass at mean
        mean_temp = round(ladder.mean())
        return from_pmf(
            {mean_temp: 1.0},
            source=source or f"{ladder.source}+collapsed",
        )

    mean = ladder.mean()
    shrink_factor = math.sqrt(1.0 - progress)
    new_pmf = {}

    for temp, prob in ladder.pmf.items():
        # Shrink toward mean
        new_temp = round(mean + (temp - mean) * shrink_factor)
        new_pmf[new_temp] = new_pmf.get(new_temp, 0.0) + prob

    return from_pmf(
        new_pmf,
        source=source or f"{ladder.source}+shrunk(progress={progress:.2f})",
    )


def compute_ev_for_contract(
    ladder: ProbabilityLadder,
    contract_type: str,
    threshold_or_range: Tuple[int, ...],
    market_price: float,
    side: str,
    fee: float = 0.0,
    profit_fee_rate: float = 0.07,
) -> float:
    """
    Compute expected value for a Kalshi weather contract.

    Uses Kalshi's actual fee model: 7% fee on winning profits only.

    Args:
        ladder: Probability ladder for temperature distribution
        contract_type: "above", "below", "between", or "exact"
        threshold_or_range: (threshold,) for above/below, (low, high) for between
        market_price: Current market price (0-1)
        side: "YES" or "NO"
        fee: Deprecated flat fee (ignored, kept for API compat)
        profit_fee_rate: Kalshi profit fee rate (default 7%)

    Returns:
        Expected value of the trade
    """
    # Get model probability
    if contract_type == "above":
        model_prob = ladder.prob_above(threshold_or_range[0])
    elif contract_type == "below":
        model_prob = ladder.prob_below(threshold_or_range[0])
    elif contract_type == "between":
        model_prob = ladder.prob_between(threshold_or_range[0], threshold_or_range[1])
    elif contract_type == "exact":
        model_prob = ladder.prob_exact(threshold_or_range[0])
    else:
        raise ValueError(f"Unknown contract type: {contract_type}")

    # Kalshi fee model: 7% fee on winning profits only
    # Net profit on win = (1 - price) * (1 - profit_fee_rate)
    net_profit_rate = 1.0 - profit_fee_rate

    if side.upper() == "YES":
        # Win (prob=model_prob): net profit = (1-price) * 0.93
        # Lose (prob=1-model_prob): lose price
        ev = model_prob * (1.0 - market_price) * net_profit_rate - (1.0 - model_prob) * market_price
    else:  # NO
        # Win (prob=1-model_prob): net profit = (1-price) * 0.93
        # Lose (prob=model_prob): lose price
        ev = (1.0 - model_prob) * (1.0 - market_price) * net_profit_rate - model_prob * market_price

    return ev
