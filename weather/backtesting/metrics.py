"""
Metrics for evaluating probability forecasts and trading strategies.

Key metrics:
- Brier Score: Mean squared error of probability forecasts
- Log Loss: Cross-entropy loss (more sensitive to confident wrong predictions)
- Calibration Curve: Predicted probability vs actual frequency
- Expected vs Realized EV: Does predicted edge translate to profit?
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import math


def brier_score(
    predicted_probs: List[float],
    actual_outcomes: List[int],
) -> float:
    """
    Compute Brier Score (mean squared probability error).

    Brier Score = (1/n) * sum((p_i - o_i)^2)

    Lower is better. Range is [0, 1] for binary outcomes.
    - Perfect: 0.0
    - Random (50%): 0.25
    - Worst: 1.0

    Args:
        predicted_probs: List of predicted probabilities (0-1)
        actual_outcomes: List of actual outcomes (0 or 1)

    Returns:
        Brier score
    """
    if len(predicted_probs) != len(actual_outcomes):
        raise ValueError("Length mismatch between predictions and outcomes")

    if not predicted_probs:
        return float("nan")

    n = len(predicted_probs)
    squared_errors = [(p - o) ** 2 for p, o in zip(predicted_probs, actual_outcomes)]
    return sum(squared_errors) / n


def log_loss(
    predicted_probs: List[float],
    actual_outcomes: List[int],
    eps: float = 1e-15,
) -> float:
    """
    Compute Log Loss (cross-entropy).

    Log Loss = -(1/n) * sum(o_i * log(p_i) + (1-o_i) * log(1-p_i))

    Lower is better. Heavily penalizes confident wrong predictions.
    - Perfect: 0.0
    - Random (50%): 0.693 (ln(2))
    - Confident wrong: Very high

    Args:
        predicted_probs: List of predicted probabilities (0-1)
        actual_outcomes: List of actual outcomes (0 or 1)
        eps: Small value to avoid log(0)

    Returns:
        Log loss
    """
    if len(predicted_probs) != len(actual_outcomes):
        raise ValueError("Length mismatch between predictions and outcomes")

    if not predicted_probs:
        return float("nan")

    n = len(predicted_probs)
    total_loss = 0.0

    for p, o in zip(predicted_probs, actual_outcomes):
        # Clip probability to avoid log(0)
        p_clipped = max(eps, min(1 - eps, p))

        if o == 1:
            total_loss -= math.log(p_clipped)
        else:
            total_loss -= math.log(1 - p_clipped)

    return total_loss / n


def calibration_curve(
    predicted_probs: List[float],
    actual_outcomes: List[int],
    n_bins: int = 10,
) -> Tuple[List[float], List[float], List[int]]:
    """
    Compute calibration curve (reliability diagram data).

    Groups predictions into bins by predicted probability and computes
    the actual frequency of positive outcomes in each bin.

    A well-calibrated model has bin_actual ≈ bin_predicted.

    Args:
        predicted_probs: List of predicted probabilities
        actual_outcomes: List of actual outcomes (0 or 1)
        n_bins: Number of probability bins

    Returns:
        Tuple of (bin_centers, bin_actuals, bin_counts)
        - bin_centers: Center of each probability bin
        - bin_actuals: Actual positive rate in each bin
        - bin_counts: Number of samples in each bin
    """
    if len(predicted_probs) != len(actual_outcomes):
        raise ValueError("Length mismatch between predictions and outcomes")

    # Initialize bins
    bin_sums = [0.0] * n_bins
    bin_counts = [0] * n_bins
    bin_actuals_sum = [0] * n_bins

    # Assign each prediction to a bin
    for p, o in zip(predicted_probs, actual_outcomes):
        # Bin index: 0 for [0, 0.1), 1 for [0.1, 0.2), ..., 9 for [0.9, 1.0]
        bin_idx = min(int(p * n_bins), n_bins - 1)
        bin_sums[bin_idx] += p
        bin_counts[bin_idx] += 1
        bin_actuals_sum[bin_idx] += o

    # Compute bin centers and actual rates
    bin_centers = []
    bin_actuals = []
    valid_counts = []

    for i in range(n_bins):
        if bin_counts[i] > 0:
            bin_centers.append(bin_sums[i] / bin_counts[i])
            bin_actuals.append(bin_actuals_sum[i] / bin_counts[i])
            valid_counts.append(bin_counts[i])

    return bin_centers, bin_actuals, valid_counts


def calibration_error(
    predicted_probs: List[float],
    actual_outcomes: List[int],
    n_bins: int = 10,
) -> float:
    """
    Compute Expected Calibration Error (ECE).

    ECE = sum(n_b / n * |actual_b - predicted_b|)

    A weighted average of the calibration error across bins.
    Lower is better. Perfect calibration = 0.

    Args:
        predicted_probs: List of predicted probabilities
        actual_outcomes: List of actual outcomes
        n_bins: Number of bins

    Returns:
        Expected calibration error
    """
    centers, actuals, counts = calibration_curve(predicted_probs, actual_outcomes, n_bins)

    if not counts:
        return float("nan")

    total_samples = sum(counts)
    ece = 0.0

    for center, actual, count in zip(centers, actuals, counts):
        weight = count / total_samples
        ece += weight * abs(actual - center)

    return ece


@dataclass
class TradeOutcome:
    """Record of a single trade outcome."""
    date: str
    market_ticker: str
    side: str  # "YES" or "NO"
    predicted_prob: float  # Model probability
    market_price: float  # Price paid
    predicted_ev: float  # Expected value at trade time
    actual_outcome: int  # 1 if contract settled YES, 0 if NO
    pnl: float  # Actual profit/loss


def expected_vs_realized_ev(
    trades: List[TradeOutcome],
) -> Dict[str, float]:
    """
    Compare predicted EV to realized returns.

    This is the ultimate test: does the model's edge translate to profit?

    Args:
        trades: List of trade outcomes

    Returns:
        Dict with various EV comparison metrics
    """
    if not trades:
        return {
            "n_trades": 0,
            "total_predicted_ev": 0.0,
            "total_realized_pnl": 0.0,
            "ev_to_pnl_ratio": float("nan"),
        }

    total_predicted_ev = sum(t.predicted_ev for t in trades)
    total_realized_pnl = sum(t.pnl for t in trades)

    # Win rate
    wins = sum(1 for t in trades if t.pnl > 0)
    win_rate = wins / len(trades)

    # Average trade metrics
    avg_predicted_ev = total_predicted_ev / len(trades)
    avg_realized_pnl = total_realized_pnl / len(trades)

    # EV to PnL ratio (should be ~1.0 for well-calibrated model)
    if abs(total_predicted_ev) > 0.001:
        ev_to_pnl_ratio = total_realized_pnl / total_predicted_ev
    else:
        ev_to_pnl_ratio = float("nan")

    # By EV bucket analysis
    ev_buckets = {
        "0-5%": [],
        "5-10%": [],
        "10-15%": [],
        "15-20%": [],
        "20%+": [],
    }

    for t in trades:
        ev_pct = t.predicted_ev * 100
        if ev_pct < 5:
            ev_buckets["0-5%"].append(t)
        elif ev_pct < 10:
            ev_buckets["5-10%"].append(t)
        elif ev_pct < 15:
            ev_buckets["10-15%"].append(t)
        elif ev_pct < 20:
            ev_buckets["15-20%"].append(t)
        else:
            ev_buckets["20%+"].append(t)

    bucket_stats = {}
    for bucket, bucket_trades in ev_buckets.items():
        if bucket_trades:
            bucket_wins = sum(1 for t in bucket_trades if t.pnl > 0)
            bucket_pnl = sum(t.pnl for t in bucket_trades)
            bucket_stats[f"bucket_{bucket}_n"] = len(bucket_trades)
            bucket_stats[f"bucket_{bucket}_win_rate"] = bucket_wins / len(bucket_trades)
            bucket_stats[f"bucket_{bucket}_pnl"] = bucket_pnl

    return {
        "n_trades": len(trades),
        "total_predicted_ev": round(total_predicted_ev, 4),
        "total_realized_pnl": round(total_realized_pnl, 4),
        "avg_predicted_ev": round(avg_predicted_ev, 4),
        "avg_realized_pnl": round(avg_realized_pnl, 4),
        "win_rate": round(win_rate, 4),
        "ev_to_pnl_ratio": round(ev_to_pnl_ratio, 4) if not math.isnan(ev_to_pnl_ratio) else float("nan"),
        **bucket_stats,
    }


def compute_drawdown(
    cumulative_pnl: List[float],
) -> Tuple[float, int, int]:
    """
    Compute maximum drawdown from cumulative PnL series.

    Args:
        cumulative_pnl: List of cumulative profit/loss values

    Returns:
        Tuple of (max_drawdown, peak_index, trough_index)
    """
    if not cumulative_pnl:
        return (0.0, 0, 0)

    peak = cumulative_pnl[0]
    peak_idx = 0
    max_dd = 0.0
    max_dd_peak_idx = 0
    max_dd_trough_idx = 0

    for i, pnl in enumerate(cumulative_pnl):
        if pnl > peak:
            peak = pnl
            peak_idx = i

        dd = peak - pnl
        if dd > max_dd:
            max_dd = dd
            max_dd_peak_idx = peak_idx
            max_dd_trough_idx = i

    return (max_dd, max_dd_peak_idx, max_dd_trough_idx)


def sharpe_ratio(
    daily_returns: List[float],
    risk_free_rate: float = 0.0,
) -> float:
    """
    Compute annualized Sharpe ratio.

    Sharpe = sqrt(252) * (mean(returns) - rf) / std(returns)

    Args:
        daily_returns: List of daily returns
        risk_free_rate: Daily risk-free rate (default 0)

    Returns:
        Annualized Sharpe ratio
    """
    if len(daily_returns) < 2:
        return float("nan")

    import statistics

    mean_return = statistics.mean(daily_returns) - risk_free_rate
    std_return = statistics.stdev(daily_returns)

    if std_return == 0:
        return float("inf") if mean_return > 0 else float("-inf") if mean_return < 0 else 0.0

    # Annualize: sqrt(252 trading days)
    return (mean_return / std_return) * (252 ** 0.5)


def compute_probability_bins_accuracy(
    trades: List[TradeOutcome],
    n_bins: int = 5,
) -> List[Dict[str, Any]]:
    """
    Compute accuracy within probability bins.

    For each probability bin, compare predicted probability to actual hit rate.

    Args:
        trades: List of trade outcomes
        n_bins: Number of bins

    Returns:
        List of dicts with bin statistics
    """
    # Group trades by predicted probability bin
    bin_width = 1.0 / n_bins
    bins = [[] for _ in range(n_bins)]

    for t in trades:
        bin_idx = min(int(t.predicted_prob / bin_width), n_bins - 1)
        bins[bin_idx].append(t)

    results = []
    for i, bin_trades in enumerate(bins):
        bin_low = i * bin_width
        bin_high = (i + 1) * bin_width

        if bin_trades:
            avg_predicted = sum(t.predicted_prob for t in bin_trades) / len(bin_trades)
            actual_rate = sum(1 for t in bin_trades if t.actual_outcome == 1) / len(bin_trades)

            results.append({
                "bin_range": f"{bin_low:.0%}-{bin_high:.0%}",
                "count": len(bin_trades),
                "avg_predicted": round(avg_predicted, 4),
                "actual_rate": round(actual_rate, 4),
                "calibration_error": round(abs(actual_rate - avg_predicted), 4),
            })

    return results
