"""
Fee-adjusted edge calculation.

Based on weather/lib/fees.py pattern - provides fee-aware EV calculation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any


# Default Kalshi fee: 2 cents per contract
DEFAULT_FEE_CENTS = 2


@dataclass(frozen=True)
class FeeSchedule:
    """
    Fee model for Kalshi trades.

    v1 fee model: flat cents-per-contract open fee.
    Keep this explicit and configurable; fee schedules can change.
    """
    open_fee_cents: int = DEFAULT_FEE_CENTS

    def open_fee_dollars(self) -> float:
        """Return fee in dollars."""
        return float(self.open_fee_cents) / 100.0


def calculate_edge_with_fee(
    fair_prob: float,
    price_cents: int,
    fee_cents: int = DEFAULT_FEE_CENTS,
) -> Dict[str, Any]:
    """
    Calculate edge with fee adjustment.

    Args:
        fair_prob: Fair probability of winning (0.0-1.0)
        price_cents: Price to buy in cents (1-99)
        fee_cents: Fee per contract in cents (default 2)

    Returns:
        Dict with:
            - raw_edge: Edge before fees (fair_prob - price)
            - net_edge: Edge after fees (what you actually keep)
            - ev_cents: Expected value in cents per contract
            - price_cents: Input price
            - fee_cents: Fee applied
            - fair_prob: Input fair probability

    Example:
        >>> result = calculate_edge_with_fee(0.60, 50)
        >>> result["raw_edge"]  # 0.10 (10% raw edge)
        >>> result["net_edge"]  # 0.08 (8% after 2 cent fee)
        >>> result["ev_cents"]  # 8.0 cents expected profit
    """
    price_dollars = float(price_cents) / 100.0
    fee_dollars = float(fee_cents) / 100.0

    # Raw edge: fair_prob - price
    # This is the theoretical edge ignoring fees
    raw_edge = fair_prob - price_dollars

    # Net edge: raw_edge minus the fee (as a fraction of the $1 contract)
    # The fee reduces your EV by fee_dollars per contract
    net_edge = raw_edge - fee_dollars

    # EV in cents: net_edge * 100
    ev_cents = net_edge * 100

    return {
        "raw_edge": raw_edge,
        "net_edge": net_edge,
        "ev_cents": ev_cents,
        "price_cents": price_cents,
        "fee_cents": fee_cents,
        "fair_prob": fair_prob,
    }


def ev_yes(fair_prob: float, price_yes: float, fee_open: float) -> float:
    """
    Net EV ($ per contract) for buying YES and holding to expiration.

    Args:
        fair_prob: Fair probability of YES outcome (0.0-1.0)
        price_yes: Price to buy YES in dollars (0.0-1.0)
        fee_open: Fee in dollars

    Returns:
        Expected value in dollars per contract
    """
    return float(fair_prob) - float(price_yes) - float(fee_open)


def ev_no(fair_prob: float, price_no: float, fee_open: float) -> float:
    """
    Net EV ($ per contract) for buying NO and holding to expiration.

    Args:
        fair_prob: Fair probability of YES outcome (0.0-1.0)
        price_no: Price to buy NO in dollars (0.0-1.0)
        fee_open: Fee in dollars

    Returns:
        Expected value in dollars per contract
    """
    return float(1.0 - fair_prob) - float(price_no) - float(fee_open)


def kelly_fraction(
    fair_prob: float,
    price_cents: int,
    fee_cents: int = DEFAULT_FEE_CENTS,
    multiplier: float = 0.5,
) -> float:
    """
    Calculate Kelly criterion fraction of bankroll to bet.

    Args:
        fair_prob: Fair probability of winning (0.0-1.0)
        price_cents: Price in cents (1-99)
        fee_cents: Fee per contract in cents
        multiplier: Kelly multiplier (0.5 = half Kelly, recommended)

    Returns:
        Fraction of bankroll to bet (0.0-1.0)

    Kelly formula for binary bets:
        kelly = (p * b - q) / b

    For Kalshi contracts:
        - Win payout: $1 - price - fee
        - Loss: price + fee
        - b = (1 - price - fee) / (price + fee)

    Simplified (ignoring fee in denominator for stability):
        kelly = edge / (1 - price)
    """
    price_dollars = float(price_cents) / 100.0
    fee_dollars = float(fee_cents) / 100.0

    # Net edge after fees
    edge = fair_prob - price_dollars - fee_dollars

    if edge <= 0:
        return 0.0

    # Denominator: potential profit per dollar risked
    # At price p, you risk p and win (1-p), so multiplier is (1-p)
    profit_multiplier = 1.0 - price_dollars

    if profit_multiplier <= 0:
        return 0.0

    full_kelly = edge / profit_multiplier

    # Apply multiplier (half Kelly is standard for safety)
    return max(0.0, min(1.0, full_kelly * multiplier))


def round_kelly(kelly: float) -> float:
    """
    Round Kelly down to nearest 5% increment, with 2.5% as an option.

    This provides cleaner bet sizing for practical use.
    """
    if kelly <= 0:
        return 0.0
    if kelly < 0.025:
        return 0.0
    if kelly < 0.05:
        return 0.025
    # Round down to nearest 5%
    return int(kelly * 20) / 20.0
