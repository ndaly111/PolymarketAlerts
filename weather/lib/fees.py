from __future__ import annotations

from dataclasses import dataclass


# Kalshi's fee rate on winning profits (as of 2024)
KALSHI_PROFIT_FEE_RATE = 0.07


@dataclass(frozen=True)
class FeeSchedule:
    """
    Kalshi fee model: 7% fee on winning profits only.

    Example at 40¢ ask:
    - Win: payout $1, cost 40¢, gross profit 60¢, fee = 60¢ * 7% = 4.2¢
    - Loss: lose 40¢, no fee

    Keep this explicit and configurable; Kalshi fee schedules can change.
    """

    profit_fee_rate: float = KALSHI_PROFIT_FEE_RATE

    # Legacy: flat open fee (deprecated but kept for backwards compatibility)
    open_fee_cents: int = 0

    def open_fee_dollars(self) -> float:
        """Legacy method - returns 0 since we now use profit-based fees."""
        return 0.0

    def fee_on_win(self, price: float) -> float:
        """Calculate fee on a winning trade given the entry price."""
        gross_profit = 1.0 - price
        return gross_profit * self.profit_fee_rate

    def net_profit_on_win(self, price: float) -> float:
        """Calculate net profit on win after fees."""
        gross_profit = 1.0 - price
        return gross_profit * (1.0 - self.profit_fee_rate)


def ev_yes(q: float, p_yes: float, fee_open: float = 0.0) -> float:
    """
    Net EV ($ per contract) for buying YES and holding to expiration.

    Uses accurate 7% fee on profits model:
    - Win (prob q): profit = (1 - price) * 0.93 (after 7% fee)
    - Loss (prob 1-q): lose price

    Args:
        q: Model's probability that YES wins
        p_yes: Price to buy YES (0-1 scale, e.g., 0.40 for 40¢)
        fee_open: Deprecated, ignored (kept for API compatibility)

    Returns:
        Expected value in dollars per contract
    """
    price = float(p_yes)
    prob = float(q)

    # Net profit if we win (after 7% fee on profits)
    gross_profit_if_win = 1.0 - price
    net_profit_if_win = gross_profit_if_win * (1.0 - KALSHI_PROFIT_FEE_RATE)

    # EV = P(win) * net_profit - P(lose) * stake
    return prob * net_profit_if_win - (1.0 - prob) * price


def ev_no(q: float, p_no: float, fee_open: float = 0.0) -> float:
    """
    Net EV ($ per contract) for buying NO and holding to expiration.

    Uses accurate 7% fee on profits model:
    - Win (prob 1-q): profit = (1 - price) * 0.93 (after 7% fee)
    - Loss (prob q): lose price

    Args:
        q: Model's probability that YES wins (so NO wins with prob 1-q)
        p_no: Price to buy NO (0-1 scale)
        fee_open: Deprecated, ignored (kept for API compatibility)

    Returns:
        Expected value in dollars per contract
    """
    price = float(p_no)
    prob_no_wins = 1.0 - float(q)

    # Net profit if we win (after 7% fee on profits)
    gross_profit_if_win = 1.0 - price
    net_profit_if_win = gross_profit_if_win * (1.0 - KALSHI_PROFIT_FEE_RATE)

    # EV = P(win) * net_profit - P(lose) * stake
    return prob_no_wins * net_profit_if_win - float(q) * price
