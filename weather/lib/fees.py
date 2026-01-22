from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class FeeSchedule:
    """
    v1 fee model: a flat cents-per-contract open fee.
    Keep this explicit and configurable; Kalshi fee schedules can change.
    """

    open_fee_cents: int = 2

    def open_fee_dollars(self) -> float:
        return float(self.open_fee_cents) / 100.0


def ev_yes(q: float, p_yes: float, fee_open: float) -> float:
    """Net EV ($ per contract) for buying YES and holding to expiration."""
    return float(q) - float(p_yes) - float(fee_open)


def ev_no(q: float, p_no: float, fee_open: float) -> float:
    """Net EV ($ per contract) for buying NO and holding to expiration."""
    return float(1.0 - q) - float(p_no) - float(fee_open)
