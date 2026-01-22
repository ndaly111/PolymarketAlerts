from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

_NUM = r"(-?\d{1,3})"

_RX_BETWEEN = re.compile(r"between\s+" + _NUM + r"\D+and\D+" + _NUM, re.IGNORECASE)
_RX_RANGE_DASH = re.compile(r"\b" + _NUM + r"\s*[-–]\s*" + _NUM + r"\b")
_RX_OR_HIGHER = re.compile(r"\b" + _NUM + r"\s*(?:°\s*)?(?:or\s+higher|and\s+above|\+)", re.IGNORECASE)
_RX_OR_LOWER = re.compile(r"\b" + _NUM + r"\s*(?:°\s*)?(?:or\s+lower|and\s+below)", re.IGNORECASE)
_RX_BELOW = re.compile(r"(?:below|under)\s+" + _NUM, re.IGNORECASE)
_RX_ABOVE = re.compile(r"(?:above|over|greater\s+than)\s+" + _NUM, re.IGNORECASE)
_RX_LESS_THAN = re.compile(r"less\s+than\s+" + _NUM, re.IGNORECASE)


@dataclass(frozen=True)
class EventSpec:
    # One of: exact, between, ge, le, lt, gt
    kind: str
    a: int
    b: Optional[int] = None

    def describe(self) -> str:
        if self.kind == "exact":
            return f"TMAX = {self.a}"
        if self.kind == "between":
            return f"{self.a} ≤ TMAX ≤ {int(self.b)}"
        if self.kind == "ge":
            return f"TMAX ≥ {self.a}"
        if self.kind == "le":
            return f"TMAX ≤ {self.a}"
        if self.kind == "lt":
            return f"TMAX < {self.a}"
        if self.kind == "gt":
            return f"TMAX > {self.a}"
        return f"{self.kind}({self.a},{self.b})"


def parse_event_spec_from_title(title: str) -> Optional[EventSpec]:
    """Parse a Kalshi weather market title into an EventSpec.

    Returns None if we cannot confidently parse the strike definition.
    """
    t = (title or "").strip()
    if not t:
        return None

    m = _RX_BETWEEN.search(t)
    if m:
        a, b = int(m.group(1)), int(m.group(2))
        lo, hi = (a, b) if a <= b else (b, a)
        return EventSpec(kind="between", a=lo, b=hi)

    m = _RX_RANGE_DASH.search(t)
    if m:
        a, b = int(m.group(1)), int(m.group(2))
        lo, hi = (a, b) if a <= b else (b, a)
        # Avoid false positives on dates like 2026-01-22.
        if abs(lo) <= 120 and abs(hi) <= 120:
            return EventSpec(kind="between", a=lo, b=hi)

    m = _RX_OR_HIGHER.search(t)
    if m:
        return EventSpec(kind="ge", a=int(m.group(1)))

    m = _RX_OR_LOWER.search(t)
    if m:
        return EventSpec(kind="le", a=int(m.group(1)))

    m = _RX_BELOW.search(t)
    if m:
        return EventSpec(kind="lt", a=int(m.group(1)))

    m = _RX_LESS_THAN.search(t)
    if m:
        return EventSpec(kind="lt", a=int(m.group(1)))

    m = _RX_ABOVE.search(t)
    if m:
        return EventSpec(kind="gt", a=int(m.group(1)))

    # Exact: titles may contain dates/other numbers, so we only accept
    # if exactly one plausible temperature integer exists.
    nums = [int(x) for x in re.findall(_NUM, t)]
    plausible = [n for n in nums if -80 <= n <= 140]
    if len(plausible) == 1:
        return EventSpec(kind="exact", a=int(plausible[0]))

    return None


def prob_event(pmf_high_f: Dict[int, float], spec: EventSpec) -> float:
    """Map a discrete PMF over TMAX to event probability q."""
    if not pmf_high_f:
        return 0.0

    if spec.kind == "exact":
        return float(pmf_high_f.get(int(spec.a), 0.0))

    if spec.kind == "between" and spec.b is not None:
        lo, hi = int(spec.a), int(spec.b)
        return float(sum(v for k, v in pmf_high_f.items() if lo <= int(k) <= hi))

    if spec.kind == "ge":
        x = int(spec.a)
        return float(sum(v for k, v in pmf_high_f.items() if int(k) >= x))

    if spec.kind == "le":
        x = int(spec.a)
        return float(sum(v for k, v in pmf_high_f.items() if int(k) <= x))

    if spec.kind == "gt":
        x = int(spec.a)
        return float(sum(v for k, v in pmf_high_f.items() if int(k) > x))

    if spec.kind == "lt":
        x = int(spec.a)
        return float(sum(v for k, v in pmf_high_f.items() if int(k) < x))

    return 0.0


def best_buy_prices_from_snapshot_row(row: Dict[str, object]) -> Tuple[Optional[float], Optional[float]]:
    """Return (p_yes, p_no) buy prices in dollars from a snapshot DB row (asks)."""
    yes_ask = row.get("yes_ask")
    no_ask = row.get("no_ask")

    def _cents_to_dollars(v) -> Optional[float]:
        try:
            if v is None:
                return None
            c = int(v)
            if c < 0:
                return None
            return float(c) / 100.0
        except Exception:
            return None

    return _cents_to_dollars(yes_ask), _cents_to_dollars(no_ask)
