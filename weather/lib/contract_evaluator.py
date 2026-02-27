"""
Contract evaluator: ranks weather trading opportunities with threshold preference.

Key design decisions:
- Threshold contracts (-T, "Will high be >= X?") are preferred over bucket contracts
  (-B, "Will high be 21-22?") because they require only directional accuracy.
- Bucket contracts require a much higher EV bar (25% vs 15%) because they have
  ~15% base probability and are harder to beat.
- Uses the proper Kalshi 7% profit fee model via ev_yes()/ev_no().
- Enforces max 1 trade per city to avoid correlated bets.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from weather.lib.probability_ladder import ProbabilityLadder
from weather.lib.kalshi_weather import (
    EventSpec,
    parse_event_spec_from_ticker,
    best_buy_prices_from_snapshot_row,
)
from weather.lib.fees import ev_yes, ev_no

log = logging.getLogger(__name__)

# Default EV thresholds
DEFAULT_THRESHOLD_MIN_EV = 0.15   # 15% for threshold contracts
DEFAULT_BUCKET_MIN_EV = 0.25      # 25% for bucket contracts (much higher bar)
DEFAULT_MIN_PROB = 0.05           # Reject if model prob < 5%
DEFAULT_MAX_PROB = 0.95           # Reject if model prob > 95%
DEFAULT_MAX_ASK_CENTS = 85        # Don't buy contracts above 85 cents


@dataclass
class Opportunity:
    """A single trading opportunity with evaluation details."""
    city_key: str
    market_ticker: str
    contract_type: str         # "threshold" or "bucket"
    event_spec: EventSpec
    event_display: str
    side: str                  # "YES" or "NO"
    model_prob: float          # Model probability of YES outcome
    ask_cents: int             # Price to buy in cents
    ev: float                  # Expected value ($ per contract)
    half_kelly: float          # Half-Kelly fraction for position sizing
    forecast_high_f: int
    ladder_source: str


@dataclass
class EvaluationConfig:
    """Configuration for contract evaluation."""
    threshold_min_ev: float = DEFAULT_THRESHOLD_MIN_EV
    bucket_min_ev: float = DEFAULT_BUCKET_MIN_EV
    min_prob: float = DEFAULT_MIN_PROB
    max_prob: float = DEFAULT_MAX_PROB
    max_ask_cents: int = DEFAULT_MAX_ASK_CENTS
    max_trades_per_city: int = 1


def _half_kelly(prob: float, price: float) -> float:
    """
    Compute half-Kelly fraction for position sizing.

    Kelly fraction = (p * b - q) / b
    where p = win probability, q = 1-p, b = net odds (profit/stake)

    Half-Kelly = Kelly / 2 for safer sizing.
    """
    if price <= 0 or price >= 1.0:
        return 0.0
    b = (1.0 - price) / price  # net odds
    q = 1.0 - prob
    kelly = (prob * b - q) / b
    if kelly <= 0:
        return 0.0
    return kelly / 2.0


def evaluate_contracts(
    ladder: ProbabilityLadder,
    markets: List[Dict[str, Any]],
    city_key: str,
    forecast_high_f: int,
    ladder_source: str = "",
    config: Optional[EvaluationConfig] = None,
) -> List[Opportunity]:
    """
    Evaluate all contracts for a city/date and return ranked opportunities.

    Args:
        ladder: Probability ladder for the temperature distribution
        markets: List of Kalshi market snapshot dicts (from DB)
        city_key: City identifier
        forecast_high_f: Point forecast high temperature
        ladder_source: Description of how the ladder was built
        config: Evaluation configuration (uses defaults if None)

    Returns:
        List of Opportunity objects, sorted: threshold first, then by EV desc
    """
    if config is None:
        config = EvaluationConfig()

    opportunities: List[Opportunity] = []

    for market in markets:
        if market.get("status") not in ("active", None):
            # Skip non-active markets; None status means we don't know (OK to try)
            if market.get("status") is not None:
                continue

        ticker = market.get("market_ticker", "")
        if not ticker:
            continue

        # Parse contract type from ticker
        spec = parse_event_spec_from_ticker(ticker)
        if spec is None:
            continue

        # Get buy prices
        p_yes, p_no = best_buy_prices_from_snapshot_row(market)

        # Determine contract type and model probability
        if spec.kind == "ge":
            contract_type = "threshold"
            model_prob = ladder.prob_above(spec.a)
            min_ev = config.threshold_min_ev
        elif spec.kind == "between" and spec.b is not None:
            contract_type = "bucket"
            model_prob = ladder.prob_between(spec.a, spec.b)
            min_ev = config.bucket_min_ev
        else:
            continue

        # Skip extreme probabilities
        if not (config.min_prob <= model_prob <= config.max_prob):
            continue

        # Evaluate YES and NO sides
        best_side = None
        best_ev = 0.0
        best_price_cents = 0
        best_price_dollars = 0.0

        if p_yes is not None:
            yes_cents = int(round(p_yes * 100))
            if yes_cents <= config.max_ask_cents:
                yes_ev = ev_yes(model_prob, p_yes)
                if yes_ev >= min_ev and yes_ev > best_ev:
                    best_side = "YES"
                    best_ev = yes_ev
                    best_price_cents = yes_cents
                    best_price_dollars = p_yes

        if p_no is not None:
            no_cents = int(round(p_no * 100))
            if no_cents <= config.max_ask_cents:
                no_ev = ev_no(model_prob, p_no)
                if no_ev >= min_ev and no_ev > best_ev:
                    best_side = "NO"
                    best_ev = no_ev
                    best_price_cents = no_cents
                    best_price_dollars = p_no

        if best_side is None:
            continue

        # Compute half-Kelly
        if best_side == "YES":
            hk = _half_kelly(model_prob, best_price_dollars)
        else:
            hk = _half_kelly(1.0 - model_prob, best_price_dollars)

        # Build event display string
        event_display = spec.describe()

        opportunities.append(Opportunity(
            city_key=city_key,
            market_ticker=ticker,
            contract_type=contract_type,
            event_spec=spec,
            event_display=event_display,
            side=best_side,
            model_prob=model_prob,
            ask_cents=best_price_cents,
            ev=best_ev,
            half_kelly=hk,
            forecast_high_f=forecast_high_f,
            ladder_source=ladder_source,
        ))

    # Sort: threshold contracts first, then by EV descending
    opportunities.sort(key=lambda o: (
        0 if o.contract_type == "threshold" else 1,
        -o.ev,
    ))

    # Enforce max trades per city
    if config.max_trades_per_city and len(opportunities) > config.max_trades_per_city:
        opportunities = opportunities[:config.max_trades_per_city]

    return opportunities


def evaluate_all_cities(
    db_path: Path,
    city_markets: Dict[str, List[Dict[str, Any]]],
    city_ladders: Dict[str, "LadderResult"],
    config: Optional[EvaluationConfig] = None,
) -> List[Opportunity]:
    """
    Evaluate contracts across all cities and return a combined ranked list.

    Args:
        db_path: Path to weather database
        city_markets: Dict of {city_key: [market_dicts]}
        city_ladders: Dict of {city_key: LadderResult} from probability_pipeline
        config: Evaluation configuration

    Returns:
        Combined list of opportunities, sorted by threshold-first then EV desc
    """
    # Avoid circular import
    from weather.lib.probability_pipeline import LadderResult

    all_opportunities: List[Opportunity] = []

    for city_key, markets in city_markets.items():
        lr = city_ladders.get(city_key)
        if lr is None:
            log.warning("No ladder for %s, skipping", city_key)
            continue

        opps = evaluate_contracts(
            ladder=lr.ladder,
            markets=markets,
            city_key=city_key,
            forecast_high_f=lr.forecast_high_f,
            ladder_source=lr.source,
            config=config,
        )
        all_opportunities.extend(opps)

    # Global sort: threshold first, then EV descending
    all_opportunities.sort(key=lambda o: (
        0 if o.contract_type == "threshold" else 1,
        -o.ev,
    ))

    return all_opportunities
