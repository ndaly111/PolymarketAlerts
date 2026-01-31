"""Orderbook imbalance analysis for Kalshi markets."""

from __future__ import annotations

from typing import Optional, Dict, Any
from dataclasses import dataclass


@dataclass
class OrderbookImbalance:
    """Orderbook imbalance data."""
    yes_bid_volume: int
    yes_ask_volume: int
    no_bid_volume: int
    no_ask_volume: int
    imbalance_ratio: float  # >1 = more buying pressure, <1 = more selling
    bid_ask_spread: float
    depth_ratio: float  # Bid depth / Ask depth


def compute_orderbook_imbalance(orderbook: Optional[Dict[str, Any]]) -> Optional[OrderbookImbalance]:
    """Compute orderbook imbalance metrics."""
    if not orderbook:
        return None

    try:
        yes_bids = orderbook.get("yes", {}).get("bids", [])
        yes_asks = orderbook.get("yes", {}).get("asks", [])
        no_bids = orderbook.get("no", {}).get("bids", [])
        no_asks = orderbook.get("no", {}).get("asks", [])

        # Sum volume at top 5 levels
        yes_bid_vol = sum(b.get("count", 0) for b in yes_bids[:5])
        yes_ask_vol = sum(a.get("count", 0) for a in yes_asks[:5])
        no_bid_vol = sum(b.get("count", 0) for b in no_bids[:5])
        no_ask_vol = sum(a.get("count", 0) for a in no_asks[:5])

        # Imbalance ratio (bid volume / ask volume)
        total_bid = yes_bid_vol + no_bid_vol
        total_ask = yes_ask_vol + no_ask_vol
        imbalance = total_bid / total_ask if total_ask > 0 else 1.0

        # Best bid/ask for spread calculation
        best_yes_bid = yes_bids[0]["price"] if yes_bids else 0
        best_yes_ask = yes_asks[0]["price"] if yes_asks else 100
        spread = (best_yes_ask - best_yes_bid) / 100  # As decimal

        # Depth ratio (how much support vs resistance)
        depth_ratio = total_bid / total_ask if total_ask > 0 else 1.0

        return OrderbookImbalance(
            yes_bid_volume=yes_bid_vol,
            yes_ask_volume=yes_ask_vol,
            no_bid_volume=no_bid_vol,
            no_ask_volume=no_ask_vol,
            imbalance_ratio=imbalance,
            bid_ask_spread=spread,
            depth_ratio=depth_ratio,
        )
    except Exception as e:
        print(f"[orderbook] Error computing imbalance: {e}")
        return None


def compute_orderbook_features(imbalance: Optional[OrderbookImbalance]) -> Dict[str, float]:
    """Compute features from orderbook imbalance."""
    if not imbalance:
        return {
            "ob_imbalance": 0,
            "ob_imbalance_strong": 0,
            "ob_spread": 0.05,
            "ob_spread_tight": 0,
            "ob_depth_ratio": 1.0,
            "ob_yes_pressure": 0,
            "ob_no_pressure": 0,
        }

    # Normalize imbalance to -1 to 1 scale
    # >1 = buying pressure (bullish), <1 = selling pressure (bearish)
    normalized = (imbalance.imbalance_ratio - 1) / max(imbalance.imbalance_ratio, 1)
    normalized = max(-1, min(1, normalized))

    # Strong imbalance indicator
    strong_imbalance = 1 if imbalance.imbalance_ratio > 1.5 or imbalance.imbalance_ratio < 0.67 else 0

    # Tight spread indicator (good liquidity)
    tight_spread = 1 if imbalance.bid_ask_spread < 0.03 else 0

    # Directional pressure
    yes_pressure = 1 if imbalance.yes_bid_volume > imbalance.yes_ask_volume * 1.2 else 0
    no_pressure = 1 if imbalance.no_bid_volume > imbalance.no_ask_volume * 1.2 else 0

    return {
        "ob_imbalance": normalized,
        "ob_imbalance_strong": strong_imbalance,
        "ob_spread": imbalance.bid_ask_spread,
        "ob_spread_tight": tight_spread,
        "ob_depth_ratio": imbalance.depth_ratio,
        "ob_yes_pressure": yes_pressure,
        "ob_no_pressure": no_pressure,
    }
