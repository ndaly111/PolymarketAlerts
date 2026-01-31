"""Kalshi BTC market integration.

Fetches BTC price prediction markets from Kalshi.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional, List, Dict, Any

# Add parent directory to path to import kalshi_auth_client
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

try:
    from kalshi_auth_client import KalshiAuthClient
except ImportError:
    KalshiAuthClient = None


@dataclass
class BTCMarket:
    """Represents a Kalshi BTC price market."""
    ticker: str
    title: str
    strike_price: float
    expiry_time: datetime
    yes_bid: Optional[float]
    yes_ask: Optional[float]
    no_bid: Optional[float]
    no_ask: Optional[float]
    last_price: Optional[float]
    volume: int
    open_interest: int
    status: str


def get_kalshi_client() -> Optional[KalshiAuthClient]:
    """Get authenticated Kalshi client from environment."""
    if KalshiAuthClient is None:
        return None

    try:
        return KalshiAuthClient.from_env()
    except Exception as e:
        print(f"[kalshi] Failed to create client: {e}")
        return None


def fetch_btc_markets(
    client: Optional[KalshiAuthClient] = None,
    series_ticker: str = "KXBTC",
) -> List[BTCMarket]:
    """Fetch active BTC price markets from Kalshi.

    Args:
        client: Kalshi auth client (created from env if None)
        series_ticker: Market series ticker (KXBTC for BTC price)

    Returns:
        List of BTCMarket objects for active markets
    """
    if client is None:
        client = get_kalshi_client()

    if client is None:
        return []

    try:
        # Fetch markets in the BTC series
        response = client.get(f"/markets", params={
            "series_ticker": series_ticker,
            "status": "open",
        })

        if not response or "markets" not in response:
            return []

        markets = []
        for m in response["markets"]:
            try:
                # Parse expiry time
                expiry_str = m.get("close_time") or m.get("expiration_time")
                if expiry_str:
                    expiry_time = datetime.fromisoformat(
                        expiry_str.replace("Z", "+00:00")
                    )
                else:
                    continue

                # Extract strike price from title or subtitle
                strike_price = _extract_strike_price(m)
                if strike_price is None:
                    continue

                markets.append(BTCMarket(
                    ticker=m["ticker"],
                    title=m.get("title", ""),
                    strike_price=strike_price,
                    expiry_time=expiry_time,
                    yes_bid=_safe_float(m.get("yes_bid")),
                    yes_ask=_safe_float(m.get("yes_ask")),
                    no_bid=_safe_float(m.get("no_bid")),
                    no_ask=_safe_float(m.get("no_ask")),
                    last_price=_safe_float(m.get("last_price")),
                    volume=int(m.get("volume", 0)),
                    open_interest=int(m.get("open_interest", 0)),
                    status=m.get("status", "unknown"),
                ))
            except Exception as e:
                continue

        return markets

    except Exception as e:
        print(f"[kalshi] Error fetching markets: {e}")
        return []


def fetch_btc_15min_markets(
    client: Optional[KalshiAuthClient] = None,
) -> List[BTCMarket]:
    """Fetch specifically 15-minute BTC markets.

    Filters for markets expiring within the next 20 minutes.
    """
    markets = fetch_btc_markets(client)

    now = datetime.now(timezone.utc)
    filtered = []

    for m in markets:
        time_to_expiry = (m.expiry_time - now).total_seconds()

        # Keep markets expiring in 1-20 minutes
        if 60 <= time_to_expiry <= 1200:
            filtered.append(m)

    # Sort by expiry time
    filtered.sort(key=lambda x: x.expiry_time)

    return filtered


def get_next_btc_market(
    client: Optional[KalshiAuthClient] = None,
) -> Optional[BTCMarket]:
    """Get the next BTC market to expire."""
    markets = fetch_btc_15min_markets(client)
    return markets[0] if markets else None


def fetch_orderbook(
    ticker: str,
    client: Optional[KalshiAuthClient] = None,
) -> Optional[Dict[str, Any]]:
    """Fetch orderbook for a specific market."""
    if client is None:
        client = get_kalshi_client()

    if client is None:
        return None

    try:
        response = client.get(f"/markets/{ticker}/orderbook")
        return response
    except Exception as e:
        print(f"[kalshi] Error fetching orderbook: {e}")
        return None


def _safe_float(value) -> Optional[float]:
    """Safely convert value to float."""
    if value is None:
        return None
    try:
        return float(value)
    except (ValueError, TypeError):
        return None


def _extract_strike_price(market: Dict[str, Any]) -> Optional[float]:
    """Extract strike price from market data.

    Kalshi BTC markets typically have the strike price in the title
    or in a structured field.
    """
    # Try structured fields first
    if "floor_strike" in market:
        return _safe_float(market["floor_strike"])
    if "cap_strike" in market:
        return _safe_float(market["cap_strike"])

    # Try to parse from title (e.g., "BTC above $100,000")
    title = market.get("title", "") + " " + market.get("subtitle", "")

    import re
    # Match patterns like $100,000 or $100000 or 100,000 or 100000
    match = re.search(r"\$?([\d,]+(?:\.\d+)?)", title)
    if match:
        try:
            return float(match.group(1).replace(",", ""))
        except ValueError:
            pass

    return None


def calculate_market_implied_prob(market: BTCMarket) -> Dict[str, float]:
    """Calculate implied probabilities from market prices.

    Returns:
        Dict with 'up_prob' and 'down_prob' as decimals (0-1)
    """
    # Use mid price if both bid/ask available, otherwise use what's available
    if market.yes_bid is not None and market.yes_ask is not None:
        yes_mid = (market.yes_bid + market.yes_ask) / 2
    elif market.yes_ask is not None:
        yes_mid = market.yes_ask
    elif market.yes_bid is not None:
        yes_mid = market.yes_bid
    elif market.last_price is not None:
        yes_mid = market.last_price
    else:
        yes_mid = 0.5

    # Kalshi prices are in cents (0-100), convert to probability
    up_prob = yes_mid / 100.0 if yes_mid > 1 else yes_mid

    return {
        "up_prob": up_prob,
        "down_prob": 1.0 - up_prob,
    }
