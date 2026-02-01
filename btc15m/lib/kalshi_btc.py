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
    series_tickers: Optional[List[str]] = None,
) -> List[BTCMarket]:
    """Fetch active BTC price markets from Kalshi.

    Args:
        client: Kalshi auth client (created from env if None)
        series_tickers: List of market series tickers to search

    Returns:
        List of BTCMarket objects for active markets
    """
    if client is None:
        client = get_kalshi_client()

    if client is None:
        return []

    # Try multiple series tickers for BTC markets
    if series_tickers is None:
        series_tickers = [
            "KXBTC",      # Standard BTC price
            "KXBTCD",     # Daily BTC
            "KXBTC15",    # 15-minute BTC
            "BTC15",      # Alt 15-min ticker
            "BTCUSD",     # BTC/USD
        ]

    all_markets = []
    for series_ticker in series_tickers:
        try:
            result = client.list_markets(series_ticker=series_ticker, status="open")
            if result:
                print(f"[kalshi] Found {len(result)} markets for {series_ticker}")
                all_markets.extend(result)
        except Exception:
            pass

    # Also try searching by ticker prefix if no results
    if not all_markets:
        try:
            all_open = client.list_markets(status="open")
            if all_open:
                btc_markets = [m for m in all_open if "BTC" in m.get("ticker", "").upper() or "BTC" in m.get("title", "").upper()]
                print(f"[kalshi] Found {len(btc_markets)} BTC markets from full scan")
                all_markets.extend(btc_markets)
        except Exception as e:
            print(f"[kalshi] Full scan failed: {e}")

    if not all_markets:
        print("[kalshi] No BTC markets found")
        return []

    markets = []
    for m in all_markets:
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
        except Exception:
            continue

    print(f"[kalshi] Parsed {len(markets)} valid BTC markets")
    return markets


def fetch_btc_15min_markets(
    client: Optional[KalshiAuthClient] = None,
    current_btc_price: Optional[float] = None,
) -> List[BTCMarket]:
    """Fetch BTC 15-minute markets from KXBTC15M series.

    These are simple up/down markets that expire every 15 minutes.
    Only 1 market exists at a time (created ~15 min before expiry).

    Args:
        client: Kalshi auth client
        current_btc_price: Current BTC price (not used for filtering, just for logging)
    """
    if client is None:
        client = get_kalshi_client()

    if client is None:
        return []

    # Fetch 15-minute BTC markets directly
    try:
        result = client.list_markets(series_ticker="KXBTC15M", status="open")
    except Exception as e:
        print(f"[kalshi] Error fetching KXBTC15M: {e}")
        return []

    if not result:
        return []

    print(f"[kalshi] Found {len(result)} 15-min BTC markets")

    now = datetime.now(timezone.utc)
    markets = []

    for m in result:
        try:
            # Parse expiry time
            expiry_str = m.get("close_time") or m.get("expiration_time")
            if expiry_str:
                expiry_time = datetime.fromisoformat(expiry_str.replace("Z", "+00:00"))
            else:
                continue

            time_to_expiry = (expiry_time - now).total_seconds()

            # Keep markets expiring in 1-15 minutes
            if not (60 <= time_to_expiry <= 900):
                continue

            # Try to extract strike from market data (title, floor_strike, etc.)
            strike_price = _extract_strike_price(m)

            # If no strike found, try parsing from ticker (e.g., KXBTC15M-26FEB010930-30)
            if strike_price is None:
                ticker = m.get("ticker", "")
                # Try to get strike from last segment of ticker
                parts = ticker.split("-")
                if len(parts) >= 2:
                    try:
                        # Last part might be strike offset or bucket (e.g., "30" for $78,030)
                        last_part = parts[-1]
                        if last_part.isdigit():
                            # Could be full strike or offset - check magnitude
                            val = int(last_part)
                            if val > 1000:  # Full strike like 78030
                                strike_price = float(val)
                            else:
                                # Offset from round number - use current price as base
                                base = round((current_btc_price or 78000) / 1000) * 1000
                                strike_price = base + val
                    except:
                        pass

            # Fallback to current price if still no strike
            if strike_price is None:
                strike_price = current_btc_price or 0

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

            tte_min = int(time_to_expiry / 60)
            print(f"[kalshi] {m['ticker']}: expires in {tte_min}m, yes_ask={m.get('yes_ask')}")

        except Exception:
            continue

    return markets


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
        return client.get_orderbook(ticker)
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
        Also includes bid/ask for accurate trade execution
    """
    # Convert from cents to probability (Kalshi uses 0-100 cents)
    def to_prob(val):
        if val is None:
            return None
        return val / 100.0 if val > 1 else val

    yes_bid = to_prob(market.yes_bid)
    yes_ask = to_prob(market.yes_ask)
    no_bid = to_prob(market.no_bid)
    no_ask = to_prob(market.no_ask)
    last_price = to_prob(market.last_price)

    # Mid price for general probability estimate
    if yes_bid is not None and yes_ask is not None:
        yes_mid = (yes_bid + yes_ask) / 2
    elif yes_ask is not None:
        yes_mid = yes_ask
    elif yes_bid is not None:
        yes_mid = yes_bid
    elif last_price is not None:
        yes_mid = last_price
    else:
        yes_mid = 0.5

    up_prob = yes_mid
    down_prob = 1.0 - up_prob

    return {
        "up_prob": up_prob,
        "down_prob": down_prob,
        # For buying YES on UP (the ask is what you pay)
        "up_ask": yes_ask if yes_ask is not None else up_prob,
        "up_bid": yes_bid if yes_bid is not None else up_prob,
        # For buying YES on DOWN (no_ask, or equivalently 1 - yes_bid)
        "down_ask": no_ask if no_ask is not None else down_prob,
        "down_bid": no_bid if no_bid is not None else down_prob,
        # Spread info
        "up_spread": (yes_ask - yes_bid) if (yes_ask and yes_bid) else None,
        "down_spread": (no_ask - no_bid) if (no_ask and no_bid) else None,
        "has_liquidity": yes_bid is not None and yes_ask is not None,
    }


def get_execution_price(market: BTCMarket, side: str) -> float:
    """Get the price we'd actually pay to enter a trade.

    Args:
        market: BTCMarket object
        side: 'UP' or 'DOWN'

    Returns:
        Execution price as probability (0-1)
    """
    probs = calculate_market_implied_prob(market)

    if side == "UP":
        # Buying YES on "price goes up" = pay the ask
        return probs["up_ask"]
    else:
        # Buying YES on "price goes down" = pay the ask on NO side
        return probs["down_ask"]


def get_orderbook_best_prices(
    ticker: str,
    client: Optional[KalshiAuthClient] = None,
) -> Optional[Dict[str, float]]:
    """Fetch orderbook and get best bid/ask prices.

    More accurate than market snapshot for actual execution.
    """
    orderbook = fetch_orderbook(ticker, client)
    if not orderbook:
        return None

    try:
        yes_bids = orderbook.get("yes", {}).get("bids", [])
        yes_asks = orderbook.get("yes", {}).get("asks", [])
        no_bids = orderbook.get("no", {}).get("bids", [])
        no_asks = orderbook.get("no", {}).get("asks", [])

        # Best bid is highest, best ask is lowest
        best_yes_bid = max((b["price"] for b in yes_bids), default=None)
        best_yes_ask = min((a["price"] for a in yes_asks), default=None)
        best_no_bid = max((b["price"] for b in no_bids), default=None)
        best_no_ask = min((a["price"] for a in no_asks), default=None)

        # Convert cents to probability
        def to_prob(val):
            return val / 100.0 if val and val > 1 else val

        return {
            "yes_bid": to_prob(best_yes_bid),
            "yes_ask": to_prob(best_yes_ask),
            "no_bid": to_prob(best_no_bid),
            "no_ask": to_prob(best_no_ask),
            "yes_bid_size": sum(b.get("count", 0) for b in yes_bids[:3]),
            "yes_ask_size": sum(a.get("count", 0) for a in yes_asks[:3]),
        }
    except Exception as e:
        print(f"[kalshi] Error parsing orderbook: {e}")
        return None
