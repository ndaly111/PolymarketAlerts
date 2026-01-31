"""Binance API client for BTC price and candle data."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional
import requests

# Try multiple endpoints (Binance blocked in some regions)
BINANCE_ENDPOINTS = [
    "https://api.binance.us",      # Binance US
    "https://api.binance.com",     # Global (blocked in US)
    "https://api1.binance.com",    # Backup
]
SYMBOL = "BTCUSDT"


def _get_working_endpoint() -> str:
    """Find a working Binance endpoint."""
    for endpoint in BINANCE_ENDPOINTS:
        try:
            resp = requests.get(f"{endpoint}/api/v3/ping", timeout=5)
            if resp.status_code == 200:
                return endpoint
        except Exception:
            continue
    return BINANCE_ENDPOINTS[0]  # Default to US


# Cache the working endpoint
_cached_endpoint: Optional[str] = None


def get_endpoint() -> str:
    global _cached_endpoint
    if _cached_endpoint is None:
        _cached_endpoint = _get_working_endpoint()
        print(f"[binance] Using endpoint: {_cached_endpoint}")
    return _cached_endpoint


@dataclass
class Candle:
    """OHLCV candle data."""
    open_time: int      # Unix timestamp ms
    open: float
    high: float
    low: float
    close: float
    volume: float
    close_time: int


def fetch_klines(
    interval: str = "1m",
    limit: int = 60,
    symbol: str = SYMBOL,
) -> List[Candle]:
    """Fetch kline/candlestick data from Binance.

    Args:
        interval: Kline interval (1m, 5m, 15m, etc.)
        limit: Number of candles to fetch (max 1000)
        symbol: Trading pair symbol

    Returns:
        List of Candle objects, oldest first
    """
    url = f"{get_endpoint()}/api/v3/klines"
    params = {
        "symbol": symbol,
        "interval": interval,
        "limit": limit,
    }

    resp = requests.get(url, params=params, timeout=10)
    resp.raise_for_status()
    data = resp.json()

    candles = []
    for row in data:
        candles.append(Candle(
            open_time=int(row[0]),
            open=float(row[1]),
            high=float(row[2]),
            low=float(row[3]),
            close=float(row[4]),
            volume=float(row[5]),
            close_time=int(row[6]),
        ))

    return candles


def fetch_current_price(symbol: str = SYMBOL) -> float:
    """Fetch current BTC price from Binance."""
    url = f"{get_endpoint()}/api/v3/ticker/price"
    params = {"symbol": symbol}

    resp = requests.get(url, params=params, timeout=10)
    resp.raise_for_status()
    data = resp.json()

    return float(data["price"])


def fetch_ticker_24h(symbol: str = SYMBOL) -> dict:
    """Fetch 24-hour ticker statistics."""
    url = f"{get_endpoint()}/api/v3/ticker/24hr"
    params = {"symbol": symbol}

    resp = requests.get(url, params=params, timeout=10)
    resp.raise_for_status()
    data = resp.json()

    return {
        "price": float(data["lastPrice"]),
        "price_change_pct": float(data["priceChangePercent"]),
        "high_24h": float(data["highPrice"]),
        "low_24h": float(data["lowPrice"]),
        "volume_24h": float(data["volume"]),
        "quote_volume_24h": float(data["quoteVolume"]),
    }


def fetch_order_book(symbol: str = SYMBOL, limit: int = 20) -> Optional[dict]:
    """Fetch order book and compute imbalance.

    Returns dict with bid/ask pressure and imbalance metrics.
    """
    url = f"{get_endpoint()}/api/v3/depth"
    params = {"symbol": symbol, "limit": limit}

    try:
        resp = requests.get(url, params=params, timeout=10)
        resp.raise_for_status()
        data = resp.json()

        bids = data.get("bids", [])
        asks = data.get("asks", [])

        # Sum up volumes at each level
        bid_volume = sum(float(b[1]) for b in bids)
        ask_volume = sum(float(a[1]) for a in asks)

        total_volume = bid_volume + ask_volume

        # Imbalance: positive = more buyers, negative = more sellers
        if total_volume > 0:
            imbalance = (bid_volume - ask_volume) / total_volume
        else:
            imbalance = 0

        # Best bid/ask
        best_bid = float(bids[0][0]) if bids else None
        best_ask = float(asks[0][0]) if asks else None

        spread = None
        spread_pct = None
        if best_bid and best_ask:
            spread = best_ask - best_bid
            spread_pct = spread / best_bid * 100

        return {
            "bid_volume": bid_volume,
            "ask_volume": ask_volume,
            "imbalance": imbalance,  # -1 to 1
            "best_bid": best_bid,
            "best_ask": best_ask,
            "spread": spread,
            "spread_pct": spread_pct,
        }

    except Exception as e:
        return None


def fetch_funding_rate() -> Optional[dict]:
    """Fetch BTC perpetual funding rate from Binance Futures.

    Positive funding = longs pay shorts (bullish sentiment)
    Negative funding = shorts pay longs (bearish sentiment)
    """
    # Binance Futures API (different endpoint)
    url = "https://fapi.binance.com/fapi/v1/fundingRate"
    params = {"symbol": "BTCUSDT", "limit": 1}

    try:
        resp = requests.get(url, params=params, timeout=10)
        resp.raise_for_status()
        data = resp.json()

        if data:
            funding_rate = float(data[0]["fundingRate"])
            funding_time = int(data[0]["fundingTime"])

            return {
                "funding_rate": funding_rate,
                "funding_rate_pct": funding_rate * 100,
                "funding_time": funding_time,
                # Sentiment interpretation
                "sentiment": "bullish" if funding_rate > 0.0001 else (
                    "bearish" if funding_rate < -0.0001 else "neutral"
                ),
            }
    except Exception:
        pass

    return None


def fetch_open_interest() -> Optional[dict]:
    """Fetch BTC perpetual open interest from Binance Futures."""
    url = "https://fapi.binance.com/fapi/v1/openInterest"
    params = {"symbol": "BTCUSDT"}

    try:
        resp = requests.get(url, params=params, timeout=10)
        resp.raise_for_status()
        data = resp.json()

        return {
            "open_interest": float(data["openInterest"]),
            "symbol": data["symbol"],
        }
    except Exception:
        pass

    return None
