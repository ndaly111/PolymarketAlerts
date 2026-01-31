"""Binance API client for BTC price and candle data."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional
import requests

BINANCE_BASE_URL = "https://api.binance.com"
SYMBOL = "BTCUSDT"


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
    url = f"{BINANCE_BASE_URL}/api/v3/klines"
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
    url = f"{BINANCE_BASE_URL}/api/v3/ticker/price"
    params = {"symbol": symbol}

    resp = requests.get(url, params=params, timeout=10)
    resp.raise_for_status()
    data = resp.json()

    return float(data["price"])


def fetch_ticker_24h(symbol: str = SYMBOL) -> dict:
    """Fetch 24-hour ticker statistics."""
    url = f"{BINANCE_BASE_URL}/api/v3/ticker/24hr"
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
