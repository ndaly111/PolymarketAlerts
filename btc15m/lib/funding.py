"""Funding rate data from perpetual futures exchanges."""

from __future__ import annotations

import requests
from typing import Optional, Dict
from dataclasses import dataclass


@dataclass
class FundingData:
    """Funding rate data."""
    btc_funding_rate: float  # Current funding rate (8h)
    btc_funding_annualized: float  # Annualized rate
    eth_funding_rate: float
    funding_sentiment: str  # "bullish", "bearish", "neutral"
    long_short_ratio: Optional[float] = None


def fetch_binance_funding() -> Optional[Dict[str, float]]:
    """Fetch funding rates from Binance Futures."""
    try:
        # BTC perpetual funding
        btc_resp = requests.get(
            "https://fapi.binance.com/fapi/v1/fundingRate",
            params={"symbol": "BTCUSDT", "limit": 1},
            timeout=10
        )
        btc_data = btc_resp.json()
        btc_rate = float(btc_data[0]["fundingRate"]) if btc_data else 0

        # ETH perpetual funding
        eth_resp = requests.get(
            "https://fapi.binance.com/fapi/v1/fundingRate",
            params={"symbol": "ETHUSDT", "limit": 1},
            timeout=10
        )
        eth_data = eth_resp.json()
        eth_rate = float(eth_data[0]["fundingRate"]) if eth_data else 0

        return {
            "btc_funding": btc_rate,
            "eth_funding": eth_rate,
        }
    except Exception as e:
        print(f"[funding] Binance error: {e}")
        return None


def fetch_bybit_funding() -> Optional[Dict[str, float]]:
    """Fetch funding rates from Bybit as backup."""
    try:
        resp = requests.get(
            "https://api.bybit.com/v5/market/tickers",
            params={"category": "linear", "symbol": "BTCUSDT"},
            timeout=10
        )
        data = resp.json()
        if data.get("result", {}).get("list"):
            ticker = data["result"]["list"][0]
            return {
                "btc_funding": float(ticker.get("fundingRate", 0)),
                "eth_funding": 0,  # Would need separate call
            }
        return None
    except Exception as e:
        print(f"[funding] Bybit error: {e}")
        return None


def fetch_long_short_ratio() -> Optional[float]:
    """Fetch BTC long/short ratio from Binance."""
    try:
        resp = requests.get(
            "https://fapi.binance.com/futures/data/globalLongShortAccountRatio",
            params={"symbol": "BTCUSDT", "period": "5m", "limit": 1},
            timeout=10
        )
        data = resp.json()
        if data:
            return float(data[0]["longShortRatio"])
        return None
    except Exception as e:
        print(f"[funding] Long/short ratio error: {e}")
        return None


def get_funding_data() -> Optional[FundingData]:
    """Get comprehensive funding data."""
    # Try Binance first, then Bybit
    funding = fetch_binance_funding()
    if not funding:
        funding = fetch_bybit_funding()

    if not funding:
        return None

    btc_rate = funding["btc_funding"]
    eth_rate = funding.get("eth_funding", 0)

    # Annualize (3 funding periods per day * 365)
    annualized = btc_rate * 3 * 365 * 100  # As percentage

    # Determine sentiment
    # High positive funding = crowded longs (contrarian bearish)
    # High negative funding = crowded shorts (contrarian bullish)
    if btc_rate > 0.0005:  # >0.05% = high positive
        sentiment = "bearish"  # Contrarian
    elif btc_rate < -0.0005:  # <-0.05% = high negative
        sentiment = "bullish"  # Contrarian
    else:
        sentiment = "neutral"

    # Get long/short ratio
    ls_ratio = fetch_long_short_ratio()

    return FundingData(
        btc_funding_rate=btc_rate,
        btc_funding_annualized=annualized,
        eth_funding_rate=eth_rate,
        funding_sentiment=sentiment,
        long_short_ratio=ls_ratio,
    )


def compute_funding_features(funding: Optional[FundingData]) -> Dict[str, float]:
    """Compute features from funding data."""
    if not funding:
        return {
            "funding_rate": 0,
            "funding_annualized": 0,
            "funding_extreme": 0,
            "funding_bullish": 0,
            "funding_bearish": 0,
            "long_short_ratio": 1.0,
            "ls_ratio_extreme": 0,
        }

    # Normalize funding rate (-1 to 1 scale, clipped)
    normalized_funding = max(-1, min(1, funding.btc_funding_rate * 1000))

    # Extreme funding indicator
    is_extreme = 1 if abs(funding.btc_funding_rate) > 0.0005 else 0

    # Sentiment indicators
    is_bullish = 1 if funding.funding_sentiment == "bullish" else 0
    is_bearish = 1 if funding.funding_sentiment == "bearish" else 0

    # Long/short ratio features
    ls_ratio = funding.long_short_ratio or 1.0
    ls_extreme = 1 if ls_ratio > 1.5 or ls_ratio < 0.67 else 0

    return {
        "funding_rate": normalized_funding,
        "funding_annualized": funding.btc_funding_annualized,
        "funding_extreme": is_extreme,
        "funding_bullish": is_bullish,
        "funding_bearish": is_bearish,
        "long_short_ratio": ls_ratio,
        "ls_ratio_extreme": ls_extreme,
    }
