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


def fetch_okx_funding() -> Optional[Dict[str, float]]:
    """Fetch funding rates from OKX (US-accessible)."""
    try:
        # BTC perpetual funding
        btc_resp = requests.get(
            "https://www.okx.com/api/v5/public/funding-rate",
            params={"instId": "BTC-USDT-SWAP"},
            timeout=10
        )
        btc_data = btc_resp.json()

        # ETH perpetual funding
        eth_resp = requests.get(
            "https://www.okx.com/api/v5/public/funding-rate",
            params={"instId": "ETH-USDT-SWAP"},
            timeout=10
        )
        eth_data = eth_resp.json()

        btc_rate = 0
        eth_rate = 0

        if btc_data.get("code") == "0" and btc_data.get("data"):
            btc_rate = float(btc_data["data"][0].get("fundingRate", 0))
        if eth_data.get("code") == "0" and eth_data.get("data"):
            eth_rate = float(eth_data["data"][0].get("fundingRate", 0))

        return {"btc_funding": btc_rate, "eth_funding": eth_rate}
    except Exception as e:
        print(f"[funding] OKX error: {e}")
        return None


def fetch_kraken_price() -> Optional[Dict[str, float]]:
    """Fetch BTC/ETH prices from Kraken as backup data source."""
    try:
        resp = requests.get(
            "https://api.kraken.com/0/public/Ticker",
            params={"pair": "XXBTZUSD,XETHZUSD"},
            timeout=10
        )
        data = resp.json()

        if data.get("error"):
            return None

        result = data.get("result", {})
        btc_price = float(result.get("XXBTZUSD", {}).get("c", [0])[0])
        eth_price = float(result.get("XETHZUSD", {}).get("c", [0])[0])

        return {"btc_price": btc_price, "eth_price": eth_price}
    except Exception as e:
        print(f"[funding] Kraken error: {e}")
        return None


def fetch_okx_long_short_ratio() -> Optional[float]:
    """Fetch BTC long/short ratio from OKX."""
    try:
        resp = requests.get(
            "https://www.okx.com/api/v5/rubik/stat/contracts/long-short-account-ratio",
            params={"instId": "BTC", "period": "5m"},
            timeout=10
        )
        data = resp.json()
        if data.get("code") == "0" and data.get("data"):
            # OKX returns timestamp, longShortRatio pairs
            return float(data["data"][0][1])
        return None
    except Exception as e:
        # This endpoint may require auth, fallback to default
        return None


def get_funding_data() -> Optional[FundingData]:
    """Get comprehensive funding data."""
    # Use OKX (US-accessible)
    funding = fetch_okx_funding()

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
    ls_ratio = fetch_okx_long_short_ratio()

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
