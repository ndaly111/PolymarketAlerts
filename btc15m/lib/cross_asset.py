"""Cross-asset correlation signals for BTC prediction."""

from __future__ import annotations

import requests
from typing import Optional, Dict, List
from dataclasses import dataclass


@dataclass
class CrossAssetData:
    """Cross-asset price data."""
    eth_price: float
    eth_change_15m: float
    eth_change_1h: float
    btc_eth_ratio: float
    eth_btc_divergence: float  # ETH outperforming or underperforming BTC
    sp500_change: Optional[float] = None
    dxy_change: Optional[float] = None


def fetch_eth_price() -> Optional[Dict[str, float]]:
    """Fetch ETH price and recent changes from Binance."""
    try:
        # Current price
        ticker_resp = requests.get(
            "https://api.binance.us/api/v3/ticker/price",
            params={"symbol": "ETHUSDT"},
            timeout=10
        )
        current_price = float(ticker_resp.json()["price"])

        # 1h klines for change calculation
        klines_resp = requests.get(
            "https://api.binance.us/api/v3/klines",
            params={"symbol": "ETHUSDT", "interval": "15m", "limit": 5},
            timeout=10
        )
        klines = klines_resp.json()

        if len(klines) >= 5:
            price_15m_ago = float(klines[-2][4])  # Close of 15m ago
            price_1h_ago = float(klines[0][4])  # Close of 1h ago
            change_15m = (current_price - price_15m_ago) / price_15m_ago
            change_1h = (current_price - price_1h_ago) / price_1h_ago
        else:
            change_15m = 0
            change_1h = 0

        return {
            "price": current_price,
            "change_15m": change_15m,
            "change_1h": change_1h,
        }
    except Exception as e:
        print(f"[cross_asset] ETH fetch error: {e}")
        return None


def fetch_btc_changes() -> Optional[Dict[str, float]]:
    """Fetch BTC recent changes for comparison."""
    try:
        klines_resp = requests.get(
            "https://api.binance.us/api/v3/klines",
            params={"symbol": "BTCUSDT", "interval": "15m", "limit": 5},
            timeout=10
        )
        klines = klines_resp.json()

        if len(klines) >= 5:
            current_price = float(klines[-1][4])
            price_15m_ago = float(klines[-2][4])
            price_1h_ago = float(klines[0][4])
            return {
                "price": current_price,
                "change_15m": (current_price - price_15m_ago) / price_15m_ago,
                "change_1h": (current_price - price_1h_ago) / price_1h_ago,
            }
        return None
    except Exception as e:
        print(f"[cross_asset] BTC fetch error: {e}")
        return None


def get_cross_asset_data() -> Optional[CrossAssetData]:
    """Get cross-asset correlation data."""
    eth_data = fetch_eth_price()
    btc_data = fetch_btc_changes()

    if not eth_data or not btc_data:
        return None

    # BTC/ETH ratio
    ratio = btc_data["price"] / eth_data["price"] if eth_data["price"] > 0 else 0

    # Divergence: positive = ETH outperforming, negative = ETH underperforming
    divergence_15m = eth_data["change_15m"] - btc_data["change_15m"]
    divergence_1h = eth_data["change_1h"] - btc_data["change_1h"]

    return CrossAssetData(
        eth_price=eth_data["price"],
        eth_change_15m=eth_data["change_15m"],
        eth_change_1h=eth_data["change_1h"],
        btc_eth_ratio=ratio,
        eth_btc_divergence=divergence_15m,
    )


def compute_cross_asset_features(data: Optional[CrossAssetData]) -> Dict[str, float]:
    """Compute features from cross-asset data."""
    if not data:
        return {
            "eth_change_15m": 0,
            "eth_change_1h": 0,
            "eth_momentum": 0,
            "eth_btc_divergence": 0,
            "eth_leading": 0,
            "eth_lagging": 0,
            "btc_eth_ratio": 25,
            "cross_asset_bullish": 0,
            "cross_asset_bearish": 0,
        }

    # ETH momentum (is ETH trending?)
    eth_momentum = 1 if abs(data.eth_change_1h) > 0.01 else 0

    # ETH leading indicator
    # If ETH moved significantly and BTC hasn't caught up, BTC may follow
    eth_leading = 1 if data.eth_btc_divergence > 0.005 else 0
    eth_lagging = 1 if data.eth_btc_divergence < -0.005 else 0

    # Cross-asset sentiment
    cross_bullish = 1 if data.eth_change_15m > 0.003 else 0
    cross_bearish = 1 if data.eth_change_15m < -0.003 else 0

    return {
        "eth_change_15m": data.eth_change_15m * 100,  # As percentage
        "eth_change_1h": data.eth_change_1h * 100,
        "eth_momentum": eth_momentum,
        "eth_btc_divergence": data.eth_btc_divergence * 100,
        "eth_leading": eth_leading,
        "eth_lagging": eth_lagging,
        "btc_eth_ratio": data.btc_eth_ratio,
        "cross_asset_bullish": cross_bullish,
        "cross_asset_bearish": cross_bearish,
    }
