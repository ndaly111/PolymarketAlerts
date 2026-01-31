"""Cross-asset correlation data for BTC prediction."""

from __future__ import annotations

from typing import List, Optional
import requests


def fetch_eth_price() -> Optional[float]:
    """Fetch current ETH price."""
    try:
        resp = requests.get(
            "https://api.binance.us/api/v3/ticker/price",
            params={"symbol": "ETHUSDT"},
            timeout=10
        )
        resp.raise_for_status()
        return float(resp.json()["price"])
    except Exception:
        pass
    return None


def fetch_eth_btc_ratio() -> Optional[dict]:
    """Fetch ETH/BTC ratio and recent trend.

    Rising ETH/BTC = risk-on, alt season
    Falling ETH/BTC = BTC dominance, flight to safety
    """
    try:
        resp = requests.get(
            "https://api.binance.us/api/v3/ticker/24hr",
            params={"symbol": "ETHBTC"},
            timeout=10
        )
        resp.raise_for_status()
        data = resp.json()

        return {
            "eth_btc_ratio": float(data["lastPrice"]),
            "eth_btc_change_24h": float(data["priceChangePercent"]),
            "eth_btc_high_24h": float(data["highPrice"]),
            "eth_btc_low_24h": float(data["lowPrice"]),
        }
    except Exception:
        pass
    return None


def fetch_btc_dominance() -> Optional[dict]:
    """Estimate BTC dominance from top coins.

    High dominance = BTC strength
    Falling dominance = alt season
    """
    try:
        # Get BTC and total market data
        btc_resp = requests.get(
            "https://api.coingecko.com/api/v3/coins/bitcoin",
            params={"localization": "false", "tickers": "false",
                    "community_data": "false", "developer_data": "false"},
            timeout=15
        )
        btc_resp.raise_for_status()
        btc_data = btc_resp.json()

        # Get global market data
        global_resp = requests.get(
            "https://api.coingecko.com/api/v3/global",
            timeout=15
        )
        global_resp.raise_for_status()
        global_data = global_resp.json()

        btc_market_cap = btc_data["market_data"]["market_cap"]["usd"]
        total_market_cap = global_data["data"]["total_market_cap"]["usd"]
        dominance = (btc_market_cap / total_market_cap) * 100

        return {
            "btc_dominance": dominance,
            "btc_market_cap": btc_market_cap,
            "total_market_cap": total_market_cap,
            "dominance_change_24h": global_data["data"].get(
                "market_cap_change_percentage_24h_usd", 0
            ),
        }
    except Exception:
        pass
    return None


def fetch_dxy_proxy() -> Optional[dict]:
    """Fetch USD strength proxy using stablecoins.

    Note: Real DXY requires paid data feed.
    We use USDT premium as a proxy for USD demand.
    """
    try:
        # USDT/USDC rate as proxy for stablecoin demand
        # When USDT trades above 1.00, people want USD = risk-off
        resp = requests.get(
            "https://api.binance.us/api/v3/ticker/price",
            params={"symbol": "USDCUSDT"},
            timeout=10
        )
        resp.raise_for_status()
        rate = float(resp.json()["price"])

        return {
            "usdc_usdt_rate": rate,
            "usd_premium": (rate - 1.0) * 100,  # % deviation from 1.0
            "usd_sentiment": "strong" if rate > 1.001 else (
                "weak" if rate < 0.999 else "neutral"
            ),
        }
    except Exception:
        pass
    return None


def fetch_gold_proxy() -> Optional[dict]:
    """Fetch gold proxy using PAXG (gold-backed token).

    BTC and gold often correlate as inflation hedges.
    """
    try:
        resp = requests.get(
            "https://api.binance.com/api/v3/ticker/24hr",
            params={"symbol": "PAXGUSDT"},
            timeout=10
        )
        resp.raise_for_status()
        data = resp.json()

        return {
            "gold_price": float(data["lastPrice"]),
            "gold_change_24h": float(data["priceChangePercent"]),
        }
    except Exception:
        pass
    return None


def compute_correlation(
    btc_returns: List[float],
    other_returns: List[float],
) -> Optional[float]:
    """Compute Pearson correlation between two return series."""
    if len(btc_returns) != len(other_returns) or len(btc_returns) < 10:
        return None

    n = len(btc_returns)
    mean_btc = sum(btc_returns) / n
    mean_other = sum(other_returns) / n

    cov = sum((btc_returns[i] - mean_btc) * (other_returns[i] - mean_other)
              for i in range(n)) / n

    var_btc = sum((x - mean_btc) ** 2 for x in btc_returns) / n
    var_other = sum((x - mean_other) ** 2 for x in other_returns) / n

    if var_btc == 0 or var_other == 0:
        return None

    return cov / (var_btc ** 0.5 * var_other ** 0.5)


def fetch_multi_asset_klines(interval: str = "5m", limit: int = 30) -> Optional[dict]:
    """Fetch klines for BTC, ETH, and compute correlations."""
    try:
        # Fetch BTC
        btc_resp = requests.get(
            "https://api.binance.us/api/v3/klines",
            params={"symbol": "BTCUSDT", "interval": interval, "limit": limit},
            timeout=10
        )
        btc_resp.raise_for_status()
        btc_data = btc_resp.json()

        # Fetch ETH
        eth_resp = requests.get(
            "https://api.binance.us/api/v3/klines",
            params={"symbol": "ETHUSDT", "interval": interval, "limit": limit},
            timeout=10
        )
        eth_resp.raise_for_status()
        eth_data = eth_resp.json()

        # Compute returns
        btc_closes = [float(k[4]) for k in btc_data]
        eth_closes = [float(k[4]) for k in eth_data]

        btc_returns = [(btc_closes[i] - btc_closes[i-1]) / btc_closes[i-1]
                       for i in range(1, len(btc_closes))]
        eth_returns = [(eth_closes[i] - eth_closes[i-1]) / eth_closes[i-1]
                       for i in range(1, len(eth_closes))]

        corr = compute_correlation(btc_returns, eth_returns)

        # ETH relative strength
        btc_change = (btc_closes[-1] - btc_closes[0]) / btc_closes[0] * 100
        eth_change = (eth_closes[-1] - eth_closes[0]) / eth_closes[0] * 100
        relative_strength = eth_change - btc_change

        return {
            "btc_eth_corr": corr,
            "btc_change_pct": btc_change,
            "eth_change_pct": eth_change,
            "eth_relative_strength": relative_strength,
        }
    except Exception:
        pass
    return None
