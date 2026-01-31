"""Derivatives and sentiment data for BTC prediction."""

from __future__ import annotations

from typing import Optional
import requests


def fetch_long_short_ratio(symbol: str = "BTCUSDT") -> Optional[dict]:
    """Fetch global long/short ratio from Binance Futures.

    High long ratio (>1.5) = crowded long, potential squeeze down
    High short ratio (<0.7) = crowded short, potential squeeze up
    """
    url = "https://fapi.binance.com/futures/data/globalLongShortAccountRatio"
    params = {"symbol": symbol, "period": "5m", "limit": 1}

    try:
        resp = requests.get(url, params=params, timeout=10)
        resp.raise_for_status()
        data = resp.json()

        if data:
            ratio = float(data[0]["longShortRatio"])
            long_account = float(data[0]["longAccount"])
            short_account = float(data[0]["shortAccount"])

            return {
                "long_short_ratio": ratio,
                "long_pct": long_account * 100,
                "short_pct": short_account * 100,
                "sentiment": "crowded_long" if ratio > 1.5 else (
                    "crowded_short" if ratio < 0.7 else "neutral"
                ),
            }
    except Exception:
        pass

    return None


def fetch_top_trader_ratio(symbol: str = "BTCUSDT") -> Optional[dict]:
    """Fetch top trader long/short ratio (positions).

    Top traders often have better information.
    """
    url = "https://fapi.binance.com/futures/data/topLongShortPositionRatio"
    params = {"symbol": symbol, "period": "5m", "limit": 1}

    try:
        resp = requests.get(url, params=params, timeout=10)
        resp.raise_for_status()
        data = resp.json()

        if data:
            return {
                "top_trader_ratio": float(data[0]["longShortRatio"]),
                "top_long_pct": float(data[0]["longAccount"]) * 100,
                "top_short_pct": float(data[0]["shortAccount"]) * 100,
            }
    except Exception:
        pass

    return None


def fetch_taker_buy_sell_ratio(symbol: str = "BTCUSDT") -> Optional[dict]:
    """Fetch taker buy/sell volume ratio.

    Taker buy > sell = aggressive buying = bullish
    Taker sell > buy = aggressive selling = bearish
    """
    url = "https://fapi.binance.com/futures/data/takerlongshortRatio"
    params = {"symbol": symbol, "period": "5m", "limit": 1}

    try:
        resp = requests.get(url, params=params, timeout=10)
        resp.raise_for_status()
        data = resp.json()

        if data:
            buy_vol = float(data[0]["buyVol"])
            sell_vol = float(data[0]["sellVol"])
            ratio = float(data[0]["buySellRatio"])

            return {
                "taker_buy_sell_ratio": ratio,
                "taker_buy_vol": buy_vol,
                "taker_sell_vol": sell_vol,
                "taker_sentiment": "bullish" if ratio > 1.1 else (
                    "bearish" if ratio < 0.9 else "neutral"
                ),
            }
    except Exception:
        pass

    return None


def fetch_futures_premium() -> Optional[dict]:
    """Calculate futures premium (basis) vs spot.

    Positive premium = futures trading higher than spot = bullish
    Negative premium = backwardation = bearish
    """
    try:
        # Get spot price
        spot_resp = requests.get(
            "https://api.binance.us/api/v3/ticker/price",
            params={"symbol": "BTCUSDT"},
            timeout=10
        )
        spot_resp.raise_for_status()
        spot_price = float(spot_resp.json()["price"])

        # Get perpetual futures price
        futures_resp = requests.get(
            "https://fapi.binance.com/fapi/v1/ticker/price",
            params={"symbol": "BTCUSDT"},
            timeout=10
        )
        futures_resp.raise_for_status()
        futures_price = float(futures_resp.json()["price"])

        premium = futures_price - spot_price
        premium_pct = (premium / spot_price) * 100

        return {
            "spot_price": spot_price,
            "futures_price": futures_price,
            "premium": premium,
            "premium_pct": premium_pct,
            "basis_sentiment": "contango" if premium_pct > 0.01 else (
                "backwardation" if premium_pct < -0.01 else "neutral"
            ),
        }
    except Exception:
        pass

    return None


def fetch_fear_greed_index() -> Optional[dict]:
    """Fetch Crypto Fear & Greed Index.

    0-25 = Extreme Fear (buy signal)
    25-45 = Fear
    45-55 = Neutral
    55-75 = Greed
    75-100 = Extreme Greed (sell signal)
    """
    url = "https://api.alternative.me/fng/"
    params = {"limit": 1}

    try:
        resp = requests.get(url, params=params, timeout=10)
        resp.raise_for_status()
        data = resp.json()

        if data and "data" in data and data["data"]:
            value = int(data["data"][0]["value"])
            classification = data["data"][0]["value_classification"]

            return {
                "fear_greed_value": value,
                "fear_greed_class": classification,
                "contrarian_signal": "buy" if value < 25 else (
                    "sell" if value > 75 else "neutral"
                ),
            }
    except Exception:
        pass

    return None


def fetch_open_interest_change(symbol: str = "BTCUSDT") -> Optional[dict]:
    """Fetch open interest and calculate recent change.

    Rising OI + Rising Price = New longs entering = Bullish
    Rising OI + Falling Price = New shorts entering = Bearish
    Falling OI + Rising Price = Short covering = Weak bullish
    Falling OI + Falling Price = Long liquidation = Weak bearish
    """
    url = "https://fapi.binance.com/futures/data/openInterestHist"
    params = {"symbol": symbol, "period": "5m", "limit": 12}  # Last hour

    try:
        resp = requests.get(url, params=params, timeout=10)
        resp.raise_for_status()
        data = resp.json()

        if data and len(data) >= 2:
            current_oi = float(data[-1]["sumOpenInterest"])
            prev_oi = float(data[0]["sumOpenInterest"])

            oi_change = current_oi - prev_oi
            oi_change_pct = (oi_change / prev_oi) * 100 if prev_oi > 0 else 0

            return {
                "open_interest": current_oi,
                "oi_change_1h": oi_change,
                "oi_change_pct_1h": oi_change_pct,
                "oi_trend": "rising" if oi_change_pct > 1 else (
                    "falling" if oi_change_pct < -1 else "stable"
                ),
            }
    except Exception:
        pass

    return None


def fetch_liquidations() -> Optional[dict]:
    """Fetch recent liquidation data.

    Large long liquidations = downward pressure exhausted
    Large short liquidations = upward pressure exhausted
    """
    # Note: Binance doesn't provide historical liquidation API
    # This would need a third-party service like Coinglass
    # For now, return None - can be added later
    return None
