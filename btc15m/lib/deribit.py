"""Deribit options data for BTC sentiment analysis."""

from __future__ import annotations

import requests
from typing import Optional, Dict, List
from dataclasses import dataclass
from datetime import datetime, timezone


DERIBIT_API = "https://www.deribit.com/api/v2/public"


@dataclass
class DeribitOptionsData:
    """Deribit BTC options market data."""
    btc_index_price: float
    btc_iv: float  # Current implied volatility
    put_call_ratio: float  # OI-weighted
    total_call_oi: float
    total_put_oi: float
    max_pain: Optional[float]  # Strike with max pain
    iv_skew: float  # Call IV - Put IV (positive = calls more expensive)


def fetch_btc_index() -> Optional[float]:
    """Fetch BTC index price from Deribit."""
    try:
        resp = requests.get(
            f"{DERIBIT_API}/get_index_price",
            params={"index_name": "btc_usd"},
            timeout=10
        )
        data = resp.json()
        if data.get("result"):
            return data["result"]["index_price"]
        return None
    except Exception as e:
        print(f"[deribit] Index error: {e}")
        return None


def fetch_btc_volatility() -> Optional[Dict[str, float]]:
    """Fetch BTC volatility index from Deribit."""
    try:
        resp = requests.get(
            f"{DERIBIT_API}/get_volatility_index_data",
            params={"currency": "BTC", "resolution": "60"},  # 1 hour
            timeout=10
        )
        data = resp.json()
        if data.get("result", {}).get("data"):
            latest = data["result"]["data"][-1]
            return {
                "iv": latest[4],  # Close IV
                "iv_high": latest[2],
                "iv_low": latest[3],
            }
        return None
    except Exception as e:
        print(f"[deribit] Volatility error: {e}")
        return None


def fetch_options_chain() -> Optional[List[Dict]]:
    """Fetch BTC options instruments."""
    try:
        resp = requests.get(
            f"{DERIBIT_API}/get_instruments",
            params={"currency": "BTC", "kind": "option", "expired": "false"},
            timeout=15
        )
        data = resp.json()
        return data.get("result", [])
    except Exception as e:
        print(f"[deribit] Instruments error: {e}")
        return None


def fetch_option_stats(instrument_name: str) -> Optional[Dict]:
    """Fetch stats for a specific option."""
    try:
        resp = requests.get(
            f"{DERIBIT_API}/ticker",
            params={"instrument_name": instrument_name},
            timeout=10
        )
        data = resp.json()
        return data.get("result")
    except Exception as e:
        return None


def calculate_put_call_ratio(instruments: List[Dict]) -> Dict[str, float]:
    """Calculate put/call ratio from instruments."""
    if not instruments:
        return {"put_call_ratio": 1.0, "total_call_oi": 0, "total_put_oi": 0}

    call_oi = 0
    put_oi = 0

    for inst in instruments:
        oi = inst.get("open_interest", 0) or 0
        if inst.get("option_type") == "call":
            call_oi += oi
        elif inst.get("option_type") == "put":
            put_oi += oi

    ratio = put_oi / call_oi if call_oi > 0 else 1.0

    return {
        "put_call_ratio": ratio,
        "total_call_oi": call_oi,
        "total_put_oi": put_oi,
    }


def get_deribit_options_data() -> Optional[DeribitOptionsData]:
    """Get comprehensive Deribit options data."""
    # Fetch index price
    index_price = fetch_btc_index()
    if not index_price:
        return None

    # Fetch volatility
    vol_data = fetch_btc_volatility()
    iv = vol_data["iv"] if vol_data else 50

    # Fetch options chain for put/call ratio
    instruments = fetch_options_chain()
    pc_data = calculate_put_call_ratio(instruments) if instruments else {
        "put_call_ratio": 1.0, "total_call_oi": 0, "total_put_oi": 0
    }

    return DeribitOptionsData(
        btc_index_price=index_price,
        btc_iv=iv,
        put_call_ratio=pc_data["put_call_ratio"],
        total_call_oi=pc_data["total_call_oi"],
        total_put_oi=pc_data["total_put_oi"],
        max_pain=None,  # Would need more complex calculation
        iv_skew=0,  # Would need per-strike IV comparison
    )


def compute_deribit_features(data: Optional[DeribitOptionsData]) -> Dict[str, float]:
    """Compute features from Deribit options data."""
    if not data:
        return {
            "deribit_iv": 50,
            "deribit_iv_high": 0,
            "deribit_iv_low": 0,
            "deribit_pc_ratio": 1.0,
            "deribit_pc_bullish": 0,
            "deribit_pc_bearish": 0,
            "deribit_call_oi": 0,
            "deribit_put_oi": 0,
            "deribit_oi_imbalance": 0,
        }

    # Put/Call ratio interpretation
    # < 0.7 = bullish (more calls than puts)
    # > 1.3 = bearish (more puts than calls)
    # Can also be contrarian at extremes
    pc_bullish = 1 if data.put_call_ratio < 0.7 else 0
    pc_bearish = 1 if data.put_call_ratio > 1.3 else 0

    # OI imbalance (-1 to 1, positive = more calls)
    total_oi = data.total_call_oi + data.total_put_oi
    if total_oi > 0:
        oi_imbalance = (data.total_call_oi - data.total_put_oi) / total_oi
    else:
        oi_imbalance = 0

    # IV level (high IV = uncertainty, potential reversals)
    iv_high = 1 if data.btc_iv > 70 else 0
    iv_low = 1 if data.btc_iv < 40 else 0

    return {
        "deribit_iv": data.btc_iv,
        "deribit_iv_high": iv_high,
        "deribit_iv_low": iv_low,
        "deribit_pc_ratio": data.put_call_ratio,
        "deribit_pc_bullish": pc_bullish,
        "deribit_pc_bearish": pc_bearish,
        "deribit_call_oi": data.total_call_oi / 1000,  # Scale down
        "deribit_put_oi": data.total_put_oi / 1000,
        "deribit_oi_imbalance": oi_imbalance,
    }
