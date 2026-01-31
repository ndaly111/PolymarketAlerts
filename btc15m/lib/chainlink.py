"""Chainlink oracle price fetcher for BTC/USD on Polygon."""

from __future__ import annotations

import json
from typing import Optional, List
import requests

# Chainlink BTC/USD Price Feed on Polygon
BTC_USD_AGGREGATOR = "0xc907E116054Ad103354f2D350FD2514433D57F6f"

# Default RPC endpoints for Polygon
DEFAULT_RPC_URLS = [
    "https://polygon-rpc.com",
    "https://rpc-mainnet.matic.quiknode.pro",
    "https://polygon-mainnet.g.alchemy.com/v2/demo",
]

# ABI for Chainlink aggregator (only what we need)
LATEST_ROUND_DATA_SIG = "0xfeaf968c"  # latestRoundData()
DECIMALS_SIG = "0x313ce567"            # decimals()

# Cache decimals (constant per aggregator)
_decimals_cache: dict[str, int] = {}


def _eth_call(rpc_url: str, to: str, data: str, timeout: float = 5.0) -> str:
    """Make an eth_call JSON-RPC request."""
    payload = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "eth_call",
        "params": [{"to": to, "data": data}, "latest"],
    }

    resp = requests.post(rpc_url, json=payload, timeout=timeout)
    resp.raise_for_status()
    result = resp.json()

    if "error" in result:
        raise RuntimeError(f"RPC error: {result['error']}")

    return result["result"]


def _fetch_decimals(rpc_url: str, aggregator: str = BTC_USD_AGGREGATOR) -> int:
    """Fetch decimals for the price feed (usually 8 for BTC/USD)."""
    if aggregator in _decimals_cache:
        return _decimals_cache[aggregator]

    result = _eth_call(rpc_url, aggregator, DECIMALS_SIG)
    decimals = int(result, 16)
    _decimals_cache[aggregator] = decimals
    return decimals


def fetch_chainlink_price(
    rpc_urls: Optional[List[str]] = None,
    aggregator: str = BTC_USD_AGGREGATOR,
) -> Optional[dict]:
    """Fetch current BTC/USD price from Chainlink oracle.

    Tries multiple RPC endpoints in order until one succeeds.

    Returns:
        dict with 'price', 'round_id', 'updated_at', 'answered_in_round'
        or None if all RPCs fail
    """
    rpc_urls = rpc_urls or DEFAULT_RPC_URLS

    for rpc_url in rpc_urls:
        try:
            # Get decimals first
            decimals = _fetch_decimals(rpc_url, aggregator)

            # Call latestRoundData()
            result = _eth_call(rpc_url, aggregator, LATEST_ROUND_DATA_SIG)

            # Decode response (5 uint256 values, each 32 bytes = 64 hex chars)
            # Returns: (roundId, answer, startedAt, updatedAt, answeredInRound)
            hex_data = result[2:]  # Strip '0x'

            round_id = int(hex_data[0:64], 16)
            answer = int(hex_data[64:128], 16)
            # startedAt = int(hex_data[128:192], 16)  # Not needed
            updated_at = int(hex_data[192:256], 16)
            answered_in_round = int(hex_data[256:320], 16)

            # Convert to USD price
            price = answer / (10 ** decimals)

            return {
                "price": price,
                "round_id": round_id,
                "updated_at": updated_at,
                "answered_in_round": answered_in_round,
                "rpc_used": rpc_url,
            }

        except Exception as e:
            # Try next RPC
            continue

    return None


def get_chainlink_btc_price() -> Optional[float]:
    """Simple helper to get just the BTC price."""
    result = fetch_chainlink_price()
    return result["price"] if result else None
