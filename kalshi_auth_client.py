#!/usr/bin/env python3
"""
Kalshi authenticated API client with RSA-PSS signing.

Supports order placement, position queries, and account management.
"""

from __future__ import annotations

import base64
import json
import os
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import requests
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding


KALSHI_API_BASE = "https://api.elections.kalshi.com/trade-api/v2"


class KalshiAuthClient:
    """Authenticated Kalshi API client with RSA-PSS signing."""

    def __init__(
        self,
        key_id: str,
        private_key_pem: str,
        base_url: str = KALSHI_API_BASE,
    ):
        self.key_id = key_id
        self.base_url = base_url.rstrip("/")
        self.private_key = serialization.load_pem_private_key(
            private_key_pem.encode("utf-8"),
            password=None,
        )
        self.session = requests.Session()
        self.session.headers.update({
            "Content-Type": "application/json",
            "Accept": "application/json",
        })

    @classmethod
    def from_env(cls) -> "KalshiAuthClient":
        """Create client from environment variables."""
        key_id = os.getenv("KALSHI_API_KEY_ID") or os.getenv("KALSHI_KEY_ID")
        private_key = os.getenv("KALSHI_PRIVATE_KEY")
        base_url = os.getenv("KALSHI_BASE", KALSHI_API_BASE)

        if not key_id or not private_key:
            raise RuntimeError(
                "Missing KALSHI_API_KEY_ID and/or KALSHI_PRIVATE_KEY env vars"
            )

        # Handle escaped newlines in env var
        private_key = private_key.replace("\\n", "\n")

        return cls(key_id=key_id, private_key_pem=private_key, base_url=base_url)

    def _sign_request(self, method: str, path: str, body: str = "") -> Dict[str, str]:
        """
        Sign a request using RSA-PSS.

        Kalshi signature format:
        - Timestamp: milliseconds since epoch
        - Message: timestamp + method + path + body
        - Sign with RSA-PSS SHA256
        """
        timestamp_ms = str(int(time.time() * 1000))
        message = f"{timestamp_ms}{method.upper()}{path}{body}"

        signature = self.private_key.sign(
            message.encode("utf-8"),
            padding.PSS(
                mgf=padding.MGF1(hashes.SHA256()),
                salt_length=padding.PSS.MAX_LENGTH,
            ),
            hashes.SHA256(),
        )

        sig_b64 = base64.b64encode(signature).decode("utf-8")

        return {
            "KALSHI-ACCESS-KEY": self.key_id,
            "KALSHI-ACCESS-SIGNATURE": sig_b64,
            "KALSHI-ACCESS-TIMESTAMP": timestamp_ms,
        }

    def _request(
        self,
        method: str,
        path: str,
        params: Optional[Dict[str, Any]] = None,
        json_body: Optional[Dict[str, Any]] = None,
    ) -> Tuple[int, Dict[str, Any]]:
        """Make an authenticated request."""
        url = f"{self.base_url}{path}"
        body_str = json.dumps(json_body) if json_body else ""

        # For GET requests with params, path includes query string for signing
        if params and method.upper() == "GET":
            from urllib.parse import urlencode
            query = urlencode(params)
            sign_path = f"{path}?{query}"
        else:
            sign_path = path

        headers = self._sign_request(method.upper(), sign_path, body_str)

        try:
            if method.upper() == "GET":
                resp = self.session.get(url, params=params, headers=headers, timeout=30)
            elif method.upper() == "POST":
                resp = self.session.post(url, data=body_str, headers=headers, timeout=30)
            elif method.upper() == "DELETE":
                resp = self.session.delete(url, headers=headers, timeout=30)
            else:
                raise ValueError(f"Unsupported method: {method}")

            try:
                data = resp.json()
            except Exception:
                data = {"_raw": resp.text}

            return resp.status_code, data

        except requests.RequestException as e:
            return 0, {"_error": str(e)}

    # --- Account & Portfolio ---

    def get_balance(self) -> Dict[str, Any]:
        """Get account balance."""
        status, data = self._request("GET", "/portfolio/balance")
        if status != 200:
            raise RuntimeError(f"Failed to get balance: {status} {data}")
        return data

    def get_positions(self, ticker: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get current positions, optionally filtered by ticker."""
        params = {}
        if ticker:
            params["ticker"] = ticker
        status, data = self._request("GET", "/portfolio/positions", params=params)
        if status != 200:
            raise RuntimeError(f"Failed to get positions: {status} {data}")
        return data.get("market_positions", [])

    def has_position(self, ticker: str) -> bool:
        """Check if we have any position in a market."""
        positions = self.get_positions(ticker=ticker)
        for pos in positions:
            if pos.get("ticker") == ticker:
                qty = pos.get("position", 0)
                if qty != 0:
                    return True
        return False

    # --- Orders ---

    def get_orders(
        self,
        ticker: Optional[str] = None,
        status: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Get orders, optionally filtered."""
        params = {}
        if ticker:
            params["ticker"] = ticker
        if status:
            params["status"] = status
        status_code, data = self._request("GET", "/portfolio/orders", params=params)
        if status_code != 200:
            raise RuntimeError(f"Failed to get orders: {status_code} {data}")
        return data.get("orders", [])

    def place_order(
        self,
        ticker: str,
        side: str,  # "yes" or "no"
        quantity: int,
        limit_price: int,  # Price in cents (1-99)
        client_order_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Place a limit order.

        Args:
            ticker: Market ticker (e.g., "KXNBAPTS-25JAN26-LEBRON-25")
            side: "yes" or "no"
            quantity: Number of contracts
            limit_price: Limit price in cents (1-99)
            client_order_id: Optional client-provided order ID

        Returns:
            Order response dict with order_id, status, etc.
        """
        if side.lower() not in ("yes", "no"):
            raise ValueError(f"Side must be 'yes' or 'no', got: {side}")
        if not 1 <= limit_price <= 99:
            raise ValueError(f"Limit price must be 1-99 cents, got: {limit_price}")
        if quantity < 1:
            raise ValueError(f"Quantity must be >= 1, got: {quantity}")

        body = {
            "ticker": ticker,
            "action": "buy",
            "side": side.lower(),
            "type": "limit",
            "count": quantity,
            "yes_price" if side.lower() == "yes" else "no_price": limit_price,
        }

        if client_order_id:
            body["client_order_id"] = client_order_id

        status, data = self._request("POST", "/portfolio/orders", json_body=body)

        if status not in (200, 201):
            raise RuntimeError(f"Failed to place order: {status} {data}")

        return data.get("order", data)

    def cancel_order(self, order_id: str) -> bool:
        """Cancel an order by ID."""
        status, data = self._request("DELETE", f"/portfolio/orders/{order_id}")
        return status in (200, 204)

    def get_order(self, order_id: str) -> Dict[str, Any]:
        """Get order details by ID."""
        status, data = self._request("GET", f"/portfolio/orders/{order_id}")
        if status != 200:
            raise RuntimeError(f"Failed to get order: {status} {data}")
        return data.get("order", data)

    # --- Markets ---

    def get_market(self, ticker: str) -> Dict[str, Any]:
        """Get market details including current orderbook."""
        status, data = self._request("GET", f"/markets/{ticker}")
        if status != 200:
            raise RuntimeError(f"Failed to get market: {status} {data}")
        return data.get("market", data)

    def get_orderbook(self, ticker: str) -> Dict[str, Any]:
        """Get orderbook for a market."""
        status, data = self._request("GET", f"/markets/{ticker}/orderbook")
        if status != 200:
            raise RuntimeError(f"Failed to get orderbook: {status} {data}")
        return data.get("orderbook", data)


def prob_to_american(prob: float) -> int:
    """Convert probability to American odds."""
    if prob <= 0 or prob >= 1:
        raise ValueError(f"Probability must be between 0 and 1, got: {prob}")
    if prob >= 0.5:
        return int(-100 * prob / (1 - prob))
    else:
        return int(100 * (1 - prob) / prob)


def american_to_prob(odds: int) -> float:
    """Convert American odds to probability."""
    if odds < 0:
        return (-odds) / ((-odds) + 100.0)
    else:
        return 100.0 / (odds + 100.0)


if __name__ == "__main__":
    # Quick test
    try:
        client = KalshiAuthClient.from_env()
        print("Kalshi auth client initialized successfully")

        balance = client.get_balance()
        print(f"Balance: ${balance.get('balance', 0) / 100:.2f}")

        positions = client.get_positions()
        print(f"Positions: {len(positions)}")

    except Exception as e:
        print(f"Error: {e}")
