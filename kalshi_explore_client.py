#!/usr/bin/env python3
"""Minimal Kalshi API client and paginator used by kalshi_explore.py."""

from __future__ import annotations

import json
import os
import time
import urllib.parse
import urllib.request
from typing import Any, Dict, List, Optional, Tuple


def _coerce_param(value: Any) -> Any:
    if isinstance(value, bool):
        return "true" if value else "false"
    return value


def _encode_params(params: Optional[Dict[str, Any]]) -> str:
    if not params:
        return ""
    clean = {k: _coerce_param(v) for k, v in params.items() if v is not None}
    return urllib.parse.urlencode(clean, doseq=True)


class KalshiClient:
    def __init__(
        self,
        base: str,
        key_id: Optional[str] = None,
        private_key: Optional[str] = None,
        auth_enabled: bool = False,
    ) -> None:
        self.base = base.rstrip("/")
        self.key_id = key_id
        self.private_key = private_key
        self.auth_enabled = auth_enabled

    @classmethod
    def from_env(cls, auth_mode: str = "public") -> "KalshiClient":
        base = os.getenv("KALSHI_BASE", "https://api.elections.kalshi.com/trade-api/v2")
        key_id = os.getenv("KALSHI_API_KEY_ID") or os.getenv("KALSHI_KEY_ID")
        private_key = os.getenv("KALSHI_PRIVATE_KEY")
        auth_enabled = auth_mode == "signed"
        return cls(base=base, key_id=key_id, private_key=private_key, auth_enabled=auth_enabled)

    def get(self, path: str, params: Optional[Dict[str, Any]] = None) -> Tuple[int, Any]:
        query = _encode_params(params)
        url = f"{self.base}/{path.lstrip('/')}"
        if query:
            url = f"{url}?{query}"
        headers = {"User-Agent": "kalshi-explore/1.0"}
        if self.auth_enabled:
            raise NotImplementedError(
                "Signed auth not implemented in this minimal client. "
                "Kalshi requires RSA-PSS with millisecond timestamps. "
                "Use public mode or an official SDK."
            )
        req = urllib.request.Request(url, headers=headers)
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                status = resp.status
                raw = resp.read().decode("utf-8")
        except urllib.error.HTTPError as exc:
            status = exc.code
            raw = exc.read().decode("utf-8")
        except Exception as exc:
            return 0, {"_error": f"{type(exc).__name__}: {exc}"}
        try:
            data = json.loads(raw) if raw else {}
        except Exception:
            data = {"_raw": raw}
        return status, data


def paginate(
    client: KalshiClient,
    path: str,
    list_key: str,
    params: Optional[Dict[str, Any]] = None,
    hard_limit: int = 200,
    sleep_s: float = 0.1,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any], List[str]]:
    items: List[Dict[str, Any]] = []
    errors: List[str] = []
    meta: Dict[str, Any] = {}
    cursor: Optional[str] = None
    params = dict(params or {})

    while len(items) < hard_limit:
        if cursor:
            params["cursor"] = cursor
        params["limit"] = min(params.get("limit", 200), 200)
        status, data = client.get(path, params=params)
        if status != 200:
            errors.append(f"status={status}")
            meta = {"status": status, "response": data}
            break
        page_items = data.get(list_key) or data.get("data") or []
        if isinstance(page_items, list):
            items.extend(page_items)
        cursor = data.get("cursor") or data.get("next_cursor")
        meta = {"cursor": cursor}
        if not cursor or not page_items:
            break
        time.sleep(sleep_s)

    return items[:hard_limit], meta, errors
