#!/usr/bin/env python3
"""Dump Kalshi series metadata to JSON for inspection."""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from typing import Any, Dict, List

from kalshi_inspect import KalshiClient, paginate


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Dump Kalshi series metadata to JSON.")
    parser.add_argument("--category", default="", help="Optional Kalshi category filter (e.g. Weather, Sports)")
    parser.add_argument("--status", default="", help="Optional series status filter (e.g. active, open)")
    parser.add_argument("--limit", type=int, default=2000, help="Max number of series to fetch")
    parser.add_argument("--out-json", required=True, help="Path to write JSON output")
    return parser.parse_args()


def ensure_parent_dir(path: str) -> None:
    parent = os.path.dirname(os.path.abspath(path))
    if parent:
        os.makedirs(parent, exist_ok=True)


def raise_for_api_errors(pages: List[Dict[str, Any]], last_response: Dict[str, Any]) -> None:
    for page in pages:
        status = page.get("status", 0)
        if isinstance(status, int) and status >= 400:
            raise RuntimeError(
                "Kalshi /series request failed. "
                f"status={status} params={page.get('params')} response={last_response}"
            )


def main() -> int:
    args = parse_args()

    limit = max(0, int(args.limit))
    category = args.category.strip()
    status = args.status.strip()

    params: Dict[str, Any] = {"limit": min(limit, 1000)}
    if category:
        params["category"] = category
    if status:
        params["status"] = status

    client = KalshiClient.from_env()
    series, meta, last = paginate(client, "/series", "series", params, hard_limit=limit)

    pages = meta.get("pages", []) if isinstance(meta, dict) else []
    last_response = last[0] if isinstance(last, list) and last else {}
    raise_for_api_errors(pages, last_response)

    payload = {
        "meta": {
            "fetched_at_utc": datetime.now(timezone.utc).isoformat(),
            "category": category,
            "status": status,
            "limit": limit,
            "series_count": len(series),
        },
        "series": series,
    }

    ensure_parent_dir(args.out_json)
    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    print("[kalshi] series dump complete")
    print(f"category={category}")
    print(f"count={len(series)}")
    print(f"output={args.out_json}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
