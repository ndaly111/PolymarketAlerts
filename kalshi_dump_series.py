#!/usr/bin/env python3
"""Dump Kalshi series metadata to JSON for inspection."""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

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


def api_error_message(pages: List[Dict[str, Any]], last_response: Optional[Dict[str, Any]]) -> Optional[str]:
    for page in pages:
        status = page.get("status", 0)
        if isinstance(status, int) and status >= 400:
            snippet = ""
            if isinstance(last_response, dict):
                snippet = json.dumps(last_response, ensure_ascii=False)[:800]
            return (
                "Kalshi /series request failed. "
                f"status={status} params={page.get('params')} response_snippet={snippet}"
            )
    return None


def missing_kalshi_env() -> List[str]:
    missing: List[str] = []
    if not os.getenv("KALSHI_PRIVATE_KEY", "").strip():
        missing.append("KALSHI_PRIVATE_KEY")
    if not (os.getenv("KALSHI_API_KEY_ID", "").strip() or os.getenv("KALSHI_KEY_ID", "").strip()):
        missing.append("KALSHI_API_KEY_ID (or KALSHI_KEY_ID)")
    return missing


def main() -> int:
    args = parse_args()

    if not args.out_json or not args.out_json.strip():
        print("Missing required --out-json path.")
        return 2

    limit = max(0, int(args.limit))
    category = args.category.strip()
    status = args.status.strip()

    params: Dict[str, Any] = {"limit": min(limit, 1000)}
    if category:
        params["category"] = category
    if status:
        params["status"] = status

    missing = missing_kalshi_env()
    if missing:
        payload = {
            "meta": {
                "endpoint": "/series",
                "fetched_at_utc": datetime.now(timezone.utc).isoformat(),
                "category": category or None,
                "status": status or None,
                "limit": limit,
                "count": 0,
                "skipped": True,
                "skip_reason": f"Missing env vars: {', '.join(missing)}",
            },
            "items": [],
        }
        ensure_parent_dir(args.out_json)
        with open(args.out_json, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)

        print("[kalshi] series dump skipped (missing credentials)")
        print(f"missing={', '.join(missing)}")
        print(f"output={args.out_json}")
        return 0

    client = KalshiClient.from_env()
    series, meta, last = paginate(client, "/series", "series", params, hard_limit=limit)

    pages = meta.get("pages", []) if isinstance(meta, dict) else []
    last_response = last[0] if isinstance(last, list) and last else {}
    error = api_error_message(pages, last_response if isinstance(last_response, dict) else None)
    if error:
        print(error)
        return 2

    payload = {
        "meta": {
            "endpoint": "/series",
            "fetched_at_utc": datetime.now(timezone.utc).isoformat(),
            "category": category or None,
            "status": status or None,
            "limit": limit,
            "count": len(series),
        },
        "items": series,
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
