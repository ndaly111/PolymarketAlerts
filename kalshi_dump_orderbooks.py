#!/usr/bin/env python3
"""Dump Kalshi market orderbooks to JSON for inspection."""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Optional

from kalshi_inspect import KalshiClient


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Dump Kalshi market orderbooks to JSON.")
    parser.add_argument("--tickers-json", default="", help="Path to JSON array of market tickers")
    parser.add_argument(
        "--from-markets-dump",
        default="",
        help="Path to markets dump JSON (extract tickers from items)",
    )
    parser.add_argument("--limit", type=int, default=0, help="Optional max number of tickers to fetch")
    parser.add_argument("--out-json", required=True, help="Path to write JSON output")
    return parser.parse_args()


def ensure_parent_dir(path: str) -> None:
    parent = os.path.dirname(os.path.abspath(path))
    if parent:
        os.makedirs(parent, exist_ok=True)


def load_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def extract_tickers_from_items(items: Iterable[Dict[str, Any]]) -> List[str]:
    tickers: List[str] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        ticker = (
            str(item.get("ticker") or "")
            or str(item.get("market_ticker") or "")
            or str(item.get("marketTicker") or "")
        ).strip()
        if ticker:
            tickers.append(ticker)
    return tickers


def load_tickers(args: argparse.Namespace) -> List[str]:
    if args.tickers_json:
        data = load_json(args.tickers_json)
        if not isinstance(data, list):
            raise ValueError("tickers-json must contain a JSON array of tickers")
        return [str(t).strip() for t in data if str(t).strip()]

    if args.from_markets_dump:
        data = load_json(args.from_markets_dump)
        if isinstance(data, dict):
            items = data.get("items")
            if items is None:
                items = data.get("markets")
        else:
            items = None
        if not isinstance(items, list):
            raise ValueError("markets dump must contain an 'items' list")
        return extract_tickers_from_items(items)

    raise ValueError("Must provide --tickers-json or --from-markets-dump")


def main() -> int:
    args = parse_args()

    if not args.out_json or not args.out_json.strip():
        print("Missing required --out-json path.")
        return 2

    try:
        tickers = load_tickers(args)
    except Exception as exc:
        print(f"Error loading tickers: {exc}")
        return 2

    if not tickers:
        print("No tickers provided.")
        return 2

    limit = int(args.limit) if args.limit else 0
    if limit > 0:
        tickers = tickers[:limit]

    client = KalshiClient.from_env()
    results: List[Dict[str, Any]] = []
    success_count = 0

    for ticker in tickers:
        status, data = client.get(f"/markets/{ticker}/orderbook", params={})
        if status >= 400:
            results.append(
                {
                    "ticker": ticker,
                    "error": "request failed",
                    "status": status,
                    "response": data,
                }
            )
            continue

        results.append({"ticker": ticker, "orderbook": data})
        success_count += 1

    if success_count == 0:
        print("All orderbook requests failed.")
        return 2

    payload = {
        "meta": {
            "endpoint": "/markets/{ticker}/orderbook",
            "fetched_at_utc": datetime.now(timezone.utc).isoformat(),
            "input_tickers_count": len(tickers),
            "limit": limit or None,
            "count": len(results),
        },
        "items": results,
    }

    ensure_parent_dir(args.out_json)
    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    print("[kalshi] orderbooks dump complete")
    print(f"count={len(results)}")
    print(f"output={args.out_json}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
