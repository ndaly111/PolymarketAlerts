#!/usr/bin/env python3
"""Collect Kalshi weather markets and store an orderbook snapshot in weather_forecast_accuracy.db.

This script intentionally reuses your existing Kalshi auth/signing implementation:
  - kalshi_inspect.KalshiClient
  - kalshi_inspect.paginate

We snapshot quotes append-only so we can later:
  - join against our computed fair prices
  - backtest / audit model performance and execution realism

Env (already used elsewhere in repo):
  - KALSHI_KEY_ID or KALSHI_API_KEY_ID
  - KALSHI_PRIVATE_KEY
  - KALSHI_BASE (optional)

Series tickers:
  - WEATHER_KALSHI_SERIES_TICKERS (comma-separated) or --series-tickers
"""

from __future__ import annotations

import argparse
import json
import os
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
from zoneinfo import ZoneInfo

import yaml

from kalshi_inspect import KalshiClient, paginate
from weather.lib import db as db_lib


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = ROOT / "weather" / "config" / "cities.yml"
DEFAULT_DB = Path(os.getenv("WEATHER_DB_PATH", str(ROOT / "weather" / "data" / "weather_forecast_accuracy.db")))


@dataclass(frozen=True)
class City:
    key: str


def _now_utc_iso() -> str:
    return datetime.now(tz=ZoneInfo("UTC")).isoformat(timespec="seconds")


def load_city_keys(config_path: Path) -> List[City]:
    data = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    out: List[City] = []
    for row in data.get("cities", []):
        out.append(City(key=str(row["key"]).strip()))
    return out


_YYYYMMDD = re.compile(r"(20\d{2})(\d{2})(\d{2})")


def infer_target_date_local_from_ticker(text: str) -> str:
    """Best-effort: pull YYYYMMDD from a ticker-like string and convert to YYYY-MM-DD."""
    m = _YYYYMMDD.search(text or "")
    if not m:
        return datetime.now(tz=ZoneInfo("UTC")).date().isoformat()
    y, mo, d = m.group(1), m.group(2), m.group(3)
    return f"{y}-{mo}-{d}"


def _as_int(v: Any) -> Optional[int]:
    try:
        if v is None:
            return None
        return int(v)
    except Exception:
        return None


def _pick_first_str(*vals: Any) -> str:
    for v in vals:
        if isinstance(v, str) and v.strip():
            return v.strip()
    return ""


def extract_market_fields(m: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize a Kalshi 'market' object into a minimal quote snapshot shape."""
    market_ticker = _pick_first_str(m.get("ticker"), m.get("market_ticker"), m.get("marketTicker"))
    event_ticker = _pick_first_str(m.get("event_ticker"), m.get("eventTicker"))
    series_ticker = _pick_first_str(m.get("series_ticker"), m.get("seriesTicker"))
    status = _pick_first_str(m.get("status"), m.get("market_status"), m.get("marketStatus"))

    yes_bid = _as_int(m.get("yes_bid", m.get("yesBid")))
    yes_ask = _as_int(m.get("yes_ask", m.get("yesAsk")))
    no_bid = _as_int(m.get("no_bid", m.get("noBid")))
    no_ask = _as_int(m.get("no_ask", m.get("noAsk")))

    # Some endpoints provide nested orderbook keys
    ob = m.get("orderbook") if isinstance(m.get("orderbook"), dict) else {}
    if ob:
        yes_bid = yes_bid if yes_bid is not None else _as_int(ob.get("yes_bid"))
        yes_ask = yes_ask if yes_ask is not None else _as_int(ob.get("yes_ask"))
        no_bid = no_bid if no_bid is not None else _as_int(ob.get("no_bid"))
        no_ask = no_ask if no_ask is not None else _as_int(ob.get("no_ask"))

    volume = _as_int(m.get("volume"))
    open_interest = _as_int(m.get("open_interest", m.get("openInterest")))

    return {
        "series_ticker": series_ticker,
        "event_ticker": event_ticker,
        "market_ticker": market_ticker,
        "status": status or "",
        "yes_bid": yes_bid,
        "yes_ask": yes_ask,
        "no_bid": no_bid,
        "no_ask": no_ask,
        "volume": volume,
        "open_interest": open_interest,
    }


def fetch_markets_for_series(client: KalshiClient, series_ticker: str, limit: int) -> List[Dict[str, Any]]:
    """Best-effort listing of markets for a series using the same endpoint patterns as kalshi_inspect."""
    params: Dict[str, Any] = {
        "limit": min(int(limit), 1000),
        "status": "open",
        "series_ticker": series_ticker,
    }
    items, last_resp, _pages = paginate(client, "/markets", "markets", params, hard_limit=int(limit))

    # If the API shape differs, try a fallback list_key if present.
    if not items and isinstance(last_resp, dict):
        alt = last_resp.get("items")
        if isinstance(alt, list):
            items = [it for it in alt if isinstance(it, dict)]
    return items


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default=str(DEFAULT_CONFIG))
    p.add_argument("--db", default=str(DEFAULT_DB))
    p.add_argument(
        "--series-tickers",
        default=os.getenv("WEATHER_KALSHI_SERIES_TICKERS", ""),
        help="Comma-separated Kalshi series tickers (or set WEATHER_KALSHI_SERIES_TICKERS).",
    )
    p.add_argument("--limit", default="2000", help="Hard cap on markets pulled per series.")
    p.add_argument(
        "--require-city-match",
        action="store_true",
        help="Only store markets whose ticker/event_ticker contains a known city_key.",
    )
    args = p.parse_args()

    series_tickers = [s.strip() for s in str(args.series_tickers).split(",") if s.strip()]
    if not series_tickers:
        raise SystemExit("No series tickers provided. Use --series-tickers or WEATHER_KALSHI_SERIES_TICKERS.")

    cities = load_city_keys(Path(args.config))
    city_keys = [c.key for c in cities]

    client = KalshiClient.from_env()
    db_lib.ensure_schema(Path(args.db))
    snapshot_time_utc = _now_utc_iso()
    wrote = 0
    skipped = 0

    for series in series_tickers:
        markets = fetch_markets_for_series(client, series, limit=int(args.limit))
        for m in markets:
            norm = extract_market_fields(m)
            market_ticker = norm["market_ticker"]
            event_ticker = norm["event_ticker"] or ""
            if not market_ticker:
                skipped += 1
                continue

            hay = f"{market_ticker} {event_ticker}".upper()

            city_key = ""
            for ck in city_keys:
                if ck.upper() in hay:
                    city_key = ck
                    break

            if args.require_city_match and not city_key:
                skipped += 1
                continue
            if not city_key:
                city_key = "UNKNOWN"

            target_date_local = infer_target_date_local_from_ticker(hay)

            db_lib.insert_kalshi_weather_market_snapshot(
                Path(args.db),
                snapshot_time_utc=snapshot_time_utc,
                city_key=city_key,
                target_date_local=target_date_local,
                series_ticker=series,
                event_ticker=event_ticker,
                market_ticker=market_ticker,
                status=norm["status"] or "",
                yes_bid=norm["yes_bid"],
                yes_ask=norm["yes_ask"],
                no_bid=norm["no_bid"],
                no_ask=norm["no_ask"],
                volume=norm["volume"],
                open_interest=norm["open_interest"],
                raw=m,
            )
            wrote += 1

    print(
        json.dumps(
            {
                "snapshot_time_utc": snapshot_time_utc,
                "series_tickers": series_tickers,
                "rows_written": wrote,
                "rows_skipped": skipped,
                "db": str(args.db),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
