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
import sys
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
    label: str


def _now_utc_iso() -> str:
    return datetime.now(tz=ZoneInfo("UTC")).isoformat(timespec="seconds")


def load_city_keys(config_path: Path) -> List[City]:
    data = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    out: List[City] = []
    for row in data.get("cities", []):
        out.append(
            City(
                key=str(row["key"]).strip(),
                label=str(row.get("label", "")).strip(),
            )
        )
    return out


def load_series_tickers_from_config(config_path: Path) -> List[str]:
    """Load Kalshi series tickers from cities.yml kalshi_series field."""
    data = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    tickers = []
    for row in data.get("cities", []):
        series = row.get("kalshi_series", "").strip()
        if series:
            tickers.append(series)
    return tickers


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


def fetch_weather_series(client: KalshiClient, limit: int = 2000) -> List[Dict[str, Any]]:
    params: Dict[str, Any] = {"limit": min(int(limit), 1000), "category": "Weather"}
    items, last_resp, _pages = paginate(client, "/series", "series", params, hard_limit=int(limit))
    if not items and isinstance(last_resp, dict):
        alt = last_resp.get("items")
        if isinstance(alt, list):
            items = [it for it in alt if isinstance(it, dict)]
    return items


def _series_ticker(series: Dict[str, Any]) -> str:
    return _pick_first_str(series.get("ticker"), series.get("series_ticker"), series.get("seriesTicker"))


def _series_title(series: Dict[str, Any]) -> str:
    return _pick_first_str(series.get("title"), series.get("name"), series.get("series_title"), series.get("seriesTitle"))


def discover_series_tickers(client: KalshiClient, cities: List[City]) -> List[str]:
    series = fetch_weather_series(client)
    if not series:
        return []

    city_tokens = {c.key.upper() for c in cities if c.key}
    city_tokens.update({c.label.upper() for c in cities if c.label})
    filtered: List[str] = []

    for s in series:
        ticker = _series_ticker(s)
        title = _series_title(s)
        hay = f"{ticker} {title}".upper()
        if not hay.strip():
            continue

        if any(token in hay for token in city_tokens) and "HIGH" in hay:
            if ticker:
                filtered.append(ticker)

    # Preserve order, de-dupe.
    seen = set()
    out: List[str] = []
    for t in filtered:
        if t in seen:
            continue
        seen.add(t)
        out.append(t)
    return out


def collect_markets(
    client: KalshiClient,
    series_tickers: List[str],
    city_tokens: Dict[str, str],
    require_city_match: bool,
    db_path: Path,
    snapshot_time_utc: str,
    limit: int,
) -> Dict[str, int]:
    wrote = 0
    skipped = 0
    total_markets = 0

    for series in series_tickers:
        markets = fetch_markets_for_series(client, series, limit=limit)
        if markets:
            total_markets += len(markets)
        for m in markets:
            norm = extract_market_fields(m)
            market_ticker = norm["market_ticker"]
            event_ticker = norm["event_ticker"] or ""
            if not market_ticker:
                skipped += 1
                continue

            hay = f"{market_ticker} {event_ticker}".upper()

            city_key = ""
            for token in sorted(city_tokens.keys(), key=len, reverse=True):
                if token in hay:
                    city_key = city_tokens[token]
                    break

            if require_city_match and not city_key:
                skipped += 1
                continue
            if not city_key:
                city_key = "UNKNOWN"

            target_date_local = infer_target_date_local_from_ticker(hay)

            db_lib.insert_kalshi_weather_market_snapshot(
                db_path,
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

    return {"rows_written": wrote, "rows_skipped": skipped, "total_markets": total_markets}


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
    p.add_argument(
        "--require-markets",
        action="store_true",
        help="Exit non-zero if no markets are written after fallback discovery.",
    )
    p.add_argument("--out-json", default="", help="Optional path to write JSON output.")
    args = p.parse_args()

    series_tickers = [s.strip() for s in str(args.series_tickers).split(",") if s.strip()]

    cities = load_city_keys(Path(args.config))
    city_tokens: Dict[str, str] = {}
    for city in cities:
        if city.key:
            city_tokens[city.key.upper()] = city.key
        if city.label:
            city_tokens[city.label.upper()] = city.key
        if "LAX" in city.key.upper() or "LOS ANGELES" in city.label.upper():
            city_tokens.setdefault("LAX", city.key)
            city_tokens.setdefault("LOS ANGELES", city.key)

    client = KalshiClient.from_env()
    db_lib.ensure_schema(Path(args.db))
    snapshot_time_utc = _now_utc_iso()
    db_path = Path(args.db)

    if not series_tickers:
        # Try loading from config file first
        series_tickers = load_series_tickers_from_config(Path(args.config))
        if series_tickers:
            print(f"[kalshi] loaded series tickers from config: {', '.join(series_tickers)}", file=sys.stderr)
        else:
            # Fall back to API discovery
            series_tickers = discover_series_tickers(client, cities)
            if series_tickers:
                print(f"[kalshi] discovered series tickers via API: {', '.join(series_tickers)}", file=sys.stderr)

    used_series = list(series_tickers)
    stats = collect_markets(
        client=client,
        series_tickers=series_tickers,
        city_tokens=city_tokens,
        require_city_match=args.require_city_match,
        db_path=db_path,
        snapshot_time_utc=snapshot_time_utc,
        limit=int(args.limit),
    )

    if stats["total_markets"] == 0 or stats["rows_written"] == 0:
        discovered = discover_series_tickers(client, cities)
        if discovered:
            print(f"[kalshi] discovered series tickers: {', '.join(discovered)}", file=sys.stderr)
            used_series = list(discovered)
            stats = collect_markets(
                client=client,
                series_tickers=discovered,
                city_tokens=city_tokens,
                require_city_match=args.require_city_match,
                db_path=db_path,
                snapshot_time_utc=snapshot_time_utc,
                limit=int(args.limit),
            )

    payload = {
        "snapshot_time_utc": snapshot_time_utc,
        "series_tickers": used_series,
        "rows_written": stats["rows_written"],
        "rows_skipped": stats["rows_skipped"],
        "db": str(args.db),
    }
    if args.out_json:
        Path(args.out_json).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))
    if args.require_markets and stats["rows_written"] == 0:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
