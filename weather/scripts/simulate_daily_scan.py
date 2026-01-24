"""
Offline simulation runner for the weather daily scan.

Goal:
  - Inject a fake forecast snapshot (e.g., Chicago tomorrow high=8F)
  - Inject a fake error model (so pricing produces probabilities)
  - Inject fake Kalshi markets + orderbook (so edge calc runs)
  - Run the existing fair price + edge pipeline to produce SUMMARY.md

This avoids hitting NWS / Kalshi during testing.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional
from zoneinfo import ZoneInfo

import yaml

from weather.lib import db as db_lib

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = ROOT / "weather" / "config" / "cities.yml"
DEFAULT_DB = ROOT / "weather" / "data" / "weather.db"


@dataclass(frozen=True)
class City:
    key: str
    label: str
    tz: str


@dataclass(frozen=True)
class FakeMarket:
    city_key: str
    target_date_local: str
    title: str
    market_ticker: str
    yes_bid_cents: int
    yes_ask_cents: int
    no_bid_cents: int
    no_ask_cents: int
    volume: int = 1000
    open_interest: int = 1000
    bucket_type: Optional[str] = None
    lo: Optional[int] = None
    hi: Optional[int] = None


def _now_utc() -> datetime:
    return datetime.now(tz=ZoneInfo("UTC"))


def _load_cities(config_path: Path) -> List[City]:
    data = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    out: List[City] = []
    for row in data.get("cities", []):
        out.append(
            City(
                key=str(row["key"]).strip(),
                label=str(row.get("label", row["key"])).strip(),
                tz=str(row.get("tz", "America/New_York")).strip(),
            )
        )
    return out


def _normalize_city_token(value: str) -> str:
    return "".join(ch for ch in value.lower().strip() if ch.isalnum())


def _resolve_city(cities: List[City], token: str) -> Optional[City]:
    if not token:
        return None
    needle = _normalize_city_token(token)
    for c in cities:
        if _normalize_city_token(c.key) == needle:
            return c
        if _normalize_city_token(c.label) == needle:
            return c
    return None


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--city", default="", help="City key or label (e.g., CHIHIGH or chicago)")
    p.add_argument("--target-date", required=True, help="YYYY-MM-DD (e.g., 2026-01-24)")
    p.add_argument("--forecast-high", type=float, required=True, help="Fake forecast high in °F")
    p.add_argument("--config", default=str(DEFAULT_CONFIG))
    p.add_argument("--db", default=str(DEFAULT_DB))
    p.add_argument("--markets-json", default="", help="Path to JSON fixtures for fake markets (optional)")
    p.add_argument("--forecast-source", default="nws_hourly_max", help="forecast_snapshots.source label")
    p.add_argument("--snapshot-hour-local", default="6", help="snapshot_hour_local to store")
    p.add_argument("--fee-cents", default="2", help="fee cents used by compute_edges")
    p.add_argument("--min-ev", default="0.02")
    p.add_argument("--min-q", default="0.05")
    p.add_argument("--min-volume", default="0")
    p.add_argument("--min-open-interest", default="0")
    p.add_argument("--max-spread-cents", default="")
    p.add_argument("--top-n", default="25")
    p.add_argument("--require-ask", action="store_true", default=True)
    p.add_argument("--no-require-ask", dest="require_ask", action="store_false")
    return p.parse_args()


def _run_main(main_fn, argv: List[str]) -> int:
    prev_argv = sys.argv
    try:
        sys.argv = [prev_argv[0]] + list(argv)
        return int(main_fn())
    finally:
        sys.argv = prev_argv


def _load_fake_markets_from_json(path: Path) -> List[FakeMarket]:
    """
    JSON format:
      {
        "markets": [
          {
            "market_ticker": "...",
            "title": "...",
            "yes_bid": 48,
            "yes_ask": 52,
            "no_bid": 48,
            "no_ask": 52,
            "volume": 2000,
            "open_interest": 1500
          }
        ]
      }
    """
    obj = json.loads(path.read_text(encoding="utf-8"))
    out: List[FakeMarket] = []
    for m in obj.get("markets", []):
        out.append(
            FakeMarket(
                city_key="",
                target_date_local="",
                title=str(m.get("title", "")),
                market_ticker=str(m["market_ticker"]),
                yes_bid_cents=int(m.get("yes_bid", 0)),
                yes_ask_cents=int(m.get("yes_ask", 0)),
                no_bid_cents=int(m.get("no_bid", 0)),
                no_ask_cents=int(m.get("no_ask", 0)),
                volume=int(m.get("volume", 0)),
                open_interest=int(m.get("open_interest", 0)),
                bucket_type=m.get("bucket_type"),
                lo=int(m["lo"]) if m.get("lo") is not None else None,
                hi=int(m["hi"]) if m.get("hi") is not None else None,
            )
        )
    return out


def _build_fake_markets(city: City, target_date: str, forecast_high: int) -> List[FakeMarket]:
    low = forecast_high - 3
    mid_low = forecast_high - 2
    mid_high = forecast_high + 2
    high = forecast_high + 3

    return [
        FakeMarket(
            city_key=city.key,
            target_date_local=target_date,
            title=f"{city.label} high temperature {low}° or lower",
            market_ticker=f"{city.key}-{target_date}-LE-{low}",
            yes_bid_cents=66,
            yes_ask_cents=70,
            no_bid_cents=30,
            no_ask_cents=34,
            volume=1800,
            open_interest=1400,
        ),
        FakeMarket(
            city_key=city.key,
            target_date_local=target_date,
            title=f"{city.label} high temperature between {mid_low} and {mid_high}",
            market_ticker=f"{city.key}-{target_date}-RANGE-{mid_low}-{mid_high}",
            yes_bid_cents=46,
            yes_ask_cents=50,
            no_bid_cents=50,
            no_ask_cents=54,
            volume=2600,
            open_interest=2100,
        ),
        FakeMarket(
            city_key=city.key,
            target_date_local=target_date,
            title=f"{city.label} high temperature {high}° or higher",
            market_ticker=f"{city.key}-{target_date}-GE-{high}",
            yes_bid_cents=18,
            yes_ask_cents=22,
            no_bid_cents=78,
            no_ask_cents=82,
            volume=2200,
            open_interest=1900,
        ),
    ]


def main() -> int:
    args = _parse_args()
    cities = _load_cities(Path(args.config))
    city = _resolve_city(cities, args.city) if args.city else None
    if not city:
        raise SystemExit(f"Unknown city '{args.city}'. Use a city key or label from {args.config}.")

    target_date = str(args.target_date).strip()
    snapshot_hour_local = int(args.snapshot_hour_local)
    forecast_high = int(round(float(args.forecast_high)))
    db_path = Path(args.db)

    os.environ["WEATHER_OFFLINE_FIXTURES"] = "1"
    os.environ["WEATHER_SNAPSHOT_HOUR_LOCAL"] = str(snapshot_hour_local)

    now_utc = _now_utc().isoformat()
    db_lib.upsert_forecast_snapshot(
        db_path,
        city_key=city.key,
        target_date_local=target_date,
        snapshot_time_utc=now_utc,
        snapshot_hour_local=snapshot_hour_local,
        snapshot_tz=city.tz,
        forecast_high_f=forecast_high,
        source=str(args.forecast_source),
        points_url="offline://points",
        forecast_url="offline://forecast",
        qc_flags=["offline_fixture"],
        raw={"fixture": True, "forecast_high_f": forecast_high},
    )
    snapshot = db_lib.fetch_forecast_snapshot(
        db_path,
        city_key=city.key,
        target_date_local=target_date,
        snapshot_hour_local=snapshot_hour_local,
        source=str(args.forecast_source),
    )
    if not snapshot or int(snapshot.get("forecast_high_f", -9999)) != forecast_high:
        raise SystemExit("Simulation forecast snapshot not found or mismatch after insert.")

    month = int(target_date.split("-")[1])
    pmf: Dict[int, float] = {
        -4: 0.05,
        -3: 0.08,
        -2: 0.12,
        -1: 0.15,
        0: 0.2,
        1: 0.15,
        2: 0.12,
        3: 0.08,
        4: 0.05,
    }
    db_lib.upsert_error_model(
        db_path,
        city_key=city.key,
        month=month,
        snapshot_hour_local=snapshot_hour_local,
        source=str(args.forecast_source),
        n_samples=1000,
        pmf=pmf,
        updated_at_utc=now_utc,
    )

    series_ticker = f"TMAX-{city.key}"
    event_ticker = f"{series_ticker}-{target_date}"
    snapshot_time_utc = now_utc

    if args.markets_json:
        markets = _load_fake_markets_from_json(Path(args.markets_json))
        markets = [
            FakeMarket(
                city_key=city.key,
                target_date_local=target_date,
                title=m.title,
                market_ticker=m.market_ticker,
                yes_bid_cents=m.yes_bid_cents,
                yes_ask_cents=m.yes_ask_cents,
                no_bid_cents=m.no_bid_cents,
                no_ask_cents=m.no_ask_cents,
                volume=m.volume,
                open_interest=m.open_interest,
                bucket_type=m.bucket_type,
                lo=m.lo,
                hi=m.hi,
            )
            for m in markets
        ]
    else:
        markets = _build_fake_markets(city, target_date, forecast_high)

    if not markets:
        raise SystemExit("No markets loaded/built for simulation.")

    if args.require_ask:
        missing = [m.market_ticker for m in markets if (m.yes_ask_cents <= 0 and m.no_ask_cents <= 0)]
        if missing:
            raise SystemExit(f"Simulation fixtures missing ask prices for: {missing[:10]}")

    for m in markets:
        raw = {"title": m.title, "fixture": True, "market_ticker": m.market_ticker}
        if m.bucket_type is not None:
            raw["bucket_type"] = m.bucket_type
        if m.lo is not None:
            raw["lo"] = m.lo
        if m.hi is not None:
            raw["hi"] = m.hi
        db_lib.upsert_kalshi_weather_market_fixture(
            db_path,
            snapshot_time_utc=snapshot_time_utc,
            city_key=m.city_key,
            target_date_local=m.target_date_local,
            series_ticker=series_ticker,
            event_ticker=event_ticker,
            market_ticker=m.market_ticker,
            status="active",
            yes_bid=m.yes_bid_cents,
            yes_ask=m.yes_ask_cents,
            no_bid=m.no_bid_cents,
            no_ask=m.no_ask_cents,
            volume=m.volume,
            open_interest=m.open_interest,
            raw=raw,
        )

    from weather.scripts import compute_edges, compute_fair_prices

    fair_args = [
        "--config",
        str(args.config),
        "--db",
        str(db_path),
        "--city",
        city.key,
        "--date",
        target_date,
        "--forecast-source",
        str(args.forecast_source),
    ]
    fair_rc = _run_main(compute_fair_prices.main, fair_args)
    if fair_rc != 0:
        return fair_rc

    edges_args = [
        "--config",
        str(args.config),
        "--db",
        str(db_path),
        "--city",
        city.key,
        "--date",
        target_date,
        "--forecast-source",
        str(args.forecast_source),
        "--fee-cents",
        str(args.fee_cents),
        "--min-ev",
        str(args.min_ev),
        "--min-q",
        str(args.min_q),
        "--min-volume",
        str(args.min_volume),
        "--min-open-interest",
        str(args.min_open_interest),
        "--top-n",
        str(args.top_n),
    ]
    if args.max_spread_cents != "":
        edges_args += ["--max-spread-cents", str(args.max_spread_cents)]
    if args.require_ask:
        edges_args += ["--require-ask"]

    edges_rc = _run_main(compute_edges.main, edges_args)
    if edges_rc != 0:
        return edges_rc

    src_sanitized = str(args.forecast_source).replace("/", "_")
    summary = (
        ROOT
        / "weather"
        / "outputs"
        / "edges"
        / src_sanitized
        / target_date
        / "SUMMARY.md"
    )
    print(f"[simulate] done. Expected summary at: {summary}")
    if summary.exists():
        print(summary.read_text(encoding="utf-8"))
    else:
        raise SystemExit("[simulate] SUMMARY.md not found — treating as failure so CI can't silently pass.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
