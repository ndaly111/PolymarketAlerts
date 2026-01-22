#!/usr/bin/env python3
"""Compute fee-adjusted EV (YES/NO) for Kalshi weather markets.

Inputs:
- Fair-price artifacts from weather/scripts/compute_fair_prices.py
- Kalshi market snapshots stored in weather.db (weather/scripts/collect_kalshi_weather_markets.py)

Outputs:
- JSON per city:
    weather/outputs/edges/{forecast_source}/{YYYY-MM-DD}/{CITY_KEY}.json
- Optional combined markdown summary:
    weather/outputs/edges/{forecast_source}/{YYYY-MM-DD}/SUMMARY.md
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
from zoneinfo import ZoneInfo

import yaml

from weather.lib import db as db_lib
from weather.lib.fees import FeeSchedule, ev_no, ev_yes
from weather.lib.kalshi_weather import (
    best_buy_prices_from_snapshot_row,
    parse_event_spec_from_title,
    prob_event,
)

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = ROOT / "weather" / "config" / "cities.yml"
DEFAULT_DB = ROOT / "weather" / "data" / "weather.db"
FAIR_BASE = ROOT / "weather" / "outputs" / "fair_prices"
OUT_BASE = ROOT / "weather" / "outputs" / "edges"


@dataclass(frozen=True)
class City:
    key: str
    label: str
    tz: str


def _now_utc() -> datetime:
    return datetime.now(tz=ZoneInfo("UTC"))


def load_cities(config_path: Path) -> List[City]:
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


def _read_fair_artifact(
    forecast_source: str,
    target_date_local: str,
    city_key: str,
) -> Optional[Dict[str, Any]]:
    src = str(forecast_source).replace("/", "_")
    path = FAIR_BASE / src / target_date_local / f"{city_key}.json"
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _market_title_from_raw(raw: Dict[str, Any]) -> str:
    for k in ("title", "market_title", "marketTitle", "name"):
        v = raw.get(k)
        if isinstance(v, str) and v.strip():
            return v.strip()
    m = raw.get("market") if isinstance(raw.get("market"), dict) else {}
    for k in ("title", "market_title", "marketTitle", "name"):
        v = m.get(k)
        if isinstance(v, str) and v.strip():
            return v.strip()
    return ""


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default=str(DEFAULT_CONFIG))
    p.add_argument("--db", default=str(DEFAULT_DB))
    p.add_argument("--city", default="", help="Optional city_key filter (default: all)")
    p.add_argument("--date", default="", help="Optional YYYY-MM-DD (default: today in each city TZ)")
    p.add_argument("--forecast-source", default="nws_hourly_max")
    p.add_argument("--fee-cents", default=os.getenv("WEATHER_BUY_FEE_CENTS", "2"))
    p.add_argument("--top-n", default=os.getenv("WEATHER_TOP_N", "25"))
    p.add_argument("--min-ev", default=os.getenv("WEATHER_MIN_EV", "0.02"))
    p.add_argument("--require-ask", action="store_true")
    args = p.parse_args()

    fee_cents = int(str(args.fee_cents).strip())
    fees = FeeSchedule(open_fee_cents=fee_cents)
    fee_open = fees.open_fee_dollars()

    top_n = int(str(args.top_n).strip())
    min_ev = float(str(args.min_ev).strip())

    db_path = Path(args.db)
    cities = load_cities(Path(args.config))

    wrote = 0
    all_rows_for_md: List[Dict[str, Any]] = []

    for c in cities:
        if args.city and c.key != args.city:
            continue

        now_local = _now_utc().astimezone(ZoneInfo(c.tz))
        target_date_local = args.date.strip() or now_local.date().isoformat()

        fair = _read_fair_artifact(args.forecast_source, target_date_local, c.key)
        if not fair:
            continue

        pmf_raw = fair.get("pmf_high_f") or {}
        pmf: Dict[int, float] = {}
        for k, v in pmf_raw.items():
            try:
                pmf[int(k)] = float(v)
            except Exception:
                continue

        snap_time = db_lib.fetch_latest_kalshi_weather_snapshot_time(
            db_path,
            city_key=c.key,
            target_date_local=target_date_local,
        )
        if not snap_time:
            continue

        markets = list(
            db_lib.fetch_kalshi_weather_markets_at_snapshot(
                db_path,
                snapshot_time_utc=snap_time,
                city_key=c.key,
                target_date_local=target_date_local,
            )
        )
        if not markets:
            continue

        scored: List[Dict[str, Any]] = []
        drops = {"no_title": 0, "unparsed": 0, "no_prices": 0, "no_positive_ev": 0}

        for row in markets:
            raw = row.get("raw") if isinstance(row.get("raw"), dict) else {}
            title = _market_title_from_raw(raw)
            if not title:
                drops["no_title"] += 1
                continue

            spec = parse_event_spec_from_title(title)
            if not spec:
                drops["unparsed"] += 1
                continue

            q = prob_event(pmf, spec)
            p_yes, p_no = best_buy_prices_from_snapshot_row(row)

            if args.require_ask and (p_yes is None and p_no is None):
                drops["no_prices"] += 1
                continue

            best_side = None
            best_ev = -1e9
            chosen_price = None

            if p_yes is not None:
                ev = ev_yes(q, float(p_yes), fee_open)
                if ev > best_ev:
                    best_ev = ev
                    best_side = "YES"
                    chosen_price = float(p_yes)

            if p_no is not None:
                ev = ev_no(q, float(p_no), fee_open)
                if ev > best_ev:
                    best_ev = ev
                    best_side = "NO"
                    chosen_price = float(p_no)

            if best_side is None or chosen_price is None:
                drops["no_prices"] += 1
                continue

            if best_ev < min_ev:
                drops["no_positive_ev"] += 1
                continue

            scored.append(
                {
                    "market_ticker": row.get("market_ticker"),
                    "title": title,
                    "event": {"kind": spec.kind, "a": spec.a, "b": spec.b, "desc": spec.describe()},
                    "model_q": float(q),
                    "prices": {"yes_ask": p_yes, "no_ask": p_no},
                    "decision": {
                        "side_to_buy": best_side,
                        "buy_price": chosen_price,
                        "fee_open": fee_open,
                        "ev": best_ev,
                    },
                    "liquidity": {"volume": row.get("volume"), "open_interest": row.get("open_interest")},
                    "snapshot": {
                        "snapshot_time_utc": row.get("snapshot_time_utc"),
                        "series_ticker": row.get("series_ticker"),
                        "event_ticker": row.get("event_ticker"),
                        "status": row.get("status"),
                    },
                }
            )

        scored.sort(key=lambda r: float(((r.get("decision") or {}).get("ev") or 0.0)), reverse=True)
        scored = scored[: max(1, top_n)]

        out = {
            "city_key": c.key,
            "label": c.label,
            "tz": c.tz,
            "target_date_local": target_date_local,
            "forecast_source": str(args.forecast_source),
            "fee": {"open_fee_cents": fee_cents, "open_fee_dollars": fee_open},
            "kalshi_snapshot_time_utc": snap_time,
            "drops": drops,
            "candidates": scored,
        }

        src = str(args.forecast_source).replace("/", "_")
        out_dir = OUT_BASE / src / target_date_local
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / f"{c.key}.json").write_text(
            json.dumps(out, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        wrote += 1

        for r in scored:
            all_rows_for_md.append(
                {
                    "city": c.key,
                    "title": r.get("title", ""),
                    "side": (r.get("decision") or {}).get("side_to_buy", ""),
                    "ev": (r.get("decision") or {}).get("ev", 0.0),
                    "price": (r.get("decision") or {}).get("buy_price", 0.0),
                    "q": r.get("model_q", 0.0),
                }
            )

    if wrote > 0 and all_rows_for_md:
        all_rows_for_md.sort(key=lambda r: float(r.get("ev", 0.0)), reverse=True)
        src = str(args.forecast_source).replace("/", "_")
        target_date = args.date.strip() or _now_utc().date().isoformat()
        out_dir = OUT_BASE / src / target_date
        out_dir.mkdir(parents=True, exist_ok=True)
        md_lines = [
            f"# Weather edges — {target_date} ({src})",
            "",
            f"Fee (open): {fee_cents}¢ (${fee_open:.2f}) | min EV: {min_ev:.2f} | top_n per city: {top_n}",
            "",
            "| City | Side | EV ($/contract) | Price | q(model) | Market |",
            "|---|---|---:|---:|---:|---|",
        ]
        for r in all_rows_for_md[:50]:
            md_lines.append(
                "| {city} | {side} | {ev:.3f} | {price:.2f} | {q:.3f} | {title} |".format(
                    city=r["city"],
                    side=r["side"],
                    ev=float(r["ev"]),
                    price=float(r["price"]),
                    q=float(r["q"]),
                    title=str(r["title"]).replace("|", "\\|"),
                )
            )
        (out_dir / "SUMMARY.md").write_text("\n".join(md_lines) + "\n", encoding="utf-8")

    print(f"[done] wrote {wrote} edge artifacts under {OUT_BASE}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
