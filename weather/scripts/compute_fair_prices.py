#!/usr/bin/env python3
"""Compute a calibrated distribution for today's high temperature.

Step 2 (foundation):
  - Read today's forecast snapshot (forecast high) from the DB
  - Read the month-of-year error PMF for the same snapshot hour
  - Produce an observed-high PMF = forecast_high + error
  - Write a small JSON artifact per city to:
      weather/outputs/fair_prices/YYYY-MM-DD/{CITY_KEY}.json
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List
from zoneinfo import ZoneInfo

import yaml

from weather.lib import db as db_lib
from weather.lib.fair import normalize_pmf, shift_pmf, summarize_pmf


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = ROOT / "weather" / "config" / "cities.yml"
DEFAULT_DB = Path(os.getenv("WEATHER_DB_PATH", str(ROOT / "weather" / "data" / "weather_forecast_accuracy.db")))
OUT_BASE = ROOT / "weather" / "outputs" / "fair_prices"


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


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default=str(DEFAULT_CONFIG))
    p.add_argument("--db", default=str(DEFAULT_DB))
    p.add_argument("--city", default="", help="Optional city_key filter (default: all)")
    p.add_argument(
        "--date",
        default="",
        help="Optional target date_local YYYY-MM-DD (default: today in each city TZ)",
    )
    p.add_argument("--forecast-source", default="nws_hourly_max", help="Forecast source to price from.")
    args = p.parse_args()

    snapshot_hour_local = int(os.getenv("WEATHER_SNAPSHOT_HOUR_LOCAL", "6"))
    fallback_source = os.getenv("WEATHER_FALLBACK_SOURCE", "").strip() or f"{args.forecast_source}_fallback"
    db_path = Path(args.db)
    cities = load_cities(Path(args.config))
    wrote = 0

    for c in cities:
        if args.city and c.key != args.city:
            continue

        now_local = _now_utc().astimezone(ZoneInfo(c.tz))
        target_date_local = args.date.strip() or now_local.date().isoformat()
        month = int(target_date_local.split("-")[1])

        snap = db_lib.fetch_forecast_snapshot(
            db_path,
            city_key=c.key,
            target_date_local=target_date_local,
            snapshot_hour_local=snapshot_hour_local,
            source=str(args.forecast_source),
        )
        snapshot_kind = "canonical"
        if not snap:
            snap = db_lib.fetch_latest_forecast_snapshot(
                db_path,
                city_key=c.key,
                target_date_local=target_date_local,
                source=fallback_source,
            )
            snapshot_kind = "fallback" if snap else "missing"
        if not snap:
            continue

        model_source = str(args.forecast_source)
        model_snapshot_hour_local = snapshot_hour_local
        model = db_lib.fetch_error_model(
            db_path,
            city_key=c.key,
            month=month,
            snapshot_hour_local=model_snapshot_hour_local,
            source=model_source,
        )

        forecast_high = int(snap["forecast_high_f"])
        if model:
            pmf_error: Dict[int, float] = model["pmf"]
            pmf_high = normalize_pmf(shift_pmf(pmf_error, forecast_high))
            error_model_n_samples = int(model["n_samples"])
            error_model_updated_at_utc = model["updated_at_utc"]
        else:
            pmf_high = {int(forecast_high): 1.0}
            error_model_n_samples = 0
            error_model_updated_at_utc = None
        summary = summarize_pmf(pmf_high)

        out = {
            "city_key": c.key,
            "label": c.label,
            "tz": c.tz,
            "target_date_local": target_date_local,
            "snapshot_hour_local": int(snap["snapshot_hour_local"]),
            "forecast_source": str(args.forecast_source),
            "snapshot_kind": snapshot_kind,
            "fallback_source": fallback_source,
            "forecast_high_f": forecast_high,
            "error_model_month": month,
            "error_model_n_samples": error_model_n_samples,
            "error_model_updated_at_utc": error_model_updated_at_utc,
            "error_model_source": model_source,
            "error_model_snapshot_hour_local": model_snapshot_hour_local,
            "error_model_missing": model is None,
            "pmf_high_f": {str(k): v for k, v in sorted(pmf_high.items())},
            "summary": {
                "mean": summary.mean,
                "p10": summary.p10,
                "p50": summary.p50,
                "p90": summary.p90,
            },
            "snapshot_meta": {
                "snapshot_time_utc": snap["snapshot_time_utc"],
                "source": snap["source"],
                "qc_flags": snap["qc_flags"],
                "points_url": snap["points_url"],
                "forecast_url": snap["forecast_url"],
            },
        }

        source_slug = str(args.forecast_source).replace("/", "_")
        out_dir = OUT_BASE / source_slug / target_date_local
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{c.key}.json"
        out_path.write_text(json.dumps(out, indent=2, sort_keys=True), encoding="utf-8")
        wrote += 1

    print(f"[done] wrote {wrote} fair-price artifacts under {OUT_BASE}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
