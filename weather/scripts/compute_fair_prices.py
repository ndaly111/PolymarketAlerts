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

import math

from weather.lib import db as db_lib
from weather.lib.fair import adjust_pmf_with_progress, normalize_pmf, shift_pmf, summarize_pmf


def _build_fallback_gaussian_pmf(center: int, std: float = 4.0, min_tail_prob: float = 0.01) -> Dict[int, float]:
    """Build a Gaussian PMF when no error model exists. Default std=4°F is typical NWS error."""
    denom = 2.0 * (std ** 2)

    def gaussian_density(k: int) -> float:
        return math.exp(-((float(k) - center) ** 2) / denom)

    out: Dict[int, float] = {center: gaussian_density(center)}

    # Extend in both directions until tail < min_tail_prob
    for direction in [-1, 1]:
        k = center + direction
        while True:
            density = gaussian_density(k)
            out[k] = density
            total = sum(out.values())
            if direction == -1:
                tail = sum(out[j] for j in out if j <= k) / total
            else:
                tail = sum(out[j] for j in out if j >= k) / total
            if tail < min_tail_prob:
                break
            k += direction
            if abs(k - center) > 30:
                break

    total = sum(out.values())
    return {k: v / total for k, v in sorted(out.items())}


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
    p.add_argument(
        "--error-model-source",
        default=os.getenv("WEATHER_ERROR_MODEL_SOURCE", ""),
        help="Source for error models (default: same as forecast-source, or mos_gfs_18z_archive if set).",
    )
    p.add_argument(
        "--error-model-hour",
        type=int,
        default=int(os.getenv("WEATHER_ERROR_MODEL_HOUR", "0")),
        help="Snapshot hour for error models (default: same as snapshot hour, or 12 for MOS).",
    )
    p.add_argument(
        "--use-latest",
        action="store_true",
        help="Use the latest snapshot by timestamp instead of a fixed snapshot hour. "
             "This ensures pricing uses the most current NWS forecast.",
    )
    p.add_argument(
        "--enable-intraday-adjustment",
        action="store_true",
        default=os.getenv("WEATHER_ENABLE_INTRADAY_ADJUSTMENT", "").lower() in ("1", "true", "yes"),
        help="Apply intraday progress adjustment to PMF (truncation + dispersion shrink).",
    )
    args = p.parse_args()

    snapshot_hour_local = int(os.getenv("WEATHER_SNAPSHOT_HOUR_LOCAL", "6"))
    fallback_source = os.getenv("WEATHER_FALLBACK_SOURCE", "").strip() or f"{args.forecast_source}_fallback"
    use_latest = args.use_latest

    # Error model source/hour - defaults to MOS archive if available, else same as forecast
    error_model_source = args.error_model_source.strip() or str(args.forecast_source)
    error_model_hour = args.error_model_hour if args.error_model_hour > 0 else snapshot_hour_local
    db_path = Path(args.db)
    cities = load_cities(Path(args.config))
    wrote = 0

    if use_latest:
        print(f"[info] Using LATEST snapshot by timestamp: source={args.forecast_source}")
    else:
        print(f"[info] Looking for snapshots: source={args.forecast_source}, hour={snapshot_hour_local}")
    print(f"[info] Error model source: {error_model_source}, hour={error_model_hour}")

    for c in cities:
        if args.city and c.key != args.city:
            continue

        now_local = _now_utc().astimezone(ZoneInfo(c.tz))
        target_date_local = args.date.strip() or now_local.date().isoformat()
        month = int(target_date_local.split("-")[1])

        # Fetch snapshot: use latest by timestamp if --use-latest, else fixed hour
        if use_latest:
            snap = db_lib.fetch_latest_forecast_snapshot(
                db_path,
                city_key=c.key,
                target_date_local=target_date_local,
                source=str(args.forecast_source),
            )
            snapshot_kind = "latest" if snap else None
        else:
            snap = db_lib.fetch_forecast_snapshot(
                db_path,
                city_key=c.key,
                target_date_local=target_date_local,
                snapshot_hour_local=snapshot_hour_local,
                source=str(args.forecast_source),
            )
            snapshot_kind = "canonical"

        # Fallback to fallback_source if primary not found
        if not snap:
            snap = db_lib.fetch_latest_forecast_snapshot(
                db_path,
                city_key=c.key,
                target_date_local=target_date_local,
                source=fallback_source,
            )
            snapshot_kind = "fallback" if snap else "missing"
        if not snap:
            if use_latest:
                print(f"[skip] {c.key}: no snapshot for date={target_date_local} source={args.forecast_source}")
            else:
                print(f"[skip] {c.key}: no snapshot for date={target_date_local} hour={snapshot_hour_local} source={args.forecast_source}")
            continue

        # Try error model from specified source first, then fallback to forecast source
        model = db_lib.fetch_error_model(
            db_path,
            city_key=c.key,
            month=month,
            snapshot_hour_local=error_model_hour,
            source=error_model_source,
        )
        model_source = error_model_source
        model_snapshot_hour_local = error_model_hour

        # If no model from error_model_source, try forecast source
        if not model and error_model_source != str(args.forecast_source):
            model = db_lib.fetch_error_model(
                db_path,
                city_key=c.key,
                month=month,
                snapshot_hour_local=snapshot_hour_local,
                source=str(args.forecast_source),
            )
            if model:
                model_source = str(args.forecast_source)
                model_snapshot_hour_local = snapshot_hour_local

        forecast_high = int(snap["forecast_high_f"])
        if model:
            pmf_error: Dict[int, float] = model["pmf"]
            pmf_high = normalize_pmf(shift_pmf(pmf_error, forecast_high))
            error_model_n_samples = int(model["n_samples"])
            error_model_updated_at_utc = model["updated_at_utc"]
        else:
            # Fallback: use Gaussian with std=4°F (typical NWS forecast error)
            pmf_high = _build_fallback_gaussian_pmf(forecast_high, std=4.0)
            error_model_n_samples = 0
            error_model_updated_at_utc = None

        # Apply intraday adjustment if enabled and we have observation data
        intraday_adjustment = None
        if args.enable_intraday_adjustment:
            intraday_state = db_lib.fetch_latest_intraday_state(
                db_path,
                city_key=c.key,
                target_date_local=target_date_local,
            )
            if intraday_state:
                pmf_high, intraday_meta = adjust_pmf_with_progress(
                    pmf_high,
                    max_observed=intraday_state["max_observed_f"],
                    progress=intraday_state["progress"],
                )
                intraday_adjustment = {
                    **intraday_meta,
                    "observation_time_utc": intraday_state["observation_time_utc"],
                    "current_temp_f": intraday_state["current_temp_f"],
                    "baseline_temp_f": intraday_state["morning_temp_f"],
                    "station_id": intraday_state["station_id"],
                }
                print(
                    f"[intraday] {c.key}: progress={intraday_state['progress']:.1%} "
                    f"max_observed={intraday_state['max_observed_f']}F"
                )

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
            "intraday_adjustment": intraday_adjustment,
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
