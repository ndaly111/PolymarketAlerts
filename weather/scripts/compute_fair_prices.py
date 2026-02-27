#!/usr/bin/env python3
"""Compute a calibrated distribution for today's high temperature.

Uses a tiered approach:
  1. PRIMARY: ML quantile regression model (LightGBM) predicting error quantiles
     conditioned on 17 features (weather context, recent bias, synoptic regime)
  2. SECONDARY: Multi-model spread from Open-Meteo (HRRR, ECMWF, GFS) + NWS
  3. FALLBACK: Static monthly error PMF (legacy approach)
  4. LAST RESORT: Gaussian with std=4°F

The ML model and multi-model spread are blended via ProbabilityLadder, then
converted back to a PMF dict for backward compatibility with compute_edges.py.

Output:
  weather/outputs/fair_prices/{source}/{YYYY-MM-DD}/{CITY_KEY}.json
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from zoneinfo import ZoneInfo

import yaml

from weather.lib import db as db_lib
from weather.lib.fair import adjust_pmf_with_progress, normalize_pmf, shift_pmf, summarize_pmf
from weather.lib.probability_ladder import (
    ProbabilityLadder,
    blend_ladders,
    from_percentiles,
    from_pmf,
    from_point_forecast_and_error_pmf,
)

# ML model imports — optional, graceful fallback if not installed
try:
    from weather.lib.ml_models import (
        QuantileForestModel,
        SimpleFallbackModel,
        get_model,
        prepare_features,
    )
    HAS_ML = True
except ImportError:
    HAS_ML = False


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


def _get_multi_model_spread(
    db_path: Path,
    city_key: str,
    target_date_local: str,
) -> Tuple[Optional[int], List[Dict[str, Any]]]:
    """Query all forecast sources for a city/date and compute model spread.

    Returns (spread, model_details) where spread is max(highs) - min(highs)
    or None if fewer than 2 sources found.
    """
    conn = db_lib.connect(db_path)
    try:
        rows = conn.execute(
            """
            SELECT source, forecast_high_f, snapshot_time_utc
            FROM forecast_snapshots
            WHERE city_key = ? AND target_date_local = ?
            ORDER BY snapshot_time_utc DESC
            """,
            (city_key, target_date_local),
        ).fetchall()
    finally:
        conn.close()

    if not rows:
        return None, []

    # Deduplicate: keep the latest snapshot per source
    seen_sources: Dict[str, Dict[str, Any]] = {}
    for source, high_f, snap_time in rows:
        if source not in seen_sources:
            seen_sources[source] = {
                "source": source,
                "forecast_high_f": int(high_f),
                "snapshot_time_utc": snap_time,
            }

    details = list(seen_sources.values())
    if len(details) < 2:
        return None, details

    highs = [d["forecast_high_f"] for d in details]
    spread = max(highs) - min(highs)
    return spread, details


def _get_prev_day_error(
    db_path: Path,
    city_key: str,
    target_date_local: str,
    forecast_source: str,
) -> Optional[int]:
    """Get yesterday's forecast error (observed - forecast)."""
    try:
        dt = datetime.strptime(target_date_local, "%Y-%m-%d")
        yesterday = (dt - timedelta(days=1)).strftime("%Y-%m-%d")
    except ValueError:
        return None

    conn = db_lib.connect(db_path)
    try:
        row = conn.execute(
            """
            SELECT o.tmax_f - f.forecast_high_f
            FROM observed_cli o
            JOIN forecast_snapshots f
              ON f.city_key = o.city_key
             AND f.target_date_local = o.date_local
             AND f.source = ?
            WHERE o.city_key = ?
              AND o.date_local = ?
            ORDER BY f.snapshot_time_utc DESC
            LIMIT 1
            """,
            (forecast_source, city_key, yesterday),
        ).fetchone()
    finally:
        conn.close()

    return int(row[0]) if row else None


def _get_rolling_bias(
    db_path: Path,
    city_key: str,
    target_date_local: str,
    forecast_source: str,
    window_days: int = 7,
) -> Optional[float]:
    """Get rolling N-day forecast bias (mean of observed - forecast)."""
    try:
        dt = datetime.strptime(target_date_local, "%Y-%m-%d")
        start = (dt - timedelta(days=window_days)).strftime("%Y-%m-%d")
        end = (dt - timedelta(days=1)).strftime("%Y-%m-%d")
    except ValueError:
        return None

    conn = db_lib.connect(db_path)
    try:
        rows = conn.execute(
            """
            SELECT o.tmax_f - f.forecast_high_f
            FROM observed_cli o
            JOIN forecast_snapshots f
              ON f.city_key = o.city_key
             AND f.target_date_local = o.date_local
             AND f.source = ?
            WHERE o.city_key = ?
              AND o.date_local BETWEEN ? AND ?
            ORDER BY o.date_local DESC
            """,
            (forecast_source, city_key, start, end),
        ).fetchall()
    finally:
        conn.close()

    if not rows:
        return None

    errors = [int(r[0]) for r in rows]
    return sum(errors) / len(errors)


def _extract_ml_features(
    db_path: Path,
    city_key: str,
    target_date_local: str,
    forecast_high: int,
    forecast_source: str,
    snap: Dict[str, Any],
) -> Dict[str, float]:
    """Extract all ML features from the database for a city/date.

    Uses prepare_features() from ml_models.py with values gathered from DB.
    """
    dt = datetime.strptime(target_date_local, "%Y-%m-%d")

    # Multi-model spread as nbm_spread proxy
    spread, _ = _get_multi_model_spread(db_path, city_key, target_date_local)

    # NBM percentile spread if available
    nbm = db_lib.fetch_latest_nbm_forecast(
        db_path, city_key=city_key, target_date_local=target_date_local
    )
    nbm_spread = None
    if nbm and nbm.get("p90") is not None and nbm.get("p10") is not None:
        nbm_spread = int(nbm["p90"]) - int(nbm["p10"])
    elif spread is not None:
        nbm_spread = spread

    # Lead hours: from snapshot to ~18:00 local (approximate settlement time)
    lead_hours = 18  # default
    snap_time_str = snap.get("snapshot_time_utc", "")
    if snap_time_str:
        try:
            snap_dt = datetime.fromisoformat(snap_time_str.replace("Z", "+00:00"))
            # Settlement is roughly 18:00 local on target date
            settle_local = datetime.strptime(target_date_local, "%Y-%m-%d").replace(
                hour=18, tzinfo=ZoneInfo(snap.get("snapshot_tz", "America/New_York"))
            )
            diff = settle_local - snap_dt.astimezone(settle_local.tzinfo)
            lead_hours = max(1, int(diff.total_seconds() / 3600))
        except Exception:
            pass

    # Weather context from forecast_context_json
    context = snap.get("forecast_context") or {}
    precip_prob = float(context.get("precip_prob", context.get("precipitation_probability", 0.0)) or 0.0)
    cloud_cover = float(context.get("cloud_cover", context.get("cloudcover", 0.5)) or 0.5)
    wind_speed = float(context.get("wind_speed", context.get("windspeed_10m", 10.0)) or 10.0)

    # Recent performance
    prev_error = _get_prev_day_error(db_path, city_key, target_date_local, forecast_source)
    rolling_bias = _get_rolling_bias(db_path, city_key, target_date_local, forecast_source)

    return prepare_features(
        forecast_high=forecast_high,
        target_date=dt,
        nbm_spread=nbm_spread,
        lead_hours=lead_hours,
        precip_prob=precip_prob,
        cloud_cover=cloud_cover,
        wind_speed=wind_speed,
        prev_day_error=prev_error or 0,
        rolling_7d_bias=rolling_bias or 0.0,
    )


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = ROOT / "weather" / "config" / "cities.yml"
DEFAULT_DB = Path(os.getenv("WEATHER_DB_PATH", str(ROOT / "weather" / "data" / "weather_forecast_accuracy.db")))
DEFAULT_MODEL_DIR = ROOT / "weather" / "models"
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


def _build_ml_ladder(
    ml_model: Any,
    forecast_high: int,
    features: Dict[str, float],
) -> Optional[ProbabilityLadder]:
    """Build a ProbabilityLadder from ML quantile regression predictions.

    Returns None if the model produces degenerate output.
    """
    try:
        p10, p25, p50, p75, p90 = ml_model.predict_observed_distribution(
            forecast_high, features
        )
    except Exception as e:
        print(f"  [warn] ML prediction failed: {e}")
        return None

    # Sanity check: quantiles must be monotonically non-decreasing
    if p10 > p50 or p50 > p90 or p10 >= p90:
        print(f"  [warn] ML model degenerate: p10={p10} p50={p50} p90={p90}")
        return None

    return from_percentiles(p10, p25, p50, p75, p90, source="ml_quantile")


def _build_pmf_ladder(
    forecast_high: int,
    error_pmf: Dict[int, float],
) -> ProbabilityLadder:
    """Build a ProbabilityLadder from the static monthly error PMF (legacy path)."""
    pmf_high = normalize_pmf(shift_pmf(error_pmf, forecast_high))
    return from_pmf(pmf_high, source="error_pmf")


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
    p.add_argument(
        "--use-ml-model",
        action="store_true",
        default=os.getenv("WEATHER_USE_ML_MODEL", "").lower() in ("1", "true", "yes"),
        help="Use ML quantile regression models instead of static error PMF.",
    )
    p.add_argument(
        "--model-dir",
        default=os.getenv("WEATHER_MODEL_DIR", str(DEFAULT_MODEL_DIR)),
        help="Directory containing trained ML model .pkl files.",
    )
    args = p.parse_args()

    snapshot_hour_local = int(os.getenv("WEATHER_SNAPSHOT_HOUR_LOCAL", "6"))
    fallback_source = os.getenv("WEATHER_FALLBACK_SOURCE", "").strip() or f"{args.forecast_source}_fallback"
    use_latest = args.use_latest
    use_ml = args.use_ml_model and HAS_ML
    model_dir = Path(args.model_dir)

    if args.use_ml_model and not HAS_ML:
        print("[warn] --use-ml-model requested but ML dependencies not available (numpy/lightgbm).")
        print("[warn] Falling back to static error PMF.")

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

    # Pre-load ML models for all cities if ML mode is enabled
    ml_models: Dict[str, Any] = {}
    if use_ml:
        print(f"[info] ML mode enabled. Loading models from {model_dir}")
        for c in cities:
            if args.city and c.key != args.city:
                continue
            ml_models[c.key] = get_model(c.key, model_dir)
            model_type = type(ml_models[c.key]).__name__
            print(f"  {c.key}: {model_type}")

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

        forecast_high = int(snap["forecast_high_f"])

        # --- Build probability distribution (tiered approach) ---
        model_source_type = "fallback_gaussian"
        ml_quantiles = None
        multi_model_info = None
        error_model_n_samples = 0
        error_model_updated_at_utc = None
        model_source = error_model_source
        model_snapshot_hour_local = error_model_hour
        error_model_missing = True
        ladder: Optional[ProbabilityLadder] = None

        # Tier 1: ML quantile regression model
        if use_ml and c.key in ml_models and ml_models[c.key].is_fitted:
            features = _extract_ml_features(
                db_path, c.key, target_date_local, forecast_high,
                str(args.forecast_source), snap,
            )
            ml_ladder = _build_ml_ladder(ml_models[c.key], forecast_high, features)

            if ml_ladder is not None:
                ladder = ml_ladder
                model_source_type = "ml_quantile"
                ml_quantiles = {
                    "p10": ml_ladder.p10,
                    "p25": ml_ladder.p25,
                    "p50": ml_ladder.p50,
                    "p75": ml_ladder.p75,
                    "p90": ml_ladder.p90,
                }
                print(
                    f"[ml] {c.key}: forecast={forecast_high}F → "
                    f"p10={ml_ladder.p10} p50={ml_ladder.p50} p90={ml_ladder.p90}"
                )

                # Optionally blend with multi-model spread ladder
                spread, multi_details = _get_multi_model_spread(
                    db_path, c.key, target_date_local,
                )
                if spread is not None and len(multi_details) >= 2:
                    multi_model_info = {
                        "spread": spread,
                        "n_models": len(multi_details),
                        "models": multi_details,
                    }
                    # Build a simple ensemble ladder from multi-model forecasts
                    model_highs = [d["forecast_high_f"] for d in multi_details]
                    ensemble_mean = sum(model_highs) / len(model_highs)
                    # Use model spread as std dev proxy (spread ≈ 2*std for uniform-ish)
                    ensemble_std = max(1.0, spread / 2.0)
                    ensemble_pmf = _build_fallback_gaussian_pmf(
                        int(round(ensemble_mean)), std=ensemble_std,
                    )
                    ensemble_ladder = from_pmf(ensemble_pmf, source="multi_model_ensemble")
                    # Blend: 70% ML, 30% ensemble
                    ladder = blend_ladders(
                        [(ml_ladder, 0.7), (ensemble_ladder, 0.3)],
                        source="ml+ensemble",
                    )
                    model_source_type = "ml+ensemble"
                    print(
                        f"  [blend] {len(multi_details)} models, spread={spread}F → "
                        f"blended p10={ladder.p10} p50={ladder.p50} p90={ladder.p90}"
                    )

        # Tier 2: Static monthly error PMF (legacy)
        if ladder is None:
            pmf_model = db_lib.fetch_error_model(
                db_path,
                city_key=c.key,
                month=month,
                snapshot_hour_local=error_model_hour,
                source=error_model_source,
            )
            model_source = error_model_source
            model_snapshot_hour_local = error_model_hour

            # If no model from error_model_source, try forecast source
            if not pmf_model and error_model_source != str(args.forecast_source):
                pmf_model = db_lib.fetch_error_model(
                    db_path,
                    city_key=c.key,
                    month=month,
                    snapshot_hour_local=snapshot_hour_local,
                    source=str(args.forecast_source),
                )
                if pmf_model:
                    model_source = str(args.forecast_source)
                    model_snapshot_hour_local = snapshot_hour_local

            if pmf_model:
                ladder = _build_pmf_ladder(forecast_high, pmf_model["pmf"])
                model_source_type = "error_pmf"
                error_model_n_samples = int(pmf_model["n_samples"])
                error_model_updated_at_utc = pmf_model["updated_at_utc"]
                error_model_missing = False

        # Tier 3: Fallback Gaussian
        if ladder is None:
            pmf_high = _build_fallback_gaussian_pmf(forecast_high, std=4.0)
            ladder = from_pmf(pmf_high, source="fallback_gaussian")
            model_source_type = "fallback_gaussian"

        # Convert ladder PMF to the output format (backward compat with compute_edges.py)
        pmf_high = dict(ladder.pmf)

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

        out: Dict[str, Any] = {
            "city_key": c.key,
            "label": c.label,
            "tz": c.tz,
            "target_date_local": target_date_local,
            "snapshot_hour_local": int(snap["snapshot_hour_local"]),
            "forecast_source": str(args.forecast_source),
            "snapshot_kind": snapshot_kind,
            "fallback_source": fallback_source,
            "forecast_high_f": forecast_high,
            "model_source": model_source_type,
            "error_model_month": month,
            "error_model_n_samples": error_model_n_samples,
            "error_model_updated_at_utc": error_model_updated_at_utc,
            "error_model_source": model_source,
            "error_model_snapshot_hour_local": model_snapshot_hour_local,
            "error_model_missing": error_model_missing,
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

        # Include ML-specific metadata when available
        if ml_quantiles is not None:
            out["ml_quantiles"] = ml_quantiles
        if multi_model_info is not None:
            out["multi_model_info"] = multi_model_info

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
