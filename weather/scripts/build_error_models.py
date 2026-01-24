#!/usr/bin/env python3
"""
Part 1 / Step 3:
Build monthly forecast-error distributions (PMFs) per city for a given snapshot hour.

We model integer error:
  error = observed_tmax_f - forecast_high_f

IMPORTANT: Error models must be built per forecast source.

Grouping:
  - by month-of-year (all Januaries combined, etc.)
  - by city_key
  - by snapshot_hour_local (e.g. 6)

Outputs:
  - Upsert into SQLite error_models
  - Write artifacts to weather/outputs/models/{city}_m{MM}_h{HH}.json
"""

from __future__ import annotations

import argparse
import json
import math
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple
from zoneinfo import ZoneInfo

import yaml

from weather.lib import db as db_lib


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = ROOT / "weather" / "config" / "cities.yml"
DEFAULT_DB = Path(os.getenv("WEATHER_DB_PATH", str(ROOT / "weather" / "data" / "weather_forecast_accuracy.db")))
OUT_DIR = ROOT / "weather" / "outputs" / "models"


def _now_utc_iso() -> str:
    return datetime.now(tz=ZoneInfo("UTC")).isoformat()


def load_city_keys(config_path: Path) -> List[str]:
    data = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    return [str(c["key"]).strip() for c in (data.get("cities") or [])]


def _clamp(x: float, lo: float, hi: float) -> float:
    try:
        return max(lo, min(hi, float(x)))
    except Exception:
        return lo


def _moments_from_counts(counts: Dict[int, int]) -> Tuple[float, float]:
    """Return (mean, stddev) for an integer-valued distribution described by counts."""
    if not counts:
        return (0.0, 0.0)
    total = float(sum(max(0, int(v)) for v in counts.values()))
    if total <= 0:
        return (0.0, 0.0)
    mean = sum(float(k) * float(max(0, int(v))) for k, v in counts.items()) / total
    var = sum(((float(k) - mean) ** 2) * float(max(0, int(v))) for k, v in counts.items()) / total
    std = math.sqrt(max(0.0, var))
    return (mean, std)


def _gaussian_pmf_int_support(mean: float, std: float, support_std: float) -> Dict[int, float]:
    """Discrete Gaussian PMF over integer errors within +/- support_std*std."""
    if std <= 0:
        k0 = int(round(mean))
        return {k0: 1.0}
    s = float(std)
    m = float(mean)
    w = max(1.0, float(support_std)) * s
    lo = int(math.floor(m - w))
    hi = int(math.ceil(m + w))
    out: Dict[int, float] = {}
    denom = 2.0 * (s ** 2)
    for k in range(lo, hi + 1):
        out[k] = math.exp(-((float(k) - m) ** 2) / denom)
    z = sum(out.values())
    if z <= 0:
        k0 = int(round(mean))
        return {k0: 1.0}
    return {k: v / z for k, v in out.items()}


def _mix_pmfs(pmf_a: Dict[int, float], pmf_b: Dict[int, float], mix_b: float) -> Dict[int, float]:
    """Return (1-mix_b)*A + mix_b*B over union support, normalized."""
    w = _clamp(mix_b, 0.0, 1.0)
    keys = set(pmf_a.keys()) | set(pmf_b.keys())
    out: Dict[int, float] = {}
    for k in keys:
        out[k] = (1.0 - w) * float(pmf_a.get(k, 0.0)) + w * float(pmf_b.get(k, 0.0))
    z = sum(out.values())
    if z <= 0:
        return {}
    return {k: v / z for k, v in out.items()}


def build_pmf_from_counts(counts: Dict[int, int], laplace_alpha: float = 0.0) -> Dict[int, float]:
    """
    Convert counts into a PMF. Optional Laplace smoothing if alpha>0.
    """
    if not counts:
        return {}
    keys = sorted(counts.keys())
    # Optional smoothing over observed support only (simple + stable)
    if laplace_alpha > 0:
        total = sum(counts.values()) + laplace_alpha * len(keys)
        return {k: (counts[k] + laplace_alpha) / total for k in keys}
    total = float(sum(counts.values()))
    return {k: counts[k] / total for k in keys}


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default=str(DEFAULT_CONFIG))
    p.add_argument("--db", default=str(DEFAULT_DB))
    p.add_argument("--snapshot-hour", type=int, default=int(os.getenv("WEATHER_SNAPSHOT_HOUR_LOCAL", "6")))
    p.add_argument("--laplace", type=float, default=0.0, help="Optional Laplace smoothing alpha (e.g., 0.5).")
    p.add_argument("--min-samples", type=int, default=10, help="Only write models with >= N samples.")
    p.add_argument("--forecast-source", default="nws_hourly_max", help="Which forecast_snapshots.source to use.")
    p.add_argument(
        "--gaussian-mix",
        type=float,
        default=float(os.getenv("WEATHER_ERROR_GAUSS_MIX", "0")),
        help="0..1. Mix weight for Gaussian-smoothed PMF (0 disables).",
    )
    p.add_argument(
        "--gaussian-support-std",
        type=float,
        default=float(os.getenv("WEATHER_ERROR_GAUSS_SUPPORT_STD", "4")),
        help="Gaussian PMF support width in std devs (e.g., 4).",
    )
    args = p.parse_args()

    db_path = Path(args.db)
    db_lib.ensure_schema(db_path)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    city_keys = load_city_keys(Path(args.config))
    updated = _now_utc_iso()
    wrote = 0
    gauss_mix = _clamp(float(args.gaussian_mix), 0.0, 1.0)
    gauss_support_std = _clamp(float(args.gaussian_support_std), 1.0, 12.0)

    for city in city_keys:
        for month in range(1, 13):
            n, counts = db_lib.fetch_joined_errors(
                db_path,
                city_key=city,
                month=month,
                snapshot_hour_local=int(args.snapshot_hour),
                forecast_source=str(args.forecast_source),
            )
            if n < int(args.min_samples):
                continue
            pmf_emp = build_pmf_from_counts(counts, laplace_alpha=float(args.laplace))
            if not pmf_emp:
                continue
            mu, sigma = _moments_from_counts(counts)
            pmf = pmf_emp
            if gauss_mix > 0.0:
                pmf_g = _gaussian_pmf_int_support(mu, sigma, support_std=gauss_support_std)
                pmf = _mix_pmfs(pmf_emp, pmf_g, mix_b=gauss_mix)
            if not pmf:
                continue

            db_lib.upsert_error_model(
                db_path,
                city_key=city,
                month=month,
                snapshot_hour_local=int(args.snapshot_hour),
                source=str(args.forecast_source),
                n_samples=n,
                pmf=pmf,
                updated_at_utc=updated,
            )

            out = {
                "city_key": city,
                "month": month,
                "snapshot_hour_local": int(args.snapshot_hour),
                "forecast_source": str(args.forecast_source),
                "n_samples": n,
                "smoothing": {
                    "laplace_alpha": float(args.laplace),
                    "gaussian_mix": float(gauss_mix),
                    "gaussian_support_std": float(gauss_support_std),
                    "error_mean": float(mu),
                    "error_std": float(sigma),
                },
                "pmf_error": {str(k): v for k, v in pmf.items()},
                "updated_at_utc": updated,
            }
            source_slug = str(args.forecast_source).replace("/", "_")
            out_path = OUT_DIR / f"{city}_m{month:02d}_h{int(args.snapshot_hour):02d}_{source_slug}.json"
            out_path.write_text(json.dumps(out, indent=2, sort_keys=True), encoding="utf-8")
            wrote += 1

    print(f"[done] wrote {wrote} monthly error models (db + artifacts).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
