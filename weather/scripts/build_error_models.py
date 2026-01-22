#!/usr/bin/env python3
"""
Part 1 / Step 3:
Build monthly forecast-error distributions (PMFs) per city for a given snapshot hour.

We model integer error:
  error = observed_tmax_f - forecast_high_f

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
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List
from zoneinfo import ZoneInfo

from weather.lib import db as db_lib


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = ROOT / "weather" / "config" / "cities.yml"
DEFAULT_DB = ROOT / "weather" / "data" / "weather.db"
OUT_DIR = ROOT / "weather" / "outputs" / "models"


def _now_utc_iso() -> str:
    return datetime.now(tz=ZoneInfo("UTC")).isoformat()


def load_city_keys(config_path: Path) -> List[str]:
    import yaml

    data = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    return [str(c["key"]).strip() for c in (data.get("cities") or [])]


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
    args = p.parse_args()

    db_path = Path(args.db)
    db_lib.ensure_schema(db_path)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    city_keys = load_city_keys(Path(args.config))
    updated = _now_utc_iso()
    wrote = 0

    for city in city_keys:
        for month in range(1, 13):
            n, counts = db_lib.fetch_joined_errors(
                db_path,
                city_key=city,
                month=month,
                snapshot_hour_local=int(args.snapshot_hour),
            )
            if n < int(args.min_samples):
                continue
            pmf = build_pmf_from_counts(counts, laplace_alpha=float(args.laplace))
            if not pmf:
                continue

            db_lib.upsert_error_model(
                db_path,
                city_key=city,
                month=month,
                snapshot_hour_local=int(args.snapshot_hour),
                n_samples=n,
                pmf=pmf,
                updated_at_utc=updated,
            )

            out = {
                "city_key": city,
                "month": month,
                "snapshot_hour_local": int(args.snapshot_hour),
                "n_samples": n,
                "pmf_error": {str(k): v for k, v in pmf.items()},
                "updated_at_utc": updated,
            }
            out_path = OUT_DIR / f"{city}_m{month:02d}_h{int(args.snapshot_hour):02d}.json"
            out_path.write_text(json.dumps(out, indent=2, sort_keys=True), encoding="utf-8")
            wrote += 1

    print(f"[done] wrote {wrote} monthly error models (db + artifacts).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
