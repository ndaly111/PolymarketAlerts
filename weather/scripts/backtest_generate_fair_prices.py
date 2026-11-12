#!/usr/bin/env python3
"""Generate historical fair-price artifacts without look-ahead bias."""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from weather.lib import db as db_lib
from weather.lib.fair import normalize_pmf, shift_pmf, summarize_pmf
from weather.scripts.build_error_models import build_pmf_from_counts


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = ROOT / "weather" / "config" / "cities.yml"
DEFAULT_DB = ROOT / "weather" / "data" / "weather.db"
OUT_BASE = ROOT / "weather" / "outputs" / "backtest_fair_prices"
SNAPSHOT_HOUR_LOCAL = int(os.getenv("WEATHER_SNAPSHOT_HOUR_LOCAL", "5"))


@dataclass(frozen=True)
class City:
    key: str
    label: str
    tz: str


def load_cities(config_path: Path) -> List[City]:
    import yaml

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


def parse_date(value: str) -> date:
    return datetime.strptime(value, "%Y-%m-%d").date()


def fetch_available_dates(
    db_path: Path,
    *,
    city_key: str,
    start_date: Optional[date],
    end_date: Optional[date],
) -> List[str]:
    db_lib.ensure_schema(db_path)
    with db_lib.connect(db_path) as conn:
        rows = conn.execute(
            """
            SELECT DISTINCT target_date_local
            FROM forecast_snapshots
            WHERE city_key=?
            ORDER BY target_date_local
            """,
            (city_key,),
        ).fetchall()

    dates: List[str] = []
    for (d,) in rows:
        d_parsed = parse_date(d)
        if start_date and d_parsed < start_date:
            continue
        if end_date and d_parsed > end_date:
            continue
        dates.append(d)
    return dates


def build_error_pmf_as_of(
    db_path: Path,
    *,
    city_key: str,
    month: int,
    snapshot_hour_local: int,
    cutoff_date_local: str,
    min_samples: int,
    laplace_alpha: float,
) -> Optional[Tuple[int, Dict[int, float]]]:
    """Build a month+hour error PMF using only dates before cutoff_date_local."""
    db_lib.ensure_schema(db_path)
    with db_lib.connect(db_path) as conn:
        rows = conn.execute(
            """
            SELECT o.date_local, o.tmax_f, f.forecast_high_f
            FROM observed_cli o
            JOIN forecast_snapshots f
              ON f.city_key=o.city_key
             AND f.target_date_local=o.date_local
             AND f.snapshot_hour_local=?
            WHERE o.city_key=?
              AND o.date_local < ?
              AND CAST(strftime('%m', o.date_local) AS INTEGER)=?
            """,
            (
                int(snapshot_hour_local),
                city_key,
                cutoff_date_local,
                int(month),
            ),
        ).fetchall()

    counts: Dict[int, int] = {}
    for (_, obs, fcst) in rows:
        err = int(obs) - int(fcst)
        counts[err] = counts.get(err, 0) + 1

    n_samples = len(rows)
    if n_samples < int(min_samples):
        return None
    pmf = build_pmf_from_counts(counts, laplace_alpha=float(laplace_alpha))
    if not pmf:
        return None
    return (n_samples, pmf)


def summarize_pmf_high(pmf_high: Dict[int, float]) -> Dict[str, float | int]:
    summary = summarize_pmf(pmf_high)
    return {
        "mean": summary.mean,
        "p10": summary.p10,
        "p50": summary.p50,
        "p90": summary.p90,
    }


def generate_fair_price_artifact(
    db_path: Path,
    *,
    city: City,
    target_date_local: str,
    snapshot_hour_local: int,
    min_samples: int,
    laplace_alpha: float,
    overwrite: bool,
    out_base: Path,
) -> bool:
    snap = db_lib.fetch_forecast_snapshot(
        db_path,
        city_key=city.key,
        target_date_local=target_date_local,
        snapshot_hour_local=snapshot_hour_local,
    )
    if not snap:
        return False

    month = int(target_date_local.split("-")[1])
    error_model = build_error_pmf_as_of(
        db_path,
        city_key=city.key,
        month=month,
        snapshot_hour_local=snapshot_hour_local,
        cutoff_date_local=target_date_local,
        min_samples=min_samples,
        laplace_alpha=laplace_alpha,
    )
    if not error_model:
        return False

    forecast_high = int(snap["forecast_high_f"])
    n_samples, pmf_error = error_model
    pmf_high = normalize_pmf(shift_pmf(pmf_error, forecast_high))
    summary = summarize_pmf_high(pmf_high)

    out = {
        "city_key": city.key,
        "label": city.label,
        "tz": city.tz,
        "target_date_local": target_date_local,
        "snapshot_hour_local": int(snapshot_hour_local),
        "forecast_high_f": forecast_high,
        "error_model_month": month,
        "error_model_n_samples": int(n_samples),
        # Backtest models are built "as of" the target date to avoid look-ahead bias.
        "error_model_cutoff_date_local": target_date_local,
        "laplace_alpha": float(laplace_alpha),
        # Keep a timestamp for traceability (this is the snapshot time, not a model update time).
        "snapshot_time_utc": snap["snapshot_time_utc"],
        "snapshot_meta": {
            "source": snap["source"],
            "snapshot_time_utc": snap["snapshot_time_utc"],
            "forecast_url": snap["forecast_url"],
            "points_url": snap["points_url"],
        },
        "pmf_high_f": {str(k): v for k, v in sorted(pmf_high.items())},
        "summary": summary,
    }

    out_dir = out_base / target_date_local
    out_path = out_dir / f"{city.key}.json"
    if out_path.exists() and not overwrite:
        return False
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2, sort_keys=True), encoding="utf-8")
    return True


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default=str(DEFAULT_CONFIG))
    p.add_argument("--db", default=str(DEFAULT_DB))
    p.add_argument("--city", default="", help="Optional city_key filter (default: all)")
    p.add_argument("--start-date", default="", help="Optional start date YYYY-MM-DD")
    p.add_argument("--end-date", default="", help="Optional end date YYYY-MM-DD")
    p.add_argument("--min-samples", type=int, default=25)
    p.add_argument("--laplace-alpha", type=float, default=0.0)
    p.add_argument("--overwrite", action="store_true", help="Overwrite existing artifacts.")
    args = p.parse_args()

    start_date = parse_date(args.start_date) if args.start_date else None
    end_date = parse_date(args.end_date) if args.end_date else None

    db_path = Path(args.db)
    cities = load_cities(Path(args.config))
    wrote = 0

    for c in cities:
        if args.city and c.key != args.city:
            continue

        dates = fetch_available_dates(
            db_path,
            city_key=c.key,
            start_date=start_date,
            end_date=end_date,
        )
        for target_date_local in dates:
            wrote += int(
                generate_fair_price_artifact(
                    db_path,
                    city=c,
                    target_date_local=target_date_local,
                    snapshot_hour_local=SNAPSHOT_HOUR_LOCAL,
                    min_samples=int(args.min_samples),
                    laplace_alpha=float(args.laplace_alpha),
                    overwrite=bool(args.overwrite),
                    out_base=OUT_BASE,
                )
            )

    print(f"[done] wrote {wrote} backtest fair-price artifacts under {OUT_BASE}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
