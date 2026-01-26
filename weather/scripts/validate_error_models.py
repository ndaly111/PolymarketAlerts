#!/usr/bin/env python3
"""
Validate forecast error models using time-based train/test split.

Evaluates:
1. Brier Score - measures prediction accuracy (lower = better, 0 = perfect)
2. Calibration - are predicted probabilities reliable?
3. Log Loss - penalizes confident wrong predictions

Usage:
    python weather/scripts/validate_error_models.py --test-year 2025
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sqlite3
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import yaml

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = ROOT / "weather" / "config" / "cities.yml"
DEFAULT_DB = Path(os.getenv("WEATHER_DB_PATH", str(ROOT / "weather" / "data" / "weather_forecast_accuracy.db")))


def load_city_keys(config_path: Path) -> List[str]:
    data = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    return [str(c["key"]).strip() for c in (data.get("cities") or [])]


def fetch_forecast_observed_pairs(
    db_path: Path,
    city_key: str,
    snapshot_hour_local: int,
    forecast_source: str,
    min_date: Optional[str] = None,
    max_date: Optional[str] = None,
) -> List[Tuple[str, int, int, int]]:
    """
    Fetch (date, month, forecast_high, observed_high) pairs.
    """
    conn = sqlite3.connect(str(db_path))
    cur = conn.cursor()

    query = """
        SELECT
            f.target_date_local,
            CAST(strftime('%m', f.target_date_local) AS INTEGER) as month,
            f.forecast_high_f,
            o.tmax_f
        FROM forecast_snapshots f
        JOIN observed_cli o
            ON f.city_key = o.city_key
            AND f.target_date_local = o.date_local
        WHERE f.city_key = ?
            AND f.snapshot_hour_local = ?
            AND f.source = ?
    """
    params = [city_key, snapshot_hour_local, forecast_source]

    if min_date:
        query += " AND f.target_date_local >= ?"
        params.append(min_date)
    if max_date:
        query += " AND f.target_date_local < ?"
        params.append(max_date)

    query += " ORDER BY f.target_date_local"

    cur.execute(query, params)
    rows = cur.fetchall()
    conn.close()
    return rows


def build_error_pmf_from_data(
    pairs: List[Tuple[str, int, int, int]],
    month: int,
) -> Tuple[Dict[int, float], float, float, int]:
    """
    Build error PMF from train data for a specific month.
    Returns (pmf, mean, std, n_samples).
    """
    errors = [obs - fcst for (date, m, fcst, obs) in pairs if m == month]
    if not errors:
        return {}, 0.0, 0.0, 0

    n = len(errors)
    mean = sum(errors) / n
    var = sum((e - mean) ** 2 for e in errors) / n
    std = math.sqrt(var) if var > 0 else 0.001

    # Build Gaussian PMF with tail cutoff (matching production approach)
    pmf = _gaussian_pmf_with_tail_cutoff(mean, std, min_tail_prob=0.01)

    return pmf, mean, std, n


def _gaussian_pmf_with_tail_cutoff(
    mean: float, std: float, min_tail_prob: float = 0.01
) -> Dict[int, float]:
    """Build discrete Gaussian PMF extending tails until cumulative prob < min_tail_prob."""
    if std <= 0:
        return {int(round(mean)): 1.0}

    center = int(round(mean))
    denom = 2.0 * (std ** 2)

    def gaussian_density(k: int) -> float:
        return math.exp(-((float(k) - mean) ** 2) / denom)

    out: Dict[int, float] = {center: gaussian_density(center)}

    # Extend left
    k = center - 1
    while True:
        density = gaussian_density(k)
        out[k] = density
        total = sum(out.values())
        cumulative_left = sum(out[j] for j in out if j <= k) / total
        if cumulative_left < min_tail_prob or k < center - 50:
            break
        k -= 1

    # Extend right
    k = center + 1
    while True:
        density = gaussian_density(k)
        out[k] = density
        total = sum(out.values())
        cumulative_right = sum(out[j] for j in out if j >= k) / total
        if cumulative_right < min_tail_prob or k > center + 50:
            break
        k += 1

    # Normalize
    total = sum(out.values())
    return {k: v / total for k, v in sorted(out.items())}


def compute_prob_over_threshold(
    pmf_error: Dict[int, float],
    forecast_high: int,
    threshold: int,
) -> float:
    """
    Compute P(observed >= threshold) given forecast and error PMF.

    observed = forecast + error
    P(observed >= threshold) = P(error >= threshold - forecast)
    """
    needed_error = threshold - forecast_high
    prob = sum(p for e, p in pmf_error.items() if e >= needed_error)
    return prob


def evaluate_calibration(
    test_pairs: List[Tuple[str, int, int, int]],
    models: Dict[int, Dict[int, float]],  # month -> pmf_error
    thresholds: List[int] = [70, 75, 80, 85, 90],
) -> Dict:
    """
    Evaluate model calibration on test data.

    For each test sample, compute predicted P(observed >= threshold) for various thresholds,
    then compare to actual outcomes.
    """
    # Collect (predicted_prob, actual_outcome) pairs
    predictions = []  # List of (pred_prob, actual_binary)

    # Also collect by probability bin for reliability diagram
    bins = defaultdict(lambda: {"count": 0, "sum_pred": 0.0, "sum_actual": 0})

    for date, month, forecast, observed in test_pairs:
        if month not in models:
            continue
        pmf = models[month]

        for thresh in thresholds:
            pred_prob = compute_prob_over_threshold(pmf, forecast, thresh)
            actual = 1 if observed >= thresh else 0

            predictions.append((pred_prob, actual))

            # Bin by predicted probability (0.1 width bins)
            bin_idx = min(9, int(pred_prob * 10))
            bins[bin_idx]["count"] += 1
            bins[bin_idx]["sum_pred"] += pred_prob
            bins[bin_idx]["sum_actual"] += actual

    if not predictions:
        return {"error": "No predictions generated"}

    # Compute Brier Score
    brier = sum((p - a) ** 2 for p, a in predictions) / len(predictions)

    # Compute Log Loss (with clipping to avoid log(0))
    eps = 1e-10
    log_loss = -sum(
        a * math.log(max(eps, p)) + (1 - a) * math.log(max(eps, 1 - p))
        for p, a in predictions
    ) / len(predictions)

    # Compute calibration by bin
    calibration = {}
    for bin_idx in sorted(bins.keys()):
        b = bins[bin_idx]
        if b["count"] > 0:
            avg_pred = b["sum_pred"] / b["count"]
            avg_actual = b["sum_actual"] / b["count"]
            calibration[f"{bin_idx * 10}-{(bin_idx + 1) * 10}%"] = {
                "n": b["count"],
                "avg_predicted": round(avg_pred, 3),
                "avg_actual": round(avg_actual, 3),
                "calibration_error": round(avg_actual - avg_pred, 3),
            }

    # Compute Expected Calibration Error (ECE)
    ece = 0.0
    total_samples = len(predictions)
    for b in bins.values():
        if b["count"] > 0:
            avg_pred = b["sum_pred"] / b["count"]
            avg_actual = b["sum_actual"] / b["count"]
            ece += (b["count"] / total_samples) * abs(avg_actual - avg_pred)

    return {
        "n_predictions": len(predictions),
        "brier_score": round(brier, 4),
        "log_loss": round(log_loss, 4),
        "expected_calibration_error": round(ece, 4),
        "calibration_by_bin": calibration,
    }


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default=str(DEFAULT_CONFIG))
    p.add_argument("--db", default=str(DEFAULT_DB))
    p.add_argument("--snapshot-hour", type=int, default=int(os.getenv("WEATHER_SNAPSHOT_HOUR_LOCAL", "12")))
    p.add_argument("--forecast-source", default="mos_gfs_18z_archive")
    p.add_argument("--test-year", type=int, default=2025, help="Year to hold out for testing")
    p.add_argument("--output", default=None, help="Output JSON file path")
    args = p.parse_args()

    db_path = Path(args.db)
    city_keys = load_city_keys(Path(args.config))

    test_start = f"{args.test_year}-01-01"

    print("=" * 60)
    print("WEATHER ERROR MODEL VALIDATION")
    print(f"Test year: {args.test_year}")
    print(f"Forecast source: {args.forecast_source}")
    print(f"Snapshot hour: {args.snapshot_hour}")
    print("=" * 60)
    print()

    all_results = {}

    for city in city_keys:
        print(f"\n--- {city} ---")

        # Fetch all data
        train_pairs = fetch_forecast_observed_pairs(
            db_path, city, args.snapshot_hour, args.forecast_source,
            max_date=test_start,
        )
        test_pairs = fetch_forecast_observed_pairs(
            db_path, city, args.snapshot_hour, args.forecast_source,
            min_date=test_start,
        )

        print(f"  Train samples: {len(train_pairs)}")
        print(f"  Test samples: {len(test_pairs)}")

        if not train_pairs or not test_pairs:
            print("  Skipping (insufficient data)")
            continue

        # Build models from train data (one per month)
        models = {}
        for month in range(1, 13):
            pmf, mean, std, n = build_error_pmf_from_data(train_pairs, month)
            if n >= 10:
                models[month] = pmf
                print(f"  Month {month:2d}: n={n:3d}, mean={mean:+.2f}, std={std:.2f}")

        # Evaluate on test data
        results = evaluate_calibration(test_pairs, models)
        all_results[city] = results

        print(f"\n  Brier Score: {results.get('brier_score', 'N/A')}")
        print(f"  Log Loss: {results.get('log_loss', 'N/A')}")
        print(f"  ECE: {results.get('expected_calibration_error', 'N/A')}")

        # Show calibration by bin
        cal = results.get("calibration_by_bin", {})
        if cal:
            print("\n  Calibration by predicted probability bin:")
            print("  Bin        | N    | Pred  | Actual | Error")
            print("  " + "-" * 45)
            for bin_name, data in cal.items():
                print(f"  {bin_name:10s} | {data['n']:4d} | {data['avg_predicted']:.1%} | {data['avg_actual']:.1%}  | {data['calibration_error']:+.1%}")

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    brier_scores = [r["brier_score"] for r in all_results.values() if "brier_score" in r]
    if brier_scores:
        avg_brier = sum(brier_scores) / len(brier_scores)
        print(f"Average Brier Score: {avg_brier:.4f}")
        print("  (0 = perfect, <0.25 = good, <0.1 = excellent)")

    ece_scores = [r["expected_calibration_error"] for r in all_results.values() if "expected_calibration_error" in r]
    if ece_scores:
        avg_ece = sum(ece_scores) / len(ece_scores)
        print(f"Average ECE: {avg_ece:.4f}")
        print("  (0 = perfectly calibrated, <0.05 = well calibrated)")

    # Save output
    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(all_results, indent=2), encoding="utf-8")
        print(f"\nResults saved to: {out_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
