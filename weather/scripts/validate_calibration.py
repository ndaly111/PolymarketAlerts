#!/usr/bin/env python3
"""
Validate calibration of the probability pipeline against historical data.

For each city and date with observed_cli data + kalshi_weather_market_snapshots:
1. Build probability ladder using the new pipeline
2. For each threshold contract, compute model_prob = ladder.prob_above(threshold)
3. Record (model_prob, actual_outcome) where actual_outcome = 1 if observed >= threshold
4. Compute calibration metrics: ECE, Brier score, per-bin predicted vs actual

Usage:
    python -m weather.scripts.validate_calibration
    python -m weather.scripts.validate_calibration --city NHIGH --start 2025-01-20 --end 2025-02-10
"""

from __future__ import annotations

import argparse
import sqlite3
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from weather.lib.probability_pipeline import build_trading_ladder
from weather.lib.kalshi_weather import parse_event_spec_from_ticker
from weather.backtesting.metrics import (
    brier_score,
    calibration_curve,
    calibration_error,
)


def load_city_keys(config_path: Path) -> List[str]:
    import yaml
    data = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    return [c["key"] for c in data.get("cities", [])]


def get_dates_with_observed(db_path: Path, city_key: str, start: str, end: str) -> List[Tuple[str, int]]:
    """Get dates where we have observed CLI data for a city."""
    conn = sqlite3.connect(str(db_path))
    rows = conn.execute(
        """
        SELECT date_local, tmax_f
        FROM observed_cli
        WHERE city_key = ? AND date_local >= ? AND date_local <= ?
        ORDER BY date_local;
        """,
        (city_key, start, end),
    ).fetchall()
    conn.close()
    return [(r[0], int(r[1])) for r in rows]


def get_market_tickers_for_date(db_path: Path, city_key: str, date_local: str) -> List[Dict[str, Any]]:
    """Get all Kalshi market tickers and prices for a city/date."""
    from weather.lib import db as db_lib

    snap_time = db_lib.fetch_latest_kalshi_weather_snapshot_time(
        db_path, city_key=city_key, target_date_local=date_local,
    )
    if not snap_time:
        return []

    return list(db_lib.fetch_kalshi_weather_markets_at_snapshot(
        db_path,
        snapshot_time_utc=snap_time,
        city_key=city_key,
        target_date_local=date_local,
    ))


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate probability pipeline calibration")
    parser.add_argument("--db", default=str(ROOT / "weather" / "data" / "weather_forecast_accuracy.db"))
    parser.add_argument("--config", default=str(ROOT / "weather" / "config" / "cities.yml"))
    parser.add_argument("--city", help="Single city to validate (default: all)")
    parser.add_argument("--start", default="2025-01-01", help="Start date YYYY-MM-DD")
    parser.add_argument("--end", default="2026-02-10", help="End date YYYY-MM-DD")
    parser.add_argument("--forecast-source", default="mos_gfs_12z")
    args = parser.parse_args()

    db_path = Path(args.db)
    if not db_path.exists():
        print(f"ERROR: Database not found: {db_path}")
        return 1

    cities = [args.city] if args.city else load_city_keys(Path(args.config))

    # Collect (predicted_prob, actual_outcome) pairs
    all_threshold_preds: List[float] = []
    all_threshold_outcomes: List[int] = []
    all_bucket_preds: List[float] = []
    all_bucket_outcomes: List[int] = []

    total_dates = 0
    total_contracts = 0

    for city_key in cities:
        dates_observed = get_dates_with_observed(db_path, city_key, args.start, args.end)
        if not dates_observed:
            print(f"  {city_key}: no observed data in range")
            continue

        city_threshold_hits = 0
        city_threshold_total = 0

        for date_local, observed_tmax in dates_observed:
            # Build ladder
            lr = build_trading_ladder(
                db_path,
                city_key=city_key,
                target_date=date_local,
                forecast_source=args.forecast_source,
            )
            if lr is None:
                continue

            # Get market tickers
            markets = get_market_tickers_for_date(db_path, city_key, date_local)
            if not markets:
                continue

            total_dates += 1

            for market in markets:
                ticker = market.get("market_ticker", "")
                spec = parse_event_spec_from_ticker(ticker)
                if spec is None:
                    continue

                total_contracts += 1

                if spec.kind == "ge":
                    # Threshold contract
                    model_prob = lr.ladder.prob_above(spec.a)
                    actual = 1 if observed_tmax >= spec.a else 0
                    all_threshold_preds.append(model_prob)
                    all_threshold_outcomes.append(actual)
                    if actual == 1:
                        city_threshold_hits += 1
                    city_threshold_total += 1

                elif spec.kind == "between" and spec.b is not None:
                    # Bucket contract
                    model_prob = lr.ladder.prob_between(spec.a, spec.b)
                    actual = 1 if spec.a <= observed_tmax <= spec.b else 0
                    all_bucket_preds.append(model_prob)
                    all_bucket_outcomes.append(actual)

        if city_threshold_total > 0:
            print(f"  {city_key}: {len(dates_observed)} dates, "
                  f"{city_threshold_total} threshold contracts evaluated")

    # --- Report ---
    print("\n" + "=" * 60)
    print("CALIBRATION VALIDATION REPORT")
    print(f"Date range: {args.start} to {args.end}")
    print(f"Forecast source: {args.forecast_source}")
    print(f"Total city-dates: {total_dates}, Total contracts: {total_contracts}")
    print("=" * 60)

    # Threshold contracts
    if all_threshold_preds:
        print(f"\n--- THRESHOLD CONTRACTS ({len(all_threshold_preds)} evaluations) ---")
        bs = brier_score(all_threshold_preds, all_threshold_outcomes)
        ece = calibration_error(all_threshold_preds, all_threshold_outcomes, n_bins=5)
        print(f"  Brier Score:  {bs:.4f}  (perfect=0, random=0.25)")
        print(f"  ECE:          {ece:.4f}  (perfect=0)")

        centers, actuals, counts = calibration_curve(
            all_threshold_preds, all_threshold_outcomes, n_bins=5,
        )
        print(f"\n  {'Predicted':>10} {'Actual':>10} {'Count':>8}")
        print(f"  {'-'*10} {'-'*10} {'-'*8}")
        for c, a, n in zip(centers, actuals, counts):
            marker = " ***" if abs(c - a) > 0.15 else ""
            print(f"  {c:10.1%} {a:10.1%} {n:8d}{marker}")

        # Pass/fail
        if ece < 0.10:
            print(f"\n  PASS: ECE {ece:.4f} < 0.10 threshold")
        else:
            print(f"\n  WARN: ECE {ece:.4f} >= 0.10 — calibration needs improvement")
    else:
        print("\n  No threshold contract evaluations available")

    # Bucket contracts
    if all_bucket_preds:
        print(f"\n--- BUCKET CONTRACTS ({len(all_bucket_preds)} evaluations) ---")
        bs = brier_score(all_bucket_preds, all_bucket_outcomes)
        ece = calibration_error(all_bucket_preds, all_bucket_outcomes, n_bins=5)
        print(f"  Brier Score:  {bs:.4f}")
        print(f"  ECE:          {ece:.4f}")

        centers, actuals, counts = calibration_curve(
            all_bucket_preds, all_bucket_outcomes, n_bins=5,
        )
        print(f"\n  {'Predicted':>10} {'Actual':>10} {'Count':>8}")
        print(f"  {'-'*10} {'-'*10} {'-'*8}")
        for c, a, n in zip(centers, actuals, counts):
            print(f"  {c:10.1%} {a:10.1%} {n:8d}")
    else:
        print("\n  No bucket contract evaluations available")

    # Bias detection
    if all_threshold_preds:
        avg_pred = sum(all_threshold_preds) / len(all_threshold_preds)
        avg_actual = sum(all_threshold_outcomes) / len(all_threshold_outcomes)
        bias = avg_actual - avg_pred
        print(f"\n--- BIAS CHECK ---")
        print(f"  Avg predicted prob:   {avg_pred:.3f}")
        print(f"  Avg actual hit rate:  {avg_actual:.3f}")
        print(f"  Bias (actual - pred): {bias:+.3f}")
        if abs(bias) > 0.05:
            print(f"  WARN: Bias {bias:+.3f} exceeds +/-0.05")
        else:
            print(f"  OK: Bias within +/-0.05")

    print("\n" + "=" * 60)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
