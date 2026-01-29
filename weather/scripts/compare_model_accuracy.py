#!/usr/bin/env python3
"""
Compare accuracy of different forecast models against CLI settlements.

Run this periodically (weekly) to see which model is performing best.
Once we have enough data (30+ days), we can make a data-driven decision
on which model to use for live trading.

Metrics:
- MAE (Mean Absolute Error): Average of |forecast - actual|
- Bias: Average of (forecast - actual), positive = forecasts too hot
- RMSE: Root mean squared error
"""

from __future__ import annotations

import argparse
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Tuple

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DB = ROOT / "weather" / "data" / "weather_forecast_accuracy.db"


def get_model_accuracy(db_path: Path, min_days: int = 7) -> Dict[str, Dict]:
    """
    Compare forecast accuracy across models.

    Returns dict of source -> {
        'days': int,
        'mae': float,
        'bias': float,
        'rmse': float,
        'by_city': {city_key: {'mae': float, 'bias': float, 'count': int}}
    }
    """
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()

    # Get all sources with matched forecasts and observations
    # Use only the earliest snapshot per (city, date, source) for fair comparison
    cur.execute("""
        WITH earliest_snapshots AS (
            SELECT fs.*
            FROM forecast_snapshots fs
            INNER JOIN (
                SELECT city_key, target_date_local, source, MIN(snapshot_hour_local) as min_hour
                FROM forecast_snapshots
                GROUP BY city_key, target_date_local, source
            ) earliest
            ON fs.city_key = earliest.city_key
            AND fs.target_date_local = earliest.target_date_local
            AND fs.source = earliest.source
            AND fs.snapshot_hour_local = earliest.min_hour
        )
        SELECT
            es.source,
            es.city_key,
            es.target_date_local,
            es.forecast_high_f,
            oc.tmax_f as observed
        FROM earliest_snapshots es
        INNER JOIN observed_cli oc
            ON es.city_key = oc.city_key
            AND es.target_date_local = oc.date_local
        WHERE oc.tmax_f IS NOT NULL
        ORDER BY es.source, es.city_key, es.target_date_local
    """)

    rows = cur.fetchall()
    conn.close()

    # Aggregate by source
    results: Dict[str, Dict] = {}

    for row in rows:
        source = row["source"]
        city = row["city_key"]
        forecast = row["forecast_high_f"]
        observed = row["observed"]
        error = forecast - observed

        if source not in results:
            results[source] = {
                "errors": [],
                "by_city": {},
                "dates": set(),
            }

        results[source]["errors"].append(error)
        results[source]["dates"].add(row["target_date_local"])

        if city not in results[source]["by_city"]:
            results[source]["by_city"][city] = {"errors": []}
        results[source]["by_city"][city]["errors"].append(error)

    # Calculate metrics
    final_results = {}
    for source, data in results.items():
        errors = data["errors"]
        if len(data["dates"]) < min_days:
            continue

        mae = sum(abs(e) for e in errors) / len(errors)
        bias = sum(errors) / len(errors)
        rmse = (sum(e**2 for e in errors) / len(errors)) ** 0.5

        by_city = {}
        for city, city_data in data["by_city"].items():
            city_errors = city_data["errors"]
            by_city[city] = {
                "count": len(city_errors),
                "mae": sum(abs(e) for e in city_errors) / len(city_errors),
                "bias": sum(city_errors) / len(city_errors),
            }

        final_results[source] = {
            "days": len(data["dates"]),
            "total_forecasts": len(errors),
            "mae": round(mae, 2),
            "bias": round(bias, 2),
            "rmse": round(rmse, 2),
            "by_city": by_city,
        }

    return final_results


def print_comparison(results: Dict[str, Dict]) -> None:
    """Print a formatted comparison of model accuracy."""
    if not results:
        print("No data available for comparison yet.")
        print("Run collect_multi_model_forecasts.py daily and wait for CLI settlements.")
        return

    print("\n" + "=" * 70)
    print("WEATHER FORECAST MODEL ACCURACY COMPARISON")
    print("=" * 70)

    # Sort by MAE (lower is better)
    sorted_sources = sorted(results.items(), key=lambda x: x[1]["mae"])

    print(f"\n{'Model':<25} {'Days':>6} {'MAE':>8} {'Bias':>8} {'RMSE':>8}")
    print("-" * 60)

    for source, data in sorted_sources:
        print(f"{source:<25} {data['days']:>6} {data['mae']:>8.2f} {data['bias']:>+8.2f} {data['rmse']:>8.2f}")

    print("\n" + "-" * 60)
    print("MAE = Mean Absolute Error (lower is better)")
    print("Bias = Average error (+ = forecasts too hot, - = too cold)")
    print("RMSE = Root Mean Squared Error (penalizes large errors)")

    # Best model recommendation
    if sorted_sources:
        best = sorted_sources[0]
        print(f"\n*** BEST MODEL: {best[0]} (MAE: {best[1]['mae']}°F) ***")

        if best[1]["days"] >= 30:
            print(f"    Recommendation: Consider switching to {best[0]}")
        else:
            print(f"    Need more data ({best[1]['days']}/30 days minimum)")

    # Per-city breakdown for top 3 models
    print("\n" + "=" * 70)
    print("PER-CITY BREAKDOWN (Top 3 Models)")
    print("=" * 70)

    for source, data in sorted_sources[:3]:
        print(f"\n{source}:")
        cities_sorted = sorted(data["by_city"].items(), key=lambda x: x[1]["mae"])
        for city, city_data in cities_sorted:
            print(f"  {city:<12} MAE: {city_data['mae']:>5.1f}°F  Bias: {city_data['bias']:>+5.1f}°F  (n={city_data['count']})")


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare forecast model accuracy")
    parser.add_argument("--db", default=str(DEFAULT_DB))
    parser.add_argument("--min-days", type=int, default=7, help="Minimum days of data required")
    parser.add_argument("--json", action="store_true", help="Output as JSON")
    args = parser.parse_args()

    results = get_model_accuracy(Path(args.db), args.min_days)

    if args.json:
        import json
        print(json.dumps(results, indent=2))
    else:
        print_comparison(results)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
