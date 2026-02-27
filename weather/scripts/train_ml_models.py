#!/usr/bin/env python3
"""
Train ML models for temperature forecast error prediction.

This script:
1. Loads historical forecast/observation data from the database
2. Prepares features for quantile regression
3. Trains LightGBM models using walk-forward validation
4. Applies isotonic calibration for probability accuracy
5. Saves trained models to weather/models/

Walk-forward validation prevents temporal leakage:
- Train on data from dates < validation start
- Validate on held-out future period
- Repeat with expanding window

Usage:
  # Train NYC model with default settings
  python weather/scripts/train_ml_models.py --city NHIGH

  # Train all cities
  python weather/scripts/train_ml_models.py --all

  # Train with custom validation period
  python weather/scripts/train_ml_models.py --city NHIGH --val-months 6
"""

from __future__ import annotations

import argparse
import json
import os
import sqlite3
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from zoneinfo import ZoneInfo

import numpy as np
import yaml

from weather.lib import db as db_lib
from weather.lib.ml_models import (
    QuantileForestModel,
    QuantileModelConfig,
    SimpleFallbackModel,
    prepare_features,
    FEATURE_NAMES,
)


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = ROOT / "weather" / "config" / "cities.yml"
DEFAULT_DB = Path(os.getenv(
    "WEATHER_DB_PATH",
    str(ROOT / "weather" / "data" / "weather_forecast_accuracy.db")
))
DEFAULT_MODEL_DIR = ROOT / "weather" / "models"


def load_cities(config_path: Path) -> List[Dict[str, Any]]:
    """Load city configurations from YAML."""
    data = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    return data.get("cities", [])


def load_weather_features(
    db_path: Path,
    city_key: str,
    dates: List[str],
) -> Dict[str, Dict[str, float]]:
    """
    Load additional weather features (snow, SST, visibility, smoke) for dates.

    Returns dict mapping date -> {snow_depth_inches, sst_f, visibility_min_miles, ...}
    """
    conn = sqlite3.connect(db_path)
    features: Dict[str, Dict[str, float]] = {d: {} for d in dates}

    # Load snow observations
    for date in dates:
        row = conn.execute(
            """
            SELECT snow_depth_inches, swe_inches
            FROM snow_observations
            WHERE city_key = ? AND date_local = ?
            ORDER BY snapshot_time_utc DESC LIMIT 1
            """,
            (city_key, date),
        ).fetchone()
        if row:
            features[date]["snow_depth_inches"] = row[0] if row[0] is not None else 0.0

    # Load SST observations
    for date in dates:
        row = conn.execute(
            """
            SELECT sst_f, sst_anomaly_f
            FROM sst_observations
            WHERE city_key = ? AND date_local = ?
            ORDER BY observation_time_utc DESC LIMIT 1
            """,
            (city_key, date),
        ).fetchone()
        if row:
            features[date]["sst_f"] = row[0] if row[0] is not None else None
            features[date]["sst_anomaly_f"] = row[1] if row[1] is not None else None

    # Load visibility summaries
    for date in dates:
        row = conn.execute(
            """
            SELECT MIN(visibility_miles), SUM(fog_present)
            FROM visibility_observations
            WHERE city_key = ? AND date_local = ?
            """,
            (city_key, date),
        ).fetchone()
        if row and row[0] is not None:
            features[date]["visibility_min_miles"] = row[0]

    # Load smoke observations
    for date in dates:
        row = conn.execute(
            """
            SELECT smoke_present
            FROM smoke_observations
            WHERE city_key = ? AND date_local = ?
            ORDER BY observation_time_utc DESC LIMIT 1
            """,
            (city_key, date),
        ).fetchone()
        if row:
            features[date]["smoke_present"] = bool(row[0])

    conn.close()
    return features


def compute_precip_accumulation(
    dates: List[str],
    precip_by_date: Dict[str, float],
    lookback_days: int = 3,
) -> Dict[str, float]:
    """
    Compute precipitation accumulation over the past N days.

    Args:
        dates: List of date strings in order
        precip_by_date: Dict mapping date -> precip_total_mm
        lookback_days: Number of days to look back

    Returns:
        Dict mapping date -> accumulated precip (mm)
    """
    result = {}
    date_objs = [datetime.strptime(d, "%Y-%m-%d") for d in dates]

    for i, (date_str, date_obj) in enumerate(zip(dates, date_objs)):
        total = 0.0
        for j in range(1, lookback_days + 1):
            lookback_date = (date_obj - timedelta(days=j)).strftime("%Y-%m-%d")
            total += precip_by_date.get(lookback_date, 0.0)
        result[date_str] = total

    return result


def load_climatology(
    db_path: Path,
    city_key: str,
    forecast_source: str,
) -> Dict[int, float]:
    """
    Compute monthly climatological average forecast high for a city.
    Returns {month: avg_forecast_high}.
    """
    conn = sqlite3.connect(db_path)
    rows = conn.execute(
        """
        SELECT CAST(strftime('%m', target_date_local) AS INTEGER) as month,
               AVG(forecast_high_f)
        FROM forecast_snapshots
        WHERE city_key = ? AND source = ?
        GROUP BY month
        """,
        (city_key, forecast_source),
    ).fetchall()
    conn.close()
    return {int(m): float(avg) for m, avg in rows if avg is not None}


def load_training_data(
    db_path: Path,
    city_key: str,
    start_date: str,
    end_date: str,
    forecast_source: str = "mos_gfs_18z_archive",
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Load training data: forecasts, observations, and features.

    Uses weather_observations_daily for precip/wind when forecast_context_json
    is NULL (which is 100% of MOS GFS archive rows).

    Returns:
        (X, y, dates) where:
        - X: feature array (n_samples, n_features)
        - y: target array (forecast errors)
        - dates: list of date strings
    """
    conn = sqlite3.connect(db_path)

    # Query joined forecast/observation data including weather context
    query = """
    SELECT
        f.target_date_local,
        f.forecast_high_f,
        o.tmax_f,
        f.snapshot_hour_local,
        f.forecast_context_json
    FROM forecast_snapshots f
    JOIN observed_cli o
        ON f.city_key = o.city_key
        AND f.target_date_local = o.date_local
    WHERE f.city_key = ?
        AND f.source = ?
        AND f.target_date_local >= ?
        AND f.target_date_local <= ?
    ORDER BY f.target_date_local
    """

    rows = conn.execute(query, (city_key, forecast_source, start_date, end_date)).fetchall()
    conn.close()

    if not rows:
        return np.array([]), np.array([]), []

    # Extract dates and contexts
    dates = [row[0] for row in rows]
    contexts_by_date = {}

    for row in rows:
        date_str = row[0]
        context_json = row[4]
        context = {}
        if context_json:
            try:
                context = json.loads(context_json)
            except (json.JSONDecodeError, TypeError):
                pass
        contexts_by_date[date_str] = context

    # Load additional weather features from database
    weather_features = load_weather_features(Path(db_lib.default_paths(ROOT).db_path), city_key, dates)

    # Load daily weather observations (precip, wind) from backfilled archive data
    daily_obs = db_lib.fetch_weather_observations_daily(
        Path(db_lib.default_paths(ROOT).db_path), city_key=city_key, dates=dates
    )

    # Build precipitation map from archive data for accumulation
    precip_by_date: Dict[str, float] = {}
    for d in dates:
        obs = daily_obs.get(d, {})
        ctx = contexts_by_date.get(d, {})
        # Prefer forecast_context_json if available, otherwise use archive
        if ctx and "precip_total_mm" in ctx:
            precip_by_date[d] = ctx["precip_total_mm"]
        else:
            precip_by_date[d] = obs.get("precipitation_mm", 0.0)

    # Compute precipitation accumulation
    precip_accum = compute_precip_accumulation(dates, precip_by_date, lookback_days=3)

    # Load climatology for forecast_anomaly
    climatology = load_climatology(Path(db_lib.default_paths(ROOT).db_path), city_key, forecast_source)

    # Build feature arrays
    X_list = []
    y_list = []
    prev_error = 0
    rolling_errors: List[float] = []

    for i, (date_str, forecast_high, observed_tmax, snapshot_hour, context_json) in enumerate(rows):
        # Compute error: observed - forecast
        error = observed_tmax - forecast_high

        # Parse date
        target_date = datetime.strptime(date_str, "%Y-%m-%d")
        month = target_date.month

        # Get forecast context and archive obs for this date
        context = contexts_by_date.get(date_str, {})
        obs = daily_obs.get(date_str, {})

        # Weather context: prefer forecast_context_json, fall back to archive observations
        if context and context.get("precip_prob_max") is not None:
            precip_prob = context["precip_prob_max"] / 100.0
        else:
            # Binary rain proxy from observed precipitation
            precip_prob = 1.0 if obs.get("precipitation_mm", 0) > 0.5 else 0.0

        if context and context.get("cloud_cover_avg") is not None:
            cloud_cover = context["cloud_cover_avg"] / 100.0
        else:
            cloud_cover = 0.5  # Keep default when no data

        if context and context.get("wind_speed_avg") is not None:
            wind_speed = context["wind_speed_avg"]
        else:
            wind_speed = obs.get("windspeed_max_mph", 10.0)

        # Get additional weather features
        wf = weather_features.get(date_str, {})

        # Compute forecast anomaly
        clim_avg = climatology.get(month, 65.0)
        forecast_anomaly = float(forecast_high) - clim_avg

        # Compute features
        features = prepare_features(
            forecast_high=forecast_high,
            target_date=target_date,
            precip_prob=precip_prob,
            cloud_cover=cloud_cover,
            wind_speed=wind_speed,
            prev_day_error=prev_error,
            rolling_7d_bias=np.mean(rolling_errors[-7:]) if rolling_errors else 0.0,
            snow_depth_inches=wf.get("snow_depth_inches", 0.0),
            sst_f=wf.get("sst_f"),
            sst_anomaly_f=wf.get("sst_anomaly_f"),
            visibility_min_miles=wf.get("visibility_min_miles"),
            precip_last_3d_mm=precip_accum.get(date_str, 0.0),
            forecast_anomaly=forecast_anomaly,
        )

        # Convert to array
        X_row = [features[name] for name in FEATURE_NAMES]
        X_list.append(X_row)
        y_list.append(error)

        # Update tracking
        prev_error = error
        rolling_errors.append(float(error))

    return np.array(X_list), np.array(y_list), dates


def load_nbm_features(
    db_path: Path,
    city_key: str,
    dates: List[str],
) -> Dict[str, Dict[str, float]]:
    """
    Load NBM features (spread) for the given dates.

    Returns dict mapping date -> {nbm_spread, p50, ...}
    """
    conn = sqlite3.connect(db_path)

    features = {}
    for date in dates:
        row = conn.execute(
            """
            SELECT p10, p50, p90
            FROM nbm_probabilistic_forecasts
            WHERE city_key = ? AND target_date_local = ? AND metric = 'tmax'
            ORDER BY model_run_utc DESC
            LIMIT 1
            """,
            (city_key, date),
        ).fetchone()

        if row and row[0] is not None and row[2] is not None:
            p10, p50, p90 = row
            features[date] = {
                "nbm_spread": p90 - p10,
                "nbm_p50": p50,
            }

    conn.close()
    return features


def add_nbm_features_to_X(
    X: np.ndarray,
    dates: List[str],
    nbm_features: Dict[str, Dict[str, float]],
) -> np.ndarray:
    """Update NBM spread column in feature matrix."""
    nbm_spread_idx = FEATURE_NAMES.index("nbm_spread")

    for i, date in enumerate(dates):
        if date in nbm_features:
            X[i, nbm_spread_idx] = nbm_features[date].get("nbm_spread", X[i, nbm_spread_idx])

    return X


def walk_forward_split(
    X: np.ndarray,
    y: np.ndarray,
    dates: List[str],
    val_months: int = 6,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, str]:
    """
    Split data for walk-forward validation.

    Train on all data before (today - val_months), validate on the rest.

    Returns:
        (X_train, y_train, X_val, y_val, cutoff_date)
    """
    if len(dates) == 0:
        return np.array([]), np.array([]), np.array([]), np.array([]), ""

    # Find cutoff date
    cutoff = datetime.now() - timedelta(days=val_months * 30)
    cutoff_date = cutoff.strftime("%Y-%m-%d")

    # Split
    train_mask = np.array([d < cutoff_date for d in dates])
    val_mask = ~train_mask

    return (
        X[train_mask],
        y[train_mask],
        X[val_mask],
        y[val_mask],
        cutoff_date,
    )


def evaluate_model(
    model: QuantileForestModel,
    X_val: np.ndarray,
    y_val: np.ndarray,
) -> Dict[str, float]:
    """Evaluate model on validation set."""
    metrics = {}

    if len(y_val) == 0:
        return metrics

    # Predict quantiles
    preds = model.predict_quantiles(X_val)

    # Compute coverage for each quantile
    for q, y_pred in preds.items():
        # What fraction of actuals are below the predicted quantile?
        coverage = np.mean(y_val <= y_pred)
        metrics[f"coverage_q{q:.2f}"] = round(coverage, 4)

        # Calibration error: |coverage - q|
        metrics[f"cal_error_q{q:.2f}"] = round(abs(coverage - q), 4)

    # Median absolute error
    if 0.50 in preds:
        mae = np.mean(np.abs(y_val - preds[0.50]))
        metrics["mae_median"] = round(mae, 2)

    # CRPS (Continuous Ranked Probability Score) - simplified
    if 0.25 in preds and 0.75 in preds:
        iqr_coverage = np.mean((y_val >= preds[0.25]) & (y_val <= preds[0.75]))
        metrics["iqr_coverage"] = round(iqr_coverage, 4)  # Should be ~0.50

    return metrics


def train_city_model(
    db_path: Path,
    city_key: str,
    model_dir: Path,
    forecast_source: str = "open_meteo",
    val_months: int = 6,
    config: Optional[QuantileModelConfig] = None,
) -> Optional[QuantileForestModel]:
    """
    Train a quantile model for a single city.

    Returns trained model or None if insufficient data.
    """
    print(f"\n{'='*60}")
    print(f"Training model for {city_key}")
    print(f"{'='*60}")

    # Load all available data
    # Use 5 years of history
    end_date = datetime.now().strftime("%Y-%m-%d")
    start_date = (datetime.now() - timedelta(days=5*365)).strftime("%Y-%m-%d")

    X, y, dates = load_training_data(
        db_path, city_key, start_date, end_date, forecast_source
    )

    if len(y) < 100:
        print(f"  [skip] Insufficient data: only {len(y)} samples")
        return None

    print(f"  Loaded {len(y)} samples from {dates[0]} to {dates[-1]}")

    # Try to add NBM features
    nbm_features = load_nbm_features(db_path, city_key, dates)
    if nbm_features:
        print(f"  Found NBM features for {len(nbm_features)} dates")
        X = add_nbm_features_to_X(X, dates, nbm_features)

    # Walk-forward split
    X_train, y_train, X_val, y_val, cutoff = walk_forward_split(
        X, y, dates, val_months=val_months
    )

    print(f"  Training samples: {len(y_train)}")
    print(f"  Validation samples: {len(y_val)} (from {cutoff})")

    if len(y_train) < 50:
        print(f"  [skip] Insufficient training data")
        return None

    # Initialize model
    model = QuantileForestModel(config=config, city_key=city_key)

    try:
        # Train
        model.fit(X_train, y_train, X_val, y_val)

        # Evaluate
        metrics = evaluate_model(model, X_val, y_val)
        print(f"\nValidation metrics:")
        for name, value in sorted(metrics.items()):
            print(f"  {name}: {value}")

        # Calibrate if enough validation data
        if len(y_val) >= 50:
            model.calibrate(X_val, y_val)

        # Feature importance
        importance = model.feature_importance()
        print(f"\nFeature importance:")
        for name, imp in sorted(importance.items(), key=lambda x: -x[1])[:5]:
            print(f"  {name}: {imp:.4f}")

        # Save model
        model_path = model_dir / f"{city_key.lower()}_quantile.pkl"
        model.save(model_path)

        return model

    except ImportError as e:
        print(f"  [error] LightGBM not available: {e}")
        print("  Install with: pip install lightgbm")
        return None
    except Exception as e:
        print(f"  [error] Training failed: {e}")
        return None


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Train ML models for temperature forecast error prediction."
    )
    parser.add_argument(
        "--config",
        default=str(DEFAULT_CONFIG),
        help="Path to cities.yml configuration",
    )
    parser.add_argument(
        "--db",
        default=str(DEFAULT_DB),
        help="Path to SQLite database",
    )
    parser.add_argument(
        "--model-dir",
        default=str(DEFAULT_MODEL_DIR),
        help="Directory to save models",
    )
    parser.add_argument(
        "--city",
        default="",
        help="Specific city key to train (e.g., NHIGH)",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Train models for all cities",
    )
    parser.add_argument(
        "--forecast-source",
        default="mos_gfs_18z_archive",
        help="Forecast source to train on (default: mos_gfs_18z_archive)",
    )
    parser.add_argument(
        "--val-months",
        type=int,
        default=6,
        help="Months of data to hold out for validation (default: 6)",
    )
    parser.add_argument(
        "--n-estimators",
        type=int,
        default=500,
        help="Number of trees per quantile model (default: 500)",
    )
    args = parser.parse_args()

    config_path = Path(args.config)
    db_path = Path(args.db)
    model_dir = Path(args.model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)

    if not db_path.exists():
        print(f"[error] Database not found: {db_path}", file=sys.stderr)
        return 1

    cities = load_cities(config_path)

    # Filter cities
    if args.city:
        cities = [c for c in cities if c["key"] == args.city.upper()]
        if not cities:
            print(f"[error] City not found: {args.city}", file=sys.stderr)
            return 1
    elif not args.all:
        # Default to NYC if no city specified
        cities = [c for c in cities if c["key"] == "NHIGH"]
        if not cities:
            print("[error] NYC (NHIGH) not found in config", file=sys.stderr)
            return 1

    # Model config
    model_config = QuantileModelConfig(n_estimators=args.n_estimators)

    print(f"ML Model Training")
    print(f"  Cities: {len(cities)}")
    print(f"  Database: {db_path}")
    print(f"  Model dir: {model_dir}")
    print(f"  Forecast source: {args.forecast_source}")
    print(f"  Validation months: {args.val_months}")

    trained = 0
    for city in cities:
        model = train_city_model(
            db_path=db_path,
            city_key=city["key"],
            model_dir=model_dir,
            forecast_source=args.forecast_source,
            val_months=args.val_months,
            config=model_config,
        )
        if model is not None:
            trained += 1

    print(f"\n{'='*60}")
    print(f"Training complete: {trained}/{len(cities)} models trained")
    print(f"Models saved to: {model_dir}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
