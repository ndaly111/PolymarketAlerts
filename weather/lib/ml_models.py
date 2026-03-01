"""
ML-based temperature forecast error models using LightGBM quantile regression.

This module provides probabilistic temperature forecasting by predicting
quantiles of the forecast error distribution. Key features:

1. **Quantile Regression**: Predicts multiple quantiles (0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95)
   of the error distribution instead of just the mean.

2. **Feature Engineering**: Uses meteorological context (NBM spread, precipitation,
   cloud cover) along with temporal features (month, day of year) and recent
   forecast performance (bias tracking).

3. **Calibration**: Post-hoc isotonic regression calibration ensures predicted
   probabilities are well-calibrated.

Predictand: forecast_error = observed_tmax - forecast_tmax

Features:
- month, day_of_year (seasonality)
- forecast_high (point forecast value)
- nbm_spread (p90 - p10, model uncertainty)
- lead_hours (time to settlement)
- precip_prob, cloud_cover, wind_speed (synoptic context)
- prev_day_error, rolling_7d_bias (recent performance)

Usage:
    from weather.lib.ml_models import QuantileForestModel

    model = QuantileForestModel.load("weather/models/nyc_quantile.pkl")
    percentiles = model.predict_percentiles(features)
"""

from __future__ import annotations

import json
import pickle
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import math

import numpy as np

try:
    import lightgbm as lgb
    HAS_LIGHTGBM = True
except ImportError:
    HAS_LIGHTGBM = False

try:
    from sklearn.isotonic import IsotonicRegression
    from sklearn.calibration import calibration_curve
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False


# Quantiles to predict
DEFAULT_QUANTILES = [0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95]

# Feature names in order
# v2: Removed lead_hours (always constant 18), smoke_present (only 5 obs).
#     Added forecast_anomaly (deviation from climatological monthly avg).
# v3: Added 15 new features across 4 phases:
#     Phase 1 (multi-model spread), Phase 2 (upper-air), Phase 3 (teleconnections),
#     Phase 4 (morning temperature). Total: 32 features.
FEATURE_NAMES = [
    # Core temporal features
    "month",
    "day_of_year",
    "forecast_high",
    "nbm_spread",
    # Weather context features (from archive or forecast_context_json)
    "precip_prob",
    "cloud_cover",
    "wind_speed",
    # Recent performance features
    "prev_day_error",
    "rolling_7d_bias",
    # Snow features
    "snow_depth_inches",
    "has_snow",
    # SST features
    "sst_f",
    "sst_anomaly_f",
    # Visibility features
    "visibility_min_miles",
    # Soil moisture proxy
    "precip_last_3d_mm",
    # Forecast anomaly: forecast_high minus climatological monthly avg
    "forecast_anomaly",
    # Phase 1: Multi-model spread (from forecast_snapshots)
    "model_spread_f",    # max - min across HRRR/ECMWF/GFS/NWS
    "model_mean_f",      # mean forecast across all available models
    "ecmwf_f",           # ECMWF forecast specifically (best global model)
    "n_models",          # count of models with data (data quality signal)
    # Phase 2: Upper-air features (from upper_air_snapshots via Open-Meteo)
    "z500",              # 500mb geopotential height (m) — jetstream indicator
    "t850",              # 850mb temperature (°C) — lower-troposphere predictor
    "z500_anomaly",      # z500 minus monthly climatological mean
    "dewpoint_f",        # surface dewpoint (°F) — limits afternoon high in humid cities
    # Phase 3: Teleconnection indices (from climate_indices)
    "nao",               # North Atlantic Oscillation (monthly)
    "ao",                # Arctic Oscillation (monthly) — polar vortex
    "pna",               # Pacific North American pattern (monthly)
    "mei_enso",          # Multivariate ENSO Index (monthly)
    "pdo",               # Pacific Decadal Oscillation (monthly)
    "mjo_amplitude",     # Madden-Julian Oscillation amplitude
    # Phase 4: Morning temperature (from intraday_observations)
    "morning_temp_f",    # First intraday obs 6-8am local — strong predictor
]


@dataclass
class QuantileModelConfig:
    """Configuration for quantile regression model."""
    quantiles: List[float] = field(default_factory=lambda: DEFAULT_QUANTILES.copy())
    n_estimators: int = 500
    max_depth: int = 6
    learning_rate: float = 0.05
    min_child_samples: int = 20
    subsample: float = 0.8
    colsample_bytree: float = 0.8
    reg_alpha: float = 0.1
    reg_lambda: float = 0.1
    random_state: int = 42


@dataclass
class ModelMetadata:
    """Metadata about a trained model."""
    city_key: str
    trained_at_utc: str
    training_samples: int
    validation_samples: int
    feature_names: List[str]
    quantiles: List[float]
    validation_metrics: Dict[str, float]
    config: Dict[str, Any]


class QuantileForestModel:
    """
    LightGBM quantile regression model for temperature forecast errors.

    Trains separate models for each quantile and uses them to construct
    a full probability distribution for the forecast error.
    """

    def __init__(
        self,
        config: Optional[QuantileModelConfig] = None,
        city_key: str = "unknown",
    ):
        self.config = config or QuantileModelConfig()
        self.city_key = city_key
        self.models: Dict[float, Any] = {}  # quantile -> trained model
        self.calibrators: Dict[float, Any] = {}  # quantile -> isotonic calibrator
        self.metadata: Optional[ModelMetadata] = None
        self.is_fitted = False

    def _check_lightgbm(self):
        if not HAS_LIGHTGBM:
            raise ImportError(
                "LightGBM is required for ML models. "
                "Install with: pip install lightgbm"
            )

    def _build_lgb_params(self, quantile: float) -> Dict[str, Any]:
        """Build LightGBM parameters for a specific quantile."""
        return {
            "objective": "quantile",
            "alpha": quantile,
            "metric": "quantile",
            "n_estimators": self.config.n_estimators,
            "max_depth": self.config.max_depth,
            "learning_rate": self.config.learning_rate,
            "min_child_samples": self.config.min_child_samples,
            "subsample": self.config.subsample,
            "colsample_bytree": self.config.colsample_bytree,
            "reg_alpha": self.config.reg_alpha,
            "reg_lambda": self.config.reg_lambda,
            "random_state": self.config.random_state,
            "verbose": -1,
            "n_jobs": -1,
        }

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
    ) -> "QuantileForestModel":
        """
        Train quantile regression models.

        Args:
            X: Training features (n_samples, n_features)
            y: Training targets (forecast errors)
            X_val: Validation features (optional)
            y_val: Validation targets (optional)

        Returns:
            self
        """
        self._check_lightgbm()

        n_samples = X.shape[0]
        n_val = X_val.shape[0] if X_val is not None else 0

        print(f"Training quantile models for {self.city_key}")
        print(f"  Training samples: {n_samples}")
        print(f"  Validation samples: {n_val}")
        print(f"  Features: {X.shape[1]}")
        print(f"  Quantiles: {self.config.quantiles}")

        self.models = {}
        validation_metrics = {}

        for q in self.config.quantiles:
            print(f"  Training q={q:.2f}...", end=" ", flush=True)

            params = self._build_lgb_params(q)
            model = lgb.LGBMRegressor(**params)

            # Fit with early stopping if validation data provided
            if X_val is not None and y_val is not None:
                model.fit(
                    X, y,
                    eval_set=[(X_val, y_val)],
                    callbacks=[lgb.early_stopping(50, verbose=False)],
                )
                # Compute validation pinball loss
                y_pred = model.predict(X_val)
                loss = pinball_loss(y_val, y_pred, q)
                validation_metrics[f"pinball_q{q:.2f}"] = loss
                print(f"loss={loss:.4f}")
            else:
                model.fit(X, y)
                print("done")

            self.models[q] = model

        self.is_fitted = True

        # Store metadata
        self.metadata = ModelMetadata(
            city_key=self.city_key,
            trained_at_utc=datetime.utcnow().isoformat(),
            training_samples=n_samples,
            validation_samples=n_val,
            feature_names=FEATURE_NAMES.copy(),
            quantiles=self.config.quantiles.copy(),
            validation_metrics=validation_metrics,
            config={
                "n_estimators": self.config.n_estimators,
                "max_depth": self.config.max_depth,
                "learning_rate": self.config.learning_rate,
            },
        )

        return self

    def predict_quantiles(
        self,
        X: np.ndarray,
    ) -> Dict[float, np.ndarray]:
        """
        Predict quantiles of the error distribution.

        Args:
            X: Features (n_samples, n_features)

        Returns:
            Dict mapping quantile -> predictions
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")

        predictions = {}
        for q, model in self.models.items():
            predictions[q] = model.predict(X)

        return predictions

    def predict_error_distribution(
        self,
        features: Dict[str, float],
    ) -> Tuple[int, int, int, int, int]:
        """
        Predict error distribution for a single forecast.

        Args:
            features: Feature dict with keys matching FEATURE_NAMES

        Returns:
            (p10, p25, p50, p75, p90) error quantiles
        """
        X = self._features_to_array(features)
        quantile_preds = self.predict_quantiles(X)

        # Extract standard quantiles
        p10 = int(round(quantile_preds.get(0.10, [0])[0]))
        p25 = int(round(quantile_preds.get(0.25, [0])[0]))
        p50 = int(round(quantile_preds.get(0.50, [0])[0]))
        p75 = int(round(quantile_preds.get(0.75, [0])[0]))
        p90 = int(round(quantile_preds.get(0.90, [0])[0]))

        return (p10, p25, p50, p75, p90)

    def predict_observed_distribution(
        self,
        forecast_high: int,
        features: Dict[str, float],
    ) -> Tuple[int, int, int, int, int]:
        """
        Predict distribution of observed high temperature.

        observed = forecast + error, so we add forecast_high to error quantiles.

        Args:
            forecast_high: Point forecast temperature
            features: Feature dict

        Returns:
            (p10, p25, p50, p75, p90) observed temperature quantiles
        """
        error_p10, error_p25, error_p50, error_p75, error_p90 = self.predict_error_distribution(features)

        return (
            forecast_high + error_p10,
            forecast_high + error_p25,
            forecast_high + error_p50,
            forecast_high + error_p75,
            forecast_high + error_p90,
        )

    def _features_to_array(self, features: Dict[str, float]) -> np.ndarray:
        """Convert feature dict to numpy array."""
        arr = np.zeros((1, len(FEATURE_NAMES)))
        for i, name in enumerate(FEATURE_NAMES):
            arr[0, i] = features.get(name, 0.0)
        return arr

    def calibrate(
        self,
        X_cal: np.ndarray,
        y_cal: np.ndarray,
    ) -> "QuantileForestModel":
        """
        Fit isotonic calibrators using calibration data.

        Isotonic regression ensures that predicted probabilities are
        well-calibrated (predicted 30% events happen ~30% of the time).

        Args:
            X_cal: Calibration features
            y_cal: Calibration targets

        Returns:
            self
        """
        if not HAS_SKLEARN:
            print("  [warn] sklearn not available, skipping calibration")
            return self

        if not self.is_fitted:
            raise ValueError("Model must be fitted before calibration")

        print(f"Calibrating quantile models with {len(y_cal)} samples...")

        self.calibrators = {}
        quantile_preds = self.predict_quantiles(X_cal)

        for q in self.config.quantiles:
            # For each quantile, fit isotonic regression on P(y <= pred_q)
            preds = quantile_preds[q]

            # Binary indicator: was actual below predicted quantile?
            indicators = (y_cal <= preds).astype(float)

            # Fit isotonic regression
            calibrator = IsotonicRegression(out_of_bounds="clip")
            calibrator.fit(preds, indicators)
            self.calibrators[q] = calibrator

            print(f"  Calibrated q={q:.2f}")

        return self

    def save(self, path: Path) -> None:
        """Save model to disk."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "city_key": self.city_key,
            "config": self.config,
            "models": self.models,
            "calibrators": self.calibrators,
            "metadata": self.metadata,
            "is_fitted": self.is_fitted,
        }

        with open(path, "wb") as f:
            pickle.dump(data, f)

        print(f"Saved model to {path}")

    @classmethod
    def load(cls, path: Path) -> "QuantileForestModel":
        """Load model from disk."""
        path = Path(path)

        with open(path, "rb") as f:
            data = pickle.load(f)

        instance = cls(
            config=data.get("config"),
            city_key=data.get("city_key", "unknown"),
        )
        instance.models = data.get("models", {})
        instance.calibrators = data.get("calibrators", {})
        instance.metadata = data.get("metadata")
        instance.is_fitted = data.get("is_fitted", False)

        return instance

    def feature_importance(self) -> Dict[str, float]:
        """Get feature importance scores averaged across quantile models."""
        if not self.is_fitted:
            return {}

        importance = {name: 0.0 for name in FEATURE_NAMES}

        for model in self.models.values():
            for i, imp in enumerate(model.feature_importances_):
                if i < len(FEATURE_NAMES):
                    importance[FEATURE_NAMES[i]] += imp

        # Average across models
        n_models = len(self.models)
        if n_models > 0:
            importance = {k: v / n_models for k, v in importance.items()}

        return importance


def pinball_loss(y_true: np.ndarray, y_pred: np.ndarray, quantile: float) -> float:
    """
    Compute pinball (quantile) loss.

    Loss is asymmetric:
    - Underprediction (y_true > y_pred) penalized by factor `quantile`
    - Overprediction (y_true < y_pred) penalized by factor `(1 - quantile)`
    """
    errors = y_true - y_pred
    loss = np.where(
        errors >= 0,
        quantile * errors,
        (quantile - 1) * errors,
    )
    return float(np.mean(loss))


def prepare_features(
    forecast_high: int,
    target_date: datetime,
    nbm_spread: Optional[int] = None,
    precip_prob: float = 0.0,
    cloud_cover: float = 0.5,
    wind_speed: float = 10.0,
    prev_day_error: int = 0,
    rolling_7d_bias: float = 0.0,
    snow_depth_inches: float = 0.0,
    sst_f: Optional[float] = None,
    sst_anomaly_f: Optional[float] = None,
    visibility_min_miles: Optional[float] = None,
    precip_last_3d_mm: float = 0.0,
    forecast_anomaly: Optional[float] = None,
    # Phase 1: Multi-model spread
    model_spread_f: Optional[float] = None,
    model_mean_f: Optional[float] = None,
    ecmwf_f: Optional[float] = None,
    n_models: Optional[float] = None,
    # Phase 2: Upper-air features
    z500: Optional[float] = None,
    t850: Optional[float] = None,
    z500_anomaly: Optional[float] = None,
    dewpoint_f: Optional[float] = None,
    # Phase 3: Teleconnection indices
    nao: Optional[float] = None,
    ao: Optional[float] = None,
    pna: Optional[float] = None,
    mei_enso: Optional[float] = None,
    pdo: Optional[float] = None,
    mjo_amplitude: Optional[float] = None,
    # Phase 4: Morning temperature
    morning_temp_f: Optional[float] = None,
    # Deprecated params kept for backward compat — ignored
    lead_hours: int = 18,
    smoke_present: bool = False,
) -> Dict[str, float]:
    """
    Prepare feature dictionary for model prediction.

    Args:
        forecast_high: Point forecast temperature
        target_date: Target date
        nbm_spread: NBM p90 - p10 spread (defaults to climatological estimate)
        precip_prob: Precipitation probability (0-1)
        cloud_cover: Cloud cover fraction (0-1)
        wind_speed: Wind speed (mph)
        prev_day_error: Previous day's forecast error
        rolling_7d_bias: 7-day rolling bias
        snow_depth_inches: Snow depth in inches (0 if no snow)
        sst_f: Sea surface temperature in F (None for non-coastal cities)
        sst_anomaly_f: SST anomaly from seasonal normal
        visibility_min_miles: Minimum visibility in miles (None if not available)
        precip_last_3d_mm: Precipitation total over last 3 days (mm)
        forecast_anomaly: Forecast high minus climatological monthly avg
        model_spread_f: Max minus min across all forecast models
        model_mean_f: Mean forecast across all available models
        ecmwf_f: ECMWF-specific forecast high (best global model)
        n_models: Number of models with data for this date
        z500: 500mb geopotential height in meters
        t850: 850mb temperature in Celsius
        z500_anomaly: z500 minus monthly climatological mean
        dewpoint_f: Surface dewpoint in Fahrenheit
        nao: North Atlantic Oscillation index (monthly)
        ao: Arctic Oscillation index (monthly)
        pna: Pacific North American pattern index (monthly)
        mei_enso: Multivariate ENSO Index (monthly)
        pdo: Pacific Decadal Oscillation index (monthly)
        mjo_amplitude: Madden-Julian Oscillation amplitude
        morning_temp_f: First intraday observation (6-8am local)

    Returns:
        Feature dictionary
    """
    month = target_date.month
    day_of_year = target_date.timetuple().tm_yday

    # Default NBM spread based on month if not provided
    if nbm_spread is None:
        monthly_spreads = {
            1: 12, 2: 11, 3: 10, 4: 9, 5: 8, 6: 7,
            7: 7, 8: 7, 9: 8, 10: 9, 11: 10, 12: 11,
        }
        nbm_spread = monthly_spreads.get(month, 8)

    # Derive has_snow from snow_depth
    has_snow = 1.0 if snow_depth_inches > 0 else 0.0

    # Default values for optional features
    sst_value = float(sst_f) if sst_f is not None else 60.0
    sst_anomaly_value = float(sst_anomaly_f) if sst_anomaly_f is not None else 0.0
    visibility_value = float(visibility_min_miles) if visibility_min_miles is not None else 10.0

    # Compute forecast anomaly from climatology if not provided
    if forecast_anomaly is None:
        # Rough US monthly avg highs (national average, °F)
        monthly_clim = {
            1: 42, 2: 46, 3: 54, 4: 64, 5: 74, 6: 82,
            7: 86, 8: 85, 9: 78, 10: 67, 11: 55, 12: 44,
        }
        forecast_anomaly = float(forecast_high) - monthly_clim.get(month, 65)

    # Phase 1 defaults: use forecast_high as proxy when model data is unavailable
    model_spread_val = float(model_spread_f) if model_spread_f is not None else 0.0
    model_mean_val = float(model_mean_f) if model_mean_f is not None else float(forecast_high)
    ecmwf_val = float(ecmwf_f) if ecmwf_f is not None else float(forecast_high)
    n_models_val = float(n_models) if n_models is not None else 1.0

    # Phase 2 defaults: use 0 for anomaly-type features, climatological guess for z500/t850
    # z500 climatological ~5500m; t850 varies by season/location — use 0 as neutral
    z500_val = float(z500) if z500 is not None else 5500.0
    t850_val = float(t850) if t850 is not None else 0.0
    z500_anomaly_val = float(z500_anomaly) if z500_anomaly is not None else 0.0
    # Convert dewpoint from C to F if provided in C, or use 50°F as neutral default
    dewpoint_val = float(dewpoint_f) if dewpoint_f is not None else 50.0

    # Phase 3 defaults: use 0 for all teleconnection indices (neutral state)
    nao_val = float(nao) if nao is not None else 0.0
    ao_val = float(ao) if ao is not None else 0.0
    pna_val = float(pna) if pna is not None else 0.0
    mei_enso_val = float(mei_enso) if mei_enso is not None else 0.0
    pdo_val = float(pdo) if pdo is not None else 0.0
    mjo_amplitude_val = float(mjo_amplitude) if mjo_amplitude is not None else 0.0

    # Phase 4: morning_temp_f — use forecast_high as fallback (no info)
    morning_temp_val = float(morning_temp_f) if morning_temp_f is not None else float(forecast_high)

    return {
        "month": float(month),
        "day_of_year": float(day_of_year),
        "forecast_high": float(forecast_high),
        "nbm_spread": float(nbm_spread),
        "precip_prob": float(precip_prob),
        "cloud_cover": float(cloud_cover),
        "wind_speed": float(wind_speed),
        "prev_day_error": float(prev_day_error),
        "rolling_7d_bias": float(rolling_7d_bias),
        "snow_depth_inches": float(snow_depth_inches),
        "has_snow": has_snow,
        "sst_f": sst_value,
        "sst_anomaly_f": sst_anomaly_value,
        "visibility_min_miles": visibility_value,
        "precip_last_3d_mm": float(precip_last_3d_mm),
        "forecast_anomaly": float(forecast_anomaly),
        "model_spread_f": model_spread_val,
        "model_mean_f": model_mean_val,
        "ecmwf_f": ecmwf_val,
        "n_models": n_models_val,
        "z500": z500_val,
        "t850": t850_val,
        "z500_anomaly": z500_anomaly_val,
        "dewpoint_f": dewpoint_val,
        "nao": nao_val,
        "ao": ao_val,
        "pna": pna_val,
        "mei_enso": mei_enso_val,
        "pdo": pdo_val,
        "mjo_amplitude": mjo_amplitude_val,
        "morning_temp_f": morning_temp_val,
    }


class SimpleFallbackModel:
    """
    Simple fallback model when LightGBM is not available.

    Uses climatological error distributions based on month and lead time.
    """

    def __init__(self, city_key: str = "unknown"):
        self.city_key = city_key
        self.is_fitted = True  # Always ready

    def predict_error_distribution(
        self,
        features: Dict[str, float],
    ) -> Tuple[int, int, int, int, int]:
        """Predict error distribution using climatological estimates."""
        month = int(features.get("month", 6))
        lead_hours = features.get("lead_hours", 18)
        nbm_spread = features.get("nbm_spread", 8)

        # Base spread by month
        monthly_spreads = {
            1: 12, 2: 11, 3: 10, 4: 9, 5: 8, 6: 7,
            7: 7, 8: 7, 9: 8, 10: 9, 11: 10, 12: 11,
        }
        base_spread = monthly_spreads.get(month, 8)

        # Adjust for lead time
        lead_factor = 1.0 + (lead_hours / 72.0) * 0.3
        spread = base_spread * lead_factor

        # If we have NBM spread, blend it
        if nbm_spread > 0:
            spread = 0.6 * nbm_spread + 0.4 * spread

        # Convert spread to standard deviation
        # For normal distribution, p90 - p10 ≈ 2.56σ
        std = spread / 2.56

        # Compute quantiles (error centered around 0)
        p10 = int(round(-1.28 * std))
        p25 = int(round(-0.67 * std))
        p50 = 0
        p75 = int(round(0.67 * std))
        p90 = int(round(1.28 * std))

        return (p10, p25, p50, p75, p90)

    def predict_observed_distribution(
        self,
        forecast_high: int,
        features: Dict[str, float],
    ) -> Tuple[int, int, int, int, int]:
        """Predict observed high distribution."""
        error_p10, error_p25, error_p50, error_p75, error_p90 = self.predict_error_distribution(features)

        return (
            forecast_high + error_p10,
            forecast_high + error_p25,
            forecast_high + error_p50,
            forecast_high + error_p75,
            forecast_high + error_p90,
        )


def get_model(city_key: str, model_dir: Optional[Path] = None) -> Any:
    """
    Get the best available model for a city.

    Tries to load a trained LightGBM model, falls back to simple model.

    Args:
        city_key: City identifier
        model_dir: Directory containing saved models

    Returns:
        Model instance (QuantileForestModel or SimpleFallbackModel)
    """
    if model_dir is not None:
        model_path = model_dir / f"{city_key.lower()}_quantile.pkl"
        if model_path.exists():
            try:
                return QuantileForestModel.load(model_path)
            except Exception as e:
                print(f"  [warn] Failed to load model {model_path}: {e}")

    # Fall back to simple model
    return SimpleFallbackModel(city_key=city_key)
