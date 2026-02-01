"""
Profit-optimized predictor for BTC trading.

Uses a model trained to predict whether a trade will be profitable,
given both technical indicators AND the market entry price.
"""

import pickle
from pathlib import Path
from typing import Dict, Optional, NamedTuple
import numpy as np


class ProfitPrediction(NamedTuple):
    """Prediction result from profit model."""
    is_profitable: bool
    probability: float  # Probability of being profitable
    direction: str  # 'UP' or 'DOWN' based on original prediction
    confidence: str  # 'HIGH', 'MEDIUM', 'LOW'


class ProfitPredictor:
    """Predictor that estimates probability of profitable trade."""

    def __init__(self, model_path: Optional[Path] = None):
        if model_path is None:
            model_path = Path(__file__).parent.parent / "models" / "btc15m_profit_model.pkl"

        self.model_path = model_path
        self.model = None
        self.scaler = None
        self.feature_names = None
        self.model_name = None

        self._load_model()

    def _load_model(self):
        """Load the trained model."""
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model not found: {self.model_path}")

        with open(self.model_path, "rb") as f:
            data = pickle.load(f)

        self.model = data["model"]
        self.scaler = data["scaler"]
        self.feature_names = data["feature_names"]
        self.model_name = data.get("model_name", "Unknown")

        print(f"[ProfitPredictor] Loaded {self.model_name} model with {len(self.feature_names)} features")

    def reload_model(self):
        """Reload model from disk (for hot reloading after retraining)."""
        self._load_model()

    def predict(
        self,
        features: Dict[str, float],
        market_ask: float,
        model_prob: float,
        edge: float,
        direction: str = "UP",
        distance_pct: float = 0,
        time_to_expiry_sec: int = 600,
        volatility_1h: float = 0,
        trend_1h: float = 0,
    ) -> Optional[ProfitPrediction]:
        """
        Predict whether a trade will be profitable.

        Args:
            features: Technical indicator features
            market_ask: Entry price (0-1)
            model_prob: Original model probability
            edge: Calculated edge (prob - price)
            direction: 'UP' or 'DOWN'
            distance_pct: Distance to strike as percentage
            time_to_expiry_sec: Seconds until expiry
            volatility_1h: 1-hour volatility
            trend_1h: 1-hour trend

        Returns:
            ProfitPrediction with probability of profitability
        """
        if self.model is None:
            return None

        # Build feature vector in same order as training
        row = []

        # Extra feature names that we add explicitly
        extra_feature_set = {
            "market_ask", "edge", "model_prob", "distance_pct",
            "time_to_expiry_sec", "volatility_1h", "trend_1h"
        }

        # Build feature vector matching training order exactly
        for fname in self.feature_names:
            if fname in extra_feature_set:
                # Handle extra features explicitly
                if fname == "market_ask":
                    row.append(market_ask)
                elif fname == "edge":
                    row.append(edge)
                elif fname == "model_prob":
                    row.append(model_prob)
                elif fname == "distance_pct":
                    row.append(distance_pct)
                elif fname == "time_to_expiry_sec":
                    row.append(float(time_to_expiry_sec))
                elif fname == "volatility_1h":
                    row.append(volatility_1h)
                elif fname == "trend_1h":
                    row.append(trend_1h)
            else:
                # Base features from technical indicators
                val = features.get(fname, 0)
                if isinstance(val, bool):
                    val = int(val)
                elif val is None:
                    val = 0
                row.append(float(val))

        # Scale and predict
        X = np.array([row])
        X_scaled = self.scaler.transform(X)

        # Get probability of profitable trade
        proba = self.model.predict_proba(X_scaled)[0]
        prob_profitable = proba[1]  # Class 1 = profitable

        # Determine confidence level
        if prob_profitable >= 0.75:
            confidence = "HIGH"
        elif prob_profitable >= 0.60:
            confidence = "MEDIUM"
        else:
            confidence = "LOW"

        return ProfitPrediction(
            is_profitable=prob_profitable > 0.5,
            probability=prob_profitable,
            direction=direction,
            confidence=confidence,
        )


# Singleton instance
_predictor: Optional[ProfitPredictor] = None


def get_profit_predictor() -> Optional[ProfitPredictor]:
    """Get or create singleton predictor."""
    global _predictor
    if _predictor is None:
        try:
            _predictor = ProfitPredictor()
        except Exception as e:
            print(f"[ProfitPredictor] Failed to load: {e}")
            return None
    return _predictor
