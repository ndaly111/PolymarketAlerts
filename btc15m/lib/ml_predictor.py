"""ML model predictor for BTC 15-min direction.

Loads trained model and provides probability predictions
with confidence filtering.
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Optional, List
from dataclasses import dataclass

import numpy as np

# Default model path
DEFAULT_MODEL_PATH = Path(__file__).parent.parent / "models" / "btc15m_model.pkl"


@dataclass
class Prediction:
    """ML model prediction result."""
    direction: str       # 'UP' or 'DOWN'
    probability: float   # Probability of predicted direction (0.5-1.0)
    confidence: str      # 'LOW', 'MEDIUM', 'HIGH', 'VERY_HIGH'
    up_prob: float       # Probability of UP
    down_prob: float     # Probability of DOWN
    features_used: int   # Number of features used


class MLPredictor:
    """ML predictor for BTC 15-minute direction."""

    def __init__(self, model_path: Path = DEFAULT_MODEL_PATH):
        """Load trained model from disk."""
        self.model = None
        self.scaler = None
        self.model_path = model_path
        self._load_model()

    def _load_model(self) -> None:
        """Load model and scaler from pickle file."""
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model not found at {self.model_path}")

        with open(self.model_path, "rb") as f:
            data = pickle.load(f)
            self.model = data["model"]
            self.scaler = data["scaler"]

    def predict(self, features: dict) -> Optional[Prediction]:
        """Make a prediction given feature dictionary.

        Args:
            features: Dict with feature values (matches FEATURE_COLUMNS)

        Returns:
            Prediction object with direction, probability, and confidence
        """
        if self.model is None:
            return None

        # Build feature vector in correct order
        from scripts.train_model import FEATURE_COLUMNS

        feature_vector = []
        defaults = {
            "rsi": 50, "macd": 0, "macd_signal": 0, "macd_histogram": 0,
            "volatility": 0, "momentum": 0, "price_vs_vwap": 0,
            "bb_percent_b": 0.5, "bb_bandwidth": 0, "atr": 0, "stoch_k": 50,
            "williams_r": -50, "cci": 0, "obv_trend": 0,
            "trend_15m": 0, "trend_1h": 0, "change_15m": 0, "change_1h": 0,
            "hour": 12, "hour_sin": 0, "hour_cos": 1,
            "asia_session": 0, "europe_session": 0, "us_session": 0, "high_vol_hour": 0,
            "doji": 0, "hammer": 0, "shooting_star": 0,
            "bullish_engulfing": 0, "bearish_engulfing": 0,
            "adx": 25, "plus_di": 25, "minus_di": 25, "mfi": 50,
            "roc": 0, "trix": 0, "kc_position": 0.5, "kc_width": 0,
            "dc_position": 0.5, "dc_width": 0, "ichi_tk_cross": 0,
            "ichi_above_cloud": 0, "ichi_below_cloud": 0, "ichi_cloud_thickness": 0,
            "price_vs_vwma": 0, "cmf": 0, "chop": 50,
            "squeeze_on": 0, "squeeze_momentum": 0,
            "vol_ratio": 1, "vol_trend": 0, "vol_up_ratio": 0.5, "vol_spike": 0,
            "dist_to_high_pct": 0, "dist_to_low_pct": 0, "range_compression": 1,
            "higher_highs": 0, "lower_lows": 0, "up_streak": 0, "down_streak": 0,
            "above_pivot": 0, "dist_to_nearest_pivot": 0,
            "lr_slope": 0, "lr_deviation": 0, "lr_r_squared": 0,
            "comp_trend_composite": 0, "comp_weighted_trend": 0,
            "comp_ob_os_signal": 0, "comp_overbought_consensus": 0, "comp_oversold_consensus": 0,
            "comp_volatility_composite": 0, "comp_is_ranging": 0, "comp_is_trending": 0,
            "comp_breakout_potential": 0, "comp_vol_price_confirmation": 0,
            "comp_vol_price_divergence": 0, "comp_strong_move": 0,
            "comp_mtf_aligned": 0, "comp_mtf_direction": 0, "comp_mtf_acceleration": 0,
            "comp_breakout_score": 0, "comp_breakout_direction": 0,
            "comp_has_divergence": 0, "comp_divergence_strength": 0,
        }

        for col in FEATURE_COLUMNS:
            val = features.get(col)
            if val is None:
                val = defaults.get(col, 0)
            feature_vector.append(val)

        # Scale features
        X = np.array([feature_vector])
        X_scaled = self.scaler.transform(X)

        # Get probability predictions
        proba = self.model.predict_proba(X_scaled)[0]

        # proba[0] = DOWN probability, proba[1] = UP probability
        down_prob = proba[0]
        up_prob = proba[1]

        # Determine direction and confidence
        if up_prob > down_prob:
            direction = "UP"
            probability = up_prob
        else:
            direction = "DOWN"
            probability = down_prob

        # Classify confidence level
        if probability >= 0.65:
            confidence = "VERY_HIGH"
        elif probability >= 0.60:
            confidence = "HIGH"
        elif probability >= 0.55:
            confidence = "MEDIUM"
        else:
            confidence = "LOW"

        return Prediction(
            direction=direction,
            probability=probability,
            confidence=confidence,
            up_prob=up_prob,
            down_prob=down_prob,
            features_used=len(feature_vector),
        )

    def predict_with_threshold(
        self,
        features: dict,
        min_confidence: float = 0.55,
    ) -> Optional[Prediction]:
        """Only return prediction if confidence meets threshold.

        Args:
            features: Feature dictionary
            min_confidence: Minimum probability to return prediction (default 0.55)

        Returns:
            Prediction if confidence >= threshold, else None
        """
        pred = self.predict(features)
        if pred and pred.probability >= min_confidence:
            return pred
        return None


def compute_all_features(candles: List[dict], current_hour: int = 12) -> dict:
    """Compute all features needed for ML prediction from candle data.

    Args:
        candles: List of candle dicts with open, high, low, close, volume
        current_hour: Current hour (UTC) for time features

    Returns:
        Dict with all computed features
    """
    from lib.indicators import (
        compute_rsi, compute_macd, compute_vwap, compute_volatility,
        compute_price_momentum, compute_sma, compute_bollinger_bands,
        compute_atr, compute_stochastic, compute_williams_r, compute_cci,
        compute_obv_trend, compute_higher_timeframe_trend, compute_time_features,
        compute_candle_patterns, compute_adx, compute_mfi, compute_roc,
        compute_trix, compute_keltner_channels, compute_donchian_channels,
        compute_ichimoku, compute_vwma, compute_cmf, compute_chop,
        compute_squeeze, compute_volume_features, compute_price_action_features,
        compute_pivot_points, compute_linear_regression, compute_all_composites,
    )

    closes = [c["close"] for c in candles]

    # Original indicators
    rsi = compute_rsi(closes, period=14)
    macd_result = compute_macd(closes, fast=12, slow=26, signal=9)
    vwap = compute_vwap(candles)
    volatility = compute_volatility(closes, period=20)
    momentum = compute_price_momentum(closes, period=10)
    sma_20 = compute_sma(closes, period=20)

    price = closes[-1] if closes else 0
    price_vs_vwap = ((price - vwap) / vwap * 100) if vwap else 0

    # Batch 1 indicators
    bb = compute_bollinger_bands(closes, period=20, std_dev=2.0)
    atr = compute_atr(candles, period=14)
    stoch = compute_stochastic(candles, k_period=14, d_period=3)
    williams_r = compute_williams_r(candles, period=14)
    cci = compute_cci(candles, period=20)
    obv_trend = compute_obv_trend(candles, period=10)
    htf_trend = compute_higher_timeframe_trend(closes)
    time_feats = compute_time_features(current_hour)
    patterns = compute_candle_patterns(candles)

    # Batch 2 indicators
    adx = compute_adx(candles, period=14)
    mfi = compute_mfi(candles, period=14)
    roc = compute_roc(closes, period=10)
    trix = compute_trix(closes, period=15)
    kc = compute_keltner_channels(candles, ema_period=20, atr_period=10)
    dc = compute_donchian_channels(candles, period=20)
    ichi = compute_ichimoku(candles)
    vwma = compute_vwma(candles, period=20)
    cmf = compute_cmf(candles, period=20)
    chop = compute_chop(candles, period=14)
    squeeze = compute_squeeze(candles)
    vol_feats = compute_volume_features(candles, period=20)
    pa_feats = compute_price_action_features(candles, period=20)
    pivot = compute_pivot_points(candles)
    lr = compute_linear_regression(closes, period=20)

    # Build feature dict
    features = {
        # Original
        "rsi": rsi,
        "macd": macd_result.macd if macd_result else 0,
        "macd_signal": macd_result.signal if macd_result else 0,
        "macd_histogram": macd_result.histogram if macd_result else 0,
        "volatility": volatility,
        "momentum": momentum,
        "price_vs_vwap": price_vs_vwap,
        # Bollinger
        "bb_percent_b": bb["percent_b"] if bb else 0.5,
        "bb_bandwidth": bb["bandwidth"] if bb else 0,
        # ATR
        "atr": atr or 0,
        # Stochastic
        "stoch_k": stoch["k"] if stoch else 50,
        # Williams %R
        "williams_r": williams_r or -50,
        # CCI
        "cci": cci or 0,
        # OBV
        "obv_trend": obv_trend or 0,
        # Higher TF
        "trend_15m": htf_trend.get("trend_15m", 0),
        "trend_1h": htf_trend.get("trend_1h", 0),
        "change_15m": htf_trend.get("change_15m", 0),
        "change_1h": htf_trend.get("change_1h", 0),
        # Time
        "hour": time_feats["hour"],
        "hour_sin": time_feats["hour_sin"],
        "hour_cos": time_feats["hour_cos"],
        "asia_session": time_feats["asia_session"],
        "europe_session": time_feats["europe_session"],
        "us_session": time_feats["us_session"],
        "high_vol_hour": time_feats["high_vol_hour"],
        # Patterns
        "doji": patterns.get("doji", 0),
        "hammer": patterns.get("hammer", 0),
        "shooting_star": patterns.get("shooting_star", 0),
        "bullish_engulfing": patterns.get("bullish_engulfing", 0),
        "bearish_engulfing": patterns.get("bearish_engulfing", 0),
        # ADX
        "adx": adx["adx"] if adx else 25,
        "plus_di": adx["plus_di"] if adx else 25,
        "minus_di": adx["minus_di"] if adx else 25,
        # MFI
        "mfi": mfi or 50,
        # ROC
        "roc": roc or 0,
        # TRIX
        "trix": trix or 0,
        # Keltner
        "kc_position": kc["kc_position"] if kc else 0.5,
        "kc_width": kc["kc_width"] if kc else 0,
        # Donchian
        "dc_position": dc["dc_position"] if dc else 0.5,
        "dc_width": dc["dc_width"] if dc else 0,
        # Ichimoku
        "ichi_tk_cross": ichi["ichi_tk_cross"] if ichi else 0,
        "ichi_above_cloud": ichi["ichi_above_cloud"] if ichi else 0,
        "ichi_below_cloud": ichi["ichi_below_cloud"] if ichi else 0,
        "ichi_cloud_thickness": ichi["ichi_cloud_thickness"] if ichi else 0,
        # VWMA
        "price_vs_vwma": ((price - vwma) / vwma * 100) if vwma else 0,
        # CMF
        "cmf": cmf or 0,
        # Chop
        "chop": chop or 50,
        # Squeeze
        "squeeze_on": squeeze["squeeze_on"] if squeeze else 0,
        "squeeze_momentum": squeeze["squeeze_momentum"] if squeeze else 0,
        # Volume
        "vol_ratio": vol_feats["vol_ratio"] if vol_feats else 1,
        "vol_trend": vol_feats["vol_trend"] if vol_feats else 0,
        "vol_up_ratio": vol_feats["vol_up_ratio"] if vol_feats else 0.5,
        "vol_spike": vol_feats["vol_spike"] if vol_feats else 0,
        # Price action
        "dist_to_high_pct": pa_feats["dist_to_high_pct"] if pa_feats else 0,
        "dist_to_low_pct": pa_feats["dist_to_low_pct"] if pa_feats else 0,
        "range_compression": pa_feats["range_compression"] if pa_feats else 1,
        "higher_highs": pa_feats["higher_highs"] if pa_feats else 0,
        "lower_lows": pa_feats["lower_lows"] if pa_feats else 0,
        "up_streak": pa_feats["up_streak"] if pa_feats else 0,
        "down_streak": pa_feats["down_streak"] if pa_feats else 0,
        # Pivot
        "above_pivot": pivot["above_pivot"] if pivot else 0,
        "dist_to_nearest_pivot": pivot["dist_to_nearest"] if pivot else 0,
        # Linear regression
        "lr_slope": lr["lr_slope"] if lr else 0,
        "lr_deviation": lr["lr_deviation"] if lr else 0,
        "lr_r_squared": lr["lr_r_squared"] if lr else 0,
    }

    # Composite indicators
    composite_inputs = {
        "adx": features["adx"],
        "rsi": features["rsi"],
        "macd_histogram": features["macd_histogram"],
        "momentum": features["momentum"],
        "stoch_k": features["stoch_k"],
        "williams_r": features["williams_r"],
        "mfi": features["mfi"],
        "cci": features["cci"],
        "atr": features["atr"],
        "bb_bandwidth": features["bb_bandwidth"],
        "kc_width": features["kc_width"],
        "chop": features["chop"],
        "vol_ratio": features["vol_ratio"],
        "obv_trend": features["obv_trend"],
        "cmf": features["cmf"],
        "price_change": 0,
        "trend_15m": features["trend_15m"],
        "trend_1h": features["trend_1h"],
        "change_15m": features["change_15m"],
        "change_1h": features["change_1h"],
        "squeeze_on": features["squeeze_on"],
        "range_compression": features["range_compression"],
        "bb_percent_b": features["bb_percent_b"],
        "dc_position": features["dc_position"],
    }
    composites = compute_all_composites(closes, candles, composite_inputs)

    # Add composite features
    features.update({
        "comp_trend_composite": composites.get("comp_trend_composite", 0),
        "comp_weighted_trend": composites.get("comp_weighted_trend", 0),
        "comp_ob_os_signal": composites.get("comp_ob_os_signal", 0),
        "comp_overbought_consensus": composites.get("comp_overbought_consensus", 0),
        "comp_oversold_consensus": composites.get("comp_oversold_consensus", 0),
        "comp_volatility_composite": composites.get("comp_volatility_composite", 0),
        "comp_is_ranging": composites.get("comp_is_ranging", 0),
        "comp_is_trending": composites.get("comp_is_trending", 0),
        "comp_breakout_potential": composites.get("comp_breakout_potential", 0),
        "comp_vol_price_confirmation": composites.get("comp_vol_price_confirmation", 0),
        "comp_vol_price_divergence": composites.get("comp_vol_price_divergence", 0),
        "comp_strong_move": composites.get("comp_strong_move", 0),
        "comp_mtf_aligned": composites.get("comp_mtf_aligned", 0),
        "comp_mtf_direction": composites.get("comp_mtf_direction", 0),
        "comp_mtf_acceleration": composites.get("comp_mtf_acceleration", 0),
        "comp_breakout_score": composites.get("comp_breakout_score", 0),
        "comp_breakout_direction": composites.get("comp_breakout_direction", 0),
        "comp_has_divergence": composites.get("comp_has_divergence", 0),
        "comp_divergence_strength": composites.get("comp_divergence_strength", 0),
    })

    return features
