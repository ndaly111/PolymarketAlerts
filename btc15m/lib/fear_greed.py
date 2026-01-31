"""Fear & Greed Index from alternative.me API."""

from __future__ import annotations

import requests
from typing import Optional, Dict
from dataclasses import dataclass


@dataclass
class FearGreedData:
    """Fear & Greed Index data."""
    value: int  # 0-100
    classification: str  # "Extreme Fear", "Fear", "Neutral", "Greed", "Extreme Greed"
    timestamp: int


def fetch_fear_greed() -> Optional[FearGreedData]:
    """Fetch current Fear & Greed Index."""
    try:
        resp = requests.get(
            "https://api.alternative.me/fng/",
            params={"limit": 1},
            timeout=10
        )
        data = resp.json()

        if data.get("data"):
            item = data["data"][0]
            return FearGreedData(
                value=int(item["value"]),
                classification=item["value_classification"],
                timestamp=int(item["timestamp"]),
            )
        return None
    except Exception as e:
        print(f"[fear_greed] Error: {e}")
        return None


def compute_fear_greed_features(data: Optional[FearGreedData]) -> Dict[str, float]:
    """Compute features from Fear & Greed data."""
    if not data:
        return {
            "fg_value": 50,  # Neutral default
            "fg_normalized": 0,  # -1 to 1 scale
            "fg_extreme_fear": 0,
            "fg_fear": 0,
            "fg_neutral": 1,
            "fg_greed": 0,
            "fg_extreme_greed": 0,
            "fg_contrarian_buy": 0,
            "fg_contrarian_sell": 0,
        }

    value = data.value

    # Normalize to -1 to 1 (0 = extreme fear, 100 = extreme greed)
    normalized = (value - 50) / 50

    # Classification flags
    extreme_fear = 1 if value <= 25 else 0
    fear = 1 if 25 < value <= 45 else 0
    neutral = 1 if 45 < value <= 55 else 0
    greed = 1 if 55 < value <= 75 else 0
    extreme_greed = 1 if value > 75 else 0

    # Contrarian signals
    # Extreme fear = potential buy (market oversold)
    # Extreme greed = potential sell (market overbought)
    contrarian_buy = 1 if value <= 20 else 0
    contrarian_sell = 1 if value >= 80 else 0

    return {
        "fg_value": value,
        "fg_normalized": normalized,
        "fg_extreme_fear": extreme_fear,
        "fg_fear": fear,
        "fg_neutral": neutral,
        "fg_greed": greed,
        "fg_extreme_greed": extreme_greed,
        "fg_contrarian_buy": contrarian_buy,
        "fg_contrarian_sell": contrarian_sell,
    }
