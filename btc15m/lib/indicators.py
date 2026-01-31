"""Technical indicators for BTC price analysis.

All functions are optimized for O(n) complexity.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional


@dataclass
class MACDResult:
    """MACD calculation result."""
    macd: float          # MACD line (fast EMA - slow EMA)
    signal: float        # Signal line (EMA of MACD)
    histogram: float     # MACD - Signal
    hist_delta: Optional[float]  # Change in histogram


def compute_rsi(closes: List[float], period: int = 14) -> Optional[float]:
    """Compute RSI (Relative Strength Index).

    Args:
        closes: List of closing prices
        period: RSI lookback period (default 14)

    Returns:
        RSI value (0-100) or None if insufficient data
    """
    if len(closes) < period + 1:
        return None

    gains = 0.0
    losses = 0.0

    # Calculate initial average gain/loss
    for i in range(len(closes) - period, len(closes)):
        diff = closes[i] - closes[i - 1]
        if diff > 0:
            gains += diff
        else:
            losses += abs(diff)

    avg_gain = gains / period
    avg_loss = losses / period

    if avg_loss == 0:
        return 100.0

    rs = avg_gain / avg_loss
    rsi = 100.0 - (100.0 / (1.0 + rs))

    return max(0.0, min(100.0, rsi))


def compute_rsi_series(closes: List[float], period: int = 14) -> List[Optional[float]]:
    """Compute RSI series incrementally in O(n).

    Uses Wilder's smoothing method for efficiency.
    """
    n = len(closes)
    if n < period + 1:
        return [None] * n

    series: List[Optional[float]] = [None] * n

    # Calculate first RSI using simple average
    gains = 0.0
    losses = 0.0
    for i in range(1, period + 1):
        diff = closes[i] - closes[i - 1]
        if diff > 0:
            gains += diff
        else:
            losses += abs(diff)

    avg_gain = gains / period
    avg_loss = losses / period

    if avg_loss == 0:
        series[period] = 100.0
    else:
        rs = avg_gain / avg_loss
        series[period] = 100.0 - (100.0 / (1.0 + rs))

    # Calculate subsequent RSIs using Wilder's smoothing
    for i in range(period + 1, n):
        diff = closes[i] - closes[i - 1]
        gain = diff if diff > 0 else 0.0
        loss = abs(diff) if diff < 0 else 0.0

        avg_gain = (avg_gain * (period - 1) + gain) / period
        avg_loss = (avg_loss * (period - 1) + loss) / period

        if avg_loss == 0:
            series[i] = 100.0
        else:
            rs = avg_gain / avg_loss
            series[i] = 100.0 - (100.0 / (1.0 + rs))

    return series


def _ema(values: List[float], period: int) -> Optional[float]:
    """Compute EMA (Exponential Moving Average)."""
    if len(values) < period:
        return None

    k = 2.0 / (period + 1)
    ema = values[0]
    for i in range(1, len(values)):
        ema = values[i] * k + ema * (1 - k)

    return ema


def _ema_series(values: List[float], period: int) -> List[Optional[float]]:
    """Compute EMA series incrementally in O(n)."""
    n = len(values)
    if n < period:
        return [None] * n

    series: List[Optional[float]] = [None] * n
    k = 2.0 / (period + 1)

    # First EMA is SMA of first 'period' values
    sma = sum(values[:period]) / period
    series[period - 1] = sma

    # Subsequent EMAs
    ema = sma
    for i in range(period, n):
        ema = values[i] * k + ema * (1 - k)
        series[i] = ema

    return series


def compute_macd(
    closes: List[float],
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
) -> Optional[MACDResult]:
    """Compute MACD indicator.

    Args:
        closes: List of closing prices
        fast: Fast EMA period (default 12)
        slow: Slow EMA period (default 26)
        signal: Signal line EMA period (default 9)

    Returns:
        MACDResult or None if insufficient data
    """
    if len(closes) < slow + signal:
        return None

    # Compute EMA series incrementally
    fast_ema_series = _ema_series(closes, fast)
    slow_ema_series = _ema_series(closes, slow)

    # Build MACD series
    macd_series: List[float] = []
    for i in range(len(closes)):
        if fast_ema_series[i] is not None and slow_ema_series[i] is not None:
            macd_series.append(fast_ema_series[i] - slow_ema_series[i])

    if len(macd_series) < signal:
        return None

    # Compute signal line (EMA of MACD)
    signal_ema = _ema(macd_series, signal)
    if signal_ema is None:
        return None

    macd_line = macd_series[-1]
    histogram = macd_line - signal_ema

    # Calculate histogram delta
    hist_delta = None
    if len(macd_series) >= signal + 1:
        prev_signal = _ema(macd_series[:-1], signal)
        if prev_signal is not None:
            prev_hist = macd_series[-2] - prev_signal
            hist_delta = histogram - prev_hist

    return MACDResult(
        macd=macd_line,
        signal=signal_ema,
        histogram=histogram,
        hist_delta=hist_delta,
    )


def compute_vwap(candles: List[dict]) -> Optional[float]:
    """Compute VWAP (Volume Weighted Average Price).

    Args:
        candles: List of candle dicts with 'high', 'low', 'close', 'volume'

    Returns:
        VWAP value or None if no volume
    """
    if not candles:
        return None

    cum_pv = 0.0
    cum_v = 0.0

    for c in candles:
        tp = (c["high"] + c["low"] + c["close"]) / 3
        vol = c["volume"]
        cum_pv += tp * vol
        cum_v += vol

    if cum_v == 0:
        return None

    return cum_pv / cum_v


def compute_vwap_series(candles: List[dict]) -> List[Optional[float]]:
    """Compute VWAP series incrementally in O(n)."""
    if not candles:
        return []

    series: List[Optional[float]] = []
    cum_pv = 0.0
    cum_v = 0.0

    for c in candles:
        tp = (c["high"] + c["low"] + c["close"]) / 3
        vol = c["volume"]
        cum_pv += tp * vol
        cum_v += vol
        series.append(cum_pv / cum_v if cum_v > 0 else None)

    return series


def compute_sma(values: List[float], period: int) -> Optional[float]:
    """Compute Simple Moving Average."""
    if len(values) < period:
        return None
    return sum(values[-period:]) / period


def compute_volatility(closes: List[float], period: int = 20) -> Optional[float]:
    """Compute price volatility (standard deviation of returns)."""
    if len(closes) < period + 1:
        return None

    returns = []
    for i in range(len(closes) - period, len(closes)):
        ret = (closes[i] - closes[i - 1]) / closes[i - 1]
        returns.append(ret)

    mean = sum(returns) / len(returns)
    variance = sum((r - mean) ** 2 for r in returns) / len(returns)

    return variance ** 0.5


def compute_price_momentum(closes: List[float], period: int = 10) -> Optional[float]:
    """Compute price momentum (percent change over period)."""
    if len(closes) < period:
        return None

    return (closes[-1] - closes[-period]) / closes[-period] * 100


# ============================================================
# Additional Indicators for Better Prediction
# ============================================================

def compute_bollinger_bands(
    closes: List[float],
    period: int = 20,
    std_dev: float = 2.0,
) -> Optional[dict]:
    """Compute Bollinger Bands.

    Returns dict with 'upper', 'middle', 'lower', 'percent_b', 'bandwidth'
    """
    if len(closes) < period:
        return None

    # Middle band = SMA
    middle = sum(closes[-period:]) / period

    # Standard deviation
    variance = sum((c - middle) ** 2 for c in closes[-period:]) / period
    std = variance ** 0.5

    upper = middle + (std_dev * std)
    lower = middle - (std_dev * std)

    # %B = (Price - Lower) / (Upper - Lower)
    current = closes[-1]
    percent_b = (current - lower) / (upper - lower) if upper != lower else 0.5

    # Bandwidth = (Upper - Lower) / Middle
    bandwidth = (upper - lower) / middle if middle != 0 else 0

    return {
        "upper": upper,
        "middle": middle,
        "lower": lower,
        "percent_b": percent_b,      # 0-1, >1 = above upper, <0 = below lower
        "bandwidth": bandwidth,       # Higher = more volatile
    }


def compute_atr(
    candles: List[dict],
    period: int = 14,
) -> Optional[float]:
    """Compute Average True Range (volatility indicator).

    Args:
        candles: List of dicts with 'high', 'low', 'close'
        period: ATR period (default 14)
    """
    if len(candles) < period + 1:
        return None

    true_ranges = []
    for i in range(1, len(candles)):
        high = candles[i]["high"]
        low = candles[i]["low"]
        prev_close = candles[i - 1]["close"]

        tr = max(
            high - low,
            abs(high - prev_close),
            abs(low - prev_close),
        )
        true_ranges.append(tr)

    # Simple average of last 'period' TRs
    return sum(true_ranges[-period:]) / period


def compute_stochastic(
    candles: List[dict],
    k_period: int = 14,
    d_period: int = 3,
) -> Optional[dict]:
    """Compute Stochastic Oscillator (%K and %D).

    Returns dict with 'k', 'd', 'signal' (oversold/overbought/neutral)
    """
    if len(candles) < k_period:
        return None

    recent = candles[-k_period:]
    highest_high = max(c["high"] for c in recent)
    lowest_low = min(c["low"] for c in recent)
    current_close = candles[-1]["close"]

    if highest_high == lowest_low:
        k = 50.0
    else:
        k = ((current_close - lowest_low) / (highest_high - lowest_low)) * 100

    # %D is SMA of %K (simplified - just use current K for now)
    d = k  # Would need historical K values for proper D

    # Signal
    if k < 20:
        signal = "oversold"
    elif k > 80:
        signal = "overbought"
    else:
        signal = "neutral"

    return {"k": k, "d": d, "signal": signal}


def compute_williams_r(
    candles: List[dict],
    period: int = 14,
) -> Optional[float]:
    """Compute Williams %R (momentum indicator).

    Returns value from -100 to 0 (-80 to -100 = oversold, -20 to 0 = overbought)
    """
    if len(candles) < period:
        return None

    recent = candles[-period:]
    highest_high = max(c["high"] for c in recent)
    lowest_low = min(c["low"] for c in recent)
    current_close = candles[-1]["close"]

    if highest_high == lowest_low:
        return -50.0

    return ((highest_high - current_close) / (highest_high - lowest_low)) * -100


def compute_cci(
    candles: List[dict],
    period: int = 20,
) -> Optional[float]:
    """Compute Commodity Channel Index.

    Returns CCI value (>100 = overbought, <-100 = oversold)
    """
    if len(candles) < period:
        return None

    # Typical price = (High + Low + Close) / 3
    tps = [(c["high"] + c["low"] + c["close"]) / 3 for c in candles[-period:]]

    # SMA of typical prices
    sma_tp = sum(tps) / period

    # Mean deviation
    mean_dev = sum(abs(tp - sma_tp) for tp in tps) / period

    if mean_dev == 0:
        return 0.0

    current_tp = tps[-1]
    return (current_tp - sma_tp) / (0.015 * mean_dev)


def compute_obv_trend(
    candles: List[dict],
    period: int = 10,
) -> Optional[float]:
    """Compute On-Balance Volume trend.

    Returns the slope of OBV over the period (positive = bullish volume)
    """
    if len(candles) < period + 1:
        return None

    obv = 0
    obv_series = []

    for i in range(1, len(candles)):
        if candles[i]["close"] > candles[i - 1]["close"]:
            obv += candles[i].get("volume", 0)
        elif candles[i]["close"] < candles[i - 1]["close"]:
            obv -= candles[i].get("volume", 0)
        obv_series.append(obv)

    if len(obv_series) < period:
        return None

    # Return normalized slope
    recent = obv_series[-period:]
    slope = (recent[-1] - recent[0]) / period

    # Normalize by average volume
    avg_vol = sum(c.get("volume", 1) for c in candles[-period:]) / period
    return slope / avg_vol if avg_vol > 0 else 0


def compute_higher_timeframe_trend(
    closes_1m: List[float],
    closes_5m: List[float] = None,
) -> dict:
    """Determine trend on higher timeframes.

    Returns dict with trend signals for different timeframes.
    """
    result = {}

    # 15-minute trend (last 15 1m candles)
    if len(closes_1m) >= 15:
        change_15m = (closes_1m[-1] - closes_1m[-15]) / closes_1m[-15] * 100
        result["trend_15m"] = 1 if change_15m > 0 else -1
        result["change_15m"] = change_15m

    # 1-hour trend (last 60 1m candles)
    if len(closes_1m) >= 60:
        change_1h = (closes_1m[-1] - closes_1m[-60]) / closes_1m[-60] * 100
        result["trend_1h"] = 1 if change_1h > 0 else -1
        result["change_1h"] = change_1h

    # Use 5m candles if provided
    if closes_5m and len(closes_5m) >= 12:
        change_1h_5m = (closes_5m[-1] - closes_5m[-12]) / closes_5m[-12] * 100
        result["trend_1h_5m"] = 1 if change_1h_5m > 0 else -1

    return result


def compute_time_features(hour: int) -> dict:
    """Compute time-based features.

    Crypto markets have patterns based on time of day (UTC).
    """
    # Trading session flags
    asia_session = 0 <= hour < 8      # Asia: 00:00-08:00 UTC
    europe_session = 8 <= hour < 16   # Europe: 08:00-16:00 UTC
    us_session = 14 <= hour < 22      # US: 14:00-22:00 UTC (overlaps)

    # High volatility hours (US market open, Asia open)
    high_vol_hour = hour in [14, 15, 0, 1]  # US open, Asia open

    return {
        "hour": hour,
        "asia_session": 1 if asia_session else 0,
        "europe_session": 1 if europe_session else 0,
        "us_session": 1 if us_session else 0,
        "high_vol_hour": 1 if high_vol_hour else 0,
        "hour_sin": __import__("math").sin(2 * 3.14159 * hour / 24),  # Cyclical encoding
        "hour_cos": __import__("math").cos(2 * 3.14159 * hour / 24),
    }


def compute_candle_patterns(candles: List[dict]) -> dict:
    """Detect common candlestick patterns.

    Returns dict with pattern flags.
    """
    if len(candles) < 3:
        return {}

    c = candles[-1]  # Current candle
    p = candles[-2]  # Previous candle

    body = abs(c["close"] - c["open"])
    upper_wick = c["high"] - max(c["close"], c["open"])
    lower_wick = min(c["close"], c["open"]) - c["low"]
    total_range = c["high"] - c["low"]

    patterns = {}

    # Doji (small body, long wicks)
    if total_range > 0:
        patterns["doji"] = 1 if body / total_range < 0.1 else 0

        # Hammer (small body at top, long lower wick)
        patterns["hammer"] = 1 if (
            lower_wick > body * 2 and
            upper_wick < body * 0.5 and
            c["close"] > c["open"]
        ) else 0

        # Shooting star (small body at bottom, long upper wick)
        patterns["shooting_star"] = 1 if (
            upper_wick > body * 2 and
            lower_wick < body * 0.5 and
            c["close"] < c["open"]
        ) else 0

    # Engulfing patterns
    if p["close"] != p["open"]:
        bull_engulf = (
            c["close"] > c["open"] and  # Current is green
            p["close"] < p["open"] and  # Previous is red
            c["open"] < p["close"] and  # Current opens below prev close
            c["close"] > p["open"]      # Current closes above prev open
        )
        bear_engulf = (
            c["close"] < c["open"] and
            p["close"] > p["open"] and
            c["open"] > p["close"] and
            c["close"] < p["open"]
        )
        patterns["bullish_engulfing"] = 1 if bull_engulf else 0
        patterns["bearish_engulfing"] = 1 if bear_engulf else 0

    return patterns
