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


# ============================================================
# Additional Indicators - Batch 2
# ============================================================

def compute_adx(candles: List[dict], period: int = 14) -> Optional[dict]:
    """Compute Average Directional Index (trend strength).

    ADX > 25 = trending market
    ADX < 20 = ranging market
    +DI > -DI = bullish trend
    -DI > +DI = bearish trend
    """
    if len(candles) < period + 1:
        return None

    plus_dm = []
    minus_dm = []
    tr_list = []

    for i in range(1, len(candles)):
        high = candles[i]["high"]
        low = candles[i]["low"]
        prev_high = candles[i - 1]["high"]
        prev_low = candles[i - 1]["low"]
        prev_close = candles[i - 1]["close"]

        # True Range
        tr = max(high - low, abs(high - prev_close), abs(low - prev_close))
        tr_list.append(tr)

        # Directional Movement
        up_move = high - prev_high
        down_move = prev_low - low

        plus_dm.append(up_move if up_move > down_move and up_move > 0 else 0)
        minus_dm.append(down_move if down_move > up_move and down_move > 0 else 0)

    if len(tr_list) < period:
        return None

    # Smoothed averages (Wilder's smoothing)
    atr = sum(tr_list[-period:]) / period
    plus_di = (sum(plus_dm[-period:]) / period) / atr * 100 if atr > 0 else 0
    minus_di = (sum(minus_dm[-period:]) / period) / atr * 100 if atr > 0 else 0

    # DX and ADX
    di_sum = plus_di + minus_di
    dx = abs(plus_di - minus_di) / di_sum * 100 if di_sum > 0 else 0

    return {
        "adx": dx,  # Simplified - real ADX needs smoothing
        "plus_di": plus_di,
        "minus_di": minus_di,
        "trend_strength": "strong" if dx > 25 else "weak",
        "trend_direction": "bullish" if plus_di > minus_di else "bearish",
    }


def compute_mfi(candles: List[dict], period: int = 14) -> Optional[float]:
    """Compute Money Flow Index (volume-weighted RSI).

    MFI > 80 = overbought
    MFI < 20 = oversold
    """
    if len(candles) < period + 1:
        return None

    pos_flow = 0
    neg_flow = 0

    for i in range(len(candles) - period, len(candles)):
        tp = (candles[i]["high"] + candles[i]["low"] + candles[i]["close"]) / 3
        prev_tp = (candles[i-1]["high"] + candles[i-1]["low"] + candles[i-1]["close"]) / 3
        raw_mf = tp * candles[i].get("volume", 0)

        if tp > prev_tp:
            pos_flow += raw_mf
        else:
            neg_flow += raw_mf

    if neg_flow == 0:
        return 100.0
    if pos_flow == 0:
        return 0.0

    mf_ratio = pos_flow / neg_flow
    mfi = 100 - (100 / (1 + mf_ratio))

    return max(0, min(100, mfi))


def compute_roc(closes: List[float], period: int = 10) -> Optional[float]:
    """Compute Rate of Change (momentum).

    ROC > 0 = bullish momentum
    ROC < 0 = bearish momentum
    """
    if len(closes) < period + 1:
        return None

    return ((closes[-1] - closes[-period - 1]) / closes[-period - 1]) * 100


def compute_trix(closes: List[float], period: int = 15) -> Optional[float]:
    """Compute TRIX (triple smoothed momentum).

    TRIX > 0 = bullish
    TRIX < 0 = bearish
    Shows rate of change of triple-smoothed EMA.
    """
    if len(closes) < period * 3 + 1:
        return None

    # First EMA
    ema1 = _ema_series(closes, period)
    if not ema1 or ema1[-1] is None:
        return None

    # Second EMA (of first)
    ema1_clean = [x for x in ema1 if x is not None]
    ema2 = _ema_series(ema1_clean, period)
    if not ema2 or ema2[-1] is None:
        return None

    # Third EMA (of second)
    ema2_clean = [x for x in ema2 if x is not None]
    ema3 = _ema_series(ema2_clean, period)
    if not ema3 or len(ema3) < 2 or ema3[-1] is None or ema3[-2] is None:
        return None

    # TRIX = percent change of triple EMA
    return ((ema3[-1] - ema3[-2]) / ema3[-2]) * 100


def compute_keltner_channels(
    candles: List[dict],
    ema_period: int = 20,
    atr_period: int = 10,
    multiplier: float = 2.0,
) -> Optional[dict]:
    """Compute Keltner Channels.

    Similar to Bollinger Bands but uses ATR instead of std dev.
    More stable for trend following.
    """
    if len(candles) < max(ema_period, atr_period + 1):
        return None

    closes = [c["close"] for c in candles]
    ema = _ema(closes, ema_period)
    atr = compute_atr(candles, atr_period)

    if ema is None or atr is None:
        return None

    upper = ema + (multiplier * atr)
    lower = ema - (multiplier * atr)
    current = closes[-1]

    # Position within channel
    if upper != lower:
        position = (current - lower) / (upper - lower)
    else:
        position = 0.5

    return {
        "kc_upper": upper,
        "kc_middle": ema,
        "kc_lower": lower,
        "kc_position": position,  # 0-1, >1 above upper, <0 below lower
        "kc_width": (upper - lower) / ema if ema > 0 else 0,
    }


def compute_donchian_channels(
    candles: List[dict],
    period: int = 20,
) -> Optional[dict]:
    """Compute Donchian Channels (breakout indicator).

    Price at upper = potential breakout up
    Price at lower = potential breakout down
    """
    if len(candles) < period:
        return None

    recent = candles[-period:]
    highest = max(c["high"] for c in recent)
    lowest = min(c["low"] for c in recent)
    middle = (highest + lowest) / 2
    current = candles[-1]["close"]

    # Position within channel
    if highest != lowest:
        position = (current - lowest) / (highest - lowest)
    else:
        position = 0.5

    return {
        "dc_upper": highest,
        "dc_middle": middle,
        "dc_lower": lowest,
        "dc_position": position,
        "dc_width": (highest - lowest) / middle if middle > 0 else 0,
    }


def compute_ichimoku(candles: List[dict]) -> Optional[dict]:
    """Compute Ichimoku Cloud components.

    Tenkan/Kijun cross = trend signal
    Price vs Cloud = trend direction
    """
    if len(candles) < 52:
        return None

    def midpoint(data, period):
        recent = data[-period:]
        return (max(c["high"] for c in recent) + min(c["low"] for c in recent)) / 2

    # Tenkan-sen (Conversion Line) - 9 period
    tenkan = midpoint(candles, 9)

    # Kijun-sen (Base Line) - 26 period
    kijun = midpoint(candles, 26)

    # Senkou Span A (Leading Span A) - midpoint of Tenkan/Kijun
    senkou_a = (tenkan + kijun) / 2

    # Senkou Span B (Leading Span B) - 52 period midpoint
    senkou_b = midpoint(candles, 52)

    current = candles[-1]["close"]

    # Cloud analysis
    cloud_top = max(senkou_a, senkou_b)
    cloud_bottom = min(senkou_a, senkou_b)

    above_cloud = current > cloud_top
    below_cloud = current < cloud_bottom
    in_cloud = not above_cloud and not below_cloud

    return {
        "ichi_tenkan": tenkan,
        "ichi_kijun": kijun,
        "ichi_senkou_a": senkou_a,
        "ichi_senkou_b": senkou_b,
        "ichi_tk_cross": 1 if tenkan > kijun else -1,  # 1 = bullish, -1 = bearish
        "ichi_above_cloud": 1 if above_cloud else 0,
        "ichi_below_cloud": 1 if below_cloud else 0,
        "ichi_in_cloud": 1 if in_cloud else 0,
        "ichi_cloud_thickness": (cloud_top - cloud_bottom) / current * 100,
    }


def compute_vwma(candles: List[dict], period: int = 20) -> Optional[float]:
    """Compute Volume Weighted Moving Average.

    Price above VWMA = bullish
    Price below VWMA = bearish
    """
    if len(candles) < period:
        return None

    recent = candles[-period:]
    numerator = sum(c["close"] * c.get("volume", 1) for c in recent)
    denominator = sum(c.get("volume", 1) for c in recent)

    if denominator == 0:
        return None

    return numerator / denominator


def compute_cmf(candles: List[dict], period: int = 20) -> Optional[float]:
    """Compute Chaikin Money Flow.

    CMF > 0 = buying pressure
    CMF < 0 = selling pressure
    """
    if len(candles) < period:
        return None

    recent = candles[-period:]
    mfv_sum = 0
    vol_sum = 0

    for c in recent:
        high, low, close = c["high"], c["low"], c["close"]
        volume = c.get("volume", 0)

        if high != low:
            mf_multiplier = ((close - low) - (high - close)) / (high - low)
        else:
            mf_multiplier = 0

        mfv_sum += mf_multiplier * volume
        vol_sum += volume

    if vol_sum == 0:
        return 0

    return mfv_sum / vol_sum


def compute_chop(candles: List[dict], period: int = 14) -> Optional[float]:
    """Compute Choppiness Index.

    High CHOP (>61.8) = consolidating/ranging
    Low CHOP (<38.2) = trending
    """
    if len(candles) < period + 1:
        return None

    recent = candles[-period:]

    # Sum of ATRs
    atr_sum = 0
    for i in range(1, len(recent)):
        tr = max(
            recent[i]["high"] - recent[i]["low"],
            abs(recent[i]["high"] - recent[i-1]["close"]),
            abs(recent[i]["low"] - recent[i-1]["close"])
        )
        atr_sum += tr

    # Highest high - lowest low
    highest = max(c["high"] for c in recent)
    lowest = min(c["low"] for c in recent)
    hl_range = highest - lowest

    if hl_range == 0 or atr_sum <= 0:
        return 50

    ratio = atr_sum / hl_range
    if ratio <= 0:
        return 50

    import math
    try:
        chop = 100 * math.log10(ratio) / math.log10(period)
    except ValueError:
        return 50

    return max(0, min(100, chop))


def compute_squeeze(candles: List[dict]) -> Optional[dict]:
    """Detect Bollinger Band / Keltner Channel squeeze.

    Squeeze on = low volatility, breakout imminent
    Squeeze off = high volatility, trending
    """
    bb = compute_bollinger_bands([c["close"] for c in candles], period=20, std_dev=2.0)
    kc = compute_keltner_channels(candles, ema_period=20, atr_period=10, multiplier=1.5)

    if not bb or not kc:
        return None

    # Squeeze is on when BB is inside KC
    squeeze_on = bb["lower"] > kc["kc_lower"] and bb["upper"] < kc["kc_upper"]

    # Momentum (using simple linear regression of close prices)
    closes = [c["close"] for c in candles[-20:]]
    n = len(closes)
    if n < 5:
        return None

    # Simple momentum: difference from SMA
    sma = sum(closes) / n
    momentum = closes[-1] - sma

    return {
        "squeeze_on": 1 if squeeze_on else 0,
        "squeeze_momentum": momentum,
        "squeeze_direction": 1 if momentum > 0 else -1,
    }


def compute_volume_features(candles: List[dict], period: int = 20) -> Optional[dict]:
    """Compute various volume-based features."""
    if len(candles) < period:
        return None

    volumes = [c.get("volume", 0) for c in candles]
    recent_vols = volumes[-period:]

    avg_vol = sum(recent_vols) / period
    current_vol = volumes[-1]

    # Volume ratio
    vol_ratio = current_vol / avg_vol if avg_vol > 0 else 1

    # Volume trend (rising or falling)
    first_half = sum(recent_vols[:period//2]) / (period // 2)
    second_half = sum(recent_vols[period//2:]) / (period - period // 2)
    vol_trend = (second_half - first_half) / first_half if first_half > 0 else 0

    # Up vs down volume
    up_vol = sum(
        c.get("volume", 0) for c in candles[-period:]
        if c["close"] > c.get("open", c["close"])
    )
    down_vol = sum(
        c.get("volume", 0) for c in candles[-period:]
        if c["close"] < c.get("open", c["close"])
    )
    total_vol = up_vol + down_vol

    return {
        "vol_ratio": vol_ratio,  # Current vs average
        "vol_trend": vol_trend,  # Trend direction
        "vol_up_ratio": up_vol / total_vol if total_vol > 0 else 0.5,
        "vol_down_ratio": down_vol / total_vol if total_vol > 0 else 0.5,
        "vol_spike": 1 if vol_ratio > 2 else 0,
    }


def compute_price_action_features(candles: List[dict], period: int = 20) -> Optional[dict]:
    """Compute price action based features."""
    if len(candles) < period:
        return None

    closes = [c["close"] for c in candles]
    highs = [c["high"] for c in candles]
    lows = [c["low"] for c in candles]

    # Support/Resistance levels
    recent_high = max(highs[-period:])
    recent_low = min(lows[-period:])
    current = closes[-1]

    # Distance from support/resistance
    dist_to_high = (recent_high - current) / current * 100
    dist_to_low = (current - recent_low) / current * 100

    # Range compression (narrowing range = breakout imminent)
    range_20 = recent_high - recent_low
    range_5 = max(highs[-5:]) - min(lows[-5:])
    range_ratio = range_5 / range_20 if range_20 > 0 else 1

    # Higher highs / Lower lows trend
    hh_count = sum(1 for i in range(-5, 0) if highs[i] > highs[i-1])
    ll_count = sum(1 for i in range(-5, 0) if lows[i] < lows[i-1])

    # Consecutive up/down candles
    up_streak = 0
    down_streak = 0
    for i in range(-1, -min(10, len(candles)), -1):
        if candles[i]["close"] > candles[i]["open"]:
            if down_streak == 0:
                up_streak += 1
            else:
                break
        else:
            if up_streak == 0:
                down_streak += 1
            else:
                break

    return {
        "dist_to_high_pct": dist_to_high,
        "dist_to_low_pct": dist_to_low,
        "range_compression": range_ratio,  # <0.5 = compressed
        "higher_highs": hh_count,
        "lower_lows": ll_count,
        "up_streak": up_streak,
        "down_streak": down_streak,
    }


def compute_pivot_points(candles: List[dict]) -> Optional[dict]:
    """Compute pivot points (support/resistance levels).

    Classic pivot point formula using previous day's OHLC.
    """
    if len(candles) < 2:
        return None

    # Use recent candles as "previous period"
    prev = candles[-2]
    high = prev["high"]
    low = prev["low"]
    close = prev["close"]
    current = candles[-1]["close"]

    pivot = (high + low + close) / 3
    r1 = 2 * pivot - low
    s1 = 2 * pivot - high
    r2 = pivot + (high - low)
    s2 = pivot - (high - low)

    # Position relative to pivot
    above_pivot = current > pivot

    # Nearest level
    levels = [("S2", s2), ("S1", s1), ("PP", pivot), ("R1", r1), ("R2", r2)]
    distances = [(name, level, abs(current - level)) for name, level in levels]
    nearest = min(distances, key=lambda x: x[2])

    return {
        "pivot": pivot,
        "pivot_r1": r1,
        "pivot_r2": r2,
        "pivot_s1": s1,
        "pivot_s2": s2,
        "above_pivot": 1 if above_pivot else 0,
        "nearest_level": nearest[0],
        "dist_to_nearest": nearest[2] / current * 100,
    }


def compute_linear_regression(closes: List[float], period: int = 20) -> Optional[dict]:
    """Compute linear regression trend."""
    if len(closes) < period:
        return None

    recent = closes[-period:]
    n = len(recent)

    # Linear regression
    x_sum = sum(range(n))
    y_sum = sum(recent)
    xy_sum = sum(i * recent[i] for i in range(n))
    x2_sum = sum(i ** 2 for i in range(n))

    denom = n * x2_sum - x_sum ** 2
    if denom == 0:
        return None

    slope = (n * xy_sum - x_sum * y_sum) / denom
    intercept = (y_sum - slope * x_sum) / n

    # Predicted value and deviation
    predicted = intercept + slope * (n - 1)
    current = recent[-1]
    deviation = (current - predicted) / predicted * 100

    # R-squared (goodness of fit)
    y_mean = y_sum / n
    ss_tot = sum((y - y_mean) ** 2 for y in recent)
    ss_res = sum((recent[i] - (intercept + slope * i)) ** 2 for i in range(n))
    r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

    return {
        "lr_slope": slope / current * 100,  # Normalized slope
        "lr_deviation": deviation,
        "lr_r_squared": r_squared,
        "lr_trend": "up" if slope > 0 else "down",
    }


# ============================================================
# COMBINATION INDICATORS - Composite Signals
# ============================================================

def compute_trend_strength_composite(
    adx_value: Optional[float],
    rsi: Optional[float],
    macd_hist: Optional[float],
    momentum: Optional[float],
) -> Optional[dict]:
    """Combine multiple indicators to measure trend strength.

    Returns a composite score from -100 (strong bearish) to +100 (strong bullish).
    """
    if adx_value is None or rsi is None:
        return None

    signals = []

    # RSI signal: >70 bullish, <30 bearish
    if rsi is not None:
        rsi_signal = (rsi - 50) / 50  # Normalize to -1 to +1
        signals.append(rsi_signal)

    # MACD histogram signal
    if macd_hist is not None:
        # Normalize based on typical range
        macd_signal = max(-1, min(1, macd_hist / 100))
        signals.append(macd_signal)

    # Momentum signal
    if momentum is not None:
        mom_signal = max(-1, min(1, momentum / 5))  # 5% as max
        signals.append(mom_signal)

    if not signals:
        return None

    # Average signal
    avg_signal = sum(signals) / len(signals)

    # Weight by ADX (trend strength)
    adx_weight = min(1, adx_value / 50) if adx_value else 0.5

    return {
        "trend_composite": avg_signal * 100,
        "trend_strength": adx_weight,
        "weighted_trend": avg_signal * adx_weight * 100,
    }


def compute_overbought_oversold_composite(
    rsi: Optional[float],
    stoch_k: Optional[float],
    williams_r: Optional[float],
    mfi: Optional[float],
    cci: Optional[float],
) -> Optional[dict]:
    """Combine overbought/oversold indicators.

    Consensus signal is stronger than individual indicators.
    """
    signals = []

    # RSI: >70 overbought, <30 oversold
    if rsi is not None:
        if rsi > 70:
            signals.append(1)  # Overbought
        elif rsi < 30:
            signals.append(-1)  # Oversold
        else:
            signals.append(0)

    # Stochastic: >80 overbought, <20 oversold
    if stoch_k is not None:
        if stoch_k > 80:
            signals.append(1)
        elif stoch_k < 20:
            signals.append(-1)
        else:
            signals.append(0)

    # Williams %R: >-20 overbought, <-80 oversold
    if williams_r is not None:
        if williams_r > -20:
            signals.append(1)
        elif williams_r < -80:
            signals.append(-1)
        else:
            signals.append(0)

    # MFI: >80 overbought, <20 oversold
    if mfi is not None:
        if mfi > 80:
            signals.append(1)
        elif mfi < 20:
            signals.append(-1)
        else:
            signals.append(0)

    # CCI: >100 overbought, <-100 oversold
    if cci is not None:
        if cci > 100:
            signals.append(1)
        elif cci < -100:
            signals.append(-1)
        else:
            signals.append(0)

    if not signals:
        return None

    # Count consensus
    overbought_count = sum(1 for s in signals if s == 1)
    oversold_count = sum(1 for s in signals if s == -1)
    total = len(signals)

    return {
        "ob_os_signal": sum(signals),  # Positive = overbought, negative = oversold
        "overbought_consensus": overbought_count / total,
        "oversold_consensus": oversold_count / total,
        "ob_os_strength": abs(sum(signals)) / total,
    }


def compute_volatility_composite(
    atr: Optional[float],
    bb_bandwidth: Optional[float],
    kc_width: Optional[float],
    chop: Optional[float],
    price: float,
) -> Optional[dict]:
    """Combine volatility indicators.

    High volatility = larger price moves expected.
    """
    signals = []

    # ATR relative to price
    if atr is not None and price > 0:
        atr_pct = atr / price * 100
        signals.append(min(1, atr_pct / 2))  # 2% ATR = max

    # Bollinger bandwidth
    if bb_bandwidth is not None:
        signals.append(min(1, bb_bandwidth / 0.05))  # 5% bandwidth = max

    # Keltner width
    if kc_width is not None:
        signals.append(min(1, kc_width / 0.05))

    if not signals:
        return None

    avg_vol = sum(signals) / len(signals)

    # Choppiness indicates consolidation
    is_ranging = chop is not None and chop > 61.8
    is_trending = chop is not None and chop < 38.2

    return {
        "volatility_composite": avg_vol,
        "is_ranging": 1 if is_ranging else 0,
        "is_trending": 1 if is_trending else 0,
        "breakout_potential": avg_vol if is_ranging else 0,
    }


def compute_volume_price_composite(
    vol_ratio: Optional[float],
    obv_trend: Optional[float],
    cmf: Optional[float],
    price_change: float,
) -> Optional[dict]:
    """Combine volume and price indicators.

    Volume confirming price = stronger signal.
    """
    if vol_ratio is None:
        return None

    # Volume confirms price direction?
    price_up = price_change > 0
    volume_high = vol_ratio > 1.5

    # OBV confirms direction?
    obv_confirms = obv_trend is not None and (
        (price_up and obv_trend > 0) or
        (not price_up and obv_trend < 0)
    )

    # CMF confirms direction?
    cmf_confirms = cmf is not None and (
        (price_up and cmf > 0.1) or
        (not price_up and cmf < -0.1)
    )

    confirmation_count = sum([
        volume_high,
        obv_confirms,
        cmf_confirms,
    ])

    return {
        "vol_price_confirmation": confirmation_count / 3,
        "vol_price_divergence": 1 if (volume_high and not obv_confirms) else 0,
        "strong_move": 1 if (volume_high and confirmation_count >= 2) else 0,
    }


def compute_multi_timeframe_composite(
    trend_15m: Optional[int],
    trend_1h: Optional[int],
    change_15m: Optional[float],
    change_1h: Optional[float],
) -> Optional[dict]:
    """Combine multiple timeframe trends.

    Aligned trends = stronger signal.
    """
    if trend_15m is None and trend_1h is None:
        return None

    # Trend alignment
    aligned = trend_15m is not None and trend_1h is not None and trend_15m == trend_1h

    # Momentum acceleration
    accel = None
    if change_15m is not None and change_1h is not None:
        # Is recent move faster than overall trend?
        # 15m change should be roughly 1/4 of 1h change if constant
        expected_15m = change_1h / 4 if change_1h else 0
        accel = change_15m - expected_15m if expected_15m != 0 else 0

    return {
        "mtf_aligned": 1 if aligned else 0,
        "mtf_direction": trend_15m or trend_1h or 0,
        "mtf_acceleration": accel,
        "mtf_strength": abs(change_15m or 0) + abs(change_1h or 0) / 4,
    }


def compute_breakout_probability(
    squeeze_on: Optional[int],
    range_compression: Optional[float],
    vol_ratio: Optional[float],
    bb_percent_b: Optional[float],
    dc_position: Optional[float],
) -> Optional[dict]:
    """Estimate probability of breakout.

    Combines volatility compression with volume buildup.
    """
    signals = []

    # Squeeze indicator
    if squeeze_on is not None:
        signals.append(squeeze_on)

    # Range compression (<0.5 = compressed)
    if range_compression is not None:
        signals.append(1 if range_compression < 0.5 else 0)

    # Volume building up during compression
    if vol_ratio is not None:
        signals.append(1 if vol_ratio > 1.2 else 0)

    # Price near channel extremes
    near_breakout = False
    if bb_percent_b is not None:
        near_breakout = bb_percent_b > 0.95 or bb_percent_b < 0.05
    if dc_position is not None:
        near_breakout = near_breakout or dc_position > 0.95 or dc_position < 0.05

    if near_breakout:
        signals.append(1)

    if not signals:
        return None

    # Direction hint
    direction = 0
    if bb_percent_b is not None:
        direction = 1 if bb_percent_b > 0.5 else -1

    return {
        "breakout_score": sum(signals) / len(signals),
        "breakout_direction": direction,
        "compression_building": 1 if sum(signals) >= 2 else 0,
    }


def compute_momentum_divergence(
    closes: List[float],
    rsi_values: List[Optional[float]] = None,
    period: int = 14,
) -> Optional[dict]:
    """Detect momentum divergence (price vs RSI).

    Bullish divergence: price makes lower low, RSI makes higher low
    Bearish divergence: price makes higher high, RSI makes lower high
    """
    if len(closes) < period:
        return None

    # Simple divergence detection using recent price and computed RSI
    recent_closes = closes[-period:]
    rsi = compute_rsi(closes, period=14)
    if rsi is None:
        return None

    # Check for lower lows in price
    price_min_idx = recent_closes.index(min(recent_closes))
    price_max_idx = recent_closes.index(max(recent_closes))

    # Simplified divergence: compare first half vs second half
    first_half = recent_closes[:period//2]
    second_half = recent_closes[period//2:]

    price_trend = 1 if sum(second_half)/len(second_half) > sum(first_half)/len(first_half) else -1

    # RSI trend would need historical values - simplify for now
    rsi_trend = 1 if rsi > 50 else -1

    divergence = price_trend != rsi_trend

    return {
        "has_divergence": 1 if divergence else 0,
        "divergence_type": "bullish" if (price_trend < 0 and rsi_trend > 0) else (
            "bearish" if (price_trend > 0 and rsi_trend < 0) else "none"
        ),
        "divergence_strength": abs(rsi - 50) / 50 if divergence else 0,
    }


def compute_all_composites(
    closes: List[float],
    candles: List[dict],
    indicators: dict,
) -> dict:
    """Compute all composite indicators at once.

    Args:
        closes: List of close prices
        candles: List of candle dicts
        indicators: Dict with individual indicator values

    Returns:
        Dict with all composite values
    """
    result = {}

    # Trend strength composite
    trend_comp = compute_trend_strength_composite(
        indicators.get("adx"),
        indicators.get("rsi"),
        indicators.get("macd_histogram"),
        indicators.get("momentum"),
    )
    if trend_comp:
        result.update({f"comp_{k}": v for k, v in trend_comp.items()})

    # Overbought/oversold composite
    obos_comp = compute_overbought_oversold_composite(
        indicators.get("rsi"),
        indicators.get("stoch_k"),
        indicators.get("williams_r"),
        indicators.get("mfi"),
        indicators.get("cci"),
    )
    if obos_comp:
        result.update({f"comp_{k}": v for k, v in obos_comp.items()})

    # Volatility composite
    vol_comp = compute_volatility_composite(
        indicators.get("atr"),
        indicators.get("bb_bandwidth"),
        indicators.get("kc_width"),
        indicators.get("chop"),
        closes[-1] if closes else 0,
    )
    if vol_comp:
        result.update({f"comp_{k}": v for k, v in vol_comp.items()})

    # Volume-price composite
    vp_comp = compute_volume_price_composite(
        indicators.get("vol_ratio"),
        indicators.get("obv_trend"),
        indicators.get("cmf"),
        indicators.get("price_change", 0),
    )
    if vp_comp:
        result.update({f"comp_{k}": v for k, v in vp_comp.items()})

    # Multi-timeframe composite
    mtf_comp = compute_multi_timeframe_composite(
        indicators.get("trend_15m"),
        indicators.get("trend_1h"),
        indicators.get("change_15m"),
        indicators.get("change_1h"),
    )
    if mtf_comp:
        result.update({f"comp_{k}": v for k, v in mtf_comp.items()})

    # Breakout probability
    bo_comp = compute_breakout_probability(
        indicators.get("squeeze_on"),
        indicators.get("range_compression"),
        indicators.get("vol_ratio"),
        indicators.get("bb_percent_b"),
        indicators.get("dc_position"),
    )
    if bo_comp:
        result.update({f"comp_{k}": v for k, v in bo_comp.items()})

    # Momentum divergence
    div_comp = compute_momentum_divergence(closes)
    if div_comp:
        result.update({f"comp_{k}": v for k, v in div_comp.items()})

    return result
