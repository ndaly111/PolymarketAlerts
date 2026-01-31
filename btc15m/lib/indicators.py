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
