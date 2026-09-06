"""Technical indicators calculation module for PSX Stock Agent."""

import math
from typing import List, Dict, Any, Optional, Tuple
from src.analysis.models import DerivedMetric


def calculate_sma(prices: List[float], window: int) -> Optional[float]:
    """Calculates Simple Moving Average over specified window.
    
    Returns None if price history is shorter than window.
    """
    if not prices or len(prices) < window or window <= 0:
        return None
    return sum(prices[-window:]) / float(window)


def calculate_ema(prices: List[float], window: int) -> List[float]:
    """Calculates Exponential Moving Average series for prices."""
    if not prices or len(prices) < window or window <= 0:
        return []
    
    k = 2.0 / (window + 1)
    # Seed with initial SMA
    ema_series = [sum(prices[:window]) / float(window)]
    
    for price in prices[window:]:
        ema_val = (price * k) + (ema_series[-1] * (1.0 - k))
        ema_series.append(ema_val)
        
    return ema_series


def calculate_rsi(prices: List[float], window: int = 14) -> Optional[float]:
    """Calculates 14-period Relative Strength Index (0 to 100).
    
    Uses Wilder's Smoothing Method. Returns None if history < window + 1.
    """
    if not prices or len(prices) < window + 1 or window <= 0:
        return None

    gains = []
    losses = []

    for i in range(1, len(prices)):
        change = prices[i] - prices[i - 1]
        if change > 0:
            gains.append(change)
            losses.append(0.0)
        else:
            gains.append(abs(change))
            losses.append(abs(change))

    if len(gains) < window:
        return None

    # First average gain and loss
    avg_gain = sum(gains[:window]) / float(window)
    avg_loss = sum(losses[:window]) / float(window)

    # Wilder's smoothing
    for i in range(window, len(gains)):
        avg_gain = (avg_gain * (window - 1) + gains[i]) / float(window)
        avg_loss = (avg_loss * (window - 1) + losses[i]) / float(window)

    if avg_loss == 0:
        return 100.0 if avg_gain > 0 else 50.0

    rs = avg_gain / avg_loss
    rsi = 100.0 - (100.0 / (1.0 + rs))
    return round(rsi, 4)


def calculate_macd(
    prices: List[float], 
    fast: int = 12, 
    slow: int = 26, 
    signal: int = 9
) -> Dict[str, Optional[float]]:
    """Calculates MACD Line, Signal Line, and Histogram.
    
    Formula:
      MACD Line = EMA(12) - EMA(26)
      Signal Line = EMA(9) of MACD Line
      Histogram = MACD Line - Signal Line
    """
    if not prices or len(prices) < slow + signal:
        return {"macd": None, "signal": None, "histogram": None}

    ema_fast = calculate_ema(prices, fast)
    ema_slow = calculate_ema(prices, slow)

    # Align EMA series lengths
    # ema_slow starts at index `slow - 1` relative to prices
    # ema_fast starts at index `fast - 1` relative to prices
    offset = slow - fast
    ema_fast_trimmed = ema_fast[offset:]

    macd_line = [f - s for f, s in zip(ema_fast_trimmed, ema_slow)]

    if len(macd_line) < signal:
        return {"macd": None, "signal": None, "histogram": None}

    signal_line_series = calculate_ema(macd_line, signal)

    latest_macd = macd_line[-1]
    latest_signal = signal_line_series[-1] if signal_line_series else None
    latest_hist = (latest_macd - latest_signal) if latest_signal is not None else None

    return {
        "macd": round(latest_macd, 4),
        "signal": round(latest_signal, 4) if latest_signal is not None else None,
        "histogram": round(latest_hist, 4) if latest_hist is not None else None,
    }


def calculate_52w_high_low_distance(
    prices: List[float], 
    dates: List[str]
) -> Dict[str, DerivedMetric]:
    """Calculates 52-week (252 trading days) High and Low distances from current price.
    
    Formula:
      dist_from_52w_high = ((Current_Price - 52w_High) / 52w_High) * 100
      dist_from_52w_low = ((Current_Price - 52w_Low) / 52w_Low) * 100
    """
    period_52w = 252
    if not prices:
        return {
            "dist_from_52w_high": DerivedMetric(
                metric_name="dist_from_52w_high",
                value=None,
                unit="%",
                calculation_period="52w (252d)",
                methodology="((Current_Close - 52w_High) / 52w_High) * 100",
                input_references=["raw_ohlcv_closes"],
                status="insufficient_data",
                notes="No price series available",
            ),
            "dist_from_52w_low": DerivedMetric(
                metric_name="dist_from_52w_low",
                value=None,
                unit="%",
                calculation_period="52w (252d)",
                methodology="((Current_Close - 52w_Low) / 52w_Low) * 100",
                input_references=["raw_ohlcv_closes"],
                status="insufficient_data",
                notes="No price series available",
            ),
        }

    prices_52w = prices[-period_52w:] if len(prices) >= period_52w else prices
    current_price = prices[-1]
    high_52w = max(prices_52w)
    low_52w = min(prices_52w)

    dist_high = ((current_price - high_52w) / high_52w) * 100.0 if high_52w > 0 else None
    dist_low = ((current_price - low_52w) / low_52w) * 100.0 if low_52w > 0 else None

    return {
        "dist_from_52w_high": DerivedMetric(
            metric_name="dist_from_52w_high",
            value=round(dist_high, 4) if dist_high is not None else None,
            unit="%",
            calculation_period=f"{len(prices_52w)}d",
            methodology="((Current_Close - 52w_High) / 52w_High) * 100",
            input_references=["raw_ohlcv_closes"],
            status="success",
        ),
        "dist_from_52w_low": DerivedMetric(
            metric_name="dist_from_52w_low",
            value=round(dist_low, 4) if dist_low is not None else None,
            unit="%",
            calculation_period=f"{len(prices_52w)}d",
            methodology="((Current_Close - 52w_Low) / 52w_Low) * 100",
            input_references=["raw_ohlcv_closes"],
            status="success",
        ),
    }


def compute_all_technical_metrics(
    closes: List[float], 
    highs: List[float], 
    lows: List[float], 
    dates: List[str]
) -> Dict[str, Dict[str, Any]]:
    """Master calculator for all technical indicators."""
    results = {}

    # 1. SMA 50
    sma50_val = calculate_sma(closes, 50)
    results["sma_50"] = DerivedMetric(
        metric_name="sma_50",
        value=round(sma50_val, 4) if sma50_val is not None else None,
        unit="PKR",
        calculation_period="50d",
        methodology="Simple Moving Average over 50 trading days",
        input_references=["raw_ohlcv_closes"],
        status="success" if sma50_val is not None else "insufficient_data",
        notes=None if sma50_val is not None else "Requires at least 50 close prices",
    ).to_dict()

    # 2. SMA 200
    sma200_val = calculate_sma(closes, 200)
    results["sma_200"] = DerivedMetric(
        metric_name="sma_200",
        value=round(sma200_val, 4) if sma200_val is not None else None,
        unit="PKR",
        calculation_period="200d",
        methodology="Simple Moving Average over 200 trading days",
        input_references=["raw_ohlcv_closes"],
        status="success" if sma200_val is not None else "insufficient_data",
        notes=None if sma200_val is not None else "Requires at least 200 close prices",
    ).to_dict()

    # 3. RSI 14
    rsi_val = calculate_rsi(closes, 14)
    results["rsi_14"] = DerivedMetric(
        metric_name="rsi_14",
        value=rsi_val,
        unit="index (0-100)",
        calculation_period="14d",
        methodology="14-period Relative Strength Index using Wilder's smoothing",
        input_references=["raw_ohlcv_closes"],
        status="success" if rsi_val is not None else "insufficient_data",
        notes=None if rsi_val is not None else "Requires at least 15 close prices",
    ).to_dict()

    # 4. MACD
    macd_dict = calculate_macd(closes, 12, 26, 9)
    results["macd"] = DerivedMetric(
        metric_name="macd",
        value=macd_dict["macd"],
        unit="points",
        calculation_period="12-26-9d",
        methodology="EMA(12) - EMA(26)",
        input_references=["raw_ohlcv_closes"],
        status="success" if macd_dict["macd"] is not None else "insufficient_data",
        notes=f"Signal: {macd_dict['signal']}, Histogram: {macd_dict['histogram']}",
    ).to_dict()

    # 5. 52W High/Low Distances
    dist_dict = calculate_52w_high_low_distance(closes, dates)
    results["dist_from_52w_high"] = dist_dict["dist_from_52w_high"].to_dict()
    results["dist_from_52w_low"] = dist_dict["dist_from_52w_low"].to_dict()

    return results
