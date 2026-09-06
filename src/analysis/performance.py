from datetime import datetime, timedelta
import math
from typing import List, Dict, Any, Optional
from src.analysis.models import DerivedMetric


def _parse_trading_date(dt_str: str) -> Optional[datetime]:
    """Parses date string into a datetime object using standard PSX date formats."""
    if not dt_str or not isinstance(dt_str, str):
        return None
    for fmt in ["%Y-%m-%d", "%d-%m-%Y", "%Y/%m/%d", "%d/%m/%Y"]:
        try:
            return datetime.strptime(dt_str.strip(), fmt)
        except ValueError:
            continue
    return None


def calculate_period_return_by_date(
    prices: List[float], 
    dates: List[str], 
    years: float = 1.0,
    max_date_tolerance_days: int = 45
) -> Dict[str, Any]:
    """Calculates percentage return based on actual calendar date difference.
    
    Guarantees that a multi-year return (e.g. 3Y) is NEVER silently calculated from
    short datasets (e.g. 1Y). Returns status='insufficient_data' if coverage is inadequate.
    """
    if not prices or not dates or len(prices) != len(dates) or len(prices) < 2:
        return {
            "value": None,
            "start_date": None,
            "end_date": None,
            "start_price": None,
            "end_price": None,
            "status": "insufficient_data",
            "notes": "Insufficient price or date records"
        }

    end_dt = _parse_trading_date(dates[-1])
    earliest_dt = _parse_trading_date(dates[0])
    if not end_dt or not earliest_dt:
        return {
            "value": None,
            "start_date": None,
            "end_date": None,
            "start_price": None,
            "end_price": None,
            "status": "insufficient_data",
            "notes": "Invalid date format in OHLCV series"
        }

    target_dt = end_dt - timedelta(days=int(years * 365.25))

    # Strict coverage check: if earliest available date is later than target_dt (+ tolerance),
    # then history does NOT cover the requested time window. NEVER fall back to oldest price!
    if earliest_dt > (target_dt + timedelta(days=max_date_tolerance_days)):
        coverage_days = (end_dt - earliest_dt).days
        return {
            "value": None,
            "start_date": None,
            "end_date": dates[-1],
            "start_price": None,
            "end_price": prices[-1],
            "status": "insufficient_data",
            "notes": f"Dataset covers only {coverage_days} calendar days ({round(coverage_days / 365.25, 1)} years) from {dates[0]} to {dates[-1]}. Insufficient historical coverage for requested {years}-year return."
        }

    # Find the trading observation closest to target_dt
    best_idx = None
    min_diff = None

    for i, d_str in enumerate(dates):
        d_parsed = _parse_trading_date(d_str)
        if d_parsed:
            diff = abs((d_parsed - target_dt).days)
            if min_diff is None or diff < min_diff:
                min_diff = diff
                best_idx = i

    if best_idx is None or min_diff > max_date_tolerance_days:
        return {
            "value": None,
            "start_date": None,
            "end_date": dates[-1],
            "start_price": None,
            "end_price": prices[-1],
            "status": "insufficient_data",
            "notes": f"No trading observation found within {max_date_tolerance_days} days of target date {target_dt.strftime('%Y-%m-%d')}"
        }

    start_price = prices[best_idx]
    latest_price = prices[-1]

    if start_price <= 0:
        return {
            "value": None,
            "start_date": dates[best_idx],
            "end_date": dates[-1],
            "start_price": start_price,
            "end_price": latest_price,
            "status": "insufficient_data",
            "notes": "Start price is zero or negative"
        }

    ret_pct = ((latest_price - start_price) / start_price) * 100.0

    return {
        "value": round(ret_pct, 4),
        "start_date": dates[best_idx],
        "end_date": dates[-1],
        "start_price": start_price,
        "end_price": latest_price,
        "status": "success",
        "notes": f"{years}Y Return calculated from {dates[best_idx]} (PKR {start_price}) to {dates[-1]} (PKR {latest_price})"
    }


def calculate_period_return(
    prices: List[float], 
    dates: List[str], 
    target_days: int
) -> Optional[float]:
    """Calculates percentage return over target trading days using calendar date resolution."""
    years = target_days / 252.0
    res = calculate_period_return_by_date(prices, dates, years=years)
    return res["value"]


def calculate_cagr(
    prices: List[float], 
    dates: List[str], 
    years: float = 5.0
) -> Optional[float]:
    """Calculates Compound Annual Growth Rate (CAGR) over specified years.
    
    Formula: ((P_latest / P_start) ** (1 / years)) - 1.0
    Requires adequate multi-year calendar coverage.
    """
    res = calculate_period_return_by_date(prices, dates, years=years)
    if res["value"] is None or res["start_price"] is None or res["start_price"] <= 0:
        return None

    start_price = res["start_price"]
    latest_price = res["end_price"]
    cagr_ratio = (latest_price / start_price) ** (1.0 / years) - 1.0
    return cagr_ratio * 100.0


def calculate_max_drawdown(prices: List[float]) -> Optional[float]:
    """Calculates Maximum Percentage Peak-to-Trough Drawdown over the price series.
    
    Formula: min((P_i - Peak_t) / Peak_t) * 100
    Always returns a non-positive percentage (e.g. -24.5%).
    """
    if not prices or len(prices) < 2:
        return None

    peak = prices[0]
    max_dd = 0.0

    for price in prices:
        if price > peak:
            peak = price
        if peak > 0:
            dd = ((price - peak) / peak) * 100.0
            if dd < max_dd:
                max_dd = dd

    return max_dd


def calculate_annualized_volatility(prices: List[float]) -> Optional[float]:
    """Calculates Annualized Price Volatility from daily percentage returns.
    
    Formula: StdDev(daily_returns) * sqrt(252) * 100
    """
    if not prices or len(prices) < 5:
        return None

    daily_returns = []
    for i in range(1, len(prices)):
        if prices[i - 1] > 0:
            ret = (prices[i] - prices[i - 1]) / prices[i - 1]
            daily_returns.append(ret)

    if not daily_returns:
        return None

    mean_ret = sum(daily_returns) / len(daily_returns)
    variance = sum((r - mean_ret) ** 2 for r in daily_returns) / (len(daily_returns) - 1 if len(daily_returns) > 1 else 1)
    std_dev = math.sqrt(variance)

    annualized_vol = std_dev * math.sqrt(252) * 100.0
    return annualized_vol


def compute_all_performance_metrics(
    closes: List[float], 
    dates: List[str]
) -> Dict[str, Dict[str, Any]]:
    """Master calculator for all price performance and risk metrics."""
    results = {}

    # 1. 1Y Return (1 calendar year)
    res_1y = calculate_period_return_by_date(closes, dates, years=1.0)
    results["return_1y"] = DerivedMetric(
        metric_name="return_1y",
        value=res_1y["value"],
        unit="%",
        calculation_period="1y",
        methodology="((P_latest - P_1y_ago) / P_1y_ago) * 100",
        input_references=["raw_ohlcv_closes"],
        status=res_1y["status"],
        notes=res_1y["notes"],
    ).to_dict()

    # 2. 3Y Return (3 calendar years)
    res_3y = calculate_period_return_by_date(closes, dates, years=3.0)
    results["return_3y"] = DerivedMetric(
        metric_name="return_3y",
        value=res_3y["value"],
        unit="%",
        calculation_period="3y",
        methodology="((P_latest - P_3y_ago) / P_3y_ago) * 100",
        input_references=["raw_ohlcv_closes"],
        status=res_3y["status"],
        notes=res_3y["notes"],
    ).to_dict()

    # 3. 5Y Price CAGR
    cagr_5y = calculate_cagr(closes, dates, years=5.0)
    results["return_5y_cagr"] = DerivedMetric(
        metric_name="return_5y_cagr",
        value=round(cagr_5y, 4) if cagr_5y is not None else None,
        unit="%",
        calculation_period="5y (1260d)",
        methodology="((P_latest / P_5y_ago) ** (1/5.0)) - 1.0",
        input_references=["raw_ohlcv_closes"],
        status="success" if cagr_5y is not None else "insufficient_data",
        notes=f"5-Year CAGR across {len(closes)} available trading days",
    ).to_dict()

    # 4. Maximum Drawdown (5Y)
    max_dd = calculate_max_drawdown(closes)
    results["max_drawdown_5y"] = DerivedMetric(
        metric_name="max_drawdown_5y",
        value=round(max_dd, 4) if max_dd is not None else None,
        unit="%",
        calculation_period=f"{len(closes)}d",
        methodology="min((P_t - Peak_t) / Peak_t) * 100",
        input_references=["raw_ohlcv_closes"],
        status="success" if max_dd is not None else "insufficient_data",
    ).to_dict()

    # 5. Annualized Volatility
    vol = calculate_annualized_volatility(closes)
    results["annualized_volatility"] = DerivedMetric(
        metric_name="annualized_volatility",
        value=round(vol, 4) if vol is not None else None,
        unit="%",
        calculation_period=f"{len(closes)}d",
        methodology="StdDev(daily_returns) * sqrt(252) * 100",
        input_references=["raw_ohlcv_closes"],
        status="success" if vol is not None else "insufficient_data",
    ).to_dict()

    return results
