"""Dividend-derived metrics module for PSX Stock Agent."""

import re
from datetime import datetime
from typing import List, Dict, Any, Optional
from src.analysis.models import DerivedMetric


def _parse_cash_amount(cash_str: Any) -> Optional[float]:
    """Parses numeric cash dividend amount from string (e.g. '8.000 PKR' -> 8.0)."""
    if cash_str is None:
        return None
    if isinstance(cash_str, (int, float)):
        return float(cash_str)
    
    clean_str = str(cash_str).replace("PKR", "").replace(",", "").strip()
    match = re.search(r"[-+]?\d*\.\d+|\d+", clean_str)
    if match:
        try:
            return float(match.group(0))
        except ValueError:
            return None
    return None


def _parse_date(date_str: Any) -> Optional[datetime]:
    """Parses date string into datetime object."""
    if not date_str or not isinstance(date_str, str):
        return None
    
    # Try various PSX date formats: "Aug 18, 2026", "18/08/2026", "2026-08-18"
    formats = ["%b %d, %Y", "%d/%m/%Y", "%Y-%m-%d", "%B %d, %Y"]
    for fmt in formats:
        try:
            return datetime.strptime(date_str.strip(), fmt)
        except ValueError:
            continue
    return None


def calculate_ttm_dividend_yield(
    dividend_history: List[Dict[str, Any]], 
    current_price: Optional[float]
) -> DerivedMetric:
    """Calculates Trailing Twelve Months (TTM) Dividend Yield.
    
    Formula: (Sum of cash dividends in last 365 days / Current_Price) * 100
    """
    if current_price is None or current_price <= 0:
        return DerivedMetric(
            metric_name="ttm_dividend_yield",
            value=None,
            unit="%",
            calculation_period="TTM (365d)",
            methodology="(Sum_TTM_Dividends / Current_Price) * 100",
            input_references=["dividend_history", "mcp_quote_price"],
            status="insufficient_data",
            notes="Current stock price is missing or zero",
        )

    if not dividend_history:
        return DerivedMetric(
            metric_name="ttm_dividend_yield",
            value=0.0,
            unit="%",
            calculation_period="TTM (365d)",
            methodology="(Sum_TTM_Dividends / Current_Price) * 100",
            input_references=["dividend_history", "mcp_quote_price"],
            status="success",
            notes="No dividend payouts on record in last 12 months",
        )

    # Reference latest dividend date or current date
    latest_div_date = None
    parsed_records = []

    for r in dividend_history:
        amt = _parse_cash_amount(r.get("cash_amount"))
        dt = _parse_date(r.get("ex_dividend_date")) or _parse_date(r.get("pay_date"))
        if amt is not None and amt > 0 and dt is not None:
            parsed_records.append((dt, amt))
            if latest_div_date is None or dt > latest_div_date:
                latest_div_date = dt

    if not parsed_records or latest_div_date is None:
        return DerivedMetric(
            metric_name="ttm_dividend_yield",
            value=0.0,
            unit="%",
            calculation_period="TTM (365d)",
            methodology="(Sum_TTM_Dividends / Current_Price) * 100",
            input_references=["dividend_history"],
            status="success",
            notes="No valid cash dividend amounts found",
        )

    # Sum cash payouts within 365 days of latest dividend date
    ttm_total = 0.0
    for dt, amt in parsed_records:
        days_diff = (latest_div_date - dt).days
        if 0 <= days_diff <= 365:
            ttm_total += amt

    div_yield = (ttm_total / current_price) * 100.0

    return DerivedMetric(
        metric_name="ttm_dividend_yield",
        value=round(div_yield, 4),
        unit="%",
        calculation_period="TTM (365d)",
        methodology="(Sum_TTM_Dividends / Current_Price) * 100",
        input_references=["dividend_history", "current_price"],
        status="success",
        notes=f"TTM cash dividend payout: {round(ttm_total, 2)} PKR over current price {current_price} PKR",
    )


def calculate_annual_historical_dividend_yields(
    dividend_history: List[Dict[str, Any]], 
    ohlcv_bars: List[Dict[str, Any]]
) -> DerivedMetric:
    """Calculates annual historical dividend yield series and YoY payout growth.
    
    Formula:
      Annual Dividend Yield (Year T) = (Sum of Cash Dividends in Year T / Last Trading-Day Close in Year T) * 100
      YoY Dividend Growth (Year T) = ((Cash_Dividends_T - Cash_Dividends_T-1) / Cash_Dividends_T-1) * 100
      
    Current/latest calendar year is labeled as 'YTD/available-period'.
    Explicit limitation note added for unadjusted stock splits/bonus issues.
    """
    if not dividend_history or not ohlcv_bars:
        return DerivedMetric(
            metric_name="annual_historical_dividend_yields",
            value=[],
            unit="annual_records",
            calculation_period="Multi-Year",
            methodology="(Annual_Cash_DPS / Year_End_Last_Close) * 100",
            input_references=["dividend_history", "raw_ohlcv"],
            status="insufficient_data",
            notes="Missing dividend history or OHLCV price series",
        )

    # 1. Aggregate Cash Dividends per Calendar Year
    yearly_payouts: Dict[int, float] = {}
    for r in dividend_history:
        amt = _parse_cash_amount(r.get("cash_amount"))
        dt = _parse_date(r.get("ex_dividend_date")) or _parse_date(r.get("pay_date"))
        if amt is not None and amt > 0 and dt is not None:
            yr = dt.year
            yearly_payouts[yr] = yearly_payouts.get(yr, 0.0) + amt

    if not yearly_payouts:
        return DerivedMetric(
            metric_name="annual_historical_dividend_yields",
            value=[],
            unit="annual_records",
            calculation_period="Multi-Year",
            methodology="(Annual_Cash_DPS / Year_End_Last_Close) * 100",
            input_references=["dividend_history"],
            status="success",
            notes="No valid annual cash dividend payouts found",
        )

    # 2. Extract Year-End Last Close Prices from OHLCV
    yearly_last_close: Dict[int, float] = {}
    yearly_last_date: Dict[int, str] = {}
    
    for bar in ohlcv_bars:
        dt_str = bar.get("date", "")
        close_px = bar.get("close")
        if dt_str and close_px is not None and close_px > 0:
            try:
                yr = int(dt_str.split("-")[0])
                yearly_last_close[yr] = float(close_px)
                yearly_last_date[yr] = dt_str
            except (ValueError, IndexError):
                continue

    sorted_years = sorted(yearly_payouts.keys())
    max_year = sorted_years[-1]

    annual_records = []
    for idx, yr in enumerate(sorted_years):
        cash_dps = round(yearly_payouts[yr], 2)
        ref_price = yearly_last_close.get(yr)
        ref_date = yearly_last_date.get(yr)
        
        is_ytd = (yr == max_year)
        period_label = f"{yr} (YTD)" if is_ytd else str(yr)

        # Dividend Yield for Year
        div_yield = round((cash_dps / ref_price) * 100.0, 4) if ref_price and ref_price > 0 else None

        # YoY Dividend Growth (Only for complete prior calendar years; set None for YTD)
        yoy_growth = None
        if not is_ytd:
            prev_yr = yr - 1
            if prev_yr in yearly_payouts and yearly_payouts[prev_yr] > 0:
                prev_dps = yearly_payouts[prev_yr]
                yoy_growth = round(((cash_dps - prev_dps) / prev_dps) * 100.0, 4)

        record = {
            "year": yr,
            "period_label": period_label,
            "is_ytd": is_ytd,
            "annual_cash_dps_pkr": cash_dps,
            "annual_reference_price_pkr": ref_price,
            "annual_reference_price_date": ref_date,
            "annual_dividend_yield_pct": div_yield,
            "yoy_dividend_growth_pct": yoy_growth,
        }
        annual_records.append(record)

    limitations_note = (
        "Historical per-share cash amounts and yields are unadjusted for historical stock splits "
        "or bonus share issues, as corporate action adjustment histories are not provided."
    )

    return DerivedMetric(
        metric_name="annual_historical_dividend_yields",
        value=annual_records,
        unit="annual_records",
        calculation_period=f"{sorted_years[0]}-{sorted_years[-1]}",
        methodology="(Annual_Cash_DPS / Year_End_Last_Close) * 100",
        input_references=["dividend_history", "raw_ohlcv"],
        status="success",
        notes=limitations_note,
    )


def calculate_dividend_cagr_3y(
    dividend_history: List[Dict[str, Any]]
) -> DerivedMetric:
    """Calculates 3-Year Annual Dividend Payout CAGR (optional & defensive).
    
    Groups dividends by calendar year. Only calculates if at least 4 consecutive years are available.
    """
    if not dividend_history or len(dividend_history) < 4:
        return DerivedMetric(
            metric_name="dividend_cagr_3y",
            value=None,
            unit="%",
            calculation_period="3y",
            methodology="((Div_Year_T / Div_Year_T-3) ** (1/3.0)) - 1.0",
            input_references=["dividend_history"],
            status="insufficient_data",
            notes="Requires at least 4 distinct annual payout periods for defensive calculation",
        )

    yearly_payouts: Dict[int, float] = {}

    for r in dividend_history:
        amt = _parse_cash_amount(r.get("cash_amount"))
        dt = _parse_date(r.get("ex_dividend_date")) or _parse_date(r.get("pay_date"))
        if amt is not None and amt > 0 and dt is not None:
            yr = dt.year
            yearly_payouts[yr] = yearly_payouts.get(yr, 0.0) + amt

    years = sorted(yearly_payouts.keys())
    if len(years) < 4:
        return DerivedMetric(
            metric_name="dividend_cagr_3y",
            value=None,
            unit="%",
            calculation_period="3y",
            methodology="((Div_Year_T / Div_Year_T-3) ** (1/3.0)) - 1.0",
            input_references=["dividend_history"],
            status="insufficient_data",
            notes=f"Only {len(years)} years of dividend data on record",
        )

    latest_yr = years[-1]
    target_yr = latest_yr - 3

    if target_yr not in yearly_payouts:
        return DerivedMetric(
            metric_name="dividend_cagr_3y",
            value=None,
            unit="%",
            calculation_period="3y",
            methodology="((Div_Year_T / Div_Year_T-3) ** (1/3.0)) - 1.0",
            input_references=["dividend_history"],
            status="insufficient_data",
            notes=f"Missing payout record for 3 years prior ({target_yr})",
        )

    div_start = yearly_payouts[target_yr]
    div_latest = yearly_payouts[latest_yr]

    if div_start <= 0 or div_latest <= 0:
        return DerivedMetric(
            metric_name="dividend_cagr_3y",
            value=None,
            unit="%",
            calculation_period="3y",
            methodology="((Div_Year_T / Div_Year_T-3) ** (1/3.0)) - 1.0",
            input_references=["dividend_history"],
            status="insufficient_data",
            notes="Non-positive annual dividend payout encountered",
        )

    cagr = ((div_latest / div_start) ** (1.0 / 3.0) - 1.0) * 100.0

    return DerivedMetric(
        metric_name="dividend_cagr_3y",
        value=round(cagr, 4),
        unit="%",
        calculation_period="3y",
        methodology="((Div_Year_T / Div_Year_T-3) ** (1/3.0)) - 1.0",
        input_references=["dividend_history"],
        status="success",
        notes=f"Payout increased from {round(div_start, 2)} PKR in {target_yr} to {round(div_latest, 2)} PKR in {latest_yr}",
    )


def compute_all_dividend_metrics(
    dividend_history: List[Dict[str, Any]], 
    current_price: Optional[float],
    ohlcv_bars: Optional[List[Dict[str, Any]]] = None
) -> Dict[str, Dict[str, Any]]:
    """Master calculator for dividend-derived metrics."""
    res = {
        "ttm_dividend_yield": calculate_ttm_dividend_yield(dividend_history, current_price).to_dict(),
        "dividend_cagr_3y": calculate_dividend_cagr_3y(dividend_history).to_dict(),
    }
    if ohlcv_bars:
        res["annual_historical_dividend_yields"] = calculate_annual_historical_dividend_yields(dividend_history, ohlcv_bars).to_dict()
    return res
