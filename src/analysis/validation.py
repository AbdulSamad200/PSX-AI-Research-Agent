"""Conditional source metric validation module for PSX Stock Agent."""

import re
from typing import Dict, Any, Optional
from src.analysis.models import ValidationResult


def _parse_first_float(val: Any) -> Optional[float]:
    """Extracts the first floating point number from a multi-year pipe string or numeric value."""
    if val is None:
        return None
    if isinstance(val, (int, float)):
        return float(val)
    
    val_str = str(val).split("|")[0].replace(",", "").replace("%", "").strip()
    match = re.search(r"[-+]?\d*\.\d+|\d+", val_str)
    if match:
        try:
            return float(match.group(0))
        except ValueError:
            return None
    return None


def validate_net_profit_margin(
    annual_financials: Dict[str, Any], 
    ratios: Dict[str, Any]
) -> ValidationResult:
    """Validates source Net Profit Margin (%) against calculated (Profit After Tax / Revenue) * 100.
    
    Handles both Banking ('Mark-up Earned' / 'Total Income') and Industrial ('Sales') top-line fields.
    Returns status 'not_applicable' if required financial fields are missing.
    """
    source_margin_raw = ratios.get("Net Profit Margin (%)")
    source_margin = _parse_first_float(source_margin_raw)

    if source_margin is None:
        return ValidationResult(
            metric_name="net_profit_margin_validation",
            source_value=source_margin_raw,
            calculated_value=None,
            difference=None,
            status="not_applicable",
            notes="Source Net Profit Margin (%) is not provided in ratios",
        )

    # Extract Net Profit
    net_profit = _parse_first_float(annual_financials.get("Profit after Taxation"))
    if net_profit is None:
        return ValidationResult(
            metric_name="net_profit_margin_validation",
            source_value=source_margin,
            calculated_value=None,
            difference=None,
            status="not_applicable",
            notes="Annual 'Profit after Taxation' field is missing from statement data",
        )

    # Determine Top-line Revenue (Sales for Industrial, Mark-up Earned or Total Income for Banks)
    revenue = _parse_first_float(annual_financials.get("Mark-up Earned"))
    if revenue is None:
        revenue = _parse_first_float(annual_financials.get("Sales"))
    if revenue is None:
        revenue = _parse_first_float(annual_financials.get("Total Income"))

    if revenue is None or revenue <= 0:
        return ValidationResult(
            metric_name="net_profit_margin_validation",
            source_value=source_margin,
            calculated_value=None,
            difference=None,
            status="not_applicable",
            notes="No compatible top-line revenue metric (Sales/Mark-up/Total Income) found",
        )

    calculated_margin = round((net_profit / revenue) * 100.0, 2)
    diff = round(abs(calculated_margin - source_margin), 2)

    # Within 0.5% tolerance is considered a match due to rounding/tax adjustments
    is_match = diff <= 0.5
    status = "match" if is_match else "discrepancy"

    return ValidationResult(
        metric_name="net_profit_margin_validation",
        source_value=source_margin,
        calculated_value=calculated_margin,
        difference=diff,
        status=status,
        notes=f"Calculated ({net_profit} / {revenue}) * 100 = {calculated_margin}% vs source {source_margin}%",
    )


def validate_eps_growth(
    annual_financials: Dict[str, Any], 
    ratios: Dict[str, Any]
) -> ValidationResult:
    """Validates source EPS Growth (%) against annual EPS multi-year series.
    
    Returns status 'not_applicable' if multi-year EPS series is not available.
    """
    source_eps_growth_raw = ratios.get("EPS Growth (%)")
    source_eps_growth = _parse_first_float(source_eps_growth_raw)

    if source_eps_growth is None:
        return ValidationResult(
            metric_name="eps_growth_validation",
            source_value=source_eps_growth_raw,
            calculated_value=None,
            difference=None,
            status="not_applicable",
            notes="Source EPS Growth (%) is missing from ratios",
        )

    eps_str = annual_financials.get("EPS")
    if not eps_str or not isinstance(eps_str, str) or "|" not in eps_str:
        return ValidationResult(
            metric_name="eps_growth_validation",
            source_value=source_eps_growth,
            calculated_value=None,
            difference=None,
            status="not_applicable",
            notes="Multi-year EPS series (pipe-separated) is missing from annual financials",
        )

    parts = [ _parse_first_float(p) for p in eps_str.split("|") if _parse_first_float(p) is not None ]
    if len(parts) < 2:
        return ValidationResult(
            metric_name="eps_growth_validation",
            source_value=source_eps_growth,
            calculated_value=None,
            difference=None,
            status="not_applicable",
            notes="Requires at least 2 annual EPS data points for validation",
        )

    eps_latest = parts[0]
    eps_prev = parts[1]

    if eps_prev == 0:
        return ValidationResult(
            metric_name="eps_growth_validation",
            source_value=source_eps_growth,
            calculated_value=None,
            difference=None,
            status="not_applicable",
            notes="Previous EPS is zero; growth cannot be calculated",
        )

    calc_growth = round(((eps_latest - eps_prev) / abs(eps_prev)) * 100.0, 2)
    diff = round(abs(calc_growth - source_eps_growth), 2)
    status = "match" if diff <= 1.0 else "discrepancy"

    return ValidationResult(
        metric_name="eps_growth_validation",
        source_value=source_eps_growth,
        calculated_value=calc_growth,
        difference=diff,
        status=status,
        notes=f"Calculated (({eps_latest} - {eps_prev}) / |{eps_prev}|) * 100 = {calc_growth}% vs source {source_eps_growth}%",
    )


def compute_all_validations(
    annual_financials: Dict[str, Any], 
    ratios: Dict[str, Any]
) -> Dict[str, Dict[str, Any]]:
    """Master calculator for validation metrics."""
    return {
        "net_profit_margin_validation": validate_net_profit_margin(annual_financials, ratios).to_dict(),
        "eps_growth_validation": validate_eps_growth(annual_financials, ratios).to_dict(),
    }
