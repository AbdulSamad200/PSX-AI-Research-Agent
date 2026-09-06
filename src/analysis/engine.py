"""Master orchestration engine for PSXenius financial & technical analysis."""

import logging
from typing import Dict, Any, Optional
from src.analysis.models import SourceMetric, AnalysisResult, get_utc_timestamp
from src.analysis.technical import compute_all_technical_metrics
from src.analysis.performance import compute_all_performance_metrics
from src.analysis.dividends import compute_all_dividend_metrics
from src.analysis.validation import compute_all_validations

logger = logging.getLogger(__name__)


def extract_source_metrics(
    toolkit_data: Dict[str, Any], 
    mcp_quote: Optional[Dict[str, Any]] = None
) -> Dict[str, Dict[str, Any]]:
    """Extracts and preserves source-provided metrics from toolkit and MCP server intact."""
    sources = {}
    fundamentals = toolkit_data.get("fundamentals", {})
    profile = fundamentals.get("profile", {})
    equity = fundamentals.get("equity_profile", {})
    ratios = fundamentals.get("ratios", {})
    annuals = toolkit_data.get("financial_statements", {}).get("annual_financials", {})
    quarterlies = toolkit_data.get("financial_statements", {}).get("quarterly_financials", {})

    # Profile & Governance
    if profile.get("Business Description"):
        sources["business_description"] = SourceMetric(
            metric_name="business_description",
            value=profile["Business Description"],
            source="pypsx-toolkit",
        ).to_dict()

    # Ratios
    for k, v in ratios.items():
        key_clean = k.lower().replace(" ", "_").replace("(", "").replace(")", "").replace("%", "pct")
        sources[f"source_{key_clean}"] = SourceMetric(
            metric_name=k,
            value=v,
            source="pypsx-toolkit",
            financial_period="Multi-Year",
        ).to_dict()

    # Equity Structure
    for k, v in equity.items():
        key_clean = k.lower().replace(" ", "_").replace("(", "").replace(")", "").replace("'", "")
        sources[f"source_{key_clean}"] = SourceMetric(
            metric_name=k,
            value=v,
            source="pypsx-toolkit",
        ).to_dict()


    # Statement Tables
    sources["annual_income_statement"] = SourceMetric(
        metric_name="annual_income_statement",
        value=annuals,
        source="pypsx-toolkit",
        financial_period="Annual Multi-Year",
    ).to_dict()

    sources["quarterly_income_statement"] = SourceMetric(
        metric_name="quarterly_income_statement",
        value=quarterlies,
        source="pypsx-toolkit",
        financial_period="Quarterly 4Q",
    ).to_dict()

    # MCP Quote if available
    if mcp_quote and isinstance(mcp_quote, dict):
        sources["mcp_current_price"] = SourceMetric(
            metric_name="current_price",
            value=mcp_quote.get("current_price"),
            unit="PKR",
            source="psx-mcp-server",
        ).to_dict()

    return sources


def analyze_stock(
    toolkit_data: Dict[str, Any], 
    mcp_quote: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """Master pipeline executing source metric extraction, derived metric calculations,
    and validation checks for a given stock data payload.
    
    Reuses pre-fetched OHLCV time-series to prevent duplicate network calls.
    """
    symbol = toolkit_data.get("symbol", "UNKNOWN").upper()
    tech_series = toolkit_data.get("technical_series", {})
    
    closes = tech_series.get("close_prices", [])
    highs = tech_series.get("high_prices", [])
    lows = tech_series.get("low_prices", [])
    dates = tech_series.get("dates", [])
    
    div_history = toolkit_data.get("dividend_history", {}).get("history", [])

    # Current Price: prefer latest OHLCV close if MCP quote current_price is None
    latest_close = closes[-1] if closes else None
    current_price = (mcp_quote.get("current_price") if mcp_quote and mcp_quote.get("current_price") is not None else latest_close)

    # 1. Source Metrics
    source_metrics = extract_source_metrics(toolkit_data, mcp_quote=mcp_quote)

    # 2. Derived Metrics
    derived_metrics = {}
    
    # Technical Indicators
    tech_metrics = compute_all_technical_metrics(closes, highs, lows, dates)
    derived_metrics.update(tech_metrics)

    # Performance & Risk Metrics
    perf_metrics = compute_all_performance_metrics(closes, dates)
    derived_metrics.update(perf_metrics)

    # Dividend Metrics
    ohlcv_bars = toolkit_data.get("ohlcv_data", {}).get("ohlcv", [])
    div_metrics = compute_all_dividend_metrics(div_history, current_price, ohlcv_bars=ohlcv_bars)
    derived_metrics.update(div_metrics)


    # 3. Validation Metrics
    annuals = toolkit_data.get("financial_statements", {}).get("annual_financials", {})
    ratios = toolkit_data.get("fundamentals", {}).get("ratios", {})
    validation_results = compute_all_validations(annuals, ratios)

    analysis_res = AnalysisResult(
        symbol=symbol,
        retrieved_at=get_utc_timestamp(),
        source_metrics=source_metrics,
        derived_metrics=derived_metrics,
        validation_results=validation_results,
    )

    return analysis_res.to_dict()
