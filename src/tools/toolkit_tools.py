"""pypsx-toolkit Integration Module for PSX Stock Agent.

Provides clean Python data-wrapper functions for fetching 10-year OHLCV prices,
company fundamentals, financial statements (annual & quarterly), dividend history,
and technical raw series from pypsx-toolkit.

All functions normalize Pandas DataFrames into 100% JSON-serializable Python dicts
and lists for safe storage in LangGraph AgentState.
"""

import logging
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional, Union
import pandas as pd
import pypsx_toolkit as pt

logger = logging.getLogger(__name__)


def _get_timestamp() -> str:
    """Returns ISO 8601 UTC timestamp."""
    return datetime.now(timezone.utc).isoformat()


def _sanitize_val(val: Any) -> Any:
    """Converts numpy/pandas types (NaN, NaT, int64, float64) to native Python types."""
    if pd.isna(val):
        return None
    if hasattr(val, "item"):
        return val.item()
    if isinstance(val, (pd.Timestamp, datetime)):
        return val.isoformat()
    return val


def get_stock_ohlcv(symbol: str, period: str = "5y") -> Dict[str, Any]:
    """Fetches historical OHLCV price data for a given ticker and period.
    
    Args:
        symbol: PSX stock symbol (e.g. "MEBL", "FFC")
        period: Historical period ("1d", "1wk", "1mo", "1y", "5y", "10y")
        
    Returns:
        Dict containing symbol, metadata, and normalized list of OHLCV records:
        [{"date": "YYYY-MM-DD", "open": float, "high": float, "low": float, "close": float, "volume": int}]
    """
    symbol = symbol.upper()
    try:
        df = pt.download(symbol, period=period)
        if df is None or df.empty:
            logger.warning(f"No OHLCV data returned by pypsx-toolkit for {symbol} (period={period}).")
            return {
                "symbol": symbol,
                "period": period,
                "source": "pypsx-toolkit",
                "retrieved_at": _get_timestamp(),
                "records_count": 0,
                "ohlcv": [],
            }

        # Handle MultiIndex columns if present (e.g., ('MEBL', 'OPEN'))
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(-1)

        # Standardize column names to lowercase
        df.columns = [str(c).strip().lower() for c in df.columns]

        records = []
        for idx, row in df.iterrows():
            date_str = idx.strftime("%Y-%m-%d") if hasattr(idx, "strftime") else str(idx).split(" ")[0]
            records.append({
                "date": date_str,
                "open": _sanitize_val(row.get("open")),
                "high": _sanitize_val(row.get("high")),
                "low": _sanitize_val(row.get("low")),
                "close": _sanitize_val(row.get("close")),
                "volume": int(_sanitize_val(row.get("volume")) or 0),
            })

        return {
            "symbol": symbol,
            "period": period,
            "source": "pypsx-toolkit",
            "retrieved_at": _get_timestamp(),
            "records_count": len(records),
            "ohlcv": records,
        }
    except Exception as e:
        logger.error(f"Error fetching OHLCV for {symbol}: {e}")
        return {
            "symbol": symbol,
            "period": period,
            "source": "pypsx-toolkit",
            "retrieved_at": _get_timestamp(),
            "error": str(e),
            "records_count": 0,
            "ohlcv": [],
        }


def get_company_fundamentals(symbol: str) -> Dict[str, Any]:
    """Fetches company profile, equity structure, governance, and financial ratios.
    
    Args:
        symbol: PSX stock symbol (e.g. "MEBL", "FFC")
        
    Returns:
        Dict containing normalized profile, equity metrics, governance, and financial ratios.
    """
    symbol = symbol.upper()
    try:
        fundamentals = pt.get_company_fundamentals(symbol)
        if fundamentals is None or fundamentals.empty:
            return {
                "symbol": symbol,
                "source": "pypsx-toolkit",
                "retrieved_at": _get_timestamp(),
                "profile": {},
                "governance": {},
                "equity_profile": {},
                "ratios": {},
            }

        result = {
            "symbol": symbol,
            "source": "pypsx-toolkit",
            "retrieved_at": _get_timestamp(),
            "profile": {},
            "governance": {},
            "equity_profile": {},
            "ratios": {},
        }

        for idx, row in fundamentals.iterrows():
            if isinstance(idx, tuple):
                if len(idx) >= 3:
                    category = str(idx[1]).strip()
                    key_clean = " ".join([str(x).strip() for x in idx[2:] if str(x).strip()])
                elif len(idx) == 2:
                    category = str(idx[0]).strip()
                    key_clean = str(idx[1]).strip()
                else:
                    category = "General"
                    key_clean = str(idx[0]).strip()
            else:
                category = "General"
                key_clean = str(idx).strip()

            cat_lower = category.lower()

            if hasattr(row, "values"):
                vals = [str(_sanitize_val(v)) for v in row.values if pd.notna(v)]
                val = " | ".join(vals) if len(vals) > 1 else (vals[0] if vals else None)
            else:
                val = _sanitize_val(row)

            if "profile" in cat_lower and "equity" not in cat_lower:
                result["profile"][key_clean] = val
            elif "governance" in cat_lower:
                result["governance"][key_clean] = val
            elif "equity" in cat_lower:
                result["equity_profile"][key_clean] = val
            elif "ratio" in cat_lower:
                result["ratios"][key_clean] = val

        return result
    except Exception as e:
        logger.error(f"Error fetching fundamentals for {symbol}: {e}")
        return {
            "symbol": symbol,
            "source": "pypsx-toolkit",
            "retrieved_at": _get_timestamp(),
            "error": str(e),
            "profile": {},
            "governance": {},
            "equity_profile": {},
            "ratios": {},
        }


def get_financial_statements(symbol: str) -> Dict[str, Any]:
    """Fetches annual and quarterly financial statements (Income Statement, Revenue, EPS).
    
    Args:
        symbol: PSX stock symbol (e.g. "MEBL", "FFC")
        
    Returns:
        Dict containing normalized annual and quarterly statement tables.
    """
    symbol = symbol.upper()
    try:
        fundamentals = pt.get_company_fundamentals(symbol)
        if fundamentals is None or fundamentals.empty:
            return {
                "symbol": symbol,
                "source": "pypsx-toolkit",
                "retrieved_at": _get_timestamp(),
                "annual_financials": {},
                "quarterly_financials": {},
            }

        annual_data = {}
        quarterly_data = {}

        for idx, row in fundamentals.iterrows():
            if isinstance(idx, tuple):
                if len(idx) >= 3:
                    category = str(idx[1]).strip()
                    metric_clean = " ".join([str(x).strip() for x in idx[2:] if str(x).strip()])
                elif len(idx) == 2:
                    category = str(idx[0]).strip()
                    metric_clean = str(idx[1]).strip()
                else:
                    category = "General"
                    metric_clean = str(idx[0]).strip()
            else:
                category = "General"
                metric_clean = str(idx).strip()

            cat_lower = category.lower()

            if hasattr(row, "values"):
                vals = [str(_sanitize_val(v)) for v in row.values if pd.notna(v)]
                val = " | ".join(vals) if len(vals) > 1 else (vals[0] if vals else None)
            else:
                val = _sanitize_val(row)

            if "annual" in cat_lower:
                annual_data[metric_clean] = val
            elif "quarterly" in cat_lower:
                quarterly_data[metric_clean] = val

        return {
            "symbol": symbol,
            "source": "pypsx-toolkit",
            "retrieved_at": _get_timestamp(),
            "annual_financials": annual_data,
            "quarterly_financials": quarterly_data,
        }


    except Exception as e:
        logger.error(f"Error fetching financial statements for {symbol}: {e}")
        return {
            "symbol": symbol,
            "source": "pypsx-toolkit",
            "retrieved_at": _get_timestamp(),
            "error": str(e),
            "annual_financials": {},
            "quarterly_financials": {},
        }


def get_dividend_history(symbol: str) -> Dict[str, Any]:
    """Fetches chronological ex-dividend dates, cash payout amounts, and pay dates.
    
    Args:
        symbol: PSX stock symbol (e.g. "MEBL", "FFC")
        
    Returns:
        Dict containing normalized list of dividend payout records:
        [{"ex_dividend_date": "Aug 18, 2026", "cash_amount": "8.000 PKR", "record_date": "...", "pay_date": "..."}]
    """
    symbol = symbol.upper()
    try:
        div_df = pt.get_dividend_history(symbol)
        if div_df is None or div_df.empty:
            return {
                "symbol": symbol,
                "source": "pypsx-toolkit",
                "retrieved_at": _get_timestamp(),
                "count": 0,
                "history": [],
            }

        records = []
        for idx, row in div_df.iterrows():
            records.append({
                "ex_dividend_date": _sanitize_val(row.get("EX-DIVIDEND DATE")),
                "cash_amount": _sanitize_val(row.get("CASH AMOUNT")),
                "record_date": _sanitize_val(row.get("RECORD DATE")),
                "pay_date": _sanitize_val(row.get("PAY DATE")),
            })

        return {
            "symbol": symbol,
            "source": "pypsx-toolkit",
            "retrieved_at": _get_timestamp(),
            "count": len(records),
            "history": records,
        }
    except Exception as e:
        logger.error(f"Error fetching dividend history for {symbol}: {e}")
        return {
            "symbol": symbol,
            "source": "pypsx-toolkit",
            "retrieved_at": _get_timestamp(),
            "error": str(e),
            "count": 0,
            "history": [],
        }


def get_technical_data(
    symbol: str, 
    period: str = "5y", 
    ohlcv_payload: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """Prepares raw price and volume time-series arrays for deterministic technical calculations.
    
    Can accept a pre-fetched ohlcv_payload to prevent duplicate downloads.
    
    Args:
        symbol: PSX stock symbol (e.g. "MEBL", "FFC")
        period: Historical period for technical analysis (default "5y")
        ohlcv_payload: Optional pre-fetched payload from get_stock_ohlcv()
        
    Returns:
        Dict containing date, close, high, low, and volume numeric lists.
    """
    symbol = symbol.upper()
    if ohlcv_payload is None:
        ohlcv_payload = get_stock_ohlcv(symbol, period=period)

    records = ohlcv_payload.get("ohlcv", [])

    dates = [r["date"] for r in records if r.get("date")]
    closes = [r["close"] for r in records if r.get("close") is not None]
    highs = [r["high"] for r in records if r.get("high") is not None]
    lows = [r["low"] for r in records if r.get("low") is not None]
    volumes = [r["volume"] for r in records if r.get("volume") is not None]

    return {
        "symbol": symbol,
        "period": period,
        "source": "pypsx-toolkit",
        "retrieved_at": _get_timestamp(),
        "series_length": len(closes),
        "dates": dates,
        "close_prices": closes,
        "high_prices": highs,
        "low_prices": lows,
        "volumes": volumes,
    }


def fetch_all_toolkit_data(symbol: str, period: str = "5y") -> Dict[str, Any]:
    """Composite fetcher retrieving OHLCV, fundamentals, financials, and dividends in one call.
    
    Reuses the single downloaded OHLCV payload for technical_series to eliminate duplicate downloads.
    
    Returns a unified dictionary ready to update AgentState.
    """
    symbol = symbol.upper()
    ohlcv_data = get_stock_ohlcv(symbol, period=period)
    technical_series = get_technical_data(symbol, period=period, ohlcv_payload=ohlcv_data)

    return {
        "symbol": symbol,
        "ohlcv_data": ohlcv_data,
        "fundamentals": get_company_fundamentals(symbol),
        "financial_statements": get_financial_statements(symbol),
        "dividend_history": get_dividend_history(symbol),
        "technical_series": technical_series,
    }

