"""Tool Registry for PSX Stock Research Agent (Step 6).

Exposes psx-mcp-server tools and pypsx-toolkit tools as LangChain @tool objects
discoverable by OpenAI and LangGraph ToolNodes.
"""

import json
import asyncio
from typing import Dict, Any, List, Optional
from langchain_core.tools import tool

from src.tools.mcp_tools import (
    get_mcp_quote,
    get_mcp_company_info,
    get_mcp_dividends,
    get_mcp_announcements,
)
from src.tools.toolkit_tools import (
    get_stock_ohlcv,
    get_financial_statements,
    get_company_fundamentals,
    get_dividend_history,
)


# =============================================================================
# 1. psx-mcp-server Tools
# =============================================================================

@tool("mcp_get_quote")
def mcp_get_quote_tool(symbol: str) -> str:
    """Fetch real-time stock price quote, market statistics, LDCP, and daily range from psx-mcp-server.
    
    Args:
        symbol: PSX stock symbol (e.g. "MEBL", "FFC", "BAFL")
    """
    try:
        # Run async function in sync wrapper if needed
        res = asyncio.run(get_mcp_quote(symbol))
        return json.dumps(res, default=str)
    except Exception as e:
        return json.dumps({"error": f"Failed to fetch MCP quote for {symbol}: {str(e)}"})


@tool("mcp_get_company_info")
def mcp_get_company_info_tool(symbol: str) -> str:
    """Fetch company profile, market cap, shares outstanding, and free float percentage from psx-mcp-server.
    
    Args:
        symbol: PSX stock symbol (e.g. "MEBL", "FFC")
    """
    try:
        res = asyncio.run(get_mcp_company_info(symbol))
        return json.dumps(res, default=str)
    except Exception as e:
        return json.dumps({"error": f"Failed to fetch MCP company info for {symbol}: {str(e)}"})


@tool("mcp_get_dividends")
def mcp_get_dividends_tool(symbol: str, limit: int = 5) -> str:
    """Fetch recent and upcoming dividend declarations, ex-dividend dates, and payout amounts from psx-mcp-server.
    
    Args:
        symbol: PSX stock symbol (e.g. "MEBL")
        limit: Maximum number of dividend records to return (default: 5)
    """
    try:
        res = asyncio.run(get_mcp_dividends(symbol, limit=limit))
        return json.dumps(res, default=str)
    except Exception as e:
        return json.dumps({"error": f"Failed to fetch MCP dividends for {symbol}: {str(e)}"})


@tool("mcp_get_announcements")
def mcp_get_announcements_tool(symbol: str, limit: int = 5) -> str:
    """Fetch official corporate disclosures, board meeting outcomes, and PDF links from psx-mcp-server.
    
    Args:
        symbol: PSX stock symbol (e.g. "MEBL")
        limit: Maximum number of announcements to return (default: 5)
    """
    try:
        res = asyncio.run(get_mcp_announcements(symbol, limit=limit))
        return json.dumps(res, default=str)
    except Exception as e:
        return json.dumps({"error": f"Failed to fetch MCP announcements for {symbol}: {str(e)}"})


# =============================================================================
# 2. pypsx-toolkit Tools
# =============================================================================

@tool("toolkit_get_stock_ohlcv")
def toolkit_get_stock_ohlcv_tool(symbol: str, period: str = "5y") -> str:
    """Fetch normalized historical OHLCV price records (open, high, low, close, volume) for a ticker over a period.
    
    Args:
        symbol: PSX stock symbol (e.g. "MEBL")
        period: Historical period ('1d', '1wk', '1mo', '1y', '5y')
    """
    try:
        res = get_stock_ohlcv(symbol, period=period)
        return json.dumps(res, default=str)
    except Exception as e:
        return json.dumps({"error": f"Failed to fetch toolkit OHLCV for {symbol}: {str(e)}"})


@tool("toolkit_get_financial_statements")
def toolkit_get_financial_statements_tool(symbol: str) -> str:
    """Fetch multi-year annual and quarterly financial statements (Income statement, Balance sheet, Cash flow) from pypsx-toolkit.
    
    Args:
        symbol: PSX stock symbol (e.g. "MEBL")
    """
    try:
        res = get_financial_statements(symbol)
        return json.dumps(res, default=str)
    except Exception as e:
        return json.dumps({"error": f"Failed to fetch toolkit financial statements for {symbol}: {str(e)}"})


@tool("toolkit_get_company_fundamentals")
def toolkit_get_company_fundamentals_tool(symbol: str) -> str:
    """Fetch financial ratios (Net Margin, EPS Growth, PEG) and equity profile metrics from pypsx-toolkit.
    
    Args:
        symbol: PSX stock symbol (e.g. "MEBL")
    """
    try:
        res = get_company_fundamentals(symbol)
        return json.dumps(res, default=str)
    except Exception as e:
        return json.dumps({"error": f"Failed to fetch toolkit fundamentals for {symbol}: {str(e)}"})


@tool("toolkit_get_dividend_history")
def toolkit_get_dividend_history_tool(symbol: str) -> str:
    """Fetch complete historical cash dividend distribution log from pypsx-toolkit.
    
    Args:
        symbol: PSX stock symbol (e.g. "MEBL")
    """
    try:
        res = get_dividend_history(symbol)
        return json.dumps(res, default=str)
    except Exception as e:
        return json.dumps({"error": f"Failed to fetch toolkit dividend history for {symbol}: {str(e)}"})


# Master registry list of all Step 6 tools
ALL_TOOLS = [
    mcp_get_quote_tool,
    mcp_get_company_info_tool,
    mcp_get_dividends_tool,
    mcp_get_announcements_tool,
    toolkit_get_stock_ohlcv_tool,
    toolkit_get_financial_statements_tool,
    toolkit_get_company_fundamentals_tool,
    toolkit_get_dividend_history_tool,
]


def get_tool_by_name(name: str) -> Optional[Any]:
    """Helper to retrieve a tool object by name."""
    for t in ALL_TOOLS:
        if t.name == name:
            return t
    return None
