"""Tools package for PSX Stock Agent."""

# MCP Server Tools
from src.tools.mcp_client import get_psx_mcp_client, get_psx_mcp_tools
from src.tools.mcp_tools import (
    call_mcp_tool_by_name,
    get_mcp_quote,
    get_mcp_company_info,
    get_mcp_dividends,
    get_mcp_announcements,
    get_mcp_eod_history,
    fetch_all_mcp_data,
)

# PyPSX-Toolkit Python Tools
from src.tools.toolkit_tools import (
    get_stock_ohlcv,
    get_company_fundamentals,
    get_financial_statements,
    get_dividend_history,
    get_technical_data,
    fetch_all_toolkit_data,
)

__all__ = [
    # MCP
    "get_psx_mcp_client",
    "get_psx_mcp_tools",
    "call_mcp_tool_by_name",
    "get_mcp_quote",
    "get_mcp_company_info",
    "get_mcp_dividends",
    "get_mcp_announcements",
    "get_mcp_eod_history",
    "fetch_all_mcp_data",
    # Toolkit
    "get_stock_ohlcv",
    "get_company_fundamentals",
    "get_financial_statements",
    "get_dividend_history",
    "get_technical_data",
    "fetch_all_toolkit_data",
]
