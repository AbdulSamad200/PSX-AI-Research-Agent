"""High-level interface for invoking psx-mcp-server tools.

Provides clean async functions to execute specific MCP tools (quote, company_info,
dividends, announcements, eod_history) and return parsed data for LangGraph nodes.
"""

import json
import asyncio
import logging
from typing import Dict, Any, List, Optional
from psx_mcp_server.server import mcp as direct_fastmcp_server
from src.tools.mcp_client import get_psx_mcp_tools


logger = logging.getLogger(__name__)


def _parse_res_payload(res: Any) -> Any:
    """Helper to cleanly parse JSON text, stringified dicts, list content, or raw objects."""
    if isinstance(res, str):
        try:
            return json.loads(res)
        except json.JSONDecodeError:
            return res

    if isinstance(res, list) and len(res) > 0:
        first = res[0]
        if isinstance(first, dict) and "text" in first:
            return _parse_res_payload(first["text"])
        if hasattr(first, "text"):
            return _parse_res_payload(getattr(first, "text"))
        return [_parse_res_payload(item) for item in res]

    if hasattr(res, "content"):
        return _parse_res_payload(getattr(res, "content"))

    if isinstance(res, dict) and "text" in res:
        return _parse_res_payload(res["text"])

    return res



async def call_mcp_tool_by_name(tool_name: str, arguments: Dict[str, Any]) -> Any:
    """Invokes a specific psx-mcp-server tool by name.
    
    First attempts execution via MultiServerMCPClient tools; falls back to
    direct FastMCP server invocation if needed.
    
    Args:
        tool_name: Name of the tool (e.g., 'get_quote', 'get_dividends')
        arguments: Parameters dictionary (e.g., {'symbol': 'MEBL'})
        
    Returns:
        Parsed JSON dict/list or raw text response.
    """
    # 1. Try invoking via LangChain MCP tool adapter
    try:
        tools = await get_psx_mcp_tools()
        matching_tools = [t for t in tools if t.name == tool_name]
        if matching_tools:
            target_tool = matching_tools[0]
            raw_res = await target_tool.ainvoke(arguments)
            return _parse_res_payload(raw_res)
    except Exception as e:
        logger.warning(f"LangChain MCP client tool call failed for '{tool_name}': {e}. Falling back to direct FastMCP.")

    # 2. Fallback: Direct FastMCP server tool execution
    res = await direct_fastmcp_server.call_tool(tool_name, arguments)
    if res and len(res) > 0 and hasattr(res[0], "text"):
        return _parse_res_payload(res[0].text)
    return res


async def get_mcp_quote(symbol: str) -> Dict[str, Any]:
    """Fetches real-time price quote and market statistics for a ticker."""
    res = await call_mcp_tool_by_name("get_quote", {"symbol": symbol.upper()})
    return res if isinstance(res, dict) else {"raw": res}


async def get_mcp_company_info(symbol: str) -> Dict[str, Any]:
    """Fetches company profile, market cap, shares outstanding, and free float."""
    res = await call_mcp_tool_by_name("get_company_info", {"symbol": symbol.upper()})
    return res if isinstance(res, dict) else {"raw": res}


async def get_mcp_dividends(symbol: str, limit: int = 5) -> Dict[str, Any]:
    """Fetches upcoming and historical dividend payout details."""
    res = await call_mcp_tool_by_name("get_dividends", {"symbol": symbol.upper(), "limit": limit})
    return res if isinstance(res, dict) else {"payouts": res}


async def get_mcp_announcements(symbol: str, limit: int = 5) -> Dict[str, Any]:
    """Fetches corporate disclosures and board meeting outcomes with PDF links."""
    res = await call_mcp_tool_by_name("get_announcements", {"symbol": symbol.upper(), "limit": limit})
    return res if isinstance(res, dict) else {"announcements": res}


async def get_mcp_eod_history(symbol: str, limit: int = 10) -> List[Dict[str, Any]]:
    """Fetches End-Of-Day historical price records from psx-mcp-server."""
    res = await call_mcp_tool_by_name("get_eod_history", {"symbol": symbol.upper(), "limit": limit})
    return res if isinstance(res, list) else [res]


async def fetch_all_mcp_data(symbol: str) -> Dict[str, Any]:
    """Fetches quote, company info, dividends, and announcements concurrently.
    
    Returns a unified dictionary ready to populate Raw Data and Evidence in AgentState.
    """
    symbol = symbol.upper()
    quote, company_info, dividends, announcements = await asyncio.gather(
        get_mcp_quote(symbol),
        get_mcp_company_info(symbol),
        get_mcp_dividends(symbol),
        get_mcp_announcements(symbol),
    )

    return {
        "symbol": symbol,
        "market_data": quote,
        "company_info": company_info,
        "dividend_data": dividends,
        "announcements": announcements,
    }

