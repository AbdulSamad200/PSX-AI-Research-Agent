"""MCP Client module for connecting to psx-mcp-server via stdio transport.

Uses langchain-mcp-adapters to discover and wrap psx-mcp-server tools as
standard LangChain BaseTool objects for seamless LangGraph integration.
"""

import sys
import logging
from typing import List, Dict, Any, Optional
from langchain_core.tools import BaseTool
from langchain_mcp_adapters.client import MultiServerMCPClient

logger = logging.getLogger(__name__)

# Global client and tools cache
_mcp_client: Optional[MultiServerMCPClient] = None
_cached_tools: Optional[List[BaseTool]] = None


def get_mcp_server_config() -> Dict[str, Any]:
    """Returns the stdio transport configuration for psx-mcp-server."""
    return {
        "psx_mcp": {
            "transport": "stdio",
            "command": sys.executable,
            "args": ["-m", "psx_mcp_server.server"],
        }
    }


def get_psx_mcp_client() -> MultiServerMCPClient:
    """Instantiates and returns the singleton MultiServerMCPClient."""
    global _mcp_client
    if _mcp_client is None:
        config = get_mcp_server_config()
        logger.info("Initializing MultiServerMCPClient with psx-mcp-server stdio transport.")
        _mcp_client = MultiServerMCPClient(
            connections=config,
            tool_name_prefix=False,
            handle_tool_errors=True,
        )
    return _mcp_client


async def get_psx_mcp_tools() -> List[BaseTool]:
    """Asynchronously discovers and returns all tools from psx-mcp-server.
    
    Caches tool definitions after first discovery for maximum performance.
    
    Returns:
        List[BaseTool]: LangChain compatible BaseTool objects.
    """
    global _cached_tools
    if _cached_tools is None:
        client = get_psx_mcp_client()
        _cached_tools = await client.get_tools()
        logger.info(f"Discovered {len(_cached_tools)} tools from psx-mcp-server.")
    return _cached_tools
