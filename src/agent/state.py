"""State definitions for the PSX Investment Research LangGraph Agent."""

from typing import TypedDict, List, Dict, Any, Optional


class EvidenceItem(TypedDict):
    """Represents a single traceable financial evidence record.
    
    Designed to store verified data points from psx-mcp-server, pypsx-toolkit,
    and future sources like Tavily (news) or Gemini (annual report PDF analysis).
    """
    id: str                   # Unique evidence ID, e.g., "EVD-001"
    source: str               # Data source name, e.g., "pypsx-toolkit", "psx-mcp-server", "tavily", "gemini"
    metric_name: str          # Name of the financial/technical metric, e.g., "Annual EPS Growth"
    value: Any                # Numerical or textual value, e.g., 20.01, "574.34 PKR"
    as_of_date: str           # Date stamp or period reference, e.g., "2026-08-28" or "FY2025"
    raw_reference: str        # Origin reference (tool name, API field, PDF link, or text snippet)
    metadata: Optional[Dict[str, Any]]  # Extensible metadata key-values for future data sources


class AgentState(TypedDict, total=False):
    """Complete graph state schema for the PSX Stock Research Agent.
    
    The state is grouped into 6 logical domains:
    1. User / Intent
    2. Raw Data
    3. Calculated Analytics
    4. Evidence / Traceability
    5. Analysis / Reasoning
    6. Final Output
    """

    # -------------------------------------------------------------------------
    # 1. User / Intent Domain
    # -------------------------------------------------------------------------
    user_query: str                         # Raw user query prompt
    ticker: str                             # Extracted PSX stock ticker (e.g. "MEBL", "FFC")
    investment_strategy: Optional[str]      # Inferred strategy (e.g. "Dividend Growth", "Value")
    investment_horizon: Optional[str]       # Time horizon (e.g. "Long-term (3-5y)", "Short-term")
    user_thesis: Optional[str]              # User's initial investment hypothesis/assumption

    # -------------------------------------------------------------------------
    # 2. Raw Data Domain
    # -------------------------------------------------------------------------
    market_data: Dict[str, Any]             # Current price quote, LDCP, 52-wk high/low, volume
    historical_data: List[Dict[str, Any]]   # Daily OHLCV price history records
    financial_data: Dict[str, Any]          # Annual & Quarterly income statements, equity profile
    dividend_data: Dict[str, Any]           # Payout history, ex-dividend dates, T+2 buy deadlines
    announcements: List[Dict[str, Any]]      # Corporate disclosures, board meeting outcomes & PDF links

    # -------------------------------------------------------------------------
    # 3. Calculated Analytics Domain (Deterministic Python Output)
    # -------------------------------------------------------------------------
    technical_metrics: Dict[str, Any]       # Computed SMA50, SMA200, RSI(14), MACD, Volatility
    fundamental_metrics: Dict[str, Any]     # Computed Revenue CAGR, EPS Growth, Margins, PEG, Payout %

    # -------------------------------------------------------------------------
    # 4. Evidence & Tool Execution Domain (Step 6 Tool Calling Extensions)
    # -------------------------------------------------------------------------
    evidence_sources: List[EvidenceItem]    # Master list of traceable evidence records
    messages: List[Any]                     # Message trajectory for LangGraph tool-calling loop
    research_status: Optional[str]          # Research status ('planned', 'executing', 'completed', 'failed')
    tool_calls_executed: List[Dict[str, Any]] # Audit trail of executed tools & arguments
    tool_errors: List[Dict[str, Any]]       # Execution errors encountered during tool calls

    # -------------------------------------------------------------------------
    # 5. Analysis / Reasoning Domain (LLM Reasoning Output)
    # -------------------------------------------------------------------------
    reasoning_output: Optional[Dict[str, Any]] # Structured Pydantic output dict (investment_view, confidence, etc.)
    bull_case: Dict[str, Any]               # Bullish thesis, catalysts, upside potential
    bear_case: Dict[str, Any]               # Bearish risks, valuation traps, margin pressure
    thesis_challenge: Dict[str, Any]        # Critical evaluation & critique of user_thesis vs evidence

    # -------------------------------------------------------------------------
    # 6. Final Output Domain
    # -------------------------------------------------------------------------
    final_synthesis: Optional[str]          # Final Markdown formatted institutional research report


def create_initial_state(user_query: str, ticker: str = "") -> AgentState:
    """Helper factory function to create a clean, fully initialized AgentState.
    
    Ensures safe defaults for lists and dicts so graph nodes can be constructed
    and tested incrementally without encountering KeyError or NoneType issues.
    """
    return {
        "user_query": user_query,
        "ticker": ticker.upper(),
        "investment_strategy": None,
        "investment_horizon": None,
        "user_thesis": None,
        "market_data": {},
        "historical_data": [],
        "financial_data": {},
        "dividend_data": {},
        "announcements": [],
        "technical_metrics": {},
        "fundamental_metrics": {},
        "evidence_sources": [],
        "messages": [],
        "research_status": "planned",
        "tool_calls_executed": [],
        "tool_errors": [],
        "reasoning_output": None,
        "bull_case": {},
        "bear_case": {},
        "thesis_challenge": {},
        "final_synthesis": None,
    }
