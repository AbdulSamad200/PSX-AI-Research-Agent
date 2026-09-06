"""LangGraph nodes for PSX Stock Research Agent reasoning, analysis, and tool execution."""

import os
import json
from typing import Dict, Any, Optional, List
from openai import OpenAI
from pydantic import ValidationError

from src.agent.state import AgentState, EvidenceItem
from src.agent.models import StructuredReasoningOutput
from src.analysis.technical import compute_all_technical_metrics
from src.analysis.performance import compute_all_performance_metrics
from src.analysis.dividends import compute_all_dividend_metrics
from src.analysis.validation import compute_all_validations
from src.analysis.engine import extract_source_metrics


REASONING_SYSTEM_PROMPT = """You are the Senior Financial Reasoning Engine for PSXenius (Pakistan Stock Exchange Stock Analysis Agent).
Your sole responsibility is to INTERPRET and ANALYZE the financial evidence, technical indicators, and fundamental metrics provided in the user's research state.

CRITICAL RULES & GROUNDING CONSTRAINTS:
1. STRICT EVIDENCE GROUNDING: Every numerical or factual claim must be supported by a value present in the provided state/evidence context. Do not introduce financial ratios, figures, or metrics that are absent from the supplied state.
2. DO NOT RECALCULATE METRICS: Do not attempt to compute or recalculate moving averages, CAGR, margins, or yields. Interpret the calculated metrics provided.
3. DISTINGUISH METRIC PROVENANCE:
   - Source-provided metrics (e.g., reported Net Margin, EPS Growth, PEG, Market Cap) come directly from PSX data sources/tools.
   - PSXenius-derived metrics (e.g., Return 1Y/3Y, SMA 50/200, RSI 14, MACD, Max Drawdown, Volatility, TTM Yield, Dividend CAGR) are calculated deterministically by our Python engine.
4. FACTS VS. INTERPRETATION: State factual data points clearly before providing your professional analytical interpretation.
5. EXPLICIT EVIDENCE CITING: Every evidence ID included in the 'evidence_ids' array of your output MUST actually exist in the supplied 'evidence_sources' list. Cite specific Evidence IDs (e.g. EVD-FUND-MEBL-NPM, EVD-TECH-MEBL-SMA50) whenever referencing data points.
6. NEGATIVE PEG RATIO GROUNDING:
   - Do NOT treat a negative PEG ratio as conventional evidence of undervaluation.
   - When PEG is negative and EPS growth is negative, explain that the PEG ratio is negative due to negative earnings growth in that period and is not a conventional valuation signal.
7. DIVIDEND & PERIOD SEMANTIC ACCURACY:
   - Always distinguish historical annual values (e.g. FY2024, FY2025) from current values, TTM (Trailing Twelve Months) totals, and YTD values (e.g. 2026 YTD).
   - NEVER compare a YTD dividend total with a prior full-year dividend total as YoY growth or YoY decline.
   - NEVER call a YTD dividend amount a projected full-year dividend or annual forecast.
8. NO UNSUPPORTED FORECASTS OR GUARANTEES:
   - Do not make unsupported predictions, forward projections, or guarantees about future returns (e.g., 'guaranteed stable returns over the next 3-5 years').
   - The user's 3-5 year horizon may be discussed as an investment thesis context, but historical evidence must not be presented as a forecast.
   - Technical indicators describe historical/current market conditions; they do not guarantee future price movement.
9. HANDLE MISSING EVIDENCE GRACEFULLY: If specific required metrics or data points are missing or unavailable in state, explicitly state that they are unavailable rather than guessing or hallucinating.
10. NO SCORED BUY/SELL/HOLD: Do NOT output explicit BUY/SELL/HOLD recommendations or numerical stock ratings. Keep the output strictly descriptive, objective, and analytical.
"""


def _format_state_payload(state: AgentState) -> str:
    """Formats the relevant sections of AgentState into a structured JSON string for the prompt."""
    payload = {
        "ticker": state.get("ticker", "UNKNOWN"),
        "user_query": state.get("user_query", ""),
        "user_thesis": state.get("user_thesis"),
        "investment_strategy": state.get("investment_strategy"),
        "investment_horizon": state.get("investment_horizon"),
        "fundamental_metrics": state.get("fundamental_metrics", {}),
        "technical_metrics": state.get("technical_metrics", {}),
        "dividend_data": state.get("dividend_data", {}),
        "announcements": state.get("announcements", []),
        "evidence_sources": state.get("evidence_sources", []),
    }
    return json.dumps(payload, indent=2, default=str)


def run_openai_reasoning(
    state: AgentState,
    client: Optional[Any] = None,
    model: Optional[str] = None
) -> StructuredReasoningOutput:
    """Executes structured OpenAI reasoning over the provided AgentState.
    
    Args:
        state: The current AgentState payload.
        client: Optional OpenAI client or mock client for dependency injection.
        model: Optional model name override. Defaults to OPENAI_MODEL environment variable or 'gpt-4o-mini'.
        
    Returns:
        StructuredReasoningOutput Pydantic model instance.
    """
    if client is None:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable is not set.")
        client = OpenAI(api_key=api_key)

    model_name = model or os.getenv("OPENAI_MODEL", "gpt-4o-mini")

    state_json = _format_state_payload(state)
    messages = [
        {"role": "system", "content": REASONING_SYSTEM_PROMPT},
        {"role": "user", "content": f"Please interpret and analyze the following stock research state evidence:\n\n{state_json}"}
    ]

    completion = client.chat.completions.parse(
        model=model_name,
        messages=messages,
        response_format=StructuredReasoningOutput,
    )

    message = completion.choices[0].message
    if message.parsed:
        return message.parsed
    elif message.refusal:
        raise ValueError(f"OpenAI model refused reasoning request: {message.refusal}")
    else:
        raise ValueError("Failed to parse response into StructuredReasoningOutput.")


def openai_reasoning_node(
    state: AgentState,
    client: Optional[Any] = None,
    model: Optional[str] = None
) -> Dict[str, Any]:
    """LangGraph node function that executes OpenAI reasoning and updates state.
    
    Returns:
        Dict update for AgentState containing {'reasoning_output': dict}.
    """
    structured_output = run_openai_reasoning(state=state, client=client, model=model)
    return {"reasoning_output": structured_output.model_dump()}


# =============================================================================
# Step 7: Deterministic Analysis Engine Node (Consumes Raw State Data)
# =============================================================================

def analysis_engine_node(state: AgentState) -> Dict[str, Any]:
    """Graph node executing deterministic financial and technical analysis using state raw data.
    
    Consumes ONLY raw data already collected in AgentState (historical_data, market_data,
    financial_data, dividend_data). Performs ZERO redundant external downloads.
    """
    ticker = state.get("ticker", "UNKNOWN").upper()
    historical_data = state.get("historical_data", [])
    market_data = state.get("market_data", {})
    financial_data = state.get("financial_data", {})
    dividend_data = state.get("dividend_data", {})

    # Extract OHLCV series from state historical_data
    closes, highs, lows, dates = [], [], [], []
    if isinstance(historical_data, list) and len(historical_data) > 0:
        for bar in historical_data:
            if isinstance(bar, dict) and "close" in bar and "date" in bar:
                closes.append(bar["close"])
                highs.append(bar.get("high", bar["close"]))
                lows.append(bar.get("low", bar["close"]))
                dates.append(bar["date"])

    # Extract current price: prefer market_data current_price, else latest OHLCV close
    current_price = market_data.get("current_price")
    if current_price is None and closes:
        current_price = closes[-1]

    # 1. Fundamental / Source Metrics Extraction
    toolkit_payload = {
        "symbol": ticker,
        "fundamentals": financial_data.get("fundamentals", financial_data),
        "financial_statements": financial_data.get("financial_statements", financial_data),
    }
    source_metrics = extract_source_metrics(toolkit_payload, mcp_quote=market_data)

    # 2. Technical & Performance Derived Metrics
    technical_metrics = {}
    if closes:
        tech_results = compute_all_technical_metrics(closes, highs, lows, dates)
        perf_results = compute_all_performance_metrics(closes, dates)
        technical_metrics.update(tech_results)
        technical_metrics.update(perf_results)

    # 3. Dividend Derived Metrics (Prioritize toolkit cash history)
    div_history = dividend_data.get("history")
    if not div_history and isinstance(dividend_data.get("payouts"), list):
        div_history = dividend_data["payouts"]
    if isinstance(div_history, list) and div_history:
        div_results = compute_all_dividend_metrics(div_history, current_price, ohlcv_bars=historical_data)
        technical_metrics.update(div_results)

    # 4. Statement vs Ratio Validation Check
    annuals = financial_data.get("financial_statements", {}).get("annual_financials", financial_data.get("annual_financials", {}))
    ratios = financial_data.get("fundamentals", {}).get("ratios", financial_data.get("ratios", {}))
    validation_results = compute_all_validations(annuals, ratios)
    source_metrics["net_profit_margin_validation"] = validation_results.get("net_profit_margin_validation", {"status": "not_applicable"})

    # 5. Build Traceable Evidence Sources
    evidence_sources = list(state.get("evidence_sources", []))
    existing_ids = {item.get("id") if isinstance(item, dict) else item.get("id") for item in evidence_sources if isinstance(item, dict)}
    as_of_date = dates[-1] if dates else "Latest"

    def add_evd(evd_id: str, source: str, metric_name: str, value: Any, date_ref: str, raw_ref: str, is_derived: bool):
        if value is not None and evd_id not in existing_ids:
            evidence_sources.append(EvidenceItem(
                id=evd_id,
                source=source,
                metric_name=metric_name,
                value=str(value) if not isinstance(value, (str, dict, list)) else value,
                as_of_date=date_ref,
                raw_reference=raw_ref,
                metadata={"metric_type": "psxenius_derived" if is_derived else "source_provided"}
            ))
            existing_ids.add(evd_id)

    # Fundamental Source Evidence
    npm_val = source_metrics.get("source_net_profit_margin_pct", {}).get("value")
    add_evd(f"EVD-FUND-{ticker}-NPM", "pypsx-toolkit", "Net Profit Margin (%)", npm_val, "Multi-Year", "toolkit_get_company_fundamentals", False)

    epsg_val = source_metrics.get("source_eps_growth_pct", {}).get("value")
    add_evd(f"EVD-FUND-{ticker}-EPSG", "pypsx-toolkit", "EPS Growth (%)", epsg_val, "Multi-Year", "toolkit_get_company_fundamentals", False)

    peg_val = source_metrics.get("source_peg", {}).get("value")
    add_evd(f"EVD-FUND-{ticker}-PEG", "pypsx-toolkit", "PEG Ratio", peg_val, "Multi-Year", "toolkit_get_company_fundamentals", False)

    mcap_val = source_metrics.get("source_market_cap_000s", {}).get("value")
    add_evd(f"EVD-FUND-{ticker}-MCAP", "pypsx-toolkit", "Market Cap (000s PKR)", mcap_val, "Latest", "toolkit_get_company_fundamentals", False)

    # Technical & Performance Derived Evidence
    ret1y = technical_metrics.get("return_1y", {}).get("value")
    add_evd(f"EVD-TECH-{ticker}-RET1Y", "PSXenius-Derived Engine", "1-Year Return (%)", f"{ret1y}%" if ret1y is not None else None, as_of_date, "compute_all_performance_metrics", True)

    ret3y = technical_metrics.get("return_3y", {}).get("value")
    add_evd(f"EVD-TECH-{ticker}-RET3Y", "PSXenius-Derived Engine", "3-Year Return (%)", f"{ret3y}%" if ret3y is not None else None, as_of_date, "compute_all_performance_metrics", True)

    sma50 = technical_metrics.get("sma_50", {}).get("value")
    add_evd(f"EVD-TECH-{ticker}-SMA50", "PSXenius-Derived Engine", "50-Day SMA", f"{sma50} PKR" if sma50 is not None else None, as_of_date, "compute_all_technical_metrics", True)

    sma200 = technical_metrics.get("sma_200", {}).get("value")
    add_evd(f"EVD-TECH-{ticker}-SMA200", "PSXenius-Derived Engine", "200-Day SMA", f"{sma200} PKR" if sma200 is not None else None, as_of_date, "compute_all_technical_metrics", True)

    rsi14 = technical_metrics.get("rsi_14", {}).get("value")
    add_evd(f"EVD-TECH-{ticker}-RSI14", "PSXenius-Derived Engine", "14-Day RSI", rsi14, as_of_date, "compute_all_technical_metrics", True)

    # Dividend Derived Evidence
    ttm_yield = technical_metrics.get("ttm_dividend_yield", {}).get("value")
    add_evd(f"EVD-DIV-{ticker}-TTMYIELD", "PSXenius-Derived Engine", "TTM Dividend Yield (%)", f"{ttm_yield}%" if ttm_yield is not None else None, "Latest", "compute_all_dividend_metrics", True)

    div_cagr = technical_metrics.get("dividend_cagr_3y", {}).get("value")
    add_evd(f"EVD-DIV-{ticker}-CAGR3Y", "PSXenius-Derived Engine", "3-Year Dividend CAGR (%)", f"{div_cagr}%" if div_cagr is not None else None, "Latest", "compute_all_dividend_metrics", True)

    hist_yields = technical_metrics.get("annual_historical_dividend_yields", {}).get("value")
    add_evd(f"EVD-DIV-{ticker}-HIST", "PSXenius-Derived Engine", "Annual Historical Dividend Yields & Payouts", hist_yields, "Multi-Year", "compute_all_dividend_metrics", True)

    return {
        "fundamental_metrics": source_metrics,
        "technical_metrics": technical_metrics,
        "evidence_sources": evidence_sources,
        "research_status": "analyzed"
    }


# =============================================================================
# Step 6 & 7: LangGraph Tool Calling & Research Planning Nodes
# =============================================================================

from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, ToolMessage
from langchain_openai import ChatOpenAI
from src.agent.tools_registry import ALL_TOOLS, get_tool_by_name

PLANNER_SYSTEM_PROMPT = """You are the PSX Stock Research Planner.
Your sole responsibility is to analyze the user's research query and determine which PSX tools are required to gather the requested data.

AVAILABLE TOOLS & TOOL RESPONSIBILITIES:
- mcp_get_quote: For real-time price quotes, LDCP, daily high/low range from psx-mcp-server.
- mcp_get_company_info: For company profile, market cap, shares outstanding, free float from psx-mcp-server.
- mcp_get_dividends: For recent dividend announcements, declarations, book closure dates, and upcoming payout events from psx-mcp-server. Do NOT use this for calculating historical cash yield or dividend CAGR.
- mcp_get_announcements: For official corporate disclosures & board meeting outcomes from psx-mcp-server.
- toolkit_get_stock_ohlcv: For historical OHLCV price records from pypsx-toolkit.
  * If the user query requires 3-year performance, 3–5 year investment analysis, 5-year performance, or multi-year technical/performance analysis, you MUST pass period="5y".
  * If only approximately 1-year or short-term performance is requested, period="1y" is acceptable.
- toolkit_get_financial_statements: For annual & quarterly income statements, balance sheets from pypsx-toolkit.
- toolkit_get_company_fundamentals: For key ratios (Net Margin, EPS Growth, PEG) & equity profile from pypsx-toolkit.
- toolkit_get_dividend_history: PREFERRED for all dividend history, historical cash dividend amounts, annual dividend totals, TTM dividend yield, and dividend CAGR from pypsx-toolkit. Whenever the query asks about dividend yield, dividend growth, dividend history, or dividend analysis, you MUST select toolkit_get_dividend_history.

RULES & ANTI-REDUNDANCY PROTECTION:
1. Select ONLY tools that are strictly required to fulfill the user's query.
2. For queries involving multi-year (3-5y) performance or investment horizons, ALWAYS call toolkit_get_stock_ohlcv with period="5y".
3. For queries involving dividend yields, dividend growth, or dividend history, ALWAYS call toolkit_get_dividend_history.
4. Check previously executed tools before requesting any tool. Do NOT request a tool that has already been executed successfully.
5. If all required data has already been retrieved, output your final plan message without tool_calls.
6. Do NOT provide financial recommendations or stock analyses. Your goal is strictly tool selection.
"""


def research_planner_node(
    state: AgentState,
    llm: Optional[Any] = None,
    model: Optional[str] = None
) -> Dict[str, Any]:
    """Research Planner node that analyzes state/user_query and generates tool calls."""
    messages = list(state.get("messages", []))
    executed_tools = [c.get("tool_name") for c in state.get("tool_calls_executed", [])]

    # Ensure SystemMessage(content=PLANNER_SYSTEM_PROMPT) is always present
    if not messages or not any(isinstance(m, SystemMessage) for m in messages):
        user_query = state.get("user_query", "")
        if not user_query and messages:
            first_msg = messages[0]
            if isinstance(first_msg, tuple):
                user_query = first_msg[1]
            elif hasattr(first_msg, "content"):
                user_query = first_msg.content
        ticker = state.get("ticker", "")
        content = f"Query: {user_query}" if not ticker else f"Ticker: {ticker}. Query: {user_query}"
        if executed_tools:
            content += f"\nPreviously executed tools: {executed_tools}"
        messages = [
            SystemMessage(content=PLANNER_SYSTEM_PROMPT),
            HumanMessage(content=content)
        ]

    if llm is None:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable is not set.")
        model_name = model or os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        llm = ChatOpenAI(model=model_name, api_key=api_key)

    llm_with_tools = llm.bind_tools(ALL_TOOLS)
    response = llm_with_tools.invoke(messages)

    updated_messages = messages + [response]
    has_tool_calls = hasattr(response, "tool_calls") and bool(response.tool_calls)
    next_status = "executing" if has_tool_calls else "completed"

    return {
        "messages": updated_messages,
        "research_status": next_status
    }


def tool_execution_node(state: AgentState) -> Dict[str, Any]:
    """Tool execution node that invokes tools requested by the Research Planner and updates AgentState."""
    messages = list(state.get("messages", []))
    if not messages:
        return {}

    last_msg = messages[-1]
    tool_calls = getattr(last_msg, "tool_calls", [])
    if not tool_calls:
        return {"research_status": "completed"}

    market_data = dict(state.get("market_data", {}))
    historical_data = list(state.get("historical_data", []))
    financial_data = dict(state.get("financial_data", {}))
    dividend_data = dict(state.get("dividend_data", {}))
    announcements = list(state.get("announcements", []))

    executed_log = list(state.get("tool_calls_executed", []))
    errors_log = list(state.get("tool_errors", []))
    already_executed_names = {c.get("tool_name") for c in executed_log if c.get("status") == "success"}
    new_messages = []

    for call in tool_calls:
        tool_name = call.get("name")
        args = call.get("args", {})
        call_id = call.get("id", f"call_{len(executed_log)}")

        # Redundant call protection: if already executed successfully, reuse
        if tool_name in already_executed_names:
            skip_msg = f"Tool '{tool_name}' already executed successfully. Skipping duplicate network call."
            new_messages.append(ToolMessage(content=json.dumps({"info": skip_msg}), tool_call_id=call_id))
            continue

        tool_obj = get_tool_by_name(tool_name)
        if not tool_obj:
            err_msg = f"Tool '{tool_name}' not found in registry."
            errors_log.append({"tool_name": tool_name, "error": err_msg})
            new_messages.append(ToolMessage(content=json.dumps({"error": err_msg}), tool_call_id=call_id))
            continue

        try:
            res_str = tool_obj.invoke(args)
            try:
                res_data = json.loads(res_str)
            except Exception:
                res_data = res_str

            # Populate state domains based on tool type
            if tool_name in ["mcp_get_quote", "mcp_get_company_info"]:
                market_data.update(res_data if isinstance(res_data, dict) else {"raw": res_data})
            elif tool_name in ["mcp_get_dividends", "toolkit_get_dividend_history"]:
                dividend_data.update(res_data if isinstance(res_data, dict) else {"raw": res_data})
            elif tool_name == "mcp_get_announcements":
                announcements.extend(res_data if isinstance(res_data, list) else [res_data])
            elif tool_name == "toolkit_get_stock_ohlcv":
                if isinstance(res_data, dict) and "ohlcv" in res_data:
                    historical_data = res_data["ohlcv"]
                elif isinstance(res_data, list):
                    historical_data = res_data
            elif tool_name in ["toolkit_get_financial_statements", "toolkit_get_company_fundamentals"]:
                financial_data.update(res_data if isinstance(res_data, dict) else {"raw": res_data})

            executed_log.append({
                "tool_name": tool_name,
                "args": args,
                "status": "success",
                "tool_call_id": call_id
            })
            already_executed_names.add(tool_name)
            new_messages.append(ToolMessage(content=res_str, tool_call_id=call_id))

        except Exception as e:
            err_str = f"Execution error in '{tool_name}': {str(e)}"
            errors_log.append({"tool_name": tool_name, "args": args, "error": err_str})
            executed_log.append({
                "tool_name": tool_name,
                "args": args,
                "status": "failed",
                "tool_call_id": call_id
            })
            new_messages.append(ToolMessage(content=json.dumps({"error": err_str}), tool_call_id=call_id))

    return {
        "messages": messages + new_messages,
        "market_data": market_data,
        "historical_data": historical_data,
        "financial_data": financial_data,
        "dividend_data": dividend_data,
        "announcements": announcements,
        "tool_calls_executed": executed_log,
        "tool_errors": errors_log,
        "research_status": "planned"
    }


def should_continue_tools(state: AgentState) -> str:
    """Conditional edge function determining whether to execute tool calls or route to analysis engine."""
    messages = state.get("messages", [])
    if not messages:
        return "analysis_engine"

    last_msg = messages[-1]
    tool_calls = getattr(last_msg, "tool_calls", [])
    
    if tool_calls and len(state.get("tool_calls_executed", [])) < 10:
        return "tools"
    return "analysis_engine"
