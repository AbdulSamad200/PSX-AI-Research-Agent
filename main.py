"""PSXenius Stock Agent Main CLI Entrypoint (Step 7 Integrated Graph).

Invokes the single unified LangGraph research workflow:
User Query -> Research Planner & Tool Execution -> Deterministic Analysis Engine -> OpenAI Reasoning -> Final State Report
"""

import sys
import os
import json
import argparse
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# UTF-8 stdout encoding for Windows console compatibility
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from src.agent.state import create_initial_state
from src.agent.graph import create_full_research_graph


# =============================================================================
# 📌 USER CONFIGURATION VARIABLE
# Change this ticker variable to analyze any stock (e.g., "MEBL", "FFC", "BAFL", "HUBC")
# =============================================================================
COMPANY_TICKER = "MEBL"


def run_agent_workflow(ticker: str, user_query: str):
    """Executes the complete single-graph PSX Stock Agent workflow."""
    ticker = ticker.upper()
    print(f"\n================================================================================")
    print(f" 🚀 PSXenius Agent Unified Graph Execution for: [{ticker}]")
    print(f" Query: {user_query}")
    print(f"================================================================================\n")

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key or api_key == "your_openai_api_key_here":
        print("⚠️ WARNING: OPENAI_API_KEY is not configured in your .env file!")
        print("Please edit the .env file in project root and replace 'your_openai_api_key_here' with your real API key.\n")

    # Step 1: Initialize Agent State
    state = create_initial_state(user_query=user_query, ticker=ticker)

    # Step 2: Invoke Single Integrated LangGraph Workflow
    print("📌 Executing Unified LangGraph Workflow (Planning -> Tools -> Engine -> Reasoning)...")
    graph = create_full_research_graph()

    try:
        final_state = graph.invoke(state)

        executed_tools = [call.get("tool_name") for call in final_state.get("tool_calls_executed", [])]
        print(f"\n✓ Executed Tools: {executed_tools if executed_tools else 'None'}")

        tech = final_state.get("technical_metrics", {})
        if tech:
            print(f"✓ Technical Metrics Calculated:")
            print(f"   • SMA 50: {tech.get('sma_50', {}).get('value')} | SMA 200: {tech.get('sma_200', {}).get('value')}")
            print(f"   • RSI 14: {tech.get('rsi_14', {}).get('value')} | 5Y CAGR: {tech.get('return_5y_cagr', {}).get('value')}%")
            print(f"   • TTM Dividend Yield: {tech.get('ttm_dividend_yield', {}).get('value')}% | Max Drawdown: {tech.get('max_drawdown_5y', {}).get('value')}%")

        reasoning = final_state.get("reasoning_output")
        if reasoning:
            print(f"\n================================================================================")
            print(f" 🤖 OPENAI STRUCTURED REASONING REPORT: [{ticker}]")
            print(f"================================================================================")
            print(f"\n💡 Investment View: {reasoning.get('investment_view')}")
            print(f"🎯 Confidence Score: {reasoning.get('confidence')} / 1.00")
            print(f"🔗 Referenced Evidence IDs: {reasoning.get('evidence_ids')}\n")
            
            print(f"📊 Fundamental Interpretation:\n   {reasoning.get('fundamental_interpretation')}\n")
            print(f"📈 Technical Interpretation:\n   {reasoning.get('technical_interpretation')}\n")
            print(f"💰 Dividend Interpretation:\n   {reasoning.get('dividend_interpretation')}\n")
            
            print("✅ Key Strengths:")
            for strength in reasoning.get('key_strengths', []):
                print(f"   • {strength}")
                
            print("\n⚠️ Key Risks:")
            for risk in reasoning.get('key_risks', []):
                print(f"   • {risk}")

            print(f"\n📋 Thesis Assessment:\n   {reasoning.get('thesis_assessment')}")
            print(f"================================================================================\n")

    except Exception as e:
        print(f"❌ Graph execution error: {e}")


def main():
    parser = argparse.ArgumentParser(description="PSXenius Stock Agent CLI")
    parser.add_argument("ticker", nargs="?", default=COMPANY_TICKER, help=f"PSX stock ticker (default: {COMPANY_TICKER})")
    parser.add_argument("--query", "-q", help="Custom research query prompt")

    args = parser.parse_args()

    ticker = args.ticker.upper()
    query = args.query or f"Analyze {ticker} for a long-term dividend-growth investment over 3–5 years. Evaluate its recent price performance, fundamentals, dividends, and technical position"

    run_agent_workflow(ticker=ticker, user_query=query)


if __name__ == "__main__":
    main()
