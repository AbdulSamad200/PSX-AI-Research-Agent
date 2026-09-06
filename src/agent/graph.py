"""LangGraph workflow definitions for the PSX Stock Research Agent (Step 5, Step 6, & Step 7)."""

from typing import Optional, Any
from langgraph.graph import StateGraph, START, END

from src.agent.state import AgentState
from src.agent.nodes import (
    openai_reasoning_node,
    research_planner_node,
    tool_execution_node,
    analysis_engine_node,
    should_continue_tools,
)


def create_full_research_graph(
    llm: Optional[Any] = None,
    client: Optional[Any] = None,
    model: Optional[str] = None
):
    """Constructs and compiles the complete Step 7 end-to-end LangGraph research workflow.
    
    Graph Topology:
        START ---> planner ───(should_continue_tools)───> tools (tool_execution_node)
                     │                                       │
                     │ ("analysis_engine")                   │
                     ▼                                       │
            analysis_engine <────────────────────────────────┘
                     │
                     ▼
              openai_reasoning
                     │
                     ▼
                    END
    """
    def planner_wrapper(state: AgentState):
        return research_planner_node(state, llm=llm, model=model)

    def reasoning_wrapper(state: AgentState):
        return openai_reasoning_node(state, client=client, model=model)

    builder = StateGraph(AgentState)
    builder.add_node("planner", planner_wrapper)
    builder.add_node("tools", tool_execution_node)
    builder.add_node("analysis_engine", analysis_engine_node)
    builder.add_node("openai_reasoning", reasoning_wrapper)

    builder.add_edge(START, "planner")
    builder.add_conditional_edges(
        "planner",
        should_continue_tools,
        {
            "tools": "tools",
            "analysis_engine": "analysis_engine"
        }
    )
    builder.add_edge("tools", "planner")
    builder.add_edge("analysis_engine", "openai_reasoning")
    builder.add_edge("openai_reasoning", END)

    return builder.compile()


def create_tool_calling_graph(llm: Optional[Any] = None, model: Optional[str] = None):
    """Constructs and compiles the Step 6 Tool Calling & Research Planning graph."""
    def planner_wrapper(state: AgentState):
        return research_planner_node(state, llm=llm, model=model)

    builder = StateGraph(AgentState)
    builder.add_node("planner", planner_wrapper)
    builder.add_node("tools", tool_execution_node)

    builder.add_edge(START, "planner")
    builder.add_conditional_edges(
        "planner",
        should_continue_tools,
        {
            "tools": "tools",
            "analysis_engine": END
        }
    )
    builder.add_edge("tools", "planner")

    return builder.compile()


def create_research_graph(client: Optional[Any] = None, model: Optional[str] = None):
    """Constructs and compiles the Step 5 Reasoning graph."""
    def node_wrapper(state: AgentState):
        return openai_reasoning_node(state, client=client, model=model)

    builder = StateGraph(AgentState)
    builder.add_node("openai_reasoning", node_wrapper)
    builder.add_edge(START, "openai_reasoning")
    builder.add_edge("openai_reasoning", END)

    return builder.compile()
