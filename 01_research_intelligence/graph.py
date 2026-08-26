"""
Graph assembly for the AI Research Intelligence Pipeline.

This is the only module that actually builds and compiles the
LangGraph `StateGraph`. The execution path is:

    START -> research -> analysis -> report -> END

The graph — not a plain Python function calling three functions in a
row — owns orchestration: node registration, edges, and (optionally)
per-node retry policy for transient LLM/API failures.
"""

from __future__ import annotations

from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import RetryPolicy

from config import load_settings
from nodes import LLMClient, make_analysis_node, make_report_node, make_research_node
from state import ResearchState


def get_llm() -> LLMClient:
    """Build the real Gemini client used outside of tests.

    Imported lazily inside the function so that modules which only need
    the graph *shape* (e.g. `test_graph.py` with a FakeLLM) never need
    `langchain_google_genai` to be importable with a configured key.
    """
    from langchain_google_genai import ChatGoogleGenerativeAI

    settings = load_settings()
    if not settings.google_api_key:
        raise RuntimeError(
            "GOOGLE_API_KEY is not set. Copy .env.example to .env and add your "
            "key, or set it via Colab Secrets (see README.md)."
        )
    return ChatGoogleGenerativeAI(
        model=settings.gemini_model,
        google_api_key=settings.google_api_key,
    )


def build_graph(llm: LLMClient | None = None) -> CompiledStateGraph:
    """Construct and compile the research pipeline StateGraph.

    Args:
        llm: An object implementing `.invoke(prompt) -> response`. If
            omitted, a real `ChatGoogleGenerativeAI` client is created
            from environment configuration. Tests pass a `FakeLLM`
            here instead, so the graph can be compiled and executed
            without any API key or network access.
    """
    resolved_llm = llm if llm is not None else get_llm()
    settings = load_settings()

    retry_policy = RetryPolicy(
        max_attempts=settings.max_retries,
        initial_interval=settings.retry_initial_interval,
        backoff_factor=settings.retry_backoff_factor,
        retry_on=Exception,
    )

    graph = StateGraph(ResearchState)

    graph.add_node("research", make_research_node(resolved_llm), retry_policy=retry_policy)
    graph.add_node("analysis", make_analysis_node(resolved_llm), retry_policy=retry_policy)
    graph.add_node("report", make_report_node(resolved_llm), retry_policy=retry_policy)

    graph.add_edge(START, "research")
    graph.add_edge("research", "analysis")
    graph.add_edge("analysis", "report")
    graph.add_edge("report", END)

    return graph.compile()
