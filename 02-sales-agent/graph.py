"""
graph.py

Builds the actual LangGraph workflow for the sales lead agent.

Two conditional edges drive the whole project:

1. After ``normalize`` -- obviously invalid leads never reach the LLM.
2. After ``qualify`` -- qualified leads go to research/outreach, everyone
   else goes to nurture.

Both routing decisions are made with ``add_conditional_edges`` reading
state that LangGraph itself manages. There is no ordinary Python
``if/else`` wrapped around graph execution to fake branching.
"""

from __future__ import annotations

from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import RetryPolicy

from nodes import (
    generate_nurture,
    generate_outreach,
    normalize_lead,
    qualify_lead,
    research_lead,
)
from state import SalesState
from utils import TransientLLMError, logger

# Nodes that call an LLM get a graph-level retry policy on top of the
# node-internal retry helper, so transient provider failures are retried
# even if they escape the node's own retry loop. Validation-style errors
# are plain ValueErrors/dict updates, never exceptions, so they are never
# retried here.
_LLM_RETRY_POLICY = RetryPolicy(max_attempts=2, retry_on=(TransientLLMError,))


def route_after_normalize(state: SalesState) -> str:
    """Return the next route key after normalization.

    An obviously invalid lead (missing required fields) must never reach
    the qualification LLM call -- it is routed straight to the end of the
    graph with its errors already recorded in state.
    """

    if state.get("errors"):
        logger.info("Routing lead: invalid_input")
        return "invalid"
    logger.info("Routing lead: valid_input")
    return "valid"


def route_lead(state: SalesState) -> str:
    """Return the next route key after qualification.

    This is the central conditional-routing decision of the project: the
    graph -- not ad-hoc Python control flow -- decides whether a lead
    proceeds to research/outreach or to nurture, based purely on state
    that the qualification node wrote.
    """

    if state.get("errors") and not state.get("qualification_status"):
        # Qualification itself failed (e.g. exhausted retries).
        logger.info("Routing lead: qualification_failed -> nurture")
        return "unqualified"

    status = state.get("qualification_status", "unqualified")
    logger.info("Routing lead: %s", status)
    return status


def build_graph() -> CompiledStateGraph:
    """Construct and compile the sales lead StateGraph."""

    builder = StateGraph(SalesState)

    builder.add_node("normalize", normalize_lead)
    builder.add_node("qualify", qualify_lead, retry_policy=_LLM_RETRY_POLICY)
    builder.add_node("research", research_lead, retry_policy=_LLM_RETRY_POLICY)
    builder.add_node("outreach", generate_outreach, retry_policy=_LLM_RETRY_POLICY)
    builder.add_node("nurture", generate_nurture, retry_policy=_LLM_RETRY_POLICY)

    builder.add_edge(START, "normalize")

    builder.add_conditional_edges(
        "normalize",
        route_after_normalize,
        {
            "valid": "qualify",
            "invalid": END,
        },
    )

    builder.add_conditional_edges(
        "qualify",
        route_lead,
        {
            "qualified": "research",
            "unqualified": "nurture",
        },
    )

    builder.add_edge("research", "outreach")
    builder.add_edge("outreach", END)
    builder.add_edge("nurture", END)

    return builder.compile()


def get_graph_ascii() -> str:
    """Return a printable representation of the compiled graph, for demos."""

    graph = build_graph()
    try:
        return graph.get_graph().draw_ascii()
    except Exception:  # noqa: BLE001 - ascii rendering needs an extra dep
        return graph.get_graph().print_ascii() or ""
