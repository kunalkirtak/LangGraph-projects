"""
Graph assembly for the Enterprise Human Approval Workflow.

    START
      |
    validate
      |
    analyze
      |
    [route_risk] --low-risk--> execute --> END
      |
    high-risk
      |
    human_approval  (interrupt() pauses here; checkpointer persists state)
      |
    [route_approval] --approve--> execute --> END
      |
    reject --> reject --> END

Compiling with a checkpointer is what makes `interrupt()` meaningful: the
graph's state is saved at the interruption point, keyed by thread_id, so a
completely separate later call with `Command(resume=...)` can pick up
exactly where execution left off.
"""

from __future__ import annotations

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import RetryPolicy

from nodes import (
    TransientLLMError,
    analyze_request,
    execute_request,
    human_approval,
    reject_request,
    route_approval,
    route_risk,
    route_validation,
    validate_request,
)
from state import ApprovalState

# Only transient LLM failures are retried (see nodes.TransientLLMError).
# Validation errors, bad human input, and programming errors are never
# retried automatically - see README "Retry Logic" for the rationale.
ANALYZE_RETRY_POLICY = RetryPolicy(
    retry_on=TransientLLMError,
    max_attempts=3,
    initial_interval=0.5,
    backoff_factor=2.0,
)


def build_graph(checkpointer=None):
    """Construct and compile the approval workflow graph.

    Args:
        checkpointer: a LangGraph checkpointer. Defaults to an in-memory
            saver, which is sufficient for this portfolio demonstration but
            is explicitly NOT durable across process restarts (see README
            "Limitations"). Pass a different checkpointer (e.g. a Postgres
            one) to swap in durable persistence without touching this graph.
    """
    if checkpointer is None:
        checkpointer = InMemorySaver()

    builder = StateGraph(ApprovalState)

    builder.add_node("validate", validate_request)
    builder.add_node("analyze", analyze_request, retry_policy=ANALYZE_RETRY_POLICY)
    builder.add_node("human_approval", human_approval)
    builder.add_node("execute", execute_request)
    builder.add_node("reject", reject_request)

    builder.add_edge(START, "validate")

    builder.add_conditional_edges(
        "validate",
        route_validation,
        {
            "analyze": "analyze",
            "invalid": "execute",
        },
    )

    builder.add_conditional_edges(
        "analyze",
        route_risk,
        {
            "execute": "execute",
            "approval": "human_approval",
        },
    )

    builder.add_conditional_edges(
        "human_approval",
        route_approval,
        {
            "execute": "execute",
            "reject": "reject",
        },
    )

    builder.add_edge("execute", END)
    builder.add_edge("reject", END)

    return builder.compile(checkpointer=checkpointer)
