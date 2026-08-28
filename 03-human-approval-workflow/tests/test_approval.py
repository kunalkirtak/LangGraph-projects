"""
Focused tests for the human-in-the-loop interrupt/resume mechanism itself,
isolated from the risk-analysis LLM call.

These tests build a tiny graph containing only human_approval, execute, and
reject, wired exactly like the real workflow's high-risk branch, and drive
it with a real LangGraph checkpointer. This proves the graph genuinely
pauses at interrupt() and genuinely resumes via Command(resume=...) rather
than merely calling a Python function that asks for input().
"""

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command

from nodes import execute_request, human_approval, reject_request, route_approval
from state import ApprovalState


def _build_approval_only_graph():
    builder = StateGraph(ApprovalState)
    builder.add_node("human_approval", human_approval)
    builder.add_node("execute", execute_request)
    builder.add_node("reject", reject_request)

    builder.add_edge(START, "human_approval")
    builder.add_conditional_edges(
        "human_approval",
        route_approval,
        {"execute": "execute", "reject": "reject"},
    )
    builder.add_edge("execute", END)
    builder.add_edge("reject", END)

    return builder.compile(checkpointer=InMemorySaver())


def _base_state():
    return {
        "request_id": "REQ-TEST0001",
        "requester": "Alex",
        "department": "Engineering",
        "action": "Approve a large infrastructure purchase",
        "amount": 50000,
        "reason": "Production capacity expansion",
        "risk_score": 80,
        "risk_level": "HIGH",
        "risk_reason": "High financial exposure",
        "errors": [],
        "metadata": {},
    }


def test_graph_actually_pauses_at_interrupt():
    graph = _build_approval_only_graph()
    config = {"configurable": {"thread_id": "test-thread-pause"}}

    result = graph.invoke(_base_state(), config=config)

    # The graph must stop BEFORE execute/reject ever run.
    assert result.get("execution_status") in (None, "not_started")
    assert result.get("human_decision") is None

    snapshot = graph.get_state(config)
    assert len(snapshot.next) > 0  # graph has NOT reached END
    assert snapshot.interrupts, "expected a pending interrupt on the checkpoint"


def test_resume_with_approve_executes():
    graph = _build_approval_only_graph()
    config = {"configurable": {"thread_id": "test-thread-approve"}}

    graph.invoke(_base_state(), config=config)  # pauses here
    final_state = graph.invoke(Command(resume="approve"), config=config)

    assert final_state["human_decision"] == "approve"
    assert final_state["execution_status"] == "executed"
    assert "REQ-TEST0001" in final_state["execution_result"]

    snapshot = graph.get_state(config)
    assert len(snapshot.next) == 0  # graph reached END


def test_resume_with_reject_rejects():
    graph = _build_approval_only_graph()
    config = {"configurable": {"thread_id": "test-thread-reject"}}

    graph.invoke(_base_state(), config=config)  # pauses here
    final_state = graph.invoke(Command(resume="reject"), config=config)

    assert final_state["human_decision"] == "reject"
    assert final_state["execution_status"] == "rejected"
    assert final_state["approval_status"] == "rejected"


def test_resume_with_invalid_decision_fails_safe_to_reject():
    graph = _build_approval_only_graph()
    config = {"configurable": {"thread_id": "test-thread-invalid"}}

    graph.invoke(_base_state(), config=config)  # pauses here
    final_state = graph.invoke(Command(resume="maybe"), config=config)

    # An unrecognized decision must never result in execution.
    assert final_state["execution_status"] == "rejected"
    assert any("invalid human decision" in e for e in final_state["errors"])


def test_each_thread_id_is_independent():
    graph = _build_approval_only_graph()
    config_a = {"configurable": {"thread_id": "thread-a"}}
    config_b = {"configurable": {"thread_id": "thread-b"}}

    graph.invoke(_base_state(), config=config_a)
    graph.invoke(_base_state(), config=config_b)

    result_a = graph.invoke(Command(resume="approve"), config=config_a)
    result_b = graph.invoke(Command(resume="reject"), config=config_b)

    assert result_a["execution_status"] == "executed"
    assert result_b["execution_status"] == "rejected"
