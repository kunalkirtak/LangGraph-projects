"""
End-to-end graph tests using a mocked LLM, exercising the full
validate -> analyze -> route -> (execute | human_approval -> execute/reject)
pipeline for every branch: low risk, high risk + approve, and high risk +
reject. No Gemini API key or network access is required.
"""

import pytest
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

import nodes
from graph import build_graph
from nodes import RiskAssessment, TransientLLMError
from state import new_state


class FakeStructuredLLM:
    """Stands in for `base_llm.with_structured_output(RiskAssessment)`."""

    def __init__(self, responses):
        # responses: list of RiskAssessment | Exception, consumed in order
        self._responses = list(responses)

    def invoke(self, _prompt):
        response = self._responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response


@pytest.fixture(autouse=True)
def _reset_llm_cache():
    nodes.reset_llm_cache()
    yield
    nodes.reset_llm_cache()


def _mock_llm(monkeypatch, responses):
    fake = FakeStructuredLLM(responses)
    monkeypatch.setattr(nodes, "_get_structured_llm", lambda: fake)
    return fake


def _new_graph():
    return build_graph(checkpointer=InMemorySaver())


# ---------------------------------------------------------------------------
# LOW RISK
# ---------------------------------------------------------------------------
def test_low_risk_request_executes_without_interruption(monkeypatch):
    _mock_llm(
        monkeypatch,
        [RiskAssessment(score=15, level="LOW", reason="Trivial purchase", risk_factors=[])],
    )

    graph = _new_graph()
    config = {"configurable": {"thread_id": "low-risk-thread"}}
    state = new_state("Alex", "Engineering", "Buy a mouse", 25, "Broken mouse")

    result = graph.invoke(state, config=config)

    assert result["risk_level"] == "LOW"
    assert result["approval_required"] is False
    assert result["execution_status"] == "executed"
    assert "human_decision" not in result or result.get("human_decision") is None

    snapshot = graph.get_state(config)
    assert len(snapshot.next) == 0  # reached END, no pause


# ---------------------------------------------------------------------------
# HIGH RISK
# ---------------------------------------------------------------------------
def test_high_risk_request_pauses_for_approval(monkeypatch):
    _mock_llm(
        monkeypatch,
        [RiskAssessment(score=85, level="HIGH", reason="Large spend", risk_factors=["amount"])],
    )

    graph = _new_graph()
    config = {"configurable": {"thread_id": "high-risk-pause-thread"}}
    state = new_state("Alex", "Engineering", "Buy production servers", 50000, "Capacity expansion")

    result = graph.invoke(state, config=config)

    assert result["risk_level"] == "HIGH"
    assert result["approval_required"] is True
    assert result.get("execution_status") in (None, "not_started")

    snapshot = graph.get_state(config)
    assert snapshot.interrupts, "expected the graph to pause for human approval"


def test_high_risk_request_approve_resumes_and_executes(monkeypatch):
    _mock_llm(
        monkeypatch,
        [RiskAssessment(score=85, level="HIGH", reason="Large spend", risk_factors=["amount"])],
    )

    graph = _new_graph()
    config = {"configurable": {"thread_id": "high-risk-approve-thread"}}
    state = new_state("Alex", "Engineering", "Buy production servers", 50000, "Capacity expansion")

    graph.invoke(state, config=config)  # pauses
    final_state = graph.invoke(Command(resume="approve"), config=config)

    assert final_state["human_decision"] == "approve"
    assert final_state["execution_status"] == "executed"


def test_high_risk_request_reject_resumes_and_rejects(monkeypatch):
    _mock_llm(
        monkeypatch,
        [RiskAssessment(score=90, level="HIGH", reason="Large spend", risk_factors=["amount"])],
    )

    graph = _new_graph()
    config = {"configurable": {"thread_id": "high-risk-reject-thread"}}
    state = new_state("Alex", "Engineering", "Buy production servers", 75000, "Capacity expansion")

    graph.invoke(state, config=config)  # pauses
    final_state = graph.invoke(Command(resume="reject"), config=config)

    assert final_state["human_decision"] == "reject"
    assert final_state["execution_status"] == "rejected"
    assert final_state["approval_status"] == "rejected"


# ---------------------------------------------------------------------------
# INVALID INPUT NEVER REACHES THE LLM
# ---------------------------------------------------------------------------
def test_invalid_request_never_calls_the_model(monkeypatch):
    calls = []

    class ExplodingLLM:
        def invoke(self, prompt):
            calls.append(prompt)
            raise AssertionError("the LLM must never be called for invalid input")

    monkeypatch.setattr(nodes, "_get_structured_llm", lambda: ExplodingLLM())

    graph = _new_graph()
    config = {"configurable": {"thread_id": "invalid-request-thread"}}
    state = new_state("", "Engineering", "Buy production servers", 50000, "Capacity expansion")

    result = graph.invoke(state, config=config)

    assert calls == []
    assert result["execution_status"] == "failed"
    assert any("requester" in e for e in result["errors"])


# ---------------------------------------------------------------------------
# RETRY BEHAVIOR
# ---------------------------------------------------------------------------
def test_transient_llm_failure_is_retried_then_succeeds(monkeypatch):
    _mock_llm(
        monkeypatch,
        [
            TransientLLMError("simulated transient provider timeout"),
            RiskAssessment(score=10, level="LOW", reason="Small purchase", risk_factors=[]),
        ],
    )

    graph = _new_graph()
    config = {"configurable": {"thread_id": "retry-thread"}}
    state = new_state("Alex", "Engineering", "Buy a keyboard", 40, "Broken keyboard")

    result = graph.invoke(state, config=config)

    # Despite the first (simulated) call failing transiently, the retry
    # policy on the analyze node retries and the workflow completes.
    assert result["risk_level"] == "LOW"
    assert result["execution_status"] == "executed"
