from __future__ import annotations

from tests.conftest import FakeContentLLM, FakeQualificationLLM, RaisingLLM

import nodes
from graph import build_graph
from utils import TransientLLMError

QUALIFIED_LEAD = {
    "lead_name": "Sarah Chen",
    "company": "Acme Technologies",
    "role": "CTO",
    "industry": "SaaS",
    "company_size": 500,
    "need": "Automating internal AI workflows",
    "budget": "$75000",
    "urgency": "High",
}

UNQUALIFIED_LEAD = {
    "lead_name": "Jordan Miles",
    "company": "Miles Family Bakery",
    "role": "Marketing Intern",
    "industry": "Food & Beverage",
    "company_size": 6,
    "need": "Curious about AI for social captions",
    "budget": "unsure",
    "urgency": "none",
}

INVALID_LEAD = {
    "lead_name": "No Company Here",
    "role": "Owner",
    "need": "Something",
}


def test_graph_compiles():
    graph = build_graph()
    assert graph is not None


def test_qualified_path_reaches_research_and_outreach(monkeypatch):
    monkeypatch.setattr(nodes, "get_qualification_llm", lambda: FakeQualificationLLM(85, "qualified"))
    monkeypatch.setattr(nodes, "get_research_llm", lambda: FakeContentLLM("Research brief text."))
    monkeypatch.setattr(nodes, "get_content_llm", lambda: FakeContentLLM("Outreach email text."))

    graph = build_graph()
    final_state = graph.invoke(QUALIFIED_LEAD)

    assert final_state["qualification_status"] == "qualified"
    assert final_state["qualification_score"] == 85
    assert "Research brief text." in final_state["research"]
    assert final_state["outreach_message"] == "Outreach email text."
    assert final_state["next_action"] == "sales_outreach"
    assert "nurture_message" not in final_state or final_state["nurture_message"] == ""


def test_unqualified_path_reaches_nurture(monkeypatch):
    monkeypatch.setattr(nodes, "get_qualification_llm", lambda: FakeQualificationLLM(25, "unqualified"))
    monkeypatch.setattr(nodes, "get_content_llm", lambda: FakeContentLLM("Nurture email text."))

    graph = build_graph()
    final_state = graph.invoke(UNQUALIFIED_LEAD)

    assert final_state["qualification_status"] == "unqualified"
    assert final_state["qualification_score"] == 25
    assert final_state["nurture_message"] == "Nurture email text."
    assert final_state["next_action"] == "nurture"
    assert "outreach_message" not in final_state or final_state["outreach_message"] == ""
    assert "research" not in final_state or final_state["research"] == ""


def test_invalid_lead_never_reaches_qualification(monkeypatch):
    calls = {"count": 0}

    class ExplodingQualificationLLM:
        def invoke(self, messages):
            calls["count"] += 1
            raise AssertionError("Qualification LLM should never be called for invalid input")

    monkeypatch.setattr(nodes, "get_qualification_llm", lambda: ExplodingQualificationLLM())

    graph = build_graph()
    final_state = graph.invoke(INVALID_LEAD)

    assert calls["count"] == 0
    assert final_state["errors"]
    assert "qualification_status" not in final_state


def test_qualification_transient_failure_still_routes_and_records_error(monkeypatch):
    # A qualification failure has no status in state, so route_lead's safe
    # default sends the lead to nurture rather than dropping it entirely --
    # but the failure is still recorded in `errors` for observability.
    monkeypatch.setattr(
        nodes,
        "get_qualification_llm",
        lambda: RaisingLLM(TransientLLMError("simulated provider timeout")),
    )
    monkeypatch.setattr(nodes, "get_content_llm", lambda: FakeContentLLM("Nurture fallback."))

    graph = build_graph()
    final_state = graph.invoke(QUALIFIED_LEAD)

    assert "qualification_status" not in final_state
    assert any("Qualification node failed" in e for e in final_state["errors"])
    assert final_state["next_action"] == "nurture"


def test_qualification_and_nurture_both_failing_surfaces_human_review(monkeypatch):
    monkeypatch.setattr(
        nodes,
        "get_qualification_llm",
        lambda: RaisingLLM(TransientLLMError("simulated provider timeout")),
    )
    monkeypatch.setattr(
        nodes,
        "get_content_llm",
        lambda: RaisingLLM(TransientLLMError("simulated provider timeout")),
    )

    graph = build_graph()
    final_state = graph.invoke(QUALIFIED_LEAD)

    assert final_state["next_action"] == "human_review"
    assert len(final_state["errors"]) >= 2


def test_mocked_full_workflow_completes_for_both_branches(monkeypatch):
    monkeypatch.setattr(nodes, "get_research_llm", lambda: FakeContentLLM("r"))
    monkeypatch.setattr(nodes, "get_content_llm", lambda: FakeContentLLM("m"))
    graph = build_graph()

    monkeypatch.setattr(nodes, "get_qualification_llm", lambda: FakeQualificationLLM(90, "qualified"))
    qualified_result = graph.invoke(QUALIFIED_LEAD)
    assert qualified_result["next_action"] == "sales_outreach"

    monkeypatch.setattr(nodes, "get_qualification_llm", lambda: FakeQualificationLLM(10, "unqualified"))
    unqualified_result = graph.invoke(UNQUALIFIED_LEAD)
    assert unqualified_result["next_action"] == "nurture"
