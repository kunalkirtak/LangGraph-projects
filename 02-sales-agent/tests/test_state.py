from __future__ import annotations

from state import SalesState


def test_state_can_be_created_empty():
    state: SalesState = {}
    assert state == {}


def test_state_accepts_all_documented_fields():
    state: SalesState = {
        "lead_id": "lead_123",
        "lead_name": "Sarah",
        "company": "Acme",
        "role": "CTO",
        "industry": "SaaS",
        "company_size": 500,
        "need": "Automation",
        "budget": "$75000",
        "urgency": "High",
        "qualification_score": 82,
        "qualification_status": "qualified",
        "qualification_reason": "Strong fit",
        "qualification_strengths": ["Clear need"],
        "qualification_concerns": [],
        "research": "brief",
        "outreach_message": "hello",
        "nurture_message": "",
        "next_action": "sales_outreach",
        "errors": [],
        "metadata": {"foo": "bar"},
    }
    assert state["lead_name"] == "Sarah"
    assert state["qualification_status"] == "qualified"


def test_state_is_a_plain_dict_at_runtime():
    # TypedDict instances are ordinary dicts at runtime -- required fields
    # are a static-typing concept only, not enforced by Python itself.
    state: SalesState = {"lead_name": "Only one field"}
    assert isinstance(state, dict)
    assert "company" not in state
