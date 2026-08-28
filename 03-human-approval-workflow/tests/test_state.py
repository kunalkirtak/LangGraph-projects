from state import new_state


def test_new_state_contains_required_fields():
    state = new_state(
        requester="Alex",
        department="Engineering",
        action="Buy a laptop",
        amount=1200,
        reason="Replacement hardware",
    )

    assert state["requester"] == "Alex"
    assert state["department"] == "Engineering"
    assert state["action"] == "Buy a laptop"
    assert state["amount"] == 1200
    assert state["reason"] == "Replacement hardware"


def test_new_state_initializes_governance_fields():
    state = new_state("Alex", "Engineering", "Buy a laptop", 1200, "Replacement hardware")

    assert state["errors"] == []
    assert state["metadata"] == {}
    assert state["approval_status"] == "not_required"
    assert state["execution_status"] == "not_started"


def test_new_state_does_not_prematurely_set_risk_fields():
    state = new_state("Alex", "Engineering", "Buy a laptop", 1200, "Replacement hardware")

    assert "risk_score" not in state
    assert "risk_level" not in state
    assert "human_decision" not in state
