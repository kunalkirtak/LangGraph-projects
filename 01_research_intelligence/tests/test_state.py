from state import initial_state


def test_initial_state_has_all_fields():
    state = initial_state("LLM inference optimization")

    assert state["topic"] == "LLM inference optimization"
    assert state["research"] == ""
    assert state["analysis"] == ""
    assert state["report"] == ""
    assert state["status"] == "started"
    assert state["errors"] == []
    assert state["metadata"] == {}


def test_initial_state_is_independent_between_calls():
    state_a = initial_state("Topic A")
    state_b = initial_state("Topic B")

    state_a["errors"].append("boom")

    # Mutating one call's state must not leak into a fresh call.
    assert state_b["errors"] == []
    assert state_a["topic"] != state_b["topic"]


def test_state_supports_partial_updates_like_a_node_would_return():
    state = initial_state("Topic")
    update = {"research": "some findings", "status": "research_complete"}

    state.update(update)

    assert state["research"] == "some findings"
    assert state["status"] == "research_complete"
    # Fields the "node" did not touch remain unchanged.
    assert state["analysis"] == ""
