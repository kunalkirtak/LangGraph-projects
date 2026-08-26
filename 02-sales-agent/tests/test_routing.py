from __future__ import annotations

from graph import route_after_normalize, route_lead


# ---------------------------------------------------------------------------
# route_lead: score >= threshold -> qualified, score < threshold -> unqualified
# ---------------------------------------------------------------------------


def test_route_lead_returns_qualified_when_status_is_qualified():
    state = {"qualification_status": "qualified", "qualification_score": 82}
    assert route_lead(state) == "qualified"


def test_route_lead_returns_unqualified_when_status_is_unqualified():
    state = {"qualification_status": "unqualified", "qualification_score": 34}
    assert route_lead(state) == "unqualified"


def test_route_lead_defaults_to_unqualified_when_status_missing():
    # A lead with no qualification_status (e.g. qualification node failed)
    # must never silently fall through to the qualified path.
    state = {}
    assert route_lead(state) == "unqualified"


def test_route_lead_treats_qualification_failure_as_unqualified():
    state = {"errors": ["Qualification node failed: timeout"]}
    assert route_lead(state) == "unqualified"


def test_route_lead_boundary_score_high_end():
    # Boundary case: exactly qualified.
    state = {"qualification_status": "qualified", "qualification_score": 60}
    assert route_lead(state) == "qualified"


def test_route_lead_boundary_score_low_end():
    state = {"qualification_status": "unqualified", "qualification_score": 59}
    assert route_lead(state) == "unqualified"


# ---------------------------------------------------------------------------
# route_after_normalize: invalid input never reaches qualification
# ---------------------------------------------------------------------------


def test_route_after_normalize_valid_input():
    state = {"lead_name": "Sarah", "errors": []}
    assert route_after_normalize(state) == "valid"


def test_route_after_normalize_invalid_input():
    state = {"errors": ["Missing required field: 'company'."]}
    assert route_after_normalize(state) == "invalid"


def test_route_after_normalize_no_errors_key_defaults_valid():
    state = {"lead_name": "Sarah"}
    assert route_after_normalize(state) == "valid"
