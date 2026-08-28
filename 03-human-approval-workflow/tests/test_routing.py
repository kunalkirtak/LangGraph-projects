from nodes import route_approval, route_risk, route_validation


def test_route_validation_sends_invalid_requests_to_execute():
    state = {"errors": ["requester must not be empty"]}
    assert route_validation(state) == "invalid"


def test_route_validation_sends_valid_requests_to_analyze():
    state = {"errors": []}
    assert route_validation(state) == "analyze"


def test_route_risk_low_score_goes_to_execute():
    state = {"risk_score": 20, "risk_level": "LOW", "approval_required": False, "errors": []}
    assert route_risk(state) == "execute"


def test_route_risk_high_score_goes_to_approval():
    state = {"risk_score": 75, "risk_level": "HIGH", "approval_required": True, "errors": []}
    assert route_risk(state) == "approval"


def test_route_risk_with_prior_errors_goes_to_execute():
    # execute_request is responsible for turning this into a failed outcome.
    state = {"errors": ["invalid risk assessment from model: bad output"], "approval_required": True}
    assert route_risk(state) == "execute"


def test_route_approval_approve_goes_to_execute():
    state = {"human_decision": "approve"}
    assert route_approval(state) == "execute"


def test_route_approval_reject_goes_to_reject():
    state = {"human_decision": "reject"}
    assert route_approval(state) == "reject"
