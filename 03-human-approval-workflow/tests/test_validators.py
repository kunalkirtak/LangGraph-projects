import pytest

from validators import (
    validate_fields,
    validate_human_decision,
    normalize_decision,
)


def test_valid_request_has_no_errors():
    errors = validate_fields(
        requester="Alex",
        department="Engineering",
        action="Buy a laptop",
        amount=1200,
        reason="Replacement hardware",
    )
    assert errors == []


def test_missing_requester_is_rejected():
    errors = validate_fields(
        requester="",
        department="Engineering",
        action="Buy a laptop",
        amount=1200,
        reason="Replacement hardware",
    )
    assert any("requester" in e for e in errors)


def test_missing_action_is_rejected():
    errors = validate_fields(
        requester="Alex",
        department="Engineering",
        action="   ",
        amount=1200,
        reason="Replacement hardware",
    )
    assert any("action" in e for e in errors)


def test_negative_amount_is_rejected():
    errors = validate_fields(
        requester="Alex",
        department="Engineering",
        action="Buy a laptop",
        amount=-50,
        reason="Replacement hardware",
    )
    assert any("amount" in e for e in errors)


def test_non_numeric_amount_is_rejected():
    errors = validate_fields(
        requester="Alex",
        department="Engineering",
        action="Buy a laptop",
        amount="a lot of money",
        reason="Replacement hardware",
    )
    assert any("amount" in e for e in errors)


def test_missing_reason_is_rejected():
    errors = validate_fields(
        requester="Alex",
        department="Engineering",
        action="Buy a laptop",
        amount=1200,
        reason="",
    )
    assert any("reason" in e for e in errors)


@pytest.mark.parametrize("decision", ["approve", "APPROVE", " Reject ", "reject"])
def test_valid_decisions_are_accepted(decision):
    assert validate_human_decision(decision) is True


@pytest.mark.parametrize("decision", ["yes", "maybe", "", None, "approve please"])
def test_invalid_decisions_are_rejected(decision):
    assert validate_human_decision(decision) is False


def test_normalize_decision_lowercases_and_strips():
    assert normalize_decision(" Approve ") == "approve"
