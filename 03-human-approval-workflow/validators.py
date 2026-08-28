"""
Input validation for incoming approval requests.

This module never calls the LLM. Validation is pure, deterministic
Python — if a request is malformed, we reject it before spending a
single token on it.
"""

from __future__ import annotations

from state import ApprovalState
from utils import generate_request_id


def validate_fields(
    requester: str,
    department: str,
    action: str,
    amount: float,
    reason: str,
) -> list[str]:
    """Return a list of human-readable validation errors (empty if valid)."""
    errors: list[str] = []

    if not requester or not str(requester).strip():
        errors.append("requester must not be empty")

    if not department or not str(department).strip():
        errors.append("department must not be empty")

    if not action or not str(action).strip():
        errors.append("action must not be empty")

    if not reason or not str(reason).strip():
        errors.append("reason must not be empty")

    try:
        numeric_amount = float(amount)
        if numeric_amount < 0:
            errors.append("amount must be non-negative")
    except (TypeError, ValueError):
        errors.append("amount must be numeric")

    return errors


VALID_DECISIONS = {"approve", "reject"}


def validate_human_decision(decision: str) -> bool:
    """Return True only for exactly 'approve' or 'reject' (case-insensitive)."""
    if not isinstance(decision, str):
        return False
    return decision.strip().lower() in VALID_DECISIONS


def normalize_decision(decision: str) -> str:
    """Normalize a validated decision string to lowercase canonical form."""
    return decision.strip().lower()


def ensure_request_id(state: ApprovalState) -> str:
    """Return the existing request_id, or generate a new one."""
    return state.get("request_id") or generate_request_id()
