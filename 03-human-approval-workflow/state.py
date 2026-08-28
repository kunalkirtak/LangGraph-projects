"""
Graph state definition for the Enterprise Human Approval Workflow.

The state is the single source of truth that flows through every node.
It represents the complete lifecycle of a request:

    REQUEST -> ANALYSIS -> RISK -> APPROVAL DECISION -> HUMAN DECISION -> EXECUTION

Nothing about routing, execution, or approval lives outside this state;
the graph reads and writes it at every step, which is what makes the
workflow inspectable, auditable, and resumable.
"""

from __future__ import annotations

from typing import Literal, TypedDict


class ApprovalState(TypedDict, total=False):
    # --- Request identity -------------------------------------------------
    request_id: str
    requester: str
    department: str

    # --- Request payload ----------------------------------------------------
    action: str
    amount: float
    reason: str

    # --- Risk assessment (produced by analyze_request) ----------------------
    risk_score: int
    risk_level: Literal["LOW", "HIGH"]
    risk_reason: str
    risk_factors: list[str]

    # --- Policy decision (produced deterministically from risk_score) -------
    approval_required: bool
    approval_status: Literal["not_required", "pending", "approved", "rejected"]

    # --- Human decision (produced by human_approval / interrupt) ------------
    human_decision: str

    # --- Execution outcome ----------------------------------------------------
    execution_status: Literal["not_started", "executed", "rejected", "failed"]
    execution_result: str

    # --- Governance / audit ---------------------------------------------------
    errors: list[str]
    metadata: dict


def new_state(
    requester: str,
    department: str,
    action: str,
    amount: float,
    reason: str,
) -> ApprovalState:
    """Build a fresh, minimally-populated state for a new request.

    Everything else (request_id, risk fields, approval fields, execution
    fields) is filled in by the graph nodes as the workflow progresses.
    """
    return ApprovalState(
        requester=requester,
        department=department,
        action=action,
        amount=amount,
        reason=reason,
        errors=[],
        metadata={},
        approval_status="not_required",
        execution_status="not_started",
    )
