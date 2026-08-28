"""
Node implementations for the Enterprise Human Approval Workflow.

Five nodes make up the graph:

    validate_request  -> pure input validation, no LLM call
    analyze_request    -> LLM risk assessment + deterministic policy decision
    human_approval      -> interrupt() and wait for a human decision
    execute_request     -> simulated execution of an approved/low-risk request
    reject_request       -> records a rejection

Design principle (see README "Policy vs LLM"): the LLM only produces a
*risk assessment*. Whether that assessment requires human approval is
decided by a deterministic threshold in `decide_policy`, never by the
model itself.
"""

from __future__ import annotations

from typing import Literal

from langgraph.types import interrupt
from pydantic import BaseModel, Field, ValidationError

from config import load_settings
from state import ApprovalState
from utils import generate_request_id, logger, utc_now_iso
from validators import (
    normalize_decision,
    validate_fields,
    validate_human_decision,
)
from prompts import RISK_ASSESSMENT_PROMPT


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------
class TransientLLMError(Exception):
    """Raised for retryable failures (timeouts, rate limits, connection drops).

    Only this exception type should trigger LangGraph's retry policy on the
    analyze_request node. Invalid input or malformed model output must NOT
    be retried — retrying a bad prompt or bad input forever just wastes
    calls and hides real bugs.
    """


# ---------------------------------------------------------------------------
# Structured LLM output
# ---------------------------------------------------------------------------
class RiskAssessment(BaseModel):
    score: int = Field(ge=0, le=100)
    level: Literal["LOW", "HIGH"]
    reason: str
    risk_factors: list[str] = Field(default_factory=list)


_llm = None  # lazily constructed so importing this module never requires an API key


def _get_structured_llm():
    global _llm
    if _llm is None:
        from langchain_google_genai import ChatGoogleGenerativeAI

        settings = load_settings()
        base = ChatGoogleGenerativeAI(
            model=settings.gemini_model,
            google_api_key=settings.google_api_key,
            temperature=0,
        )
        _llm = base.with_structured_output(RiskAssessment)
    return _llm


def reset_llm_cache() -> None:
    """Used by tests to force a fresh (mockable) LLM instance."""
    global _llm
    _llm = None


def decide_policy(score: int, threshold: int) -> tuple[str, bool]:
    """Deterministic policy layer: turns a raw score into a level + approval flag.

    This is intentionally NOT part of the LLM call. The model assesses risk;
    this function — plain, testable Python — decides what the business does
    about it.
    """
    level = "HIGH" if score >= threshold else "LOW"
    approval_required = level == "HIGH"
    return level, approval_required


# ---------------------------------------------------------------------------
# Node: validate_request
# ---------------------------------------------------------------------------
def validate_request(state: ApprovalState) -> ApprovalState:
    logger.info(
        "Request validation started | request_id=%s", state.get("request_id", "pending")
    )

    errors = validate_fields(
        requester=state.get("requester", ""),
        department=state.get("department", ""),
        action=state.get("action", ""),
        amount=state.get("amount", None),
        reason=state.get("reason", ""),
    )

    request_id = state.get("request_id") or generate_request_id()
    metadata = dict(state.get("metadata", {}))
    metadata.setdefault("created_at", utc_now_iso())
    metadata["updated_at"] = utc_now_iso()
    metadata["workflow_status"] = "validated" if not errors else "validation_failed"

    if errors:
        logger.info("Request validation failed | request_id=%s | errors=%s", request_id, errors)

    return {
        **state,
        "request_id": request_id,
        "errors": errors,
        "metadata": metadata,
    }


# ---------------------------------------------------------------------------
# Node: analyze_request
# ---------------------------------------------------------------------------
def analyze_request(state: ApprovalState) -> ApprovalState:
    request_id = state["request_id"]
    logger.info("Risk analysis started | request_id=%s", request_id)

    settings = load_settings()
    prompt = RISK_ASSESSMENT_PROMPT.format(
        requester=state.get("requester", ""),
        department=state.get("department", ""),
        action=state.get("action", ""),
        amount=state.get("amount", ""),
        reason=state.get("reason", ""),
    )

    try:
        raw_result = _get_structured_llm().invoke(prompt)
    except (TimeoutError, ConnectionError) as exc:
        # Known-transient failure classes: let LangGraph's retry policy handle it.
        raise TransientLLMError(str(exc)) from exc
    except Exception as exc:  # noqa: BLE001 - re-raise as transient, see docstring
        # Any other unexpected error from the model call is treated as
        # transient (network hiccup, provider hiccup) rather than crashing
        # the workflow outright. Malformed *output* is handled below, not here.
        raise TransientLLMError(str(exc)) from exc

    try:
        assessment = (
            raw_result
            if isinstance(raw_result, RiskAssessment)
            else RiskAssessment.model_validate(raw_result)
        )
    except ValidationError as exc:
        # The model returned something we don't trust. This is NOT retried
        # blindly - we record it as a hard error and let the workflow stop
        # rather than act on an assessment we can't validate.
        errors = list(state.get("errors", []))
        errors.append(f"invalid risk assessment from model: {exc}")
        logger.info("Risk assessment invalid | request_id=%s", request_id)
        return {**state, "errors": errors}

    level, approval_required = decide_policy(assessment.score, settings.high_risk_threshold)

    logger.info(
        "Risk assessment completed | request_id=%s | risk_score=%s | risk_level=%s",
        request_id,
        assessment.score,
        level,
    )

    metadata = dict(state.get("metadata", {}))
    metadata["updated_at"] = utc_now_iso()
    metadata["workflow_status"] = "risk_assessed"

    return {
        **state,
        "risk_score": assessment.score,
        "risk_level": level,
        "risk_reason": assessment.reason,
        "risk_factors": assessment.risk_factors,
        "approval_required": approval_required,
        "approval_status": "pending" if approval_required else "not_required",
        "metadata": metadata,
    }


# ---------------------------------------------------------------------------
# Routers
# ---------------------------------------------------------------------------
def route_validation(state: ApprovalState) -> str:
    """Skip the LLM entirely when the request itself failed validation.

    Per the project requirements, invalid input must never reach the model.
    Invalid requests are routed straight to execute_request, which detects
    the recorded errors and safely reports execution_status="failed"
    without attempting anything.
    """
    return "invalid" if state.get("errors") else "analyze"


def route_risk(state: ApprovalState) -> str:
    """Decide, purely from state, whether to auto-execute or ask a human."""
    if state.get("errors"):
        return "execute"  # execute_request will notice errors and short-circuit safely
    return "approval" if state.get("approval_required") else "execute"


def route_approval(state: ApprovalState) -> str:
    decision = state.get("human_decision", "")
    return "execute" if decision == "approve" else "reject"


# ---------------------------------------------------------------------------
# Node: human_approval
# ---------------------------------------------------------------------------
def human_approval(state: ApprovalState) -> ApprovalState:
    """Pause the graph and wait for a real external human decision.

    This is the core human-in-the-loop mechanism: `interrupt()` halts graph
    execution at this exact point. LangGraph persists the current state via
    the configured checkpointer. Execution only continues when the graph is
    invoked again with `Command(resume=<decision>)` against the SAME thread
    id. Nothing below this line runs until that happens.
    """
    request_id = state["request_id"]
    logger.info("Workflow interrupted for human approval | request_id=%s", request_id)

    payload = {
        "type": "approval_required",
        "request_id": request_id,
        "requester": state.get("requester"),
        "department": state.get("department"),
        "action": state.get("action"),
        "amount": state.get("amount"),
        "reason": state.get("reason"),
        "risk_score": state.get("risk_score"),
        "risk_level": state.get("risk_level"),
        "risk_reason": state.get("risk_reason"),
    }

    raw_decision = interrupt(payload)

    if not validate_human_decision(raw_decision):
        logger.info(
            "Invalid human decision received | request_id=%s | value=%r",
            request_id,
            raw_decision,
        )
        errors = list(state.get("errors", []))
        errors.append(f"invalid human decision: {raw_decision!r} (must be 'approve' or 'reject')")
        # Fail safe: an unrecognized decision is treated as a rejection so the
        # sensitive action is never executed on ambiguous human input.
        return {
            **state,
            "human_decision": "reject",
            "approval_status": "rejected",
            "errors": errors,
        }

    decision = normalize_decision(raw_decision)
    logger.info("Human decision received: %s | request_id=%s", decision, request_id)

    return {
        **state,
        "human_decision": decision,
        "approval_status": "approved" if decision == "approve" else "rejected",
    }


# ---------------------------------------------------------------------------
# Node: execute_request
# ---------------------------------------------------------------------------
def execute_request(state: ApprovalState) -> ApprovalState:
    request_id = state["request_id"]

    if state.get("errors"):
        logger.info("Execution skipped due to prior errors | request_id=%s", request_id)
        return {
            **state,
            "execution_status": "failed",
            "execution_result": (
                f"Request {request_id} could not be executed due to prior errors: "
                f"{'; '.join(state['errors'])}"
            ),
        }

    logger.info("Execution started | request_id=%s", request_id)

    # Simulated execution only. See README "Execution Node" and "Security
    # Design" - this project intentionally never touches a real financial,
    # production, or access-control system.
    result = f"Request {request_id} approved and execution simulated successfully."

    metadata = dict(state.get("metadata", {}))
    metadata["updated_at"] = utc_now_iso()
    metadata["workflow_status"] = "executed"

    logger.info("Execution completed | request_id=%s", request_id)

    return {
        **state,
        "approval_status": state.get("approval_status") or "not_required",
        "execution_status": "executed",
        "execution_result": result,
        "metadata": metadata,
    }


# ---------------------------------------------------------------------------
# Node: reject_request
# ---------------------------------------------------------------------------
def reject_request(state: ApprovalState) -> ApprovalState:
    request_id = state["request_id"]
    logger.info("Execution rejected by policy/human decision | request_id=%s", request_id)

    metadata = dict(state.get("metadata", {}))
    metadata["updated_at"] = utc_now_iso()
    metadata["workflow_status"] = "rejected"

    return {
        **state,
        "approval_status": "rejected",
        "execution_status": "rejected",
        "execution_result": f"Request {request_id} was rejected by the human reviewer.",
        "metadata": metadata,
    }
