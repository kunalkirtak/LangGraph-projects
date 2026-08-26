"""
nodes.py

The actual LangGraph node functions for the sales lead agent. Each node:

1. Reads only the state it needs.
2. Performs one responsibility.
3. Returns a partial state update (never mutates state in place).
4. Handles failures by recording them in ``errors`` rather than raising.
5. Logs its execution.

The LLM objects are built lazily via small factory functions
(``get_qualification_llm``, ``get_research_llm``, ``get_content_llm``) rather
than at import time. This is what makes the nodes testable: unit tests
monkeypatch these factories to return a deterministic fake model instead of
calling Gemini.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, field_validator

import prompts
from config import settings
from utils import TransientLLMError, call_with_retry, generate_lead_id, logger
from validators import validate_lead


# ---------------------------------------------------------------------------
# Structured qualification schema
# ---------------------------------------------------------------------------


class QualificationResult(BaseModel):
    """Structured output produced by the qualification LLM call."""

    score: int = Field(ge=0, le=100, description="Overall lead fit score, 0-100.")
    status: Literal["qualified", "unqualified"] = Field(
        description="Categorical qualification outcome."
    )
    reason: str = Field(min_length=1, description="Justification for the score/status.")
    strengths: list[str] = Field(default_factory=list)
    concerns: list[str] = Field(default_factory=list)

    @field_validator("reason")
    @classmethod
    def _reason_not_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("reason must not be blank")
        return value


# ---------------------------------------------------------------------------
# LLM factories (overridden in tests)
# ---------------------------------------------------------------------------


def _base_chat_model():
    """Build the underlying Gemini chat model.

    Imported lazily so that the ``langchain_google_genai`` dependency is
    only required when a node actually needs to call the model (tests never
    hit this function because they monkeypatch the factories below).
    """

    from langchain_google_genai import ChatGoogleGenerativeAI

    if not settings.google_api_key:
        raise TransientLLMError(
            "GOOGLE_API_KEY is not configured; cannot call the Gemini model."
        )

    return ChatGoogleGenerativeAI(
        model=settings.gemini_model,
        google_api_key=settings.google_api_key,
        temperature=0.3,
    )


def get_qualification_llm():
    """Return a model configured to emit :class:`QualificationResult`."""

    return _base_chat_model().with_structured_output(QualificationResult)


def get_research_llm():
    """Return a plain chat model used for the research brief."""

    return _base_chat_model()


def get_content_llm():
    """Return a plain chat model used for outreach/nurture copy."""

    return _base_chat_model()


# ---------------------------------------------------------------------------
# Node: normalize_lead
# ---------------------------------------------------------------------------


def normalize_lead(state: dict) -> dict:
    """Clean and validate the raw lead before it enters the rest of the graph."""

    logger.info("Lead normalization started")

    validation = validate_lead(state)
    if not validation.is_valid:
        logger.info("Lead normalization failed validation: %s", validation.errors)
        return {
            "errors": list(state.get("errors", [])) + validation.errors,
            "next_action": "human_review",
            "metadata": {**state.get("metadata", {}), "validation_failed": True},
        }

    lead_id = state.get("lead_id") or generate_lead_id()

    def _clean_str(value: object) -> str:
        return value.strip() if isinstance(value, str) else ""

    company_size_raw = state.get("company_size")
    company_size: int | None
    if company_size_raw in (None, ""):
        company_size = None
    else:
        try:
            company_size = int(company_size_raw)
        except (TypeError, ValueError):
            company_size = None

    budget_raw = _clean_str(state.get("budget", "")) or "Not specified"
    urgency = _clean_str(state.get("urgency", "")) or "Not specified"
    industry = _clean_str(state.get("industry", "")) or "Not specified"

    normalized = {
        "lead_id": lead_id,
        "lead_name": _clean_str(state.get("lead_name", "")),
        "company": _clean_str(state.get("company", "")),
        "role": _clean_str(state.get("role", "")),
        "industry": industry,
        "need": _clean_str(state.get("need", "")),
        "budget": budget_raw,
        "urgency": urgency,
        "errors": list(state.get("errors", [])),
        "metadata": {**state.get("metadata", {}), "normalized": True},
    }
    if company_size is not None:
        normalized["company_size"] = company_size

    logger.info("Lead normalization completed | lead_id=%s", lead_id)
    return normalized


# ---------------------------------------------------------------------------
# Node: qualify_lead
# ---------------------------------------------------------------------------


def qualify_lead(state: dict) -> dict:
    """Score and classify the lead using the qualification LLM.

    LangGraph -- not the LLM -- makes the final routing decision. This node
    only produces the structured qualification result and stores it in
    state; ``route_lead`` (in graph.py) reads that state to pick a path.
    """

    lead_id = state.get("lead_id", "unknown")
    logger.info("Qualification started | lead_id=%s", lead_id)

    prompt_kwargs = {
        "lead_name": state.get("lead_name", ""),
        "company": state.get("company", ""),
        "role": state.get("role", ""),
        "industry": state.get("industry", "Not specified"),
        "company_size": state.get("company_size", "Not specified"),
        "need": state.get("need", ""),
        "budget": state.get("budget", "Not specified"),
        "urgency": state.get("urgency", "Not specified"),
    }

    def _invoke() -> QualificationResult:
        llm = get_qualification_llm()
        messages = [
            ("system", prompts.QUALIFICATION_SYSTEM_PROMPT),
            ("user", prompts.QUALIFICATION_USER_PROMPT.format(**prompt_kwargs)),
        ]
        try:
            result = llm.invoke(messages)
        except TransientLLMError:
            raise
        except Exception as exc:  # noqa: BLE001 - genuinely unknown provider errors
            raise TransientLLMError(str(exc)) from exc

        if isinstance(result, QualificationResult):
            return result
        # Fallback: some providers return a dict-like structured output.
        try:
            return QualificationResult.model_validate(result)
        except Exception as exc:  # noqa: BLE001
            raise TransientLLMError(f"Malformed structured output: {exc}") from exc

    try:
        qualification = call_with_retry(_invoke, node_name="qualify_lead")
    except TransientLLMError as exc:
        logger.info("Qualification failed after retries | lead_id=%s | %s", lead_id, exc)
        return {
            "errors": list(state.get("errors", [])) + [f"Qualification node failed: {exc}"],
            "next_action": "human_review",
        }

    # Enforce the configurable threshold as the authoritative routing signal,
    # even if the model's own "status" field disagrees with its own score.
    status: Literal["qualified", "unqualified"] = (
        "qualified" if qualification.score >= settings.qualification_threshold else "unqualified"
    )

    logger.info(
        "Qualification score: %d | status=%s | lead_id=%s",
        qualification.score,
        status,
        lead_id,
    )

    return {
        "qualification_score": qualification.score,
        "qualification_status": status,
        "qualification_reason": qualification.reason,
        "qualification_strengths": qualification.strengths,
        "qualification_concerns": qualification.concerns,
    }


# ---------------------------------------------------------------------------
# Node: research_lead (qualified path)
# ---------------------------------------------------------------------------


def research_lead(state: dict) -> dict:
    """Produce an LLM-based lead research synthesis.

    This is explicitly an LLM-based synthesis from the information already
    on the lead -- not live web research. No external research tool is
    wired up in this project.
    """

    lead_id = state.get("lead_id", "unknown")
    logger.info("Lead research started | lead_id=%s", lead_id)

    prompt_kwargs = {
        "lead_name": state.get("lead_name", ""),
        "company": state.get("company", ""),
        "role": state.get("role", ""),
        "industry": state.get("industry", "Not specified"),
        "company_size": state.get("company_size", "Not specified"),
        "need": state.get("need", ""),
        "budget": state.get("budget", "Not specified"),
        "urgency": state.get("urgency", "Not specified"),
        "qualification_reason": state.get("qualification_reason", ""),
    }

    def _invoke() -> str:
        llm = get_research_llm()
        messages = [
            ("system", prompts.LEAD_RESEARCH_SYSTEM_PROMPT),
            ("user", prompts.LEAD_RESEARCH_USER_PROMPT.format(**prompt_kwargs)),
        ]
        try:
            response = llm.invoke(messages)
        except Exception as exc:  # noqa: BLE001
            raise TransientLLMError(str(exc)) from exc
        content = getattr(response, "content", response)
        if not isinstance(content, str) or not content.strip():
            raise TransientLLMError("Research LLM returned empty content.")
        return content.strip()

    try:
        research_text = call_with_retry(_invoke, node_name="research_lead")
    except TransientLLMError as exc:
        logger.info("Lead research failed after retries | lead_id=%s | %s", lead_id, exc)
        return {
            "errors": list(state.get("errors", [])) + [f"Research node failed: {exc}"],
            "research": "",
        }

    logger.info("Lead research completed | lead_id=%s", lead_id)
    return {"research": f"[LLM-based lead research synthesis]\n{research_text}"}


# ---------------------------------------------------------------------------
# Node: generate_outreach (qualified path)
# ---------------------------------------------------------------------------


def generate_outreach(state: dict) -> dict:
    """Generate a personalized outreach message for a qualified lead."""

    lead_id = state.get("lead_id", "unknown")
    logger.info("Outreach generation started | lead_id=%s", lead_id)

    prompt_kwargs = {
        "lead_name": state.get("lead_name", ""),
        "company": state.get("company", ""),
        "role": state.get("role", ""),
        "need": state.get("need", ""),
        "qualification_reason": state.get("qualification_reason", ""),
        "research": state.get("research", ""),
    }

    def _invoke() -> str:
        llm = get_content_llm()
        messages = [
            ("system", prompts.OUTREACH_SYSTEM_PROMPT),
            ("user", prompts.OUTREACH_USER_PROMPT.format(**prompt_kwargs)),
        ]
        try:
            response = llm.invoke(messages)
        except Exception as exc:  # noqa: BLE001
            raise TransientLLMError(str(exc)) from exc
        content = getattr(response, "content", response)
        if not isinstance(content, str) or not content.strip():
            raise TransientLLMError("Outreach LLM returned empty content.")
        return content.strip()

    try:
        message = call_with_retry(_invoke, node_name="generate_outreach")
    except TransientLLMError as exc:
        logger.info("Outreach generation failed after retries | lead_id=%s | %s", lead_id, exc)
        return {
            "errors": list(state.get("errors", [])) + [f"Outreach node failed: {exc}"],
            "outreach_message": "",
            "next_action": "human_review",
        }

    logger.info("Outreach generation completed | lead_id=%s", lead_id)
    return {"outreach_message": message, "next_action": "sales_outreach"}


# ---------------------------------------------------------------------------
# Node: generate_nurture (unqualified path)
# ---------------------------------------------------------------------------


def generate_nurture(state: dict) -> dict:
    """Generate a relationship-preserving nurture message for a weaker-fit lead."""

    lead_id = state.get("lead_id", "unknown")
    logger.info("Nurture generation started | lead_id=%s", lead_id)

    prompt_kwargs = {
        "lead_name": state.get("lead_name", ""),
        "company": state.get("company", ""),
        "role": state.get("role", ""),
        "need": state.get("need", ""),
        "qualification_reason": state.get("qualification_reason", ""),
        "concerns": "; ".join(state.get("qualification_concerns", [])) or "Not specified",
    }

    def _invoke() -> str:
        llm = get_content_llm()
        messages = [
            ("system", prompts.NURTURE_SYSTEM_PROMPT),
            ("user", prompts.NURTURE_USER_PROMPT.format(**prompt_kwargs)),
        ]
        try:
            response = llm.invoke(messages)
        except Exception as exc:  # noqa: BLE001
            raise TransientLLMError(str(exc)) from exc
        content = getattr(response, "content", response)
        if not isinstance(content, str) or not content.strip():
            raise TransientLLMError("Nurture LLM returned empty content.")
        return content.strip()

    try:
        message = call_with_retry(_invoke, node_name="generate_nurture")
    except TransientLLMError as exc:
        logger.info("Nurture generation failed after retries | lead_id=%s | %s", lead_id, exc)
        return {
            "errors": list(state.get("errors", [])) + [f"Nurture node failed: {exc}"],
            "nurture_message": "",
            "next_action": "human_review",
        }

    logger.info("Nurture generation completed | lead_id=%s", lead_id)
    return {"nurture_message": message, "next_action": "nurture"}
