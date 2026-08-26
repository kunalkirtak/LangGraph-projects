"""
state.py

Defines the shared graph state for the Intelligent Sales Lead Agent.

This file intentionally contains NO business logic. It only describes the
shape of the data that flows between LangGraph nodes.
"""

from __future__ import annotations

from typing import Any, Literal, TypedDict


class SalesState(TypedDict, total=False):
    """Shared state passed between every node in the sales lead graph.

    Fields are grouped by the stage of the workflow that populates them.
    All fields are optional (``total=False``) because the state is built up
    incrementally as it flows through the graph -- a freshly submitted lead
    will not yet have a ``qualification_score``, for example.
    """

    # --- Identity -----------------------------------------------------
    lead_id: str
    """Stable identifier for the lead. Generated during normalization if
    the caller did not supply one."""

    # --- Raw / normalized lead data ------------------------------------
    lead_name: str
    """The contact's name."""

    company: str
    """The company the lead works for."""

    role: str
    """The lead's job title / role (e.g. 'CTO', 'VP Engineering')."""

    industry: str
    """The industry the lead's company operates in. Optional."""

    company_size: int
    """Approximate employee count. Optional, normalized to an int."""

    need: str
    """Free-text description of the lead's business need/problem."""

    budget: str
    """Free-text budget figure as supplied by the lead (e.g. '$75000').
    Kept as a string because leads rarely give clean numeric input, but
    normalization will strip stray characters where possible."""

    urgency: str
    """Free-text urgency signal (e.g. 'High', 'Low', 'Q3 rollout')."""

    # --- Qualification --------------------------------------------------
    qualification_score: int
    """0-100 score produced by the qualification node."""

    qualification_status: Literal["qualified", "unqualified"]
    """Categorical qualification outcome derived from the score/LLM."""

    qualification_reason: str
    """Human-readable justification for the qualification outcome."""

    qualification_strengths: list[str]
    """Positive signals identified during qualification."""

    qualification_concerns: list[str]
    """Risk factors / concerns identified during qualification."""

    # --- Downstream content ---------------------------------------------
    research: str
    """LLM-generated sales research brief (qualified leads only)."""

    outreach_message: str
    """Personalized outreach message (qualified leads only)."""

    nurture_message: str
    """Relationship-preserving message (unqualified leads only)."""

    # --- Workflow control -------------------------------------------------
    next_action: Literal["sales_outreach", "nurture", "human_review"]
    """What the sales system should do next with this lead."""

    # --- Observability / error handling -----------------------------------
    errors: list[str]
    """Accumulated error messages. Never silently discarded."""

    metadata: dict[str, Any]
    """Free-form bag for timing info, retry counters, node history, etc."""
