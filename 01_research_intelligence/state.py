"""
Shared graph state for the AI Research Intelligence Pipeline.

This module defines the schema only. It intentionally contains no
business logic — nodes in `nodes.py` are the only place state is
produced or transformed.

Every node reads a subset of this state and returns only the keys it
owns. LangGraph merges each node's returned dict into the running
state, which is how information (research findings, analysis, etc.)
propagates from one node to the next without global variables or
tightly-coupled function calls.
"""

from __future__ import annotations

from typing import TypedDict


class ResearchState(TypedDict, total=False):
    """Shared state that flows through the entire graph.

    Fields:
        topic: The user-supplied research topic. Set once, read by
            every node.
        research: LLM-generated research synthesis. Written by
            `research_node`, read by `analysis_node` and `report_node`.
        analysis: Findings/opportunities/risks derived from `research`.
            Written by `analysis_node`, read by `report_node`.
        report: The final Markdown research report. Written by
            `report_node`.
        status: A short machine-readable status string, e.g.
            "started", "research_complete", "analysis_complete",
            "report_complete", "research_failed", etc.
        errors: Accumulated error messages. Nodes append to this list
            instead of raising uncaught exceptions, so a partial
            pipeline result is always inspectable.
        metadata: Execution metadata such as execution_id, start_time,
            and completed_nodes — used for basic observability.
    """

    topic: str
    research: str
    analysis: str
    report: str
    status: str
    errors: list[str]
    metadata: dict


def initial_state(topic: str) -> ResearchState:
    """Build a fresh, well-formed state for a new workflow run."""
    return ResearchState(
        topic=topic,
        research="",
        analysis="",
        report="",
        status="started",
        errors=[],
        metadata={},
    )
