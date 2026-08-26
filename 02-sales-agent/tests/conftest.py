"""
conftest.py

Deterministic fake LLMs used across the test suite. No test in this
project talks to a real Gemini API -- every LLM factory in ``nodes.py`` is
monkeypatched to return one of these fakes instead.
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

# Make the project root importable when pytest is run from the repo root.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from nodes import QualificationResult  # noqa: E402


class FakeQualificationLLM:
    """Deterministic stand-in for ``get_qualification_llm()``."""

    def __init__(self, score: int, status: str, reason: str = "Mock qualification"):
        self.score = score
        self.status = status
        self.reason = reason

    def invoke(self, messages):
        return QualificationResult(
            score=self.score,
            status=self.status,
            reason=self.reason,
            strengths=["Clear stated need"] if self.score >= 60 else [],
            concerns=["Budget unclear"] if self.score < 60 else [],
        )


class FakeContentLLM:
    """Deterministic stand-in for research/outreach/nurture LLMs."""

    def __init__(self, text: str):
        self.text = text

    def invoke(self, messages):
        return SimpleNamespace(content=self.text)


class RaisingLLM:
    """Fake LLM that always raises, to simulate a transient provider failure."""

    def __init__(self, exc: Exception):
        self.exc = exc
        self.calls = 0

    def invoke(self, messages):
        self.calls += 1
        raise self.exc


@pytest.fixture
def qualified_llm_factory():
    return lambda: FakeQualificationLLM(score=85, status="qualified")


@pytest.fixture
def unqualified_llm_factory():
    return lambda: FakeQualificationLLM(score=25, status="unqualified")


@pytest.fixture
def content_llm_factory():
    return lambda: FakeContentLLM(text="This is a mock generated message.")


@pytest.fixture(autouse=True)
def _patch_llms(monkeypatch, request):
    """Patch every node-level LLM factory with fakes for every test.

    Individual tests can further monkeypatch ``nodes.get_qualification_llm``
    etc. to install a different fake (e.g. one that raises) after this
    fixture has run.
    """

    import nodes

    monkeypatch.setattr(nodes, "get_qualification_llm", lambda: FakeQualificationLLM(85, "qualified"))
    monkeypatch.setattr(nodes, "get_research_llm", lambda: FakeContentLLM("Mock research brief."))
    monkeypatch.setattr(nodes, "get_content_llm", lambda: FakeContentLLM("Mock generated message."))
    yield
