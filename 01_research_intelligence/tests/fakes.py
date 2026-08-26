"""
A minimal fake LLM used across the test suite.

Satisfies the same `.invoke(prompt) -> response.content` interface as
`ChatGoogleGenerativeAI`, so `build_graph(llm=FakeLLM())` runs the real
graph, with real state propagation, without any network access or API
key. This is what lets `pytest` pass with zero external dependencies.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class _FakeResponse:
    content: str


class FakeLLM:
    """Returns a canned, prompt-aware response for each `.invoke()` call."""

    def __init__(self, fail_on: set[str] | None = None, fail_times: int = 0):
        # fail_on: substrings of the prompt that should raise on the first call.
        self.fail_on = fail_on or set()
        self.fail_times = fail_times
        self._fail_counts: dict[str, int] = {}
        self.calls: list[str] = []

    def invoke(self, prompt: str) -> _FakeResponse:
        self.calls.append(prompt)

        for marker in self.fail_on:
            if marker in prompt:
                seen = self._fail_counts.get(marker, 0)
                if seen < self.fail_times:
                    self._fail_counts[marker] = seen + 1
                    raise RuntimeError(f"Simulated transient failure for '{marker}'")

        if "Produce a structured research synthesis" in prompt:
            return _FakeResponse(content="## Research\nFake research synthesis body.")
        if "Analyze this research" in prompt:
            return _FakeResponse(content="## Analysis\nFake analysis body.")
        if "Write a polished Markdown report" in prompt:
            return _FakeResponse(
                content=(
                    "# Executive Summary\nFake summary.\n\n"
                    "# Background\nFake background.\n\n"
                    "# Key Findings\nFake findings.\n\n"
                    "# Analysis\nFake analysis.\n\n"
                    "# Opportunities\nFake opportunities.\n\n"
                    "# Risks\nFake risks.\n\n"
                    "# Engineering Recommendations\nFake recommendations.\n\n"
                    "# Conclusion\nFake conclusion.\n"
                )
            )
        return _FakeResponse(content="Fake generic response.")
