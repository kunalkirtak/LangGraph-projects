"""
LangGraph node implementations.

Each node is produced by a small factory function (`make_research_node`,
`make_analysis_node`, `make_report_node`) that closes over an LLM
client. This keeps nodes pure functions of `(state) -> state update`
while making the LLM dependency explicit and injectable — which is
exactly what lets `tests/test_graph.py` run the whole graph against a
`FakeLLM` with no network access and no API key.

Each node:
1. Reads only the state it needs.
2. Performs its operation.
3. Returns only the state keys it owns.
4. Catches and records errors instead of letting exceptions escape.
5. Logs start/completion via the standard `logging` module.
"""

from __future__ import annotations

import time
from typing import Callable, Protocol

from config import configure_logging, load_settings
from prompts import ANALYSIS_PROMPT, REPORT_PROMPT, RESEARCH_PROMPT
from state import ResearchState

logger = configure_logging()


class LLMClient(Protocol):
    """Minimal interface nodes depend on.

    `ChatGoogleGenerativeAI` satisfies this via `.invoke(prompt)` ->
    object with a `.content` attribute, and so does the `FakeLLM` used
    in tests. Nodes never import `langchain_google_genai` directly,
    which keeps them decoupled from a specific provider.
    """

    def invoke(self, prompt: str) -> object: ...


def _text(response: object) -> str:
    """Extract plain text from an LLM response object."""
    content = getattr(response, "content", response)
    return content if isinstance(content, str) else str(content)


NodeFn = Callable[[ResearchState], ResearchState]


def invoke_with_retry(llm: LLMClient, prompt: str, node_name: str) -> str:
    """Call the LLM with a bounded exponential-backoff retry.

    This is the primary retry mechanism for transient LLM/API failures
    (rate limits, brief network blips, etc.). It is intentionally a
    plain loop rather than a separate retry framework: settings come
    from `config.load_settings()`, and every attempt is logged so the
    "failure -> retry -> success" path is visible in the logs.

    The compiled graph *also* configures a `RetryPolicy` per node (see
    `graph.py`) as a second, orchestration-level safety net for errors
    that occur outside this function. Because this loop already
    resolves ordinary LLM failures internally, that graph-level policy
    is a defense-in-depth measure and is not expected to trigger in
    normal operation.
    """
    settings = load_settings()
    delay = settings.retry_initial_interval
    last_exc: Exception | None = None

    for attempt in range(1, settings.max_retries + 1):
        try:
            response = llm.invoke(prompt)
            return _text(response)
        except Exception as exc:  # noqa: BLE001 - retried, then re-raised for the caller
            last_exc = exc
            logger.warning(
                "%s | attempt %d/%d failed: %s",
                node_name,
                attempt,
                settings.max_retries,
                exc,
            )
            if attempt < settings.max_retries:
                time.sleep(delay)
                delay *= settings.retry_backoff_factor

    assert last_exc is not None
    raise last_exc


def make_research_node(llm: LLMClient) -> NodeFn:
    def research_node(state: ResearchState) -> ResearchState:
        topic = state.get("topic", "")
        logger.info("Research node started | topic=%s", topic)
        try:
            prompt = RESEARCH_PROMPT.format(topic=topic)
            research_text = invoke_with_retry(llm, prompt, "research")
            logger.info("Research node completed")
            return {
                "research": research_text,
                "status": "research_complete",
            }
        except Exception as exc:  # noqa: BLE001 - intentionally broad, recorded in state
            logger.error("Research node failed: %s", exc)
            return {
                "errors": state.get("errors", []) + [f"Research node failed: {exc}"],
                "status": "research_failed",
            }

    return research_node


def make_analysis_node(llm: LLMClient) -> NodeFn:
    def analysis_node(state: ResearchState) -> ResearchState:
        topic = state.get("topic", "")
        research = state.get("research", "")
        logger.info("Analysis node started | topic=%s", topic)

        if state.get("status") == "research_failed":
            logger.info("Analysis node skipped | upstream research failed")
            return {"status": "analysis_skipped"}

        try:
            prompt = ANALYSIS_PROMPT.format(topic=topic, research=research)
            analysis_text = invoke_with_retry(llm, prompt, "analysis")
            logger.info("Analysis node completed")
            return {
                "analysis": analysis_text,
                "status": "analysis_complete",
            }
        except Exception as exc:  # noqa: BLE001
            logger.error("Analysis node failed: %s", exc)
            return {
                "errors": state.get("errors", []) + [f"Analysis node failed: {exc}"],
                "status": "analysis_failed",
            }

    return analysis_node


def make_report_node(llm: LLMClient) -> NodeFn:
    def report_node(state: ResearchState) -> ResearchState:
        topic = state.get("topic", "")
        research = state.get("research", "")
        analysis = state.get("analysis", "")
        logger.info("Report node started | topic=%s", topic)

        if state.get("status") in {"research_failed", "analysis_failed", "analysis_skipped"}:
            logger.info("Report node skipped | upstream failure")
            return {"status": "report_skipped"}

        try:
            prompt = REPORT_PROMPT.format(topic=topic, research=research, analysis=analysis)
            report_text = invoke_with_retry(llm, prompt, "report")
            logger.info("Report node completed")
            return {
                "report": report_text,
                "status": "report_complete",
            }
        except Exception as exc:  # noqa: BLE001
            logger.error("Report node failed: %s", exc)
            return {
                "errors": state.get("errors", []) + [f"Report node failed: {exc}"],
                "status": "report_failed",
            }

    return report_node
