"""
utils.py

Small, dependency-light helpers shared across nodes: logging setup, lead ID
generation, and a retry helper for transient LLM/API failures.
"""

from __future__ import annotations

import logging
import time
import uuid
from typing import Callable, TypeVar

from config import settings

T = TypeVar("T")


def configure_logging() -> logging.Logger:
    """Configure and return the package-wide logger.

    Safe to call multiple times (e.g. once from app.py, once from tests) --
    handlers are only attached once.
    """

    logger = logging.getLogger("sales_agent")
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter("%(levelname)s | %(message)s")
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(getattr(logging, settings.log_level.upper(), logging.INFO))
        logger.propagate = False
    return logger


logger = configure_logging()


def generate_lead_id() -> str:
    """Generate a short, stable-looking lead identifier."""

    return f"lead_{uuid.uuid4().hex[:10]}"


class TransientLLMError(Exception):
    """Raised by LLM wrapper calls to signal a retryable failure.

    This is distinct from validation errors (which are never retried) and
    from unrecoverable errors (which are logged and surfaced, not retried
    forever).
    """


def call_with_retry(
    fn: Callable[[], T],
    *,
    max_retries: int | None = None,
    base_delay_seconds: float = 0.0,
    node_name: str = "node",
) -> T:
    """Call ``fn`` with retry-on-``TransientLLMError`` semantics.

    Only :class:`TransientLLMError` triggers a retry. Any other exception
    (including validation-style errors) is raised immediately -- retrying
    invalid input is never appropriate.
    """

    attempts = (max_retries if max_retries is not None else settings.max_llm_retries) + 1
    last_error: Exception | None = None

    for attempt in range(1, attempts + 1):
        try:
            return fn()
        except TransientLLMError as exc:
            last_error = exc
            logger.info(
                "%s: transient failure on attempt %d/%d: %s",
                node_name,
                attempt,
                attempts,
                exc,
            )
            if attempt < attempts and base_delay_seconds > 0:
                time.sleep(base_delay_seconds)

    assert last_error is not None
    raise last_error
