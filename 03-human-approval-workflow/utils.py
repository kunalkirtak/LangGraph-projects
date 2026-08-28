"""
Small shared utilities: logging setup, ID generation, and timestamps.

Kept deliberately tiny — this project favors a handful of clear modules
over a sprawling utils grab-bag.
"""

from __future__ import annotations

import logging
import uuid
from datetime import datetime, timezone


def setup_logging(level: int = logging.INFO) -> logging.Logger:
    """Configure and return the shared application logger.

    Safe to call multiple times (e.g. once per notebook cell) without
    duplicating log handlers.
    """
    logger = logging.getLogger("approval_workflow")
    logger.setLevel(level)

    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            fmt="%(asctime)s | %(levelname)s | %(message)s",
            datefmt="%H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    logger.propagate = False
    return logger


def generate_request_id() -> str:
    """Generate a short, human-readable, unique request identifier."""
    return f"REQ-{uuid.uuid4().hex[:8].upper()}"


def utc_now_iso() -> str:
    """Return the current UTC time as an ISO-8601 string."""
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


logger = setup_logging()
