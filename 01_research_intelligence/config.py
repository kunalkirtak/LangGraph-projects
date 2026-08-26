"""
Central configuration for the AI Research Intelligence Pipeline.

All environment-dependent settings live here. No secrets are ever
hardcoded — the Gemini API key is read exclusively from the environment
(or from Colab Secrets when running in Google Colab, see README.md).
"""

from __future__ import annotations

import os
import logging
from dataclasses import dataclass

from dotenv import load_dotenv

# Load a local .env file if present (no-op in Colab, harmless in prod).
load_dotenv()


def _get_api_key() -> str | None:
    """Read the Gemini API key from the environment.

    Returns None instead of raising so that modules which do not need
    the LLM (state, graph shape, mock-based tests) can still be imported
    without a key being configured.
    """
    return os.getenv("GOOGLE_API_KEY")


@dataclass(frozen=True)
class Settings:
    google_api_key: str | None
    gemini_model: str
    log_level: str
    max_retries: int
    retry_initial_interval: float
    retry_backoff_factor: float


def load_settings() -> Settings:
    """Build a Settings object from the current environment.

    Called lazily (not at import time) so tests and tooling can mutate
    environment variables before configuration is resolved.
    """
    return Settings(
        google_api_key=_get_api_key(),
        gemini_model=os.getenv("GEMINI_MODEL", "gemini-2.5-flash"),
        log_level=os.getenv("LOG_LEVEL", "INFO"),
        max_retries=int(os.getenv("MAX_RETRIES", "3")),
        retry_initial_interval=float(os.getenv("RETRY_INITIAL_INTERVAL", "0.5")),
        retry_backoff_factor=float(os.getenv("RETRY_BACKOFF_FACTOR", "2.0")),
    )


def configure_logging(level: str | None = None) -> logging.Logger:
    """Configure and return the package-wide logger.

    Uses the standard `logging` module (never bare `print`) so that log
    output is filterable, timestamped, and safe for production use.
    """
    settings = load_settings()
    resolved_level = level or settings.log_level

    logger = logging.getLogger("research_pipeline")
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter("%(levelname)s | %(name)s | %(message)s")
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    logger.setLevel(resolved_level)
    logger.propagate = False
    return logger
