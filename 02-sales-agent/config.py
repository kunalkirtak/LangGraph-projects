"""
config.py

Centralizes environment/configuration handling for the sales lead agent.
No secrets are ever hardcoded here -- everything is read from environment
variables, with sane (non-secret) defaults for local development.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv

# Load a local .env file if present. Safe no-op if it doesn't exist.
load_dotenv()


@dataclass(frozen=True)
class Settings:
    """Runtime configuration for the sales lead agent."""

    google_api_key: str | None
    gemini_model: str
    qualification_threshold: int
    max_llm_retries: int
    log_level: str


def _get_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def load_settings() -> Settings:
    """Build a :class:`Settings` instance from the current environment.

    Called lazily (not at import time) so that tests and notebooks can set
    environment variables *before* the settings are constructed.
    """

    return Settings(
        google_api_key=os.environ.get("GOOGLE_API_KEY"),
        gemini_model=os.environ.get("GEMINI_MODEL", "gemini-3.6-flash"),
        qualification_threshold=_get_int("QUALIFICATION_THRESHOLD", 60),
        max_llm_retries=_get_int("MAX_LLM_RETRIES", 2),
        log_level=os.environ.get("LOG_LEVEL", "INFO"),
    )


settings = load_settings()
