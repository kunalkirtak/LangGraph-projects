"""
Configuration for the Enterprise Human Approval Workflow.

All configuration is read from environment variables. Nothing sensitive
is hardcoded here. In Colab, credentials should be injected via
`google.colab.userdata` before this module is imported (see the README
and the generated notebook).
"""

from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv

# Loads a local .env file if present. In Colab, environment variables are
# instead set directly via google.colab.userdata (see README / notebook).
load_dotenv()


@dataclass(frozen=True)
class Settings:
    google_api_key: str
    gemini_model: str
    high_risk_threshold: int


def _get_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def load_settings() -> Settings:
    """Load settings from the current process environment.

    Called lazily (not at import time) so that test suites and CLI tools
    that never touch the LLM do not require GOOGLE_API_KEY to be set.
    """
    return Settings(
        google_api_key=os.environ.get("GOOGLE_API_KEY", ""),
        gemini_model=os.environ.get("GEMINI_MODEL", "gemini-2.0-flash"),
        high_risk_threshold=_get_int("HIGH_RISK_THRESHOLD", 40),
    )
