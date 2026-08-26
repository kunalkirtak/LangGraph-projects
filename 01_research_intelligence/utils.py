"""
Small, independently-testable utility functions used across the
pipeline: report persistence and lightweight execution metadata.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from pathlib import Path


def save_report(report: str, filename: str = "research_report.md") -> Path:
    """Write a generated report to disk as a Markdown file.

    Returns the resolved Path so callers can confirm/print the location.
    """
    path = Path(filename)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(report, encoding="utf-8")
    return path


def new_execution_metadata() -> dict:
    """Create a fresh metadata dict for a single pipeline run.

    Kept intentionally small: an execution id, a start timestamp, and a
    running list of completed nodes. This is enough to demonstrate
    basic production-style observability without over-engineering a
    full tracing system.
    """
    return {
        "execution_id": str(uuid.uuid4()),
        "start_time": datetime.now(timezone.utc).isoformat(),
        "completed_nodes": [],
    }


def mark_node_complete(metadata: dict, node_name: str) -> dict:
    """Return a copy of metadata with `node_name` appended to completed_nodes."""
    updated = dict(metadata)
    completed = list(updated.get("completed_nodes", []))
    completed.append(node_name)
    updated["completed_nodes"] = completed
    return updated
