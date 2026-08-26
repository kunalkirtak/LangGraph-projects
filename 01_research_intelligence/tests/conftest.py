"""
Shared pytest fixtures.

Retry backoff intervals default to values sensible for production
(0.5s initial, 2x backoff). Tests override them to be effectively
instant so the retry-recovery test in `test_graph.py` runs fast
without weakening what it actually verifies (number of attempts,
final state, error propagation).
"""

import os

import pytest


@pytest.fixture(autouse=True)
def fast_retry_settings(monkeypatch):
    monkeypatch.setenv("RETRY_INITIAL_INTERVAL", "0.01")
    monkeypatch.setenv("RETRY_BACKOFF_FACTOR", "1.0")
    monkeypatch.setenv("MAX_RETRIES", "3")
    yield
