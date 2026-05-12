"""LangSmith tracing setup for Lab 24 scripts."""

from __future__ import annotations

import os

from config import LANGSMITH_API_KEY, LANGSMITH_ENDPOINT, LANGSMITH_PROJECT, LANGSMITH_TRACING


def configure_langsmith(run_name: str | None = None) -> bool:
    """Configure LangSmith/LangChain tracing environment variables.

    Returns True when tracing is enabled and an API key is present.
    """
    enabled = LANGSMITH_TRACING in {"1", "true", "yes", "on"}
    if not enabled or not LANGSMITH_API_KEY:
        return False

    os.environ["LANGSMITH_TRACING"] = "true"
    os.environ["LANGSMITH_API_KEY"] = LANGSMITH_API_KEY
    os.environ["LANGSMITH_PROJECT"] = LANGSMITH_PROJECT
    os.environ["LANGSMITH_ENDPOINT"] = LANGSMITH_ENDPOINT

    # Compatibility with older LangChain/RAGAS integrations.
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_API_KEY"] = LANGSMITH_API_KEY
    os.environ["LANGCHAIN_PROJECT"] = LANGSMITH_PROJECT
    os.environ["LANGCHAIN_ENDPOINT"] = LANGSMITH_ENDPOINT

    if run_name:
        os.environ["LANGSMITH_RUN_NAME"] = run_name
    return True
