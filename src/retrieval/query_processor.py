"""Backward-compatible query processing API.

The retrieval stack now uses :mod:`src.retrieval.query_planner` internally.
This module keeps the older import paths stable for scripts and callers.
"""

from __future__ import annotations

from src.retrieval.query_planner import (
    QueryPlan,
    classify_query,
    expand_query,
    get_rrf_weights,
    normalize_query,
    plan_query,
    unique_preserve_order,
)


def process_query(question: str) -> QueryPlan:
    """Return the enhanced query plan for a question."""
    return plan_query(question)


def detect_question_type(normalized_query: str, keywords: list[str] | None = None) -> str:
    """Backward-compatible alias for older retrieval code."""
    del keywords
    return classify_query(normalized_query)


__all__ = [
    "QueryPlan",
    "classify_query",
    "detect_question_type",
    "expand_query",
    "get_rrf_weights",
    "normalize_query",
    "plan_query",
    "process_query",
    "unique_preserve_order",
]
