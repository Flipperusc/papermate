"""Evaluation utilities placeholder."""

from __future__ import annotations


class EvaluationService:
    """Evaluate retrieval and generation quality."""

    def evaluate_answer(self, question: str, answer: str, context: str) -> dict[str, float]:
        """Evaluate a generated answer."""
        raise NotImplementedError("Answer evaluation is not implemented yet.")
