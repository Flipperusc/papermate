"""Feedback and bad case persistence."""

from __future__ import annotations

from typing import Any

from src.db import get_db_connection, init_db


POSITIVE_FEEDBACK = "有帮助"
NEGATIVE_FEEDBACK_TYPES = {
    "不准确",
    "引用不支持答案",
    "回答太空泛",
    "模型编造",
    "没有回答我的问题",
}
FEEDBACK_OPTIONS = [
    POSITIVE_FEEDBACK,
    "不准确",
    "引用不支持答案",
    "回答太空泛",
    "模型编造",
    "没有回答我的问题",
]


def is_negative_feedback(feedback_type: str) -> bool:
    """Return whether a feedback option should be treated as negative."""
    return feedback_type in NEGATIVE_FEEDBACK_TYPES


def save_feedback(
    paper_id: str,
    question: str,
    answer: str,
    feedback_type: str,
    comment: str = "",
    qa_log_id: int | None = None,
    chunk_id: str | None = None,
) -> dict[str, Any]:
    """Save feedback and create a bad case for negative feedback."""
    init_db()
    is_negative = is_negative_feedback(feedback_type)
    rating = 1 if feedback_type == POSITIVE_FEEDBACK else 0

    with get_db_connection() as connection:
        feedback_cursor = connection.execute(
            """
            INSERT INTO feedback (
                paper_id,
                chunk_id,
                qa_log_id,
                rating,
                feedback_type,
                is_negative,
                comment
            )
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                paper_id,
                chunk_id,
                qa_log_id,
                rating,
                feedback_type,
                1 if is_negative else 0,
                comment,
            ),
        )
        feedback_id = int(feedback_cursor.lastrowid)
        bad_case_id: int | None = None

        if is_negative:
            bad_case_cursor = connection.execute(
                """
                INSERT INTO bad_cases (
                    paper_id,
                    question,
                    answer,
                    error_type,
                    reason,
                    solution,
                    status,
                    actual_answer,
                    notes
                )
                VALUES (?, ?, ?, ?, '', '', 'open', ?, ?)
                """,
                (
                    paper_id,
                    question,
                    answer,
                    feedback_type,
                    answer,
                    comment,
                ),
            )
            bad_case_id = int(bad_case_cursor.lastrowid)

    return {
        "feedback_id": feedback_id,
        "bad_case_id": bad_case_id,
        "is_negative": is_negative,
    }


def list_feedback_records(limit: int = 200) -> list[dict[str, Any]]:
    """Return recent feedback records with paper and QA context."""
    init_db()

    with get_db_connection() as connection:
        rows = connection.execute(
            """
            SELECT
                f.feedback_id,
                f.paper_id,
                p.file_name,
                f.qa_log_id,
                q.question,
                q.answer,
                f.feedback_type,
                f.is_negative,
                f.comment,
                f.created_at
            FROM feedback f
            LEFT JOIN papers p ON p.paper_id = f.paper_id
            LEFT JOIN qa_logs q ON q.qa_log_id = f.qa_log_id
            ORDER BY f.created_at DESC, f.feedback_id DESC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()

    return [dict(row) for row in rows]


def list_bad_cases(limit: int = 200) -> list[dict[str, Any]]:
    """Return recent bad cases with paper context."""
    init_db()

    with get_db_connection() as connection:
        rows = connection.execute(
            """
            SELECT
                b.bad_case_id,
                b.paper_id,
                p.file_name,
                b.question,
                b.answer,
                b.error_type,
                b.reason,
                b.solution,
                b.status,
                b.notes,
                b.created_at
            FROM bad_cases b
            LEFT JOIN papers p ON p.paper_id = b.paper_id
            ORDER BY b.created_at DESC, b.bad_case_id DESC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()

    return [dict(row) for row in rows]


class FeedbackService:
    """Collect and store user feedback."""

    def record(self, item_id: str, rating: int, comment: str = "") -> None:
        """Record feedback for backward compatibility."""
        feedback_type = POSITIVE_FEEDBACK if rating > 0 else "不准确"
        save_feedback(
            paper_id=item_id,
            question="",
            answer="",
            feedback_type=feedback_type,
            comment=comment,
        )
