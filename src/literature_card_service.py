"""Persistence helpers for saved literature cards."""

from __future__ import annotations

import re
from typing import Any

from src.card_pipeline import UNKNOWN_VALUE
from src.db import get_db_connection, init_db


CARD_FIELD_LABELS: dict[str, str] = {
    "title": "论文标题",
    "authors": "作者",
    "year": "年份",
    "research_field": "研究领域",
    "research_question": "研究问题",
    "method_summary": "方法概述",
    "datasets": "实验数据集",
}


def extract_card_fields(markdown: str) -> dict[str, str]:
    """Extract structured card fields from a markdown literature card."""
    fields: dict[str, str] = {}
    for key, label in CARD_FIELD_LABELS.items():
        pattern = rf"^##\s+{re.escape(label)}\s*\n(?P<value>.*?)(?=^##\s+|\Z)"
        match = re.search(pattern, markdown, flags=re.MULTILINE | re.DOTALL)
        value = match.group("value").strip() if match else ""
        fields[key] = value or UNKNOWN_VALUE
    return fields


def build_card_markdown(fields: dict[str, str]) -> str:
    """Build normalized markdown from structured literature-card fields."""
    lines = ["# 文献卡片", ""]
    for key, label in CARD_FIELD_LABELS.items():
        value = str(fields.get(key) or "").strip() or UNKNOWN_VALUE
        lines.extend([f"## {label}", value, ""])
    return "\n".join(lines).strip()


def save_literature_card(paper_id: str, markdown: str) -> int:
    """Create one saved literature card for a paper."""
    init_db()
    fields = extract_card_fields(markdown)
    normalized_markdown = build_card_markdown(fields)

    with get_db_connection() as connection:
        cursor = connection.execute(
            """
            INSERT INTO literature_cards (
                paper_id,
                title,
                authors,
                year,
                research_field,
                research_question,
                method_summary,
                datasets,
                markdown
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                paper_id,
                fields["title"],
                fields["authors"],
                fields["year"],
                fields["research_field"],
                fields["research_question"],
                fields["method_summary"],
                fields["datasets"],
                normalized_markdown,
            ),
        )

    return int(cursor.lastrowid)


def list_literature_cards() -> list[dict[str, Any]]:
    """Return all saved literature cards with their source paper metadata."""
    init_db()

    with get_db_connection() as connection:
        rows = connection.execute(
            """
            SELECT
                c.card_id,
                c.paper_id,
                c.title,
                c.authors,
                c.year,
                c.research_field,
                c.research_question,
                c.method_summary,
                c.datasets,
                c.markdown,
                c.created_at,
                c.updated_at,
                p.file_name,
                p.save_path,
                p.page_count,
                p.file_size_bytes
            FROM literature_cards c
            LEFT JOIN papers p ON p.paper_id = c.paper_id
            ORDER BY c.updated_at DESC, c.card_id DESC
            """
        ).fetchall()

    return [dict(row) for row in rows]


def get_literature_card(card_id: int) -> dict[str, Any] | None:
    """Return one saved literature card by id."""
    init_db()

    with get_db_connection() as connection:
        row = connection.execute(
            """
            SELECT
                c.card_id,
                c.paper_id,
                c.title,
                c.authors,
                c.year,
                c.research_field,
                c.research_question,
                c.method_summary,
                c.datasets,
                c.markdown,
                c.created_at,
                c.updated_at,
                p.file_name,
                p.save_path,
                p.page_count,
                p.file_size_bytes
            FROM literature_cards c
            LEFT JOIN papers p ON p.paper_id = c.paper_id
            WHERE c.card_id = ?
            """,
            (card_id,),
        ).fetchone()

    return dict(row) if row else None


def get_literature_card_by_paper(paper_id: str) -> dict[str, Any] | None:
    """Return the latest saved literature card for a paper if it exists."""
    init_db()

    with get_db_connection() as connection:
        row = connection.execute(
            """
            SELECT *
            FROM literature_cards
            WHERE paper_id = ?
            ORDER BY updated_at DESC, card_id DESC
            LIMIT 1
            """,
            (paper_id,),
        ).fetchone()

    return dict(row) if row else None


def update_literature_card(card_id: int, fields: dict[str, str]) -> None:
    """Update structured fields and regenerate markdown."""
    init_db()
    normalized_fields = {
        key: str(fields.get(key) or "").strip() or UNKNOWN_VALUE
        for key in CARD_FIELD_LABELS
    }
    markdown = build_card_markdown(normalized_fields)

    with get_db_connection() as connection:
        connection.execute(
            """
            UPDATE literature_cards
            SET
                title = ?,
                authors = ?,
                year = ?,
                research_field = ?,
                research_question = ?,
                method_summary = ?,
                datasets = ?,
                markdown = ?,
                updated_at = CURRENT_TIMESTAMP
            WHERE card_id = ?
            """,
            (
                normalized_fields["title"],
                normalized_fields["authors"],
                normalized_fields["year"],
                normalized_fields["research_field"],
                normalized_fields["research_question"],
                normalized_fields["method_summary"],
                normalized_fields["datasets"],
                markdown,
                card_id,
            ),
        )


def delete_literature_card(card_id: int) -> None:
    """Delete one saved literature card."""
    init_db()

    with get_db_connection() as connection:
        connection.execute("DELETE FROM literature_cards WHERE card_id = ?", (card_id,))


def delete_literature_cards(card_ids: list[int]) -> int:
    """Delete multiple saved literature cards and return the affected count."""
    init_db()
    clean_ids = [int(card_id) for card_id in card_ids]
    if not clean_ids:
        return 0

    placeholders = ",".join("?" for _ in clean_ids)
    with get_db_connection() as connection:
        cursor = connection.execute(
            f"DELETE FROM literature_cards WHERE card_id IN ({placeholders})",
            clean_ids,
        )
        return int(cursor.rowcount)
