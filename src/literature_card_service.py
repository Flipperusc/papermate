"""Persistence helpers for saved literature cards."""

from __future__ import annotations

import re
import sqlite3
from typing import Any

from src.card_pipeline import UNKNOWN_VALUE
from src.db import get_db_connection, init_db


DEFAULT_LIBRARY_NAME = "默认卡片库"

CARD_FIELD_LABELS: dict[str, str] = {
    "title": "论文标题",
    "authors": "作者",
    "year": "年份",
    "research_field": "研究领域",
    "research_question": "研究问题",
    "method_summary": "方法概述",
    "datasets": "实验数据集",
}


def normalize_library_name(name: str) -> str:
    """Normalize a user-provided literature-card library name."""
    clean_name = re.sub(r"\s+", " ", name or "").strip()
    if not clean_name:
        raise ValueError("卡片库名字不能为空。")
    if len(clean_name) > 40:
        raise ValueError("卡片库名字最多 40 个字符。")
    return clean_name


def ensure_default_card_library(user_id: int, team_id: int | None = None) -> dict[str, Any]:
    """Ensure a user always has at least one literature-card library."""
    init_db()
    with get_db_connection() as connection:
        team_filter = "AND team_id = ?" if team_id is not None else "AND team_id IS NULL"
        parameters: list[Any] = [int(user_id)]
        if team_id is not None:
            parameters.append(int(team_id))
        row = connection.execute(
            f"""
            SELECT library_id, user_id, name, created_at, updated_at
            FROM card_libraries
            WHERE user_id = ?
                {team_filter}
            ORDER BY updated_at DESC, library_id DESC
            LIMIT 1
            """,
            parameters,
        ).fetchone()
        if row:
            return dict(row)

        try:
            cursor = connection.execute(
                """
                INSERT INTO card_libraries (user_id, team_id, name)
                VALUES (?, ?, ?)
                """,
                (int(user_id), team_id, DEFAULT_LIBRARY_NAME),
            )
        except sqlite3.IntegrityError:
            legacy_row = connection.execute(
                """
                SELECT library_id, user_id, team_id, name, created_at, updated_at
                FROM card_libraries
                WHERE user_id = ?
                    AND name = ?
                ORDER BY
                    CASE
                        WHEN team_id = ? THEN 1
                        WHEN team_id IS NULL THEN 2
                        ELSE 3
                    END,
                    updated_at DESC,
                    library_id DESC
                LIMIT 1
                """,
                (int(user_id), DEFAULT_LIBRARY_NAME, team_id),
            ).fetchone()
            if legacy_row:
                if team_id is not None and legacy_row["team_id"] is None:
                    connection.execute(
                        """
                        UPDATE card_libraries
                        SET team_id = ?, updated_at = CURRENT_TIMESTAMP
                        WHERE library_id = ?
                        """,
                        (int(team_id), int(legacy_row["library_id"])),
                    )
                    return get_card_library(int(legacy_row["library_id"]), user_id, team_id=team_id) or dict(legacy_row)
                return dict(legacy_row)
            raise
        library_id = int(cursor.lastrowid)

    return get_card_library(library_id, user_id, team_id=team_id) or {
        "library_id": library_id,
        "user_id": int(user_id),
        "team_id": team_id,
        "name": DEFAULT_LIBRARY_NAME,
    }


def list_card_libraries(user_id: int, team_id: int | None = None) -> list[dict[str, Any]]:
    """Return all literature-card libraries owned by one user."""
    init_db()
    ensure_default_card_library(user_id, team_id=team_id)

    where_sql = "WHERE l.user_id = ?"
    parameters: list[Any] = [int(user_id)]
    if team_id is not None:
        where_sql += " AND l.team_id = ?"
        parameters.append(int(team_id))
    with get_db_connection() as connection:
        rows = connection.execute(
            f"""
            SELECT
                l.library_id,
                l.user_id,
                l.team_id,
                l.visibility,
                l.name,
                l.created_at,
                l.updated_at,
                COUNT(c.card_id) AS card_count
            FROM card_libraries l
            LEFT JOIN literature_cards c
                ON c.library_id = l.library_id AND c.user_id = l.user_id
            {where_sql}
            GROUP BY l.library_id
            ORDER BY l.updated_at DESC, l.library_id DESC
            """,
            parameters,
        ).fetchall()

    return [dict(row) for row in rows]


def get_card_library(
    library_id: int,
    user_id: int,
    team_id: int | None = None,
) -> dict[str, Any] | None:
    """Return one card library owned by a user."""
    init_db()
    where_sql = "WHERE library_id = ? AND user_id = ?"
    parameters: list[Any] = [int(library_id), int(user_id)]
    if team_id is not None:
        where_sql += " AND team_id = ?"
        parameters.append(int(team_id))
    with get_db_connection() as connection:
        row = connection.execute(
            f"""
            SELECT library_id, user_id, team_id, visibility, name, created_at, updated_at
            FROM card_libraries
            {where_sql}
            """,
            parameters,
        ).fetchone()

    return dict(row) if row else None


def create_card_library(user_id: int, name: str, team_id: int | None = None) -> dict[str, Any]:
    """Create a literature-card library for one user."""
    clean_name = normalize_library_name(name)
    init_db()

    with get_db_connection() as connection:
        cursor = connection.execute(
            """
            INSERT INTO card_libraries (user_id, team_id, name)
            VALUES (?, ?, ?)
            """,
            (int(user_id), team_id, clean_name),
        )
        library_id = int(cursor.lastrowid)

    return get_card_library(library_id, user_id, team_id=team_id) or {
        "library_id": library_id,
        "user_id": int(user_id),
        "team_id": team_id,
        "name": clean_name,
    }


def update_card_library(
    library_id: int,
    user_id: int,
    name: str,
    team_id: int | None = None,
) -> None:
    """Rename a literature-card library owned by one user."""
    clean_name = normalize_library_name(name)
    init_db()
    team_sql = "AND team_id = ?" if team_id is not None else ""
    parameters: list[Any] = [clean_name, int(library_id), int(user_id)]
    if team_id is not None:
        parameters.append(int(team_id))
    with get_db_connection() as connection:
        connection.execute(
            f"""
            UPDATE card_libraries
            SET name = ?, updated_at = CURRENT_TIMESTAMP
            WHERE library_id = ? AND user_id = ?
                {team_sql}
            """,
            parameters,
        )


def claim_unassigned_literature_cards(user_id: int, team_id: int | None = None) -> int:
    """Assign legacy cards without an owner to the current user's default library."""
    library_id = int(ensure_default_card_library(user_id, team_id=team_id)["library_id"])
    init_db()
    with get_db_connection() as connection:
        cursor = connection.execute(
            """
            UPDATE literature_cards
            SET
                user_id = ?,
                library_id = COALESCE(library_id, ?),
                team_id = COALESCE(team_id, ?),
                updated_at = CURRENT_TIMESTAMP
            WHERE user_id IS NULL
            """,
            (int(user_id), library_id, team_id),
        )
        return int(cursor.rowcount)


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


def save_literature_card(
    paper_id: str,
    markdown: str,
    user_id: int | None = None,
    library_id: int | None = None,
    team_id: int | None = None,
    project_id: int | None = None,
) -> int:
    """Create one saved literature card for a paper."""
    init_db()
    if user_id is not None:
        if library_id is None:
            library_id = int(ensure_default_card_library(user_id, team_id=team_id)["library_id"])
        elif not get_card_library(library_id, user_id, team_id=team_id):
            raise ValueError("没有找到这个卡片库，或它不属于当前用户。")

    fields = extract_card_fields(markdown)
    normalized_markdown = build_card_markdown(fields)

    with get_db_connection() as connection:
        if team_id is None or project_id is None:
            paper_row = connection.execute(
                """
                SELECT team_id, project_id
                FROM papers
                WHERE paper_id = ?
                """,
                (paper_id,),
            ).fetchone()
            if paper_row:
                team_id = team_id if team_id is not None else paper_row["team_id"]
                project_id = project_id if project_id is not None else paper_row["project_id"]
        cursor = connection.execute(
            """
            INSERT INTO literature_cards (
                user_id,
                library_id,
                team_id,
                project_id,
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
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                user_id,
                library_id,
                team_id,
                project_id,
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


def list_literature_cards(
    user_id: int | None = None,
    library_id: int | None = None,
    team_id: int | None = None,
    project_id: int | None = None,
) -> list[dict[str, Any]]:
    """Return all saved literature cards with their source paper metadata."""
    init_db()
    where_clauses: list[str] = []
    parameters: list[Any] = []
    if user_id is not None:
        where_clauses.append("c.user_id = ?")
        parameters.append(int(user_id))
    if library_id is not None:
        where_clauses.append("c.library_id = ?")
        parameters.append(int(library_id))
    if team_id is not None:
        where_clauses.append("c.team_id = ?")
        parameters.append(int(team_id))
    if project_id is not None:
        where_clauses.append("c.project_id = ?")
        parameters.append(int(project_id))
    where_sql = f"WHERE {' AND '.join(where_clauses)}" if where_clauses else ""

    with get_db_connection() as connection:
        rows = connection.execute(
            f"""
            SELECT
                c.card_id,
                c.user_id,
                c.library_id,
                c.team_id,
                c.project_id,
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
                l.name AS library_name,
                p.file_name,
                p.save_path,
                p.page_count,
                p.file_size_bytes,
                pr.name AS project_name
            FROM literature_cards c
            LEFT JOIN card_libraries l ON l.library_id = c.library_id
            LEFT JOIN papers p ON p.paper_id = c.paper_id
            LEFT JOIN projects pr ON pr.project_id = c.project_id
            {where_sql}
            ORDER BY c.updated_at DESC, c.card_id DESC
            """,
            parameters,
        ).fetchall()

    return [dict(row) for row in rows]


def get_literature_card(
    card_id: int,
    user_id: int | None = None,
    team_id: int | None = None,
) -> dict[str, Any] | None:
    """Return one saved literature card by id."""
    init_db()
    where_sql = "WHERE c.card_id = ?"
    parameters: list[Any] = [int(card_id)]
    if user_id is not None:
        where_sql += " AND c.user_id = ?"
        parameters.append(int(user_id))
    if team_id is not None:
        where_sql += " AND c.team_id = ?"
        parameters.append(int(team_id))

    with get_db_connection() as connection:
        row = connection.execute(
            f"""
            SELECT
                c.card_id,
                c.user_id,
                c.library_id,
                c.team_id,
                c.project_id,
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
                l.name AS library_name,
                p.file_name,
                p.save_path,
                p.page_count,
                p.file_size_bytes,
                pr.name AS project_name
            FROM literature_cards c
            LEFT JOIN card_libraries l ON l.library_id = c.library_id
            LEFT JOIN papers p ON p.paper_id = c.paper_id
            LEFT JOIN projects pr ON pr.project_id = c.project_id
            {where_sql}
            """,
            parameters,
        ).fetchone()

    return dict(row) if row else None


def get_literature_card_by_paper(
    paper_id: str,
    user_id: int | None = None,
    library_id: int | None = None,
    team_id: int | None = None,
) -> dict[str, Any] | None:
    """Return the latest saved literature card for a paper if it exists."""
    init_db()
    where_clauses = ["paper_id = ?"]
    parameters: list[Any] = [paper_id]
    if user_id is not None:
        where_clauses.append("user_id = ?")
        parameters.append(int(user_id))
    if library_id is not None:
        where_clauses.append("library_id = ?")
        parameters.append(int(library_id))
    if team_id is not None:
        where_clauses.append("team_id = ?")
        parameters.append(int(team_id))

    with get_db_connection() as connection:
        row = connection.execute(
            f"""
            SELECT *
            FROM literature_cards
            WHERE {' AND '.join(where_clauses)}
            ORDER BY updated_at DESC, card_id DESC
            LIMIT 1
            """,
            parameters,
        ).fetchone()

    return dict(row) if row else None


def update_literature_card(
    card_id: int,
    fields: dict[str, str],
    user_id: int | None = None,
    team_id: int | None = None,
) -> None:
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
                AND (? IS NULL OR user_id = ?)
                AND (? IS NULL OR team_id = ?)
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
                user_id,
                user_id,
                team_id,
                team_id,
            ),
        )


def delete_literature_card(
    card_id: int,
    user_id: int | None = None,
    team_id: int | None = None,
) -> None:
    """Delete one saved literature card."""
    init_db()

    with get_db_connection() as connection:
        connection.execute(
            """
            DELETE FROM literature_cards
            WHERE card_id = ?
                AND (? IS NULL OR user_id = ?)
                AND (? IS NULL OR team_id = ?)
            """,
            (int(card_id), user_id, user_id, team_id, team_id),
        )


def delete_literature_cards(
    card_ids: list[int],
    user_id: int | None = None,
    team_id: int | None = None,
) -> int:
    """Delete multiple saved literature cards and return the affected count."""
    init_db()
    clean_ids = [int(card_id) for card_id in card_ids]
    if not clean_ids:
        return 0

    placeholders = ",".join("?" for _ in clean_ids)
    user_filter = " AND user_id = ?" if user_id is not None else ""
    team_filter = " AND team_id = ?" if team_id is not None else ""
    parameters: list[Any] = [*clean_ids]
    if user_id is not None:
        parameters.append(int(user_id))
    if team_id is not None:
        parameters.append(int(team_id))

    with get_db_connection() as connection:
        cursor = connection.execute(
            f"DELETE FROM literature_cards WHERE card_id IN ({placeholders}){user_filter}{team_filter}",
            parameters,
        )
        return int(cursor.rowcount)
