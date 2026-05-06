"""SQLite database utilities for PaperMate."""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any

from config import settings


def ensure_data_directories() -> tuple[Path, Path]:
    """Ensure local data directories exist."""
    settings.upload_dir.mkdir(parents=True, exist_ok=True)
    settings.chroma_dir.mkdir(parents=True, exist_ok=True)
    settings.mineru_output_dir.mkdir(parents=True, exist_ok=True)
    settings.db_path.parent.mkdir(parents=True, exist_ok=True)
    return settings.upload_dir, settings.chroma_dir


def get_db_connection(db_path: str | Path | None = None) -> sqlite3.Connection:
    """Create a SQLite connection with PaperMate defaults."""
    ensure_data_directories()
    path = Path(db_path) if db_path is not None else settings.db_path
    path.parent.mkdir(parents=True, exist_ok=True)

    connection = sqlite3.connect(path)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")
    return connection


def init_db(db_path: str | Path | None = None) -> None:
    """Create PaperMate database tables if they do not already exist."""
    with get_db_connection(db_path) as connection:
        connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS papers (
                paper_id TEXT PRIMARY KEY,
                file_name TEXT NOT NULL,
                file_size_bytes INTEGER NOT NULL,
                save_path TEXT NOT NULL,
                page_count INTEGER NOT NULL DEFAULT 0,
                total_chars INTEGER NOT NULL DEFAULT 0,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            );

            CREATE TABLE IF NOT EXISTS chunks (
                chunk_id TEXT PRIMARY KEY,
                paper_id TEXT NOT NULL,
                chunk_index INTEGER NOT NULL,
                page_num INTEGER NOT NULL,
                section_title TEXT,
                text TEXT NOT NULL,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (paper_id) REFERENCES papers (paper_id) ON DELETE CASCADE
            );

            CREATE UNIQUE INDEX IF NOT EXISTS idx_chunks_paper_index
                ON chunks (paper_id, chunk_index);

            CREATE TABLE IF NOT EXISTS qa_logs (
                qa_log_id INTEGER PRIMARY KEY AUTOINCREMENT,
                paper_id TEXT,
                question TEXT NOT NULL,
                answer TEXT NOT NULL,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (paper_id) REFERENCES papers (paper_id) ON DELETE SET NULL
            );

            CREATE TABLE IF NOT EXISTS feedback (
                feedback_id INTEGER PRIMARY KEY AUTOINCREMENT,
                paper_id TEXT,
                chunk_id TEXT,
                qa_log_id INTEGER,
                rating INTEGER,
                feedback_type TEXT,
                is_negative INTEGER NOT NULL DEFAULT 0,
                comment TEXT,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (paper_id) REFERENCES papers (paper_id) ON DELETE SET NULL,
                FOREIGN KEY (chunk_id) REFERENCES chunks (chunk_id) ON DELETE SET NULL,
                FOREIGN KEY (qa_log_id) REFERENCES qa_logs (qa_log_id) ON DELETE SET NULL
            );

            CREATE TABLE IF NOT EXISTS bad_cases (
                bad_case_id INTEGER PRIMARY KEY AUTOINCREMENT,
                paper_id TEXT,
                question TEXT,
                answer TEXT,
                error_type TEXT,
                reason TEXT NOT NULL DEFAULT '',
                solution TEXT NOT NULL DEFAULT '',
                status TEXT NOT NULL DEFAULT 'open',
                expected_answer TEXT,
                actual_answer TEXT,
                notes TEXT,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (paper_id) REFERENCES papers (paper_id) ON DELETE SET NULL
            );

            CREATE TABLE IF NOT EXISTS literature_cards (
                card_id INTEGER PRIMARY KEY AUTOINCREMENT,
                paper_id TEXT NOT NULL,
                title TEXT NOT NULL DEFAULT '',
                authors TEXT NOT NULL DEFAULT '',
                year TEXT NOT NULL DEFAULT '',
                research_field TEXT NOT NULL DEFAULT '',
                research_question TEXT NOT NULL DEFAULT '',
                method_summary TEXT NOT NULL DEFAULT '',
                datasets TEXT NOT NULL DEFAULT '',
                markdown TEXT NOT NULL,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (paper_id) REFERENCES papers (paper_id) ON DELETE CASCADE
            );

            CREATE INDEX IF NOT EXISTS idx_literature_cards_updated_at
                ON literature_cards (updated_at DESC);

            CREATE INDEX IF NOT EXISTS idx_literature_cards_paper_id
                ON literature_cards (paper_id);
            """
        )
        ensure_schema_migrations(connection)


def ensure_schema_migrations(connection: sqlite3.Connection) -> None:
    """Add new columns for existing local databases."""
    # This app keeps user data in local SQLite files, so startup migrations must
    # be repeatable and additive instead of assuming a freshly created schema.
    migrate_literature_cards_allow_duplicates(connection)
    ensure_columns(
        connection,
        "feedback",
        {
            "feedback_type": "TEXT",
            "is_negative": "INTEGER NOT NULL DEFAULT 0",
        },
    )
    ensure_columns(
        connection,
        "bad_cases",
        {
            "answer": "TEXT",
            "error_type": "TEXT",
            "reason": "TEXT NOT NULL DEFAULT ''",
            "solution": "TEXT NOT NULL DEFAULT ''",
            "status": "TEXT NOT NULL DEFAULT 'open'",
        },
    )
    ensure_columns(
        connection,
        "literature_cards",
        {
            "title": "TEXT NOT NULL DEFAULT ''",
            "authors": "TEXT NOT NULL DEFAULT ''",
            "year": "TEXT NOT NULL DEFAULT ''",
            "research_field": "TEXT NOT NULL DEFAULT ''",
            "research_question": "TEXT NOT NULL DEFAULT ''",
            "method_summary": "TEXT NOT NULL DEFAULT ''",
            "datasets": "TEXT NOT NULL DEFAULT ''",
            "markdown": "TEXT NOT NULL DEFAULT ''",
            "updated_at": "TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP",
        },
    )


def migrate_literature_cards_allow_duplicates(connection: sqlite3.Connection) -> None:
    """Remove the old one-card-per-paper unique constraint if it exists."""
    row = connection.execute(
        """
        SELECT sql
        FROM sqlite_master
        WHERE type = 'table' AND name = 'literature_cards'
        """
    ).fetchone()
    if not row or "paper_id TEXT NOT NULL UNIQUE" not in str(row["sql"]):
        connection.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_literature_cards_paper_id
                ON literature_cards (paper_id)
            """
        )
        return

    connection.execute("ALTER TABLE literature_cards RENAME TO literature_cards_old")
    connection.executescript(
        """
        CREATE TABLE literature_cards (
            card_id INTEGER PRIMARY KEY AUTOINCREMENT,
            paper_id TEXT NOT NULL,
            title TEXT NOT NULL DEFAULT '',
            authors TEXT NOT NULL DEFAULT '',
            year TEXT NOT NULL DEFAULT '',
            research_field TEXT NOT NULL DEFAULT '',
            research_question TEXT NOT NULL DEFAULT '',
            method_summary TEXT NOT NULL DEFAULT '',
            datasets TEXT NOT NULL DEFAULT '',
            markdown TEXT NOT NULL,
            created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
            updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (paper_id) REFERENCES papers (paper_id) ON DELETE CASCADE
        );

        INSERT INTO literature_cards (
            card_id,
            paper_id,
            title,
            authors,
            year,
            research_field,
            research_question,
            method_summary,
            datasets,
            markdown,
            created_at,
            updated_at
        )
        SELECT
            card_id,
            paper_id,
            title,
            authors,
            year,
            research_field,
            research_question,
            method_summary,
            datasets,
            markdown,
            created_at,
            updated_at
        FROM literature_cards_old;

        DROP TABLE literature_cards_old;

        CREATE INDEX IF NOT EXISTS idx_literature_cards_updated_at
            ON literature_cards (updated_at DESC);

        CREATE INDEX IF NOT EXISTS idx_literature_cards_paper_id
            ON literature_cards (paper_id);
        """
    )


def ensure_columns(
    connection: sqlite3.Connection,
    table_name: str,
    columns: dict[str, str],
) -> None:
    """Ensure columns exist on a SQLite table."""
    existing_columns = {
        row["name"]
        for row in connection.execute(f"PRAGMA table_info({table_name})").fetchall()
    }
    for column_name, column_sql in columns.items():
        if column_name not in existing_columns:
            connection.execute(f"ALTER TABLE {table_name} ADD COLUMN {column_name} {column_sql}")


def save_paper_and_chunks(paper: dict[str, Any], chunks: list[dict[str, Any]]) -> None:
    """Persist one paper and its generated chunks in a single transaction."""
    init_db()

    with get_db_connection() as connection:
        connection.execute(
            """
            INSERT OR REPLACE INTO papers (
                paper_id,
                file_name,
                file_size_bytes,
                save_path,
                page_count,
                total_chars
            )
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                paper["paper_id"],
                paper["file_name"],
                paper["file_size_bytes"],
                paper["save_path"],
                paper["page_count"],
                paper["total_chars"],
            ),
        )

        connection.execute("DELETE FROM chunks WHERE paper_id = ?", (paper["paper_id"],))
        connection.executemany(
            """
            INSERT INTO chunks (
                chunk_id,
                paper_id,
                chunk_index,
                page_num,
                section_title,
                text
            )
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            [
                (
                    chunk["chunk_id"],
                    chunk["paper_id"],
                    chunk["chunk_index"],
                    chunk["page_num"],
                    chunk["section_title"],
                    chunk["text"],
                )
                for chunk in chunks
            ],
        )


def save_qa_log(paper_id: str, question: str, answer: str) -> int:
    """Persist one question-answer record and return its id."""
    init_db()

    with get_db_connection() as connection:
        cursor = connection.execute(
            """
            INSERT INTO qa_logs (paper_id, question, answer)
            VALUES (?, ?, ?)
            """,
            (paper_id, question, answer),
        )
        return int(cursor.lastrowid)


def get_paper_chunks(paper_id: str) -> list[dict[str, Any]]:
    """Return chunks for a paper ordered by chunk index."""
    init_db()

    with get_db_connection() as connection:
        rows = connection.execute(
            """
            SELECT
                chunk_id,
                paper_id,
                chunk_index,
                page_num,
                section_title,
                text
            FROM chunks
            WHERE paper_id = ?
            ORDER BY chunk_index ASC
            """,
            (paper_id,),
        ).fetchall()

    return [dict(row) for row in rows]
