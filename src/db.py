"""SQLite database utilities for PaperMate."""

from __future__ import annotations

import json
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
            CREATE TABLE IF NOT EXISTS users (
                user_id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT NOT NULL UNIQUE COLLATE NOCASE,
                password_hash TEXT NOT NULL,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                last_login_at TEXT
            );

            CREATE TABLE IF NOT EXISTS teams (
                team_id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                owner_user_id INTEGER NOT NULL,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (owner_user_id) REFERENCES users (user_id) ON DELETE CASCADE
            );

            CREATE TABLE IF NOT EXISTS team_members (
                team_id INTEGER NOT NULL,
                user_id INTEGER NOT NULL,
                role TEXT NOT NULL DEFAULT 'viewer',
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (team_id, user_id),
                FOREIGN KEY (team_id) REFERENCES teams (team_id) ON DELETE CASCADE,
                FOREIGN KEY (user_id) REFERENCES users (user_id) ON DELETE CASCADE
            );

            CREATE TABLE IF NOT EXISTS projects (
                project_id INTEGER PRIMARY KEY AUTOINCREMENT,
                team_id INTEGER NOT NULL,
                name TEXT NOT NULL,
                created_by_user_id INTEGER,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (team_id) REFERENCES teams (team_id) ON DELETE CASCADE,
                FOREIGN KEY (created_by_user_id) REFERENCES users (user_id) ON DELETE SET NULL,
                UNIQUE (team_id, name)
            );

            CREATE TABLE IF NOT EXISTS card_libraries (
                library_id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                team_id INTEGER,
                visibility TEXT NOT NULL DEFAULT 'private',
                name TEXT NOT NULL,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (team_id) REFERENCES teams (team_id) ON DELETE CASCADE,
                FOREIGN KEY (user_id) REFERENCES users (user_id) ON DELETE CASCADE,
                UNIQUE (user_id, team_id, name)
            );

            CREATE TABLE IF NOT EXISTS papers (
                paper_id TEXT PRIMARY KEY,
                file_name TEXT NOT NULL,
                file_size_bytes INTEGER NOT NULL,
                save_path TEXT NOT NULL,
                owner_user_id INTEGER,
                team_id INTEGER,
                project_id INTEGER,
                visibility TEXT NOT NULL DEFAULT 'team',
                file_sha256 TEXT NOT NULL DEFAULT '',
                parse_status TEXT NOT NULL DEFAULT 'succeeded',
                index_status TEXT NOT NULL DEFAULT 'unknown',
                translation_status TEXT NOT NULL DEFAULT 'not_started',
                parser TEXT NOT NULL DEFAULT '',
                markdown_path TEXT,
                translated_markdown_path TEXT,
                content_list_path TEXT,
                images_json TEXT NOT NULL DEFAULT '[]',
                page_count INTEGER NOT NULL DEFAULT 0,
                total_chars INTEGER NOT NULL DEFAULT 0,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (owner_user_id) REFERENCES users (user_id) ON DELETE SET NULL,
                FOREIGN KEY (team_id) REFERENCES teams (team_id) ON DELETE SET NULL,
                FOREIGN KEY (project_id) REFERENCES projects (project_id) ON DELETE SET NULL
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
                user_id INTEGER,
                team_id INTEGER,
                project_id INTEGER,
                question TEXT NOT NULL,
                answer TEXT NOT NULL,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users (user_id) ON DELETE SET NULL,
                FOREIGN KEY (team_id) REFERENCES teams (team_id) ON DELETE SET NULL,
                FOREIGN KEY (project_id) REFERENCES projects (project_id) ON DELETE SET NULL,
                FOREIGN KEY (paper_id) REFERENCES papers (paper_id) ON DELETE SET NULL
            );

            CREATE TABLE IF NOT EXISTS feedback (
                feedback_id INTEGER PRIMARY KEY AUTOINCREMENT,
                paper_id TEXT,
                chunk_id TEXT,
                qa_log_id INTEGER,
                user_id INTEGER,
                team_id INTEGER,
                project_id INTEGER,
                rating INTEGER,
                feedback_type TEXT,
                is_negative INTEGER NOT NULL DEFAULT 0,
                comment TEXT,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users (user_id) ON DELETE SET NULL,
                FOREIGN KEY (team_id) REFERENCES teams (team_id) ON DELETE SET NULL,
                FOREIGN KEY (project_id) REFERENCES projects (project_id) ON DELETE SET NULL,
                FOREIGN KEY (paper_id) REFERENCES papers (paper_id) ON DELETE SET NULL,
                FOREIGN KEY (chunk_id) REFERENCES chunks (chunk_id) ON DELETE SET NULL,
                FOREIGN KEY (qa_log_id) REFERENCES qa_logs (qa_log_id) ON DELETE SET NULL
            );

            CREATE TABLE IF NOT EXISTS bad_cases (
                bad_case_id INTEGER PRIMARY KEY AUTOINCREMENT,
                paper_id TEXT,
                user_id INTEGER,
                team_id INTEGER,
                project_id INTEGER,
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
                FOREIGN KEY (user_id) REFERENCES users (user_id) ON DELETE SET NULL,
                FOREIGN KEY (team_id) REFERENCES teams (team_id) ON DELETE SET NULL,
                FOREIGN KEY (project_id) REFERENCES projects (project_id) ON DELETE SET NULL,
                FOREIGN KEY (paper_id) REFERENCES papers (paper_id) ON DELETE SET NULL
            );

            CREATE TABLE IF NOT EXISTS literature_cards (
                card_id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER,
                library_id INTEGER,
                team_id INTEGER,
                project_id INTEGER,
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
                FOREIGN KEY (user_id) REFERENCES users (user_id) ON DELETE CASCADE,
                FOREIGN KEY (library_id) REFERENCES card_libraries (library_id) ON DELETE SET NULL,
                FOREIGN KEY (team_id) REFERENCES teams (team_id) ON DELETE SET NULL,
                FOREIGN KEY (project_id) REFERENCES projects (project_id) ON DELETE SET NULL,
                FOREIGN KEY (paper_id) REFERENCES papers (paper_id) ON DELETE CASCADE
            );

            CREATE TABLE IF NOT EXISTS jobs (
                job_id INTEGER PRIMARY KEY AUTOINCREMENT,
                job_type TEXT NOT NULL,
                status TEXT NOT NULL DEFAULT 'queued',
                paper_id TEXT,
                team_id INTEGER,
                project_id INTEGER,
                user_id INTEGER,
                payload_json TEXT NOT NULL DEFAULT '{}',
                result_json TEXT NOT NULL DEFAULT '{}',
                error_message TEXT NOT NULL DEFAULT '',
                attempt_count INTEGER NOT NULL DEFAULT 0,
                max_attempts INTEGER NOT NULL DEFAULT 3,
                locked_at TEXT,
                started_at TEXT,
                finished_at TEXT,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (paper_id) REFERENCES papers (paper_id) ON DELETE SET NULL,
                FOREIGN KEY (team_id) REFERENCES teams (team_id) ON DELETE SET NULL,
                FOREIGN KEY (project_id) REFERENCES projects (project_id) ON DELETE SET NULL,
                FOREIGN KEY (user_id) REFERENCES users (user_id) ON DELETE SET NULL
            );

            CREATE INDEX IF NOT EXISTS idx_card_libraries_user_id
                ON card_libraries (user_id);

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
        "card_libraries",
        {
            "team_id": "INTEGER",
            "visibility": "TEXT NOT NULL DEFAULT 'private'",
        },
    )
    ensure_columns(
        connection,
        "papers",
        {
            "owner_user_id": "INTEGER",
            "team_id": "INTEGER",
            "project_id": "INTEGER",
            "visibility": "TEXT NOT NULL DEFAULT 'team'",
            "file_sha256": "TEXT NOT NULL DEFAULT ''",
            "parse_status": "TEXT NOT NULL DEFAULT 'succeeded'",
            "index_status": "TEXT NOT NULL DEFAULT 'unknown'",
            "translation_status": "TEXT NOT NULL DEFAULT 'not_started'",
            "parser": "TEXT NOT NULL DEFAULT ''",
            "markdown_path": "TEXT",
            "translated_markdown_path": "TEXT",
            "content_list_path": "TEXT",
            "images_json": "TEXT NOT NULL DEFAULT '[]'",
            "updated_at": "TEXT",
        },
    )
    ensure_columns(
        connection,
        "qa_logs",
        {
            "user_id": "INTEGER",
            "team_id": "INTEGER",
            "project_id": "INTEGER",
        },
    )
    ensure_columns(
        connection,
        "feedback",
        {
            "user_id": "INTEGER",
            "team_id": "INTEGER",
            "project_id": "INTEGER",
            "feedback_type": "TEXT",
            "is_negative": "INTEGER NOT NULL DEFAULT 0",
        },
    )
    ensure_columns(
        connection,
        "bad_cases",
        {
            "user_id": "INTEGER",
            "team_id": "INTEGER",
            "project_id": "INTEGER",
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
            "user_id": "INTEGER",
            "library_id": "INTEGER",
            "team_id": "INTEGER",
            "project_id": "INTEGER",
            "title": "TEXT NOT NULL DEFAULT ''",
            "authors": "TEXT NOT NULL DEFAULT ''",
            "year": "TEXT NOT NULL DEFAULT ''",
            "research_field": "TEXT NOT NULL DEFAULT ''",
            "research_question": "TEXT NOT NULL DEFAULT ''",
            "method_summary": "TEXT NOT NULL DEFAULT ''",
            "datasets": "TEXT NOT NULL DEFAULT ''",
            "markdown": "TEXT NOT NULL DEFAULT ''",
            "updated_at": "TEXT",
        },
    )
    connection.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_card_libraries_user_id
            ON card_libraries (user_id)
        """
    )
    connection.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_card_libraries_team_id
            ON card_libraries (team_id)
        """
    )
    connection.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_literature_cards_user_library
            ON literature_cards (user_id, library_id, updated_at DESC)
        """
    )
    connection.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_literature_cards_team_project
            ON literature_cards (team_id, project_id, updated_at DESC)
        """
    )
    connection.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_papers_team_project
            ON papers (team_id, project_id, updated_at DESC)
        """
    )
    connection.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_papers_file_sha256
            ON papers (team_id, file_sha256)
        """
    )
    connection.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_jobs_status_created
            ON jobs (status, created_at)
        """
    )
    migrate_legacy_team_scope(connection)


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


def migrate_legacy_team_scope(connection: sqlite3.Connection) -> None:
    """Move pre-team local data into the first user's default team/project."""
    first_user = connection.execute(
        """
        SELECT user_id, username
        FROM users
        ORDER BY user_id ASC
        LIMIT 1
        """
    ).fetchone()
    if not first_user:
        return

    user_id = int(first_user["user_id"])
    username = str(first_user["username"])
    team = connection.execute(
        """
        SELECT t.team_id
        FROM teams t
        JOIN team_members m ON m.team_id = t.team_id
        WHERE m.user_id = ? AND m.role = 'owner'
        ORDER BY t.team_id ASC
        LIMIT 1
        """,
        (user_id,),
    ).fetchone()
    if team:
        team_id = int(team["team_id"])
    else:
        cursor = connection.execute(
            """
            INSERT INTO teams (name, owner_user_id)
            VALUES (?, ?)
            """,
            (f"{username} 的团队", user_id),
        )
        team_id = int(cursor.lastrowid)
        connection.execute(
            """
            INSERT OR IGNORE INTO team_members (team_id, user_id, role)
            VALUES (?, ?, 'owner')
            """,
            (team_id, user_id),
        )

    connection.execute(
        """
        INSERT OR IGNORE INTO team_members (team_id, user_id, role)
        VALUES (?, ?, 'owner')
        """,
        (team_id, user_id),
    )

    project = connection.execute(
        """
        SELECT project_id
        FROM projects
        WHERE team_id = ?
        ORDER BY project_id ASC
        LIMIT 1
        """,
        (team_id,),
    ).fetchone()
    if project:
        project_id = int(project["project_id"])
    else:
        cursor = connection.execute(
            """
            INSERT INTO projects (team_id, name, created_by_user_id)
            VALUES (?, '默认项目', ?)
            """,
            (team_id, user_id),
        )
        project_id = int(cursor.lastrowid)

    connection.execute(
        """
        UPDATE papers
        SET
            owner_user_id = COALESCE(owner_user_id, ?),
            team_id = COALESCE(team_id, ?),
            project_id = COALESCE(project_id, ?),
            visibility = COALESCE(NULLIF(visibility, ''), 'team'),
            parse_status = COALESCE(NULLIF(parse_status, ''), 'succeeded'),
            index_status = COALESCE(NULLIF(index_status, ''), 'unknown'),
            translation_status = COALESCE(NULLIF(translation_status, ''), 'not_started'),
            updated_at = COALESCE(updated_at, created_at, CURRENT_TIMESTAMP)
        WHERE team_id IS NULL
        """,
        (user_id, team_id, project_id),
    )
    connection.execute(
        """
        UPDATE card_libraries
        SET team_id = COALESCE(team_id, ?)
        WHERE team_id IS NULL
        """,
        (team_id,),
    )
    connection.execute(
        """
        UPDATE literature_cards
        SET
            user_id = COALESCE(user_id, ?),
            team_id = COALESCE(
                team_id,
                (SELECT team_id FROM papers WHERE papers.paper_id = literature_cards.paper_id),
                ?
            ),
            project_id = COALESCE(
                project_id,
                (SELECT project_id FROM papers WHERE papers.paper_id = literature_cards.paper_id),
                ?
            )
        WHERE team_id IS NULL
        """,
        (user_id, team_id, project_id),
    )
    for table_name in ("qa_logs", "feedback", "bad_cases"):
        connection.execute(
            f"""
            UPDATE {table_name}
            SET
                user_id = COALESCE(user_id, ?),
                team_id = COALESCE(
                    team_id,
                    (SELECT team_id FROM papers WHERE papers.paper_id = {table_name}.paper_id),
                    ?
                ),
                project_id = COALESCE(
                    project_id,
                    (SELECT project_id FROM papers WHERE papers.paper_id = {table_name}.paper_id),
                    ?
                )
            WHERE team_id IS NULL
            """,
            (user_id, team_id, project_id),
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
            INSERT INTO papers (
                paper_id,
                file_name,
                file_size_bytes,
                save_path,
                owner_user_id,
                team_id,
                project_id,
                visibility,
                file_sha256,
                parse_status,
                index_status,
                translation_status,
                parser,
                markdown_path,
                translated_markdown_path,
                content_list_path,
                images_json,
                page_count,
                total_chars,
                updated_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
            ON CONFLICT(paper_id) DO UPDATE SET
                file_name = excluded.file_name,
                file_size_bytes = excluded.file_size_bytes,
                save_path = excluded.save_path,
                owner_user_id = COALESCE(excluded.owner_user_id, papers.owner_user_id),
                team_id = COALESCE(excluded.team_id, papers.team_id),
                project_id = COALESCE(excluded.project_id, papers.project_id),
                visibility = excluded.visibility,
                file_sha256 = excluded.file_sha256,
                parse_status = excluded.parse_status,
                index_status = excluded.index_status,
                translation_status = excluded.translation_status,
                parser = excluded.parser,
                markdown_path = excluded.markdown_path,
                translated_markdown_path = excluded.translated_markdown_path,
                content_list_path = excluded.content_list_path,
                images_json = excluded.images_json,
                page_count = excluded.page_count,
                total_chars = excluded.total_chars,
                updated_at = CURRENT_TIMESTAMP
            """,
            (
                paper["paper_id"],
                paper["file_name"],
                paper["file_size_bytes"],
                paper["save_path"],
                paper.get("owner_user_id"),
                paper.get("team_id"),
                paper.get("project_id"),
                paper.get("visibility", "team"),
                paper.get("file_sha256", ""),
                paper.get("parse_status", "succeeded"),
                paper.get("index_status", "unknown"),
                paper.get("translation_status", "not_started"),
                paper.get("parser", ""),
                paper.get("markdown_path"),
                paper.get("translated_markdown_path"),
                paper.get("content_list_path"),
                paper.get("images_json")
                if isinstance(paper.get("images_json"), str)
                else json.dumps(paper.get("images", []), ensure_ascii=False),
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


def save_qa_log(
    paper_id: str,
    question: str,
    answer: str,
    user_id: int | None = None,
    team_id: int | None = None,
    project_id: int | None = None,
) -> int:
    """Persist one question-answer record and return its id."""
    init_db()

    with get_db_connection() as connection:
        if team_id is None or project_id is None:
            row = connection.execute(
                """
                SELECT team_id, project_id
                FROM papers
                WHERE paper_id = ?
                """,
                (paper_id,),
            ).fetchone()
            if row:
                team_id = team_id if team_id is not None else row["team_id"]
                project_id = project_id if project_id is not None else row["project_id"]
        cursor = connection.execute(
            """
            INSERT INTO qa_logs (paper_id, user_id, team_id, project_id, question, answer)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (paper_id, user_id, team_id, project_id, question, answer),
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
