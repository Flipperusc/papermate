"""Smoke checks for PaperMate team schema migration."""

from __future__ import annotations

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.db import get_db_connection, init_db


REQUIRED_COLUMNS = {
    "papers": {
        "owner_user_id",
        "team_id",
        "project_id",
        "visibility",
        "file_sha256",
        "parse_status",
        "index_status",
        "translation_status",
    },
    "qa_logs": {"user_id", "team_id", "project_id"},
    "feedback": {"user_id", "team_id", "project_id"},
    "bad_cases": {"user_id", "team_id", "project_id"},
    "literature_cards": {"team_id", "project_id"},
    "card_libraries": {"team_id", "visibility"},
    "jobs": {"job_type", "status", "payload_json", "result_json", "attempt_count"},
}


def table_columns(table_name: str) -> set[str]:
    """Return column names for a SQLite table."""
    with get_db_connection() as connection:
        rows = connection.execute(f"PRAGMA table_info({table_name})").fetchall()
    return {str(row["name"]) for row in rows}


def unique_index_columns(table_name: str) -> list[list[str]]:
    """Return unique index column lists for a SQLite table."""
    with get_db_connection() as connection:
        indexes = [
            dict(row)
            for row in connection.execute(f"PRAGMA index_list({table_name})").fetchall()
            if int(row["unique"]) == 1
        ]
        return [
            [
                str(info["name"])
                for info in connection.execute(f"PRAGMA index_info({index['name']})").fetchall()
            ]
            for index in indexes
        ]


def main() -> None:
    init_db()
    for table_name, expected_columns in REQUIRED_COLUMNS.items():
        missing = expected_columns - table_columns(table_name)
        assert not missing, f"{table_name} missing columns: {sorted(missing)}"

    card_library_unique_indexes = unique_index_columns("card_libraries")
    assert ["user_id", "team_id", "name"] in card_library_unique_indexes, (
        "card_libraries should be unique per user/team/name, not globally per user/name"
    )
    assert ["user_id", "name"] not in card_library_unique_indexes, (
        "legacy card_libraries UNIQUE(user_id, name) should be migrated away"
    )

    with get_db_connection() as connection:
        user_count = connection.execute("SELECT COUNT(*) FROM users").fetchone()[0]
        if int(user_count) > 0:
            team_count = connection.execute("SELECT COUNT(*) FROM teams").fetchone()[0]
            project_count = connection.execute("SELECT COUNT(*) FROM projects").fetchone()[0]
            assert int(team_count) > 0, "existing users should have a migrated default team"
            assert int(project_count) > 0, "existing users should have a migrated default project"
    print("team schema migration tests passed")


if __name__ == "__main__":
    main()
