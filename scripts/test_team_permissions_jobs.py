"""Smoke tests for team permissions and SQLite jobs."""

from __future__ import annotations

import sys
from pathlib import Path
from uuid import uuid4


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.auth_service import create_user
from src.db import get_db_connection, init_db
from src.job_service import cancel_job, enqueue_job, get_job, list_jobs, retry_job
from src.team_service import (
    add_team_member_by_username,
    create_project,
    ensure_user_workspace,
    get_user_team_role,
    require_team_role,
    update_team_member_role,
)


def main() -> None:
    suffix = uuid4().hex[:10]
    owner = create_user(f"pm_owner_{suffix}", "password123")
    member = create_user(f"pm_member_{suffix}", "password123")
    job_id: int | None = None

    try:
        workspace = ensure_user_workspace(int(owner["user_id"]))
        team_id = int(workspace["team_id"])
        assert get_user_team_role(int(owner["user_id"]), team_id) == "owner"

        added = add_team_member_by_username(
            int(owner["user_id"]),
            team_id,
            str(member["username"]),
            "viewer",
        )
        assert added["role"] == "viewer"
        assert require_team_role(int(member["user_id"]), team_id, "viewer") == "viewer"
        try:
            require_team_role(int(member["user_id"]), team_id, "editor")
        except PermissionError:
            pass
        else:
            raise AssertionError("viewer should not satisfy editor permission")

        update_team_member_role(int(owner["user_id"]), team_id, int(member["user_id"]), "editor")
        assert require_team_role(int(member["user_id"]), team_id, "editor") == "editor"

        project = create_project(int(owner["user_id"]), team_id, f"项目 {suffix}")
        job_id = enqueue_job(
            "eval",
            user_id=int(member["user_id"]),
            team_id=team_id,
            project_id=int(project["project_id"]),
            payload={"case": "permission-smoke"},
        )
        jobs = list_jobs(int(owner["user_id"]), team_id)
        assert any(int(job["job_id"]) == job_id for job in jobs)
        cancel_job(int(member["user_id"]), job_id)
        assert get_job(job_id)["status"] == "canceled"
        retry_job(int(member["user_id"]), job_id)
        assert get_job(job_id)["status"] == "queued"
        print("team permissions and jobs tests passed")
    finally:
        cleanup_users([int(owner["user_id"]), int(member["user_id"])])


def cleanup_users(user_ids: list[int]) -> None:
    """Remove test users and their team data from the local runtime DB."""
    placeholders = ",".join("?" for _ in user_ids)
    with get_db_connection() as connection:
        connection.execute(f"DELETE FROM jobs WHERE user_id IN ({placeholders})", user_ids)
        connection.execute(f"DELETE FROM teams WHERE owner_user_id IN ({placeholders})", user_ids)
        connection.execute(f"DELETE FROM users WHERE user_id IN ({placeholders})", user_ids)


if __name__ == "__main__":
    init_db()
    main()
