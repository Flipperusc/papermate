"""Smoke tests for team permissions and SQLite jobs."""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path
from uuid import uuid4


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.auth_service import create_user
from src import db as db_module
from src.db import get_db_connection, init_db
from src.job_service import (
    cancel_job,
    clear_team_queued_jobs,
    claim_next_job,
    enqueue_job,
    get_job,
    list_jobs,
    queue_progress_summary,
    requeue_running_jobs,
    retry_job,
)
from src.team_service import (
    add_team_member_by_username,
    create_project,
    ensure_user_workspace,
    get_user_team_role,
    require_team_role,
    update_team_member_role,
)


def main() -> None:
    test_init_db_cached_wal()
    test_claim_next_job_type_lanes_and_paper_mutex()
    test_requeue_running_jobs_restores_paper_status()
    test_clear_team_queued_jobs_restores_manual_status()
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
        parse_job_id = enqueue_job(
            "parse",
            user_id=int(member["user_id"]),
            team_id=team_id,
            project_id=int(project["project_id"]),
            payload={"case": "queue-progress"},
        )
        index_job_id = enqueue_job(
            "index",
            user_id=int(member["user_id"]),
            team_id=team_id,
            project_id=int(project["project_id"]),
            payload={"case": "queue-progress"},
        )
        with get_db_connection() as connection:
            connection.execute(
                """
                UPDATE jobs
                SET status = 'running',
                    started_at = CURRENT_TIMESTAMP,
                    updated_at = CURRENT_TIMESTAMP
                WHERE job_id = ?
                """,
                (parse_job_id,),
            )
        queue_summary = queue_progress_summary(int(owner["user_id"]), team_id)
        assert queue_summary["running_count"] >= 1
        assert int(queue_summary["running_by_type"]["parse"]["job_id"]) == parse_job_id
        assert int(queue_summary["queued_by_type"]["index"]["job_id"]) == index_job_id
        assert any(int(job["job_id"]) == index_job_id for job in queue_summary["queued"])
        print("team permissions and jobs tests passed")
    finally:
        cleanup_users([int(owner["user_id"]), int(member["user_id"])])


def test_init_db_cached_wal() -> None:
    """Verify DB initialization is cached per path and enables WAL."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        db_path = Path(tmp_dir) / "papermate-test.db"
        init_db(db_path, force=True)
        key = db_module.db_init_key(db_path)
        assert key in db_module._INITIALIZED_DB_PATHS
        initialized_count = len(db_module._INITIALIZED_DB_PATHS)
        init_db(db_path)
        assert len(db_module._INITIALIZED_DB_PATHS) == initialized_count
        with get_db_connection(db_path) as connection:
            journal_mode = str(connection.execute("PRAGMA journal_mode").fetchone()[0]).lower()
            assert journal_mode == "wal"


def test_claim_next_job_type_lanes_and_paper_mutex() -> None:
    """Verify parse and index lanes can run different papers but not the same paper."""
    original_db_path = db_module.settings.db_path
    with tempfile.TemporaryDirectory() as tmp_dir:
        temp_db_path = Path(tmp_dir) / "papermate-claim-test.db"
        object.__setattr__(db_module.settings, "db_path", temp_db_path)
        try:
            init_db(force=True)
            suffix = uuid4().hex[:10]
            owner = create_user(f"pm_claim_owner_{suffix}", "password123")
            workspace = ensure_user_workspace(int(owner["user_id"]))
            team_id = int(workspace["team_id"])
            project_id = int(workspace["project_id"])
            paper_a = f"paper-a-{suffix}"
            paper_b = f"paper-b-{suffix}"
            paper_c = f"paper-c-{suffix}"
            paper_d = f"paper-d-{suffix}"
            insert_test_paper(paper_a, int(owner["user_id"]), team_id, project_id)
            insert_test_paper(paper_b, int(owner["user_id"]), team_id, project_id)
            insert_test_paper(paper_c, int(owner["user_id"]), team_id, project_id, parse_status="running")
            insert_test_paper(paper_d, int(owner["user_id"]), team_id, project_id)
            parse_a_id = enqueue_job(
                "parse",
                user_id=int(owner["user_id"]),
                team_id=team_id,
                project_id=project_id,
                paper_id=paper_a,
                payload={"paper_id": paper_a},
            )
            index_a_id = enqueue_job(
                "index",
                user_id=int(owner["user_id"]),
                team_id=team_id,
                project_id=project_id,
                paper_id=paper_a,
                payload={"paper_id": paper_a},
            )
            index_b_id = enqueue_job(
                "index",
                user_id=int(owner["user_id"]),
                team_id=team_id,
                project_id=project_id,
                paper_id=paper_b,
                payload={"paper_id": paper_b},
            )

            claimed_parse = claim_next_job(job_types=("parse",))
            assert claimed_parse is not None
            assert int(claimed_parse["job_id"]) == parse_a_id
            assert claimed_parse["status"] == "running"

            claimed_index = claim_next_job(job_types=("index",))
            assert claimed_index is not None
            assert int(claimed_index["job_id"]) == index_b_id
            assert claimed_index["paper_id"] == paper_b
            assert get_job(index_a_id)["status"] == "queued"
            index_c_id = enqueue_job(
                "index",
                user_id=int(owner["user_id"]),
                team_id=team_id,
                project_id=project_id,
                paper_id=paper_c,
                payload={"paper_id": paper_c},
            )
            index_d_id = enqueue_job(
                "index",
                user_id=int(owner["user_id"]),
                team_id=team_id,
                project_id=project_id,
                paper_id=paper_d,
                payload={"paper_id": paper_d},
            )
            lane_summary = queue_progress_summary(int(owner["user_id"]), team_id)
            assert int(lane_summary["blocked_by_type"]["index"]["job_id"]) in {index_a_id, index_c_id}
            assert int(lane_summary["queued_by_type"]["index"]["job_id"]) == index_d_id
            claimed_next_index = claim_next_job(job_types=("index",))
            assert claimed_next_index is not None
            assert int(claimed_next_index["job_id"]) == index_d_id
            assert get_job(index_c_id)["status"] == "queued"
            assert claim_next_job(job_types=("index",)) is None
        finally:
            object.__setattr__(db_module.settings, "db_path", original_db_path)


def test_requeue_running_jobs_restores_paper_status() -> None:
    """Verify interrupted running worker jobs can be safely returned to the queue."""
    original_db_path = db_module.settings.db_path
    with tempfile.TemporaryDirectory() as tmp_dir:
        temp_db_path = Path(tmp_dir) / "papermate-requeue-test.db"
        object.__setattr__(db_module.settings, "db_path", temp_db_path)
        try:
            init_db(force=True)
            suffix = uuid4().hex[:10]
            owner = create_user(f"pm_requeue_owner_{suffix}", "password123")
            workspace = ensure_user_workspace(int(owner["user_id"]))
            team_id = int(workspace["team_id"])
            project_id = int(workspace["project_id"])
            paper_id = f"paper-requeue-{suffix}"
            insert_test_paper(paper_id, int(owner["user_id"]), team_id, project_id, parse_status="running")
            with get_db_connection() as connection:
                connection.execute(
                    "UPDATE papers SET index_status = 'running' WHERE paper_id = ?",
                    (paper_id,),
                )
            parse_job_id = enqueue_job(
                "parse",
                user_id=int(owner["user_id"]),
                team_id=team_id,
                project_id=project_id,
                paper_id=paper_id,
                payload={"paper_id": paper_id},
            )
            index_job_id = enqueue_job(
                "index",
                user_id=int(owner["user_id"]),
                team_id=team_id,
                project_id=project_id,
                paper_id=paper_id,
                payload={"paper_id": paper_id},
            )
            with get_db_connection() as connection:
                connection.execute(
                    """
                    UPDATE jobs
                    SET status = 'running',
                        locked_at = CURRENT_TIMESTAMP,
                        started_at = CURRENT_TIMESTAMP,
                        updated_at = CURRENT_TIMESTAMP
                    WHERE job_id IN (?, ?)
                    """,
                    (parse_job_id, index_job_id),
                )

            recovered = requeue_running_jobs()
            recovered_ids = {int(job["job_id"]) for job in recovered}
            assert {parse_job_id, index_job_id}.issubset(recovered_ids)
            assert get_job(parse_job_id)["status"] == "queued"
            assert get_job(index_job_id)["status"] == "queued"
            with get_db_connection() as connection:
                paper = connection.execute(
                    "SELECT parse_status, index_status FROM papers WHERE paper_id = ?",
                    (paper_id,),
                ).fetchone()
            assert paper["parse_status"] == "queued"
            assert paper["index_status"] == "queued"
        finally:
            object.__setattr__(db_module.settings, "db_path", original_db_path)


def test_clear_team_queued_jobs_restores_manual_status() -> None:
    """Verify UI queue clearing cancels queued work and makes papers manually selectable."""
    original_db_path = db_module.settings.db_path
    with tempfile.TemporaryDirectory() as tmp_dir:
        temp_db_path = Path(tmp_dir) / "papermate-clear-queue-test.db"
        object.__setattr__(db_module.settings, "db_path", temp_db_path)
        try:
            init_db(force=True)
            suffix = uuid4().hex[:10]
            owner = create_user(f"pm_clear_owner_{suffix}", "password123")
            workspace = ensure_user_workspace(int(owner["user_id"]))
            team_id = int(workspace["team_id"])
            project_id = int(workspace["project_id"])
            parse_paper_id = f"paper-clear-parse-{suffix}"
            index_paper_id = f"paper-clear-index-{suffix}"
            insert_test_paper(parse_paper_id, int(owner["user_id"]), team_id, project_id, parse_status="queued")
            insert_test_paper(index_paper_id, int(owner["user_id"]), team_id, project_id, parse_status="succeeded")
            parse_job_id = enqueue_job(
                "parse",
                user_id=int(owner["user_id"]),
                team_id=team_id,
                project_id=project_id,
                paper_id=parse_paper_id,
                payload={"paper_id": parse_paper_id},
            )
            index_job_id = enqueue_job(
                "index",
                user_id=int(owner["user_id"]),
                team_id=team_id,
                project_id=project_id,
                paper_id=index_paper_id,
                payload={"paper_id": index_paper_id},
            )

            result = clear_team_queued_jobs(
                int(owner["user_id"]),
                team_id,
                job_types=("parse", "index"),
                reason="test clear",
            )
            assert result["cleared_count"] == 2
            assert get_job(parse_job_id)["status"] == "canceled"
            assert get_job(index_job_id)["status"] == "canceled"
            with get_db_connection() as connection:
                parse_paper = connection.execute(
                    "SELECT parse_status, index_status FROM papers WHERE paper_id = ?",
                    (parse_paper_id,),
                ).fetchone()
                index_paper = connection.execute(
                    "SELECT parse_status, index_status FROM papers WHERE paper_id = ?",
                    (index_paper_id,),
                ).fetchone()
            assert parse_paper["parse_status"] == "not_started"
            assert parse_paper["index_status"] == "unknown"
            assert index_paper["parse_status"] == "succeeded"
            assert index_paper["index_status"] == "unknown"
        finally:
            object.__setattr__(db_module.settings, "db_path", original_db_path)


def insert_test_paper(
    paper_id: str,
    owner_user_id: int,
    team_id: int,
    project_id: int,
    parse_status: str = "succeeded",
) -> None:
    """Insert a minimal paper row for job FK tests."""
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
                index_status
            )
            VALUES (?, ?, 1, ?, ?, ?, ?, 'team', ?, ?, 'queued')
            """,
            (
                paper_id,
                f"{paper_id}.pdf",
                str(Path("data/uploads") / f"{paper_id}.pdf"),
                int(owner_user_id),
                int(team_id),
                int(project_id),
                paper_id,
                parse_status,
            ),
        )


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
