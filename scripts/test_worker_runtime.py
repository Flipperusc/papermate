"""Smoke tests for worker runtime lease and retry behavior."""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path
from uuid import uuid4


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts import worker
from src import db as db_module
from src.auth_service import create_user
from src.db import get_db_connection, init_db
from src.job_service import enqueue_job, get_job
from src.team_service import ensure_user_workspace


def main() -> None:
    test_worker_completes_eval_job()
    test_worker_retries_failed_job_with_backoff()
    print("worker runtime tests passed")


def test_worker_completes_eval_job() -> None:
    with isolated_runtime():
        user_id, team_id, project_id = create_workspace_user("worker_ok")
        job_id = enqueue_job(
            "eval",
            user_id=user_id,
            team_id=team_id,
            project_id=project_id,
            payload={"case": "worker-success"},
        )

        completed = worker.run_worker(once=True, job_types=("eval",), lease_seconds=30)
        assert completed == 1
        job = get_job(job_id)
        assert job["status"] == "succeeded"
        assert job["worker_id"] is None
        assert job["lease_expires_at"] is None


def test_worker_retries_failed_job_with_backoff() -> None:
    original_run_job = worker.run_job
    with isolated_runtime():
        user_id, team_id, project_id = create_workspace_user("worker_retry")
        job_id = enqueue_job(
            "eval",
            user_id=user_id,
            team_id=team_id,
            project_id=project_id,
            payload={"case": "worker-retry"},
            max_attempts=2,
        )

        calls = {"count": 0}

        def flaky_run_job(job: dict) -> dict:
            calls["count"] += 1
            if calls["count"] == 1:
                raise RuntimeError("temporary test failure")
            return {"ok": True}

        try:
            worker.run_job = flaky_run_job
            completed = worker.run_worker(once=True, job_types=("eval",), lease_seconds=30)
            assert completed == 1
            retried_job = get_job(job_id)
            assert retried_job["status"] == "queued"
            assert retried_job["attempt_count"] == 1
            assert retried_job["next_run_at"]
            assert retried_job["worker_id"] is None

            with get_db_connection() as connection:
                connection.execute(
                    """
                    UPDATE jobs
                    SET next_run_at = datetime(CURRENT_TIMESTAMP, '-1 seconds')
                    WHERE job_id = ?
                    """,
                    (job_id,),
                )

            completed = worker.run_worker(once=True, job_types=("eval",), lease_seconds=30)
            assert completed == 1
            finished_job = get_job(job_id)
            assert finished_job["status"] == "succeeded"
            assert finished_job["attempt_count"] == 2
            assert finished_job["next_run_at"] is None
        finally:
            worker.run_job = original_run_job


class isolated_runtime:
    """Temporarily redirect the database to a private test file."""

    def __enter__(self) -> Path:
        self.tmp = tempfile.TemporaryDirectory()
        self.original_db_path = db_module.settings.db_path
        self.db_path = Path(self.tmp.name) / "papermate-worker-test.db"
        object.__setattr__(db_module.settings, "db_path", self.db_path)
        init_db(force=True)
        return self.db_path

    def __exit__(self, exc_type, exc_value, traceback) -> bool:
        object.__setattr__(db_module.settings, "db_path", self.original_db_path)
        self.tmp.cleanup()
        return False


def create_workspace_user(label: str) -> tuple[int, int, int]:
    suffix = uuid4().hex[:10]
    user = create_user(f"pm_{label}_{suffix}", "password123")
    workspace = ensure_user_workspace(int(user["user_id"]))
    return int(user["user_id"]), int(workspace["team_id"]), int(workspace["project_id"])


if __name__ == "__main__":
    main()

