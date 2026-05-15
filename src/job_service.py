"""SQLite-backed background job helpers."""

from __future__ import annotations

import json
from typing import Any

from src.db import get_db_connection, init_db
from src.team_service import require_team_role


JOB_TYPES = {"parse", "index", "translate", "card", "eval"}
JOB_STATUSES = {"queued", "running", "succeeded", "failed", "canceled"}


def enqueue_job(
    job_type: str,
    user_id: int,
    team_id: int,
    project_id: int | None = None,
    paper_id: str | None = None,
    payload: dict[str, Any] | None = None,
    max_attempts: int = 3,
) -> int:
    """Create a queued job."""
    clean_type = normalize_job_type(job_type)
    require_team_role(user_id, team_id, "editor")
    init_db()
    with get_db_connection() as connection:
        cursor = connection.execute(
            """
            INSERT INTO jobs (
                job_type,
                status,
                paper_id,
                team_id,
                project_id,
                user_id,
                payload_json,
                max_attempts
            )
            VALUES (?, 'queued', ?, ?, ?, ?, ?, ?)
            """,
            (
                clean_type,
                paper_id,
                int(team_id),
                project_id,
                int(user_id),
                json.dumps(payload or {}, ensure_ascii=False),
                max(1, int(max_attempts)),
            ),
        )
        return int(cursor.lastrowid)


def list_jobs(
    user_id: int,
    team_id: int,
    paper_id: str | None = None,
    limit: int = 50,
) -> list[dict[str, Any]]:
    """Return recent jobs visible to a team member."""
    require_team_role(user_id, team_id, "viewer")
    where_clauses = ["j.team_id = ?"]
    parameters: list[Any] = [int(team_id)]
    if paper_id:
        where_clauses.append("j.paper_id = ?")
        parameters.append(paper_id)
    parameters.append(max(1, int(limit)))
    init_db()
    with get_db_connection() as connection:
        rows = connection.execute(
            f"""
            SELECT
                j.*,
                p.file_name,
                u.username
            FROM jobs j
            LEFT JOIN papers p ON p.paper_id = j.paper_id
            LEFT JOIN users u ON u.user_id = j.user_id
            WHERE {' AND '.join(where_clauses)}
            ORDER BY j.created_at DESC, j.job_id DESC
            LIMIT ?
            """,
            parameters,
        ).fetchall()
    return [normalize_job_row(dict(row)) for row in rows]


def latest_job_for_paper(team_id: int, paper_id: str, job_type: str | None = None) -> dict[str, Any] | None:
    """Return latest job for one paper."""
    where_clauses = ["team_id = ?", "paper_id = ?"]
    parameters: list[Any] = [int(team_id), paper_id]
    if job_type:
        where_clauses.append("job_type = ?")
        parameters.append(normalize_job_type(job_type))
    init_db()
    with get_db_connection() as connection:
        row = connection.execute(
            f"""
            SELECT *
            FROM jobs
            WHERE {' AND '.join(where_clauses)}
            ORDER BY created_at DESC, job_id DESC
            LIMIT 1
            """,
            parameters,
        ).fetchone()
    return normalize_job_row(dict(row)) if row else None


def claim_next_job() -> dict[str, Any] | None:
    """Atomically claim the oldest queued job."""
    init_db()
    connection = get_db_connection()
    try:
        connection.execute("BEGIN IMMEDIATE")
        row = connection.execute(
            """
            SELECT *
            FROM jobs
            WHERE status = 'queued'
                AND attempt_count < max_attempts
            ORDER BY created_at ASC, job_id ASC
            LIMIT 1
            """
        ).fetchone()
        if not row:
            connection.commit()
            return None
        job_id = int(row["job_id"])
        connection.execute(
            """
            UPDATE jobs
            SET
                status = 'running',
                attempt_count = attempt_count + 1,
                locked_at = CURRENT_TIMESTAMP,
                started_at = COALESCE(started_at, CURRENT_TIMESTAMP),
                updated_at = CURRENT_TIMESTAMP
            WHERE job_id = ?
            """,
            (job_id,),
        )
        updated = connection.execute(
            "SELECT * FROM jobs WHERE job_id = ?",
            (job_id,),
        ).fetchone()
        connection.commit()
        return normalize_job_row(dict(updated)) if updated else None
    except Exception:
        connection.rollback()
        raise
    finally:
        connection.close()


def complete_job(job_id: int, result: dict[str, Any] | None = None) -> None:
    """Mark a job succeeded."""
    init_db()
    with get_db_connection() as connection:
        connection.execute(
            """
            UPDATE jobs
            SET
                status = 'succeeded',
                result_json = ?,
                error_message = '',
                finished_at = CURRENT_TIMESTAMP,
                updated_at = CURRENT_TIMESTAMP
            WHERE job_id = ?
            """,
            (json.dumps(result or {}, ensure_ascii=False), int(job_id)),
        )


def fail_job(job_id: int, error_message: str, result: dict[str, Any] | None = None) -> None:
    """Mark a job failed."""
    init_db()
    with get_db_connection() as connection:
        connection.execute(
            """
            UPDATE jobs
            SET
                status = 'failed',
                result_json = ?,
                error_message = ?,
                finished_at = CURRENT_TIMESTAMP,
                updated_at = CURRENT_TIMESTAMP
            WHERE job_id = ?
            """,
            (
                json.dumps(result or {}, ensure_ascii=False),
                str(error_message or "")[:4000],
                int(job_id),
            ),
        )


def retry_job(user_id: int, job_id: int) -> None:
    """Retry a failed or canceled job if requester can edit the team."""
    job = get_job(job_id)
    if not job:
        raise ValueError("没有找到这个任务。")
    require_team_role(user_id, int(job["team_id"]), "editor")
    with get_db_connection() as connection:
        connection.execute(
            """
            UPDATE jobs
            SET
                status = 'queued',
                error_message = '',
                finished_at = NULL,
                updated_at = CURRENT_TIMESTAMP
            WHERE job_id = ?
                AND status IN ('failed', 'canceled')
                AND attempt_count < max_attempts
            """,
            (int(job_id),),
        )


def cancel_job(user_id: int, job_id: int) -> None:
    """Cancel a queued or running job."""
    job = get_job(job_id)
    if not job:
        raise ValueError("没有找到这个任务。")
    require_team_role(user_id, int(job["team_id"]), "editor")
    with get_db_connection() as connection:
        connection.execute(
            """
            UPDATE jobs
            SET
                status = 'canceled',
                finished_at = CURRENT_TIMESTAMP,
                updated_at = CURRENT_TIMESTAMP
            WHERE job_id = ?
                AND status IN ('queued', 'running')
            """,
            (int(job_id),),
        )


def get_job(job_id: int) -> dict[str, Any] | None:
    """Return a job by id."""
    init_db()
    with get_db_connection() as connection:
        row = connection.execute(
            "SELECT * FROM jobs WHERE job_id = ?",
            (int(job_id),),
        ).fetchone()
    return normalize_job_row(dict(row)) if row else None


def normalize_job_type(job_type: str) -> str:
    """Validate a job type."""
    clean_type = str(job_type or "").strip().lower()
    if clean_type not in JOB_TYPES:
        raise ValueError("不支持的任务类型。")
    return clean_type


def normalize_job_row(row: dict[str, Any]) -> dict[str, Any]:
    """Parse JSON fields on a job row."""
    item = dict(row)
    item["payload"] = parse_json_field(item.get("payload_json"))
    item["result"] = parse_json_field(item.get("result_json"))
    return item


def parse_json_field(value: Any) -> dict[str, Any]:
    """Parse a JSON object field."""
    if isinstance(value, dict):
        return value
    try:
        parsed = json.loads(str(value or "{}"))
    except (TypeError, ValueError):
        return {}
    return parsed if isinstance(parsed, dict) else {}
