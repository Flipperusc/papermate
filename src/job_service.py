"""SQLite-backed background job helpers."""

from __future__ import annotations

import json
from typing import Any

from src.db import get_db_connection, init_db
from src.team_service import require_team_role


JOB_TYPES = {"parse", "index", "translate", "card", "eval"}
JOB_STATUSES = {"queued", "running", "succeeded", "failed", "canceled"}
DEFAULT_JOB_LEASE_SECONDS = 300
DEFAULT_RETRY_BASE_DELAY_SECONDS = 30
DEFAULT_RETRY_MAX_DELAY_SECONDS = 900


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
                p.parse_status AS paper_parse_status,
                p.index_status AS paper_index_status,
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


def queue_progress_summary(
    user_id: int,
    team_id: int,
    job_types: tuple[str, ...] = ("parse", "index"),
    queued_limit: int = 10,
) -> dict[str, Any]:
    """Return lightweight queue activity for the global UI progress bar."""
    require_team_role(user_id, team_id, "viewer")
    clean_types = tuple(normalize_job_type(job_type) for job_type in job_types)
    placeholders = ", ".join("?" for _ in clean_types)
    init_db()
    with get_db_connection() as connection:
        running_rows = connection.execute(
            f"""
            SELECT
                j.*,
                p.file_name,
                p.parse_status AS paper_parse_status,
                p.index_status AS paper_index_status,
                u.username
            FROM jobs j
            LEFT JOIN papers p ON p.paper_id = j.paper_id
            LEFT JOIN users u ON u.user_id = j.user_id
            WHERE j.team_id = ?
                AND j.job_type IN ({placeholders})
                AND j.status = 'running'
            ORDER BY COALESCE(j.started_at, j.locked_at, j.updated_at, j.created_at) ASC, j.job_id ASC
            """,
            [int(team_id), *clean_types],
        ).fetchall()
        queued_rows = connection.execute(
            f"""
            SELECT
                j.*,
                p.file_name,
                p.parse_status AS paper_parse_status,
                p.index_status AS paper_index_status,
                u.username
            FROM jobs j
            LEFT JOIN papers p ON p.paper_id = j.paper_id
            LEFT JOIN users u ON u.user_id = j.user_id
            WHERE j.team_id = ?
                AND j.job_type IN ({placeholders})
                AND j.status = 'queued'
            ORDER BY j.created_at ASC, j.job_id ASC
            LIMIT ?
            """,
            [int(team_id), *clean_types, max(1, int(queued_limit))],
        ).fetchall()
        queued_candidate_rows = connection.execute(
            f"""
            SELECT
                j.*,
                p.file_name,
                p.parse_status AS paper_parse_status,
                p.index_status AS paper_index_status,
                u.username
            FROM jobs j
            LEFT JOIN papers p ON p.paper_id = j.paper_id
            LEFT JOIN users u ON u.user_id = j.user_id
            WHERE j.team_id = ?
                AND j.job_type IN ({placeholders})
                AND j.status = 'queued'
            ORDER BY j.created_at ASC, j.job_id ASC
            LIMIT ?
            """,
            [int(team_id), *clean_types, max(100, int(queued_limit))],
        ).fetchall()
        count_rows = connection.execute(
            f"""
            SELECT status, COUNT(*) AS count
            FROM jobs
            WHERE team_id = ?
                AND job_type IN ({placeholders})
                AND status IN ('queued', 'running')
            GROUP BY status
            """,
            [int(team_id), *clean_types],
        ).fetchall()

    running = [normalize_job_row(dict(row)) for row in running_rows]
    queued = [normalize_job_row(dict(row)) for row in queued_rows]
    queued_candidates = [normalize_job_row(dict(row)) for row in queued_candidate_rows]
    counts = {str(row["status"]): int(row["count"]) for row in count_rows}
    running_by_type: dict[str, dict[str, Any]] = {}
    for job in running:
        running_by_type.setdefault(str(job["job_type"]), job)
    running_parse_index_papers = {
        str(job.get("paper_id"))
        for job in running
        if job.get("paper_id") and str(job.get("job_type") or "") in {"parse", "index"}
    }

    def mark_queue_block_reason(job: dict[str, Any]) -> dict[str, Any]:
        job_type = str(job.get("job_type") or "")
        is_index_waiting_for_parse = (
            job_type == "index"
            and bool(job.get("paper_id"))
            and (
                str(job.get("paper_parse_status") or "") != "succeeded"
                or str(job.get("paper_id")) in running_parse_index_papers
            )
        )
        if is_index_waiting_for_parse:
            job["queue_block_reason"] = "waiting_for_parse"
        return job

    queued = [mark_queue_block_reason(job) for job in queued]
    queued_candidates = [mark_queue_block_reason(job) for job in queued_candidates]
    queued_by_type: dict[str, dict[str, Any]] = {}
    blocked_by_type: dict[str, dict[str, Any]] = {}
    for job in queued_candidates:
        job_type = str(job.get("job_type") or "")
        if job.get("queue_block_reason"):
            blocked_by_type.setdefault(job_type, job)
        else:
            queued_by_type.setdefault(job_type, job)
    return {
        "running": running,
        "queued": queued,
        "running_by_type": running_by_type,
        "queued_by_type": queued_by_type,
        "blocked_by_type": blocked_by_type,
        "running_count": counts.get("running", 0),
        "queued_count": counts.get("queued", 0),
    }


def clear_team_queued_jobs(
    user_id: int,
    team_id: int,
    job_types: tuple[str, ...] | list[str] | None = None,
    reason: str = "queue cleared by UI refresh",
) -> dict[str, Any]:
    """Cancel queued jobs for one team and restore paper statuses for manual scheduling."""
    require_team_role(user_id, team_id, "editor")
    clean_types = tuple(
        dict.fromkeys(normalize_job_type(job_type) for job_type in (job_types or JOB_TYPES))
    )
    placeholders = ", ".join("?" for _ in clean_types)
    parameters: list[Any] = [int(team_id), *clean_types]
    init_db()
    with get_db_connection() as connection:
        rows = connection.execute(
            f"""
            SELECT job_id, job_type, paper_id
            FROM jobs
            WHERE team_id = ?
                AND job_type IN ({placeholders})
                AND status = 'queued'
            ORDER BY job_id ASC
            """,
            parameters,
        ).fetchall()
        jobs = [dict(row) for row in rows]
        if not jobs:
            return {"cleared_count": 0, "jobs": []}

        job_ids = [int(job["job_id"]) for job in jobs]
        id_placeholders = ", ".join("?" for _ in job_ids)
        connection.execute(
            f"""
            UPDATE jobs
            SET
                status = 'canceled',
                error_message = ?,
                finished_at = CURRENT_TIMESTAMP,
                updated_at = CURRENT_TIMESTAMP
            WHERE job_id IN ({id_placeholders})
                AND status = 'queued'
            """,
            [str(reason or "")[:4000], *job_ids],
        )
        restore_paper_statuses_after_queue_clear(connection, jobs)
        return {"cleared_count": len(jobs), "jobs": [normalize_job_row(job) for job in jobs]}


def restore_paper_statuses_after_queue_clear(connection: Any, jobs: list[dict[str, Any]]) -> None:
    """Reset paper status fields after queued jobs are canceled."""
    paper_ids_by_type: dict[str, list[str]] = {}
    for job in jobs:
        paper_id = str(job.get("paper_id") or "")
        job_type = str(job.get("job_type") or "")
        if not paper_id:
            continue
        paper_ids_by_type.setdefault(job_type, []).append(paper_id)

    parse_paper_ids = paper_ids_by_type.get("parse") or []
    if parse_paper_ids:
        placeholders = ", ".join("?" for _ in parse_paper_ids)
        connection.execute(
            f"""
            UPDATE papers
            SET
                parse_status = CASE
                    WHEN COALESCE(markdown_path, '') != ''
                        OR EXISTS (
                            SELECT 1 FROM chunks c WHERE c.paper_id = papers.paper_id LIMIT 1
                        )
                    THEN 'succeeded'
                    ELSE 'not_started'
                END,
                index_status = CASE
                    WHEN index_status = 'queued' THEN 'unknown'
                    ELSE index_status
                END,
                updated_at = CURRENT_TIMESTAMP
            WHERE paper_id IN ({placeholders})
                AND parse_status = 'queued'
            """,
            parse_paper_ids,
        )

    index_paper_ids = paper_ids_by_type.get("index") or []
    if index_paper_ids:
        placeholders = ", ".join("?" for _ in index_paper_ids)
        connection.execute(
            f"""
            UPDATE papers
            SET
                index_status = 'unknown',
                updated_at = CURRENT_TIMESTAMP
            WHERE paper_id IN ({placeholders})
                AND index_status = 'queued'
            """,
            index_paper_ids,
        )

    translate_paper_ids = paper_ids_by_type.get("translate") or []
    if translate_paper_ids:
        placeholders = ", ".join("?" for _ in translate_paper_ids)
        connection.execute(
            f"""
            UPDATE papers
            SET
                translation_status = 'not_started',
                updated_at = CURRENT_TIMESTAMP
            WHERE paper_id IN ({placeholders})
                AND translation_status = 'queued'
            """,
            translate_paper_ids,
        )


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


def claim_next_job(job_types: tuple[str, ...] | list[str] | None = None) -> dict[str, Any] | None:
    """Atomically claim the oldest queued job, optionally constrained by type."""
    return claim_next_job_for_worker(job_types=job_types)


def claim_next_job_for_worker(
    job_types: tuple[str, ...] | list[str] | None = None,
    worker_id: str | None = None,
    lease_seconds: int = DEFAULT_JOB_LEASE_SECONDS,
) -> dict[str, Any] | None:
    """Atomically claim the oldest runnable queued job for a worker."""
    clean_types = tuple(dict.fromkeys(normalize_job_type(job_type) for job_type in (job_types or ())))
    where_clauses = [
        "j.status = 'queued'",
        "j.attempt_count < j.max_attempts",
        "(j.next_run_at IS NULL OR j.next_run_at <= CURRENT_TIMESTAMP)",
        """
        NOT (
            j.job_type IN ('parse', 'index')
            AND j.paper_id IS NOT NULL
            AND EXISTS (
                SELECT 1
                FROM jobs running
                WHERE running.status = 'running'
                    AND running.job_type IN ('parse', 'index')
                    AND running.paper_id = j.paper_id
            )
        )
        """,
        """
        (
            j.job_type != 'index'
            OR j.paper_id IS NULL
            OR EXISTS (
                SELECT 1
                FROM papers p
                WHERE p.paper_id = j.paper_id
                    AND p.parse_status = 'succeeded'
            )
        )
        """,
    ]
    parameters: list[Any] = []
    if clean_types:
        placeholders = ", ".join("?" for _ in clean_types)
        where_clauses.append(f"j.job_type IN ({placeholders})")
        parameters.extend(clean_types)

    init_db()
    connection = get_db_connection()
    try:
        connection.execute("BEGIN IMMEDIATE")
        row = connection.execute(
            f"""
            SELECT j.*
            FROM jobs j
            WHERE {' AND '.join(where_clauses)}
            ORDER BY created_at ASC, job_id ASC
            LIMIT 1
            """,
            parameters,
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
                heartbeat_at = CURRENT_TIMESTAMP,
                worker_id = ?,
                lease_expires_at = datetime(CURRENT_TIMESTAMP, ?),
                next_run_at = NULL,
                started_at = COALESCE(started_at, CURRENT_TIMESTAMP),
                updated_at = CURRENT_TIMESTAMP
            WHERE job_id = ?
            """,
            (
                worker_id,
                sqlite_seconds_modifier(lease_seconds),
                job_id,
            ),
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


def heartbeat_job(
    job_id: int,
    worker_id: str | None = None,
    lease_seconds: int = DEFAULT_JOB_LEASE_SECONDS,
) -> bool:
    """Extend the lease for a running job owned by worker_id."""
    init_db()
    with get_db_connection() as connection:
        if worker_id:
            cursor = connection.execute(
                """
                UPDATE jobs
                SET
                    heartbeat_at = CURRENT_TIMESTAMP,
                    lease_expires_at = datetime(CURRENT_TIMESTAMP, ?),
                    updated_at = CURRENT_TIMESTAMP
                WHERE job_id = ?
                    AND status = 'running'
                    AND worker_id = ?
                """,
                (sqlite_seconds_modifier(lease_seconds), int(job_id), worker_id),
            )
        else:
            cursor = connection.execute(
                """
                UPDATE jobs
                SET
                    heartbeat_at = CURRENT_TIMESTAMP,
                    lease_expires_at = datetime(CURRENT_TIMESTAMP, ?),
                    updated_at = CURRENT_TIMESTAMP
                WHERE job_id = ?
                    AND status = 'running'
                """,
                (sqlite_seconds_modifier(lease_seconds), int(job_id)),
            )
        return bool(cursor.rowcount)


def requeue_expired_jobs(job_types: tuple[str, ...] | list[str] | None = None) -> list[dict[str, Any]]:
    """Requeue running jobs whose worker lease has expired."""
    clean_types = tuple(dict.fromkeys(normalize_job_type(job_type) for job_type in (job_types or ())))
    where_clauses = [
        "status = 'running'",
        "attempt_count < max_attempts",
        "lease_expires_at IS NOT NULL",
        "lease_expires_at <= CURRENT_TIMESTAMP",
    ]
    parameters: list[Any] = []
    if clean_types:
        placeholders = ", ".join("?" for _ in clean_types)
        where_clauses.append(f"job_type IN ({placeholders})")
        parameters.extend(clean_types)

    init_db()
    with get_db_connection() as connection:
        rows = connection.execute(
            f"""
            SELECT job_id, job_type, paper_id
            FROM jobs
            WHERE {' AND '.join(where_clauses)}
            ORDER BY COALESCE(lease_expires_at, locked_at, started_at, updated_at, created_at) ASC, job_id ASC
            """,
            parameters,
        ).fetchall()
        jobs = [dict(row) for row in rows]
        if not jobs:
            return []

        job_ids = [int(job["job_id"]) for job in jobs]
        id_placeholders = ", ".join("?" for _ in job_ids)
        connection.execute(
            f"""
            UPDATE jobs
            SET
                status = 'queued',
                worker_id = NULL,
                locked_at = NULL,
                heartbeat_at = NULL,
                lease_expires_at = NULL,
                started_at = NULL,
                error_message = 'worker lease expired; job returned to queue',
                updated_at = CURRENT_TIMESTAMP
            WHERE job_id IN ({id_placeholders})
                AND status = 'running'
            """,
            job_ids,
        )
        update_running_paper_statuses(connection, jobs)
        return [normalize_job_row(job) for job in jobs]


def requeue_running_jobs(job_types: tuple[str, ...] | list[str] | None = None) -> list[dict[str, Any]]:
    """Requeue jobs left running by an interrupted worker process."""
    clean_types = tuple(dict.fromkeys(normalize_job_type(job_type) for job_type in (job_types or ())))
    where_clauses = ["status = 'running'", "attempt_count < max_attempts"]
    parameters: list[Any] = []
    if clean_types:
        placeholders = ", ".join("?" for _ in clean_types)
        where_clauses.append(f"job_type IN ({placeholders})")
        parameters.extend(clean_types)

    init_db()
    with get_db_connection() as connection:
        rows = connection.execute(
            f"""
            SELECT job_id, job_type, paper_id
            FROM jobs
            WHERE {' AND '.join(where_clauses)}
            ORDER BY COALESCE(locked_at, started_at, updated_at, created_at) ASC, job_id ASC
            """,
            parameters,
        ).fetchall()
        jobs = [dict(row) for row in rows]
        if not jobs:
            return []

        job_ids = [int(job["job_id"]) for job in jobs]
        id_placeholders = ", ".join("?" for _ in job_ids)
        connection.execute(
            f"""
            UPDATE jobs
            SET
                status = 'queued',
                worker_id = NULL,
                locked_at = NULL,
                heartbeat_at = NULL,
                lease_expires_at = NULL,
                next_run_at = NULL,
                started_at = NULL,
                error_message = '',
                updated_at = CURRENT_TIMESTAMP
            WHERE job_id IN ({id_placeholders})
            """,
            job_ids,
        )
        update_running_paper_statuses(connection, jobs)
        return [normalize_job_row(job) for job in jobs]


def update_running_paper_statuses(connection: Any, jobs: list[dict[str, Any]]) -> None:
    """Mirror recovered running job status back to paper status columns."""
    status_by_type = {
        "parse": "parse_status",
        "index": "index_status",
        "translate": "translation_status",
    }
    for job_type, status_column in status_by_type.items():
        paper_ids = [
            str(job.get("paper_id"))
            for job in jobs
            if job.get("paper_id") and str(job.get("job_type")) == job_type
        ]
        if not paper_ids:
            continue
        placeholders = ", ".join("?" for _ in paper_ids)
        connection.execute(
            f"""
            UPDATE papers
            SET {status_column} = 'queued',
                updated_at = CURRENT_TIMESTAMP
            WHERE paper_id IN ({placeholders})
                AND {status_column} = 'running'
            """,
            paper_ids,
        )


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
                worker_id = NULL,
                locked_at = NULL,
                heartbeat_at = NULL,
                lease_expires_at = NULL,
                next_run_at = NULL,
                finished_at = CURRENT_TIMESTAMP,
                updated_at = CURRENT_TIMESTAMP
            WHERE job_id = ?
            """,
            (json.dumps(result or {}, ensure_ascii=False), int(job_id)),
        )


def fail_job(
    job_id: int,
    error_message: str,
    result: dict[str, Any] | None = None,
    *,
    auto_retry: bool = False,
    retry_delay_seconds: int | None = None,
    error_code: str = "",
) -> None:
    """Mark a job failed, or return it to the queue when auto-retry is enabled."""
    init_db()
    with get_db_connection() as connection:
        row = connection.execute(
            """
            SELECT attempt_count, max_attempts
            FROM jobs
            WHERE job_id = ?
            """,
            (int(job_id),),
        ).fetchone()
        should_retry = False
        delay_seconds = int(retry_delay_seconds or 0)
        if row and auto_retry:
            attempt_count = int(row["attempt_count"] or 0)
            max_attempts = int(row["max_attempts"] or 1)
            should_retry = attempt_count < max_attempts
            if should_retry and delay_seconds <= 0:
                delay_seconds = retry_delay_for_attempt(attempt_count)

        if should_retry:
            connection.execute(
                """
                UPDATE jobs
                SET
                    status = 'queued',
                    result_json = ?,
                    error_message = ?,
                    last_error_code = ?,
                    worker_id = NULL,
                    locked_at = NULL,
                    heartbeat_at = NULL,
                    lease_expires_at = NULL,
                    next_run_at = datetime(CURRENT_TIMESTAMP, ?),
                    started_at = NULL,
                    finished_at = NULL,
                    updated_at = CURRENT_TIMESTAMP
                WHERE job_id = ?
                """,
                (
                    json.dumps(result or {}, ensure_ascii=False),
                    str(error_message or "")[:4000],
                    str(error_code or "")[:120],
                    sqlite_seconds_modifier(delay_seconds),
                    int(job_id),
                ),
            )
            return

        connection.execute(
            """
            UPDATE jobs
            SET
                status = 'failed',
                result_json = ?,
                error_message = ?,
                last_error_code = ?,
                worker_id = NULL,
                locked_at = NULL,
                heartbeat_at = NULL,
                lease_expires_at = NULL,
                next_run_at = NULL,
                finished_at = CURRENT_TIMESTAMP,
                updated_at = CURRENT_TIMESTAMP
            WHERE job_id = ?
            """,
            (
                json.dumps(result or {}, ensure_ascii=False),
                str(error_message or "")[:4000],
                str(error_code or "")[:120],
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
                worker_id = NULL,
                locked_at = NULL,
                heartbeat_at = NULL,
                lease_expires_at = NULL,
                next_run_at = NULL,
                started_at = NULL,
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
                worker_id = NULL,
                locked_at = NULL,
                heartbeat_at = NULL,
                lease_expires_at = NULL,
                next_run_at = NULL,
                finished_at = CURRENT_TIMESTAMP,
                updated_at = CURRENT_TIMESTAMP
            WHERE job_id = ?
                AND status IN ('queued', 'running')
            """,
            (int(job_id),),
        )


def cancel_queued_job(user_id: int, job_id: int) -> bool:
    """Cancel one queued job and restore paper status for manual scheduling."""
    job = get_job(job_id)
    if not job:
        raise ValueError("没有找到这个任务。")
    require_team_role(user_id, int(job["team_id"]), "editor")
    if str(job.get("status") or "") != "queued":
        return False
    with get_db_connection() as connection:
        cursor = connection.execute(
            """
            UPDATE jobs
            SET
                status = 'canceled',
                error_message = 'removed from queue by user',
                worker_id = NULL,
                locked_at = NULL,
                heartbeat_at = NULL,
                lease_expires_at = NULL,
                next_run_at = NULL,
                finished_at = CURRENT_TIMESTAMP,
                updated_at = CURRENT_TIMESTAMP
            WHERE job_id = ?
                AND status = 'queued'
            """,
            (int(job_id),),
        )
        if cursor.rowcount:
            restore_paper_statuses_after_queue_clear(connection, [job])
        return bool(cursor.rowcount)


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


def retry_delay_for_attempt(attempt_count: int) -> int:
    """Return exponential retry delay after the current failed attempt."""
    exponent = max(0, int(attempt_count) - 1)
    delay = DEFAULT_RETRY_BASE_DELAY_SECONDS * (2**exponent)
    return min(DEFAULT_RETRY_MAX_DELAY_SECONDS, delay)


def sqlite_seconds_modifier(seconds: int | float) -> str:
    """Return a SQLite datetime modifier for a positive second interval."""
    clean_seconds = max(1, int(seconds or 1))
    return f"+{clean_seconds} seconds"
