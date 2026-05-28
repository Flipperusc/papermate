"""Backend diagnostics for local PaperMate deployments."""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from config import settings
from src.db import get_db_connection, init_db
from src.embedding_client import OPENAI_COMPATIBLE_PROVIDERS, ZHIPU_PROVIDERS


CHECK_OK = "ok"
CHECK_WARNING = "warning"
CHECK_ERROR = "error"

SUPPORTED_PDF_PROVIDERS = {"mineru", "pymupdf"}
REQUIRED_TABLES = {
    "users",
    "teams",
    "team_members",
    "projects",
    "papers",
    "chunks",
    "qa_logs",
    "feedback",
    "bad_cases",
    "literature_cards",
    "card_libraries",
    "jobs",
}
REQUIRED_COLUMNS = {
    "papers": {
        "paper_id",
        "owner_user_id",
        "team_id",
        "project_id",
        "file_sha256",
        "parse_status",
        "index_status",
        "translation_status",
        "markdown_path",
        "translated_markdown_path",
        "content_list_path",
    },
    "chunks": {
        "chunk_id",
        "paper_id",
        "chunk_index",
        "page_num",
        "section_title",
        "text",
        "chunk_type",
        "images_json",
        "tables_json",
    },
    "jobs": {
        "job_id",
        "job_type",
        "status",
        "paper_id",
        "team_id",
        "project_id",
        "user_id",
        "payload_json",
        "result_json",
        "error_message",
        "attempt_count",
        "max_attempts",
        "worker_id",
        "locked_at",
        "heartbeat_at",
        "lease_expires_at",
        "next_run_at",
        "last_error_code",
        "started_at",
        "finished_at",
    },
}
REQUIRED_INDEXES = {
    "idx_jobs_status_created",
    "idx_jobs_type_status_next_run",
    "idx_jobs_running_lease",
}


@dataclass(frozen=True)
class DiagnosticCheck:
    """One backend health check result."""

    name: str
    status: str
    message: str
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-safe representation."""
        return asdict(self)


def collect_backend_diagnostics(*, initialize_database: bool = True) -> dict[str, Any]:
    """Collect local backend diagnostics without exposing secret values."""
    checks: list[DiagnosticCheck] = []
    checks.extend(check_runtime_paths())
    checks.extend(check_configuration())
    checks.extend(check_retry_configuration())
    checks.extend(check_sqlite_database(initialize_database=initialize_database))
    checks.extend(check_queue_state())
    return build_report(checks)


def build_report(checks: list[DiagnosticCheck]) -> dict[str, Any]:
    """Build the aggregate diagnostic report."""
    summary = {
        CHECK_OK: sum(1 for check in checks if check.status == CHECK_OK),
        CHECK_WARNING: sum(1 for check in checks if check.status == CHECK_WARNING),
        CHECK_ERROR: sum(1 for check in checks if check.status == CHECK_ERROR),
    }
    if summary[CHECK_ERROR]:
        status = CHECK_ERROR
    elif summary[CHECK_WARNING]:
        status = CHECK_WARNING
    else:
        status = CHECK_OK
    return {
        "status": status,
        "summary": summary,
        "checks": [check.to_dict() for check in checks],
    }


def check_runtime_paths() -> list[DiagnosticCheck]:
    """Verify runtime directories can be created and written."""
    path_checks: list[DiagnosticCheck] = []
    paths = {
        "data_dir": settings.data_dir,
        "upload_dir": settings.upload_dir,
        "markdown_dir": settings.markdown_dir,
        "mineru_output_dir": settings.mineru_output_dir,
        "chroma_dir": settings.chroma_dir,
        "bm25_dir": settings.bm25_dir,
        "log_dir": settings.log_dir,
        "db_parent": settings.db_path.parent,
    }
    for label, path in paths.items():
        path_checks.append(check_directory_writable(label, path))
    return path_checks


def check_directory_writable(label: str, path: Path) -> DiagnosticCheck:
    """Return whether a directory path is writable."""
    try:
        path.mkdir(parents=True, exist_ok=True)
        test_path = path / f".papermate_write_test_{os.getpid()}"
        test_path.write_text("ok", encoding="utf-8")
        test_path.unlink(missing_ok=True)
    except OSError as exc:
        return DiagnosticCheck(
            name=f"runtime_path:{label}",
            status=CHECK_ERROR,
            message="runtime directory is not writable",
            details={"path": str(path), "error": compact_error(exc)},
        )
    return DiagnosticCheck(
        name=f"runtime_path:{label}",
        status=CHECK_OK,
        message="runtime directory is writable",
        details={"path": str(path)},
    )


def check_configuration() -> list[DiagnosticCheck]:
    """Validate config shape while redacting all secret values."""
    checks = [
        credential_check(
            "config:deepseek_api_key",
            "DEEPSEEK_API_KEY",
            bool(settings.deepseek_api_key),
            "DeepSeek key is configured",
            "DEEPSEEK_API_KEY is missing; Q&A, translation, rerank, and card generation will fail",
        ),
        credential_check(
            "config:embedding_api_key",
            "EMBEDDING_API_KEY",
            bool(settings.embedding_api_key),
            "embedding key is configured",
            "EMBEDDING_API_KEY is missing; vector indexing will fail",
        ),
    ]

    pdf_provider = str(settings.pdf_parse_provider or "").lower()
    if pdf_provider not in SUPPORTED_PDF_PROVIDERS:
        checks.append(
            DiagnosticCheck(
                name="config:pdf_parse_provider",
                status=CHECK_ERROR,
                message="unsupported PDF_PARSE_PROVIDER",
                details={
                    "provider": pdf_provider,
                    "supported": sorted(SUPPORTED_PDF_PROVIDERS),
                },
            )
        )
    else:
        checks.append(
            DiagnosticCheck(
                name="config:pdf_parse_provider",
                status=CHECK_OK,
                message="PDF parser provider is supported",
                details={"provider": pdf_provider},
            )
        )

    if pdf_provider == "mineru":
        checks.append(
            credential_check(
                "config:mineru_api_token",
                "MINERU_API_TOKEN",
                bool(settings.mineru_api_token),
                "MinerU token is configured",
                "MINERU_API_TOKEN is missing while PDF_PARSE_PROVIDER=mineru",
            )
        )

    embedding_provider = str(settings.embedding_provider or "").lower().replace("-", "_")
    supported_embedding_providers = sorted(ZHIPU_PROVIDERS | OPENAI_COMPATIBLE_PROVIDERS)
    if embedding_provider not in (ZHIPU_PROVIDERS | OPENAI_COMPATIBLE_PROVIDERS):
        checks.append(
            DiagnosticCheck(
                name="config:embedding_provider",
                status=CHECK_ERROR,
                message="unsupported EMBEDDING_PROVIDER",
                details={
                    "provider": embedding_provider,
                    "supported": supported_embedding_providers,
                },
            )
        )
    else:
        checks.append(
            DiagnosticCheck(
                name="config:embedding_provider",
                status=CHECK_OK,
                message="embedding provider is supported",
                details={"provider": embedding_provider},
            )
        )

    if (
        embedding_provider in ZHIPU_PROVIDERS
        and settings.embedding_model == "embedding-3"
        and settings.embedding_dimensions not in {256, 512, 1024, 2048}
    ):
        checks.append(
            DiagnosticCheck(
                name="config:embedding_dimensions",
                status=CHECK_ERROR,
                message="embedding-3 dimensions must be one of 256, 512, 1024, 2048",
                details={"dimensions": settings.embedding_dimensions},
            )
        )
    else:
        checks.append(
            DiagnosticCheck(
                name="config:embedding_dimensions",
                status=CHECK_OK,
                message="embedding dimensions are valid",
                details={
                    "model": settings.embedding_model,
                    "dimensions": settings.embedding_dimensions,
                },
            )
        )

    if settings.vlm_enabled:
        checks.append(
            credential_check(
                "config:vlm_api_key",
                "VLM_API_KEY",
                bool(settings.vlm_api_key),
                "VLM key is configured",
                "VLM_API_KEY or DASHSCOPE_API_KEY is missing while VLM_ENABLED=true",
            )
        )
    else:
        checks.append(
            DiagnosticCheck(
                name="config:vlm_enabled",
                status=CHECK_OK,
                message="VLM image description is disabled",
                details={"enabled": False},
            )
        )

    if str(settings.app_env).lower() not in {"local", "development", "dev", "test"} and not settings.app_password:
        checks.append(
            DiagnosticCheck(
                name="config:app_password",
                status=CHECK_WARNING,
                message="PAPERMATE_APP_PASSWORD is empty outside local development",
                details={"configured": False},
            )
        )

    return checks


def credential_check(
    name: str,
    env_name: str,
    configured: bool,
    ok_message: str,
    missing_message: str,
) -> DiagnosticCheck:
    """Return a credential presence check without exposing the credential."""
    return DiagnosticCheck(
        name=name,
        status=CHECK_OK if configured else CHECK_WARNING,
        message=ok_message if configured else missing_message,
        details={"env": env_name, "configured": bool(configured)},
    )


def check_retry_configuration() -> list[DiagnosticCheck]:
    """Validate shared external retry settings."""
    max_attempts = int(settings.external_api_max_attempts or 0)
    base_delay = float(settings.external_api_retry_base_seconds or 0)
    max_delay = float(settings.external_api_retry_max_seconds or 0)
    details = {
        "max_attempts": max_attempts,
        "base_delay_seconds": base_delay,
        "max_delay_seconds": max_delay,
    }
    if max_attempts < 1:
        return [
            DiagnosticCheck(
                name="config:external_api_retry",
                status=CHECK_ERROR,
                message="EXTERNAL_API_MAX_ATTEMPTS must be at least 1",
                details=details,
            )
        ]
    if base_delay < 0 or max_delay < 0:
        return [
            DiagnosticCheck(
                name="config:external_api_retry",
                status=CHECK_ERROR,
                message="external retry delays must be non-negative",
                details=details,
            )
        ]
    if max_delay < base_delay:
        return [
            DiagnosticCheck(
                name="config:external_api_retry",
                status=CHECK_WARNING,
                message="EXTERNAL_API_RETRY_MAX_SECONDS is lower than base delay; retry helper will clamp it",
                details=details,
            )
        ]
    return [
        DiagnosticCheck(
            name="config:external_api_retry",
            status=CHECK_OK,
            message="external retry settings are valid",
            details=details,
        )
    ]


def check_sqlite_database(*, initialize_database: bool = True) -> list[DiagnosticCheck]:
    """Validate SQLite connectivity, schema, and important runtime pragmas."""
    checks: list[DiagnosticCheck] = []
    try:
        if initialize_database:
            init_db()
        with get_db_connection() as connection:
            table_rows = connection.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table'"
            ).fetchall()
            table_names = {str(row["name"]) for row in table_rows}
            missing_tables = sorted(REQUIRED_TABLES - table_names)
            checks.append(
                DiagnosticCheck(
                    name="sqlite:tables",
                    status=CHECK_ERROR if missing_tables else CHECK_OK,
                    message="required tables are present" if not missing_tables else "required tables are missing",
                    details={"missing": missing_tables, "count": len(table_names)},
                )
            )

            for table_name, required_columns in REQUIRED_COLUMNS.items():
                if table_name not in table_names:
                    continue
                column_rows = connection.execute(f"PRAGMA table_info({table_name})").fetchall()
                column_names = {str(row["name"]) for row in column_rows}
                missing_columns = sorted(required_columns - column_names)
                checks.append(
                    DiagnosticCheck(
                        name=f"sqlite:columns:{table_name}",
                        status=CHECK_ERROR if missing_columns else CHECK_OK,
                        message=(
                            f"{table_name} columns are present"
                            if not missing_columns
                            else f"{table_name} columns are missing"
                        ),
                        details={"missing": missing_columns, "count": len(column_names)},
                    )
                )

            index_rows = connection.execute(
                "SELECT name FROM sqlite_master WHERE type = 'index'"
            ).fetchall()
            index_names = {str(row["name"]) for row in index_rows}
            missing_indexes = sorted(REQUIRED_INDEXES - index_names)
            checks.append(
                DiagnosticCheck(
                    name="sqlite:indexes",
                    status=CHECK_ERROR if missing_indexes else CHECK_OK,
                    message="required indexes are present" if not missing_indexes else "required indexes are missing",
                    details={"missing": missing_indexes},
                )
            )

            foreign_keys = int(connection.execute("PRAGMA foreign_keys").fetchone()[0])
            journal_mode = str(connection.execute("PRAGMA journal_mode").fetchone()[0]).lower()
            checks.append(
                DiagnosticCheck(
                    name="sqlite:pragmas",
                    status=CHECK_OK if foreign_keys == 1 else CHECK_ERROR,
                    message="SQLite pragmas are configured" if foreign_keys == 1 else "SQLite foreign_keys pragma is disabled",
                    details={
                        "foreign_keys": bool(foreign_keys),
                        "journal_mode": journal_mode,
                        "db_path": str(settings.db_path),
                    },
                )
            )
    except Exception as exc:
        checks.append(
            DiagnosticCheck(
                name="sqlite:connection",
                status=CHECK_ERROR,
                message="SQLite diagnostics failed",
                details={"db_path": str(settings.db_path), "error": compact_error(exc)},
            )
        )
    return checks


def check_queue_state() -> list[DiagnosticCheck]:
    """Summarize durable queue state without requiring a user session."""
    try:
        with get_db_connection() as connection:
            status_rows = connection.execute(
                """
                SELECT status, COUNT(*) AS count
                FROM jobs
                GROUP BY status
                ORDER BY status
                """
            ).fetchall()
            type_status_rows = connection.execute(
                """
                SELECT job_type, status, COUNT(*) AS count
                FROM jobs
                WHERE status IN ('queued', 'running')
                GROUP BY job_type, status
                ORDER BY job_type, status
                """
            ).fetchall()
            expired_running = int(
                connection.execute(
                    """
                    SELECT COUNT(*)
                    FROM jobs
                    WHERE status = 'running'
                        AND lease_expires_at IS NOT NULL
                        AND lease_expires_at <= CURRENT_TIMESTAMP
                    """
                ).fetchone()[0]
            )
            scheduled_retry = int(
                connection.execute(
                    """
                    SELECT COUNT(*)
                    FROM jobs
                    WHERE status = 'queued'
                        AND next_run_at IS NOT NULL
                        AND next_run_at > CURRENT_TIMESTAMP
                    """
                ).fetchone()[0]
            )
        status_counts = {str(row["status"]): int(row["count"]) for row in status_rows}
        active_by_type = [
            {
                "job_type": str(row["job_type"]),
                "status": str(row["status"]),
                "count": int(row["count"]),
            }
            for row in type_status_rows
        ]
        return [
            DiagnosticCheck(
                name="jobs:queue_state",
                status=CHECK_WARNING if expired_running else CHECK_OK,
                message=(
                    "queue has expired running jobs that can be recovered"
                    if expired_running
                    else "queue state is readable"
                ),
                details={
                    "by_status": status_counts,
                    "active_by_type": active_by_type,
                    "expired_running_count": expired_running,
                    "scheduled_retry_count": scheduled_retry,
                },
            )
        ]
    except Exception as exc:
        return [
            DiagnosticCheck(
                name="jobs:queue_state",
                status=CHECK_ERROR,
                message="queue state could not be read",
                details={"error": compact_error(exc)},
            )
        ]


def compact_error(error: BaseException, limit: int = 300) -> str:
    """Return a compact, bounded error message for diagnostic output."""
    return " ".join(str(error or type(error).__name__).split())[: max(1, int(limit))]
