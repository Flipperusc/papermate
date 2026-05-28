"""Smoke tests for PaperMate backend diagnostics."""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.doctor_backend import exit_code_for_report
from src import db as db_module
from src.diagnostics import CHECK_OK, CHECK_WARNING, collect_backend_diagnostics
from src.db import get_db_connection, init_db


PATCHED_SETTINGS = {
    "data_dir",
    "upload_dir",
    "markdown_dir",
    "chroma_dir",
    "bm25_dir",
    "log_dir",
    "db_path",
    "mineru_output_dir",
    "deepseek_api_key",
    "embedding_api_key",
    "mineru_api_token",
    "vlm_api_key",
    "pdf_parse_provider",
    "embedding_provider",
    "embedding_model",
    "embedding_dimensions",
    "vlm_enabled",
    "app_env",
    "app_password",
    "external_api_max_attempts",
    "external_api_retry_base_seconds",
    "external_api_retry_max_seconds",
}


def main() -> None:
    test_diagnostics_redact_configured_secrets_and_report_ok()
    test_diagnostics_report_missing_secret_warnings()
    test_diagnostics_warn_on_expired_running_jobs()
    print("backend diagnostics tests passed")


def test_diagnostics_redact_configured_secrets_and_report_ok() -> None:
    secret_values = {
        "deepseek_api_key": "deep-secret-value",
        "embedding_api_key": "embedding-secret-value",
        "mineru_api_token": "mineru-secret-value",
        "vlm_api_key": "vlm-secret-value",
    }
    with patched_runtime_settings(**secret_values):
        report = collect_backend_diagnostics()
    payload = json.dumps(report, ensure_ascii=False)
    assert report["status"] == CHECK_OK
    for secret in secret_values.values():
        assert secret not in payload
    assert '"configured": true' in payload
    assert exit_code_for_report(report) == 0


def test_diagnostics_report_missing_secret_warnings() -> None:
    with patched_runtime_settings(
        deepseek_api_key="",
        embedding_api_key="",
        mineru_api_token="",
        vlm_api_key="",
        pdf_parse_provider="mineru",
        vlm_enabled=True,
    ):
        report = collect_backend_diagnostics()
    assert report["status"] == CHECK_WARNING
    warning_names = {
        str(check["name"])
        for check in report["checks"]
        if check["status"] == CHECK_WARNING
    }
    assert {
        "config:deepseek_api_key",
        "config:embedding_api_key",
        "config:mineru_api_token",
        "config:vlm_api_key",
    }.issubset(warning_names)
    assert exit_code_for_report(report) == 0
    assert exit_code_for_report(report, strict=True) == 2


def test_diagnostics_warn_on_expired_running_jobs() -> None:
    with patched_runtime_settings() as runtime:
        init_db(force=True)
        with get_db_connection() as connection:
            connection.execute(
                """
                INSERT INTO jobs (
                    job_type,
                    status,
                    payload_json,
                    worker_id,
                    heartbeat_at,
                    lease_expires_at,
                    started_at
                )
                VALUES (
                    'parse',
                    'running',
                    '{}',
                    'stale-worker',
                    datetime(CURRENT_TIMESTAMP, '-10 minutes'),
                    datetime(CURRENT_TIMESTAMP, '-5 minutes'),
                    datetime(CURRENT_TIMESTAMP, '-10 minutes')
                )
                """
            )
        report = collect_backend_diagnostics()
    queue_check = next(check for check in report["checks"] if check["name"] == "jobs:queue_state")
    assert queue_check["status"] == CHECK_WARNING
    assert queue_check["details"]["expired_running_count"] == 1
    assert str(runtime["db_path"]).endswith("papermate-diagnostics-test.db")


class patched_runtime_settings:
    """Temporarily redirect global settings to an isolated local runtime."""

    def __init__(self, **overrides: Any) -> None:
        self.overrides = overrides
        self.originals = {name: getattr(db_module.settings, name) for name in PATCHED_SETTINGS}
        self.runtime: dict[str, Any] = {}

    def __enter__(self) -> dict[str, Any]:
        self.tmp_dir = tempfile.TemporaryDirectory()
        root = Path(self.tmp_dir.name)
        values: dict[str, Any] = {
            "data_dir": root / "data",
            "upload_dir": root / "data" / "uploads",
            "markdown_dir": root / "data" / "markdown",
            "chroma_dir": root / "data" / "chroma_db",
            "bm25_dir": root / "data" / "bm25",
            "log_dir": root / "logs",
            "db_path": root / "data" / "papermate-diagnostics-test.db",
            "mineru_output_dir": root / "data" / "markdown",
            "deepseek_api_key": "test-deepseek-key",
            "embedding_api_key": "test-embedding-key",
            "mineru_api_token": "test-mineru-token",
            "vlm_api_key": "test-vlm-key",
            "pdf_parse_provider": "mineru",
            "embedding_provider": "zhipu",
            "embedding_model": "embedding-3",
            "embedding_dimensions": 2048,
            "vlm_enabled": True,
            "app_env": "local",
            "app_password": "",
            "external_api_max_attempts": 2,
            "external_api_retry_base_seconds": 0.0,
            "external_api_retry_max_seconds": 0.0,
        }
        values.update(self.overrides)
        for name, value in values.items():
            object.__setattr__(db_module.settings, name, value)
        self.runtime = values
        return values

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        for name, value in self.originals.items():
            object.__setattr__(db_module.settings, name, value)
        self.tmp_dir.cleanup()


if __name__ == "__main__":
    main()
