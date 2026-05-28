"""Smoke tests for application-level paper workflow orchestration."""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path
from uuid import uuid4


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src import db as db_module
from src.application import paper_workflow
from src.auth_service import create_user
from src.db import get_db_connection, init_db, save_paper_and_chunks
from src.job_service import get_job
from src.team_service import ensure_user_workspace


PDF_BYTES = b"%PDF-1.4\n% PaperMate workflow smoke test\n"


def main() -> None:
    test_upload_pipeline_reuses_active_jobs()
    test_manual_parse_and_index_queueing()
    print("paper workflow service tests passed")


def test_upload_pipeline_reuses_active_jobs() -> None:
    with isolated_runtime() as runtime:
        user_id, team_id, project_id = create_workspace_user("upload")

        first = paper_workflow.save_uploaded_pdf_to_library(
            file_name="workflow.pdf",
            file_bytes=PDF_BYTES,
            user_id=user_id,
            team_id=team_id,
            project_id=project_id,
            signature="upload-1",
        )
        assert first["reused"] is False
        assert first["parse_job_id"]
        assert first["index_job_id"]
        assert first["parse_job_status"] == "queued"
        assert first["index_job_status"] == "queued"
        assert first["signature"] == "upload-1"
        assert Path(first["paper"]["save_path"]).exists()

        parse_job = get_job(int(first["parse_job_id"]))
        index_job = get_job(int(first["index_job_id"]))
        assert parse_job["payload"]["auto_created_from_upload"] is True
        assert index_job["payload"]["waiting_for_parse"] is True

        second = paper_workflow.save_uploaded_pdf_to_library(
            file_name="workflow.pdf",
            file_bytes=PDF_BYTES,
            user_id=user_id,
            team_id=team_id,
            project_id=project_id,
            signature="upload-2",
        )
        assert second["reused"] is True
        assert second["paper"]["paper_id"] == first["paper"]["paper_id"]
        assert second["parse_job_id"] == first["parse_job_id"]
        assert second["index_job_id"] == first["index_job_id"]
        assert second["parse_reused"] is True
        assert second["index_reused"] is True
        assert second["signature"] == "upload-2"
        assert len(list(runtime["upload_dir"].glob("*.pdf"))) == 1


def test_manual_parse_and_index_queueing() -> None:
    with isolated_runtime():
        user_id, team_id, project_id = create_workspace_user("manual")
        parsed_paper_id = insert_paper(
            user_id=user_id,
            team_id=team_id,
            project_id=project_id,
            parse_status="succeeded",
            index_status="unknown",
        )

        index_result = paper_workflow.queue_index_build_for_paper(
            parsed_paper_id,
            user_id=user_id,
        )
        assert index_result["reused_existing"] is False
        assert index_result["status"] == "queued"
        assert get_job(int(index_result["job_id"]))["payload"]["paper_id"] == parsed_paper_id

        reused_index = paper_workflow.queue_index_build_for_paper(
            parsed_paper_id,
            user_id=user_id,
        )
        assert reused_index["reused_existing"] is True
        assert reused_index["job_id"] == index_result["job_id"]

        unparsed_paper_id = insert_paper(
            user_id=user_id,
            team_id=team_id,
            project_id=project_id,
            parse_status="not_started",
            index_status="unknown",
        )
        try:
            paper_workflow.queue_index_build_for_paper(unparsed_paper_id, user_id=user_id)
        except ValueError:
            pass
        else:
            raise AssertionError("unparsed paper should not be indexed when require_parsed is true")

        parse_result = paper_workflow.queue_paper_parse(
            unparsed_paper_id,
            user_id=user_id,
            include_images=True,
        )
        assert parse_result["reused"] is False
        parse_job = get_job(int(parse_result["parse_job_id"]))
        assert parse_job["payload"]["include_images"] is True
        assert parse_job["payload"]["auto_index"] is False

        reused_parse = paper_workflow.queue_paper_parse(unparsed_paper_id, user_id=user_id)
        assert reused_parse["reused"] is True
        assert reused_parse["parse_job_id"] == parse_result["parse_job_id"]


class isolated_runtime:
    """Temporarily redirect runtime paths to a private test directory."""

    def __enter__(self) -> dict[str, Path]:
        self.tmp = tempfile.TemporaryDirectory()
        root = Path(self.tmp.name)
        self.originals = {
            "db_path": db_module.settings.db_path,
            "upload_dir": db_module.settings.upload_dir,
            "chroma_dir": db_module.settings.chroma_dir,
            "mineru_output_dir": db_module.settings.mineru_output_dir,
        }
        paths = {
            "db_path": root / "papermate-test.db",
            "upload_dir": root / "uploads",
            "chroma_dir": root / "chroma",
            "mineru_output_dir": root / "markdown",
        }
        for name, path in paths.items():
            object.__setattr__(db_module.settings, name, path)
        init_db(force=True)
        self.paths = paths
        return paths

    def __exit__(self, exc_type, exc_value, traceback) -> bool:
        for name, value in self.originals.items():
            object.__setattr__(db_module.settings, name, value)
        self.tmp.cleanup()
        return False


def create_workspace_user(label: str) -> tuple[int, int, int]:
    suffix = uuid4().hex[:10]
    user = create_user(f"pm_{label}_{suffix}", "password123")
    workspace = ensure_user_workspace(int(user["user_id"]))
    return int(user["user_id"]), int(workspace["team_id"]), int(workspace["project_id"])


def insert_paper(
    *,
    user_id: int,
    team_id: int,
    project_id: int,
    parse_status: str,
    index_status: str,
) -> str:
    paper_id = f"paper-{uuid4().hex[:12]}"
    upload_dir = db_module.settings.upload_dir
    upload_dir.mkdir(parents=True, exist_ok=True)
    save_path = upload_dir / f"{paper_id}.pdf"
    save_path.write_bytes(PDF_BYTES)
    save_paper_and_chunks(
        {
            "paper_id": paper_id,
            "file_name": save_path.name,
            "file_size_bytes": len(PDF_BYTES),
            "save_path": str(save_path),
            "owner_user_id": user_id,
            "team_id": team_id,
            "project_id": project_id,
            "file_sha256": uuid4().hex,
            "parse_status": parse_status,
            "index_status": index_status,
            "translation_status": "not_started",
            "page_count": 1,
            "total_chars": 0,
        },
        [],
    )
    with get_db_connection() as connection:
        row = connection.execute(
            "SELECT paper_id FROM papers WHERE paper_id = ?",
            (paper_id,),
        ).fetchone()
    assert row is not None
    return paper_id


if __name__ == "__main__":
    main()

