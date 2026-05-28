"""Smoke tests for application-level study workflow orchestration."""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path
from uuid import uuid4


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src import db as db_module
from src.application import study_workflow
from src.auth_service import create_user
from src.db import get_db_connection, init_db, save_paper_and_chunks
from src.job_service import get_job
from src.literature_card_service import ensure_default_card_library, get_literature_card
from src.team_service import ensure_user_workspace


PDF_BYTES = b"%PDF-1.4\n% PaperMate study workflow smoke test\n"


def main() -> None:
    test_answer_paper_question_builds_record_and_saves_log()
    test_translation_and_card_workflows()
    print("study workflow service tests passed")


def test_answer_paper_question_builds_record_and_saves_log() -> None:
    original_answer_question = study_workflow.rag_answer_question
    with isolated_runtime():
        user_id, team_id, project_id = create_workspace_user("qa")
        paper_id = insert_paper(user_id=user_id, team_id=team_id, project_id=project_id)

        def fake_answer_question(paper_id: str, question: str, user_id: int | None = None) -> dict:
            return {
                "answer": f"answer for {question}",
                "citations": [{"citation_id": 1, "chunk_id": "chunk-1"}],
                "source_chunks": [{"chunk_id": "chunk-1", "text": "source"}],
                "retrieval_details": {"strategy": "fake"},
                "qa_id": None,
            }

        try:
            study_workflow.rag_answer_question = fake_answer_question
            result = study_workflow.answer_paper_question(
                paper_id,
                "  核心方法是什么？  ",
                user_id=user_id,
            )
        finally:
            study_workflow.rag_answer_question = original_answer_question

        assert result["qa_id"] is not None
        assert result["qa_log_save_failed"] is False
        assert result["qa_record"]["paper_id"] == paper_id
        assert result["qa_record"]["question"] == "核心方法是什么？"
        assert result["qa_record"]["answer"] == "answer for 核心方法是什么？"
        assert result["qa_record"]["retrieval_details"]["strategy"] == "fake"

        with get_db_connection() as connection:
            count = connection.execute(
                "SELECT COUNT(*) FROM qa_logs WHERE paper_id = ?",
                (paper_id,),
            ).fetchone()[0]
        assert int(count) == 1


def test_translation_and_card_workflows() -> None:
    with isolated_runtime():
        user_id, team_id, project_id = create_workspace_user("study")
        paper_id = insert_paper(user_id=user_id, team_id=team_id, project_id=project_id)
        library = ensure_default_card_library(user_id, team_id=team_id)

        translate_job_id = study_workflow.queue_markdown_translation(
            paper_id,
            user_id=user_id,
            input_md_path="paper.md",
            output_md_path="paper.zh.md",
            force=True,
        )
        translate_job = get_job(translate_job_id)
        assert translate_job["job_type"] == "translate"
        assert translate_job["payload"]["input_md_path"] == "paper.md"
        assert translate_job["payload"]["output_md_path"] == "paper.zh.md"
        assert translate_job["payload"]["force"] is True

        with get_db_connection() as connection:
            paper_status = connection.execute(
                "SELECT translation_status FROM papers WHERE paper_id = ?",
                (paper_id,),
            ).fetchone()["translation_status"]
        assert paper_status == "queued"

        card_job_id = study_workflow.queue_literature_card_generation(
            paper_id,
            user_id=user_id,
            library_id=int(library["library_id"]),
        )
        card_job = get_job(card_job_id)
        assert card_job["job_type"] == "card"
        assert int(card_job["payload"]["library_id"]) == int(library["library_id"])

        card_id = study_workflow.save_literature_card_markdown(
            paper_id,
            CARD_MARKDOWN,
            user_id=user_id,
            library_id=int(library["library_id"]),
        )
        card = get_literature_card(card_id, user_id=user_id, team_id=team_id)
        assert card is not None
        assert card["paper_id"] == paper_id
        assert card["title"] == "Workflow Card"


class isolated_runtime:
    """Temporarily redirect the database and runtime directories."""

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


def insert_paper(*, user_id: int, team_id: int, project_id: int) -> str:
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
            "parse_status": "succeeded",
            "index_status": "succeeded",
            "translation_status": "not_started",
            "page_count": 1,
            "total_chars": 12,
        },
        [],
    )
    return paper_id


CARD_MARKDOWN = """# Workflow Card

## 论文标题
Workflow Card

## 作者
PaperMate

## 年份
2026

## 研究领域
Software Architecture

## 研究问题
How to keep workflow orchestration maintainable.

## 方法概述
Move UI-independent logic into application services.

## 实验数据集
原文未明确说明
"""


if __name__ == "__main__":
    main()
