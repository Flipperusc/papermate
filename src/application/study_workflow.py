"""Paper study workflow orchestration.

The functions here back user-facing study actions such as asking questions,
requesting translations, and creating literature cards. They enforce paper
access and coordinate lower-level services without depending on Streamlit.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any

from src.db import save_qa_log
from src.job_service import enqueue_job
from src.literature_card_service import save_literature_card
from src.paper_service import get_accessible_paper, update_paper_status
from src.rag_pipeline import answer_question as rag_answer_question


def answer_paper_question(
    paper_id: str,
    question: str,
    *,
    user_id: int,
) -> dict[str, Any]:
    """Answer one paper question and return the UI-ready QA record."""
    clean_question = str(question or "").strip()
    if not clean_question:
        raise ValueError("请输入问题。")
    if not get_accessible_paper(paper_id, user_id, minimum_role="editor"):
        raise PermissionError("没有找到当前论文或无权提问。")

    rag_result = rag_answer_question(paper_id, clean_question, user_id=user_id)
    answer = str(rag_result.get("answer") or "")
    qa_log_id = rag_result.get("qa_id")
    qa_log_save_failed = False
    if qa_log_id is None:
        try:
            qa_log_id = save_qa_log(paper_id, clean_question, answer, user_id=user_id)
        except (OSError, sqlite3.Error):
            qa_log_save_failed = True

    qa_record = {
        "paper_id": paper_id,
        "question": clean_question,
        "answer": answer,
        "citations": rag_result.get("citations") or [],
        "source_chunks": rag_result.get("source_chunks") or [],
        "retrieval_details": rag_result.get("retrieval_details") or rag_result.get("retrieval_debug", {}),
        "qa_log_id": qa_log_id,
    }
    result = dict(rag_result)
    result["qa_id"] = qa_log_id
    result["qa_log_save_failed"] = qa_log_save_failed
    result["qa_record"] = qa_record
    return result


def queue_literature_card_generation(
    paper_id: str,
    *,
    user_id: int,
    library_id: int,
) -> int:
    """Queue a worker job that generates and saves a literature card."""
    paper = _require_editable_paper(paper_id, user_id)
    job_id = enqueue_job(
        "card",
        user_id=user_id,
        team_id=int(paper["team_id"]),
        project_id=paper.get("project_id"),
        paper_id=paper_id,
        payload={"paper_id": paper_id, "library_id": int(library_id)},
    )
    return int(job_id)


def save_literature_card_markdown(
    paper_id: str,
    markdown: str,
    *,
    user_id: int,
    library_id: int,
) -> int:
    """Persist user-reviewed literature-card Markdown for one paper."""
    paper = _require_editable_paper(paper_id, user_id)
    return int(
        save_literature_card(
            paper_id,
            markdown,
            user_id=user_id,
            library_id=int(library_id),
            team_id=int(paper["team_id"]),
            project_id=paper.get("project_id"),
        )
    )


def queue_markdown_translation(
    paper_id: str,
    *,
    user_id: int,
    input_md_path: str | Path,
    output_md_path: str | Path,
    force: bool = False,
) -> int:
    """Queue a worker job that translates paper Markdown to Chinese."""
    paper = _require_editable_paper(paper_id, user_id)
    job_id = enqueue_job(
        "translate",
        user_id=user_id,
        team_id=int(paper["team_id"]),
        project_id=paper.get("project_id"),
        paper_id=paper_id,
        payload={
            "paper_id": paper_id,
            "input_md_path": str(input_md_path),
            "output_md_path": str(output_md_path),
            "force": bool(force),
        },
    )
    update_paper_status(paper_id, translation_status="queued")
    return int(job_id)


def _require_editable_paper(paper_id: str, user_id: int) -> dict[str, Any]:
    paper = get_accessible_paper(paper_id, user_id, minimum_role="editor")
    if not paper:
        raise PermissionError("没有找到当前论文或无权执行该操作。")
    return paper

