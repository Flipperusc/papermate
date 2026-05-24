"""Run PaperMate SQLite background jobs."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import settings
from src.card_pipeline import generate_literature_card
from src.chunker import chunk_pages
from src.db import get_db_connection, init_db, save_paper_and_chunks
from src.job_service import claim_next_job, complete_job, enqueue_job, fail_job, latest_job_for_paper
from src.literature_card_service import save_literature_card
from src.markdown_translator import translate_markdown_to_chinese
from src.paper_service import update_paper_status
from src.pdf_parser import parse_pdf
from src.retrieval.bm25_store import BM25Store
from src.vector_store import VectorStore


def run_job(job: dict[str, Any]) -> dict[str, Any]:
    """Run one claimed job and return its result payload."""
    job_type = str(job["job_type"])
    if job_type == "parse":
        return run_parse_job(job)
    if job_type == "index":
        return run_index_job(job)
    if job_type == "translate":
        return run_translate_job(job)
    if job_type == "card":
        return run_card_job(job)
    if job_type == "eval":
        return {"message": "eval job placeholder completed"}
    raise ValueError(f"Unsupported job type: {job_type}")


def run_parse_job(job: dict[str, Any]) -> dict[str, Any]:
    """Parse a saved PDF and persist chunks."""
    paper_id = str(job.get("paper_id") or job.get("payload", {}).get("paper_id") or "")
    if not paper_id:
        raise ValueError("parse job missing paper_id")

    paper = load_paper(paper_id)
    if not paper:
        raise ValueError(f"paper not found: {paper_id}")
    update_paper_status(paper_id, parse_status="running")

    parsed_pdf = parse_pdf(paper["save_path"], paper_id)
    chunks = chunk_pages(
        paper_id,
        parsed_pdf["pages"],
        chunk_size=settings.rag_chunk_size,
        overlap=settings.rag_chunk_overlap,
        elements=parsed_pdf.get("elements"),
    )
    total_chars = len("\n\n".join(page["text"] for page in parsed_pdf["pages"]))
    save_paper_and_chunks(
        {
            "paper_id": paper_id,
            "file_name": paper["file_name"],
            "file_size_bytes": paper["file_size_bytes"],
            "save_path": paper["save_path"],
            "owner_user_id": paper.get("owner_user_id") or job.get("user_id"),
            "team_id": paper.get("team_id") or job.get("team_id"),
            "project_id": paper.get("project_id") or job.get("project_id"),
            "visibility": paper.get("visibility") or "team",
            "file_sha256": paper.get("file_sha256") or "",
            "parse_status": "succeeded",
            "index_status": paper.get("index_status") or "unknown",
            "translation_status": paper.get("translation_status") or "not_started",
            "parser": parsed_pdf.get("parser", ""),
            "markdown_path": parsed_pdf.get("markdown_path"),
            "translated_markdown_path": parsed_pdf.get("translated_markdown_path"),
            "content_list_path": parsed_pdf.get("content_list_path"),
            "images": parsed_pdf.get("images", []),
            "page_count": parsed_pdf["page_count"],
            "total_chars": total_chars,
        },
        chunks,
    )
    result = {
        "paper_id": paper_id,
        "chunk_count": len(chunks),
        "page_count": parsed_pdf["page_count"],
        "markdown_path": parsed_pdf.get("markdown_path"),
    }
    if bool(job.get("payload", {}).get("auto_index")):
        existing_index_job = latest_job_for_paper(int(job["team_id"]), paper_id, "index")
        if existing_index_job and existing_index_job.get("status") in {"queued", "running", "succeeded"}:
            result["index_job_id"] = existing_index_job.get("job_id")
            result["index_status"] = existing_index_job.get("status")
        else:
            update_paper_status(paper_id, index_status="queued")
            result["index_job_id"] = enqueue_job(
                "index",
                user_id=int(job["user_id"]),
                team_id=int(job["team_id"]),
                project_id=job.get("project_id"),
                paper_id=paper_id,
                payload={
                    "paper_id": paper_id,
                    "auto_created_from_parse_job_id": int(job["job_id"]),
                },
            )
            result["index_status"] = "queued"
    return result


def run_index_job(job: dict[str, Any]) -> dict[str, Any]:
    """Build vector and BM25 indexes for a paper."""
    paper_id = str(job.get("paper_id") or job.get("payload", {}).get("paper_id") or "")
    if not paper_id:
        raise ValueError("index job missing paper_id")

    update_paper_status(paper_id, index_status="running")
    chunks = load_chunks(paper_id)
    if not chunks:
        update_paper_status(paper_id, index_status="failed")
        raise ValueError("paper has no chunks; parse it before indexing")

    vector_count: int | None = None
    bm25_result: dict[str, Any] | None = None
    errors: list[str] = []
    try:
        vector_count = VectorStore().add_chunks(chunks)
    except Exception as exc:
        errors.append(f"vector: {exc}")
    try:
        bm25_result = BM25Store(str(settings.bm25_dir)).build_index(paper_id, chunks)
    except Exception as exc:
        errors.append(f"bm25: {exc}")

    if vector_count is None and bm25_result is None:
        update_paper_status(paper_id, index_status="failed")
        raise RuntimeError("; ".join(errors) or "index build failed")

    status = "succeeded" if not errors else "partial"
    update_paper_status(paper_id, index_status=status)
    return {
        "paper_id": paper_id,
        "index_status": status,
        "vector_count": vector_count,
        "bm25_count": bm25_result.get("chunk_count") if bm25_result else None,
        "errors": errors,
    }


def run_translate_job(job: dict[str, Any]) -> dict[str, Any]:
    """Translate one paper Markdown file."""
    payload = job.get("payload") or {}
    paper_id = str(job.get("paper_id") or payload.get("paper_id") or "")
    paper = load_paper(paper_id) if paper_id else None
    input_md_path = payload.get("input_md_path") or (paper or {}).get("markdown_path")
    if not input_md_path:
        raise ValueError("translate job missing input_md_path")
    output_md_path = payload.get("output_md_path") or translated_output_path(str(input_md_path))
    force = bool(payload.get("force", False))

    if paper_id:
        update_paper_status(paper_id, translation_status="running")
    translated_path = translate_markdown_to_chinese(
        input_md_path=str(input_md_path),
        output_md_path=str(output_md_path),
        model=settings.translation_model,
        chunk_size=settings.translation_chunk_size,
        force=force,
        timeout=settings.translation_timeout,
    )
    if paper_id:
        update_paper_status(
            paper_id,
            translation_status="succeeded",
            translated_markdown_path=translated_path,
        )
    return {"paper_id": paper_id, "translated_markdown_path": translated_path}


def run_card_job(job: dict[str, Any]) -> dict[str, Any]:
    """Generate and save one literature card."""
    payload = job.get("payload") or {}
    paper_id = str(job.get("paper_id") or payload.get("paper_id") or "")
    if not paper_id:
        raise ValueError("card job missing paper_id")
    markdown = generate_literature_card(paper_id)
    card_id = save_literature_card(
        paper_id,
        markdown,
        user_id=job.get("user_id"),
        library_id=payload.get("library_id"),
        team_id=job.get("team_id"),
        project_id=job.get("project_id"),
    )
    return {"paper_id": paper_id, "card_id": card_id}


def load_paper(paper_id: str) -> dict[str, Any] | None:
    """Load one paper by id without permission filtering for worker use."""
    init_db()
    with get_db_connection() as connection:
        row = connection.execute("SELECT * FROM papers WHERE paper_id = ?", (paper_id,)).fetchone()
    return dict(row) if row else None


def load_chunks(paper_id: str) -> list[dict[str, Any]]:
    """Load chunks for worker use."""
    init_db()
    with get_db_connection() as connection:
        rows = connection.execute(
            """
            SELECT
                chunk_id,
                paper_id,
                chunk_index,
                page_num,
                section_title,
                text,
                chunk_type,
                images_json,
                tables_json
            FROM chunks
            WHERE paper_id = ?
            ORDER BY chunk_index ASC
            """,
            (paper_id,),
        ).fetchall()
    return [dict(row) for row in rows]


def translated_output_path(markdown_path: str) -> str:
    """Return the non-destructive Chinese Markdown output path."""
    source = Path(markdown_path)
    if source.suffix.lower() == ".md":
        return str(source.with_name(f"{source.stem}.zh.md"))
    return str(source.with_name(f"{source.name}.zh.md"))


def run_worker(once: bool = False, poll_interval: float = 3.0, limit: int | None = None) -> int:
    """Run jobs until stopped, or once when requested."""
    init_db()
    completed = 0
    while True:
        job = claim_next_job()
        if not job:
            if once:
                return completed
            time.sleep(max(0.5, poll_interval))
            continue

        try:
            result = run_job(job)
        except Exception as exc:
            if job.get("paper_id") and job.get("job_type") == "parse":
                update_paper_status(str(job["paper_id"]), parse_status="failed")
            elif job.get("paper_id") and job.get("job_type") == "index":
                update_paper_status(str(job["paper_id"]), index_status="failed")
            elif job.get("paper_id") and job.get("job_type") == "translate":
                update_paper_status(str(job["paper_id"]), translation_status="failed")
            fail_job(int(job["job_id"]), str(exc))
            print(f"FAILED #{job['job_id']} {job['job_type']}: {exc}", flush=True)
        else:
            complete_job(int(job["job_id"]), result)
            print(f"DONE #{job['job_id']} {job['job_type']}: {result}", flush=True)
        completed += 1
        if once or (limit is not None and completed >= limit):
            return completed


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Run PaperMate SQLite background jobs.")
    parser.add_argument("--once", action="store_true", help="process at most one queued job and exit")
    parser.add_argument("--limit", type=int, default=None, help="process at most N jobs and exit")
    parser.add_argument("--poll-interval", type=float, default=3.0, help="seconds between polls")
    return parser.parse_args()


def main() -> int:
    """Worker entry point."""
    args = parse_args()
    try:
        run_worker(once=args.once, poll_interval=args.poll_interval, limit=args.limit)
    except KeyboardInterrupt:
        print("Worker stopped by user.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
