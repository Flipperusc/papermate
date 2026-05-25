"""Run PaperMate SQLite background jobs."""

from __future__ import annotations

import argparse
import sys
import threading
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
from src.job_service import (
    claim_next_job,
    complete_job,
    enqueue_job,
    fail_job,
    latest_job_for_paper,
    requeue_running_jobs,
)
from src.literature_card_service import save_literature_card
from src.markdown_translator import translate_markdown_to_chinese
from src.paper_service import update_paper_status
from src.pdf_parser import parse_pdf
from src.retrieval.bm25_store import BM25Store
from src.vector_store import VectorStore


DEFAULT_WORKER_LANES: tuple[tuple[str, ...], ...] = (
    ("parse",),
    ("index",),
    ("translate", "card", "eval"),
)


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
    payload = job.get("payload") or {}
    paper_id = str(job.get("paper_id") or payload.get("paper_id") or "")
    if not paper_id:
        raise ValueError("parse job missing paper_id")

    paper = load_paper(paper_id)
    if not paper:
        raise ValueError(f"paper not found: {paper_id}")
    update_paper_status(paper_id, parse_status="running")

    include_images = payload_bool(payload, "include_images", default=False)
    parsed_pdf = parse_pdf(paper["save_path"], paper_id, include_images=include_images)
    chunks = chunk_pages(
        paper_id,
        parsed_pdf["pages"],
        chunk_size=settings.rag_chunk_size,
        overlap=settings.rag_chunk_overlap,
        elements=parsed_pdf.get("elements"),
        include_images=include_images,
    )
    total_chars = len("\n\n".join(page["text"] for page in parsed_pdf["pages"]))
    images = parsed_pdf.get("images", []) if include_images else []
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
            "images": images,
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
        "include_images": include_images,
        "image_count": len(images),
    }
    if payload_bool(payload, "auto_index", default=False):
        force_reindex = payload_bool(payload, "force_reindex", default=False)
        existing_index_job = latest_job_for_paper(int(job["team_id"]), paper_id, "index")
        reusable_statuses = {"queued", "running"} if force_reindex else {"queued", "running", "succeeded"}
        if existing_index_job and existing_index_job.get("status") in reusable_statuses:
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


def payload_bool(payload: dict[str, Any], key: str, default: bool = False) -> bool:
    """Read a boolean value from a job payload."""
    value = payload.get(key, default)
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "on"}
    return default


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


def fail_pending_index_jobs(paper_id: str, error_message: str) -> None:
    """Fail queued index jobs that can no longer run because parsing failed."""
    init_db()
    with get_db_connection() as connection:
        connection.execute(
            """
            UPDATE jobs
            SET
                status = 'failed',
                error_message = ?,
                finished_at = CURRENT_TIMESTAMP,
                updated_at = CURRENT_TIMESTAMP
            WHERE paper_id = ?
                AND job_type = 'index'
                AND status = 'queued'
            """,
            (str(error_message or "")[:4000], paper_id),
        )


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


def parse_job_types(raw_types: str | None) -> tuple[str, ...] | None:
    """Parse a comma-separated job type lane filter."""
    if not raw_types:
        return None
    job_types = tuple(job_type.strip().lower() for job_type in raw_types.split(",") if job_type.strip())
    return job_types or None


def worker_lane_label(job_types: tuple[str, ...]) -> str:
    """Return a readable label for a worker lane."""
    return ",".join(job_types)


def run_worker_lane(job_types: tuple[str, ...], poll_interval: float) -> None:
    """Run one named worker lane forever."""
    label = worker_lane_label(job_types)
    print(f"Worker lane started: {label}", flush=True)
    run_worker(poll_interval=poll_interval, job_types=job_types)


def run_worker_lanes(poll_interval: float = 3.0, recover_running: bool = False) -> int:
    """Run default parse/index/other lanes concurrently."""
    if recover_running:
        report_recovered_jobs(requeue_running_jobs())
    threads: list[threading.Thread] = []
    for job_types in DEFAULT_WORKER_LANES:
        thread = threading.Thread(
            target=run_worker_lane,
            args=(job_types, poll_interval),
            name=f"papermate-worker-{worker_lane_label(job_types)}",
            daemon=True,
        )
        thread.start()
        threads.append(thread)

    while True:
        for thread in threads:
            thread.join(timeout=1.0)
        dead_threads = [thread.name for thread in threads if not thread.is_alive()]
        if dead_threads:
            raise RuntimeError(f"worker lane stopped unexpectedly: {', '.join(dead_threads)}")


def run_worker(
    once: bool = False,
    poll_interval: float = 3.0,
    limit: int | None = None,
    job_types: tuple[str, ...] | None = None,
    recover_running: bool = False,
) -> int:
    """Run jobs until stopped, or once when requested."""
    init_db()
    if recover_running:
        report_recovered_jobs(requeue_running_jobs(job_types=job_types))
    completed = 0
    while True:
        job = claim_next_job(job_types=job_types)
        if not job:
            if once:
                return completed
            time.sleep(max(0.5, poll_interval))
            continue

        try:
            result = run_job(job)
        except Exception as exc:
            if job.get("paper_id") and job.get("job_type") == "parse":
                update_paper_status(str(job["paper_id"]), parse_status="failed", index_status="failed")
                fail_pending_index_jobs(str(job["paper_id"]), f"parse failed before indexing: {exc}")
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
    parser.add_argument(
        "--serial",
        action="store_true",
        help="run the legacy single queue loop instead of default parse/index/other lanes",
    )
    parser.add_argument(
        "--types",
        default=None,
        help="comma-separated job types this worker may claim, for example parse or index",
    )
    parser.add_argument(
        "--recover-running",
        action="store_true",
        help="requeue jobs left running by an interrupted worker before processing",
    )
    return parser.parse_args()


def report_recovered_jobs(jobs: list[dict[str, Any]]) -> None:
    """Print a concise startup recovery summary."""
    if not jobs:
        print("Recovered 0 interrupted running jobs.", flush=True)
        return
    job_labels = ", ".join(f"#{job['job_id']} {job['job_type']}" for job in jobs[:10])
    suffix = "" if len(jobs) <= 10 else f", +{len(jobs) - 10} more"
    print(f"Recovered {len(jobs)} interrupted running job(s): {job_labels}{suffix}", flush=True)


def main() -> int:
    """Worker entry point."""
    args = parse_args()
    job_types = parse_job_types(args.types)
    try:
        if job_types or args.once or args.limit is not None or args.serial:
            run_worker(
                once=args.once,
                poll_interval=args.poll_interval,
                limit=args.limit,
                job_types=job_types,
                recover_running=args.recover_running,
            )
        else:
            run_worker_lanes(poll_interval=args.poll_interval, recover_running=args.recover_running)
    except KeyboardInterrupt:
        print("Worker stopped by user.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
