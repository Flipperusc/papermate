"""Paper upload and processing workflow orchestration.

This module is the boundary between UI actions and backend services for paper
lifecycle tasks. It coordinates persistence, duplicate detection, and job
scheduling while remaining independent of Streamlit session state.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
from uuid import uuid4

from src.db import ensure_data_directories, save_paper_and_chunks
from src.errors import ErrorCode, UploadError
from src.job_service import enqueue_job, latest_job_for_paper
from src.paper_service import (
    file_sha256,
    find_team_paper_by_hash,
    format_file_size,
    get_accessible_paper,
    update_paper_status,
)


ACTIVE_JOB_STATUSES = {"queued", "running"}
DUPLICATE_PARSE_STATUSES = ("succeeded", "running", "queued", "failed")


def save_pdf_upload(
    file_name: str,
    file_bytes: bytes,
    *,
    file_digest: str | None = None,
) -> dict[str, Any]:
    """Validate and save uploaded PDF bytes to the configured upload directory."""
    original_name = Path(file_name).name
    if Path(original_name).suffix.lower() != ".pdf":
        raise UploadError(ErrorCode.INVALID_FILE_TYPE)
    if not file_bytes:
        raise UploadError(ErrorCode.EMPTY_FILE)

    paper_id = uuid4().hex
    digest = file_digest or file_sha256(file_bytes)

    try:
        upload_dir, _ = ensure_data_directories()
        save_path = upload_dir / f"{paper_id}_{original_name}"
        save_path.write_bytes(file_bytes)
    except OSError as exc:
        raise UploadError(ErrorCode.SAVE_FAILED, detail=str(exc)) from exc

    return {
        "file_name": original_name,
        "paper_id": paper_id,
        "file_size_bytes": len(file_bytes),
        "file_size": format_file_size(len(file_bytes)),
        "save_path": str(save_path.resolve()),
        "file_sha256": digest,
    }


def queue_index_build_for_paper(
    paper_id: str,
    *,
    user_id: int,
    require_parsed: bool = True,
    payload_extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Queue an index build for one paper, reusing an active index job."""
    paper = get_accessible_paper(paper_id, user_id, minimum_role="editor")
    if not paper:
        raise PermissionError("没有找到当前论文或无权构建索引。")

    parse_status = str(paper.get("parse_status") or "")
    if require_parsed and parse_status != "succeeded":
        raise ValueError("这篇论文还没有解析完成，暂时不能构建索引。")

    latest_job = latest_job_for_paper(int(paper["team_id"]), paper_id, "index")
    latest_status = str((latest_job or {}).get("status") or "").strip().lower()
    if latest_job and latest_status in ACTIVE_JOB_STATUSES:
        update_paper_status(paper_id, index_status=latest_status)
        return {
            "job_id": int(latest_job["job_id"]),
            "reused_existing": True,
            "status": latest_status,
            "state_label": index_state_label(latest_status),
        }

    payload = {"paper_id": paper_id}
    payload.update(payload_extra or {})
    job_id = enqueue_job(
        "index",
        user_id=user_id,
        team_id=int(paper["team_id"]),
        project_id=paper.get("project_id"),
        paper_id=paper_id,
        payload=payload,
    )
    update_paper_status(paper_id, index_status="queued")
    return {
        "job_id": int(job_id),
        "reused_existing": False,
        "status": "queued",
        "state_label": index_state_label("queued"),
    }


def queue_upload_processing_pipeline(
    paper: dict[str, Any],
    *,
    user_id: int,
    include_images: bool = False,
) -> dict[str, Any]:
    """Queue parse and index jobs for an uploaded paper, reusing active jobs."""
    paper_id = str(paper.get("paper_id") or "")
    team_id = int(paper.get("team_id") or 0)
    if not paper_id or team_id <= 0:
        raise ValueError("paper metadata is missing paper_id or team_id")

    parse_status = str(paper.get("parse_status") or "").strip().lower()
    index_status = str(paper.get("index_status") or "").strip().lower()

    latest_parse_job = latest_job_for_paper(team_id, paper_id, "parse")
    latest_parse_status = str((latest_parse_job or {}).get("status") or "").strip().lower()
    parse_job_id: int | None = None
    parse_job_status = ""
    parse_reused = False

    parse_needed = parse_status != "succeeded" or latest_parse_status in ACTIVE_JOB_STATUSES
    if latest_parse_status in ACTIVE_JOB_STATUSES:
        parse_job_id = int(latest_parse_job["job_id"])
        parse_job_status = latest_parse_status
        parse_reused = True
        update_paper_status(paper_id, parse_status=latest_parse_status)
    elif parse_needed:
        update_paper_status(paper_id, parse_status="queued", index_status="queued")
        parse_job_id = enqueue_job(
            "parse",
            user_id=user_id,
            team_id=team_id,
            project_id=paper.get("project_id"),
            paper_id=paper_id,
            payload={
                "paper_id": paper_id,
                "save_path": paper.get("save_path"),
                "auto_created_from_upload": True,
                "auto_index": False,
                "include_images": bool(include_images),
            },
        )
        parse_job_status = "queued"

    latest_index_job = latest_job_for_paper(team_id, paper_id, "index")
    latest_index_status = str((latest_index_job or {}).get("status") or "").strip().lower()
    index_job_id: int | None = None
    index_job_status = ""
    index_reused = False
    should_queue_index = parse_needed or index_status != "succeeded"

    if latest_index_status in ACTIVE_JOB_STATUSES:
        index_job_id = int(latest_index_job["job_id"])
        index_job_status = latest_index_status
        index_reused = True
        update_paper_status(paper_id, index_status=latest_index_status)
    elif should_queue_index:
        index_job_id = enqueue_job(
            "index",
            user_id=user_id,
            team_id=team_id,
            project_id=paper.get("project_id"),
            paper_id=paper_id,
            payload={
                "paper_id": paper_id,
                "auto_created_from_upload": True,
                "waiting_for_parse": parse_status != "succeeded" or parse_job_id is not None,
            },
        )
        update_paper_status(paper_id, index_status="queued")
        index_job_status = "queued"

    return {
        "parse_job_id": parse_job_id,
        "index_job_id": index_job_id,
        "parse_job_status": parse_job_status,
        "index_job_status": index_job_status,
        "parse_reused": parse_reused,
        "index_reused": index_reused,
    }


def save_uploaded_pdf_to_library(
    *,
    file_name: str,
    file_bytes: bytes,
    user_id: int,
    team_id: int,
    project_id: int | None,
    signature: str | None = None,
    include_images: bool = False,
) -> dict[str, Any]:
    """Save an uploaded PDF record and queue its processing pipeline."""
    digest = file_sha256(file_bytes)
    existing_paper = find_team_paper_by_hash(
        team_id,
        digest,
        statuses=DUPLICATE_PARSE_STATUSES,
    )
    if existing_paper:
        pipeline = queue_upload_processing_pipeline(
            existing_paper,
            user_id=user_id,
            include_images=include_images,
        )
        refreshed_paper = get_accessible_paper(str(existing_paper["paper_id"]), user_id)
        return upload_result(
            paper=refreshed_paper or existing_paper,
            pipeline=pipeline,
            reused=True,
            signature=signature,
        )

    saved_file = save_pdf_upload(file_name, file_bytes, file_digest=digest)
    save_paper_and_chunks(
        {
            "paper_id": saved_file["paper_id"],
            "file_name": saved_file["file_name"],
            "file_size_bytes": saved_file["file_size_bytes"],
            "save_path": saved_file["save_path"],
            "owner_user_id": user_id,
            "team_id": team_id,
            "project_id": project_id,
            "file_sha256": saved_file.get("file_sha256", digest),
            "parse_status": "not_started",
            "index_status": "unknown",
            "translation_status": "not_started",
            "page_count": 0,
            "total_chars": 0,
        },
        [],
    )
    paper = get_accessible_paper(saved_file["paper_id"], user_id)
    if not paper:
        raise RuntimeError("saved paper is not accessible after upload")

    pipeline = queue_upload_processing_pipeline(
        paper,
        user_id=user_id,
        include_images=include_images,
    )
    paper = get_accessible_paper(saved_file["paper_id"], user_id) or paper
    return upload_result(
        paper=paper,
        pipeline=pipeline,
        reused=False,
        signature=signature,
    )


def queue_paper_parse(
    paper_id: str,
    *,
    user_id: int,
    include_images: bool = False,
) -> dict[str, Any]:
    """Queue parsing for an existing paper without automatically indexing it."""
    paper = get_accessible_paper(paper_id, user_id, minimum_role="editor")
    if not paper:
        raise PermissionError("没有找到当前论文或无权解析。")

    latest_parse_job = latest_job_for_paper(int(paper["team_id"]), paper_id, "parse")
    latest_status = str((latest_parse_job or {}).get("status") or "").strip().lower()
    if latest_parse_job and latest_status in ACTIVE_JOB_STATUSES:
        return {
            "parse_job_id": int(latest_parse_job["job_id"]),
            "index_job_id": None,
            "reused": True,
            "message": f"解析任务已在队列中：#{latest_parse_job['job_id']}。",
        }

    update_paper_status(paper_id, parse_status="queued", index_status="unknown")
    parse_job_id = enqueue_job(
        "parse",
        user_id=user_id,
        team_id=int(paper["team_id"]),
        project_id=paper.get("project_id"),
        paper_id=paper_id,
        payload={
            "paper_id": paper_id,
            "save_path": paper.get("save_path"),
            "auto_index": False,
            "include_images": bool(include_images),
        },
    )
    image_note = "，包含图片识别" if include_images else ""
    return {
        "parse_job_id": int(parse_job_id),
        "index_job_id": None,
        "reused": False,
        "message": f"已提交解析任务：#{parse_job_id}{image_note}。解析完成后可手动选择是否构建索引。",
    }


def upload_result(
    *,
    paper: dict[str, Any],
    pipeline: dict[str, Any],
    reused: bool,
    signature: str | None,
) -> dict[str, Any]:
    """Build the common result payload returned after an upload action."""
    return {
        "paper": paper,
        "job_id": pipeline.get("parse_job_id"),
        "parse_job_id": pipeline.get("parse_job_id"),
        "index_job_id": pipeline.get("index_job_id"),
        "parse_job_status": pipeline.get("parse_job_status", ""),
        "index_job_status": pipeline.get("index_job_status", ""),
        "parse_reused": pipeline.get("parse_reused", False),
        "index_reused": pipeline.get("index_reused", False),
        "reused": reused,
        "signature": signature,
    }


def index_state_label(status: str) -> str:
    """Return the UI-facing index state label for a queued/running job."""
    return "构建中" if str(status).strip().lower() == "running" else "排队中"
