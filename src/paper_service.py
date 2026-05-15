"""Paper library persistence and access helpers."""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path
from typing import Any

from config import settings
from src.db import get_db_connection, get_paper_chunks, init_db
from src.team_service import require_team_role


def file_sha256(file_bytes: bytes) -> str:
    """Return the SHA-256 digest for uploaded file bytes."""
    return hashlib.sha256(file_bytes).hexdigest()


def format_file_size(size_bytes: int) -> str:
    """Format a byte count for display."""
    if size_bytes < 1024:
        return f"{size_bytes} B"
    if size_bytes < 1024 * 1024:
        return f"{size_bytes / 1024:.1f} KB"
    return f"{size_bytes / (1024 * 1024):.1f} MB"


def list_accessible_papers(
    user_id: int,
    team_id: int,
    project_id: int | None = None,
    uploader_user_id: int | None = None,
    parse_status: str | None = None,
    index_status: str | None = None,
    search_query: str = "",
    limit: int = 300,
) -> list[dict[str, Any]]:
    """Return papers visible to one team member."""
    require_team_role(user_id, team_id, "viewer")
    where_clauses = ["p.team_id = ?"]
    parameters: list[Any] = [int(team_id)]
    if project_id is not None:
        where_clauses.append("p.project_id = ?")
        parameters.append(int(project_id))
    if uploader_user_id is not None:
        where_clauses.append("p.owner_user_id = ?")
        parameters.append(int(uploader_user_id))
    if parse_status:
        where_clauses.append("p.parse_status = ?")
        parameters.append(parse_status)
    if index_status:
        where_clauses.append("p.index_status = ?")
        parameters.append(index_status)
    if search_query.strip():
        where_clauses.append("(p.file_name LIKE ? OR p.paper_id LIKE ?)")
        needle = f"%{search_query.strip()}%"
        parameters.extend([needle, needle])
    parameters.append(max(1, int(limit)))

    init_db()
    with get_db_connection() as connection:
        rows = connection.execute(
            f"""
            SELECT
                p.paper_id,
                p.file_name,
                p.file_size_bytes,
                p.save_path,
                p.owner_user_id,
                u.username AS owner_username,
                p.team_id,
                t.name AS team_name,
                p.project_id,
                pr.name AS project_name,
                p.visibility,
                p.file_sha256,
                p.parse_status,
                p.index_status,
                p.translation_status,
                p.parser,
                p.markdown_path,
                p.translated_markdown_path,
                p.content_list_path,
                p.images_json,
                p.page_count,
                p.total_chars,
                p.created_at,
                p.updated_at,
                COUNT(c.chunk_id) AS chunk_count
            FROM papers p
            LEFT JOIN users u ON u.user_id = p.owner_user_id
            LEFT JOIN teams t ON t.team_id = p.team_id
            LEFT JOIN projects pr ON pr.project_id = p.project_id
            LEFT JOIN chunks c ON c.paper_id = p.paper_id
            WHERE {' AND '.join(where_clauses)}
            GROUP BY p.paper_id
            ORDER BY p.updated_at DESC, p.created_at DESC
            LIMIT ?
            """,
            parameters,
        ).fetchall()
    return [dict(row) for row in rows]


def get_accessible_paper(paper_id: str, user_id: int, minimum_role: str = "viewer") -> dict[str, Any] | None:
    """Return one paper if user has access to its team."""
    init_db()
    with get_db_connection() as connection:
        row = connection.execute(
            """
            SELECT
                p.*,
                u.username AS owner_username,
                t.name AS team_name,
                pr.name AS project_name
            FROM papers p
            LEFT JOIN users u ON u.user_id = p.owner_user_id
            LEFT JOIN teams t ON t.team_id = p.team_id
            LEFT JOIN projects pr ON pr.project_id = p.project_id
            WHERE p.paper_id = ?
            """,
            (paper_id,),
        ).fetchone()
    if not row:
        return None
    paper = dict(row)
    if paper.get("team_id") is None:
        return None
    require_team_role(user_id, int(paper["team_id"]), minimum_role)
    return paper


def find_team_duplicate_paper(team_id: int, file_digest: str) -> dict[str, Any] | None:
    """Return an existing parsed paper with the same file hash in a team."""
    return find_team_paper_by_hash(team_id, file_digest, statuses=["succeeded"])


def find_team_paper_by_hash(
    team_id: int,
    file_digest: str,
    statuses: list[str] | tuple[str, ...] | None = None,
) -> dict[str, Any] | None:
    """Return an existing team paper with the same file hash."""
    if not file_digest:
        return None
    parameters: list[Any] = [int(team_id), file_digest]
    status_clause = ""
    if statuses:
        placeholders = ", ".join("?" for _ in statuses)
        status_clause = f"AND parse_status IN ({placeholders})"
        parameters.extend(str(status) for status in statuses)
    init_db()
    with get_db_connection() as connection:
        row = connection.execute(
            f"""
            SELECT *
            FROM papers
            WHERE team_id = ?
                AND file_sha256 = ?
                {status_clause}
            ORDER BY
                CASE parse_status
                    WHEN 'succeeded' THEN 1
                    WHEN 'running' THEN 2
                    WHEN 'queued' THEN 3
                    WHEN 'failed' THEN 4
                    ELSE 5
                END,
                updated_at DESC,
                created_at DESC
            LIMIT 1
            """,
            parameters,
        ).fetchone()
    return dict(row) if row else None


def update_paper_status(
    paper_id: str,
    parse_status: str | None = None,
    index_status: str | None = None,
    translation_status: str | None = None,
    translated_markdown_path: str | None = None,
) -> None:
    """Update paper status fields."""
    assignments: list[str] = []
    parameters: list[Any] = []
    if parse_status is not None:
        assignments.append("parse_status = ?")
        parameters.append(parse_status)
    if index_status is not None:
        assignments.append("index_status = ?")
        parameters.append(index_status)
    if translation_status is not None:
        assignments.append("translation_status = ?")
        parameters.append(translation_status)
    if translated_markdown_path is not None:
        assignments.append("translated_markdown_path = ?")
        parameters.append(translated_markdown_path)
    if not assignments:
        return

    assignments.append("updated_at = CURRENT_TIMESTAMP")
    parameters.append(paper_id)
    init_db()
    with get_db_connection() as connection:
        connection.execute(
            f"""
            UPDATE papers
            SET {', '.join(assignments)}
            WHERE paper_id = ?
            """,
            parameters,
        )


def delete_team_papers(
    user_id: int,
    team_id: int,
    paper_ids: list[str] | tuple[str, ...],
    delete_files: bool = True,
) -> dict[str, Any]:
    """Delete papers from one team after checking editor permissions."""
    require_team_role(user_id, team_id, "editor")
    clean_ids = [str(paper_id).strip() for paper_id in paper_ids if str(paper_id).strip()]
    if not clean_ids:
        return {"deleted": 0, "requested": 0, "missing": [], "file_errors": []}

    init_db()
    placeholders = ", ".join("?" for _ in clean_ids)
    with get_db_connection() as connection:
        rows = connection.execute(
            f"""
            SELECT *
            FROM papers
            WHERE team_id = ?
                AND paper_id IN ({placeholders})
            """,
            [int(team_id), *clean_ids],
        ).fetchall()
        papers = [dict(row) for row in rows]
        found_ids = {str(paper["paper_id"]) for paper in papers}
        missing = [paper_id for paper_id in clean_ids if paper_id not in found_ids]

        if found_ids:
            found_placeholders = ", ".join("?" for _ in found_ids)
            found_parameters = list(found_ids)
            connection.execute(
                f"DELETE FROM jobs WHERE paper_id IN ({found_placeholders})",
                found_parameters,
            )
            connection.execute(
                f"DELETE FROM papers WHERE paper_id IN ({found_placeholders})",
                found_parameters,
            )

    file_errors: list[str] = []
    if delete_files:
        for paper in papers:
            file_errors.extend(delete_paper_files(paper))

    return {
        "deleted": len(found_ids),
        "requested": len(clean_ids),
        "missing": missing,
        "file_errors": file_errors,
    }


def delete_paper_files(paper: dict[str, Any]) -> list[str]:
    """Best-effort cleanup of local artifacts for one paper."""
    errors: list[str] = []
    paths: list[Path] = []
    for key in ("save_path", "markdown_path", "translated_markdown_path", "content_list_path"):
        value = paper.get(key)
        if value:
            paths.append(Path(str(value)))

    for image in load_images_json(paper.get("images_json")):
        value = image.get("path") if isinstance(image, dict) else None
        if value:
            paths.append(Path(str(value)))

    for path in paths:
        errors.extend(delete_file_if_inside_data(path))

    markdown_path = paper.get("markdown_path")
    paper_id = str(paper.get("paper_id") or "")
    if markdown_path and paper_id:
        md_parent = Path(str(markdown_path)).parent
        errors.extend(delete_directory_if_safe(md_parent, paper_id))

    for bm25_path in (
        settings.bm25_dir / f"{paper_id}_payloads.json",
        settings.bm25_dir / f"{paper_id}_bm25.pkl",
    ):
        errors.extend(delete_file_if_inside_data(bm25_path))

    return errors


def delete_file_if_inside_data(path: Path) -> list[str]:
    """Delete a file only if it resolves under the configured data directory."""
    try:
        resolved = path.resolve()
        data_root = settings.data_dir.resolve()
        if not resolved.is_relative_to(data_root):
            return [f"skip outside data dir: {resolved}"]
        if resolved.exists() and resolved.is_file():
            resolved.unlink()
    except OSError as exc:
        return [f"{path}: {exc}"]
    return []


def delete_directory_if_safe(path: Path, expected_name: str) -> list[str]:
    """Delete one generated artifact directory when it is inside markdown_dir and named as expected."""
    try:
        resolved = path.resolve()
        markdown_root = settings.markdown_dir.resolve()
        if resolved.name != expected_name or not resolved.is_relative_to(markdown_root):
            return []
        if resolved.exists() and resolved.is_dir():
            shutil.rmtree(resolved)
    except OSError as exc:
        return [f"{path}: {exc}"]
    return []


def paper_to_processed_pdf(paper: dict[str, Any]) -> dict[str, Any]:
    """Build the in-memory processed_pdf shape from persisted paper metadata."""
    chunks = get_paper_chunks(str(paper["paper_id"]))
    images = load_images_json(paper.get("images_json"))
    markdown_path = str(paper.get("markdown_path") or "")
    translated_path = str(paper.get("translated_markdown_path") or "")
    markdown = ""
    if markdown_path and Path(markdown_path).exists():
        try:
            markdown = Path(markdown_path).read_text(encoding="utf-8")
        except OSError:
            markdown = ""
    if not markdown.strip() and chunks:
        markdown = chunks_to_markdown(chunks)
    preview = "\n\n".join(str(chunk.get("text") or "") for chunk in chunks)[:1000]
    parsed_pdf: dict[str, Any] = {
        "paper_id": paper["paper_id"],
        "page_count": int(paper.get("page_count") or 0),
        "pages": [],
        "parser": paper.get("parser") or "persisted",
        "markdown": markdown,
        "markdown_path": markdown_path or None,
        "translated_markdown_path": translated_path or None,
        "content_list_path": paper.get("content_list_path"),
        "images": images,
    }
    return {
        "signature": f"persisted:{paper['paper_id']}:{paper.get('updated_at') or ''}",
        "saved_file": {
            "file_name": paper.get("file_name") or "",
            "paper_id": paper["paper_id"],
            "file_size_bytes": int(paper.get("file_size_bytes") or 0),
            "file_size": format_file_size(int(paper.get("file_size_bytes") or 0)),
            "save_path": paper.get("save_path") or "",
            "file_sha256": paper.get("file_sha256") or "",
        },
        "parsed_pdf": parsed_pdf,
        "chunks": chunks,
        "total_chars": int(paper.get("total_chars") or 0),
        "preview": preview,
        "db_save_failed": False,
        "team_id": paper.get("team_id"),
        "project_id": paper.get("project_id"),
        "parse_status": paper.get("parse_status") or "unknown",
        "index_status": paper.get("index_status") or "unknown",
        "translation_status": paper.get("translation_status") or "not_started",
    }


def chunks_to_markdown(chunks: list[dict[str, Any]]) -> str:
    """Rebuild readable full text from persisted chunks when the Markdown file is unavailable."""
    parts: list[str] = []
    last_page: int | None = None
    last_section = ""

    for chunk in chunks:
        text = str(chunk.get("text") or "").strip()
        if not text:
            continue

        section = str(chunk.get("section_title") or "").strip()
        if section and section != last_section:
            parts.append(f"## {section}")
            last_section = section

        page_num = int(chunk.get("page_num") or 0)
        if page_num and page_num != last_page:
            parts.append(f"<!-- page {page_num} -->")
            last_page = page_num

        parts.append(text)

    return "\n\n".join(parts)


def load_images_json(value: Any) -> list[dict[str, Any]]:
    """Parse stored images metadata."""
    if isinstance(value, list):
        return value
    if not value:
        return []
    try:
        parsed = json.loads(str(value))
    except (TypeError, ValueError):
        return []
    return parsed if isinstance(parsed, list) else []
