"""Streamlit entry point for PaperMate."""

from __future__ import annotations

import base64
import hashlib
import html
import hmac
import io
import json
import os
import sqlite3
import re
import zipfile
from pathlib import Path
from typing import Any, Protocol
from urllib.parse import unquote
from uuid import uuid4

import streamlit as st
import streamlit.components.v1 as components

from config import settings
from src import __version__
from src.auth_service import authenticate_user, create_user, get_user_by_id
from src.bilingual_aligner import align_markdown_bilingual
from src.card_pipeline import generate_literature_card
from src.chunker import CHUNKER_VERSION, chunk_pages
from src.db import ensure_data_directories, init_db, save_paper_and_chunks, save_qa_log
from src.errors import (
    AppError,
    EmbeddingError,
    ErrorCode,
    LLMError,
    MinerUError,
    PdfParseError,
    UploadError,
    VectorStoreError,
)
from src.feedback_service import FEEDBACK_OPTIONS, list_bad_cases, list_feedback_records, save_feedback
from src.job_service import (
    cancel_job,
    cancel_queued_job,
    clear_team_queued_jobs,
    enqueue_job,
    latest_job_for_paper,
    list_jobs,
    queue_progress_summary,
    retry_job,
)
from src.literature_card_service import (
    CARD_FIELD_LABELS,
    claim_unassigned_literature_cards,
    create_card_library,
    delete_literature_card,
    delete_literature_cards,
    ensure_default_card_library,
    get_card_library,
    get_literature_card,
    get_literature_card_by_paper,
    list_card_libraries,
    list_literature_cards,
    save_literature_card,
    update_card_library,
    update_literature_card,
)
from src.logger import get_logger
from src.markdown_translator import translate_markdown_to_chinese
from src.paper_service import (
    chunks_to_markdown,
    delete_team_papers,
    file_sha256,
    find_team_paper_by_hash,
    get_accessible_paper,
    list_accessible_papers,
    paper_to_processed_pdf,
    update_paper_status,
)
from src.pdf_parser import parse_pdf
from src.rag_pipeline import answer_question
from src.retrieval.bm25_store import BM25Store
from src.team_service import (
    TEAM_ROLES,
    add_team_member_by_username,
    can_manage_team,
    can_write,
    create_project,
    create_team,
    ensure_user_workspace,
    get_user_team_role,
    list_projects,
    list_team_members,
    list_user_teams,
    remove_team_member,
    update_team_member_role,
)
from src.vector_store import VectorStore


logger = get_logger(__name__)
BILINGUAL_ALIGNMENT_CACHE_VERSION = "header-image-notices-v2"
QUEUE_REFRESH_SECONDS = 5
QUEUE_HOVER_LIMIT = 10
QUEUE_CANCEL_QUERY_PARAM = "pm_cancel_queue_job"
SOURCE_READING_MODE = "\u539f\u6587"
SOURCE_JUMP_LABEL = "\u56de\u5230\u539f\u6587"

CARD_PALETTES = [
    {"top": "#eaf3ff", "accent": "#2563eb", "field": "#f6faff"},
    {"top": "#eafaf4", "accent": "#0f766e", "field": "#f3fbf8"},
    {"top": "#fff4dc", "accent": "#b45309", "field": "#fffaf0"},
    {"top": "#fff0f4", "accent": "#be123c", "field": "#fff7f9"},
    {"top": "#f1f0ff", "accent": "#6d28d9", "field": "#faf9ff"},
    {"top": "#eefdf3", "accent": "#15803d", "field": "#f7fef9"},
]


class UploadedFile(Protocol):
    """Minimal uploaded-file interface used by the save helper."""

    name: str

    def getvalue(self) -> bytes:
        """Return uploaded file bytes."""
        ...


def inject_styles() -> None:
    """Inject compact UI styles."""
    st.markdown(
        """
        <style>
        :root {
            --pm-bg: #f7f7f5;
            --pm-surface: #ffffff;
            --pm-surface-soft: #f3f4f2;
            --pm-border: #deded8;
            --pm-text: #202123;
            --pm-muted: #6b6f76;
            --pm-accent-main: #10a37f;
            --pm-accent-strong: #0f7f66;
            --pm-blue: #2563eb;
        }
        .stApp {
            background: var(--pm-bg);
            color: var(--pm-text);
        }
        html, body,
        [data-testid="stAppViewContainer"],
        [data-testid="stHeader"] {
            background: var(--pm-bg) !important;
            color: var(--pm-text) !important;
        }
        [data-testid="stDecoration"] {
            display: none;
        }
        [data-testid="stToolbar"],
        [data-testid="collapsedControl"],
        [data-testid="stSidebarCollapsedControl"] {
            visibility: visible !important;
            opacity: 1 !important;
            pointer-events: auto !important;
            z-index: 1000000 !important;
        }
        [data-testid="collapsedControl"] button,
        [data-testid="stSidebarCollapsedControl"] button {
            border: 1px solid var(--pm-border) !important;
            border-radius: 8px !important;
            background: #ffffff !important;
            box-shadow: 0 8px 20px rgba(15, 23, 42, 0.10) !important;
        }
        .block-container {
            padding-top: 1rem;
            padding-bottom: 2.4rem;
            max-width: 1560px;
        }
        section[data-testid="stSidebar"] {
            background: #f4f4f1 !important;
            border-right: 1px solid #deded8;
            color: var(--pm-text);
        }
        section[data-testid="stSidebar"] > div {
            padding-top: 18px;
        }
        section[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] h3 {
            font-size: 17px;
            letter-spacing: 0;
            margin-bottom: 4px;
            color: #202123;
        }
        section[data-testid="stSidebar"] [role="radiogroup"] {
            display: grid;
            gap: 6px;
            margin-top: 10px;
        }
        section[data-testid="stSidebar"] [role="radiogroup"] label {
            border-radius: 10px;
            padding: 8px 10px;
            color: #202123;
            transition: background 120ms ease, box-shadow 120ms ease;
        }
        section[data-testid="stSidebar"] [role="radiogroup"] label > div:first-child {
            margin-right: 8px;
        }
        section[data-testid="stSidebar"] [role="radiogroup"] label:hover {
            background: #e9e9e4;
        }
        section[data-testid="stSidebar"] [role="radiogroup"] label:has(input:checked) {
            background: #e3f2ec;
            box-shadow: inset 3px 0 0 var(--pm-accent-main);
        }
        h1, h2, h3, h4, h5 {
            color: var(--pm-text);
            letter-spacing: 0;
        }
        p, li, label, span, div {
            color-scheme: light;
        }
        div[data-testid="stMetric"] {
            background: var(--pm-surface);
            border: 1px solid var(--pm-border);
            border-radius: 8px;
            padding: 10px 12px;
        }
        .stButton > button,
        .stDownloadButton > button,
        button[kind="primary"] {
            border-radius: 8px;
            border: 1px solid transparent;
            font-weight: 600;
        }
        .stButton > button[kind="primary"],
        .stDownloadButton > button[kind="primary"] {
            background: var(--pm-accent-main);
            border-color: var(--pm-accent-main);
        }
        .stButton > button:hover,
        .stDownloadButton > button:hover {
            border-color: var(--pm-accent-main);
        }
        div[data-testid="stExpander"] {
            border-color: var(--pm-border);
            border-radius: 8px;
            background: var(--pm-surface);
        }
        div[data-testid="stTextArea"] textarea,
        div[data-testid="stTextInput"] input {
            border-radius: 8px;
            border-color: var(--pm-border);
            background: #ffffff;
            color: var(--pm-text);
        }
        .pm-sidebar-brand {
            border-radius: 10px;
            padding: 10px 10px 12px 10px;
            background: #ffffff;
            border: 1px solid #e3e3de;
            margin-bottom: 10px;
        }
        .pm-sidebar-brand-title {
            font-size: 18px;
            font-weight: 750;
            line-height: 1.2;
            color: #202123;
            margin-bottom: 3px;
        }
        .pm-sidebar-brand-subtitle {
            font-size: 12px;
            color: #6b6f76;
            line-height: 1.45;
        }
        .pm-sidebar-section {
            margin: 16px 0 6px 0;
            color: #6b6f76;
            font-size: 12px;
            font-weight: 700;
        }
        .pm-hero {
            border: 1px solid #cfe4dc;
            border-radius: 8px;
            padding: 18px 22px;
            background: #ecf7f2;
            margin-bottom: 16px;
            box-shadow: 0 1px 2px rgba(15, 23, 42, 0.04);
        }
        .pm-hero h1 {
            margin: 0 0 4px 0;
            font-size: 29px;
            letter-spacing: 0;
        }
        .pm-hero p {
            margin: 0;
            color: #3f5f55;
        }
        .pm-card {
            border: 1px solid var(--pm-border);
            border-radius: 8px;
            padding: 0;
            background: #ffffff;
            box-shadow: 0 10px 24px rgba(15, 23, 42, 0.06);
            margin-bottom: 14px;
            overflow: hidden;
        }
        .pm-card-top {
            padding: 16px 18px 14px 18px;
            border-left: 6px solid var(--pm-accent);
            background: var(--pm-top);
        }
        .pm-card-title {
            font-size: 20px;
            line-height: 1.35;
            font-weight: 700;
            color: #172033;
            margin-bottom: 10px;
        }
        .pm-meta {
            display: flex;
            gap: 8px;
            flex-wrap: wrap;
            margin-bottom: 14px;
        }
        .pm-chip {
            border: 1px solid rgba(15, 23, 42, 0.08);
            border-radius: 999px;
            padding: 4px 10px;
            font-size: 12px;
            color: #172033;
            background: rgba(255, 255, 255, 0.74);
        }
        .pm-card-body {
            display: grid;
            gap: 10px;
            padding: 14px 16px 16px 16px;
        }
        .pm-field {
            border-radius: 8px;
            padding: 10px 12px;
            background: var(--pm-field-bg);
        }
        .pm-field-label {
            font-size: 12px;
            color: var(--pm-accent);
            font-weight: 700;
            margin-bottom: 4px;
        }
        .pm-field-value {
            color: #172033;
            line-height: 1.65;
            white-space: pre-wrap;
        }
        .pm-small {
            color: #64748b;
            font-size: 13px;
        }
        .pm-source-anchor {
            display: block;
            scroll-margin-top: 16px;
            height: 1px;
        }
        .pm-source-link {
            display: inline-flex;
            align-items: center;
            border: 1px solid #b7d5f8;
            border-radius: 999px;
            padding: 3px 10px;
            background: #eff6ff;
            color: #1d4ed8;
            font-size: 13px;
            text-decoration: none;
        }
        iframe.pm-pdf {
            border: 1px solid #d8dee9;
            border-radius: 8px;
            background: #f8fafc;
        }
        .pm-auth-shell {
            max-width: 960px;
            margin: 24px auto 0 auto;
            display: grid;
            grid-template-columns: 0.92fr 1.08fr;
            gap: 22px;
            align-items: stretch;
        }
        .pm-auth-panel,
        .pm-auth-copy,
        .pm-library-panel {
            border: 1px solid var(--pm-border);
            border-radius: 8px;
            background: #ffffff;
            box-shadow: 0 10px 24px rgba(15, 23, 42, 0.06);
        }
        .pm-auth-copy {
            padding: 22px;
            background: #f2faf6;
            border-color: #cae7dc;
        }
        .pm-auth-copy h2 {
            font-size: 24px;
            margin: 0 0 8px 0;
        }
        .pm-auth-copy p {
            color: #3f5f55;
            line-height: 1.72;
            margin: 0 0 12px 0;
        }
        .pm-auth-panel {
            padding: 18px 20px 8px 20px;
        }
        .pm-user-pill {
            border: 1px solid #cfe4dc;
            border-radius: 8px;
            background: #ecf7f2;
            padding: 8px 10px;
            color: #235347;
            font-size: 13px;
            margin: 8px 0 12px 0;
        }
        .pm-library-panel {
            padding: 14px 16px;
            margin-bottom: 16px;
        }
        .pm-library-panel h3 {
            font-size: 18px;
            margin: 0 0 8px 0;
        }
        .pm-library-panel p {
            margin: 0;
            color: var(--pm-muted);
            line-height: 1.55;
        }
        div[data-testid="stDialog"] div[role="dialog"] {
            border-radius: 14px;
            border: 1px solid #cae7dc;
            box-shadow: 0 22px 70px rgba(15, 23, 42, 0.24);
        }
        .pm-welcome {
            padding: 4px 2px 2px 2px;
        }
        .pm-welcome-kicker {
            display: inline-flex;
            border-radius: 999px;
            padding: 4px 10px;
            background: #ecf7f2;
            color: #0f7f66;
            font-size: 12px;
            font-weight: 700;
            margin-bottom: 10px;
        }
        .pm-welcome h2 {
            font-size: 25px;
            margin: 0 0 8px 0;
        }
        .pm-welcome p,
        .pm-welcome li {
            color: #475569;
            line-height: 1.68;
        }
        @media (max-width: 860px) {
            .pm-auth-shell {
                grid-template-columns: 1fr;
                margin-top: 12px;
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


















def current_user() -> dict[str, Any] | None:
    """Return the authenticated user stored in the Streamlit session."""
    user = st.session_state.get("current_user")
    if not isinstance(user, dict) or not user.get("user_id"):
        return None

    user_id = int(user["user_id"])
    fresh_user = get_user_by_id(user_id)
    if not fresh_user:
        st.session_state.pop("current_user", None)
        return None
    return {"user_id": user_id, "username": str(fresh_user["username"])}


def current_user_id() -> int:
    """Return the current user id or raise if the app is not authenticated."""
    user = current_user()
    if not user:
        raise RuntimeError("Current user is required.")
    return int(user["user_id"])


def set_current_user(user: dict[str, Any]) -> None:
    """Persist authenticated user data in session state."""
    st.session_state["current_user"] = {
        "user_id": int(user["user_id"]),
        "username": str(user["username"]),
    }


def prepare_user_workspace(user_id: int) -> None:
    """Ensure required per-user card storage exists."""
    workspace = ensure_user_workspace(user_id)
    team_id = int(workspace["team_id"])
    ensure_default_card_library(user_id, team_id=team_id)
    claim_unassigned_literature_cards(user_id, team_id=team_id)
    st.session_state[f"workspace_prepared_{int(user_id)}"] = True


def prepare_user_workspace_once(user_id: int) -> None:
    """Prepare a user's writable workspace at most once per Streamlit session."""
    state_key = f"workspace_prepared_{int(user_id)}"
    if st.session_state.get(state_key):
        return
    prepare_user_workspace(user_id)


def current_team_context() -> dict[str, Any]:
    """Return the selected team and project for the current user."""
    user_id = current_user_id()
    teams = list_user_teams(user_id)
    if not teams:
        workspace = ensure_user_workspace(user_id)
        teams = list_user_teams(user_id)
        if not teams:
            raise RuntimeError(f"无法创建团队工作区：{workspace}")

    team_ids = [int(team["team_id"]) for team in teams]
    pending_team_id = st.session_state.pop("pending_current_team_id", None)
    selected_team_id = pending_team_id if pending_team_id is not None else st.session_state.get("current_team_id")
    should_sync_team_id = pending_team_id is not None
    if int(selected_team_id or 0) not in team_ids:
        selected_team_id = team_ids[0]
        should_sync_team_id = True
    if should_sync_team_id:
        st.session_state["current_team_id"] = int(selected_team_id)

    projects = list_projects(user_id, int(selected_team_id))
    project_ids = [int(project["project_id"]) for project in projects]
    selected_project_id = st.session_state.get("current_project_id")
    if project_ids and int(selected_project_id or 0) not in project_ids:
        selected_project_id = project_ids[0]
        st.session_state["current_project_id"] = selected_project_id

    team = next(team for team in teams if int(team["team_id"]) == int(selected_team_id))
    project = (
        next((project for project in projects if int(project["project_id"]) == int(selected_project_id)), None)
        if project_ids
        else None
    )
    return {
        "team": team,
        "teams": teams,
        "team_id": int(selected_team_id),
        "role": str(team.get("role") or "viewer"),
        "projects": projects,
        "project": project,
        "project_id": int(project["project_id"]) if project else None,
    }


def clear_authenticated_session() -> None:
    """Clear user-scoped session state on logout."""
    keep_keys = {"welcome_seen"}
    for key in list(st.session_state.keys()):
        if key not in keep_keys:
            del st.session_state[key]


def render_sidebar_navigation(user: dict[str, Any]) -> str:
    """Backward-compatible sidebar wrapper."""
    return render_sidebar(user)


def navigate_to_page(page_label: str) -> None:
    """Request a sidebar navigation change before the next radio widget render."""
    st.session_state["pm_pending_nav_page"] = page_label
    st.rerun()


def format_file_size(size_bytes: int) -> str:
    """Format a byte count for display."""
    if size_bytes < 1024:
        return f"{size_bytes} B"
    if size_bytes < 1024 * 1024:
        return f"{size_bytes / 1024:.1f} KB"
    return f"{size_bytes / (1024 * 1024):.1f} MB"


def save_uploaded_pdf(uploaded_file: UploadedFile) -> dict[str, Any]:
    """Validate and save an uploaded PDF file."""
    original_name = Path(uploaded_file.name).name
    if Path(original_name).suffix.lower() != ".pdf":
        raise UploadError(ErrorCode.INVALID_FILE_TYPE)

    file_bytes = uploaded_file.getvalue()
    if not file_bytes:
        raise UploadError(ErrorCode.EMPTY_FILE)

    paper_id = uuid4().hex
    digest = file_sha256(file_bytes)

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


def build_text_preview(parsed_pdf: dict[str, Any], limit: int = 1000) -> tuple[int, str]:
    """Build total character count and preview from parsed pages."""
    pages = parsed_pdf["pages"]
    full_text = "\n\n".join(page["text"] for page in pages)
    return len(full_text), full_text[:limit]


def normalize_markdown_image_ref(value: str) -> str:
    """Normalize a Markdown image reference for matching extracted images."""
    return unquote(str(value or "").split()[0].strip("\"'").replace("\\", "/"))


def image_source_lookup(images: list[dict[str, Any]] | None) -> dict[str, dict[str, Any]]:
    """Index extracted images by archive path, source path, and filename."""
    lookup: dict[str, dict[str, Any]] = {}
    for image in images or []:
        refs = list(image.get("source_paths") or [])
        for key in ("archive_name", "file_name"):
            value = image.get(key)
            if isinstance(value, str) and value.strip():
                refs.append(value)

        for ref in refs:
            normalized = normalize_markdown_image_ref(str(ref))
            if not normalized:
                continue
            lookup.setdefault(normalized, image)
            lookup.setdefault(Path(normalized).name, image)
    return lookup


def visual_kind_from_label(label: str) -> str:
    """Infer visual kind from a Markdown label."""
    label = str(label or "")
    if re.search(r"(?:表|table)", label, flags=re.IGNORECASE):
        return "table"
    if re.search(r"(?:图片|图|image|fig(?:ure)?\.?)", label, flags=re.IGNORECASE):
        return "image"
    if re.search(r"(?:公式|equation|formula)", label, flags=re.IGNORECASE):
        return "equation"
    return ""


def visual_label_image_index(label: str) -> int | None:
    """Return the visible image index encoded in a figure label."""
    match = re.search(
        r"(?:图片|图|image|fig(?:ure)?\.?)\s*(\d+)",
        str(label or ""),
        flags=re.IGNORECASE,
    )
    return int(match.group(1)) if match else None


def visual_image_is_table_only(image: dict[str, Any] | None) -> bool:
    """Return whether an extracted visual is a pure table screenshot."""
    if not image:
        return False
    kind = str(image.get("kind") or "").lower()
    source_kinds = {
        str(kind_value).lower()
        for kind_value in image.get("source_kinds") or []
        if str(kind_value).strip()
    }
    return kind == "table" and "image" not in source_kinds


def visual_image_is_previewable(image: dict[str, Any] | None) -> bool:
    """Return whether a visual should be shown through inline image preview."""
    return bool(image) and not visual_image_is_table_only(image)


def markdown_for_display(markdown: str, images: list[dict[str, Any]] | None = None) -> str:
    """Return full Markdown with image payloads replaced by text placeholders."""
    image_by_ref = image_source_lookup(images)
    next_image_index = 1

    def placeholder_for_visual(label: str = "", image: dict[str, Any] | None = None) -> str:
        nonlocal next_image_index
        if visual_kind_from_label(label) == "table" or visual_image_is_table_only(image):
            return ""

        image_label = str(image.get("label") or "") if image else ""
        image_index = visual_label_image_index(label) or visual_label_image_index(image_label)
        if image_index is None:
            image_index = next_image_index
            next_image_index += 1
        else:
            next_image_index = max(next_image_index, image_index + 1)
        return f"**此处图片{image_index}已省略**"

    safe_markdown = markdown or ""
    safe_markdown = re.sub(
        r"!\[(?P<alt>[^\]]*)\]\(data:image[^)]*\)",
        lambda match: placeholder_for_visual(match.group("alt")),
        safe_markdown,
        flags=re.IGNORECASE,
    )

    def replace_data_uri_link(match: re.Match[str]) -> str:
        return placeholder_for_visual(match.group("label"))

    safe_markdown = re.sub(
        r"\[(?P<label>[^\]]*)\]\(data:image[^)]*\)",
        replace_data_uri_link,
        safe_markdown,
        flags=re.IGNORECASE,
    )
    safe_markdown = re.sub(
        r"\[(?P<label>[^\]]*(?:此处|图|表|公式|image|figure|table|equation|formula)[^\]]*)\]\(\[\[PM_(?:DOC_)?PROTECTED_\d+\]\]\)",
        replace_data_uri_link,
        safe_markdown,
        flags=re.IGNORECASE,
    )

    def replace_markdown_image(match: re.Match[str]) -> str:
        raw_target = match.group("target").strip().strip("\"'")
        normalized = normalize_markdown_image_ref(raw_target)
        image = image_by_ref.get(normalized) or image_by_ref.get(Path(normalized).name)
        return placeholder_for_visual(match.group("alt"), image)

    safe_markdown = re.sub(
        r"!\[(?P<alt>[^\]]*)\]\((?P<target>[^)]+)\)",
        replace_markdown_image,
        safe_markdown,
    )

    def replace_html_image(match: re.Match[str]) -> str:
        tag = match.group(0)
        src_match = re.search(r"\bsrc\s*=\s*[\"']?([^\"'>\s]+)", tag, flags=re.IGNORECASE)
        image = None
        if src_match:
            normalized = normalize_markdown_image_ref(src_match.group(1))
            image = image_by_ref.get(normalized) or image_by_ref.get(Path(normalized).name)
        return placeholder_for_visual("", image)

    safe_markdown = re.sub(
        r"<img\b[^>]*>",
        replace_html_image,
        safe_markdown,
        flags=re.IGNORECASE,
    )
    return safe_markdown.strip()


IMAGE_PLACEHOLDER_PATTERN = re.compile(r"\*\*此处图片(?P<index>\d+)已省略\*\*")


def image_preview_state_key(key_prefix: str) -> str:
    """Return the session key used for inline image preview visibility."""
    return f"{key_prefix}_loaded_image_previews"


def image_by_display_index(images: list[dict[str, Any]]) -> dict[int, dict[str, Any]]:
    """Map visible Markdown image placeholder indexes to extracted images."""
    indexed: dict[int, dict[str, Any]] = {}
    fallback_sequence = 0
    for image in images or []:
        if not visual_image_is_previewable(image):
            continue
        label = str(image.get("label") or "")
        image_index = visual_label_image_index(label)
        if image_index is not None:
            indexed.setdefault(image_index, image)
            fallback_sequence = max(fallback_sequence, image_index)
            continue

        fallback_sequence += 1
        indexed.setdefault(fallback_sequence, image)
    return indexed


def markdown_image_placeholder_occurrences(markdown_text: str) -> list[dict[str, int]]:
    """Return image placeholder occurrences with display indexes."""
    occurrences: list[dict[str, int]] = []
    for occurrence, match in enumerate(IMAGE_PLACEHOLDER_PATTERN.finditer(markdown_text or ""), start=1):
        occurrences.append({"occurrence": occurrence, "index": int(match.group("index"))})
    return occurrences


def render_inline_image_preview_controls(
    markdown_text: str,
    images: list[dict[str, Any]],
    key_prefix: str,
) -> None:
    """Render one-click controls for Markdown inline image previews."""
    image_lookup = image_by_display_index(images)
    occurrences = [
        item
        for item in markdown_image_placeholder_occurrences(markdown_text)
        if item["index"] in image_lookup
    ]
    if not occurrences or not images:
        return
    occurrence_ids = [item["occurrence"] for item in occurrences]

    state_key = image_preview_state_key(key_prefix)
    loaded = st.session_state.get(state_key)
    if not isinstance(loaded, set):
        loaded = set(loaded or [])
    st.session_state[state_key] = loaded

    loaded_count = len([occurrence for occurrence in occurrence_ids if occurrence in loaded])
    control_cols = st.columns([0.62, 0.19, 0.19], gap="small")
    with control_cols[0]:
        st.caption(f"图片预览：已加载 {loaded_count}/{len(occurrence_ids)} 张")
    with control_cols[1]:
        if st.button(
            "加载全部图片预览",
            use_container_width=True,
            disabled=loaded_count == len(occurrence_ids),
            key=f"{key_prefix}_load_all_image_previews",
        ):
            st.session_state[state_key] = set(occurrence_ids)
            st.rerun()
    with control_cols[2]:
        if st.button(
            "收起全部预览",
            use_container_width=True,
            disabled=loaded_count == 0,
            key=f"{key_prefix}_hide_all_image_previews",
        ):
            st.session_state[state_key] = set()
            st.rerun()


def render_markdown_with_image_previews(
    markdown_text: str,
    images: list[dict[str, Any]] | None,
    key_prefix: str,
) -> None:
    """Render Markdown and add per-placeholder image preview buttons."""
    markdown_text = markdown_text or ""
    image_list = images or []
    if not markdown_text.strip():
        st.markdown("暂无 Markdown 内容", unsafe_allow_html=True)
        return

    render_inline_image_preview_controls(markdown_text, image_list, key_prefix)
    image_lookup = image_by_display_index(image_list)
    state_key = image_preview_state_key(key_prefix)
    loaded = st.session_state.get(state_key)
    if not isinstance(loaded, set):
        loaded = set(loaded or [])
        st.session_state[state_key] = loaded

    position = 0
    placeholder_count = 0
    for match in IMAGE_PLACEHOLDER_PATTERN.finditer(markdown_text):
        segment = markdown_text[position:match.start()]
        if segment.strip():
            st.markdown(segment, unsafe_allow_html=True)

        image_index = int(match.group("index"))
        placeholder_count += 1
        image = image_lookup.get(image_index)
        if image is None:
            st.markdown(match.group(0), unsafe_allow_html=True)
            position = match.end()
            continue

        button_label = "隐藏预览" if placeholder_count in loaded else "加载图片预览"
        notice_cols = st.columns([0.74, 0.26], gap="small")
        with notice_cols[0]:
            st.markdown(f"**此处图片{image_index}已省略**")
        with notice_cols[1]:
            clicked = st.button(
                button_label,
                use_container_width=True,
                disabled=image is None,
                key=f"{key_prefix}_image_preview_{image_index}_{placeholder_count}",
            )
        if clicked:
            updated = set(loaded)
            if placeholder_count in updated:
                updated.remove(placeholder_count)
            else:
                updated.add(placeholder_count)
            st.session_state[state_key] = updated
            st.rerun()

        if placeholder_count in loaded and image:
            image_path = Path(image.get("path", ""))
            if image_path.exists() and image_path.is_file():
                caption = str(image.get("label") or f"图片{image_index}")
                st.image(str(image_path), caption=caption, use_container_width=True)
            else:
                st.warning(f"图片文件不存在：{image_path}")

        position = match.end()

    tail = markdown_text[position:]
    if tail.strip():
        st.markdown(tail, unsafe_allow_html=True)


def split_paper_header(markdown: str) -> tuple[dict[str, str], str]:
    """Extract likely title and author lines from the beginning of Markdown."""
    lines = (markdown or "").splitlines()
    title_index: int | None = None
    title = ""

    for index, line in enumerate(lines[:40]):
        candidate = clean_header_line(line)
        if not candidate or is_front_matter_noise(candidate):
            continue
        title_index = index
        title = candidate
        break

    if title_index is None:
        return {}, markdown.strip()

    author_lines: list[str] = []
    body_start = title_index + 1
    for index in range(title_index + 1, min(len(lines), title_index + 18)):
        raw_line = lines[index]
        if re.match(r"^\s{0,3}#{1,6}\s+", raw_line):
            body_start = index
            break
        candidate = clean_header_line(raw_line)
        if not candidate:
            if author_lines:
                body_start = index + 1
            continue
        if is_front_matter_noise(candidate):
            continue
        if is_body_start_heading(candidate):
            body_start = index
            break
        if len(author_lines) >= 6:
            body_start = index
            break
        author_lines.append(candidate)
        body_start = index + 1

    body_markdown = "\n".join(lines[body_start:]).strip()
    return {"title": title, "authors": "；".join(author_lines)}, body_markdown


def clean_header_line(line: str) -> str:
    """Normalize one potential title or author line."""
    cleaned = re.sub(r"^\s{0,3}#{1,6}\s*", "", line.strip()).strip()
    cleaned = re.sub(r"\s+", " ", cleaned)
    return cleaned.strip(" -–—")


def is_front_matter_noise(text: str) -> bool:
    """Return whether a leading Markdown line is not paper title/author text."""
    lowered = text.lower()
    plain_text = text.strip("*_ ")
    return (
        lowered.startswith("<!--")
        or lowered.startswith("![")
        or lowered.startswith("[此处")
        or plain_text.startswith("此处图片")
        or lowered in {"paper", "title"}
    )


def is_body_start_heading(text: str) -> bool:
    """Return whether a line likely begins the paper body after title/authors."""
    normalized = text.strip().lower().rstrip(":：")
    if normalized.startswith(("abstract:", "abstract：", "摘要:", "摘要：", "提要:", "提要：")):
        return True
    return normalized in {
        "abstract",
        "摘要",
        "提要",
        "keywords",
        "key words",
        "关键词",
        "introduction",
        "1 introduction",
        "i introduction",
    }


def prepare_bilingual_reader_markdown(
    markdown: str,
    images: list[dict[str, str]] | None = None,
) -> tuple[dict[str, str], str]:
    """Return display-safe Markdown with title/author lines removed for bilingual alignment."""
    safe_markdown = markdown_for_display(markdown, images or [])
    paper_header, body_markdown = split_paper_header(safe_markdown)
    return paper_header, body_markdown or safe_markdown


def split_bilingual_image_notices(markdown: str) -> tuple[list[str], str]:
    """Extract image placeholder lines so they are not rendered as bilingual pairs."""
    notices: list[str] = []
    body_lines: list[str] = []
    for line in (markdown or "").splitlines():
        notice = canonical_image_notice(line)
        if notice:
            notices.append(notice)
        else:
            body_lines.append(line)
    return notices, "\n".join(body_lines).strip()


def canonical_image_notice(line: str) -> str:
    """Return a normalized image placeholder label for one Markdown line."""
    stripped = (line or "").strip()
    if not stripped:
        return ""

    link_match = re.fullmatch(r"!?\[(?P<label>[^\]]+)\]\((?P<target>[^)]*)\)\s*", stripped)
    label = link_match.group("label").strip() if link_match else stripped
    cleaned = re.sub(r"^[*_`]+|[*_`]+$", "", label).strip()
    cleaned = re.sub(r"\s+", " ", cleaned)
    lowered = cleaned.lower()
    if "此处" not in cleaned and not lowered.startswith(("image", "figure", "fig.")):
        return ""

    image_match = re.search(r"(?:图|图片|image|figure|fig\.?)\s*(\d+)", cleaned, flags=re.IGNORECASE)
    if image_match:
        return f"此处图片{image_match.group(1)}已省略"
    if cleaned.startswith("此处图片") or cleaned.startswith("此处含有图"):
        return cleaned
    return ""


def merge_image_notices(*notice_groups: list[str]) -> list[str]:
    """Merge image notices while preserving first-seen order."""
    merged: list[str] = []
    seen: set[str] = set()
    for notices in notice_groups:
        for notice in notices:
            key = notice.strip()
            if not key or key in seen:
                continue
            seen.add(key)
            merged.append(key)
    return merged


def bilingual_image_notice_html(notice: str) -> str:
    """Render a single non-bilingual image placeholder notice."""
    return (
        '<section class="pm-bilingual-image-notice" data-block-type="image-notice">'
        f"{html.escape(notice)}"
        "</section>"
    )


def build_images_zip(images: list[dict[str, str]]) -> tuple[bytes, int]:
    """Create an in-memory zip containing extracted image files."""
    buffer = io.BytesIO()
    written_names: set[str] = set()
    written_count = 0

    with zipfile.ZipFile(buffer, mode="w", compression=zipfile.ZIP_DEFLATED) as archive:
        for index, image in enumerate(images, start=1):
            image_path = Path(image.get("path", ""))
            if not image_path.exists() or not image_path.is_file():
                continue

            archive_name = unique_zip_name(
                image.get("file_name") or image_path.name or f"image_{index}{image_path.suffix}",
                written_names,
            )
            archive.write(image_path, arcname=archive_name)
            written_count += 1

    return buffer.getvalue(), written_count


def unique_zip_name(file_name: str, used_names: set[str]) -> str:
    """Return a unique zip member name."""
    clean_name = re.sub(r"[^A-Za-z0-9._\-\u4e00-\u9fff]+", "_", file_name).strip("._")
    if not clean_name:
        clean_name = f"image_{len(used_names) + 1}.png"

    stem = Path(clean_name).stem
    suffix = Path(clean_name).suffix or ".png"
    candidate = f"{stem}{suffix}"
    counter = 2
    while candidate.lower() in used_names:
        candidate = f"{stem}_{counter}{suffix}"
        counter += 1
    used_names.add(candidate.lower())
    return candidate


def chunk_anchor_id(chunk_id: Any) -> str:
    """Build a browser-safe anchor id for one chunk."""
    raw_chunk_id = str(chunk_id or "").strip()
    safe_id = re.sub(r"[^A-Za-z0-9_-]+", "-", raw_chunk_id).strip("-")
    if not safe_id:
        safe_id = hashlib.sha1(raw_chunk_id.encode("utf-8")).hexdigest()[:12]
    return f"pm-source-{safe_id}"


def chunk_anchor_html(chunk_id: Any, fallback: bool = False) -> str:
    """Return an HTML anchor marker for chunk-level source navigation."""
    anchor_id = html.escape(chunk_anchor_id(chunk_id), quote=True)
    fallback_attr = ' data-anchor-fallback="true"' if fallback else ""
    return f'<span id="{anchor_id}" class="pm-source-anchor"{fallback_attr}></span>'


def request_source_jump(chunk_id: Any) -> None:
    """Request original-reader mode and scroll to the chunk anchor on rerun."""
    clean_chunk_id = str(chunk_id or "").strip()
    if not clean_chunk_id:
        return
    st.session_state["pm_pending_source_anchor"] = chunk_anchor_id(clean_chunk_id)
    st.rerun()


def source_jump_button_key(prefix: str, chunk_id: Any) -> str:
    """Build a stable Streamlit key for one source-jump button."""
    digest = hashlib.sha1(str(chunk_id or "").encode("utf-8")).hexdigest()[:12]
    safe_prefix = re.sub(r"[^A-Za-z0-9_-]+", "_", str(prefix or "source")).strip("_")
    return f"source_jump_{safe_prefix}_{digest}"


def render_source_jump_button(chunk_id: Any, key_prefix: str, label: str = SOURCE_JUMP_LABEL) -> None:
    """Render a reliable button for jumping from QA evidence to source text."""
    clean_chunk_id = str(chunk_id or "").strip()
    if st.button(
        label,
        key=source_jump_button_key(key_prefix, clean_chunk_id),
        disabled=not clean_chunk_id,
        use_container_width=False,
    ):
        request_source_jump(clean_chunk_id)


def add_chunk_anchors_to_markdown(
    markdown: str,
    chunks: list[dict[str, Any]],
) -> tuple[str, list[dict[str, Any]]]:
    """Insert chunk anchors into Markdown, with page/order fallbacks for misses."""
    if not markdown or not chunks:
        return markdown, chunks

    normalized_markdown, offset_map = normalize_with_offsets(markdown)
    inserts_by_position: dict[int, list[str]] = {}
    missing_chunks: list[dict[str, Any]] = []
    matched_chunks: list[dict[str, Any]] = []
    next_search_start = 0

    for ordinal, chunk in enumerate(chunks):
        chunk_id = str(chunk.get("chunk_id") or "").strip()
        if not chunk_id:
            missing_chunks.append(chunk)
            continue

        match = find_chunk_anchor_match(normalized_markdown, offset_map, chunk, next_search_start)
        if match is None:
            missing_chunks.append(chunk)
            continue

        original_position, normalized_position = match
        inserts_by_position.setdefault(original_position, []).append(chunk_anchor_html(chunk_id))
        matched_chunks.append(
            {
                "chunk": chunk,
                "position": original_position,
                "chunk_index": chunk_sort_index(chunk, ordinal),
                "page_num": chunk_page_num(chunk),
            }
        )
        next_search_start = min(len(normalized_markdown), max(next_search_start, normalized_position + 1))

    for ordinal, chunk in enumerate(missing_chunks):
        chunk_id = str(chunk.get("chunk_id") or "").strip()
        if not chunk_id:
            continue
        position = fallback_chunk_anchor_position(chunk, matched_chunks, ordinal)
        inserts_by_position.setdefault(position, []).append(chunk_anchor_html(chunk_id, fallback=True))

    anchored_markdown = markdown
    for position, anchors in sorted(inserts_by_position.items(), key=lambda item: item[0], reverse=True):
        anchor_html = "\n".join(anchors)
        anchored_markdown = f"{anchored_markdown[:position]}{anchor_html}\n{anchored_markdown[position:]}"

    return anchored_markdown, missing_chunks


def find_chunk_anchor_match(
    normalized_markdown: str,
    offset_map: list[int],
    chunk: dict[str, Any],
    search_start: int = 0,
) -> tuple[int, int] | None:
    """Return original and normalized positions for a chunk text match."""
    if not normalized_markdown or not offset_map:
        return None

    for candidate in chunk_search_candidates(str(chunk.get("text") or "")):
        normalized_candidate, _ = normalize_with_offsets(candidate)
        if len(normalized_candidate) < 24:
            continue

        match_position = normalized_markdown.find(normalized_candidate, max(0, search_start))
        if match_position < 0 and search_start > 0:
            match_position = normalized_markdown.find(normalized_candidate)
        if match_position >= 0:
            return offset_map[match_position], match_position
    return None


def fallback_chunk_anchor_position(
    chunk: dict[str, Any],
    matched_chunks: list[dict[str, Any]],
    ordinal: int,
) -> int:
    """Pick the nearest already-located source position for an unmatched chunk."""
    if not matched_chunks:
        return 0

    chunk_index = chunk_sort_index(chunk, ordinal)
    page_num = chunk_page_num(chunk)
    candidates = [
        matched
        for matched in matched_chunks
        if page_num is not None and matched.get("page_num") == page_num
    ]
    if not candidates:
        candidates = matched_chunks

    nearest = min(
        candidates,
        key=lambda matched: (
            abs(int(matched.get("chunk_index", 0)) - chunk_index),
            int(matched.get("chunk_index", 0)),
        ),
    )
    return max(0, int(nearest.get("position") or 0))


def chunk_sort_index(chunk: dict[str, Any], fallback: int = 0) -> int:
    """Return a numeric chunk order for fallback anchor placement."""
    value = chunk.get("chunk_index")
    try:
        return int(value)
    except (TypeError, ValueError):
        pass

    match = re.search(r"(?:chunk[_-])?(\d+)\s*$", str(chunk.get("chunk_id") or ""))
    if match:
        return int(match.group(1))
    return int(fallback)


def chunk_page_num(chunk: dict[str, Any]) -> int | None:
    """Return a numeric page value when the chunk has one."""
    try:
        page_num = int(chunk.get("page_num") or 0)
    except (TypeError, ValueError):
        return None
    return page_num if page_num > 0 else None


def normalize_with_offsets(text: str) -> tuple[str, list[int]]:
    """Normalize text for fuzzy position search while keeping original offsets."""
    normalized_chars: list[str] = []
    offset_map: list[int] = []
    previous_space = True

    for index, char in enumerate(text):
        if char.isspace():
            if not previous_space and normalized_chars:
                normalized_chars.append(" ")
                offset_map.append(index)
            previous_space = True
            continue

        normalized_chars.append(char.lower())
        offset_map.append(index)
        previous_space = False

    if normalized_chars and normalized_chars[-1] == " ":
        normalized_chars.pop()
        offset_map.pop()

    return "".join(normalized_chars), offset_map


def chunk_search_candidate(text: str) -> str:
    """Pick a stable searchable prefix from a chunk."""
    candidates = chunk_search_candidates(text)
    return candidates[0] if candidates else ""


def chunk_search_candidates(text: str) -> list[str]:
    """Build searchable snippets from several positions inside a chunk."""
    cleaned = clean_chunk_search_text(text)
    if not cleaned:
        return []

    candidates: list[str] = []
    seen: set[str] = set()

    def add_candidate(candidate: str) -> None:
        normalized_candidate, _ = normalize_with_offsets(candidate)
        if len(normalized_candidate) < 24:
            return
        if normalized_candidate in seen:
            return
        seen.add(normalized_candidate)
        candidates.append(candidate.strip())

    for length in (220, 160, 120, 80, 48):
        add_candidate(cleaned[: min(length, len(cleaned))])

    words = cleaned.split()
    if len(words) >= 8:
        for window_size in (36, 28, 20, 14, 10):
            if len(words) < window_size:
                continue
            starts = {
                0,
                max(0, len(words) // 4),
                max(0, len(words) // 2),
                max(0, len(words) - window_size),
            }
            for start in sorted(starts):
                add_candidate(" ".join(words[start : start + window_size]))
    else:
        for length in (160, 120, 80, 48):
            if len(cleaned) <= length:
                continue
            starts = {
                0,
                max(0, len(cleaned) // 4),
                max(0, len(cleaned) // 2),
                max(0, len(cleaned) - length),
            }
            for start in sorted(starts):
                add_candidate(cleaned[start : start + length])

    return candidates


def clean_chunk_search_text(text: str) -> str:
    """Remove generated display notices before source-position matching."""
    cleaned = str(text or "")
    cleaned = re.sub(r"\*\*(?:此处图片|姝ゅ鍥剧墖)\d+[^*]*\*\*", " ", cleaned)
    cleaned = re.sub(r"\[(?:此处含有图|姝ゅ鍚湁鍥?)[^\]]*\]\([^)]*\)", " ", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


def get_uploaded_file_signature(uploaded_file: UploadedFile) -> str:
    """Return a stable signature for the uploaded file within a Streamlit session."""
    file_bytes = uploaded_file.getvalue()
    digest = hashlib.sha256(file_bytes).hexdigest()
    parse_settings = ":".join(
        [
            settings.pdf_parse_provider,
            settings.mineru_model_version,
            str(settings.mineru_is_ocr),
            settings.mineru_language,
            CHUNKER_VERSION,
            settings.rag_chunk_strategy,
            str(settings.rag_chunk_size),
            str(settings.rag_chunk_overlap),
            settings.vlm_base_url,
            settings.vlm_model,
        ]
    )
    return f"{uploaded_file.name}:{len(file_bytes)}:{digest}:{parse_settings}"


def process_uploaded_pdf(
    uploaded_file: UploadedFile,
    user_id: int | None = None,
    team_id: int | None = None,
    project_id: int | None = None,
) -> dict[str, Any]:
    """Save, parse, chunk, and persist an uploaded PDF once per session file."""
    signature = get_uploaded_file_signature(uploaded_file)
    cached_result = st.session_state.get("processed_pdf")
    # MinerU parsing can be slow and billable; include parser settings in the
    # signature so a changed configuration forces a fresh parse.
    if cached_result and cached_result.get("signature") == signature:
        return cached_result

    saved_file = save_uploaded_pdf(uploaded_file)
    parsed_pdf = parse_pdf(saved_file["save_path"], saved_file["paper_id"])
    chunks = chunk_pages(
        saved_file["paper_id"],
        parsed_pdf["pages"],
        chunk_size=settings.rag_chunk_size,
        overlap=settings.rag_chunk_overlap,
        elements=parsed_pdf.get("elements"),
    )
    total_chars, preview = build_text_preview(parsed_pdf)

    db_save_failed = False
    try:
        # The UI can still show the parsed Markdown if SQLite fails, but RAG and
        # literature cards need these persisted chunks for later page actions.
        save_paper_and_chunks(
            {
                "paper_id": saved_file["paper_id"],
                "file_name": saved_file["file_name"],
                "file_size_bytes": saved_file["file_size_bytes"],
                "save_path": saved_file["save_path"],
                "owner_user_id": user_id,
                "team_id": team_id,
                "project_id": project_id,
                "file_sha256": saved_file.get("file_sha256", ""),
                "parse_status": "succeeded",
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
    except (OSError, sqlite3.Error):
        db_save_failed = True

    result = {
        "signature": signature,
        "saved_file": saved_file,
        "parsed_pdf": parsed_pdf,
        "chunks": chunks,
        "total_chars": total_chars,
        "preview": preview,
        "db_save_failed": db_save_failed,
    }
    st.session_state["processed_pdf"] = result
    return result


def index_state_from_paper_status(
    index_status: str,
    vector_status: str = "未知",
    bm25_status: str = "未知",
    overwrite: bool = False,
) -> tuple[str, str]:
    """Map persisted paper index status to the two UI badges."""
    clean_status = str(index_status or "").strip().lower()
    if clean_status == "succeeded":
        return "已构建", "已构建"
    status_label = {
        "queued": "排队中",
        "running": "构建中",
        "failed": "失败",
    }.get(clean_status)
    if not status_label:
        return vector_status, bm25_status
    if overwrite:
        return status_label, status_label
    return (
        vector_status if vector_status != "未知" else status_label,
        bm25_status if bm25_status != "未知" else status_label,
    )


def refresh_index_status_for_paper(paper_id: str, rerun: bool = True) -> None:
    """Clear cached index UI state, reload paper metadata, and rerun the page."""
    clean_paper_id = str(paper_id or "").strip()
    if clean_paper_id:
        st.session_state.pop(f"index_state_{clean_paper_id}", None)
    processed_pdf = st.session_state.get("processed_pdf")
    current_paper_id = str((processed_pdf or {}).get("saved_file", {}).get("paper_id") or "")
    if processed_pdf and clean_paper_id and current_paper_id == clean_paper_id:
        try:
            refresh_workspace_paper(processed_pdf)
        except Exception:
            logger.exception("Index status refresh failed. paper_id=%s", clean_paper_id)
    st.session_state["pm_workspace_notice"] = "已刷新索引状态。"
    if rerun:
        st.rerun()

def local_index_state(paper_id: str, allow_db: bool = True) -> dict[str, str]:
    """Return UI-only index status without triggering remote embedding calls."""
    state_key = f"index_state_{paper_id}"
    state = st.session_state.get(state_key, {})
    vector_status = str(state.get("vector") or "未知")
    bm25_status = str(state.get("bm25") or "未知")

    if allow_db:
        try:
            paper = get_accessible_paper(paper_id, current_user_id())
        except Exception:
            paper = None
        if paper:
            vector_status, bm25_status = index_state_from_paper_status(
                str(paper.get("index_status") or ""),
                vector_status,
                bm25_status,
                overwrite=True,
            )

    if vector_status == "未知" or bm25_status == "未知":
        session_paper = st.session_state.get("processed_pdf") or {}
        session_paper_id = str((session_paper.get("saved_file") or {}).get("paper_id") or "")
        index_status = str(session_paper.get("index_status") or "")
        if paper_id and session_paper_id == str(paper_id) and index_status:
            vector_status, bm25_status = index_state_from_paper_status(index_status, vector_status, bm25_status)

    if allow_db and (vector_status == "未知" or bm25_status == "未知"):
        try:
            paper = get_accessible_paper(paper_id, current_user_id())
        except Exception:
            paper = None
        if paper:
            vector_status, bm25_status = index_state_from_paper_status(
                str(paper.get("index_status") or ""),
                vector_status,
                bm25_status,
            )

    bm25_payload = settings.bm25_dir / f"{paper_id}_payloads.json"
    bm25_pickle = settings.bm25_dir / f"{paper_id}_bm25.pkl"
    fallback_payload = Path("data/bm25") / f"{paper_id}_payloads.json"
    fallback_pickle = Path("data/bm25") / f"{paper_id}_bm25.pkl"
    if bm25_status == "未知" and (
        (bm25_payload.exists() and bm25_pickle.exists())
        or (fallback_payload.exists() and fallback_pickle.exists())
    ):
        bm25_status = "已构建"

    return {"vector": vector_status, "bm25": bm25_status}


def index_status_type(status: str) -> str:
    """Map index status text to badge/card status."""
    if status == "已构建":
        return "success"
    if status == "失败":
        return "error"
    return "warning"


def render_header() -> None:
    """Render the shared page header."""
    render_app_header(
        "PaperMate",
        "论文阅读 RAG 助手",
        [render_status_badge("可信引用", "info")],
    )





def render_extracted_images(images: list[dict[str, str]]) -> None:
    """Render extracted paper image downloads and optional inspection."""
    if not images:
        return

    with st.expander(f"论文图片（{len(images)} 张）", expanded=False):
        zip_bytes, image_count = build_images_zip(images)
        if image_count:
            st.download_button(
                "下载全部图片 ZIP",
                data=zip_bytes,
                file_name="paper_images.zip",
                mime="application/zip",
                use_container_width=True,
            )
        else:
            st.warning("没有找到可打包的图片文件。")

        st.caption("正文中图片已用“此处图片 N 已省略”表示，原图可通过 ZIP 下载。")
        load_previews = st.checkbox("加载图片预览（最多 10 张）", value=False)
        visible_images = images[:10] if load_previews else []

        for image in images[:20]:
            st.caption(f"{image.get('label', '图片')}：{image.get('path', '')}")
        if len(images) > 20:
            st.caption(f"其余 {len(images) - 20} 张图片请在 MinerU 输出目录查看。")

        for image in visible_images:
            image_path = Path(image["path"])
            if not image_path.exists():
                continue
            st.markdown(f"**{image['label']}**")
            st.image(str(image_path), use_container_width=True)




def render_chunk_preview(chunks: list[dict[str, Any]]) -> None:
    """Render chunk count and the first few chunk previews."""
    st.markdown("#### 切分结果")
    if not chunks:
        st.warning("论文正文为空，无法构建索引")
        return

    st.write("chunk 数量：", len(chunks))
    for chunk in chunks[:3]:
        section = chunk["section_title"] or "未识别章节"
        chunk_type = chunk.get("chunk_type", "text")
        title = f"Chunk {chunk['chunk_index']} | {chunk_type} | 第 {chunk['page_num']} 页 | {section}"
        with st.expander(title, expanded=chunk["chunk_index"] == 0):
            st.caption(f"chunk_id：{chunk['chunk_id']}")
            image_count = chunk_metadata_count(chunk, "images")
            table_count = chunk_metadata_count(chunk, "tables")
            if image_count or table_count:
                st.caption(f"metadata：{image_count} images / {table_count} tables")
            st.text_area(
                "chunk 预览",
                chunk["text"][:1000],
                height=180,
                key=f"chunk_preview_{chunk['chunk_id']}",
            )








def render_source_chunks(source_chunks: list[dict[str, Any]]) -> None:
    """Render retrieved source snippets."""
    st.markdown("#### 原文片段")
    if not source_chunks:
        st.write("无原文片段。")
        return

    for chunk in source_chunks:
        source_id = chunk.get("source_id") or f"片段{chunk.get('citation_id', '')}"
        title = (
            f"[{source_id}] {format_page_label(chunk.get('page_num'))} | "
            f"{chunk.get('section_title', '未知章节')} | "
            f"{chunk.get('chunk_type', 'text')} | {chunk.get('chunk_id', '')}"
        )
        with st.expander(title):
            render_source_jump_button(
                chunk.get("chunk_id"),
                f"source_chunk_{source_id}_{chunk.get('chunk_id', '')}",
            )
            image_count = chunk_metadata_count(chunk, "images")
            table_count = chunk_metadata_count(chunk, "tables")
            if image_count or table_count:
                st.caption(f"metadata：{image_count} images / {table_count} tables")
            st.write(chunk.get("text", ""))


def chunk_metadata_count(chunk: dict[str, Any], key: str) -> int:
    """Return number of image/table metadata records on a chunk."""
    value = chunk.get(key)
    if isinstance(value, list):
        return len(value)
    raw_json = chunk.get(f"{key}_json")
    if not raw_json:
        return 0
    try:
        parsed = json.loads(str(raw_json))
    except (TypeError, ValueError, json.JSONDecodeError):
        return 0
    return len(parsed) if isinstance(parsed, list) else 0


def format_page_label(page_num: Any) -> str:
    """Format both legacy numeric pages and new citation page labels."""
    if page_num is None:
        return "未知页"
    text = str(page_num).strip()
    if not text:
        return "未知页"
    if "页" in text:
        return text
    return f"第 {text} 页"


def render_retrieval_details(details: dict[str, Any]) -> None:
    """Render hybrid retrieval diagnostics for the latest answer."""
    st.markdown("#### 检索细节")
    if not details:
        st.write("暂无检索细节。")
        return

    with st.expander("查看 Hybrid 检索细节", expanded=False):
        strategy = details.get("strategy", "")
        st.info(strategy_message(strategy))
        st.write("strategy：", strategy or "未知")
        st.write("query_type：", details.get("query_type") or details.get("question_type") or "default")
        st.write("expanded_query：", details.get("expanded_query") or "无")
        st.write("vector_top_k：", details.get("vector_top_k", ""))
        st.write("bm25_top_k：", details.get("bm25_top_k", ""))
        st.write("final_top_k：", details.get("final_top_k", ""))
        st.write("rrf_k：", details.get("rrf_k", ""))
        st.write("latency_ms：", details.get("latency_ms", ""))

        expanded_terms = [str(term) for term in details.get("expanded_terms", []) if term]
        if expanded_terms:
            st.write("扩展关键词：", "、".join(expanded_terms[:30]))

        rows = details.get("retrieved_chunks") or []
        if not rows:
            st.write("没有可展示的融合排序结果。")
            return

        display_rows: list[dict[str, Any]] = []
        for row in rows:
            sources = row.get("retrieval_sources") or []
            source_text = " + ".join("向量" if source == "vector" else "BM25" for source in sources)
            display_rows.append(
                {
                    "RRF排名": row.get("rank"),
                    "chunk_id": row.get("chunk_id"),
                    "页码": row.get("page_num"),
                    "章节": row.get("section_title") or "未识别章节",
                    "类型": row.get("chunk_type", "text"),
                    "来源": source_text,
                    "RRF分数": format_optional_float(row.get("rrf_score")),
                    "向量排名": row.get("vector_rank") or "",
                    "BM25排名": row.get("bm25_rank") or "",
                    "BM25分数": format_optional_float(row.get("bm25_score")),
                    "向量距离": format_optional_float(row.get("vector_distance")),
                }
            )

        st.dataframe(display_rows, use_container_width=True, hide_index=True)


def format_optional_float(value: Any) -> str:
    """Format optional numeric values for retrieval details."""
    if value is None or value == "":
        return ""
    try:
        return f"{float(value):.4f}"
    except (TypeError, ValueError):
        return str(value)


def format_retrieval_sources(sources: Any) -> str:
    """Format retrieval source labels for citations and debug UI."""
    if not sources:
        return "未知"
    if isinstance(sources, str):
        sources = [sources]
    labels = []
    for source in sources:
        if source == "vector":
            labels.append("vector")
        elif source == "bm25":
            labels.append("bm25")
        else:
            labels.append(str(source))
    return "+".join(labels)


def strategy_message(strategy: str) -> str:
    """Return a short Chinese explanation for the retrieval strategy."""
    if strategy == "hybrid_rrf":
        return "当前使用：向量检索 + 关键词检索 + RRF 融合"
    if strategy == "vector_fallback":
        return "当前使用：仅向量检索。关键词索引可能尚未构建。"
    if strategy == "bm25_fallback":
        return "当前使用：仅关键词检索。向量索引可能尚未构建。"
    if strategy in {"empty", "hybrid_empty"}:
        return "未检索到足够依据。"
    return "当前检索策略未知，请查看日志或检查索引配置。"




def needs_index_warning(details: dict[str, Any]) -> bool:
    """Infer an index-missing state from hybrid retrieval debug info."""
    if not details:
        return False
    strategy = details.get("strategy")
    if strategy not in {"empty", "hybrid_empty"}:
        return False
    return bool(details.get("vector_error") and details.get("bm25_error"))


def render_feedback_form(qa_record: dict[str, Any]) -> None:
    """Render feedback controls under an answer."""
    team_context = current_team_context()
    if not can_write(team_context["role"]):
        st.caption("当前团队角色为只读，不能提交新的反馈。")
        return
    paper_id = qa_record["paper_id"]
    qa_log_id = qa_record.get("qa_log_id")
    record_key = qa_log_id or hashlib.sha256(
        f"{paper_id}:{qa_record['question']}:{qa_record['answer']}".encode("utf-8")
    ).hexdigest()[:12]
    feedback_state_key = f"feedback_saved_{record_key}"

    st.markdown("#### 用户反馈")
    with st.form(key=f"feedback_form_{record_key}"):
        feedback_type = st.radio(
            "请选择反馈类型",
            FEEDBACK_OPTIONS,
            horizontal=False,
            key=f"feedback_type_{record_key}",
        )
        comment = st.text_area(
            "补充说明（可选）",
            key=f"feedback_comment_{record_key}",
            height=90,
        )
        submitted = st.form_submit_button("提交回答反馈")

    if submitted:
        try:
            save_feedback(
                paper_id=paper_id,
                question=qa_record["question"],
                answer=qa_record["answer"],
                feedback_type=feedback_type,
                comment=comment.strip(),
                qa_log_id=qa_log_id,
                user_id=current_user_id(),
                team_id=int(team_context["team_id"]),
                project_id=team_context.get("project_id"),
            )
        except (OSError, sqlite3.Error):
            st.error("反馈保存失败，请检查 SQLite 数据库权限。")
            return

        st.session_state[feedback_state_key] = True

    if st.session_state.get(feedback_state_key):
        st.success("反馈已记录，将用于后续优化")




def escaped_text(value: Any) -> str:
    """Escape text for safe HTML rendering."""
    return html.escape(str(value or "原文未明确说明")).replace("\n", "<br>")


def card_palette(card: dict[str, Any]) -> dict[str, str]:
    """Pick a stable visual palette for a literature card."""
    try:
        card_id = int(card.get("card_id") or 0)
    except (TypeError, ValueError):
        card_id = 0
    return CARD_PALETTES[card_id % len(CARD_PALETTES)]






@st.cache_data(show_spinner=False, max_entries=8)
def cached_pdf_bytes(save_path: str, mtime_ns: int, size_bytes: int) -> bytes:
    """Read PDF bytes keyed by path, mtime, and size for explicit previews."""
    del mtime_ns, size_bytes
    return Path(save_path).read_bytes()


def pdf_viewer_state_key(pdf_path: Path, mtime_ns: int, size_bytes: int) -> str:
    """Return a stable session-state key for one PDF preview version."""
    digest = hashlib.sha1(f"{pdf_path.resolve()}|{mtime_ns}|{size_bytes}".encode("utf-8")).hexdigest()[:16]
    return f"pdf_preview_loaded_{digest}"


def render_pdf_viewer(save_path: str | None, eager: bool = False) -> None:
    """Render a PDF viewer for a locally saved paper."""
    if not save_path:
        st.warning("没有找到该论文的 PDF 保存路径。")
        return

    pdf_path = Path(save_path)
    if not pdf_path.exists():
        st.warning(f"PDF 文件不存在：{pdf_path}")
        return

    stat = pdf_path.stat()
    state_key = pdf_viewer_state_key(pdf_path, int(stat.st_mtime_ns), int(stat.st_size))
    if eager:
        st.session_state[state_key] = True

    st.caption(f"PDF：{pdf_path.name} · {format_file_size(int(stat.st_size))}")
    if not st.session_state.get(state_key):
        if st.button("加载 PDF 预览 / 下载", use_container_width=True, key=f"{state_key}_button"):
            st.session_state[state_key] = True
            st.rerun()
        st.info("为保持页面切换流畅，PDF 文件会在点击后再读取和嵌入。")
        return

    pdf_bytes = cached_pdf_bytes(str(pdf_path.resolve()), int(stat.st_mtime_ns), int(stat.st_size))
    st.download_button(
        "下载完整 PDF",
        data=pdf_bytes,
        file_name=pdf_path.name,
        mime="application/pdf",
        use_container_width=True,
    )
    encoded_pdf = base64.b64encode(pdf_bytes).decode("ascii")
    st.markdown(
        f"""
        <iframe
            class="pm-pdf"
            src="data:application/pdf;base64,{encoded_pdf}"
            width="100%"
            height="760"
            type="application/pdf">
        </iframe>
        """,
        unsafe_allow_html=True,
    )


def card_option_label(card: dict[str, Any]) -> str:
    """Build a readable label for card selection."""
    title = str(card.get("title") or "未命名论文").strip()
    year = str(card.get("year") or "年份未知").strip()
    library_name = str(card.get("library_name") or "未分组").strip()
    file_name = str(card.get("file_name") or "PDF 未关联").strip()
    return f"{title} · {year} · {library_name} · {file_name}"


def render_card_edit_form(card: dict[str, Any], user_id: int, team_id: int | None = None) -> None:
    """Render edit form for one literature card."""
    with st.form(key=f"edit_card_{card['card_id']}"):
        values: dict[str, str] = {}
        for key, label in CARD_FIELD_LABELS.items():
            current_value = str(card.get(key) or "")
            if key in {"method_summary", "research_question", "datasets"}:
                values[key] = st.text_area(label, value=current_value, height=110)
            else:
                values[key] = st.text_input(label, value=current_value)

        submitted = st.form_submit_button("保存修改", type="primary")

    if submitted:
        try:
            update_literature_card(int(card["card_id"]), values, user_id=user_id, team_id=team_id)
        except (OSError, sqlite3.Error):
            st.error("文献卡片更新失败，请检查 SQLite 数据库权限。")
            return
        st.success("文献卡片已更新。")
        st.rerun()


def render_card_delete(card_id: int, user_id: int, team_id: int | None = None) -> None:
    """Render delete confirmation controls."""
    confirm = st.checkbox("确认删除这张文献卡片", key=f"confirm_delete_{card_id}")
    if st.button("删除文献卡片", disabled=not confirm, use_container_width=True):
        try:
            delete_literature_card(card_id, user_id=user_id, team_id=team_id)
        except (OSError, sqlite3.Error):
            st.error("文献卡片删除失败，请检查 SQLite 数据库权限。")
            return
        st.success("文献卡片已删除。")
        st.rerun()


def library_option_label(library: dict[str, Any]) -> str:
    """Build a readable label for a card library option."""
    return f"{library['name']}（{int(library.get('card_count') or 0)} 张）"


def render_library_create_form(user_id: int, key_suffix: str = "", team_id: int | None = None) -> None:
    """Render a form that creates a user-owned card library."""
    with st.form(key=f"create_library_form{key_suffix}"):
        new_library_name = st.text_input(
            "新卡片库名称",
            placeholder="例如：综述必读、方法对照、毕业论文核心文献",
            key=f"new_library_name{key_suffix}",
        )
        submitted = st.form_submit_button("创建卡片库", type="primary", use_container_width=True)

    if submitted:
        try:
            create_card_library(user_id, new_library_name, team_id=team_id)
        except sqlite3.IntegrityError:
            st.error("这个卡片库名字已经存在。")
        except (ValueError, OSError, sqlite3.Error) as exc:
            st.error(str(exc) or "卡片库创建失败。")
        else:
            st.success("卡片库已创建。")
            st.rerun()


def render_library_rename_form(user_id: int, library: dict[str, Any], team_id: int | None = None) -> None:
    """Render a form that renames a user-owned card library."""
    with st.form(key=f"rename_library_{library['library_id']}"):
        new_name = st.text_input("新的卡片库名称", value=str(library["name"]))
        submitted = st.form_submit_button("保存名称", type="primary", use_container_width=True)

    if submitted:
        try:
            update_card_library(int(library["library_id"]), user_id, new_name, team_id=team_id)
        except sqlite3.IntegrityError:
            st.error("这个卡片库名字已经存在。")
        except (ValueError, OSError, sqlite3.Error) as exc:
            st.error(str(exc) or "卡片库重命名失败。")
        else:
            st.success("卡片库名称已更新。")
            st.rerun()




def feedback_admin_password() -> str:
    """Return the feedback-page administrator password."""
    return (os.getenv("PAPERMATE_ADMIN_PASSWORD") or settings.app_password or "").strip()






def _pm_text(value: Any, default: str = "未提供") -> str:
    """Return display-safe text for UI cards."""
    text = str(value or "").strip()
    return text or default


def _pm_compact(value: Any, limit: int = 180, default: str = "原文未明确说明") -> str:
    """Compact long card text without changing the stored value."""
    text = re.sub(r"\s+", " ", str(value or "")).strip()
    if not text:
        return default
    if len(text) <= limit:
        return text
    return f"{text[: limit - 1].rstrip()}…"


def _pm_method_chips(card: dict[str, Any], limit: int = 4) -> list[str]:
    """Derive lightweight method chips from structured card fields."""
    source = " ".join(
        str(card.get(key) or "")
        for key in ("research_field", "method_summary", "datasets")
    )
    parts = [
        item.strip(" #，,；;、/|()[]{}")
        for item in re.split(r"[,，;；、/|]\s*|\s{2,}", source)
    ]
    chips: list[str] = []
    for part in parts:
        if not part or len(part) > 26:
            continue
        if part not in chips:
            chips.append(part)
        if len(chips) >= limit:
            break
    return chips or ["方法待补充"]


def _pm_markdown_section(markdown: str, labels: tuple[str, ...], default: str = "原文未明确说明") -> str:
    """Read optional generated-card sections without changing persistence format."""
    for label in labels:
        pattern = rf"^##\s+{re.escape(label)}\s*\n(?P<value>.*?)(?=^##\s+|\Z)"
        match = re.search(pattern, markdown or "", flags=re.MULTILINE | re.DOTALL)
        if match and match.group("value").strip():
            return match.group("value").strip()
    return default


def _pm_recent_week_count(cards: list[dict[str, Any]]) -> int:
    """Count cards updated in the last seven days when timestamps are parseable."""
    from datetime import datetime, timedelta

    threshold = datetime.now() - timedelta(days=7)
    count = 0
    for card in cards:
        raw_time = str(card.get("updated_at") or card.get("created_at") or "").strip()
        if not raw_time:
            continue
        try:
            parsed = datetime.fromisoformat(raw_time.replace("Z", "+00:00").split("+")[0])
        except ValueError:
            continue
        if parsed >= threshold:
            count += 1
    return count


def render_status_badge(text: str, type: str = "default") -> str:
    """Return reusable status-badge HTML."""
    normalized = str(type or "default").strip().lower()
    if normalized == "error":
        normalized = "danger"
    safe_type = html.escape(normalized)
    return f'<span class="pm-badge pm-badge-{safe_type}">{html.escape(str(text))}</span>'


def enqueue_index_build_for_paper(
    paper_id: str,
    *,
    require_parsed: bool = True,
    payload_extra: dict[str, Any] | None = None,
) -> tuple[int, bool]:
    """Queue an index build for one paper and return (job_id, reused_existing)."""
    paper = get_accessible_paper(paper_id, current_user_id(), minimum_role="editor")
    if not paper:
        raise PermissionError("没有找到当前论文或无权构建索引。")
    parse_status = str(paper.get("parse_status") or "")
    if require_parsed and parse_status != "succeeded":
        raise ValueError("这篇论文还没有解析完成，暂时不能构建索引。")

    latest_job = latest_job_for_paper(int(paper["team_id"]), paper_id, "index")
    if latest_job and latest_job.get("status") in {"queued", "running"}:
        update_paper_status(paper_id, index_status=str(latest_job["status"]))
        state_label = "构建中" if latest_job.get("status") == "running" else "排队中"
        st.session_state[f"index_state_{paper_id}"] = {"vector": state_label, "bm25": state_label}
        return int(latest_job["job_id"]), True

    payload = {"paper_id": paper_id}
    payload.update(payload_extra or {})
    job_id = enqueue_job(
        "index",
        user_id=current_user_id(),
        team_id=int(paper["team_id"]),
        project_id=paper.get("project_id"),
        paper_id=paper_id,
        payload=payload,
    )
    update_paper_status(paper_id, index_status="queued")
    st.session_state[f"index_state_{paper_id}"] = {"vector": "排队中", "bm25": "排队中"}
    return job_id, False


def enqueue_upload_processing_pipeline(
    paper: dict[str, Any],
    *,
    include_images: bool = False,
) -> dict[str, Any]:
    """Queue parse and index jobs for an uploaded paper, reusing active jobs."""
    paper_id = str(paper.get("paper_id") or "")
    team_id = int(paper.get("team_id") or 0)
    if not paper_id or team_id <= 0:
        raise ValueError("paper metadata is missing paper_id or team_id")

    active_statuses = {"queued", "running"}
    parse_status = str(paper.get("parse_status") or "").strip().lower()
    index_status = str(paper.get("index_status") or "").strip().lower()

    latest_parse_job = latest_job_for_paper(team_id, paper_id, "parse")
    latest_parse_status = str((latest_parse_job or {}).get("status") or "").strip().lower()
    parse_job_id: int | None = None
    parse_reused = False

    parse_needed = parse_status != "succeeded" or latest_parse_status in active_statuses
    if latest_parse_status in active_statuses:
        parse_job_id = int(latest_parse_job["job_id"])
        parse_reused = True
        update_paper_status(paper_id, parse_status=latest_parse_status)
    elif parse_needed:
        update_paper_status(paper_id, parse_status="queued", index_status="queued")
        parse_job_id = enqueue_job(
            "parse",
            user_id=current_user_id(),
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

    latest_index_job = latest_job_for_paper(team_id, paper_id, "index")
    latest_index_status = str((latest_index_job or {}).get("status") or "").strip().lower()
    index_job_id: int | None = None
    index_reused = False
    should_queue_index = parse_needed or index_status != "succeeded"

    if latest_index_status in active_statuses:
        index_job_id = int(latest_index_job["job_id"])
        index_reused = True
        update_paper_status(paper_id, index_status=latest_index_status)
    elif should_queue_index:
        index_job_id = enqueue_job(
            "index",
            user_id=current_user_id(),
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
        st.session_state[f"index_state_{paper_id}"] = {"vector": "排队中", "bm25": "排队中"}

    return {
        "parse_job_id": parse_job_id,
        "index_job_id": index_job_id,
        "parse_reused": parse_reused,
        "index_reused": index_reused,
    }


def upload_processing_message(reused: bool, pipeline: dict[str, Any]) -> str:
    """Return the upload result message for the automatic processing pipeline."""
    prefix = "团队中已存在相同 PDF，已打开已有论文" if reused else "已保存到论文库"
    parse_job_id = pipeline.get("parse_job_id")
    index_job_id = pipeline.get("index_job_id")
    if parse_job_id and index_job_id:
        return f"{prefix}，已自动加入解析队列 #{parse_job_id} 和索引队列 #{index_job_id}。索引会等解析成功后再执行。"
    if parse_job_id:
        return f"{prefix}，已自动加入解析队列 #{parse_job_id}。"
    if index_job_id:
        return f"{prefix}，已自动加入索引队列 #{index_job_id}。"
    return f"{prefix}，解析和索引已可用。"


def queue_job_type_label(job_type: str) -> str:
    """Return a compact Chinese label for queue job types."""
    return {
        "parse": "解析",
        "index": "索引",
    }.get(str(job_type or "").strip().lower(), str(job_type or "任务"))


def queue_job_paper_label(job: dict[str, Any] | None) -> str:
    """Return a safe short paper label for queue displays."""
    if not job:
        return "空闲"
    payload = job.get("payload") if isinstance(job.get("payload"), dict) else {}
    save_path = str(payload.get("save_path") or "").strip()
    label = str(
        job.get("file_name")
        or payload.get("file_name")
        or (Path(save_path).name if save_path else "")
        or job.get("paper_id")
        or "未关联论文"
    ).strip()
    return label or "未关联论文"


def queue_job_display_state(job: dict[str, Any] | None) -> tuple[str, str]:
    """Return CSS modifier and readable state for one queue item."""
    if not job:
        return "idle", "空闲"
    status = str(job.get("status") or "").strip().lower()
    job_type = str(job.get("job_type") or "").strip().lower()
    if status == "running":
        return "active", f"运行中 · #{job.get('job_id')}"
    if (
        job_type == "index"
        and bool(job.get("paper_id"))
        and (job.get("queue_block_reason") or str(job.get("paper_parse_status") or "") != "succeeded")
    ):
        return "blocked", f"等待解析完成 · #{job.get('job_id')}"
    if status == "queued":
        return "queued", f"排队中 · #{job.get('job_id')}"
    return "idle", status or "空闲"


def render_queue_lane(label: str, job: dict[str, Any] | None) -> str:
    """Render one queue lane with running or waiting paper content."""
    paper_label = queue_job_paper_label(job)
    modifier, state = queue_job_display_state(job)
    return (
        f'<div class="pm-queue-lane pm-queue-lane-{modifier}">'
        f'<div class="pm-queue-lane-label">{html.escape(label)}</div>'
        f'<div class="pm-queue-lane-paper">{html.escape(paper_label)}</div>'
        f'<div class="pm-queue-lane-state">{html.escape(state)}</div>'
        "</div>"
    )


def queue_row_state_label(job: dict[str, Any]) -> str:
    """Return a short state for queue hover rows."""
    modifier, state = queue_job_display_state(job)
    if modifier == "blocked":
        return "等解析"
    if modifier == "queued":
        return "排队中"
    if modifier == "active":
        return "运行中"
    return state


def queue_remove_href(job_id: Any) -> str:
    """Return the URL used by the queue hover panel to remove one queued job."""
    return f"?{QUEUE_CANCEL_QUERY_PARAM}={html.escape(str(job_id))}"


def render_queue_rows(queued_jobs: list[dict[str, Any]], queued_count: int) -> str:
    """Render up to QUEUE_HOVER_LIMIT queued papers for the hover panel."""
    if not queued_jobs:
        return '<div class="pm-queue-empty">当前没有等待中的解析或索引任务。</div>'
    rows = []
    for position, job in enumerate(queued_jobs[:QUEUE_HOVER_LIMIT], start=1):
        job_type = queue_job_type_label(str(job.get("job_type") or ""))
        paper_label = queue_job_paper_label(job)
        job_id = job.get("job_id")
        state_label = queue_row_state_label(job)
        remove_link = ""
        if str(job.get("status") or "") == "queued":
            remove_link = (
                f'<a class="pm-queue-remove" href="{queue_remove_href(job_id)}" '
                'title="移除队列任务" aria-label="移除队列任务">×</a>'
            )
        rows.append(
            (
                '<div class="pm-queue-row">'
                f'<div class="pm-queue-type">{html.escape(job_type)}</div>'
                f'<div class="pm-queue-paper" title="{html.escape(paper_label)}">{html.escape(paper_label)}</div>'
                f'<div class="pm-queue-meta">#{html.escape(str(job_id))} · {html.escape(state_label)}</div>'
                f"{remove_link}"
                "</div>"
            )
        )
    more_count = max(0, int(queued_count) - len(queued_jobs[:QUEUE_HOVER_LIMIT]))
    if more_count:
        rows.append(f'<div class="pm-queue-more">还有 {more_count} 条排队任务未显示</div>')
    return "\n".join(rows)


@st.fragment(run_every=QUEUE_REFRESH_SECONDS)
def render_global_queue_progress(user_id: int, team_id: int) -> None:
    """Render the auto-refreshing global parse/index queue bar."""
    state_key = f"queue_progress_summary_{int(team_id)}"
    try:
        summary = queue_progress_summary(
            int(user_id),
            int(team_id),
            job_types=("parse", "index"),
            queued_limit=QUEUE_HOVER_LIMIT,
        )
        st.session_state[state_key] = summary
    except Exception as exc:  # pragma: no cover - defensive UI guard
        logger.warning("Queue progress summary failed. team_id=%s error=%s", team_id, exc)
        summary = st.session_state.get(state_key)
        if not isinstance(summary, dict):
            return

    running_by_type = summary.get("running_by_type") or {}
    queued_by_type = summary.get("queued_by_type") or {}
    blocked_by_type = summary.get("blocked_by_type") or {}
    parse_job = running_by_type.get("parse") or queued_by_type.get("parse")
    index_job = running_by_type.get("index") or queued_by_type.get("index") or blocked_by_type.get("index")
    running_count = int(summary.get("running_count") or 0)
    queued_count = int(summary.get("queued_count") or 0)
    queued_jobs = list(summary.get("queued") or [])
    active_total = running_count + queued_count
    has_active_work = active_total > 0
    fill_height = 100 if running_count else (46 if queued_count else 0)
    fill_class = "pm-queue-fill" if has_active_work else "pm-queue-fill pm-queue-fill-idle"
    queue_class = "pm-queue-bar pm-queue-active" if has_active_work else "pm-queue-bar pm-queue-idle"
    title = (
        f"后台队列：{running_count} 个处理中 · {queued_count} 个排队"
        if running_count
        else f"后台队列：{queued_count} 个等待中"
        if queued_count
        else "后台队列空闲"
    )
    subtitle = (
        "每 5 秒自动刷新 · 悬停查看排队论文"
        if has_active_work
        else "没有正在处理的解析或索引任务"
    )
    queued_title = f"排队论文（最多显示 {QUEUE_HOVER_LIMIT} 条，共 {queued_count} 条）"
    lanes_html = (
        "\n".join(
            [
                '<div class="pm-queue-current">',
                render_queue_lane("解析队列", parse_job),
                render_queue_lane("索引队列", index_job),
                "</div>",
            ]
        )
        if has_active_work
        else ""
    )
    idle_html = '<div class="pm-queue-idle-card">当前没有正在处理的解析或索引任务。</div>' if not has_active_work else ""
    popover_html = (
        "\n".join(
            [
                '<div class="pm-queue-popover">',
                f'<div class="pm-queue-popover-title">{html.escape(queued_title)}</div>',
                render_queue_rows(queued_jobs, queued_count),
                "</div>",
            ]
        )
        if has_active_work
        else ""
    )

    queue_html = "\n".join(
        [
            f'<div class="{queue_class}" role="status" aria-live="polite">',
            '<div class="pm-queue-header">',
            "<div>",
            f'<div class="pm-queue-title">{html.escape(title)}</div>',
            f'<div class="pm-queue-subtitle">{html.escape(subtitle)}</div>',
            "</div>",
            "</div>",
            '<div class="pm-queue-body">',
            '<div class="pm-queue-track" aria-hidden="true">',
            f'<div class="{fill_class}" style="height:{fill_height}%"></div>',
            "</div>",
            lanes_html,
            idle_html,
            "</div>",
            popover_html,
            "</div>",
        ]
    )
    st.markdown(
        queue_html,
        unsafe_allow_html=True,
    )


def clear_queue_for_ui_session_once(user_id: int, team_context: dict[str, Any]) -> None:
    """Clear queued jobs once when a new UI session starts for the selected team."""
    team_id = int(team_context.get("team_id") or 0)
    if not team_id or not can_write(str(team_context.get("role") or "")):
        return

    state_key = f"pm_queue_cleared_for_session_{team_id}"
    if st.session_state.get(state_key):
        return
    st.session_state[state_key] = True
    try:
        result = clear_team_queued_jobs(
            int(user_id),
            team_id,
            reason="queue cleared by UI refresh",
        )
    except Exception:
        logger.exception("Failed to clear queued jobs for UI session. team_id=%s", team_id)
        return

    cleared_count = int(result.get("cleared_count") or 0)
    if cleared_count:
        st.session_state["pm_queue_clear_notice"] = f"已清空 {cleared_count} 个未开始的后台任务，请手动选择当前要处理的论文。"


def render_queue_clear_notice() -> None:
    """Show a short notice after startup queue cleanup."""
    notice = st.session_state.pop("pm_queue_clear_notice", None)
    if notice:
        st.toast(str(notice))




def render_queue_action_notice() -> None:
    """Show a short notice after a queue item action."""
    notice = st.session_state.pop("pm_queue_action_notice", None)
    if notice:
        st.toast(str(notice))


def handle_queue_cancel_query(user_id: int) -> None:
    """Handle queue item removal links emitted inside the HTML hover panel."""
    try:
        raw_job_id = st.query_params.get(QUEUE_CANCEL_QUERY_PARAM)
    except Exception:
        return
    if isinstance(raw_job_id, list):
        raw_job_id = raw_job_id[0] if raw_job_id else ""
    if not raw_job_id:
        return

    try:
        job_id = int(str(raw_job_id))
        removed = cancel_queued_job(int(user_id), job_id)
        st.session_state["pm_queue_action_notice"] = (
            f"已移除队列任务 #{job_id}。"
            if removed
            else f"任务 #{job_id} 不在排队中，未移除。"
        )
    except Exception as exc:  # pragma: no cover - defensive UI guard
        logger.warning("Queue job removal failed. job_id=%s error=%s", raw_job_id, exc)
        st.session_state["pm_queue_action_notice"] = f"移除队列任务失败：{exc}"
    finally:
        try:
            del st.query_params[QUEUE_CANCEL_QUERY_PARAM]
        except Exception:
            try:
                st.query_params.clear()
            except Exception:
                pass
        st.rerun()

def render_app_shell() -> None:
    """Render a lightweight marker for the shared app shell."""
    st.markdown('<div class="pm-app-shell"></div>', unsafe_allow_html=True)








def render_empty_state(
    title: str,
    description: str,
    action_label: str | None = None,
    icon: str = "📄",
) -> None:
    """Render a calm empty state with an optional action hint."""
    action_html = (
        f'<div class="pm-empty-action">{html.escape(action_label)}</div>'
        if action_label
        else ""
    )
    st.markdown(
        f"""
        <div class="pm-empty-state">
          <div class="pm-empty-icon">{html.escape(icon)}</div>
          <div class="pm-empty-title">{html.escape(title)}</div>
          <div class="pm-empty-description">{html.escape(description)}</div>
          {action_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_error_card(title: str, message: str, detail: str | None = None) -> None:
    """Render a friendly error card and fold technical detail."""
    st.markdown(
        f"""
        <div class="pm-error-card">
          <div class="pm-section-title">{html.escape(title)}</div>
          <div class="pm-section-description">{html.escape(message)}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    if detail:
        with st.expander("技术详情", expanded=False):
            st.code(str(detail), language="text")


def render_auth_hero() -> None:
    """Render the branded product hero for the auth page."""
    st.markdown(
        f"""
        <div class="pm-auth-hero">
          <div class="pm-brand-pill">
            <span class="pm-brand-dot">PM</span>
            <span>PaperMate · AI Research Workspace</span>
          </div>
          <div class="pm-hero-title">把论文阅读、可信问答和文献卡片整理在一个工作台里</div>
          <div class="pm-hero-subtitle">
            上传 PDF，选择解析和索引，并基于原文引用回答问题。
          </div>
          <div class="pm-badges">
            {render_status_badge("数据私有", "primary")}
            {render_status_badge("可信引用", "info")}
            {render_status_badge("Markdown 导出", "default")}
          </div>
          <div class="pm-feature-grid">
            <div class="pm-feature-card"><strong>PDF 智能解析</strong><span>自动提取正文、章节、表格与 Markdown。</span></div>
            <div class="pm-feature-card"><strong>可信引用问答</strong><span>回答基于论文原文片段和检索证据。</span></div>
            <div class="pm-feature-card"><strong>文献卡片沉淀</strong><span>将研究问题、方法、贡献和局限保存进卡片库。</span></div>
          </div>
          <div class="pm-product-preview">
            <div class="pm-preview-top">
              <strong>论文工作台预览</strong>
              {render_status_badge("RAG 就绪", "success")}
            </div>
            <div class="pm-floating-card">
              <div class="pm-card-meta">论文正文 · Method · Page 3</div>
              <div class="pm-preview-line mid"></div>
              <div class="pm-preview-line"></div>
              <div class="pm-preview-line short"></div>
            </div>
            <div class="pm-floating-card">
              <div class="pm-card-meta">Ask PaperMate</div>
              <strong>核心方法是什么？</strong>
              <div class="pm-preview-line mid"></div>
              <div class="pm-badges" style="margin-top:10px;">
                {render_status_badge("[1] Page 3", "info")}
                {render_status_badge("答案有依据", "success")}
              </div>
            </div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_auth_card() -> None:
    """Render login and registration controls as a product card."""
    st.markdown(
        """
        <div class="pm-auth-card">
          <div class="pm-login-panel">
            <h2>欢迎回来</h2>
            <p>继续你的论文阅读与研究整理。</p>
          </div>
        """,
        unsafe_allow_html=True,
    )

    login_tab, register_tab = st.tabs(["登录", "注册"])

    with login_tab:
        with st.form("login_form_v2"):
            username = st.text_input("用户名", key="login_username", placeholder="输入用户名")
            password = st.text_input("密码", type="password", key="login_password", placeholder="输入密码")
            submitted = st.form_submit_button("登录 PaperMate", type="primary", use_container_width=True)

        if submitted:
            user = authenticate_user(username, password)
            if not user:
                render_error_card("登录失败", "请检查用户名或密码后再试。")
            else:
                prepare_user_workspace(int(user["user_id"]))
                set_current_user(user)
                st.toast("登录成功。")
                st.rerun()

    with register_tab:
        with st.form("register_form_v2"):
            username = st.text_input("用户名", key="register_username", placeholder="3-32 位用户名")
            password = st.text_input("密码", type="password", key="register_password", placeholder="至少 6 位")
            password_confirm = st.text_input(
                "确认密码",
                type="password",
                key="register_password_confirm",
                placeholder="再次输入密码",
            )
            submitted = st.form_submit_button("创建账户", type="primary", use_container_width=True)

        if submitted:
            if password != password_confirm:
                render_error_card("注册失败", "两次输入的密码不一致。")
            else:
                try:
                    user = create_user(username, password)
                    prepare_user_workspace(int(user["user_id"]))
                except (ValueError, OSError, sqlite3.Error) as exc:
                    render_error_card("注册失败", str(exc) or "注册失败，请稍后再试。")
                else:
                    set_current_user(user)
                    st.toast("账户已创建。")
                    st.rerun()

    st.markdown("</div>", unsafe_allow_html=True)




def render_sidebar(user: dict[str, Any]) -> str:
    """Render the research-workspace sidebar and return the selected page."""
    team_context = current_team_context()
    role = team_context["role"]
    processed_pdf = st.session_state.get("processed_pdf")
    paper_id = str((processed_pdf or {}).get("saved_file", {}).get("paper_id") or "")
    index_state = local_index_state(paper_id, allow_db=False) if paper_id else {"vector": "未知", "bm25": "未知"}
    rag_ready = index_state.get("vector") == "已构建" or index_state.get("bm25") == "已构建"

    st.sidebar.markdown(
        f"""
          <div class="pm-sidebar-brand">
            <div class="pm-sidebar-logo">PM</div>
            <div class="pm-sidebar-brand-title">PaperMate</div>
            <div class="pm-sidebar-brand-subtitle">RAG 论文阅读器</div>
          </div>
        <div class="pm-user-pill">
          当前用户
          <strong>{html.escape(str(user["username"]))}</strong>
        </div>
        <div class="pm-sidebar-section">研究工作台</div>
        """,
        unsafe_allow_html=True,
    )
    team_options = [int(team["team_id"]) for team in team_context["teams"]]
    selected_team_id = st.sidebar.selectbox(
        "团队",
        options=team_options,
        index=team_options.index(int(team_context["team_id"])),
        format_func=lambda team_id: next(
            f"{team['name']}（{team.get('role', 'viewer')}）"
            for team in team_context["teams"]
            if int(team["team_id"]) == int(team_id)
        ),
        key="current_team_id",
    )
    if int(selected_team_id) != int(team_context["team_id"]):
        st.session_state.pop("processed_pdf", None)
        st.rerun()

    project_options = [int(project["project_id"]) for project in team_context["projects"]]
    if project_options:
        selected_project_id = st.sidebar.selectbox(
            "项目",
            options=project_options,
            index=project_options.index(int(team_context["project_id"])),
            format_func=lambda project_id: next(
                project["name"]
                for project in team_context["projects"]
                if int(project["project_id"]) == int(project_id)
            ),
            key="current_project_id",
        )
        if int(selected_project_id) != int(team_context["project_id"]):
            st.session_state.pop("processed_pdf", None)
            st.rerun()

    page_labels = {
        "📄 论文工作台": "论文工作台",
        "📚 论文库": "论文库",
        "🗂 文献卡片库": "文献卡片库",
        "🧪 反馈记录": "反馈记录",
    }
    if can_manage_team(role):
        page_labels["👥 团队管理"] = "团队管理"
    pending_page = st.session_state.pop("pm_pending_nav_page", None)
    if pending_page in page_labels:
        st.session_state["pm_nav_page"] = pending_page
    if st.session_state.get("pm_nav_page") not in page_labels:
        st.session_state["pm_nav_page"] = next(iter(page_labels))
    selected_label = st.sidebar.radio(
        "页面",
        list(page_labels.keys()),
        key="pm_nav_page",
        label_visibility="collapsed",
    )

    st.sidebar.markdown(
        f"""
        <div class="pm-sidebar-footer">
          <div class="pm-badges">
            {render_status_badge("私有存储", "primary")}
          </div>
          <div style="margin-top:10px;">{render_status_badge("RAG 就绪" if rag_ready else "RAG 未就绪", "success" if rag_ready else "warning")}</div>
          <div style="margin-top:10px;">{render_status_badge(f"角色 {role}", "info")}</div>
          <div style="margin-top:10px;">版本：{html.escape(__version__)}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    if st.sidebar.button("退出登录", use_container_width=True):
        clear_authenticated_session()
        st.rerun()

    return page_labels[selected_label]








def render_index_builder(chunks: list[dict[str, Any]]) -> None:
    """Render the hybrid index build action with clearer status."""
    if not chunks:
        render_empty_state("还没有可检索内容", "论文正文为空，当前没有可入库的 chunk。", icon="🧩")
        return

    paper_id = str(chunks[0].get("paper_id") or "")
    index_state = local_index_state(paper_id)
    team_context = current_team_context()
    role = team_context["role"]
    can_edit = can_write(role)
    index_busy = index_state["vector"] in {"排队中", "构建中"} or index_state["bm25"] in {"排队中", "构建中"}
    st.markdown(
        f"""
        <div class="pm-section-card">
          <div class="pm-section-heading">
            <div>
              <h3 class="pm-section-title">构建论文索引</h3>
            </div>
            <div class="pm-badges">
              {render_status_badge(f"向量 {index_state['vector']}", index_status_type(index_state['vector']))}
              {render_status_badge(f"BM25 {index_state['bm25']}", index_status_type(index_state['bm25']))}
            </div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    latest_index_job = latest_job_for_paper(int(team_context["team_id"]), paper_id, "index")
    if latest_index_job:
        st.caption(
            f"最近索引任务：#{latest_index_job['job_id']} · {latest_index_job['status']} · {latest_index_job.get('updated_at') or latest_index_job.get('created_at')}"
        )
    if index_state["vector"] in {"排队中", "构建中"} or index_state["bm25"] in {"排队中", "构建中"}:
        st.info("索引正在后台排队或构建中。请保持 worker 运行并耐心等待，完成后即可开始基于原文问答。")

    if st.button(
        "构建论文索引",
        type="primary",
        use_container_width=True,
        key="build_paper_index",
        disabled=not can_edit or index_busy,
    ):
        try:
            job_id, reused_existing = enqueue_index_build_for_paper(paper_id)
            if reused_existing:
                st.info(f"索引任务已在队列中：#{job_id}。worker 会按顺序构建。")
            else:
                st.success(f"论文索引已入队：#{job_id}。worker 会在后台构建 Chroma 与 BM25 索引。")
            st.session_state["pm_workspace_notice"] = (
                f"索引任务已在队列中：#{job_id}。worker 会按顺序构建。"
                if reused_existing
                else f"论文索引已入队：#{job_id}。worker 会在后台构建 Chroma 与 BM25 索引。"
            )
            st.rerun()
        except Exception as exc:
            logger.exception("Index job enqueue failed. paper_id=%s", paper_id)
            render_error_card("索引任务创建失败", "请检查团队权限和数据库状态。", str(exc))
    if not can_edit:
        st.caption("当前团队角色为只读，无法构建索引。")


@st.cache_resource(show_spinner=False)
def get_safe_markdown_renderer() -> Any:
    """Return a Markdown renderer that does not allow raw HTML."""
    try:
        from markdown_it import MarkdownIt
    except ImportError:
        return None

    try:
        return MarkdownIt("default", {"html": False})
    except Exception:
        return MarkdownIt("commonmark", {"html": False})


def safe_inline_markdown(text: str) -> str:
    """Render a small safe subset of inline Markdown."""
    escaped = html.escape(text or "")
    escaped = re.sub(r"`([^`]+)`", r"<code>\1</code>", escaped)
    escaped = re.sub(r"\*\*([^*]+)\*\*", r"<strong>\1</strong>", escaped)
    escaped = re.sub(r"__([^_]+)__", r"<strong>\1</strong>", escaped)
    escaped = re.sub(r"(?<!\*)\*([^*\n]+)\*(?!\*)", r"<em>\1</em>", escaped)
    escaped = re.sub(r"(?<!_)_([^_\n]+)_(?!_)", r"<em>\1</em>", escaped)
    return escaped


def basic_markdown_to_html_fragment(markdown_text: str) -> str:
    """Fallback Markdown renderer for answers when markdown-it is unavailable."""
    lines = (markdown_text or "").splitlines()
    parts: list[str] = []
    paragraph: list[str] = []
    list_items: list[str] = []
    list_tag = ""

    def flush_paragraph() -> None:
        nonlocal paragraph
        if paragraph:
            parts.append(f"<p>{'<br>'.join(safe_inline_markdown(line.strip()) for line in paragraph)}</p>")
            paragraph = []

    def flush_list() -> None:
        nonlocal list_items, list_tag
        if list_items:
            items = "".join(f"<li>{item}</li>" for item in list_items)
            parts.append(f"<{list_tag}>{items}</{list_tag}>")
            list_items = []
            list_tag = ""

    for line in lines:
        stripped = line.strip()
        if not stripped:
            flush_paragraph()
            flush_list()
            continue

        heading_match = re.match(r"^(#{1,6})\s+(.+)$", stripped)
        if heading_match:
            flush_paragraph()
            flush_list()
            level = len(heading_match.group(1))
            parts.append(f"<h{level}>{safe_inline_markdown(heading_match.group(2))}</h{level}>")
            continue

        unordered_match = re.match(r"^[-*+]\s+(.+)$", stripped)
        ordered_match = re.match(r"^\d+[.)]\s+(.+)$", stripped)
        if unordered_match or ordered_match:
            flush_paragraph()
            current_tag = "ol" if ordered_match else "ul"
            if list_tag and list_tag != current_tag:
                flush_list()
            list_tag = current_tag
            item_text = (ordered_match or unordered_match).group(1)
            list_items.append(safe_inline_markdown(item_text))
            continue

        flush_list()
        paragraph.append(line)

    flush_paragraph()
    flush_list()
    return "".join(parts)


def answer_markdown_to_html(content: str) -> str:
    """Render model answers as Markdown while escaping raw HTML."""
    markdown_text = str(content or "")
    if not markdown_text.strip():
        return ""

    renderer = get_safe_markdown_renderer()
    if renderer is None:
        return basic_markdown_to_html_fragment(markdown_text)

    try:
        return renderer.render(markdown_text)
    except Exception:
        logger.debug("Markdown-it failed to render QA answer.", exc_info=True)
        return basic_markdown_to_html_fragment(markdown_text)


def render_chat_message(role: str, content: str) -> None:
    """Render a ChatGPT-style message bubble."""
    normalized_role = "user" if role == "user" else "assistant"
    role_label = "你" if normalized_role == "user" else "PaperMate"
    if normalized_role == "assistant":
        rendered_content = answer_markdown_to_html(str(content or ""))
    else:
        rendered_content = html.escape(str(content or "")).replace("\n", "<br>")
    st.markdown(
        f"""
        <div class="pm-chat-message pm-chat-{normalized_role}">
          <div class="pm-chat-role">{role_label}</div>
          <div class="pm-chat-content">{rendered_content}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_reference_card(ref: dict[str, Any], index: int, expanded: bool = False) -> None:
    """Render one trusted citation as an evidence card."""
    citation_id = ref.get("citation_id") or index
    page_label = format_page_label(ref.get("page_num"))
    section_title = ref.get("section_title") or "未知章节"
    source_ranks = ref.get("source_ranks") or {}
    vector_rank = source_ranks.get("vector") or ref.get("vector_rank") or "无"
    bm25_rank = source_ranks.get("bm25") or ref.get("bm25_rank") or "无"
    rrf_score = format_optional_float(ref.get("rrf_score")) or "无"
    preview = ref.get("text_preview") or ref.get("text") or ""
    relevance = "高相关" if ref.get("rrf_score") else "答案依据"
    st.markdown(
        f"""
        <div class="pm-reference-card">
          <div class="pm-reference-head">
            <div class="pm-reference-title">[{html.escape(str(citation_id))}] {html.escape(page_label)} · {html.escape(str(section_title))}</div>
            <div class="pm-badges">{render_status_badge(relevance, "success")}</div>
          </div>
          <div class="pm-reference-text">{html.escape(str(preview))}</div>
          <div class="pm-reference-meta">
            <div class="pm-reference-meta-item">RRF 分数<strong>{html.escape(rrf_score)}</strong></div>
            <div class="pm-reference-meta-item">Vector Rank<strong>{html.escape(str(vector_rank))}</strong></div>
            <div class="pm-reference-meta-item">BM25 Rank<strong>{html.escape(str(bm25_rank))}</strong></div>
            <div class="pm-reference-meta-item">Chunk ID<strong>{html.escape(str(ref.get("chunk_id", "")))}</strong></div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    render_source_jump_button(
        ref.get("chunk_id"),
        f"citation_{index}_{citation_id}_{ref.get('chunk_id', '')}",
    )


def render_citations(citations: list[dict[str, Any]]) -> None:
    """Render citations produced from chunk metadata."""
    st.markdown("#### 可信引用")
    if not citations:
        render_empty_state("暂无可信引用", "当前回答没有可展示的引用来源。", icon="🔎")
        return
    for index, citation in enumerate(citations[:3], start=1):
        render_reference_card(citation, index, expanded=True)
    if len(citations) > 3:
        with st.expander(f"查看更多引用（{len(citations) - 3} 条）", expanded=False):
            for index, citation in enumerate(citations[3:], start=4):
                render_reference_card(citation, index)


def render_qa_box(paper_id: str) -> None:
    """Render RAG question-answering controls."""
    team_context = current_team_context()
    can_edit = can_write(team_context["role"])
    index_state = local_index_state(paper_id)
    index_ready = index_state["vector"] == "已构建" or index_state["bm25"] == "已构建"
    index_busy = index_state["vector"] in {"排队中", "构建中"} or index_state["bm25"] in {"排队中", "构建中"}
    st.markdown('<span id="pm-qa-anchor" class="pm-qa-anchor"></span>', unsafe_allow_html=True)
    header_cols = st.columns([0.48, 0.16, 0.36], gap="small", vertical_alignment="center")
    with header_cols[0]:
        st.markdown('<h3 class="pm-section-title">Ask PaperMate</h3>', unsafe_allow_html=True)
    with header_cols[1]:
        if st.button("刷新", key=f"qa_refresh_index_state_{paper_id}", help="刷新索引状态", use_container_width=True):
            refresh_index_status_for_paper(paper_id, rerun=False)
            index_state = local_index_state(paper_id)
            index_ready = index_state["vector"] == "已构建" or index_state["bm25"] == "已构建"
            index_busy = index_state["vector"] in {"排队中", "构建中"} or index_state["bm25"] in {"排队中", "构建中"}
    with header_cols[2]:
        vector_badge = render_status_badge(
            f"向量 {index_state['vector']}",
            index_status_type(index_state["vector"]),
        )
        bm25_badge = render_status_badge(
            f"BM25 {index_state['bm25']}",
            index_status_type(index_state["bm25"]),
        )
        st.markdown(
            f'<div class="pm-badges pm-ask-badges-inline">{vector_badge}{bm25_badge}</div>',
            unsafe_allow_html=True,
        )

    if not index_ready:
        render_empty_state(
            "还没有可检索索引",
            "请先构建索引，然后再开始基于论文原文问答。",
            "先构建论文索引",
            icon="🧭",
        )
        if can_edit:
            if index_busy:
                st.info("索引任务已经在后台排队或构建中。完成后即可开始问答。")
            elif st.button("排队构建论文索引", type="primary", use_container_width=True, key=f"qa_enqueue_index_{paper_id}"):
                try:
                    job_id, reused_existing = enqueue_index_build_for_paper(paper_id)
                    if reused_existing:
                        st.info(f"索引任务已在队列中：#{job_id}。")
                    else:
                        st.success(f"论文索引已入队：#{job_id}。worker 完成后即可问答。")
                except Exception as exc:
                    logger.exception("Index job enqueue from QA failed. paper_id=%s", paper_id)
                    render_error_card("索引任务创建失败", "请检查团队权限和数据库状态。", str(exc))
        else:
            st.caption("当前团队角色为只读，不能创建索引任务。")
        return

    if not can_edit:
        render_empty_state(
            "只读角色不能提问",
            "当前团队角色为 viewer，可阅读论文和查看已有内容，但不能创建新的问答记录。",
            icon="READ",
        )
        return

    quick_questions = [
        "这篇论文解决了什么问题？",
        "核心方法是什么？",
        "实验结果说明了什么？",
        "有哪些局限性？",
        "帮我总结创新点",
    ]
    quick_cols = st.columns([1, 1, 1])
    for index, quick_question in enumerate(quick_questions):
        with quick_cols[index % 3]:
            if st.button(
                quick_question,
                key=f"quick_question_{paper_id}_{index}",
                use_container_width=True,
            ):
                st.session_state[f"question_{paper_id}"] = quick_question

    with st.form(key=f"ask_form_{paper_id}"):
        question = st.text_area(
            "输入你的问题",
            placeholder="例如：这篇论文的核心方法是什么？",
            key=f"question_{paper_id}",
            height=96,
        )
        submitted = st.form_submit_button("基于论文原文提问", type="primary", use_container_width=True)

    if submitted:
        if not question.strip():
            st.warning("请输入问题。")
            return
        try:
            with st.spinner("正在检索论文片段并生成回答..."):
                rag_result = answer_question(paper_id, question, user_id=current_user_id())
        except AppError as exc:
            logger.exception("RAG question answering failed. paper_id=%s", paper_id)
            if exc.code in {ErrorCode.VECTOR_SEARCH_FAILED, ErrorCode.BM25_INDEX_MISSING}:
                render_error_card("还没有可检索索引", "请先构建论文索引后再提问。", exc.detail)
            else:
                render_error_card("问答失败", "请查看日志或检查模型与索引配置。", exc.detail)
            return
        except Exception as exc:
            logger.exception("Unexpected RAG question answering failure. paper_id=%s", paper_id)
            render_error_card("问答失败", "请查看日志或检查模型与索引配置。", str(exc))
            return

        qa_log_id = rag_result.get("qa_id")
        if qa_log_id is None:
            try:
                qa_log_id = save_qa_log(paper_id, question.strip(), rag_result["answer"], user_id=current_user_id())
            except (OSError, sqlite3.Error):
                logger.exception("QA log save failed. paper_id=%s", paper_id)
                st.warning("问答记录保存失败，但不影响当前回答。")

        st.session_state[f"last_qa_{paper_id}"] = {
            "paper_id": paper_id,
            "question": question.strip(),
            "answer": rag_result["answer"],
            "citations": rag_result["citations"],
            "source_chunks": rag_result["source_chunks"],
            "retrieval_details": rag_result.get("retrieval_details") or rag_result.get("retrieval_debug", {}),
            "qa_log_id": qa_log_id,
        }

    qa_record = st.session_state.get(f"last_qa_{paper_id}")
    if qa_record:
        render_chat_message("user", qa_record["question"])
        render_chat_message("assistant", qa_record["answer"])
        if needs_index_warning(qa_record.get("retrieval_details", {})):
            st.warning("请先构建论文索引后再提问。")
        render_citations(qa_record["citations"])
        render_retrieval_details(qa_record.get("retrieval_details", {}))
        render_source_chunks(qa_record["source_chunks"])
        render_feedback_form(qa_record)


def render_literature_card_save(
    paper_id: str,
    chunks: list[dict[str, Any]],
    db_save_failed: bool,
    user_id: int,
) -> None:
    """Render literature-card generation and persistence."""
    team_context = current_team_context()
    can_edit = can_write(team_context["role"])
    st.markdown(
        f"""
        <div class="pm-section-card">
          <div class="pm-section-heading">
            <div>
              <h3 class="pm-section-title">生成文献卡片</h3>
            </div>
            <div class="pm-badges">{render_status_badge("Markdown 可编辑", "primary")}</div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    if not chunks:
        render_empty_state("暂时无法生成卡片", "论文正文为空，暂时无法生成文献卡片。", icon="🗂")
        return
    if db_save_failed:
        st.warning("数据库保存失败，文献卡片可能无法基于完整 chunks 生成。")
    if not can_edit:
        render_empty_state("只读角色不能生成卡片", "当前团队角色为 viewer，可查看已有卡片，不能创建新卡片。", icon="CARD")
        return

    try:
        libraries = list_card_libraries(user_id, team_id=int(team_context["team_id"]))
    except (OSError, sqlite3.Error) as exc:
        render_error_card("卡片库读取失败", "请检查 SQLite 数据库权限。", str(exc))
        return
    if not libraries:
        render_empty_state("没有可用卡片库", "当前用户还没有可用的卡片库，请先创建一个卡片库。", icon="🗂")
        return

    selected_library_id = st.selectbox(
        "保存到卡片库",
        options=[int(library["library_id"]) for library in libraries],
        format_func=lambda library_id: next(
            f"{library['name']}（{library.get('card_count', 0)} 张）"
            for library in libraries
            if int(library["library_id"]) == int(library_id)
        ),
        key=f"target_library_{paper_id}",
    )
    with st.expander("新建卡片库", expanded=False):
        render_library_create_form(user_id, key_suffix=f"_for_{paper_id}", team_id=int(team_context["team_id"]))

    generated_key = f"generated_card_markdown_{paper_id}"
    if st.button("生成文献卡片", type="primary", use_container_width=True, key=f"generate_card_{paper_id}"):
        try:
            job_id = enqueue_job(
                "card",
                user_id=user_id,
                team_id=int(team_context["team_id"]),
                project_id=team_context.get("project_id"),
                paper_id=paper_id,
                payload={"paper_id": paper_id, "library_id": int(selected_library_id)},
            )
            st.success(f"文献卡片生成任务已入队：#{job_id}。worker 完成后会自动保存到所选卡片库。")
        except Exception as exc:
            render_error_card("文献卡片任务创建失败", str(exc) or "请检查团队权限和数据库状态。")
            return

    latest_card_job = latest_job_for_paper(int(team_context["team_id"]), paper_id, "card")
    if latest_card_job:
        st.caption(
            f"最近卡片任务：#{latest_card_job['job_id']} · {latest_card_job['status']} · {latest_card_job.get('updated_at') or latest_card_job.get('created_at')}"
        )

    generated_markdown = st.session_state.get(generated_key)
    if generated_markdown:
        with st.expander("生成结果预览", expanded=True):
            st.markdown(generated_markdown)
            st.text_area(
                "Markdown 原文",
                value=generated_markdown,
                height=260,
                key=f"generated_card_preview_{paper_id}",
            )

        if st.button("保存到所选卡片库", type="primary", use_container_width=True, key=f"save_card_{paper_id}"):
            try:
                card_id = save_literature_card(
                    paper_id,
                    str(st.session_state.get(f"generated_card_preview_{paper_id}") or generated_markdown),
                    user_id=user_id,
                    library_id=int(selected_library_id),
                    team_id=int(team_context["team_id"]),
                    project_id=team_context.get("project_id"),
                )
            except (ValueError, OSError, sqlite3.Error) as exc:
                render_error_card("文献卡片保存失败", str(exc) or "请检查 SQLite 数据库权限。")
                return

            st.session_state[f"saved_card_id_{paper_id}"] = card_id
            library = get_card_library(int(selected_library_id), user_id, team_id=int(team_context["team_id"]))
            library_name = library["name"] if library else "所选卡片库"
            st.toast("文献卡片已保存。")
            st.success(f"新文献卡片已保存到「{library_name}」。")

    saved_card = get_literature_card_by_paper(paper_id, user_id=user_id, team_id=int(team_context["team_id"]))
    if saved_card:
        with st.expander("最近保存的卡片预览", expanded=False):
            render_literature_card(saved_card)












def require_feedback_admin_password() -> bool:
    """Require an administrator password before showing feedback records."""
    admin_password = feedback_admin_password()
    if not admin_password:
        render_error_card(
            "缺少管理员密码",
            "反馈记录页需要管理员密码，请配置 PAPERMATE_ADMIN_PASSWORD 或 PAPERMATE_APP_PASSWORD。",
        )
        return False
    if st.session_state.get("feedback_admin_authenticated"):
        return True

    center_cols = st.columns([0.24, 0.52, 0.24])
    with center_cols[1]:
        render_section_card("管理员验证", "输入管理员密码后查看用户反馈、Bad Case 和原始记录。")
        password = st.text_input("管理员密码", type="password", key="feedback_admin_password")
        if st.button("查看反馈记录 Dashboard", type="primary", use_container_width=True):
            if hmac.compare_digest(password, admin_password):
                st.session_state["feedback_admin_authenticated"] = True
                st.rerun()
            else:
                render_error_card("验证失败", "管理员密码不正确。")
    return False


def _feedback_badge_type(feedback_type: Any, is_negative: Any = 0) -> str:
    """Map feedback categories to badge tones."""
    text = str(feedback_type or "")
    if "编造" in text or "引用不支持" in text:
        return "danger"
    if "不准确" in text or "空泛" in text or "没有回答" in text:
        return "warning"
    return "danger" if int(is_negative or 0) else "success"


def render_feedback_records_page() -> None:
    """Render saved user feedback and bad cases as a quality dashboard."""
    team_context = current_team_context()
    render_app_header(
        "反馈记录",
        "集中查看用户反馈、负面样本和 Bad Case，用于持续优化检索与回答质量。",
        [render_status_badge("管理员", "warning"), render_status_badge("质量监控", "primary")],
    )
    if not require_feedback_admin_password():
        return

    feedback_rows = list_feedback_records(team_id=int(team_context["team_id"]))
    bad_case_rows = list_bad_cases(team_id=int(team_context["team_id"]))
    negative_count = sum(1 for row in feedback_rows if int(row.get("is_negative") or 0))
    unsupported_count = sum(1 for row in feedback_rows if row.get("feedback_type") == "引用不支持答案")
    hallucination_count = sum(1 for row in feedback_rows if row.get("feedback_type") == "模型编造")

    metric_cols = st.columns(5)
    with metric_cols[0]:
        render_metric_card("总反馈数", len(feedback_rows), "所有用户反馈", icon="◎")
    with metric_cols[1]:
        render_metric_card("负面反馈数", negative_count, "需要优先排查", icon="!", status="danger" if negative_count else "success")
    with metric_cols[2]:
        render_metric_card("Bad Case 数", len(bad_case_rows), "自动归档负面反馈", icon="△", status="warning" if bad_case_rows else "success")
    with metric_cols[3]:
        render_metric_card("引用不支持答案", unsupported_count, "引用质量风险", icon="↯", status="danger" if unsupported_count else "success")
    with metric_cols[4]:
        render_metric_card("模型编造", hallucination_count, "事实性风险", icon="?", status="danger" if hallucination_count else "success")

    tab_overview, tab_bad_cases, tab_raw = st.tabs(["反馈总览", "Bad Case", "原始记录"])
    with tab_overview:
        if not feedback_rows:
            render_empty_state("没有反馈记录", "暂无反馈。用户提交反馈后，会在这里汇总展示。", icon="🧪")
        else:
            overview_rows = [
                {
                    "反馈 ID": row.get("feedback_id"),
                    "论文": row.get("file_name") or row.get("paper_id") or "未关联",
                    "问题": row.get("question") or "",
                    "回答摘要": str(row.get("answer") or "")[:160],
                    "反馈类型": row.get("feedback_type") or "",
                    "是否 Bad Case": "是" if int(row.get("is_negative") or 0) else "否",
                    "提交时间": row.get("created_at"),
                }
                for row in feedback_rows
            ]
            st.dataframe(overview_rows, use_container_width=True, hide_index=True)
            for row in feedback_rows:
                title = f"#{row['feedback_id']} · {row.get('feedback_type') or ''} · {row.get('file_name') or row.get('paper_id') or '未关联论文'}"
                with st.expander(title):
                    badge_type = _feedback_badge_type(row.get("feedback_type"), row.get("is_negative"))
                    st.markdown(render_status_badge(row.get("feedback_type") or "未标注", badge_type), unsafe_allow_html=True)
                    st.write("问题：", row.get("question") or "无")
                    st.write("回答：", row.get("answer") or "无")
                    st.write("补充说明：", row.get("comment") or "无")
                    st.caption(f"提交时间：{row.get('created_at')}")

    with tab_bad_cases:
        if not bad_case_rows:
            render_empty_state("没有 Bad Case", "暂无 Bad Case。负面反馈会自动在这里归档。", icon="✓")
        else:
            for row in bad_case_rows:
                title = f"#{row['bad_case_id']} · {row.get('error_type') or ''} · {row.get('status') or ''}"
                with st.expander(title):
                    st.markdown(render_status_badge(row.get("error_type") or "负面反馈", _feedback_badge_type(row.get("error_type"), 1)), unsafe_allow_html=True)
                    st.write("用户问题：", row.get("question") or "无")
                    st.write("模型回答：", row.get("answer") or "无")
                    st.write("反馈原因：", row.get("reason") or row.get("notes") or "无")
                    st.write("解决方案：", row.get("solution") or "无")
                    st.caption(f"论文：{row.get('file_name') or row.get('paper_id') or '未关联'}")

    with tab_raw:
        if not feedback_rows:
            render_empty_state("没有原始记录", "暂无反馈。用户提交反馈后，会在这里汇总展示。", icon="🧾")
        else:
            st.dataframe(
                feedback_rows,
                use_container_width=True,
                hide_index=True,
                column_config={
                    "feedback_id": "反馈 ID",
                    "paper_id": "paper_id",
                    "file_name": "论文文件",
                    "qa_log_id": "问答 ID",
                    "question": "问题",
                    "answer": "回答",
                    "feedback_type": "反馈类型",
                    "is_negative": "是否负面",
                    "comment": "补充说明",
                    "created_at": "提交时间",
                },
            )


def inject_global_css() -> None:
    """Inject a stable, tidy PaperMate UI theme."""
    st.markdown(
        """
        <style>
        :root {
          --pm-bg: #F8FAFC;
          --pm-surface: rgba(255,255,255,.90);
          --pm-surface-solid: #FFFFFF;
          --pm-border: rgba(148,163,184,.24);
          --pm-border-strong: rgba(100,116,139,.30);
          --pm-text: #0F172A;
          --pm-muted: #64748B;
          --pm-subtle: #94A3B8;
          --pm-primary: #4F46E5;
          --pm-primary-2: #2563EB;
          --pm-cyan: #0891B2;
          --pm-success: #16A34A;
          --pm-warning: #D97706;
          --pm-danger: #DC2626;
          --pm-radius: 16px;
          --pm-radius-lg: 22px;
          --pm-shadow: 0 10px 30px rgba(15,23,42,.07);
          --pm-shadow-soft: 0 1px 2px rgba(15,23,42,.04);
        }
        html, body, .stApp, [data-testid="stAppViewContainer"], [data-testid="stHeader"] {
          color: var(--pm-text) !important;
          background:
            radial-gradient(circle at 12% 4%, rgba(79,70,229,.11), transparent 28%),
            radial-gradient(circle at 90% 0%, rgba(8,145,178,.10), transparent 30%),
            linear-gradient(135deg, #F8FAFC 0%, #F5F3FF 54%, #ECFEFF 100%) !important;
        }
        [data-testid="stDecoration"] { display: none; }
        .block-container {
          max-width: 1380px;
          padding-top: 1.1rem;
          padding-bottom: 3rem;
        }
        h1, h2, h3, h4, h5, h6 { color: var(--pm-text); letter-spacing: 0; }
        section[data-testid="stSidebar"] {
          background: rgba(255,255,255,.78) !important;
          border-right: 1px solid rgba(148,163,184,.20);
          backdrop-filter: blur(16px);
        }
        section[data-testid="stSidebar"] [role="radiogroup"] {
          display: grid;
          gap: 7px;
          margin-top: 10px;
        }
        section[data-testid="stSidebar"] [role="radiogroup"] label {
          border-radius: 13px;
          border: 1px solid transparent;
          padding: 9px 10px;
          transition: background 120ms ease, border-color 120ms ease;
        }
        section[data-testid="stSidebar"] [role="radiogroup"] label:hover {
          background: rgba(79,70,229,.06);
          border-color: rgba(79,70,229,.12);
        }
        section[data-testid="stSidebar"] [role="radiogroup"] label:has(input:checked) {
          background: rgba(238,242,255,.95);
          border-color: rgba(79,70,229,.20);
          box-shadow: inset 3px 0 0 var(--pm-primary);
        }
        .stButton > button,
        .stDownloadButton > button {
          border-radius: 12px !important;
          border: 1px solid var(--pm-border) !important;
          background: rgba(255,255,255,.92) !important;
          color: var(--pm-text) !important;
          font-weight: 700 !important;
          min-height: 2.45rem;
          transition: transform 120ms ease, box-shadow 120ms ease, border-color 120ms ease;
        }
        .stButton > button:hover,
        .stDownloadButton > button:hover {
          transform: translateY(-1px);
          border-color: rgba(79,70,229,.32) !important;
          box-shadow: var(--pm-shadow) !important;
        }
        .stButton > button[kind="primary"],
        .stDownloadButton > button[kind="primary"],
        button[kind="primary"] {
          border: 0 !important;
          color: #fff !important;
          background: linear-gradient(135deg, var(--pm-primary), var(--pm-primary-2)) !important;
          box-shadow: 0 10px 22px rgba(37,99,235,.18) !important;
        }
        div[data-testid="stTextInput"] input,
        div[data-testid="stTextArea"] textarea,
        div[data-testid="stSelectbox"] div[data-baseweb="select"] > div {
          border-radius: 12px !important;
          border-color: var(--pm-border) !important;
          background: rgba(255,255,255,.94) !important;
          color: var(--pm-text) !important;
        }
        div[data-testid="stTextInput"] input:focus,
        div[data-testid="stTextArea"] textarea:focus {
          border-color: rgba(79,70,229,.50) !important;
          box-shadow: 0 0 0 4px rgba(79,70,229,.09) !important;
        }
        div[data-testid="stFileUploader"] {
          border: 1px dashed rgba(79,70,229,.32);
          border-radius: var(--pm-radius-lg);
          padding: 12px;
          background: rgba(255,255,255,.72);
        }
        div[data-testid="stExpander"] {
          border: 1px solid var(--pm-border) !important;
          border-radius: var(--pm-radius) !important;
          background: rgba(255,255,255,.86) !important;
          box-shadow: var(--pm-shadow-soft);
          overflow: hidden;
        }
        div[data-testid="stTabs"] [role="tablist"] {
          gap: 6px;
          border-bottom: 1px solid var(--pm-border);
        }
        div[data-testid="stTabs"] [aria-selected="true"] {
          color: var(--pm-primary) !important;
          background: rgba(79,70,229,.08);
          border-radius: 10px 10px 0 0;
        }
        .pm-panel, .pm-hero, .pm-section-card, .pm-card, .pm-auth-card,
        .pm-empty-state, .pm-error-card, .pm-detail-panel, .pm-literature-card,
        .pm-reference-card, .pm-chat-message {
          border: 1px solid var(--pm-border);
          border-radius: var(--pm-radius-lg);
          background: var(--pm-surface);
          box-shadow: var(--pm-shadow);
          backdrop-filter: blur(14px);
        }
        .pm-hero {
          padding: 22px 24px;
          margin-bottom: 16px;
        }
        .pm-hero-top {
          display: flex;
          justify-content: space-between;
          align-items: flex-start;
          gap: 18px;
          flex-wrap: wrap;
        }
        .pm-hero h1 {
          margin: 0 0 6px 0;
          font-size: 32px;
          line-height: 1.18;
        }
        .pm-hero p {
          margin: 0;
          color: var(--pm-muted);
          line-height: 1.7;
          max-width: 820px;
        }
        .pm-badges, .pm-toolbar, .pm-chip-row {
          display: flex;
          align-items: center;
          gap: 8px;
          flex-wrap: wrap;
        }
        .pm-badge {
          display: inline-flex;
          align-items: center;
          border: 1px solid var(--pm-border);
          border-radius: 999px;
          padding: 4px 10px;
          background: rgba(248,250,252,.92);
          color: #475569;
          font-size: 12px;
          font-weight: 800;
          white-space: nowrap;
        }
        .pm-badge-primary { background: #EEF2FF; color: #4338CA; border-color: #C7D2FE; }
        .pm-badge-info { background: #ECFEFF; color: #0E7490; border-color: #A5F3FC; }
        .pm-badge-success { background: #F0FDF4; color: #15803D; border-color: #BBF7D0; }
        .pm-badge-warning { background: #FFFBEB; color: #B45309; border-color: #FDE68A; }
        .pm-badge-danger { background: #FEF2F2; color: #B91C1C; border-color: #FECACA; }
        .pm-queue-bar {
          position: fixed;
          top: 92px;
          right: 22px;
          width: 276px;
          min-width: 0;
          max-height: calc(100vh - 120px);
          z-index: 999999;
          border: 1px solid rgba(148,163,184,.26);
          border-radius: 16px;
          background:
            linear-gradient(135deg, rgba(255,255,255,.96), rgba(248,250,252,.90)),
            linear-gradient(90deg, rgba(79,70,229,.08), rgba(8,145,178,.08));
          box-shadow: 0 14px 34px rgba(15,23,42,.10);
          padding: 11px 12px;
          margin: 0;
          backdrop-filter: blur(14px);
        }
        .pm-queue-idle {
          background: rgba(255,255,255,.94);
          box-shadow: 0 6px 18px rgba(15,23,42,.06);
        }
        .pm-queue-header {
          display: block;
        }
        .pm-queue-title {
          font-size: 13px;
          font-weight: 850;
          color: var(--pm-text);
          line-height: 1.3;
        }
        .pm-queue-subtitle {
          color: var(--pm-muted);
          font-size: 11px;
          line-height: 1.35;
          margin-top: 3px;
        }
        .pm-queue-body {
          display: grid;
          grid-template-columns: 8px minmax(0, 1fr);
          gap: 10px;
          align-items: stretch;
          margin-top: 10px;
        }
        .pm-queue-current {
          display: grid;
          grid-template-columns: 1fr;
          gap: 8px;
        }
        .pm-queue-lane {
          min-width: 0;
          border: 1px solid rgba(148,163,184,.25);
          border-radius: 10px;
          background: rgba(255,255,255,.74);
          padding: 8px 9px 8px 10px;
          box-shadow: inset 3px 0 0 rgba(148,163,184,.42);
        }
        .pm-queue-lane-active {
          background: #F8FFFD;
          border-color: rgba(20,184,166,.28);
          box-shadow: inset 3px 0 0 #14B8A6, 0 8px 18px rgba(20,184,166,.08);
        }
        .pm-queue-lane-queued {
          background: #F8FAFF;
          border-color: rgba(79,70,229,.22);
          box-shadow: inset 3px 0 0 var(--pm-primary);
        }
        .pm-queue-lane-blocked {
          background: #FFFBEB;
          border-color: rgba(217,119,6,.24);
          box-shadow: inset 3px 0 0 var(--pm-warning);
        }
        .pm-queue-lane-idle {
          background: rgba(248,250,252,.74);
        }
        .pm-queue-lane-label {
          color: var(--pm-muted);
          font-size: 11px;
          font-weight: 850;
          line-height: 1.1;
        }
        .pm-queue-lane-paper {
          min-width: 0;
          overflow: hidden;
          text-overflow: ellipsis;
          white-space: nowrap;
          color: var(--pm-text);
          font-size: 12px;
          font-weight: 850;
          line-height: 1.25;
          margin-top: 3px;
        }
        .pm-queue-lane-state {
          color: var(--pm-muted);
          font-size: 10px;
          line-height: 1.2;
          margin-top: 2px;
        }
        .pm-queue-lane-active .pm-queue-lane-state {
          color: #0F766E;
          font-weight: 850;
        }
        .pm-queue-lane-queued .pm-queue-lane-state {
          color: #4338CA;
          font-weight: 850;
        }
        .pm-queue-lane-blocked .pm-queue-lane-state {
          color: #B45309;
          font-weight: 850;
        }
        .pm-queue-track {
          position: relative;
          width: 8px;
          height: 100%;
          min-height: 112px;
          border-radius: 999px;
          background: #E2E8F0;
          overflow: hidden;
          margin: 0;
        }
        .pm-queue-idle .pm-queue-track {
          background: #E2E8F0;
          min-height: 48px;
        }
        .pm-queue-track::before {
          display: none;
        }
        .pm-queue-fill {
          position: absolute;
          left: 0;
          right: 0;
          bottom: 0;
          width: 100%;
          border-radius: inherit;
          background: linear-gradient(180deg, var(--pm-primary), var(--pm-cyan), var(--pm-success));
          box-shadow: 0 0 14px rgba(8,145,178,.18);
          transition: height 180ms ease;
          overflow: hidden;
        }
        .pm-queue-fill::after {
          display: none;
        }
        .pm-queue-fill-idle {
          background: transparent;
          box-shadow: none;
          animation: none;
          transition: none;
        }
        .pm-queue-fill-idle::after {
          display: none;
        }
        .pm-queue-idle-card {
          border: 1px solid rgba(148,163,184,.22);
          border-radius: 10px;
          background: rgba(248,250,252,.78);
          color: var(--pm-muted);
          font-size: 12px;
          line-height: 1.45;
          padding: 10px;
        }
        .pm-queue-popover {
          display: none;
          position: absolute;
          left: 0;
          right: 0;
          top: calc(100% + 8px);
          width: auto;
          max-height: calc(100vh - 330px);
          overflow-y: auto;
          z-index: 50;
          border: 1px solid var(--pm-border);
          border-radius: 14px;
          background: rgba(255,255,255,.98);
          box-shadow: 0 18px 45px rgba(15,23,42,.14);
          padding: 11px;
          backdrop-filter: blur(12px);
        }
        .pm-queue-bar:hover .pm-queue-popover {
          display: block;
        }
        .pm-queue-popover-title {
          color: var(--pm-muted);
          font-size: 12px;
          font-weight: 850;
          margin-bottom: 7px;
        }
        .pm-queue-row {
          display: grid;
          grid-template-columns: 64px minmax(0, 1fr) 88px;
          gap: 8px;
          align-items: center;
          position: relative;
          padding: 8px 0;
          padding-right: 24px;
          border-top: 1px solid #EEF2F7;
          font-size: 12px;
        }
        .pm-queue-row:first-of-type {
          border-top: 0;
        }
        .pm-queue-type {
          color: #0E7490;
          font-weight: 850;
        }
        .pm-queue-paper {
          min-width: 0;
          overflow: hidden;
          text-overflow: ellipsis;
          white-space: nowrap;
          color: var(--pm-text);
          font-weight: 750;
        }
        .pm-queue-meta {
          color: var(--pm-muted);
          text-align: right;
          white-space: nowrap;
        }
        .pm-queue-remove {
          position: absolute;
          right: 0;
          top: 50%;
          transform: translateY(-50%);
          display: inline-flex;
          align-items: center;
          justify-content: center;
          width: 20px;
          height: 20px;
          border-radius: 50%;
          color: #64748B;
          background: rgba(241,245,249,.96);
          border: 1px solid rgba(148,163,184,.36);
          text-decoration: none;
          font-size: 15px;
          font-weight: 850;
          line-height: 1;
          opacity: 0;
          pointer-events: none;
          transition: opacity 120ms ease, color 120ms ease, background 120ms ease;
        }
        .pm-queue-row:hover .pm-queue-remove {
          opacity: 1;
          pointer-events: auto;
        }
        .pm-queue-remove:hover {
          color: #B91C1C;
          background: #FEE2E2;
          border-color: rgba(248,113,113,.56);
        }
        .pm-queue-empty, .pm-queue-more {
          color: var(--pm-muted);
          font-size: 12px;
          padding: 8px 0 2px;
        }
        .pm-queue-more {
          border-top: 1px solid #EEF2F7;
          margin-top: 2px;
        }
        @media (min-width: 1180px) {
          .stApp:has(.pm-queue-bar) .block-container {
            padding-right: 318px;
          }
        }
        @media (max-width: 760px) {
          .pm-queue-bar {
            left: 10px;
            right: 10px;
            top: auto;
            bottom: 14px;
            width: auto;
          }
          .pm-queue-popover {
            left: 0;
            right: 0;
            top: auto;
            bottom: calc(100% + 8px);
            width: auto;
            max-height: 48vh;
          }
          .pm-queue-row {
            grid-template-columns: 52px minmax(0, 1fr) 64px;
          }
        }
        .pm-sidebar-brand, .pm-user-pill, .pm-sidebar-footer {
          border: 1px solid var(--pm-border);
          border-radius: 16px;
          background: rgba(255,255,255,.76);
          padding: 13px;
          box-shadow: var(--pm-shadow-soft);
          margin-bottom: 12px;
        }
        .pm-sidebar-logo {
          width: 34px;
          height: 34px;
          display: inline-grid;
          place-items: center;
          border-radius: 11px;
          background: linear-gradient(135deg, var(--pm-primary), var(--pm-cyan));
          color: #fff;
          font-weight: 850;
          margin-bottom: 10px;
        }
        .pm-sidebar-brand-title { font-size: 19px; font-weight: 850; }
        .pm-sidebar-brand-subtitle, .pm-user-pill, .pm-sidebar-footer {
          color: var(--pm-muted);
          font-size: 12px;
          line-height: 1.55;
        }
        .pm-sidebar-section {
          margin: 15px 0 7px;
          color: var(--pm-subtle);
          font-size: 11px;
          font-weight: 850;
          letter-spacing: .08em;
          text-transform: uppercase;
        }
        .pm-auth-shell {
          padding: 34px 0 12px;
        }
        .pm-auth-intro {
          text-align: center;
          max-width: 760px;
          margin: 0 auto 18px;
        }
        .pm-auth-intro h1 {
          margin: 0 0 8px;
          font-size: 42px;
          line-height: 1.12;
        }
        .pm-auth-intro p {
          margin: 0 auto;
          color: var(--pm-muted);
          line-height: 1.7;
          max-width: 640px;
        }
        .pm-auth-card {
          max-width: 460px;
          margin: 0 auto;
          padding: 24px;
        }
        div[data-testid="stVerticalBlockBorderWrapper"] {
          border-color: var(--pm-border) !important;
          border-radius: var(--pm-radius-lg) !important;
          background: rgba(255,255,255,.90) !important;
          box-shadow: var(--pm-shadow) !important;
        }
        .pm-auth-title {
          margin: 0 0 4px;
          font-size: 24px;
          font-weight: 850;
        }
        .pm-auth-subtitle {
          margin: 0 0 18px;
          color: var(--pm-muted);
          line-height: 1.6;
        }
        .pm-privacy-note {
          border: 1px solid rgba(22,163,74,.18);
          border-radius: 14px;
          padding: 11px 12px;
          background: rgba(240,253,244,.76);
          color: #166534;
          font-size: 13px;
          line-height: 1.55;
          margin-top: 14px;
        }
        .pm-section-card, .pm-panel {
          padding: 17px;
          margin-bottom: 14px;
        }
        .pm-section-heading {
          display: flex;
          align-items: flex-start;
          justify-content: space-between;
          gap: 12px;
        }
        .pm-ask-badges-inline {
          justify-content: flex-end;
        }
        .pm-section-title {
          margin: 0;
          color: var(--pm-text);
          font-size: 18px;
          font-weight: 850;
        }
        .pm-section-description {
          margin: 5px 0 0;
          color: var(--pm-muted);
          font-size: 13px;
          line-height: 1.6;
        }
        .pm-metric-card {
          min-height: 96px;
          border: 1px solid var(--pm-border);
          border-radius: 16px;
          background: rgba(255,255,255,.86);
          padding: 14px;
          box-shadow: var(--pm-shadow-soft);
        }
        .pm-metric-label { color: var(--pm-muted); font-size: 12px; font-weight: 850; margin-bottom: 6px; }
        .pm-metric-value { color: var(--pm-text); font-size: 18px; font-weight: 850; line-height: 1.25; word-break: break-word; }
        .pm-metric-helper { color: var(--pm-muted); font-size: 12px; line-height: 1.35; margin-top: 7px; }
        .pm-metric-icon {
          width: 28px;
          height: 28px;
          display: inline-grid;
          place-items: center;
          border-radius: 10px;
          background: #EEF2FF;
          color: var(--pm-primary);
          margin-bottom: 8px;
        }
        .pm-workflow {
          display: grid;
          grid-template-columns: repeat(5, minmax(0, 1fr));
          gap: 10px;
          margin: 2px 0 15px;
        }
        .pm-step {
          display: flex;
          align-items: center;
          gap: 9px;
          border: 1px solid var(--pm-border);
          border-radius: 15px;
          background: rgba(255,255,255,.82);
          padding: 10px;
        }
        .pm-step-dot {
          width: 25px;
          height: 25px;
          border-radius: 999px;
          display: inline-grid;
          place-items: center;
          background: #E2E8F0;
          color: #64748B;
          font-size: 12px;
          font-weight: 850;
          flex: 0 0 auto;
        }
        .pm-step-title { color: var(--pm-text); font-size: 13px; font-weight: 850; }
        .pm-step-helper { color: var(--pm-subtle); font-size: 11px; margin-top: 2px; }
        .pm-step-done .pm-step-dot { background: #DCFCE7; color: var(--pm-success); }
        .pm-step-active { background: #EEF2FF; border-color: #C7D2FE; }
        .pm-step-active .pm-step-dot { background: var(--pm-primary); color: #fff; }
        .pm-step-error { background: #FEF2F2; border-color: #FECACA; }
        .pm-step-error .pm-step-dot { background: var(--pm-danger); color: #fff; }
        .pm-file-capsule {
          display: flex;
          justify-content: space-between;
          align-items: center;
          gap: 12px;
          border: 1px solid var(--pm-border);
          border-radius: 16px;
          background: rgba(255,255,255,.82);
          padding: 12px 14px;
          margin: 8px 0 12px;
        }
        .pm-small, .pm-card-meta {
          color: var(--pm-muted);
          font-size: 12px;
          line-height: 1.5;
        }
        .pm-reader-note {
          border: 1px solid var(--pm-border);
          border-radius: var(--pm-radius-lg);
          background: rgba(255,255,255,.92);
          padding: 18px;
          margin-bottom: 16px;
          box-shadow: var(--pm-shadow);
        }
        .pm-reader-note p, .pm-reader-note li { line-height: 1.76; color: #1E293B; }
        .pm-chat-message {
          padding: 13px 14px;
          margin: 10px 0;
          line-height: 1.72;
        }
        .pm-chat-user { background: #EEF2FF; margin-left: 6%; }
        .pm-chat-assistant { background: rgba(255,255,255,.92); margin-right: 3%; }
        .pm-chat-role { color: var(--pm-muted); font-size: 12px; font-weight: 850; margin-bottom: 4px; }
        .pm-chat-content h1, .pm-chat-content h2, .pm-chat-content h3,
        .pm-chat-content h4, .pm-chat-content h5, .pm-chat-content h6 {
          margin: 8px 0 6px;
          color: var(--pm-text);
          line-height: 1.35;
        }
        .pm-chat-content h3 { font-size: 16px; }
        .pm-chat-content p { margin: 6px 0 10px; }
        .pm-chat-content p:last-child { margin-bottom: 0; }
        .pm-chat-content ul, .pm-chat-content ol { margin: 6px 0 10px 22px; padding: 0; }
        .pm-chat-content code {
          border: 1px solid rgba(148,163,184,.25);
          border-radius: 6px;
          background: #F8FAFC;
          padding: 1px 5px;
          font-size: .92em;
        }
        .pm-reference-card {
          padding: 14px;
          margin: 10px 0;
          background: rgba(255,255,255,.92);
        }
        .pm-reference-head {
          display: flex;
          justify-content: space-between;
          gap: 10px;
          flex-wrap: wrap;
          margin-bottom: 10px;
        }
        .pm-reference-title { color: #1D4ED8; font-weight: 850; }
        .pm-reference-text {
          border-left: 3px solid rgba(37,99,235,.34);
          border-radius: 12px;
          background: #F8FAFC;
          padding: 11px 12px;
          color: #1E293B;
          line-height: 1.7;
          white-space: pre-wrap;
        }
        .pm-reference-meta {
          display: grid;
          grid-template-columns: repeat(4, minmax(0, 1fr));
          gap: 8px;
          margin-top: 10px;
        }
        .pm-reference-meta-item {
          border: 1px solid rgba(148,163,184,.18);
          border-radius: 10px;
          background: #F8FAFC;
          padding: 8px;
          color: var(--pm-muted);
          font-size: 11px;
        }
        .pm-reference-meta-item strong {
          display: block;
          color: var(--pm-text);
          font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
          font-size: 12px;
          margin-top: 2px;
          word-break: break-word;
        }
        .pm-source-link {
          display: inline-flex;
          border: 1px solid #BFDBFE;
          border-radius: 999px;
          padding: 3px 10px;
          background: #EFF6FF;
          color: #1D4ED8;
          font-size: 12px;
          font-weight: 800;
          text-decoration: none;
          margin-top: 4px;
        }
        .pm-source-anchor { display: block; scroll-margin-top: 18px; height: 1px; }
        .pm-source-highlight {
          border-radius: 8px;
          outline: 2px solid rgba(245,158,11,.72);
          background: rgba(254,240,138,.58) !important;
          box-shadow: 0 0 0 5px rgba(245,158,11,.16);
          transition: background 220ms ease, box-shadow 220ms ease, outline-color 220ms ease;
          animation: pm-source-highlight-pulse 2600ms ease 1;
        }
        @keyframes pm-source-highlight-pulse {
          0% {
            background: rgba(251,191,36,.78);
            box-shadow: 0 0 0 8px rgba(245,158,11,.24);
          }
          55% {
            background: rgba(254,240,138,.62);
            box-shadow: 0 0 0 5px rgba(245,158,11,.16);
          }
          100% {
            background: rgba(254,240,138,.42);
            box-shadow: 0 0 0 4px rgba(245,158,11,.10);
          }
        }
        .pm-qa-anchor {
          display: block;
          height: 1px;
          scroll-margin-top: 18px;
        }
        .pm-return-qa-button {
          position: fixed;
          right: 24px;
          bottom: 24px;
          z-index: 2147483647;
          border: 1px solid rgba(37,99,235,.28);
          border-radius: 999px;
          background: #1D4ED8;
          color: #fff;
          padding: 10px 14px;
          font-size: 13px;
          font-weight: 850;
          line-height: 1;
          box-shadow: 0 12px 30px rgba(15,23,42,.22);
          cursor: pointer;
        }
        .pm-return-qa-button:hover {
          background: #1E40AF;
        }
        @media (max-width: 760px) {
          .pm-return-qa-button {
            right: 14px;
            bottom: 14px;
            padding: 10px 12px;
          }
        }
        .pm-literature-card {
          padding: 15px;
          margin-bottom: 12px;
          background: rgba(255,255,255,.92);
          transition: border-color 120ms ease, box-shadow 120ms ease;
        }
        .pm-literature-card-selected {
          border-color: rgba(79,70,229,.48);
          box-shadow: 0 0 0 4px rgba(79,70,229,.08), var(--pm-shadow);
        }
        .pm-card-title {
          color: var(--pm-text);
          font-size: 17px;
          font-weight: 850;
          line-height: 1.38;
          margin-bottom: 7px;
        }
        .pm-chip {
          display: inline-flex;
          border: 1px solid rgba(148,163,184,.22);
          border-radius: 999px;
          padding: 3px 9px;
          background: #F8FAFC;
          color: #475569;
          font-size: 11px;
          font-weight: 750;
        }
        .pm-card-summary {
          border: 1px solid rgba(148,163,184,.16);
          border-radius: 13px;
          background: #F8FAFC;
          padding: 10px 11px;
          color: #334155;
          font-size: 13px;
          line-height: 1.62;
          margin-top: 10px;
        }
        .pm-card-footer {
          display: flex;
          justify-content: space-between;
          gap: 10px;
          align-items: center;
          border-top: 1px solid rgba(148,163,184,.15);
          padding-top: 10px;
          margin-top: 10px;
          color: var(--pm-subtle);
          font-size: 11px;
        }
        .pm-detail-panel {
          padding: 17px;
          margin-bottom: 14px;
          background: rgba(255,255,255,.92);
        }
        .pm-detail-title {
          margin: 0 0 8px;
          color: var(--pm-text);
          font-size: 22px;
          line-height: 1.3;
          font-weight: 900;
        }
        .pm-detail-section {
          border: 1px solid rgba(148,163,184,.16);
          border-radius: 13px;
          background: #F8FAFC;
          padding: 11px 12px;
          margin-top: 9px;
        }
        .pm-detail-section strong {
          display: block;
          color: var(--pm-primary);
          font-size: 12px;
          margin-bottom: 4px;
        }
        .pm-detail-section div {
          color: #1E293B;
          line-height: 1.65;
          white-space: pre-wrap;
        }
        .pm-empty-state {
          text-align: center;
          padding: 28px 22px;
          margin: 14px 0;
          background: rgba(255,255,255,.86);
        }
        .pm-empty-icon {
          width: 44px;
          height: 44px;
          display: grid;
          place-items: center;
          border-radius: 16px;
          background: #EEF2FF;
          color: var(--pm-primary);
          margin: 0 auto 10px;
          font-size: 20px;
        }
        .pm-empty-title { color: var(--pm-text); font-size: 18px; font-weight: 850; margin-bottom: 6px; }
        .pm-empty-description { color: var(--pm-muted); line-height: 1.65; max-width: 620px; margin: 0 auto; }
        .pm-empty-action {
          display: inline-flex;
          margin-top: 13px;
          border-radius: 999px;
          padding: 6px 12px;
          background: #EEF2FF;
          color: var(--pm-primary);
          font-size: 12px;
          font-weight: 850;
        }
        .pm-error-card {
          padding: 15px;
          margin: 12px 0;
          background: #FFFBFB;
          border-color: #FECACA;
        }
        .pm-error-card .pm-section-title { color: #B91C1C; }
        .pm-interleaved-reader {
          border: 1px solid var(--pm-border);
          border-radius: var(--pm-radius-lg);
          background: var(--pm-surface-solid);
          overflow: hidden;
          box-shadow: var(--pm-shadow-soft);
          margin-top: 12px;
        }
        .pm-interleaved-toolbar {
          display: flex;
          justify-content: space-between;
          align-items: center;
          gap: 12px;
          padding: 14px 18px;
          border-bottom: 1px solid var(--pm-border);
          background: linear-gradient(180deg, #FFFFFF, #F8FAFC);
          flex-wrap: wrap;
        }
        .pm-interleaved-toolbar-title {
          color: var(--pm-text);
          font-size: 16px;
          font-weight: 900;
        }
        .pm-interleaved-toolbar-meta {
          color: var(--pm-muted);
          font-size: 12px;
          margin-top: 3px;
        }
        .pm-bilingual-flow {
          padding: 22px;
          max-height: 72vh;
          overflow-y: auto;
          background: linear-gradient(180deg, rgba(248,250,252,.42), #FFFFFF 18%);
        }
        .pm-bilingual-block {
          border: 1px solid rgba(148,163,184,.18);
          border-radius: 18px;
          background: #FFFFFF;
          margin-bottom: 18px;
          overflow: hidden;
          transition: all 0.18s ease;
        }
        .pm-bilingual-block:hover {
          border-color: rgba(79,70,229,.28);
          box-shadow: 0 12px 30px rgba(15,23,42,.07);
          transform: translateY(-1px);
        }
        .pm-source-block {
          padding: 18px 22px 12px;
          background: #FFFFFF;
          color: #111827;
          line-height: 1.72;
          font-size: 15px;
        }
        .pm-target-block {
          padding: 14px 22px 18px;
          background: linear-gradient(180deg, #FCFCFF, #F8FAFF);
          border-top: 1px solid rgba(148,163,184,.14);
          border-left: 3px solid var(--pm-primary);
          color: #1E293B;
          line-height: 1.85;
          font-size: 15px;
          font-family: "PingFang SC", "Microsoft YaHei", "Noto Sans CJK SC", system-ui, sans-serif;
        }
        .pm-lang-label {
          display: inline-flex;
          align-items: center;
          gap: 6px;
          font-size: 11px;
          font-weight: 800;
          letter-spacing: 0.04em;
          margin-bottom: 8px;
          padding: 3px 8px;
          border-radius: 999px;
        }
        .pm-lang-label-source {
          color: #475569;
          background: #F1F5F9;
        }
        .pm-lang-label-target {
          color: #4338CA;
          background: #EEF2FF;
        }
        .pm-block-content {
          overflow-wrap: anywhere;
        }
        .pm-block-content p,
        .pm-block-content li {
          line-height: inherit;
        }
        .pm-block-content h1,
        .pm-block-content h2,
        .pm-block-content h3,
        .pm-block-content h4 {
          margin: 0.2rem 0 0.75rem;
          line-height: 1.28;
          letter-spacing: 0;
        }
        .pm-block-content table {
          width: 100%;
          border-collapse: collapse;
          margin: 10px 0;
          font-size: 13px;
        }
        .pm-block-content th,
        .pm-block-content td {
          border: 1px solid rgba(148,163,184,.22);
          padding: 7px 8px;
          vertical-align: top;
        }
        .pm-block-content pre {
          border-radius: 12px;
          background: #F8FAFC;
          padding: 12px;
          overflow-x: auto;
        }
        .pm-block-placeholder {
          color: #94A3B8;
          font-style: italic;
          margin: 0;
        }
        .pm-bilingual-image-notice {
          margin: 0 0 14px;
          padding: 10px 14px;
          border: 1px dashed rgba(148,163,184,.32);
          border-radius: 12px;
          background: #F8FAFC;
          color: #64748B;
          font-size: 13px;
          font-weight: 750;
          line-height: 1.5;
        }
        .pm-bilingual-block-heading {
          border-color: rgba(79,70,229,.24);
        }
        .pm-bilingual-block-heading .pm-source-block,
        .pm-bilingual-block-heading .pm-target-block {
          padding-top: 22px;
        }
        .pm-align-warning {
          margin: 12px 0 18px;
          padding: 12px 14px;
          border-radius: 14px;
          background: #FFFBEB;
          border: 1px solid #FDE68A;
          color: #92400E;
          font-size: 14px;
        }
        @media (max-width: 980px) {
          .pm-workflow, .pm-reference-meta { grid-template-columns: 1fr; }
          .pm-auth-intro h1 { font-size: 34px; }
          .pm-chat-user, .pm-chat-assistant { margin-left: 0; margin-right: 0; }
          .pm-bilingual-flow {
            padding: 14px;
            max-height: none;
          }
          .pm-source-block,
          .pm-target-block {
            padding-left: 16px;
            padding-right: 16px;
          }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_app_header(
    title: str,
    subtitle: str,
    status_badges: list[str] | None = None,
) -> None:
    badges = "".join(status_badges or [])
    st.markdown(
        f"""
        <div class="pm-hero">
          <div class="pm-hero-top">
            <div>
              <h1>{html.escape(title)}</h1>
              <p>{html.escape(subtitle)}</p>
            </div>
            <div class="pm-badges">{badges}</div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_metric_card(
    label: str,
    value: Any,
    helper: str | None = None,
    icon: str | None = None,
    status: str | None = None,
) -> None:
    status_name = "danger" if status == "error" else str(status or "")
    icon_html = f'<div class="pm-metric-icon">{html.escape(str(icon))}</div>' if icon else ""
    helper_html = f'<div class="pm-metric-helper">{html.escape(helper)}</div>' if helper else ""
    st.markdown(
        f"""
        <div class="pm-metric-card pm-metric-card-{html.escape(status_name)}">
          {icon_html}
          <div class="pm-metric-label">{html.escape(label)}</div>
          <div class="pm-metric-value">{html.escape(str(value))}</div>
          {helper_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_section_card(title: str, description: str | None = None) -> None:
    st.markdown(
        f"""
        <div class="pm-section-card">
          <h3 class="pm-section-title">{html.escape(title)}</h3>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_auth_page() -> None:
    """Render a simple centered login/register page."""
    st.markdown(
        f"""
        <div class="pm-auth-shell">
          <div>
            <div class="pm-auth-intro">
                <div class="pm-badges" style="justify-content:center;margin-bottom:14px;">
                {render_status_badge("数据私有", "primary")}
                {render_status_badge("可信引用", "info")}
              </div>
              <h1>PaperMate</h1>
              <p>论文阅读 RAG 助手</p>
            </div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    left, middle, right = st.columns([1, 0.92, 1])
    with middle:
        with st.container(border=True):
            st.markdown(
                """
                <h2 class="pm-auth-title">欢迎回来</h2>
                """,
                unsafe_allow_html=True,
            )
            login_tab, register_tab = st.tabs(["登录", "注册"])
            with login_tab:
                with st.form("login_form_tidy"):
                    username = st.text_input("用户名", key="login_username", placeholder="输入用户名")
                    password = st.text_input("密码", type="password", key="login_password", placeholder="输入密码")
                    submitted = st.form_submit_button("登录 PaperMate", type="primary", use_container_width=True)
                if submitted:
                    user = authenticate_user(username, password)
                    if not user:
                        render_error_card("登录失败", "请检查用户名或密码后再试。")
                    else:
                        prepare_user_workspace(int(user["user_id"]))
                        set_current_user(user)
                        st.toast("登录成功。")
                        st.rerun()
            with register_tab:
                with st.form("register_form_tidy"):
                    username = st.text_input("用户名", key="register_username", placeholder="3-32 位用户名")
                    password = st.text_input("密码", type="password", key="register_password", placeholder="至少 6 位")
                    password_confirm = st.text_input("确认密码", type="password", key="register_password_confirm")
                    submitted = st.form_submit_button("创建账户", type="primary", use_container_width=True)
                if submitted:
                    if password != password_confirm:
                        render_error_card("注册失败", "两次输入的密码不一致。")
                    else:
                        try:
                            user = create_user(username, password)
                            prepare_user_workspace(int(user["user_id"]))
                        except (ValueError, OSError, sqlite3.Error) as exc:
                            render_error_card("注册失败", str(exc) or "注册失败，请稍后再试。")
                        else:
                            set_current_user(user)
                            st.toast("账户已创建。")
                            st.rerun()






def job_badge_type(status: str) -> str:
    """Map job status to a badge type."""
    if status == "succeeded":
        return "success"
    if status == "failed":
        return "danger"
    if status == "running":
        return "warning"
    if status == "canceled":
        return "default"
    return "info"


def render_jobs_panel(team_id: int, paper_id: str | None = None) -> None:
    """Render recent background jobs."""
    can_edit = can_write(current_team_context()["role"])
    jobs = list_jobs(current_user_id(), team_id, paper_id=paper_id, limit=20)
    if not jobs:
        st.caption("暂无后台任务。")
        return
    for job in jobs:
        with st.expander(f"#{job['job_id']} · {job['job_type']} · {job['status']}", expanded=job["status"] in {"running", "failed"}):
            st.markdown(render_status_badge(str(job["status"]), job_badge_type(str(job["status"]))), unsafe_allow_html=True)
            st.write("论文：", job.get("file_name") or job.get("paper_id") or "未关联")
            st.write("创建人：", job.get("username") or job.get("user_id") or "未知")
            st.caption(f"创建：{job.get('created_at')} · 更新：{job.get('updated_at')} · 尝试：{job.get('attempt_count')}/{job.get('max_attempts')}")
            if job.get("error_message"):
                st.code(str(job["error_message"]), language="text")
            action_cols = st.columns(2)
            with action_cols[0]:
                if st.button("重试", key=f"retry_job_{job['job_id']}", disabled=not can_edit or job["status"] not in {"failed", "canceled"}, use_container_width=True):
                    try:
                        retry_job(current_user_id(), int(job["job_id"]))
                        st.rerun()
                    except Exception as exc:
                        render_error_card("重试任务失败", str(exc) or "请检查权限。")
            with action_cols[1]:
                if st.button("取消", key=f"cancel_job_{job['job_id']}", disabled=not can_edit or job["status"] not in {"queued", "running"}, use_container_width=True):
                    try:
                        cancel_job(current_user_id(), int(job["job_id"]))
                        st.rerun()
                    except Exception as exc:
                        render_error_card("取消任务失败", str(exc) or "请检查权限。")


def load_paper_into_workspace(paper: dict[str, Any], signature: str | None = None) -> dict[str, Any]:
    """Load a persisted paper into the current workspace session."""
    processed_pdf = paper_to_processed_pdf(paper)
    if signature:
        processed_pdf["signature"] = signature
    processed_pdf["paper_status_signature"] = paper_status_signature(paper)
    st.session_state["processed_pdf"] = processed_pdf
    return processed_pdf


def paper_status_signature(paper: dict[str, Any] | None) -> str:
    """Return the metadata signature that controls workspace reload decisions."""
    if not paper:
        return ""
    return "|".join(
        str(paper.get(key) or "")
        for key in (
            "paper_id",
            "updated_at",
            "parse_status",
            "index_status",
            "translation_status",
        )
    )


def should_reload_processed_pdf(processed_pdf: dict[str, Any] | None, paper: dict[str, Any] | None) -> bool:
    """Return whether paper metadata changes require reading chunks/Markdown again."""
    if not processed_pdf or not paper:
        return False
    current_paper_id = str((processed_pdf.get("saved_file") or {}).get("paper_id") or "")
    new_paper_id = str(paper.get("paper_id") or "")
    if current_paper_id != new_paper_id:
        return True

    parse_status = str(paper.get("parse_status") or "")
    current_parse_status_value = current_parse_status(processed_pdf)
    if parse_status == "succeeded" and current_parse_status_value != "succeeded":
        return True
    if parse_status == "succeeded" and not processed_pdf.get("chunks"):
        return True
    return False


def update_processed_pdf_metadata(
    processed_pdf: dict[str, Any],
    paper: dict[str, Any],
    signature: str | None = None,
) -> dict[str, Any]:
    """Update lightweight paper fields without re-reading chunks or Markdown."""
    saved_file = dict(processed_pdf.get("saved_file") or {})
    saved_file.update(
        {
            "file_name": paper.get("file_name") or saved_file.get("file_name") or "",
            "paper_id": paper.get("paper_id") or saved_file.get("paper_id") or "",
            "file_size_bytes": int(paper.get("file_size_bytes") or saved_file.get("file_size_bytes") or 0),
            "file_size": format_file_size(int(paper.get("file_size_bytes") or saved_file.get("file_size_bytes") or 0)),
            "save_path": paper.get("save_path") or saved_file.get("save_path") or "",
            "file_sha256": paper.get("file_sha256") or saved_file.get("file_sha256") or "",
        }
    )
    processed_pdf["saved_file"] = saved_file

    parsed_pdf = dict(processed_pdf.get("parsed_pdf") or {})
    parsed_pdf.update(
        {
            "paper_id": paper.get("paper_id") or parsed_pdf.get("paper_id"),
            "page_count": int(paper.get("page_count") or parsed_pdf.get("page_count") or 0),
            "parser": paper.get("parser") or parsed_pdf.get("parser") or "persisted",
            "markdown_path": paper.get("markdown_path") or parsed_pdf.get("markdown_path"),
            "translated_markdown_path": paper.get("translated_markdown_path") or parsed_pdf.get("translated_markdown_path"),
            "content_list_path": paper.get("content_list_path") or parsed_pdf.get("content_list_path"),
        }
    )
    processed_pdf["parsed_pdf"] = parsed_pdf
    processed_pdf["total_chars"] = int(paper.get("total_chars") or processed_pdf.get("total_chars") or 0)
    processed_pdf["team_id"] = paper.get("team_id")
    processed_pdf["project_id"] = paper.get("project_id")
    processed_pdf["parse_status"] = paper.get("parse_status") or "unknown"
    processed_pdf["index_status"] = paper.get("index_status") or "unknown"
    processed_pdf["translation_status"] = paper.get("translation_status") or "not_started"
    processed_pdf["paper_status_signature"] = paper_status_signature(paper)
    if signature:
        processed_pdf["signature"] = signature
    st.session_state["processed_pdf"] = processed_pdf
    return processed_pdf


def refresh_workspace_paper(processed_pdf: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any] | None]:
    """Refresh current workspace metadata, loading heavy paper content only when needed."""
    paper_id = str((processed_pdf.get("saved_file") or {}).get("paper_id") or "")
    if not paper_id:
        return processed_pdf, None
    paper = get_accessible_paper(paper_id, current_user_id())
    if not paper:
        return processed_pdf, None
    if should_reload_processed_pdf(processed_pdf, paper):
        return load_paper_into_workspace(paper, signature=processed_pdf.get("signature")), paper
    return update_processed_pdf_metadata(processed_pdf, paper, signature=processed_pdf.get("signature")), paper


def save_uploaded_pdf_to_library(
    uploaded_file: UploadedFile,
    team_id: int,
    project_id: int | None,
    signature: str | None = None,
) -> dict[str, Any]:
    """Save an uploaded PDF and enqueue parsing plus indexing jobs."""
    file_bytes = uploaded_file.getvalue()
    digest = file_sha256(file_bytes)
    existing_paper = find_team_paper_by_hash(
        team_id,
        digest,
        statuses=("succeeded", "running", "queued", "failed"),
    )
    if existing_paper:
        pipeline = enqueue_upload_processing_pipeline(existing_paper)
        refreshed_paper = get_accessible_paper(str(existing_paper["paper_id"]), current_user_id())
        return {
            "paper": refreshed_paper or existing_paper,
            "job_id": pipeline.get("parse_job_id"),
            "parse_job_id": pipeline.get("parse_job_id"),
            "index_job_id": pipeline.get("index_job_id"),
            "parse_reused": pipeline.get("parse_reused", False),
            "index_reused": pipeline.get("index_reused", False),
            "reused": True,
            "signature": signature,
            "message": upload_processing_message(True, pipeline),
        }

    saved_file = save_uploaded_pdf(uploaded_file)
    save_paper_and_chunks(
        {
            "paper_id": saved_file["paper_id"],
            "file_name": saved_file["file_name"],
            "file_size_bytes": saved_file["file_size_bytes"],
            "save_path": saved_file["save_path"],
            "owner_user_id": current_user_id(),
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
    paper = get_accessible_paper(saved_file["paper_id"], current_user_id())
    if not paper:
        raise RuntimeError("saved paper is not accessible after upload")
    pipeline = enqueue_upload_processing_pipeline(paper)
    paper = get_accessible_paper(saved_file["paper_id"], current_user_id()) or paper
    return {
        "paper": paper,
        "job_id": pipeline.get("parse_job_id"),
        "parse_job_id": pipeline.get("parse_job_id"),
        "index_job_id": pipeline.get("index_job_id"),
        "parse_reused": pipeline.get("parse_reused", False),
        "index_reused": pipeline.get("index_reused", False),
        "reused": False,
        "signature": signature,
        "message": upload_processing_message(False, pipeline),
    }


def enqueue_paper_parse(paper_id: str, include_images: bool = False) -> dict[str, Any]:
    """Queue parsing for an existing paper without automatically enqueueing an index build."""
    paper = get_accessible_paper(paper_id, current_user_id(), minimum_role="editor")
    if not paper:
        raise PermissionError("没有找到当前论文或无权解析。")

    latest_parse_job = latest_job_for_paper(int(paper["team_id"]), paper_id, "parse")
    if latest_parse_job and latest_parse_job.get("status") in {"queued", "running"}:
        return {
            "parse_job_id": int(latest_parse_job["job_id"]),
            "index_job_id": None,
            "reused": True,
            "message": f"解析任务已在队列中：#{latest_parse_job['job_id']}。",
        }

    update_paper_status(paper_id, parse_status="queued", index_status="unknown")
    parse_job_id = enqueue_job(
        "parse",
        user_id=current_user_id(),
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


def current_parse_status(processed_pdf: dict[str, Any] | None) -> str:
    """Return the current parse status for workflow and gating UI."""
    if not processed_pdf:
        return "not_started"
    status = str(processed_pdf.get("parse_status") or "").strip()
    if status:
        return status
    parsed_pdf = processed_pdf.get("parsed_pdf") or {}
    if processed_pdf.get("chunks") or parsed_pdf.get("markdown_path"):
        return "succeeded"
    return "unknown"


def render_parse_progress_panel(processed_pdf: dict[str, Any]) -> tuple[dict[str, Any], bool]:
    """Show the current paper in the workspace until parsing has completed."""
    paper_id = str(processed_pdf.get("saved_file", {}).get("paper_id") or "")
    if not paper_id:
        return processed_pdf, False

    try:
        paper = get_accessible_paper(paper_id, current_user_id())
    except Exception as exc:
        render_error_card("当前论文读取失败", "请检查团队权限和数据库状态。", str(exc))
        return processed_pdf, False

    if not paper:
        render_error_card("当前论文不可访问", "这篇论文不存在，或当前用户没有访问权限。")
        st.session_state.pop("processed_pdf", None)
        return processed_pdf, False

    if should_reload_processed_pdf(processed_pdf, paper):
        refreshed_pdf = load_paper_into_workspace(paper, signature=processed_pdf.get("signature"))
    else:
        refreshed_pdf = update_processed_pdf_metadata(processed_pdf, paper, signature=processed_pdf.get("signature"))
    parse_status = current_parse_status(refreshed_pdf)
    if parse_status == "succeeded" and not refreshed_pdf.get("chunks"):
        refreshed_pdf = load_paper_into_workspace(paper, signature=processed_pdf.get("signature"))
        parse_status = current_parse_status(refreshed_pdf)
    if parse_status == "succeeded":
        return refreshed_pdf, True

    saved_file = refreshed_pdf.get("saved_file", {})
    team_context = current_team_context()
    can_edit = can_write(team_context["role"])
    parse_busy = parse_status in {"queued", "running"}
    status_type = "danger" if parse_status == "failed" else "warning" if parse_status in {"queued", "running"} else "default"
    st.markdown(
        (
            '<div class="pm-panel">'
            '<h3 class="pm-section-title">当前论文已进入工作台</h3>'
            '<div class="pm-file-capsule">'
            '<div><strong>{file_name}</strong>'
            '<div class="pm-card-meta">{file_size} · paper_id: {paper_id}</div></div>'
            '<div class="pm-badges">{parse_badge}{index_badge}</div>'
            '</div>'
            '</div>'
        ).format(
            file_name=html.escape(str(saved_file.get("file_name") or "未命名论文")),
            file_size=html.escape(str(saved_file.get("file_size") or "")),
            paper_id=html.escape(paper_id),
            parse_badge=render_status_badge(f"解析 {parse_status}", status_type),
            index_badge=render_status_badge(f"索引 {refreshed_pdf.get('index_status') or 'unknown'}", "default"),
        ),
        unsafe_allow_html=True,
    )
    if not parse_busy:
        include_images = st.checkbox(
            "本次解析包含图片识别",
            value=False,
            key=f"pending_parse_include_images_{paper_id}",
            disabled=not can_edit,
        )
        if include_images:
            st.warning("如果图片数量较多可能需要数十分钟")
        if st.button(
            "解析当前论文" if not include_images else "添加图片并解析当前论文",
            type="primary",
            use_container_width=True,
            key=f"enqueue_parse_{paper_id}",
            disabled=not can_edit,
        ):
            try:
                result = enqueue_paper_parse(paper_id, include_images=include_images)
                st.session_state["pm_workspace_notice"] = result["message"]
                st.rerun()
            except Exception as exc:
                logger.exception("Parse job enqueue failed. paper_id=%s", paper_id)
                render_error_card("解析任务创建失败", "请检查团队权限、任务队列和 SQLite 写入状态。", str(exc))
        if not can_edit:
            st.caption("当前团队角色为只读，无法创建解析任务。")

    action_cols = st.columns([0.5, 0.5], gap="small")
    with action_cols[0]:
        if st.button("刷新解析结果", type="primary", use_container_width=True, key=f"refresh_parse_{paper_id}"):
            st.rerun()
    with action_cols[1]:
        if st.button("打开论文库", use_container_width=True, key=f"open_library_from_pending_{paper_id}"):
            navigate_to_page("📚 论文库")
    if parse_status == "failed":
        st.warning("解析任务失败。请检查右侧队列状态，修复配置或网络问题后重新提交任务。")
    elif parse_busy:
        st.info("请保持 worker 运行并耐心等待。解析成功后，如已有索引任务会继续排队执行；也可以手动构建索引。点击刷新即可查看最新状态。")
    else:
        st.info("当前论文还没有进入解析队列。点击上方按钮后，worker 会处理你选择的论文。")
    return refreshed_pdf, False


def dataframe_selected_rows(key: str) -> list[int]:
    """Read selected row indices from a Streamlit dataframe widget state."""
    state = st.session_state.get(key)
    if state is None:
        return []
    try:
        return [int(row) for row in state.selection.rows]
    except AttributeError:
        if isinstance(state, dict):
            return [int(row) for row in state.get("selection", {}).get("rows", [])]
    return []


@st.dialog("删除论文")
def render_delete_papers_dialog(team_id: int, paper_ids: list[str], file_names: list[str]) -> None:
    """Confirm and delete selected papers."""
    clean_ids = [str(paper_id) for paper_id in paper_ids if str(paper_id)]
    if not clean_ids:
        st.info("请先选择要删除的论文。")
        return

    count = len(clean_ids)
    dialog_key = hashlib.sha1("|".join(clean_ids).encode("utf-8")).hexdigest()[:12]
    st.warning("删除后会移出论文库，并清理对应 chunks、文献卡片、后台任务记录和 PDF/Markdown/BM25 文件。该操作不可恢复。")
    for name in file_names[:8]:
        st.write(f"- {name}")
    if len(file_names) > 8:
        st.caption(f"另有 {len(file_names) - 8} 篇未显示。")

    confirmed = st.checkbox(f"确认删除选中的 {count} 篇论文", key=f"confirm_delete_papers_{dialog_key}")
    action_cols = st.columns(2)
    with action_cols[0]:
        if st.button(
            "确认删除",
            type="primary",
            use_container_width=True,
            disabled=not confirmed,
            key=f"confirm_delete_button_{dialog_key}",
        ):
            try:
                result = delete_team_papers(current_user_id(), team_id, clean_ids, delete_files=True)
                current_paper_id = str((st.session_state.get("processed_pdf") or {}).get("saved_file", {}).get("paper_id") or "")
                if current_paper_id in clean_ids:
                    st.session_state.pop("processed_pdf", None)
                st.session_state.pop("paper_library_selected_paper_id", None)
                st.toast(f"已删除 {result['deleted']} 篇论文。")
                if result.get("file_errors"):
                    st.warning("部分文件清理失败或被跳过：\n" + "\n".join(result["file_errors"][:8]))
                st.rerun()
            except Exception as exc:
                logger.exception("Paper deletion failed.")
                render_error_card("删除失败", "请检查团队权限和 SQLite 写入权限。", str(exc))
    with action_cols[1]:
        if st.button("取消", use_container_width=True, key=f"cancel_delete_button_{dialog_key}"):
            st.rerun()


def render_paper_library_page() -> None:
    """Render the team-scoped paper library."""
    team_context = current_team_context()
    team_id = int(team_context["team_id"])
    can_edit = can_write(team_context["role"])
    render_app_header(
        "论文库",
        "查看当前团队可访问的所有论文，按项目和状态筛选，并打开历史论文继续阅读。",
        [
            render_status_badge(str(team_context["team"]["name"]), "primary"),
            render_status_badge(f"角色 {team_context['role']}", "info"),
        ],
    )
    library_notice = st.session_state.pop("pm_library_notice", None)
    if library_notice:
        st.success(str(library_notice))
    projects = team_context["projects"]
    project_options = [0, *[int(project["project_id"]) for project in projects]]
    filter_cols = st.columns([0.24, 0.20, 0.20, 0.36], gap="small")
    with filter_cols[0]:
        project_filter = st.selectbox(
            "项目",
            project_options,
            format_func=lambda value: "全部项目" if int(value) == 0 else next(project["name"] for project in projects if int(project["project_id"]) == int(value)),
        )
    with filter_cols[1]:
        parse_filter = st.selectbox("解析状态", ["全部", "not_started", "queued", "running", "succeeded", "failed"])
    with filter_cols[2]:
        index_filter = st.selectbox("索引状态", ["全部", "unknown", "running", "succeeded", "failed", "partial"])
    with filter_cols[3]:
        search_query = st.text_input("搜索论文", placeholder="文件名或 paper_id")

    if can_edit and projects:
        with st.expander("添加论文到论文库", expanded=False):
            upload_project_id = st.selectbox(
                "保存到项目",
                options=[int(project["project_id"]) for project in projects],
                index=max(
                    0,
                    next(
                        (
                            index
                            for index, project in enumerate(projects)
                            if int(project["project_id"]) == int(team_context["project_id"])
                        ),
                        0,
                    ),
                ),
                format_func=lambda value: next(project["name"] for project in projects if int(project["project_id"]) == int(value)),
                key="paper_library_upload_project_id",
            )
            library_uploads = st.file_uploader(
                "批量选择 PDF",
                type=["pdf"],
                accept_multiple_files=True,
                key="paper_library_batch_uploads",
            )
            if st.button(
                "添加到论文库",
                type="primary",
                use_container_width=True,
                disabled=not library_uploads,
                key="paper_library_batch_upload_button",
            ):
                created = 0
                reused = 0
                errors: list[str] = []
                with st.spinner("正在保存 PDF 并创建解析与索引任务..."):
                    for upload in library_uploads or []:
                        try:
                            result = save_uploaded_pdf_to_library(upload, team_id, int(upload_project_id))
                            if result.get("reused"):
                                reused += 1
                            else:
                                created += 1
                        except Exception as exc:
                            logger.exception("Paper library batch upload failed.")
                            errors.append(f"{upload.name}: {exc}")
                if errors:
                    render_error_card("部分论文添加失败", "请检查文件、权限、SQLite 状态或解析配置。", "\n".join(errors[:8]))
                else:
                    st.success(f"已添加 {created} 篇论文；复用已有论文 {reused} 篇。上传后已自动加入解析和索引队列。")
                    st.rerun()
    elif can_edit:
        st.info("当前团队还没有项目，请先在团队管理中创建项目后再批量添加论文。")
    else:
        st.info("当前角色为 viewer，只能查看论文库，不能批量添加或删除。")

    papers = list_accessible_papers(
        current_user_id(),
        team_id=team_id,
        project_id=None if int(project_filter) == 0 else int(project_filter),
        parse_status=None if parse_filter == "全部" else parse_filter,
        index_status=None if index_filter == "全部" else index_filter,
        search_query=search_query,
    )
    metric_cols = st.columns(4)
    with metric_cols[0]:
        render_metric_card("论文数", len(papers), "当前筛选结果", icon="P")
    with metric_cols[1]:
        render_metric_card("已解析", sum(1 for paper in papers if paper.get("parse_status") == "succeeded"), "可打开阅读", icon="MD")
    with metric_cols[2]:
        render_metric_card("已索引", sum(1 for paper in papers if paper.get("index_status") == "succeeded"), "可进行问答", icon="R")
    with metric_cols[3]:
        render_metric_card("任务中", sum(1 for paper in papers if paper.get("parse_status") in {"queued", "running"} or paper.get("index_status") in {"queued", "running"}), "后台处理", icon="J")

    if not papers:
        render_empty_state("当前没有论文", "在论文工作台上传 PDF 后，论文会显示在这里，并自动进入解析和索引队列。", "去论文工作台", icon="PDF")
        if st.button("去论文工作台", type="primary", use_container_width=True):
            navigate_to_page("📄 论文工作台")
        return

    table_rows = [
        {
            "文件名": paper["file_name"],
            "PDF": "点击行预览",
            "项目": paper.get("project_name") or "未分组",
            "上传人": paper.get("owner_username") or "未知",
            "解析": paper.get("parse_status"),
            "索引": paper.get("index_status"),
            "翻译": paper.get("translation_status"),
            "Chunks": paper.get("chunk_count"),
            "更新": paper.get("updated_at"),
        }
        for paper in papers
    ]
    toolbar_cols = st.columns([0.24, 0.52, 0.24], gap="small")
    with toolbar_cols[0]:
        multi_select_enabled = st.toggle(
            "多选",
            value=False,
            key="paper_library_multi_select",
            disabled=not can_edit,
            help="开启后可以一次选择多篇论文；关闭时为单篇选择。",
        )
    table_key = "paper_library_table_multi" if multi_select_enabled else "paper_library_table_single"
    selection_mode = "multi-row" if multi_select_enabled else "single-row"
    selected_rows_before = dataframe_selected_rows(table_key)
    selected_paper_ids_before = [
        str(papers[row_index]["paper_id"])
        for row_index in selected_rows_before
        if 0 <= row_index < len(papers)
    ]
    selected_file_names_before = [
        str(papers[row_index].get("file_name") or papers[row_index]["paper_id"])
        for row_index in selected_rows_before
        if 0 <= row_index < len(papers)
    ]
    with toolbar_cols[1]:
        selected_count = len(selected_paper_ids_before)
        st.caption(
            f"已选 {selected_count} 篇；点击行可预览 PDF，并手动选择解析或索引任务。"
            if selected_count
            else "点击行可预览 PDF；开启多选后可批量选择论文。"
        )
    with toolbar_cols[2]:
        if st.button(
            "删除选中论文",
            type="primary",
            use_container_width=True,
            disabled=not can_edit or not selected_paper_ids_before,
            key="delete_selected_papers_toolbar",
        ):
            render_delete_papers_dialog(team_id, selected_paper_ids_before, selected_file_names_before)

    table_state = st.dataframe(
        table_rows,
        use_container_width=True,
        hide_index=True,
        key=table_key,
        on_select="rerun",
        selection_mode=selection_mode,
        column_config={
            "PDF": st.column_config.TextColumn("PDF", help="点击论文行后，下方会预览第一篇选中论文的原始 PDF。"),
        },
    )
    selected_rows = []
    try:
        selected_rows = [int(row) for row in table_state.selection.rows]
    except AttributeError:
        if isinstance(table_state, dict):
            selected_rows = [int(row) for row in table_state.get("selection", {}).get("rows", [])]

    selected_paper: dict[str, Any] | None = None
    selected_paper_ids = [
        str(papers[int(row_index)]["paper_id"])
        for row_index in selected_rows
        if 0 <= int(row_index) < len(papers)
    ]
    if selected_rows:
        row_index = int(selected_rows[0])
        if 0 <= row_index < len(papers):
            selected_paper = papers[row_index]
            st.session_state["paper_library_selected_paper_id"] = str(selected_paper["paper_id"])
    else:
        remembered_id = st.session_state.get("paper_library_selected_paper_id")
        if remembered_id:
            selected_paper = next((paper for paper in papers if str(paper["paper_id"]) == str(remembered_id)), None)

    if selected_paper is None:
        st.info("点击表格中的论文行，可直接在下方预览原始 PDF；owner/admin/editor 可开启多选后批量删除。解析完成后也可以打开到论文工作台继续阅读和问答。")
        if st.button("刷新论文库", use_container_width=True):
            st.rerun()
        return

    selected_paper_id = str(selected_paper["paper_id"])
    st.caption(
        f"当前选择：{selected_paper.get('file_name')} · {selected_paper.get('parse_status')} · {selected_paper.get('project_name') or '未分组'}"
        + (f" · 已选 {len(selected_paper_ids)} 篇" if selected_paper_ids else "")
    )
    open_cols = st.columns([0.45, 0.55], gap="small")
    with open_cols[0]:
        if st.button("打开到论文工作台", type="primary", use_container_width=True):
            if selected_paper.get("parse_status") != "succeeded":
                st.warning("这篇论文还没有解析完成，请等待 worker 完成解析任务。")
            else:
                fresh_paper = get_accessible_paper(selected_paper_id, current_user_id())
                if not fresh_paper:
                    render_error_card("无法打开论文", "当前用户没有访问这篇论文的权限。")
                    return
                load_paper_into_workspace(fresh_paper)
                navigate_to_page("📄 论文工作台")
    with open_cols[1]:
        if st.button("刷新论文库", use_container_width=True):
            st.rerun()
    if can_edit:
        st.markdown("#### 手动任务")
        task_cols = st.columns([0.34, 0.33, 0.33], gap="small")
        parse_status = str(selected_paper.get("parse_status") or "")
        index_status = str(selected_paper.get("index_status") or "")
        with task_cols[0]:
            include_images = st.checkbox(
                "包含图片识别",
                value=False,
                key=f"library_parse_include_images_{selected_paper_id}",
                disabled=parse_status in {"queued", "running"},
            )
        if include_images:
            st.warning("如果图片数量较多可能需要数十分钟")
        with task_cols[1]:
            if st.button(
                "解析选中论文" if not include_images else "添加图片并解析",
                type="primary",
                use_container_width=True,
                disabled=parse_status in {"queued", "running"},
                key=f"library_enqueue_parse_{selected_paper_id}",
            ):
                try:
                    result = enqueue_paper_parse(selected_paper_id, include_images=include_images)
                    st.success(result["message"])
                    st.rerun()
                except Exception as exc:
                    logger.exception("Library parse enqueue failed. paper_id=%s", selected_paper_id)
                    render_error_card("解析任务创建失败", "请检查团队权限、任务队列和 SQLite 写入状态。", str(exc))
        with task_cols[2]:
            if st.button(
                "构建选中论文索引",
                use_container_width=True,
                disabled=parse_status != "succeeded" or index_status in {"queued", "running"},
                key=f"library_enqueue_index_{selected_paper_id}",
            ):
                try:
                    job_id, reused_existing = enqueue_index_build_for_paper(selected_paper_id)
                    if reused_existing:
                        st.info(f"索引任务已在队列中：#{job_id}。")
                    else:
                        st.success(f"索引任务已入队：#{job_id}。")
                    st.session_state["pm_library_notice"] = (
                        f"索引任务已在队列中：#{job_id}。"
                        if reused_existing
                        else f"索引任务已入队：#{job_id}。"
                    )
                    st.rerun()
                except Exception as exc:
                    logger.exception("Library index enqueue failed. paper_id=%s", selected_paper_id)
                    render_error_card("索引任务创建失败", "请检查解析状态、团队权限和数据库状态。", str(exc))
    st.markdown("#### 原始 PDF")
    render_pdf_viewer(selected_paper.get("save_path"))


def render_team_management_page() -> None:
    """Render team and project administration."""
    team_context = current_team_context()
    team_id = int(team_context["team_id"])
    role = team_context["role"]
    render_app_header(
        "团队管理",
        "管理团队、项目和成员角色。owner/admin 可添加成员和创建项目。",
        [render_status_badge(f"当前角色 {role}", "warning" if can_manage_team(role) else "default")],
    )
    if not can_manage_team(role):
        render_empty_state("没有团队管理权限", "只有 owner 和 admin 可以管理团队成员和项目。", icon="TEAM")
        return

    section_cols = st.columns([0.48, 0.52], gap="large")
    with section_cols[0]:
        render_section_card("团队", "创建新团队，或在侧边栏切换当前团队。")
        with st.form("create_team_form"):
            team_name = st.text_input("新团队名称", placeholder="例如：视觉组、毕业论文小组")
            submitted = st.form_submit_button("创建团队", type="primary", use_container_width=True)
        if submitted:
            try:
                team = create_team(current_user_id(), team_name)
                st.session_state["pending_current_team_id"] = int(team["team_id"])
                st.session_state.pop("processed_pdf", None)
                st.success("团队已创建。")
                st.rerun()
            except Exception as exc:
                render_error_card("团队创建失败", str(exc) or "请检查输入。")

        render_section_card("项目", "项目用于把同一团队下的论文分组。")
        projects = list_projects(current_user_id(), team_id)
        st.dataframe([{"项目 ID": project["project_id"], "名称": project["name"], "创建时间": project["created_at"]} for project in projects], use_container_width=True, hide_index=True)
        with st.form("create_project_form"):
            project_name = st.text_input("新项目名称", placeholder="例如：RAG 必读、论文综述")
            submitted = st.form_submit_button("创建项目", type="primary", use_container_width=True)
        if submitted:
            try:
                create_project(current_user_id(), team_id, project_name)
                st.success("项目已创建。")
                st.rerun()
            except Exception as exc:
                render_error_card("项目创建失败", str(exc) or "请检查输入。")

    with section_cols[1]:
        render_section_card("成员", "按已有用户名添加成员，并设置 viewer/editor/admin 角色。")
        members = list_team_members(current_user_id(), team_id)
        st.dataframe([{"用户 ID": member["user_id"], "用户名": member["username"], "角色": member["role"], "加入时间": member["created_at"]} for member in members], use_container_width=True, hide_index=True)
        with st.form("add_team_member_form"):
            username = st.text_input("已有用户名")
            role_value = st.selectbox("角色", ["viewer", "editor", "admin"])
            submitted = st.form_submit_button("添加或更新成员", type="primary", use_container_width=True)
        if submitted:
            try:
                add_team_member_by_username(current_user_id(), team_id, username, role_value)
                st.success("团队成员已更新。")
                st.rerun()
            except Exception as exc:
                render_error_card("成员添加失败", str(exc) or "请检查用户名。")

        editable_members = [member for member in members if member["role"] != "owner" and int(member["user_id"]) != current_user_id()]
        if editable_members:
            with st.expander("修改或移除成员", expanded=False):
                target_user_id = st.selectbox(
                    "成员",
                    options=[int(member["user_id"]) for member in editable_members],
                    format_func=lambda value: next(member["username"] for member in editable_members if int(member["user_id"]) == int(value)),
                )
                new_role = st.selectbox("新角色", ["viewer", "editor", "admin"], key="team_member_new_role")
                action_cols = st.columns(2)
                with action_cols[0]:
                    if st.button("保存角色", use_container_width=True):
                        update_team_member_role(current_user_id(), team_id, int(target_user_id), new_role)
                        st.rerun()
                with action_cols[1]:
                    if st.button("移除成员", use_container_width=True):
                        remove_team_member(current_user_id(), team_id, int(target_user_id))
                        st.rerun()


def render_workspace_page() -> None:
    """Render a tidy paper workspace page."""
    team_context = current_team_context()
    team_id = int(team_context["team_id"])
    project_id = team_context.get("project_id")
    can_edit = can_write(team_context["role"])
    processed_pdf: dict[str, Any] | None = st.session_state.get("processed_pdf")
    paper_id = str((processed_pdf or {}).get("saved_file", {}).get("paper_id") or "")
    if processed_pdf and paper_id:
        try:
            processed_pdf, latest_paper = refresh_workspace_paper(processed_pdf)
            if latest_paper:
                paper_id = str(processed_pdf.get("saved_file", {}).get("paper_id") or "")
        except Exception:
            logger.exception("Workspace paper refresh failed. paper_id=%s", paper_id)
    parse_status = current_parse_status(processed_pdf)
    paper_open = bool(processed_pdf)
    parse_done = parse_status == "succeeded"
    index_state = local_index_state(paper_id) if paper_id else {"vector": "未知", "bm25": "未知"}
    index_ready = index_state["vector"] == "已构建" or index_state["bm25"] == "已构建"
    qa_done = bool(paper_id and st.session_state.get(f"last_qa_{paper_id}"))
    card_done = bool(paper_id and st.session_state.get(f"saved_card_id_{paper_id}"))

    render_app_header(
        "论文工作台",
        "上传 PDF 后自动进入解析和索引队列，并基于可信引用追问论文。",
        [
            render_status_badge("论文工作台", "success"),
            render_status_badge("可信引用", "info"),
            render_status_badge("知识沉淀", "primary"),
        ],
    )
    render_workflow_steps(
        [
            {"title": "上传 PDF", "helper": "选择论文文件", "status": "done" if paper_open else "active"},
            {"title": "解析正文", "helper": "Markdown 与表格", "status": "done" if parse_done else ("active" if paper_open else "pending")},
            {"title": "构建索引", "helper": "Chroma + BM25", "status": "done" if index_ready else ("active" if parse_done else "pending")},
            {"title": "开始问答", "helper": "基于原文引用", "status": "done" if qa_done else ("active" if index_ready else "pending")},
            {"title": "生成卡片", "helper": "沉淀研究笔记", "status": "done" if card_done else ("active" if qa_done else "pending")},
        ]
    )

    saved_file = processed_pdf.get("saved_file", {}) if processed_pdf else {}
    chunks = processed_pdf.get("chunks", []) if processed_pdf else []
    metric_cols = st.columns(5)
    with metric_cols[0]:
        render_metric_card("当前论文", saved_file.get("file_name") or "未上传", "上传 PDF 后自动排队", icon="PDF")
    with metric_cols[1]:
        render_metric_card("解析状态", parse_status, "正文、表格与页码", icon="MD", status="success" if parse_done else "warning")
    with metric_cols[2]:
        render_metric_card("Chunk 数量", len(chunks), "用于检索的论文片段", icon="CH")
    with metric_cols[3]:
        render_metric_card("向量索引", index_state["vector"], "Chroma 语义检索", icon="V", status=index_status_type(index_state["vector"]))
    with metric_cols[4]:
        render_metric_card("BM25 状态", index_state["bm25"], "关键词精确检索", icon="B", status=index_status_type(index_state["bm25"]))

    render_section_card("上传与解析", "拖拽或选择一篇 PDF。论文会立即进入论文库，并自动加入解析和索引队列；worker 完成后可继续阅读、问答和生成卡片。")
    workspace_notice = st.session_state.pop("pm_workspace_notice", None)
    if workspace_notice:
        st.success(str(workspace_notice))
    if not can_edit:
        st.info("当前团队角色为 viewer，只能阅读论文库中的已有论文，不能上传或创建任务。")
        uploaded_file = None
    else:
        uploaded_file = st.file_uploader("选择一篇 PDF 论文", type=["pdf"], accept_multiple_files=False)

    if uploaded_file is None:
        if not processed_pdf:
            render_empty_state("还没有打开论文", "从论文库打开历史论文，或上传 PDF 自动创建解析和索引任务。", "去论文库", icon="PDF")
            if st.button("打开论文库", type="primary", use_container_width=True):
                navigate_to_page("📚 论文库")
            return
        st.info("已恢复当前会话中的论文工作台内容。需要换论文时，重新上传 PDF 即可。")
    else:
        signature = get_uploaded_file_signature(uploaded_file)
        cached_pdf = processed_pdf if processed_pdf and processed_pdf.get("signature") == signature else None
        st.markdown(
            f"""
            <div class="pm-file-capsule">
              <div>
                <strong>{html.escape(uploaded_file.name)}</strong>
                <div class="pm-card-meta">{format_file_size(len(uploaded_file.getvalue()))} · 已选择，将保存并自动排队</div>
              </div>
              <div class="pm-badges">{render_status_badge("PDF", "primary")}{render_status_badge("已保存", "success")}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        if cached_pdf:
            processed_pdf = cached_pdf
            if st.button("重新保存当前 PDF", use_container_width=True):
                st.session_state.pop("processed_pdf", None)
                st.session_state.pop("pm_failed_upload_signature", None)
                st.session_state.pop("pm_failed_upload_error", None)
                st.rerun()
        else:
            processed_pdf = None
            failed_signature = st.session_state.get("pm_failed_upload_signature")
            if failed_signature == signature:
                render_error_card(
                    "PDF 保存失败",
                    "上一次保存论文失败。请修复问题后点击重试，系统会重新保存 PDF 并自动创建解析和索引任务。",
                    str(st.session_state.get("pm_failed_upload_error") or ""),
                )
                if st.button("重试保存论文", type="primary", use_container_width=True, key="retry_auto_parse"):
                    st.session_state.pop("pm_failed_upload_signature", None)
                    st.session_state.pop("pm_failed_upload_error", None)
                    st.rerun()
            else:
                try:
                    with st.spinner("正在保存 PDF 并创建解析与索引任务..."):
                        result = save_uploaded_pdf_to_library(uploaded_file, team_id, project_id, signature=signature)
                        paper = result.get("paper")
                        if paper:
                            load_paper_into_workspace(paper, signature=signature)
                        st.session_state.pop("pm_failed_upload_signature", None)
                        st.session_state.pop("pm_failed_upload_error", None)
                        st.session_state["pm_workspace_notice"] = (
                            result["message"]
                        )
                        st.rerun()
                except (UploadError, PdfParseError, MinerUError) as exc:
                    logger.exception("PDF upload or parse failed.")
                    st.session_state["pm_failed_upload_signature"] = signature
                    st.session_state["pm_failed_upload_error"] = f"{exc.message} 错误码：{exc.code.value}"
                    render_error_card(
                        "PDF 保存失败",
                        f"{exc.message} 请检查文件、解析服务配置或网络连接后重试。",
                        f"错误码：{exc.code.value}",
                    )
                    processed_pdf = None
                except Exception as exc:
                    logger.exception("PDF library save failed.")
                    st.session_state["pm_failed_upload_signature"] = signature
                    st.session_state["pm_failed_upload_error"] = str(exc)
                    render_error_card("PDF 保存失败", "请检查团队权限和 SQLite 写入权限。", str(exc))
                    processed_pdf = None

    if not processed_pdf:
        return
    processed_pdf, parse_ready = render_parse_progress_panel(processed_pdf)
    if not parse_ready:
        return
    if processed_pdf.get("db_save_failed"):
        render_error_card("数据保存失败", "解析结果可以继续查看，但 RAG 和文献卡片可能无法使用。请检查 SQLite 数据库权限。")

    render_processed_pdf_summary(processed_pdf)
    render_index_builder(processed_pdf["chunks"])

    st.divider()
    left_col, right_col = st.columns([0.60, 0.40], gap="large")
    with left_col:
        render_markdown_document(processed_pdf)
    with right_col:
        render_qa_box(processed_pdf["saved_file"]["paper_id"])
        render_literature_card_save(
            processed_pdf["saved_file"]["paper_id"],
            processed_pdf["chunks"],
            processed_pdf["db_save_failed"],
            current_user_id(),
        )


def render_literature_card(card: dict[str, Any], selected: bool = False) -> None:
    title = _pm_text(card.get("title"), "未命名论文")
    authors = _pm_text(card.get("authors"), "作者未识别")
    year = _pm_text(card.get("year"), "年份未知")
    library_name = _pm_text(card.get("library_name"), "未分组")
    question = _pm_compact(card.get("research_question"), 150)
    method = _pm_compact(card.get("method_summary"), 135)
    updated_at = _pm_text(card.get("updated_at"), "暂无更新时间")
    chips = "".join(f'<span class="pm-chip">{html.escape(chip)}</span>' for chip in _pm_method_chips(card, limit=3))
    selected_class = " pm-literature-card-selected" if selected else ""
    st.markdown(
        f"""
        <div class="pm-literature-card{selected_class}">
          <div class="pm-card-title">{html.escape(title)}</div>
          <div class="pm-card-meta">{html.escape(authors)} · {html.escape(year)}</div>
          <div class="pm-chip-row">{chips}</div>
          <div class="pm-card-summary"><strong>研究问题</strong><br>{html.escape(question)}</div>
          <div class="pm-card-summary"><strong>核心方法</strong><br>{html.escape(method)}</div>
          <div class="pm-card-footer">
            <span>{html.escape(library_name)}</span>
            <span>{html.escape(updated_at)}</span>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )




def render_card_visual(card: dict[str, Any]) -> None:
    render_literature_card(card)


def render_card_library_page(user_id: int) -> None:
    """Render a tidy literature-card knowledge base."""
    team_context = current_team_context()
    team_id = int(team_context["team_id"])
    can_edit = can_write(team_context["role"])
    render_app_header(
        "文献卡片库",
        "沉淀论文阅读记录、方法线索、实验结论和研究灵感。",
        [
            render_status_badge("已保存", "success"),
            render_status_badge("Markdown 可编辑", "info"),
        ],
    )
    try:
        libraries = list_card_libraries(user_id, team_id=team_id)
        all_cards = list_literature_cards(team_id=team_id)
    except (OSError, sqlite3.Error) as exc:
        render_error_card("卡片库读取失败", "请检查 SQLite 数据库权限后重试。", str(exc))
        return

    library_options = [0, *[int(library["library_id"]) for library in libraries]]
    selected_library_id = st.session_state.get("library_filter_tidy", 0)
    if int(selected_library_id) not in library_options:
        selected_library_id = 0
    selected_library = (
        next((library for library in libraries if int(library["library_id"]) == int(selected_library_id)), None)
        if int(selected_library_id) != 0
        else None
    )
    current_library_name = selected_library["name"] if selected_library else "全部卡片库"
    library_filter = None if int(selected_library_id) == 0 else int(selected_library_id)
    cards = list_literature_cards(library_id=library_filter, team_id=team_id)

    metric_cols = st.columns(5)
    with metric_cols[0]:
        render_metric_card("卡片总数", len(all_cards), f"{len(libraries)} 个卡片库", icon="C")
    with metric_cols[1]:
        render_metric_card("当前库卡片数", len(cards), current_library_name, icon="L")
    with metric_cols[2]:
        render_metric_card("最近更新", max((str(card.get("updated_at") or "") for card in all_cards), default="暂无"), "按更新时间统计", icon="U")
    with metric_cols[3]:
        render_metric_card("关联 PDF", len({card.get("paper_id") for card in all_cards if card.get("paper_id")}), "按 paper_id 统计", icon="P")
    with metric_cols[4]:
        render_metric_card("本周新增", _pm_recent_week_count(all_cards), "最近 7 天更新", icon="+")

    render_section_card("筛选与管理", "先选择卡片库和筛选条件，再在左侧选择卡片，右侧查看详情。")
    top_cols = st.columns([0.28, 0.24, 0.18, 0.18, 0.12], gap="small")
    with top_cols[0]:
        selected_library_id = st.selectbox(
            "卡片库",
            options=library_options,
            index=library_options.index(int(selected_library_id)),
            format_func=lambda library_id: "全部卡片库" if int(library_id) == 0 else library_option_label(
                next(library for library in libraries if int(library["library_id"]) == int(library_id))
            ),
            key="library_filter_tidy",
        )
    library_filter = None if int(selected_library_id) == 0 else int(selected_library_id)
    cards = list_literature_cards(library_id=library_filter, team_id=team_id)
    field_options = ["全部领域", *sorted({_pm_text(card.get("research_field"), "") for card in cards if card.get("research_field")})]
    year_options = ["全部年份", *sorted({_pm_text(card.get("year"), "") for card in cards if card.get("year")}, reverse=True)]
    with top_cols[1]:
        search_query = st.text_input("搜索", placeholder="标题、方法、贡献、关键词")
    with top_cols[2]:
        field_filter = st.selectbox("领域", field_options)
    with top_cols[3]:
        year_filter = st.selectbox("年份", year_options)
    with top_cols[4]:
        sort_mode = st.selectbox("排序", ["最近", "标题", "年份"])

    action_cols = st.columns([0.34, 0.34, 0.32], gap="large")
    with action_cols[0]:
        if can_edit:
            with st.expander("新建卡片库", expanded=False):
                render_library_create_form(user_id, key_suffix="_tidy", team_id=team_id)
        else:
            st.caption("只读角色不能新建卡片库。")
    with action_cols[1]:
        selected_library = (
            next((library for library in libraries if int(library["library_id"]) == int(selected_library_id)), None)
            if int(selected_library_id) != 0
            else None
        )
        if selected_library:
            if can_edit:
                with st.expander("重命名当前库", expanded=False):
                    render_library_rename_form(user_id, selected_library, team_id=team_id)
            else:
                st.caption("只读角色不能重命名卡片库。")
        else:
            st.caption("选择具体卡片库后可重命名。")
    with action_cols[2]:
        with st.expander("批量删除", expanded=False):
            selected_batch_ids = st.multiselect(
                "选择要删除的卡片",
                options=[int(card["card_id"]) for card in cards],
                format_func=lambda card_id: card_option_label(
                    next(card for card in cards if int(card["card_id"]) == int(card_id))
                ),
            )
            confirm_batch_delete = st.checkbox("确认删除，操作不可撤销")
            if st.button("确认批量删除", disabled=not can_edit or not selected_batch_ids or not confirm_batch_delete, use_container_width=True):
                try:
                    deleted_count = delete_literature_cards(selected_batch_ids, user_id=None, team_id=team_id)
                except (OSError, sqlite3.Error) as exc:
                    render_error_card("批量删除失败", "请检查 SQLite 数据库权限。", str(exc))
                    return
                st.success(f"已删除 {deleted_count} 张文献卡片。")
                st.rerun()

    if field_filter != "全部领域":
        cards = [card for card in cards if _pm_text(card.get("research_field"), "") == field_filter]
    if year_filter != "全部年份":
        cards = [card for card in cards if _pm_text(card.get("year"), "") == year_filter]
    if search_query.strip():
        needle = search_query.strip().lower()
        cards = [
            card for card in cards
            if needle in " ".join(str(card.get(key) or "") for key in (
                "title", "authors", "year", "research_field", "research_question",
                "method_summary", "datasets", "library_name", "file_name", "markdown"
            )).lower()
        ]
    if sort_mode == "标题":
        cards = sorted(cards, key=lambda card: str(card.get("title") or ""))
    elif sort_mode == "年份":
        cards = sorted(cards, key=lambda card: str(card.get("year") or ""), reverse=True)

    if not cards:
        render_empty_state(
            "还没有文献卡片",
            "在论文工作台中生成第一张文献卡片，把阅读成果沉淀成你的研究知识库。",
            "去论文工作台",
            icon="CARD",
        )
        if st.button("去论文工作台", type="primary", use_container_width=True):
            navigate_to_page("📄 论文工作台")
        return

    if "selected_literature_card_id_tidy" not in st.session_state or not any(
        int(card["card_id"]) == int(st.session_state["selected_literature_card_id_tidy"]) for card in cards
    ):
        st.session_state["selected_literature_card_id_tidy"] = int(cards[0]["card_id"])

    selected_card_id = st.selectbox(
        "当前查看的文献卡片",
        options=[int(card["card_id"]) for card in cards],
        index=[int(card["card_id"]) for card in cards].index(int(st.session_state["selected_literature_card_id_tidy"])),
        format_func=lambda card_id: card_option_label(next(card for card in cards if int(card["card_id"]) == int(card_id))),
    )
    st.session_state["selected_literature_card_id_tidy"] = int(selected_card_id)

    list_col, detail_col = st.columns([0.42, 0.58], gap="large")
    with list_col:
        st.caption(f"当前显示 {len(cards)} 张卡片")
        for card in cards:
            render_literature_card(card, selected=int(card["card_id"]) == int(selected_card_id))
    with detail_col:
        selected_card = get_literature_card(int(selected_card_id), team_id=team_id)
        if not selected_card:
            render_empty_state("没有找到选中的卡片", "请重新选择一张文献卡片。", icon="CARD")
            return
        tab_card, tab_edit, tab_pdf = st.tabs(["详情", "编辑", "原 PDF"])
        with tab_card:
            render_literature_detail(selected_card, mode="preview")
            with st.expander("Markdown 原文", expanded=False):
                st.text_area("Markdown", selected_card["markdown"], height=320, key=f"card_markdown_tidy_{selected_card['card_id']}")
        with tab_edit:
            if can_edit:
                render_literature_detail(selected_card, mode="edit")
                render_card_edit_form(selected_card, user_id=None, team_id=team_id)
                st.divider()
                render_card_delete(int(selected_card["card_id"]), user_id=None, team_id=team_id)
            else:
                render_empty_state("只读角色不能编辑卡片", "当前团队角色为 viewer，只能查看文献卡片。", icon="CARD")
        with tab_pdf:
            render_pdf_viewer(selected_card.get("save_path"))


def render_workflow_steps(steps: list[dict[str, str]]) -> None:
    """Render workflow steps without Markdown treating HTML as code."""
    items: list[str] = []
    for index, step in enumerate(steps, start=1):
        status = html.escape(str(step.get("status") or "pending"))
        title = html.escape(str(step.get("title") or ""))
        helper = html.escape(str(step.get("helper") or ""))
        items.append(
            '<div class="pm-step pm-step-{status}">'
            '<div class="pm-step-dot">{index}</div>'
            '<div><div class="pm-step-title">{title}</div>'
            '<div class="pm-step-helper">{helper}</div></div>'
            '</div>'.format(status=status, index=index, title=title, helper=helper)
        )
    st.markdown(
        '<div class="pm-workflow">{items}</div>'.format(items="".join(items)),
        unsafe_allow_html=True,
    )


def render_literature_detail(card: dict[str, Any], mode: str = "preview") -> None:
    """Render literature-card detail without exposing raw HTML."""
    markdown = str(card.get("markdown") or "")
    title = _pm_text(card.get("title"), "未命名论文")
    authors = _pm_text(card.get("authors"), "作者未识别")
    year = _pm_text(card.get("year"), "年份未知")
    library_name = _pm_text(card.get("library_name"), "未分组")
    updated_at = _pm_text(card.get("updated_at"), "暂无更新时间")
    editing_badge = render_status_badge("正在编辑", "warning") if mode == "edit" else ""
    sections = [
        ("研究问题", card.get("research_question")),
        ("核心方法", card.get("method_summary")),
        ("主要贡献", _pm_markdown_section(markdown, ("主要贡献", "贡献摘要", "论文贡献"))),
        ("实验结论", card.get("datasets")),
        ("局限性", _pm_markdown_section(markdown, ("局限性", "局限", "不足"))),
        ("我的笔记", _pm_markdown_section(markdown, ("我的笔记", "个人笔记", "复习线索"))),
    ]
    section_html = "".join(
        '<div class="pm-detail-section"><strong>{label}</strong><div>{value}</div></div>'.format(
            label=html.escape(label),
            value=html.escape(_pm_text(value, "原文未明确说明")),
        )
        for label, value in sections
    )
    badges = "".join(
        [
            render_status_badge(year, "primary"),
            render_status_badge(library_name, "default"),
            render_status_badge("Markdown 可编辑", "info"),
            editing_badge,
        ]
    )
    detail_html = (
        '<div class="pm-detail-panel">'
        '<h2 class="pm-detail-title">{title}</h2>'
        '<div class="pm-badges" style="margin-bottom:10px;">{badges}</div>'
        '<div class="pm-card-meta">{authors} · 更新于 {updated_at}</div>'
        '{sections}'
        '</div>'
    ).format(
        title=html.escape(title),
        badges=badges,
        authors=html.escape(authors),
        updated_at=html.escape(updated_at),
        sections=section_html,
    )
    st.markdown(detail_html, unsafe_allow_html=True)








def render_processed_pdf_summary(processed_pdf: dict[str, Any]) -> None:
    """Render uploaded-file metadata."""
    saved_file = processed_pdf["saved_file"]
    parsed_pdf = processed_pdf["parsed_pdf"]
    indexed_image_count = len(parsed_pdf.get("images", []))
    zh_path = parsed_translated_markdown_path(parsed_pdf)
    zh_badge = (
        render_status_badge("中文译文已生成", "success")
        if zh_path and zh_path.exists()
        else render_status_badge("中文译文未生成", "warning")
    )
    st.markdown(
        (
            '<div class="pm-panel">'
            '<h3 class="pm-section-title">论文已解析</h3>'
            '<div class="pm-file-capsule">'
            '<div><strong>{file_name}</strong>'
            '<div class="pm-card-meta">{file_size} · paper_id: {paper_id}</div></div>'
            '<div class="pm-badges">{page_badge}{image_badge}{char_badge}{zh_badge}</div>'
            '</div>'
            '</div>'
        ).format(
            file_name=html.escape(saved_file["file_name"]),
            file_size=html.escape(saved_file["file_size"]),
            paper_id=html.escape(saved_file["paper_id"]),
            page_badge=render_status_badge(f"页数 {parsed_pdf.get('page_count', 0)}", "info"),
            image_badge=render_status_badge(
                f"图片索引 {indexed_image_count} 张" if indexed_image_count else "图片未索引",
                "primary" if indexed_image_count else "default",
            ),
            char_badge=render_status_badge(f"字符 {processed_pdf.get('total_chars', 0)}", "default"),
            zh_badge=zh_badge,
        ),
        unsafe_allow_html=True,
    )
    render_reparse_controls(processed_pdf)


def render_reparse_controls(processed_pdf: dict[str, Any]) -> None:
    """Render explicit reparse actions for text-only or image-aware parsing."""
    team_context = current_team_context()
    can_edit = can_write(team_context["role"])
    paper_id = str(processed_pdf.get("saved_file", {}).get("paper_id") or "")
    if not paper_id:
        return

    parse_status = current_parse_status(processed_pdf)
    busy = parse_status in {"queued", "running"}
    with st.expander("重新解析", expanded=False):
        include_images = st.checkbox(
            "本次重新解析包含图片识别",
            value=False,
            key=f"reparse_include_images_{paper_id}",
            disabled=not can_edit or busy,
        )
        if include_images:
            st.warning("如果图片数量较多可能需要数十分钟")

        button_label = "添加图片并重新解析" if include_images else "重新解析正文"
        if st.button(
            button_label,
            type="primary" if include_images else "secondary",
            use_container_width=True,
            disabled=not can_edit or busy,
            key=f"reparse_paper_{paper_id}",
        ):
            try:
                result = enqueue_paper_parse(paper_id, include_images=include_images)
                st.session_state["pm_workspace_notice"] = result["message"]
                st.rerun()
            except Exception as exc:
                logger.exception("Paper reparse enqueue failed. paper_id=%s", paper_id)
                render_error_card("重新解析任务创建失败", "请检查团队权限、任务队列和 SQLite 写入状态。", str(exc))

        if busy:
            st.caption("当前已有解析任务在排队或运行，完成后才能再次提交。")
        elif not can_edit:
            st.caption("当前角色没有重新解析权限。")




def render_translation_controls(processed_pdf: dict[str, Any]) -> None:
    """Render Markdown translation action, progress, and downloads."""
    team_context = current_team_context()
    can_edit = can_write(team_context["role"])
    parsed_pdf = processed_pdf["parsed_pdf"]
    markdown_path_text = parsed_pdf.get("markdown_path")
    if not markdown_path_text:
        st.caption("当前解析结果没有可翻译的 Markdown 文件。")
        return

    markdown_path = Path(markdown_path_text)
    zh_path = parsed_translated_markdown_path(parsed_pdf) or translated_markdown_output_path(markdown_path)
    zh_exists = zh_path.exists()

    if not settings.translation_enabled:
        st.info("中文 Markdown 翻译功能当前未启用。可在 .env 中设置 TRANSLATION_ENABLED=true。")
    else:
        translate_label = "重新翻译中文 Markdown" if zh_exists else "翻译为中文 Markdown"
        translate_cols = st.columns([0.58, 0.42], gap="small")
        with translate_cols[0]:
            if st.button(
                translate_label,
                type="primary" if not zh_exists else "secondary",
                use_container_width=True,
                key=f"translate_markdown_{processed_pdf['saved_file']['paper_id']}",
                disabled=not can_edit,
            ):
                try:
                    job_id = enqueue_job(
                        "translate",
                        user_id=current_user_id(),
                        team_id=int(team_context["team_id"]),
                        project_id=team_context.get("project_id"),
                        paper_id=processed_pdf["saved_file"]["paper_id"],
                        payload={
                            "paper_id": processed_pdf["saved_file"]["paper_id"],
                            "input_md_path": str(markdown_path),
                            "output_md_path": str(zh_path),
                            "force": bool(zh_exists),
                        },
                    )
                    update_paper_status(processed_pdf["saved_file"]["paper_id"], translation_status="queued")
                    st.success(f"翻译任务已入队：#{job_id}。worker 完成后可下载中文 Markdown。")
                except Exception as exc:
                    logger.exception("Markdown translation failed.")
                    render_error_card(
                        "中文 Markdown 翻译任务创建失败",
                        "请检查团队权限和 SQLite 写入权限。",
                        str(exc),
                    )
        with translate_cols[1]:
            st.caption(
                f"模型：{settings.translation_model} · 分块：{settings.translation_chunk_size} 字符 · 超时：{settings.translation_timeout} 秒/片段"
            )
            latest_translate_job = latest_job_for_paper(int(team_context["team_id"]), processed_pdf["saved_file"]["paper_id"], "translate")
            if latest_translate_job:
                st.caption(f"最近翻译任务：#{latest_translate_job['job_id']} · {latest_translate_job['status']}")

    download_cols = st.columns(2, gap="small")
    with download_cols[0]:
        if markdown_path.exists():
            st.download_button(
                "下载原文 Markdown",
                data=markdown_path.read_bytes(),
                file_name=markdown_path.name,
                mime="text/markdown",
                use_container_width=True,
            )
    with download_cols[1]:
        if zh_path.exists():
            parsed_pdf["translated_markdown_path"] = str(zh_path.resolve())
            st.download_button(
                "下载中文 Markdown",
                data=zh_path.read_bytes(),
                file_name=zh_path.name,
                mime="text/markdown",
                use_container_width=True,
            )
        else:
            st.button("下载中文 Markdown", disabled=True, use_container_width=True)


def get_translated_markdown_path(original_md_path: str) -> str:
    """Return the non-destructive Chinese Markdown path for a source Markdown file."""
    source = Path(original_md_path)
    if source.suffix.lower() == ".md":
        return str(source.with_name(f"{source.stem}.zh.md"))
    return str(source.with_name(f"{source.name}.zh.md"))


def translated_markdown_output_path(markdown_path: str | Path) -> Path:
    """Return the sibling .zh.md path for one Markdown file."""
    return Path(get_translated_markdown_path(str(markdown_path)))


def parsed_translated_markdown_path(parsed_pdf: dict[str, Any]) -> Path | None:
    """Return the translated Markdown path when known or derivable."""
    existing = parsed_pdf.get("translated_markdown_path")
    if existing:
        return Path(existing)
    markdown_path = parsed_pdf.get("markdown_path")
    if not markdown_path:
        return None
    zh_path = translated_markdown_output_path(markdown_path)
    if zh_path.exists():
        parsed_pdf["translated_markdown_path"] = str(zh_path.resolve())
    return zh_path


def render_segmented_choice(label: str, options: list[str], key: str, default: str) -> str:
    """Render a segmented control with radio fallback for older Streamlit versions."""
    if st.session_state.get(key) not in options:
        st.session_state[key] = default
    segmented_control = getattr(st, "segmented_control", None)
    if callable(segmented_control):
        try:
            selected = segmented_control(label, options, key=key)
        except TypeError:
            selected = st.radio(label, options, key=key, horizontal=True)
    else:
        selected = st.radio(label, options, key=key, horizontal=True)
    if selected not in options:
        selected = default
        st.session_state[key] = default
    return selected


def render_reader_translation_button(
    processed_pdf: dict[str, Any],
    markdown_path: Path,
    zh_path: Path,
    zh_exists: bool,
) -> None:
    """Render the reader-level translation action."""
    team_context = current_team_context()
    can_edit = can_write(team_context["role"])
    if not settings.translation_enabled:
        st.button("翻译为中文 Markdown", disabled=True, use_container_width=True)
        st.caption("中文翻译功能未启用。")
        return

    parsed_pdf = processed_pdf["parsed_pdf"]
    paper_id = processed_pdf["saved_file"]["paper_id"]
    label = "重新翻译中文 Markdown" if zh_exists else "翻译为中文 Markdown"
    if not st.button(
        label,
        type="primary" if not zh_exists else "secondary",
        use_container_width=True,
        key=f"translate_markdown_reader_{paper_id}",
        disabled=not can_edit,
    ):
        if not can_edit:
            st.caption("当前团队角色为只读，不能创建翻译任务。")
        return

    try:
        del parsed_pdf
        job_id = enqueue_job(
            "translate",
            user_id=current_user_id(),
            team_id=int(team_context["team_id"]),
            project_id=team_context.get("project_id"),
            paper_id=paper_id,
            payload={
                "paper_id": paper_id,
                "input_md_path": str(markdown_path),
                "output_md_path": str(zh_path),
                "force": bool(zh_exists),
            },
        )
        update_paper_status(paper_id, translation_status="queued")
        st.success(f"翻译任务已入队：#{job_id}。worker 完成后可以切换中文译文或双语对照。")
    except Exception as exc:
        logger.exception("Markdown translation failed from reader.")
        render_error_card(
            "中文 Markdown 翻译任务创建失败",
            "请检查团队权限和 SQLite 写入权限。",
            str(exc),
        )


@st.cache_resource(show_spinner=False)
def get_markdown_renderer() -> Any:
    """Return a Markdown-it renderer when available."""
    try:
        from markdown_it import MarkdownIt
    except ImportError:
        return None

    try:
        return MarkdownIt("default", {"html": True})
    except Exception:
        return MarkdownIt("commonmark", {"html": True})


def markdown_to_html_fragment(markdown_text: str, images: list[dict[str, str]] | None = None) -> str:
    """Render a Markdown fragment to HTML for the interleaved reader."""
    safe_markdown = markdown_for_display(markdown_text or "", images or [])
    if not safe_markdown.strip():
        return ""

    renderer = get_markdown_renderer()
    if renderer is None:
        return f"<p>{html.escape(safe_markdown).replace(chr(10), '<br>')}</p>"

    try:
        return renderer.render(safe_markdown)
    except Exception:
        logger.debug("Markdown-it failed to render bilingual block.", exc_info=True)
        return f"<pre>{html.escape(safe_markdown)}</pre>"


def bilingual_alignment_cache_key(
    source_path: Path | None,
    translated_path: Path | None,
    align_mode: str,
    source_markdown: str,
    translated_markdown: str,
) -> str:
    """Build a cache key for bilingual alignment."""
    path_parts: list[str] = [BILINGUAL_ALIGNMENT_CACHE_VERSION, align_mode]
    for path in (source_path, translated_path):
        if path and path.exists():
            stat = path.stat()
            path_parts.append(str(path.resolve()))
            path_parts.append(str(stat.st_mtime_ns))
            path_parts.append(str(stat.st_size))
        else:
            path_parts.append("<missing>")
    if not source_path or not translated_path:
        digest = hashlib.sha1((source_markdown + "\n---\n" + translated_markdown).encode("utf-8")).hexdigest()
        path_parts.append(digest)
    return "|".join(path_parts)


def get_cached_bilingual_blocks(
    source_markdown: str,
    translated_markdown: str,
    align_mode: str,
    source_path: Path | None,
    translated_path: Path | None,
    images: list[dict[str, str]] | None = None,
) -> list[dict[str, Any]]:
    """Return cached bilingual alignment blocks when the source inputs are unchanged."""
    source_for_alignment = markdown_for_display(source_markdown, images or [])
    translated_for_alignment = markdown_for_display(translated_markdown, images or [])
    cache_key = bilingual_alignment_cache_key(
        source_path,
        translated_path,
        align_mode,
        source_for_alignment,
        translated_for_alignment,
    )
    if (
        st.session_state.get("bilingual_cache_key") == cache_key
        and isinstance(st.session_state.get("bilingual_aligned_blocks"), list)
    ):
        return st.session_state["bilingual_aligned_blocks"]

    blocks = align_markdown_bilingual(source_for_alignment, translated_for_alignment, mode=align_mode)
    st.session_state["bilingual_cache_key"] = cache_key
    st.session_state["bilingual_aligned_blocks"] = blocks
    return blocks


def render_interleaved_bilingual_reader(
    source_markdown: str,
    translated_markdown: str,
    align_mode: str = "section",
    source_path: Path | None = None,
    translated_path: Path | None = None,
    images: list[dict[str, str]] | None = None,
) -> None:
    """Render a vertical interleaved bilingual Markdown reader."""
    if not source_markdown.strip():
        render_empty_state("还没有可阅读的论文正文", "请先上传 PDF 并完成解析。", icon="MD")
        return
    if not translated_markdown.strip():
        render_empty_state(
            "还没有中文译文",
            "点击“翻译为中文 Markdown”后，即可开启一段英文、一段中文的双语对照阅读。",
            "翻译为中文 Markdown",
            icon="ZH",
        )
        return

    image_list = images or []
    paper_header, source_body_markdown = prepare_bilingual_reader_markdown(source_markdown, image_list)
    _translated_header, translated_body_markdown = prepare_bilingual_reader_markdown(translated_markdown, image_list)
    if paper_header:
        st.caption(f"标题：{paper_header.get('title') or '未识别'}")
        st.caption(f"作者：{paper_header.get('authors') or '未识别'}")

    try:
        blocks = get_cached_bilingual_blocks(
            source_body_markdown,
            translated_body_markdown,
            align_mode,
            source_path,
            translated_path,
            None,
        )
    except Exception as exc:
        logger.exception("Bilingual alignment failed.")
        render_error_card(
            "双语对照暂时无法生成",
            "系统无法完成当前论文的双语对齐，请稍后重试或切换到原文/中文译文模式。",
            str(exc),
        )
        return

    warning = next((block.get("alignment_warning") for block in blocks if block.get("alignment_warning")), "")
    missing_source = sum(1 for block in blocks if not block.get("source"))
    missing_target = sum(1 for block in blocks if not block.get("target"))
    if warning or missing_source or missing_target:
        st.markdown(
            '<div class="pm-align-warning"><strong>双语对照对齐不完整</strong><br>'
            "系统已按段落顺序尽量展示英文原文和中文译文。</div>",
            unsafe_allow_html=True,
        )
        with st.expander("技术详情", expanded=False):
            st.write(
                {
                    "warning": warning or None,
                    "missing_source_blocks": missing_source,
                    "missing_target_blocks": missing_target,
                    "total_blocks": len(blocks),
                    "align_mode": align_mode,
                }
            )

    if len(blocks) > 200:
        st.info("当前论文较长，双语对照渲染可能稍慢。")

    align_label = "章节对齐" if align_mode == "section" else "段落对齐"
    parts = [
        '<div class="pm-interleaved-reader">',
        '<div class="pm-interleaved-toolbar">',
        '<div><div class="pm-interleaved-toolbar-title">双语对照阅读</div>',
        f'<div class="pm-interleaved-toolbar-meta">显示模式：段落交错 · 对齐方式：{html.escape(align_label)}</div></div>',
        f'<div class="pm-badges">{render_status_badge("纵向交错", "primary")}{render_status_badge(align_label, "info")}</div>',
        "</div>",
        '<div class="pm-bilingual-flow">',
    ]

    for block in blocks:
        block_type = str(block.get("type") or "other")
        block_class = "pm-bilingual-block"
        if block_type == "heading":
            block_class += " pm-bilingual-block-heading"
        anchor = html.escape(str(block.get("anchor") or block.get("id") or ""))
        source_notices, source_content = split_bilingual_image_notices(str(block.get("source") or ""))
        target_notices, target_content = split_bilingual_image_notices(str(block.get("target") or ""))
        for notice in merge_image_notices(source_notices, target_notices):
            parts.append(bilingual_image_notice_html(notice))

        source_html = markdown_to_html_fragment(source_content, None)
        target_html = markdown_to_html_fragment(target_content, None)
        if not source_html and not target_html:
            continue
        if not source_html:
            source_html = '<p class="pm-block-placeholder">无对应原文</p>'
        if not target_html:
            target_html = '<p class="pm-block-placeholder">暂无对应译文</p>'

        parts.extend(
            [
                f'<section id="bilingual-{anchor}" class="{block_class}" data-block-type="{html.escape(block_type)}">',
                '<div class="pm-source-block">',
                '<div class="pm-lang-label pm-lang-label-source">原文</div>',
                f'<div class="pm-block-content">{source_html}</div>',
                "</div>",
                '<div class="pm-target-block">',
                '<div class="pm-lang-label pm-lang-label-target">中文译文</div>',
                f'<div class="pm-block-content">{target_html}</div>',
                "</div>",
                "</section>",
            ]
        )

    # TODO: support jumping from citation chunk_id to interleaved bilingual block.
    parts.extend(["</div>", "</div>"])
    st.markdown("".join(parts), unsafe_allow_html=True)


def read_original_markdown(processed_pdf: dict[str, Any]) -> tuple[str, Path | None]:
    """Read the original Markdown from disk when possible."""
    parsed_pdf = processed_pdf["parsed_pdf"]
    markdown_path_text = parsed_pdf.get("markdown_path")
    markdown_path = Path(markdown_path_text) if markdown_path_text else None
    if markdown_path and markdown_path.exists():
        try:
            return markdown_path.read_text(encoding="utf-8"), markdown_path
        except OSError:
            logger.warning("Failed to read original Markdown from disk.", exc_info=True)
    chunk_markdown = chunks_to_markdown(processed_pdf.get("chunks", []))
    return parsed_pdf.get("markdown", "") or chunk_markdown or processed_pdf.get("preview", ""), markdown_path


def render_pending_source_scroll() -> None:
    """Execute a pending scroll to a source anchor after the reader is rendered."""
    anchor_id = str(st.session_state.pop("pm_pending_source_anchor", "") or "").strip()
    if not anchor_id:
        return

    components.html(
        f"""
        <script>
        const anchorId = {json.dumps(anchor_id)};
        const qaAnchorId = "pm-qa-anchor";
        const sourceHighlightSelector = ".pm-source-highlight";

        const getHighlightTarget = (anchor) => {{
          const contentSelector = "p, li, h1, h2, h3, h4, h5, h6, blockquote, pre, table";
          if (anchor.parentElement && anchor.parentElement.matches(contentSelector)) {{
            return anchor.parentElement;
          }}

          let element = anchor.nextElementSibling;
          while (element) {{
            if (element.matches(contentSelector)) {{
              return element;
            }}
            const nested = element.querySelector?.(contentSelector);
            if (nested) {{
              return nested;
            }}
            element = element.nextElementSibling;
          }}

          const container = anchor.closest("[data-testid='stMarkdownContainer'], .stMarkdown");
          return container || anchor.parentElement || anchor;
        }};

        const highlightSourceAnchor = (target) => {{
          const parentDocument = window.parent.document;
          parentDocument
            .querySelectorAll(sourceHighlightSelector)
            .forEach((element) => element.classList.remove("pm-source-highlight"));

          const highlightTarget = getHighlightTarget(target);
          if (highlightTarget) {{
            highlightTarget.classList.add("pm-source-highlight");
          }}
        }};

        const ensureReturnToQaButton = () => {{
          const parentDocument = window.parent.document;
          parentDocument.getElementById("pm-return-qa-button")?.remove();

          const button = parentDocument.createElement("button");
          button.id = "pm-return-qa-button";
          button.type = "button";
          button.className = "pm-return-qa-button";
          button.textContent = "回到问答区";
          button.addEventListener("click", () => {{
            const qaAnchor = parentDocument.getElementById(qaAnchorId);
            if (qaAnchor) {{
              qaAnchor.scrollIntoView({{ behavior: "smooth", block: "start", inline: "nearest" }});
            }}
            button.remove();
          }});
          parentDocument.body.appendChild(button);
        }};

        const scrollToSourceAnchor = () => {{
          const parentDocument = window.parent.document;
          const target = parentDocument.getElementById(anchorId);
          if (!target) {{
            return;
          }}
          target.scrollIntoView({{ behavior: "smooth", block: "start", inline: "nearest" }});
          highlightSourceAnchor(target);
          ensureReturnToQaButton();
        }};
        window.setTimeout(scrollToSourceAnchor, 80);
        window.setTimeout(scrollToSourceAnchor, 360);
        </script>
        """,
        height=0,
        scrolling=False,
    )


def render_markdown_document(processed_pdf: dict[str, Any]) -> None:
    """Render original, Chinese, or interleaved bilingual Markdown for reading."""
    parsed_pdf = processed_pdf["parsed_pdf"]
    paper_id = processed_pdf["saved_file"]["paper_id"]
    source_markdown, markdown_path = read_original_markdown(processed_pdf)
    if not source_markdown.strip():
        render_empty_state("还没有可阅读的论文正文", "请先上传 PDF 并完成解析。", icon="MD")
        return

    if markdown_path is None:
        zh_path = parsed_translated_markdown_path(parsed_pdf)
    else:
        zh_path = parsed_translated_markdown_path(parsed_pdf) or translated_markdown_output_path(markdown_path)
    zh_exists = bool(zh_path and zh_path.exists())

    if st.session_state.get("pm_pending_source_anchor"):
        st.session_state["reading_mode"] = SOURCE_READING_MODE

    reading_mode = render_segmented_choice("阅读模式", ["原文", "中文译文", "双语对照"], "reading_mode", "原文")
    align_label = st.session_state.get("bilingual_align_mode")
    if align_label not in {"章节对齐", "段落对齐"}:
        st.session_state["bilingual_align_mode"] = "章节对齐"
        align_label = "章节对齐"
    align_mode = "section" if align_label == "章节对齐" else "paragraph"

    mode_badge_type = {"原文": "primary", "中文译文": "success", "双语对照": "info"}.get(reading_mode, "default")
    st.markdown(
        (
            '<div class="pm-panel">'
            '<h3 class="pm-section-title">论文正文</h3>'
            '<div class="pm-toolbar" style="margin-top:12px;">'
            '{mode_badge}{parser_badge}{page_badge}{image_badge}{chunk_badge}'
            '</div></div>'
        ).format(
            mode_badge=render_status_badge(reading_mode, mode_badge_type),
            parser_badge=render_status_badge(f"解析方式 {parsed_pdf.get('parser', '未知')}", "primary"),
            page_badge=render_status_badge(f"页数 {parsed_pdf.get('page_count', 0)}", "info"),
            image_badge=render_status_badge(
                f"图片索引 {len(parsed_pdf.get('images', []))} 张"
                if parsed_pdf.get("images")
                else "图片未索引",
                "default",
            ),
            chunk_badge=render_status_badge(f"Chunks {len(processed_pdf.get('chunks', []))}", "success"),
        ),
        unsafe_allow_html=True,
    )

    if reading_mode == "双语对照":
        align_label = render_segmented_choice(
            "双语对齐方式",
            ["章节对齐", "段落对齐"],
            "bilingual_align_mode",
            "章节对齐",
        )
        align_mode = "section" if align_label == "章节对齐" else "paragraph"

    if markdown_path:
        action_cols = st.columns(3, gap="small")
        with action_cols[0]:
            render_reader_translation_button(processed_pdf, markdown_path, zh_path or translated_markdown_output_path(markdown_path), zh_exists)
        with action_cols[1]:
            st.download_button(
                "下载原文 Markdown",
                data=markdown_path.read_bytes() if markdown_path.exists() else source_markdown.encode("utf-8"),
                file_name=markdown_path.name,
                mime="text/markdown",
                use_container_width=True,
                key=f"download_source_markdown_reader_{paper_id}",
            )
        with action_cols[2]:
            if zh_exists and zh_path:
                parsed_pdf["translated_markdown_path"] = str(zh_path.resolve())
                st.download_button(
                    "下载中文译文",
                    data=zh_path.read_bytes(),
                    file_name=zh_path.name,
                    mime="text/markdown",
                    use_container_width=True,
                    key=f"download_zh_markdown_reader_{paper_id}",
                )
            else:
                st.button("下载中文译文", disabled=True, use_container_width=True, key=f"download_zh_disabled_{paper_id}")

    if reading_mode in {"中文译文", "双语对照"} and (not zh_exists or zh_path is None):
        render_empty_state(
            "还没有中文译文",
            "点击“翻译为中文 Markdown”后，即可开启一段英文、一段中文的双语对照阅读。",
            "翻译为中文 Markdown",
            icon="ZH",
        )
        return

    images = parsed_pdf.get("images", [])
    if reading_mode == "中文译文":
        translated_markdown = zh_path.read_text(encoding="utf-8") if zh_path else ""
        safe_markdown = markdown_for_display(translated_markdown, images)
        paper_header, body_markdown = split_paper_header(safe_markdown)
        if paper_header:
            st.caption(f"标题：{paper_header.get('title') or '未识别'}")
            st.caption(f"作者：{paper_header.get('authors') or '未识别'}")
        render_markdown_with_image_previews(
            body_markdown or safe_markdown or "暂无 Markdown 内容",
            images,
            f"reader_{paper_id}_zh",
        )
        return

    if reading_mode == "双语对照":
        translated_markdown = zh_path.read_text(encoding="utf-8") if zh_path else ""
        render_interleaved_bilingual_reader(
            source_markdown,
            translated_markdown,
            align_mode=align_mode,
            source_path=markdown_path,
            translated_path=zh_path,
            images=images,
        )
        return

    safe_markdown = markdown_for_display(source_markdown, images)
    paper_header, body_markdown = split_paper_header(safe_markdown)
    anchored_body, _missing_chunks = add_chunk_anchors_to_markdown(
        body_markdown or safe_markdown,
        processed_pdf.get("chunks", []),
    )
    if paper_header:
        st.caption(f"标题：{paper_header.get('title') or '未识别'}")
        st.caption(f"作者：{paper_header.get('authors') or '未识别'}")
    render_markdown_with_image_previews(
        anchored_body or "暂无 Markdown 内容",
        images,
        f"reader_{paper_id}_source",
    )
    render_pending_source_scroll()


def render_app() -> None:
    """Render the PaperMate app."""
    st.set_page_config(page_title=settings.app_name, layout="wide", initial_sidebar_state="expanded")
    inject_global_css()
    render_app_shell()

    init_db()
    user = current_user()
    if not user:
        render_auth_page()
        return

    handle_queue_cancel_query(int(user["user_id"]))
    prepare_user_workspace_once(int(user["user_id"]))
    page = render_sidebar_navigation(user)
    team_context = current_team_context()
    selected_team_id = int(team_context.get("team_id") or 0)
    if selected_team_id:
        render_queue_clear_notice()
        render_queue_action_notice()
        render_global_queue_progress(int(user["user_id"]), selected_team_id)

    if page == "论文工作台":
        render_workspace_page()
    elif page == "论文库":
        render_paper_library_page()
    elif page == "文献卡片库":
        render_card_library_page(int(user["user_id"]))
    elif page == "团队管理":
        render_team_management_page()
    else:
        render_feedback_records_page()


if __name__ == "__main__":
    render_app()
