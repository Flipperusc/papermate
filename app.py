"""Streamlit entry point for PaperMate."""

from __future__ import annotations

import base64
import hashlib
import html
import hmac
import io
import os
import sqlite3
import re
import zipfile
from pathlib import Path
from typing import Any, Protocol
from uuid import uuid4

import streamlit as st

from config import settings
from src.auth_service import authenticate_user, create_user, get_user_by_id
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
from src.pdf_parser import parse_pdf
from src.rag_pipeline import answer_question
from src.retrieval.bm25_store import BM25Store
from src.vector_store import VectorStore


logger = get_logger(__name__)

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
    ensure_default_card_library(user_id)
    claim_unassigned_literature_cards(user_id)


def clear_authenticated_session() -> None:
    """Clear user-scoped session state on logout."""
    keep_keys = {"welcome_seen"}
    for key in list(st.session_state.keys()):
        if key not in keep_keys:
            del st.session_state[key]


def render_welcome_dialog() -> None:
    """Show the first-visit PaperMate intro dialog."""
    if st.session_state.get("welcome_seen"):
        return

    @st.dialog("欢迎来到 PaperMate")
    def welcome() -> None:
        st.markdown(
            """
            <div class="pm-welcome">
              <div class="pm-welcome-kicker">你的论文搭子</div>
              <h2>把 PDF 丢进来，剩下交给 PaperMate 。</h2>
              <p>我会帮你把论文转成 Markdown，抽出可检索的片段，严格根据论文本身回答问题，还能把重点整理成文献卡片。</p>
              <ul>
                <li>上传 PDF，解析正文和图片。</li>
                <li>构建 Hybrid RAG 索引，问问题不靠玄学。</li>
                <li>把卡片存进你自己的卡片库，按主题分门别类。</li>
              </ul>
            </div>
            """,
            unsafe_allow_html=True,
        )
        if st.button("开始整理我的论文", type="primary", use_container_width=True):
            st.session_state["welcome_seen"] = True
            st.rerun()

    welcome()


def render_auth_page() -> None:
    """Render login and registration controls."""
    render_header()
    st.markdown(
        """
        <div class="pm-auth-shell">
          <div class="pm-auth-copy">
            <h2>先认个门牌号</h2>
            <p>登录后，文献卡片会只进你的库。别人看不到你的卡片，你也不会误删别人的整理成果。</p>
            <p>新用户注册后会自动获得一个“默认卡片库”，之后可以继续创建新的主题库。</p>
          </div>
        """,
        unsafe_allow_html=True,
    )

    login_tab, register_tab = st.tabs(["登录", "注册"])

    with login_tab:
        with st.form("login_form"):
            username = st.text_input("用户名", key="login_username")
            password = st.text_input("密码", type="password", key="login_password")
            submitted = st.form_submit_button("登录", type="primary", use_container_width=True)

        if submitted:
            user = authenticate_user(username, password)
            if not user:
                st.error("用户名或密码不正确。")
                return
            prepare_user_workspace(int(user["user_id"]))
            set_current_user(user)
            st.success("登录成功。")
            st.rerun()

    with register_tab:
        with st.form("register_form"):
            username = st.text_input("用户名", key="register_username")
            password = st.text_input("密码", type="password", key="register_password")
            password_confirm = st.text_input("确认密码", type="password", key="register_password_confirm")
            submitted = st.form_submit_button("注册并进入", type="primary", use_container_width=True)

        if submitted:
            if password != password_confirm:
                st.error("两次输入的密码不一致。")
                return
            try:
                user = create_user(username, password)
                prepare_user_workspace(int(user["user_id"]))
            except (ValueError, OSError, sqlite3.Error) as exc:
                st.error(str(exc) or "注册失败，请稍后再试。")
                return
            set_current_user(user)
            st.success("注册成功。")
            st.rerun()

    st.markdown("</div></div>", unsafe_allow_html=True)


def render_sidebar_navigation(user: dict[str, Any]) -> str:
    """Render the sidebar and return the selected page."""
    st.sidebar.markdown(
        f"""
        <div class="pm-sidebar-brand">
          <div class="pm-sidebar-brand-title">PaperMate</div>
          <div class="pm-sidebar-brand-subtitle">论文阅读、问答、卡片和反馈管理</div>
        </div>
        <div class="pm-user-pill">当前用户：{html.escape(str(user["username"]))}</div>
        <div class="pm-sidebar-section">工作区</div>
        """,
        unsafe_allow_html=True,
    )
    page = st.sidebar.radio(
        "页面",
        ["论文工作台", "文献卡片库", "反馈记录"],
        label_visibility="collapsed",
    )

    st.sidebar.caption("PDF 转 Markdown · Hybrid RAG · 文献卡片")
    if st.sidebar.button("退出登录", use_container_width=True):
        clear_authenticated_session()
        st.rerun()

    return page


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
    }


def build_text_preview(parsed_pdf: dict[str, Any], limit: int = 1000) -> tuple[int, str]:
    """Build total character count and preview from parsed pages."""
    pages = parsed_pdf["pages"]
    full_text = "\n\n".join(page["text"] for page in pages)
    return len(full_text), full_text[:limit]


def markdown_for_display(markdown: str, images: list[dict[str, str]] | None = None) -> str:
    """Return full Markdown with image payloads replaced by text placeholders."""
    next_image_index = 1

    def next_placeholder() -> str:
        nonlocal next_image_index
        index = next_image_index
        next_image_index += 1
        return f"**此处图片{index}已省略**"

    safe_markdown = markdown or ""
    safe_markdown = re.sub(
        r"!\[[^\]]*\]\(data:image[^)]*\)",
        lambda _match: next_placeholder(),
        safe_markdown,
        flags=re.IGNORECASE,
    )

    def replace_data_uri_link(match: re.Match[str]) -> str:
        nonlocal next_image_index
        label = match.group("label")
        image_match = re.search(r"图\s*(\d+)", label)
        if image_match:
            image_index = int(image_match.group(1))
            next_image_index = max(next_image_index, image_index + 1)
            return f"**此处图片{image_index}已省略**"
        return next_placeholder()

    safe_markdown = re.sub(
        r"\[(?P<label>[^\]]*)\]\(data:image[^)]*\)",
        replace_data_uri_link,
        safe_markdown,
        flags=re.IGNORECASE,
    )
    safe_markdown = re.sub(
        r"!\[[^\]]*\]\([^)]+\)",
        lambda _match: next_placeholder(),
        safe_markdown,
    )
    safe_markdown = re.sub(
        r"<img\b[^>]*>",
        lambda _match: next_placeholder(),
        safe_markdown,
        flags=re.IGNORECASE,
    )
    return safe_markdown.strip()


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
        candidate = clean_header_line(lines[index])
        if not candidate:
            if author_lines:
                body_start = index + 1
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
    return normalized in {
        "abstract",
        "摘要",
        "keywords",
        "key words",
        "关键词",
        "introduction",
        "1 introduction",
        "i introduction",
    }


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


def chunk_anchor_html(chunk_id: Any) -> str:
    """Return an HTML anchor marker for chunk-level source navigation."""
    anchor_id = html.escape(chunk_anchor_id(chunk_id), quote=True)
    return f'<span id="{anchor_id}" class="pm-source-anchor"></span>'


def source_anchor_link(chunk_id: Any, label: str = "回到原文") -> str:
    """Return a small HTML link that navigates to a chunk anchor."""
    anchor_id = html.escape(chunk_anchor_id(chunk_id), quote=True)
    return f'<a class="pm-source-link" href="#{anchor_id}">{html.escape(label)}</a>'


def add_chunk_anchors_to_markdown(
    markdown: str,
    chunks: list[dict[str, Any]],
) -> tuple[str, list[dict[str, Any]]]:
    """Insert chunk anchors into Markdown when chunk text can be located."""
    if not markdown or not chunks:
        return markdown, chunks

    normalized_markdown, offset_map = normalize_with_offsets(markdown)
    inserts: list[tuple[int, str]] = []
    missing_chunks: list[dict[str, Any]] = []
    used_positions: set[int] = set()

    for chunk in chunks:
        candidate = chunk_search_candidate(str(chunk.get("text") or ""))
        normalized_candidate, _ = normalize_with_offsets(candidate)
        if len(normalized_candidate) < 24:
            missing_chunks.append(chunk)
            continue

        match_position = normalized_markdown.find(normalized_candidate)
        if match_position < 0:
            missing_chunks.append(chunk)
            continue

        original_position = offset_map[match_position]
        while original_position in used_positions and original_position < len(markdown):
            original_position += 1
        used_positions.add(original_position)
        inserts.append((original_position, chunk_anchor_html(chunk.get("chunk_id"))))

    anchored_markdown = markdown
    for position, anchor in sorted(inserts, key=lambda item: item[0], reverse=True):
        anchored_markdown = f"{anchored_markdown[:position]}{anchor}\n{anchored_markdown[position:]}"

    return anchored_markdown, missing_chunks


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
    cleaned = re.sub(r"\s+", " ", text).strip()
    cleaned = re.sub(r"\*\*此处图片\d+已省略\*\*", " ", cleaned)
    if not cleaned:
        return ""
    return cleaned[: min(220, len(cleaned))]


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
        ]
    )
    return f"{uploaded_file.name}:{len(file_bytes)}:{digest}:{parse_settings}"


def process_uploaded_pdf(uploaded_file: UploadedFile) -> dict[str, Any]:
    """Save, parse, chunk, and persist an uploaded PDF once per session file."""
    signature = get_uploaded_file_signature(uploaded_file)
    cached_result = st.session_state.get("processed_pdf")
    # MinerU parsing can be slow and billable; include parser settings in the
    # signature so a changed configuration forces a fresh parse.
    if cached_result and cached_result.get("signature") == signature:
        return cached_result

    saved_file = save_uploaded_pdf(uploaded_file)
    parsed_pdf = parse_pdf(saved_file["save_path"], saved_file["paper_id"])
    chunks = chunk_pages(saved_file["paper_id"], parsed_pdf["pages"])
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


def render_header() -> None:
    """Render the shared page header."""
    st.markdown(
        """
        <div class="pm-hero">
          <h1>PaperMate</h1>
          <p>论文阅读 RAG 助手：PDF 转 Markdown、检索问答、反馈记录和文献卡片管理。</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_processed_pdf_summary(processed_pdf: dict[str, Any]) -> None:
    """Render uploaded file metadata and source downloads."""
    saved_file = processed_pdf["saved_file"]
    parsed_pdf = processed_pdf["parsed_pdf"]

    st.success("PDF 已保存并完成 Markdown 解析。")
    col_a, col_b, col_c = st.columns(3)
    col_a.metric("页数", parsed_pdf["page_count"])
    col_b.metric("字符数", processed_pdf["total_chars"])
    col_c.metric("图片数", len(parsed_pdf.get("images", [])))

    st.caption(f"文件名：{saved_file['file_name']}")
    st.caption(f"paper_id：{saved_file['paper_id']}")
    st.caption(f"PDF 路径：{saved_file['save_path']}")
    if parsed_pdf.get("markdown_path"):
        markdown_path = Path(parsed_pdf["markdown_path"])
        st.caption(f"Markdown 路径：{markdown_path}")
        if markdown_path.exists():
            st.download_button(
                "下载完整 Markdown",
                data=markdown_path.read_bytes(),
                file_name="full.md",
                mime="text/markdown",
                use_container_width=True,
            )

    render_extracted_images(parsed_pdf.get("images", []))


def render_markdown_document(processed_pdf: dict[str, Any]) -> None:
    """Render the full paper Markdown inline, excluding image payloads."""
    parsed_pdf = processed_pdf["parsed_pdf"]
    markdown = parsed_pdf.get("markdown", "") or processed_pdf["preview"]
    safe_markdown = markdown_for_display(markdown, parsed_pdf.get("images", []))
    paper_header, body_markdown = split_paper_header(safe_markdown)
    anchored_body, _missing_chunks = add_chunk_anchors_to_markdown(
        body_markdown or safe_markdown,
        processed_pdf.get("chunks", []),
    )

    st.markdown("#### 论文正文")
    if paper_header:
        st.markdown("##### 开头信息")
        st.write("标题：", paper_header.get("title") or "未识别")
        st.write("作者：", paper_header.get("authors") or "未识别")

    st.markdown(anchored_body or "暂无 Markdown 内容", unsafe_allow_html=True)

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


def render_literature_card_save(
    paper_id: str,
    chunks: list[dict[str, Any]],
    db_save_failed: bool,
    user_id: int,
) -> None:
    """Render literature card generation and persistence button."""
    st.markdown("#### 文献卡片")
    if not chunks:
        st.warning("论文正文为空，暂时无法生成文献卡片。")
        return
    if db_save_failed:
        st.warning("数据库保存失败，文献卡片可能无法基于完整 chunks 生成。")

    try:
        libraries = list_card_libraries(user_id)
    except (OSError, sqlite3.Error):
        st.error("卡片库读取失败，请检查 SQLite 数据库权限。")
        return

    if not libraries:
        st.warning("当前用户还没有可用卡片库。")
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
        with st.form(key=f"create_library_for_{paper_id}"):
            new_library_name = st.text_input("卡片库名称", placeholder="例如：综述必读、方法对照、毕业论文核心文献")
            submitted = st.form_submit_button("创建卡片库", type="primary", use_container_width=True)
        if submitted:
            try:
                create_card_library(user_id, new_library_name)
            except sqlite3.IntegrityError:
                st.error("这个卡片库名字已经存在。")
            except (ValueError, OSError, sqlite3.Error) as exc:
                st.error(str(exc) or "卡片库创建失败。")
            else:
                st.success("卡片库已创建。")
                st.rerun()

    if st.button("生成并保存为新文献卡片", type="primary", use_container_width=True, key=f"save_card_{paper_id}"):
        try:
            with st.spinner("正在生成文献卡片并保存..."):
                markdown = generate_literature_card(paper_id)
                card_id = save_literature_card(
                    paper_id,
                    markdown,
                    user_id=user_id,
                    library_id=int(selected_library_id),
                )
        except (LLMError, ValueError, OSError, sqlite3.Error) as exc:
            if isinstance(exc, LLMError):
                st.error(f"{exc.message}（错误码：{exc.code.value}）")
            else:
                st.error(str(exc) or "文献卡片保存失败，请检查 SQLite 数据库权限。")
            return

        st.session_state[f"saved_card_id_{paper_id}"] = card_id
        library = get_card_library(int(selected_library_id), user_id)
        library_name = library["name"] if library else "所选卡片库"
        st.success(f"新文献卡片已保存到「{library_name}」，可在“文献卡片库”页面查看、批量管理和编辑。")

    saved_card = get_literature_card_by_paper(paper_id, user_id=user_id)
    if saved_card:
        with st.expander("最近保存的卡片预览", expanded=False):
            render_card_visual(saved_card)


def render_chunk_preview(chunks: list[dict[str, Any]]) -> None:
    """Render chunk count and the first few chunk previews."""
    st.markdown("#### 切分结果")
    if not chunks:
        st.warning("论文正文为空，无法构建索引")
        return

    st.write("chunk 数量：", len(chunks))
    for chunk in chunks[:3]:
        section = chunk["section_title"] or "未识别章节"
        title = f"Chunk {chunk['chunk_index']} | 第 {chunk['page_num']} 页 | {section}"
        with st.expander(title, expanded=chunk["chunk_index"] == 0):
            st.caption(f"chunk_id：{chunk['chunk_id']}")
            st.text_area(
                "chunk 预览",
                chunk["text"][:1000],
                height=180,
                key=f"chunk_preview_{chunk['chunk_id']}",
            )


def render_index_builder(chunks: list[dict[str, Any]]) -> None:
    """Render the hybrid index build action."""
    st.markdown("#### 构建论文索引")
    if not chunks:
        st.info("没有可入库的 chunk。")
        return

    if st.button("构建论文索引", type="primary", use_container_width=True, key="build_paper_index"):
        paper_id = str(chunks[0].get("paper_id") or "")
        vector_count: int | None = None
        bm25_result: dict[str, Any] | None = None
        vector_error: Exception | None = None
        bm25_error: Exception | None = None

        with st.spinner("正在构建 Hybrid 索引"):
            # Build the two indexes independently so the app can degrade to the
            # working retrieval mode instead of failing the whole indexing step.
            try:
                vector_count = VectorStore().add_chunks(chunks)
            except (EmbeddingError, VectorStoreError) as exc:
                vector_error = exc
                logger.exception("Vector index build failed. paper_id=%s", paper_id)

            try:
                bm25_result = BM25Store().build_index(paper_id, chunks)
            except AppError as exc:
                bm25_error = exc
                logger.exception("BM25 index build failed. paper_id=%s", paper_id)

        if vector_error is None and bm25_error is None:
            st.success("论文索引构建完成，可开始问答。")
            st.caption(
                f"向量 chunk：{vector_count or 0}；关键词 chunk：{bm25_result.get('chunk_count', 0) if bm25_result else 0}"
            )
            return

        if vector_error is None and bm25_error is not None:
            st.warning("向量索引已构建成功，但关键词索引构建失败，当前仍可使用语义检索。")
            return

        if vector_error is not None and bm25_error is None:
            st.warning("关键词索引已构建成功，但向量索引构建失败，当前只能使用关键词检索。")
            return

        st.error("索引构建失败，请检查 API Key、网络连接和本地写入权限后重试。")


def render_citations(citations: list[dict[str, Any]]) -> None:
    """Render citations produced from chunk metadata."""
    st.markdown("#### 引用来源")
    if not citations:
        st.write("无引用来源。")
        return

    for index, citation in enumerate(citations, start=1):
        citation_id = citation.get("citation_id") or index
        page_label = format_page_label(citation.get("page_num"))
        section_title = citation.get("section_title") or "未知章节"
        title = f"引用片段 {citation_id}｜{page_label}｜{section_title}"
        source_ranks = citation.get("source_ranks") or {}

        with st.expander(title, expanded=index == 1):
            st.write("chunk_id：", citation.get("chunk_id", ""))
            st.markdown(source_anchor_link(citation.get("chunk_id")), unsafe_allow_html=True)
            st.write("检索来源：", format_retrieval_sources(citation.get("retrieval_sources")))
            st.write("source_ranks：", source_ranks or "无")

            vector_rank = source_ranks.get("vector")
            bm25_rank = source_ranks.get("bm25")
            if vector_rank:
                st.write("向量排名：", vector_rank)
            if bm25_rank:
                st.write("BM25 排名：", bm25_rank)

            rrf_score = citation.get("rrf_score")
            if rrf_score is not None:
                st.write("RRF 分数：", format_optional_float(rrf_score))

            preview = citation.get("text_preview") or ""
            if preview:
                st.markdown("**原文预览**")
                st.write(preview)


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
            f"{chunk.get('section_title', '未知章节')} | {chunk.get('chunk_id', '')}"
        )
        with st.expander(title):
            st.markdown(source_anchor_link(chunk.get("chunk_id")), unsafe_allow_html=True)
            st.write(chunk.get("text", ""))


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


def render_qa_box(paper_id: str) -> None:
    """Render RAG question-answering controls."""
    st.markdown("#### 论文问答")
    question = st.text_area(
        "输入你的问题",
        placeholder="例如：这篇论文的核心方法是什么？",
        key=f"question_{paper_id}",
        height=96,
    )

    if st.button("提问", key=f"ask_{paper_id}", type="primary", use_container_width=True):
        if not question.strip():
            st.warning("请输入问题。")
            return

        try:
            with st.spinner("正在检索论文片段并回答..."):
                rag_result = answer_question(paper_id, question)
        except AppError as exc:
            logger.exception("RAG question answering failed. paper_id=%s", paper_id)
            if exc.code in {ErrorCode.VECTOR_SEARCH_FAILED, ErrorCode.BM25_INDEX_MISSING}:
                st.error("请先构建论文索引后再提问。")
            else:
                st.error("问答失败，请查看日志或检查模型与索引配置。")
            return
        except Exception:
            logger.exception("Unexpected RAG question answering failure. paper_id=%s", paper_id)
            st.error("问答失败，请查看日志或检查模型与索引配置。")
            return

        qa_log_id = rag_result.get("qa_id")
        if qa_log_id is None:
            # The pipeline normally saves this; keep a fallback for tests and
            # older pipeline payloads that only return the answer object.
            try:
                qa_log_id = save_qa_log(paper_id, question.strip(), rag_result["answer"])
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
        st.markdown("#### 回答")
        st.write(qa_record["answer"])
        if needs_index_warning(qa_record.get("retrieval_details", {})):
            st.warning("请先构建论文索引后再提问。")
        render_citations(qa_record["citations"])
        render_retrieval_details(qa_record.get("retrieval_details", {}))
        render_source_chunks(qa_record["source_chunks"])
        render_feedback_form(qa_record)


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
        submitted = st.form_submit_button("提交反馈")

    if submitted:
        try:
            save_feedback(
                paper_id=paper_id,
                question=qa_record["question"],
                answer=qa_record["answer"],
                feedback_type=feedback_type,
                comment=comment.strip(),
                qa_log_id=qa_log_id,
            )
        except (OSError, sqlite3.Error):
            st.error("反馈保存失败，请检查 SQLite 数据库权限。")
            return

        st.session_state[feedback_state_key] = True

    if st.session_state.get(feedback_state_key):
        st.success("反馈已记录，将用于后续优化")


def render_workspace_page() -> None:
    """Render the two-column paper workspace page."""
    render_header()

    processed_pdf: dict[str, Any] | None = st.session_state.get("processed_pdf")

    st.subheader("上传 PDF")
    uploaded_file = st.file_uploader(
        "选择一篇 PDF 论文",
        type=["pdf"],
        accept_multiple_files=False,
    )

    if uploaded_file is None:
        st.info("上传 PDF 后，点击开始解析；解析完成后页面会显示完整 Markdown，右侧可以进行论文问答。")
        return

    signature = get_uploaded_file_signature(uploaded_file)
    cached_pdf = processed_pdf if processed_pdf and processed_pdf.get("signature") == signature else None
    st.caption(f"已选择文件：{uploaded_file.name}，大小：{format_file_size(len(uploaded_file.getvalue()))}")

    if cached_pdf:
        processed_pdf = cached_pdf
    else:
        processed_pdf = None
        st.info("PDF 已上传到页面，点击下方按钮后开始解析。MinerU 解析可能需要数十秒到数分钟。")
        if st.button("开始解析 PDF", type="primary", use_container_width=True, key="start_pdf_parse"):
            try:
                with st.spinner("正在保存 PDF 并转换为 Markdown..."):
                    processed_pdf = process_uploaded_pdf(uploaded_file)
            except (UploadError, PdfParseError, MinerUError) as exc:
                logger.exception("PDF upload or parse failed.")
                st.error(f"{exc.message}（错误码：{exc.code.value}）")
                processed_pdf = None

    if not processed_pdf:
        return

    if processed_pdf["db_save_failed"]:
        st.warning("数据保存失败，但不影响当前解析结果，请检查 SQLite 数据库权限。")

    render_processed_pdf_summary(processed_pdf)
    render_index_builder(processed_pdf["chunks"])

    st.divider()
    left_col, right_col = st.columns([1.12, 0.88], gap="large")
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


def render_card_visual(card: dict[str, Any]) -> None:
    """Render one saved literature card as a compact visual card."""
    title = escaped_text(card.get("title"))
    authors = escaped_text(card.get("authors"))
    year = escaped_text(card.get("year"))
    field = escaped_text(card.get("research_field"))
    library_name = escaped_text(card.get("library_name") or "未分组")
    question = escaped_text(card.get("research_question"))
    method = escaped_text(card.get("method_summary"))
    datasets = escaped_text(card.get("datasets"))
    palette = card_palette(card)

    st.markdown(
        f"""
        <div
          class="pm-card"
          style="--pm-top: {palette['top']}; --pm-accent: {palette['accent']}; --pm-field-bg: {palette['field']};"
        >
          <div class="pm-card-top">
            <div class="pm-card-title">{title}</div>
            <div class="pm-meta">
              <span class="pm-chip">作者：{authors}</span>
              <span class="pm-chip">年份：{year}</span>
              <span class="pm-chip">领域：{field}</span>
              <span class="pm-chip">库：{library_name}</span>
            </div>
          </div>
          <div class="pm-card-body">
            <div class="pm-field">
              <div class="pm-field-label">研究问题</div>
              <div class="pm-field-value">{question}</div>
            </div>
            <div class="pm-field">
              <div class="pm-field-label">方法概述</div>
              <div class="pm-field-value">{method}</div>
            </div>
            <div class="pm-field">
              <div class="pm-field-label">实验数据集</div>
              <div class="pm-field-value">{datasets}</div>
            </div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_pdf_viewer(save_path: str | None) -> None:
    """Render a PDF viewer for a locally saved paper."""
    if not save_path:
        st.warning("没有找到该论文的 PDF 保存路径。")
        return

    pdf_path = Path(save_path)
    if not pdf_path.exists():
        st.warning(f"PDF 文件不存在：{pdf_path}")
        return

    pdf_bytes = pdf_path.read_bytes()
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


def render_card_edit_form(card: dict[str, Any], user_id: int) -> None:
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
            update_literature_card(int(card["card_id"]), values, user_id=user_id)
        except (OSError, sqlite3.Error):
            st.error("文献卡片更新失败，请检查 SQLite 数据库权限。")
            return
        st.success("文献卡片已更新。")
        st.rerun()


def render_card_delete(card_id: int, user_id: int) -> None:
    """Render delete confirmation controls."""
    confirm = st.checkbox("确认删除这张文献卡片", key=f"confirm_delete_{card_id}")
    if st.button("删除文献卡片", disabled=not confirm, use_container_width=True):
        try:
            delete_literature_card(card_id, user_id=user_id)
        except (OSError, sqlite3.Error):
            st.error("文献卡片删除失败，请检查 SQLite 数据库权限。")
            return
        st.success("文献卡片已删除。")
        st.rerun()


def library_option_label(library: dict[str, Any]) -> str:
    """Build a readable label for a card library option."""
    return f"{library['name']}（{int(library.get('card_count') or 0)} 张）"


def render_library_create_form(user_id: int, key_suffix: str = "") -> None:
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
            create_card_library(user_id, new_library_name)
        except sqlite3.IntegrityError:
            st.error("这个卡片库名字已经存在。")
        except (ValueError, OSError, sqlite3.Error) as exc:
            st.error(str(exc) or "卡片库创建失败。")
        else:
            st.success("卡片库已创建。")
            st.rerun()


def render_library_rename_form(user_id: int, library: dict[str, Any]) -> None:
    """Render a form that renames a user-owned card library."""
    with st.form(key=f"rename_library_{library['library_id']}"):
        new_name = st.text_input("新的卡片库名称", value=str(library["name"]))
        submitted = st.form_submit_button("保存名称", type="primary", use_container_width=True)

    if submitted:
        try:
            update_card_library(int(library["library_id"]), user_id, new_name)
        except sqlite3.IntegrityError:
            st.error("这个卡片库名字已经存在。")
        except (ValueError, OSError, sqlite3.Error) as exc:
            st.error(str(exc) or "卡片库重命名失败。")
        else:
            st.success("卡片库名称已更新。")
            st.rerun()


def render_card_library_page(user_id: int) -> None:
    """Render saved literature-card management page."""
    render_header()
    st.subheader("文献卡片库")

    try:
        libraries = list_card_libraries(user_id)
    except (OSError, sqlite3.Error):
        st.error("卡片库读取失败，请检查 SQLite 数据库权限。")
        return

    library_count = len(libraries)
    card_total = sum(int(library.get("card_count") or 0) for library in libraries)
    st.markdown(
        f"""
        <div class="pm-library-panel">
          <h3>你的卡片，只归你管</h3>
          <p>当前共有 {library_count} 个卡片库、{card_total} 张文献卡片。可以按课程、课题或论文阶段拆成不同库，保存时直接选目标库。</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    library_options = [0, *[int(library["library_id"]) for library in libraries]]
    selected_library_id = st.selectbox(
        "查看范围",
        options=library_options,
        format_func=lambda library_id: "全部卡片库" if int(library_id) == 0 else library_option_label(
            next(library for library in libraries if int(library["library_id"]) == int(library_id))
        ),
    )
    library_filter = None if int(selected_library_id) == 0 else int(selected_library_id)
    selected_library = (
        next((library for library in libraries if int(library["library_id"]) == library_filter), None)
        if library_filter
        else None
    )

    tools_col, rename_col = st.columns([0.48, 0.52], gap="large")
    with tools_col:
        with st.expander("创建新卡片库", expanded=False):
            render_library_create_form(user_id, key_suffix="_library_page")
    with rename_col:
        if selected_library:
            with st.expander("重命名当前卡片库", expanded=False):
                render_library_rename_form(user_id, selected_library)
        else:
            st.caption("选择某个具体卡片库后，可以在这里修改它的名字。")

    cards = list_literature_cards(user_id=user_id, library_id=library_filter)
    if not cards:
        st.info("这个范围里还没有文献卡片。去“论文工作台”上传 PDF，生成卡片时选一个目标库即可。")
        return

    scope_name = selected_library["name"] if selected_library else "全部卡片库"
    st.caption(f"「{scope_name}」共 {len(cards)} 张文献卡片。支持多选批量删除，也可以选择单张卡片查看、修改和打开对应 PDF。")

    with st.expander("批量管理", expanded=False):
        selected_batch_ids = st.multiselect(
            "选择要删除的文献卡片",
            options=[int(card["card_id"]) for card in cards],
            format_func=lambda card_id: card_option_label(
                next(card for card in cards if int(card["card_id"]) == int(card_id))
            ),
        )
        confirm_batch_delete = st.checkbox("确认批量删除选中的文献卡片")
        if st.button(
            "批量删除",
            disabled=not selected_batch_ids or not confirm_batch_delete,
            use_container_width=True,
        ):
            try:
                deleted_count = delete_literature_cards(selected_batch_ids, user_id=user_id)
            except (OSError, sqlite3.Error):
                st.error("批量删除失败，请检查 SQLite 数据库权限。")
                return
            st.success(f"已删除 {deleted_count} 张文献卡片。")
            st.rerun()

    with st.expander("卡片概览", expanded=True):
        grid_columns = st.columns(2)
        for index, card in enumerate(cards):
            with grid_columns[index % 2]:
                render_card_visual(card)

    list_col, detail_col = st.columns([0.38, 0.62], gap="large")
    selected_card: dict[str, Any] | None = None

    with list_col:
        st.markdown("#### 单张管理")
        selected_card_id = st.selectbox(
            "选择文献卡片",
            options=[int(card["card_id"]) for card in cards],
            format_func=lambda card_id: card_option_label(
                next(card for card in cards if int(card["card_id"]) == int(card_id))
            ),
            label_visibility="collapsed",
        )
        selected_card = get_literature_card(int(selected_card_id), user_id=user_id)
        if selected_card:
            st.caption(f"所属卡片库：{selected_card.get('library_name') or '未分组'}")
            st.caption(f"paper_id：{selected_card['paper_id']}")
            st.caption(f"更新时间：{selected_card['updated_at']}")
            st.caption(f"PDF：{selected_card.get('file_name') or '未关联'}")

    if not selected_card:
        st.warning("没有找到选中的文献卡片。")
        return

    with detail_col:
        tab_card, tab_edit, tab_pdf = st.tabs(["卡片预览", "修改", "完整 PDF"])

        with tab_card:
            render_card_visual(selected_card)
            with st.expander("Markdown 内容", expanded=False):
                st.text_area(
                    "Markdown",
                    selected_card["markdown"],
                    height=360,
                    key=f"card_markdown_{selected_card['card_id']}",
                )

        with tab_edit:
            render_card_edit_form(selected_card, user_id)
            st.divider()
            render_card_delete(int(selected_card["card_id"]), user_id)

        with tab_pdf:
            render_pdf_viewer(selected_card.get("save_path"))


def feedback_admin_password() -> str:
    """Return the feedback-page administrator password."""
    return (os.getenv("PAPERMATE_ADMIN_PASSWORD") or settings.app_password or "").strip()


def require_feedback_admin_password() -> bool:
    """Require an administrator password before showing feedback records."""
    admin_password = feedback_admin_password()
    if not admin_password:
        st.error("反馈记录页需要管理员密码，请配置 PAPERMATE_ADMIN_PASSWORD 或 PAPERMATE_APP_PASSWORD。")
        return False

    if st.session_state.get("feedback_admin_authenticated"):
        return True

    st.subheader("管理员验证")
    password = st.text_input("管理员密码", type="password", key="feedback_admin_password")
    if st.button("查看反馈记录", type="primary", use_container_width=True):
        if hmac.compare_digest(password, admin_password):
            st.session_state["feedback_admin_authenticated"] = True
            st.rerun()
        else:
            st.error("管理员密码不正确。")

    return False


def render_feedback_records_page() -> None:
    """Render saved user feedback and bad cases."""
    render_header()
    st.subheader("反馈记录")
    if not require_feedback_admin_password():
        return

    feedback_rows = list_feedback_records()
    bad_case_rows = list_bad_cases()

    col_a, col_b, col_c = st.columns(3)
    col_a.metric("反馈总数", len(feedback_rows))
    col_b.metric("负面反馈", sum(1 for row in feedback_rows if int(row.get("is_negative") or 0)))
    col_c.metric("Bad Case", len(bad_case_rows))

    tab_feedback, tab_bad_cases = st.tabs(["用户反馈", "Bad Case"])

    with tab_feedback:
        if not feedback_rows:
            st.info("暂无用户反馈。")
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

            for row in feedback_rows:
                title = f"#{row['feedback_id']} · {row.get('feedback_type') or ''} · {row.get('file_name') or row.get('paper_id') or '未关联论文'}"
                with st.expander(title):
                    st.write("问题：", row.get("question") or "无")
                    st.write("回答：", row.get("answer") or "无")
                    st.write("补充说明：", row.get("comment") or "无")
                    st.caption(f"提交时间：{row.get('created_at')}")

    with tab_bad_cases:
        if not bad_case_rows:
            st.info("暂无 Bad Case。")
        else:
            st.dataframe(
                bad_case_rows,
                use_container_width=True,
                hide_index=True,
                column_config={
                    "bad_case_id": "Bad Case ID",
                    "paper_id": "paper_id",
                    "file_name": "论文文件",
                    "question": "问题",
                    "answer": "回答",
                    "error_type": "错误类型",
                    "reason": "原因",
                    "solution": "解决方案",
                    "status": "状态",
                    "notes": "备注",
                    "created_at": "创建时间",
                },
            )

            for row in bad_case_rows:
                title = f"#{row['bad_case_id']} · {row.get('error_type') or ''} · {row.get('status') or ''}"
                with st.expander(title):
                    st.write("问题：", row.get("question") or "无")
                    st.write("回答：", row.get("answer") or "无")
                    st.write("备注：", row.get("notes") or "无")
                    st.caption(f"论文：{row.get('file_name') or row.get('paper_id') or '未关联'}")


def render_app() -> None:
    """Render the PaperMate app."""
    st.set_page_config(page_title=settings.app_name, layout="wide", initial_sidebar_state="expanded")
    inject_styles()

    init_db()
    render_welcome_dialog()

    user = current_user()
    if not user:
        render_auth_page()
        return

    prepare_user_workspace(int(user["user_id"]))
    page = render_sidebar_navigation(user)

    if page == "论文工作台":
        render_workspace_page()
    elif page == "文献卡片库":
        render_card_library_page(int(user["user_id"]))
    else:
        render_feedback_records_page()


if __name__ == "__main__":
    render_app()
