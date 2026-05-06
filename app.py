"""Streamlit entry point for PaperMate."""

from __future__ import annotations

import base64
import hashlib
import html
import hmac
import sqlite3
import re
from pathlib import Path
from typing import Any, Protocol
from uuid import uuid4

import streamlit as st

from config import settings
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
    delete_literature_card,
    delete_literature_cards,
    get_literature_card,
    get_literature_card_by_paper,
    list_literature_cards,
    save_literature_card,
    update_literature_card,
)
from src.logger import get_logger
from src.pdf_parser import parse_pdf
from src.rag_pipeline import answer_question
from src.retrieval.bm25_store import BM25Store
from src.vector_store import VectorStore


logger = get_logger(__name__)

MARKDOWN_PREVIEW_CHARS = 20000


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
        .block-container {
            padding-top: 1.2rem;
            padding-bottom: 2rem;
            max-width: 1500px;
        }
        .pm-hero {
            border: 0;
            border-radius: 8px;
            padding: 20px 22px;
            background: #eef6ff;
            margin-bottom: 16px;
            box-shadow: inset 0 0 0 1px #c7ddf5, 0 1px 2px rgba(15, 23, 42, 0.05);
        }
        .pm-hero h1 {
            margin: 0 0 4px 0;
            font-size: 30px;
            letter-spacing: 0;
        }
        .pm-hero p {
            margin: 0;
            color: #31506f;
        }
        .pm-card {
            border: 1px solid #e3e7ee;
            border-radius: 8px;
            padding: 18px 20px;
            background: #ffffff;
            box-shadow: 0 1px 2px rgba(15, 23, 42, 0.04);
            margin-bottom: 14px;
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
            border: 1px solid #d9e1ee;
            border-radius: 999px;
            padding: 4px 10px;
            font-size: 12px;
            color: #475569;
            background: #f8fafc;
        }
        .pm-field {
            margin-top: 12px;
        }
        .pm-field-label {
            font-size: 12px;
            color: #64748b;
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
        iframe.pm-pdf {
            border: 1px solid #d8dee9;
            border-radius: 8px;
            background: #f8fafc;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


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


def markdown_preview_text(markdown: str, limit: int = MARKDOWN_PREVIEW_CHARS) -> tuple[str, bool]:
    """Return a browser-safe Markdown preview without inline image payloads."""
    safe_markdown = re.sub(
        r"!\[[^\]]*\]\(data:image[^)]*\)",
        "[图片内容已省略，可在 MinerU 输出目录查看]",
        markdown or "",
        flags=re.IGNORECASE,
    )
    safe_markdown = re.sub(
        r"\[([^\]]*)\]\(data:image[^)]*\)",
        r"\1（图片内容已省略）",
        safe_markdown,
        flags=re.IGNORECASE,
    )
    truncated = len(safe_markdown) > limit
    return safe_markdown[:limit], truncated


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
    if cached_result and cached_result.get("signature") == signature:
        return cached_result

    saved_file = save_uploaded_pdf(uploaded_file)
    parsed_pdf = parse_pdf(saved_file["save_path"], saved_file["paper_id"])
    chunks = chunk_pages(saved_file["paper_id"], parsed_pdf["pages"])
    total_chars, preview = build_text_preview(parsed_pdf)

    db_save_failed = False
    try:
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


def render_upload_and_markdown(processed_pdf: dict[str, Any]) -> None:
    """Render uploaded file metadata and full markdown document."""
    saved_file = processed_pdf["saved_file"]
    parsed_pdf = processed_pdf["parsed_pdf"]
    markdown = parsed_pdf.get("markdown", "") or processed_pdf["preview"]
    preview_text, is_truncated = markdown_preview_text(markdown)

    st.success("PDF 已保存并完成 Markdown 解析。")
    col_a, col_b = st.columns(2)
    col_a.metric("页数", parsed_pdf["page_count"])
    col_b.metric("字符数", processed_pdf["total_chars"])

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

    st.markdown("#### 论文 Markdown")
    if is_truncated:
        st.info(f"当前仅显示前 {MARKDOWN_PREVIEW_CHARS} 个字符，完整内容请下载 Markdown 文件查看。")
    with st.container(height=720, border=True):
        st.text_area(
            "Markdown 预览",
            preview_text or "暂无 Markdown 内容",
            height=680,
            label_visibility="collapsed",
        )

    render_extracted_images(parsed_pdf.get("images", []))


def render_extracted_images(images: list[dict[str, str]]) -> None:
    """Render extracted paper images for direct inspection."""
    if not images:
        return

    with st.expander(f"论文图片（{len(images)} 张）", expanded=False):
        st.caption("图片已单独保存到 MinerU 输出目录；Markdown 中的“此处含有图 N”链接可打开对应原图。")
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


def render_literature_card_save(paper_id: str, chunks: list[dict[str, Any]], db_save_failed: bool) -> None:
    """Render literature card generation and persistence button."""
    st.markdown("#### 文献卡片")
    if not chunks:
        st.warning("论文正文为空，暂时无法生成文献卡片。")
        return
    if db_save_failed:
        st.warning("数据库保存失败，文献卡片可能无法基于完整 chunks 生成。")

    existing_card = get_literature_card_by_paper(paper_id)
    if st.button("生成并保存为新文献卡片", type="primary", use_container_width=True, key=f"save_card_{paper_id}"):
        try:
            with st.spinner("正在调用 DeepSeek 生成文献卡片并保存..."):
                markdown = generate_literature_card(paper_id)
                card_id = save_literature_card(paper_id, markdown)
        except (LLMError, OSError, sqlite3.Error) as exc:
            if isinstance(exc, LLMError):
                st.error(f"{exc.message}（错误码：{exc.code.value}）")
            else:
                st.error("文献卡片保存失败，请检查 SQLite 数据库权限。")
            return

        st.session_state[f"saved_card_id_{paper_id}"] = card_id
        st.success("新文献卡片已保存，可在“文献卡片库”页面查看、批量管理和编辑。")

    saved_card = get_literature_card_by_paper(paper_id)
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

    if st.button("构建论文索引（向量 + 关键词）", type="primary", use_container_width=True, key="build_paper_index"):
        paper_id = str(chunks[0].get("paper_id") or "")
        vector_count: int | None = None
        bm25_result: dict[str, Any] | None = None
        vector_error: Exception | None = None
        bm25_error: Exception | None = None

        with st.spinner("正在构建 Hybrid 索引：向量索引 + 关键词索引..."):
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
            st.success("论文索引构建完成，可使用 Hybrid RAG 问答。")
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
            with st.spinner("正在检索论文片段并调用 DeepSeek 生成回答..."):
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

    left_col, right_col = st.columns([1.1, 0.9], gap="large")
    processed_pdf: dict[str, Any] | None = st.session_state.get("processed_pdf")

    with left_col:
        st.subheader("上传 PDF 并转为 Markdown")
        uploaded_file = st.file_uploader(
            "选择一篇 PDF 论文",
            type=["pdf"],
            accept_multiple_files=False,
        )

        if uploaded_file is None:
            st.info("上传 PDF 后，点击开始解析；解析完成后左侧会显示 Markdown 预览，右侧可以构建索引并进行问答。")
        else:
            signature = get_uploaded_file_signature(uploaded_file)
            cached_pdf = processed_pdf if processed_pdf and processed_pdf.get("signature") == signature else None
            st.caption(f"已选择文件：{uploaded_file.name}，大小：{format_file_size(len(uploaded_file.getvalue()))}")

            if cached_pdf:
                processed_pdf = cached_pdf
            else:
                processed_pdf = None
                st.info("PDF 已上传到页面，点击下方按钮后才会开始解析。MinerU 解析可能需要数十秒到数分钟。")
                if st.button("开始解析 PDF", type="primary", use_container_width=True, key="start_pdf_parse"):
                    try:
                        with st.spinner("正在保存 PDF，并调用 MinerU 转换 Markdown..."):
                            processed_pdf = process_uploaded_pdf(uploaded_file)
                    except (UploadError, PdfParseError, MinerUError) as exc:
                        logger.exception("PDF upload or parse failed.")
                        st.error(f"{exc.message}（错误码：{exc.code.value}）")
                        processed_pdf = None

            if processed_pdf:
                if processed_pdf["db_save_failed"]:
                    st.warning("数据保存失败，但不影响当前解析结果，请检查 SQLite 数据库权限。")

                render_upload_and_markdown(processed_pdf)
                render_literature_card_save(
                    processed_pdf["saved_file"]["paper_id"],
                    processed_pdf["chunks"],
                    processed_pdf["db_save_failed"],
                )

    with right_col:
        if not processed_pdf:
            st.subheader("索引与问答")
            st.info("请先在左侧上传并解析 PDF。")
            return

        render_index_builder(processed_pdf["chunks"])
        render_qa_box(processed_pdf["saved_file"]["paper_id"])


def escaped_text(value: Any) -> str:
    """Escape text for safe HTML rendering."""
    return html.escape(str(value or "原文未明确说明")).replace("\n", "<br>")


def render_card_visual(card: dict[str, Any]) -> None:
    """Render one saved literature card as a compact visual card."""
    title = escaped_text(card.get("title"))
    authors = escaped_text(card.get("authors"))
    year = escaped_text(card.get("year"))
    field = escaped_text(card.get("research_field"))
    question = escaped_text(card.get("research_question"))
    method = escaped_text(card.get("method_summary"))
    datasets = escaped_text(card.get("datasets"))

    st.markdown(
        f"""
        <div class="pm-card">
          <div class="pm-card-title">{title}</div>
          <div class="pm-meta">
            <span class="pm-chip">作者：{authors}</span>
            <span class="pm-chip">年份：{year}</span>
            <span class="pm-chip">领域：{field}</span>
          </div>
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
    file_name = str(card.get("file_name") or "PDF 未关联").strip()
    return f"{title} · {year} · {file_name}"


def render_card_edit_form(card: dict[str, Any]) -> None:
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
            update_literature_card(int(card["card_id"]), values)
        except (OSError, sqlite3.Error):
            st.error("文献卡片更新失败，请检查 SQLite 数据库权限。")
            return
        st.success("文献卡片已更新。")
        st.rerun()


def render_card_delete(card_id: int) -> None:
    """Render delete confirmation controls."""
    confirm = st.checkbox("确认删除这张文献卡片", key=f"confirm_delete_{card_id}")
    if st.button("删除文献卡片", disabled=not confirm, use_container_width=True):
        try:
            delete_literature_card(card_id)
        except (OSError, sqlite3.Error):
            st.error("文献卡片删除失败，请检查 SQLite 数据库权限。")
            return
        st.success("文献卡片已删除。")
        st.rerun()


def render_card_library_page() -> None:
    """Render saved literature-card management page."""
    render_header()
    st.subheader("文献卡片库")

    cards = list_literature_cards()
    if not cards:
        st.info("还没有保存文献卡片。请先在“论文工作台”上传 PDF，并点击“生成并保存文献卡片”。")
        return

    st.caption(f"共 {len(cards)} 张文献卡片。支持多选批量删除，也可以选择单张卡片查看、修改和打开对应 PDF。")

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
                deleted_count = delete_literature_cards(selected_batch_ids)
            except (OSError, sqlite3.Error):
                st.error("批量删除失败，请检查 SQLite 数据库权限。")
                return
            st.success(f"已删除 {deleted_count} 张文献卡片。")
            st.rerun()

    with st.expander("全部卡片概览", expanded=True):
        grid_columns = st.columns(2)
        for index, card in enumerate(cards):
            with grid_columns[index % 2]:
                render_card_visual(card)

    list_col, detail_col = st.columns([0.38, 0.62], gap="large")

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
        selected_card = get_literature_card(int(selected_card_id))
        if selected_card:
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
            render_card_edit_form(selected_card)
            st.divider()
            render_card_delete(int(selected_card["card_id"]))

        with tab_pdf:
            render_pdf_viewer(selected_card.get("save_path"))


def require_app_password() -> bool:
    """Gate the app with an optional password for cloud deployments."""
    if not settings.app_password:
        return True

    if st.session_state.get("app_authenticated"):
        return True

    st.title(settings.app_name)
    st.caption("请输入访问密码。")
    password = st.text_input("访问密码", type="password")
    if st.button("进入", type="primary"):
        if hmac.compare_digest(password, settings.app_password):
            st.session_state["app_authenticated"] = True
            st.rerun()
        else:
            st.error("访问密码不正确。")

    return False


def render_feedback_records_page() -> None:
    """Render saved user feedback and bad cases."""
    render_header()
    st.subheader("反馈记录")

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
    st.set_page_config(page_title=settings.app_name, layout="wide")
    inject_styles()
    if not require_app_password():
        return

    init_db()

    page = st.sidebar.radio(
        "页面",
        ["论文工作台", "文献卡片库", "反馈记录"],
        label_visibility="collapsed",
    )

    st.sidebar.markdown("### PaperMate")
    st.sidebar.caption("PDF 转 Markdown · RAG 问答 · 文献卡片")

    if page == "论文工作台":
        render_workspace_page()
    elif page == "文献卡片库":
        render_card_library_page()
    else:
        render_feedback_records_page()


if __name__ == "__main__":
    render_app()
