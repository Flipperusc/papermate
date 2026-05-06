"""Build LLM context and system-owned citations from retrieved chunks."""

from __future__ import annotations

from typing import Any


UNKNOWN_PAGE = "未知页"
UNKNOWN_SECTION = "未知章节"


def build_context(
    chunks: list[dict],
    max_chars: int = 6000,
) -> tuple[str, list[dict]]:
    """Convert final retrieval chunks into prompt context and citations.

    The function only includes complete snippets. If adding the next snippet
    would exceed max_chars, it stops instead of truncating snippet metadata.
    """
    if not chunks or max_chars <= 0:
        return "", []

    context_parts: list[str] = []
    citations: list[dict[str, Any]] = []
    current_len = 0

    for chunk in chunks:
        citation_id = len(citations) + 1
        chunk_id = _clean_value(chunk.get("chunk_id"), "")
        page_label = _format_page(chunk.get("page_num"))
        section_title = _clean_value(chunk.get("section_title"), UNKNOWN_SECTION)
        text = _clean_value(chunk.get("text"), "")

        snippet = (
            f"[片段{citation_id} | chunk_id={chunk_id} | 页码={page_label} | 章节={section_title}]\n"
            f"{text}"
        )
        candidate = snippet if not context_parts else f"\n\n---\n\n{snippet}"
        if current_len + len(candidate) > max_chars:
            break

        context_parts.append(snippet)
        current_len += len(candidate)
        # The UI trusts this citation list, not model-written source text.
        citations.append(
            {
                "citation_id": citation_id,
                "chunk_id": chunk_id,
                "paper_id": chunk.get("paper_id", ""),
                "chunk_index": chunk.get("chunk_index", ""),
                "page_num": page_label,
                "section_title": section_title,
                "text_preview": text[:300],
                "rrf_score": chunk.get("rrf_score"),
                "retrieval_sources": _as_list(chunk.get("retrieval_sources")),
                "source_ranks": _as_dict(chunk.get("source_ranks")),
                "vector_distance": chunk.get("vector_distance"),
                "bm25_score": chunk.get("bm25_score"),
            }
        )

    return "\n\n---\n\n".join(context_parts), citations


def _format_page(page_num: Any) -> str:
    page = _clean_value(page_num, "")
    if not page:
        return UNKNOWN_PAGE
    if "页" in page:
        return page
    return f"第{page}页"


def _clean_value(value: Any, default: str) -> str:
    if value is None:
        return default
    text = str(value).strip()
    return text or default


def _as_list(value: Any) -> list:
    if isinstance(value, list):
        return list(value)
    if value is None:
        return []
    return [value]


def _as_dict(value: Any) -> dict:
    if isinstance(value, dict):
        return dict(value)
    return {}
