"""Build LLM context and system-owned citations from retrieved chunks."""

from __future__ import annotations

from typing import Any

from src.retrieval.tokenizer import tokenize_text


UNKNOWN_PAGE = "未知页"
UNKNOWN_SECTION = "未知章节"


def build_context(
    chunks: list[dict],
    max_chars: int = 9000,
) -> tuple[str, list[dict]]:
    """Convert final retrieval chunks into prompt context and citations.

    Core chunks are kept before expanded neighbors. The function only includes
    complete snippets; neighbor chunks are skipped first when the budget is
    tight, so they never displace core retrieval hits.
    """
    if not chunks or max_chars <= 0:
        return "", []

    context_parts: list[str] = []
    citations: list[dict[str, Any]] = []
    current_len = 0

    for chunk in _dedupe_chunks(chunks):
        citation_id = len(citations) + 1
        chunk_id = _clean_value(chunk.get("chunk_id"), "")
        page_label = _format_page(chunk.get("page_num"))
        section_title = _clean_value(chunk.get("section_title"), UNKNOWN_SECTION)
        text = _clean_value(chunk.get("text"), "")
        expanded_neighbor = bool(chunk.get("expanded_neighbor", False))

        neighbor_note = ""
        if expanded_neighbor:
            parent = _clean_value(chunk.get("parent_chunk_id"), "")
            neighbor_note = f" | 邻近扩展={parent}" if parent else " | 邻近扩展=true"

        snippet = (
            f"[片段{citation_id} | chunk_id={chunk_id} | 页码={page_label} | 章节={section_title}{neighbor_note}]\n"
            f"{text}"
        )
        candidate = snippet if not context_parts else f"\n\n---\n\n{snippet}"
        if current_len + len(candidate) > max_chars:
            if expanded_neighbor:
                continue
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
                "chunk_type": chunk.get("chunk_type", "text"),
                "images_json": chunk.get("images_json", "[]"),
                "tables_json": chunk.get("tables_json", "[]"),
                "text_preview": text[:300],
                "rrf_score": chunk.get("rrf_score"),
                "rerank_score": chunk.get("rerank_score"),
                "final_score": chunk.get("final_score"),
                "section_boost": chunk.get("section_boost"),
                "expanded_neighbor": expanded_neighbor,
                "parent_chunk_id": chunk.get("parent_chunk_id", ""),
                "index_version": chunk.get("index_version", ""),
                "retrieval_sources": _as_list(chunk.get("retrieval_sources")),
                "source_ranks": _as_dict(chunk.get("source_ranks")),
                "vector_distance": chunk.get("vector_distance"),
                "bm25_score": chunk.get("bm25_score"),
            }
        )

    return "\n\n---\n\n".join(context_parts), citations


def _dedupe_chunks(chunks: list[dict]) -> list[dict]:
    deduped: list[dict] = []
    seen_ids: set[str] = set()
    seen_texts: list[str] = []

    for chunk in chunks:
        chunk_id = _clean_value(chunk.get("chunk_id"), "")
        if chunk_id and chunk_id in seen_ids:
            continue
        text = _clean_value(chunk.get("text"), "")
        if text and any(_highly_similar(text, existing) for existing in seen_texts):
            continue
        if chunk_id:
            seen_ids.add(chunk_id)
        if text:
            seen_texts.append(text)
        deduped.append(chunk)
    return deduped


def _highly_similar(left: str, right: str) -> bool:
    left_norm = " ".join(left.lower().split())
    right_norm = " ".join(right.lower().split())
    if len(left_norm) >= 80 and len(right_norm) >= 80:
        shorter, longer = sorted((left_norm, right_norm), key=len)
        if shorter in longer:
            return True

    left_tokens = set(tokenize_text(left_norm))
    right_tokens = set(tokenize_text(right_norm))
    if not left_tokens or not right_tokens:
        return False
    intersection = len(left_tokens & right_tokens)
    union = len(left_tokens | right_tokens)
    return union > 0 and intersection / union >= 0.92


def _format_page(page_num: Any) -> str:
    page = _clean_value(page_num, "")
    if not page:
        return UNKNOWN_PAGE
    if "页" in page:
        return page
    return f"第{page}页"


def _clean_value(value: Any, default: str = "") -> str:
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
