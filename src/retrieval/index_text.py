"""Helpers for enriched retrieval index text."""

from __future__ import annotations

from typing import Any

from src.retrieval.query_planner import extract_entities, unique_preserve_order


def build_enriched_chunk_text(chunk: dict[str, Any]) -> str:
    """Return text used for vector embedding.

    The original chunk body stays dominant, while section and page metadata give
    the embedding model enough anchors for table, section, and page questions.
    """
    text = _clean(chunk.get("text"))
    section_title = _clean(chunk.get("section_title"))
    page_num = _clean(chunk.get("page_num"))
    entities = extract_index_entities(chunk)

    parts: list[str] = []
    if section_title:
        parts.append(f"Section: {section_title}")
    if page_num:
        parts.append(f"Page: {page_num}")
    if entities:
        parts.append(f"Entities: {' '.join(entities)}")
    parts.append(text)
    return "\n".join(part for part in parts if part)


def build_bm25_search_text(chunk: dict[str, Any]) -> str:
    """Return weighted text used for BM25 indexing.

    BM25 has no native field weights in this lightweight store, so important
    metadata is repeated to make exact section/entity matches more competitive.
    """
    text = _clean(chunk.get("text"))
    section_title = _clean(chunk.get("section_title"))
    page_num = _clean(chunk.get("page_num"))
    entities = extract_index_entities(chunk)

    weighted_parts: list[str] = []
    if section_title:
        weighted_parts.extend([section_title, section_title, f"section {section_title}"])
    if page_num:
        weighted_parts.append(f"page {page_num}")
    if entities:
        entity_text = " ".join(entities)
        weighted_parts.extend([entity_text, entity_text])
    weighted_parts.append(text)
    return "\n".join(part for part in weighted_parts if part)


def extract_index_entities(chunk: dict[str, Any]) -> list[str]:
    """Extract exact-match entities from chunk metadata and body text."""
    fields = [
        _clean(chunk.get("section_title")),
        _clean(chunk.get("text")),
    ]
    entities: list[str] = []
    for field in fields:
        entities.extend(extract_entities(field))
    return unique_preserve_order(entities)


def _clean(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()
