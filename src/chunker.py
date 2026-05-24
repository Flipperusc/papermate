"""Semantic, multimodal, and table-aware chunking utilities."""

from __future__ import annotations

import json
import math
import re
from collections import defaultdict
from html.parser import HTMLParser
from typing import Any

from config import settings
from src.embedding_client import EmbeddingClient
from src.logger import get_logger
from src.vlm_client import QwenVLMClient


CHUNKER_VERSION = "semantic-multimodal-v2-qwen-vlm"
DEFAULT_CHUNK_SIZE = 512
DEFAULT_OVERLAP = 100
logger = get_logger(__name__)

SECTION_ALIASES = {
    "abstract": "Abstract",
    "summary": "Abstract",
    "introduction": "Introduction",
    "related work": "Related Work",
    "related works": "Related Work",
    "background": "Related Work",
    "preliminaries": "Related Work",
    "method": "Method",
    "methods": "Method",
    "materials and methods": "Method",
    "methodology": "Method",
    "approach": "Method",
    "proposed method": "Method",
    "model": "Method",
    "experiments": "Experiments",
    "experiment": "Experiments",
    "experimental results": "Experiments",
    "experimental setup": "Experiments",
    "results": "Results",
    "results and discussion": "Results",
    "discussion": "Results",
    "analysis": "Results",
    "limitations": "Results",
    "conclusion": "Conclusion",
    "conclusions": "Conclusion",
    "conclusion and future work": "Conclusion",
    "future work": "Conclusion",
    "references": "References",
    "bibliography": "References",
    "acknowledgements": "Acknowledgements",
    "acknowledgments": "Acknowledgements",
    "appendix": "Appendix",
}

SECTION_PREFIXES = sorted(SECTION_ALIASES, key=len, reverse=True)
SECTION_PREFIX_PATTERN = re.compile(
    r"^(?P<prefix>(?:\d+(?:\.\d+)*[\.)]?\s*|[ivxlcdm]+[\.)]\s*|[ivxlcdm]+\s+)?)"
    r"(?P<title>"
    + "|".join(re.escape(title) for title in SECTION_PREFIXES)
    + r")"
    r"(?P<sep>\s*[:.\-\u2013\u2014]?\s+|\s*[:.\-\u2013\u2014]\s*|$)"
    r"(?P<body>.*)$",
    flags=re.IGNORECASE,
)
GENERIC_NUMBERED_HEADING_PATTERN = re.compile(
    r"^(?P<number>\d+(?:\.\d+)*)(?:[\.)])?\s+(?P<title>.+)$"
)
NUMERIC_ONLY_HEADING_PATTERN = re.compile(r"^\d+(?:\.\d+)*$")
GENERIC_HEADING_REJECT_PREFIXES = (
    "arxiv",
    "http",
    "www",
    "table ",
    "figure ",
    "fig. ",
    "appendix table",
)


def strip_markdown_heading_marker(text: str) -> str:
    """Remove Markdown heading markers before section detection."""
    return re.sub(r"^\s{0,3}#{1,6}\s+", "", text.strip()).strip()


def detect_section_title(text: str) -> str | None:
    """Return a normalized section title when text looks like a paper heading."""
    candidate = re.sub(r"\s+", " ", strip_markdown_heading_marker(text))
    if not candidate or len(candidate) > 140:
        return None

    unnumbered_candidate = re.sub(
        r"^(?:\d+(?:\.\d+)*[\.)]?\s*|[ivxlcdm]+[\.)]\s*|[ivxlcdm]+\s+)",
        "",
        candidate,
        flags=re.IGNORECASE,
    )
    normalized = unnumbered_candidate.strip(" .:-\u2013\u2014").lower()
    if normalized in SECTION_ALIASES:
        return SECTION_ALIASES[normalized]

    return detect_generic_numbered_heading(candidate)


def detect_generic_numbered_heading(text: str) -> str | None:
    """Detect non-standard numbered paper headings and keep their title."""
    candidate = re.sub(r"\s+", " ", strip_markdown_heading_marker(text))
    match = GENERIC_NUMBERED_HEADING_PATTERN.match(candidate)
    if not match:
        return None

    title = match.group("title").strip(" .:-\u2013\u2014")
    normalized_title = title.lower()
    if not title or len(title) > 120:
        return None
    if len(title) == 1:
        return None
    if not re.search(r"[A-Za-z]", title):
        return None
    if not title[0].isalpha() or not title[0].isupper():
        return None
    if title.endswith((",", ";", ".")):
        return None
    if any(normalized_title.startswith(prefix) for prefix in GENERIC_HEADING_REJECT_PREFIXES):
        return None
    if "@" in title or "http" in normalized_title:
        return None
    if "..." in title or "\u2026" in title:
        return None
    if "[" in title or "]" in title or "answer is" in normalized_title:
        return None
    if re.search(r"\b\d+\.\s+\w+", title):
        return None
    if len(title.split()) == 1 and re.search(r"[\d_/-]", title):
        return None
    if len(title.split()) > 10:
        return None

    return SECTION_ALIASES.get(normalized_title, title)


def split_section_prefix(text: str) -> tuple[str | None, str]:
    """Split section heading prefixes from text when extraction merged lines."""
    candidate = re.sub(r"\s+", " ", strip_markdown_heading_marker(text))
    if not candidate:
        return None, ""

    match = SECTION_PREFIX_PATTERN.match(candidate)
    if not match:
        return None, text.strip()

    title = SECTION_ALIASES[match.group("title").lower()]
    body = match.group("body").strip()
    return title, body


def split_page_paragraphs(text: str) -> list[str]:
    """Split page text into paragraph-like blocks."""
    paragraphs: list[str] = []
    for block in re.split(r"\n\s*\n+", text.strip()):
        lines = [line.strip() for line in block.splitlines() if line.strip()]
        if not lines:
            continue
        paragraphs.extend(split_lines_by_headings(lines))

    return paragraphs


def split_lines_by_headings(lines: list[str]) -> list[str]:
    """Split text lines whenever a likely paper heading appears."""
    paragraphs: list[str] = []
    current_lines: list[str] = []
    index = 0

    def flush_current() -> None:
        if current_lines:
            paragraphs.append(" ".join(current_lines).strip())
            current_lines.clear()

    while index < len(lines):
        line = lines[index].strip()
        if not line:
            index += 1
            continue

        heading = detect_section_title(line)
        heading_paragraph = line
        skip_next = False

        if not heading and NUMERIC_ONLY_HEADING_PATTERN.match(strip_markdown_heading_marker(line)):
            if index + 1 < len(lines):
                combined_heading = f"{line} {lines[index + 1].strip()}"
                heading = detect_section_title(combined_heading)
                if heading:
                    heading_paragraph = combined_heading
                    skip_next = True

        if heading:
            flush_current()
            paragraphs.append(heading_paragraph)
            index += 2 if skip_next else 1
            continue

        section_title, body = split_section_prefix(line)
        if section_title:
            flush_current()
            paragraphs.append(section_title)
            if body:
                current_lines.append(body)
            index += 1
            continue

        current_lines.append(line)
        index += 1

    flush_current()
    return paragraphs


def split_sentences(text: str) -> list[str]:
    """Split text into sentence-like units while keeping punctuation."""
    sentences = [
        match.group(0).strip()
        for match in re.finditer(r"[^.!?\u3002\uff01\uff1f]+[.!?\u3002\uff01\uff1f]*", text)
        if match.group(0).strip()
    ]
    return sentences or [text.strip()]


def split_by_words(text: str, max_size: int) -> list[str]:
    """Split long text by words, falling back to length only for unbroken text."""
    text = text.strip()
    if not text:
        return []
    words = text.split()
    if len(words) <= 1:
        return [text[index : index + max_size].strip() for index in range(0, len(text), max_size)]

    chunks: list[str] = []
    current: list[str] = []
    for word in words:
        if len(word) > max_size:
            if current:
                chunks.append(" ".join(current))
                current = []
            chunks.extend(split_by_words(word, max_size))
            continue

        candidate = " ".join([*current, word]) if current else word
        if len(candidate) <= max_size:
            current.append(word)
        else:
            chunks.append(" ".join(current))
            current = [word]

    if current:
        chunks.append(" ".join(current))
    return [chunk for chunk in chunks if chunk]


def chunk_pages(
    paper_id: str,
    pages: list[dict[str, Any]] | None,
    chunk_size: int | None = None,
    overlap: int | None = None,
    elements: list[dict[str, Any]] | None = None,
    embedding_client: Any | None = None,
    vlm_client: Any | None = None,
    table_large_row_chunk_size: int | None = None,
    table_wide_column_group_size: int | None = None,
) -> list[dict[str, Any]]:
    """Split parsed PDF pages/elements into semantic multimodal chunks."""
    safe_chunk_size = max(1, int(chunk_size or settings.rag_chunk_size or DEFAULT_CHUNK_SIZE))
    safe_overlap = max(0, min(int(overlap if overlap is not None else settings.rag_chunk_overlap), safe_chunk_size // 2))
    row_chunk_size = max(1, int(table_large_row_chunk_size or settings.table_large_row_chunk_size))
    column_group_size = max(1, int(table_wide_column_group_size or settings.table_wide_column_group_size))
    source_elements = elements if elements is not None else elements_from_pages(pages or [])
    return chunk_elements(
        paper_id=paper_id,
        elements=source_elements,
        chunk_size=safe_chunk_size,
        overlap=safe_overlap,
        embedding_client=embedding_client,
        vlm_client=vlm_client,
        row_chunk_size=row_chunk_size,
        column_group_size=column_group_size,
    )


def chunk_elements(
    paper_id: str,
    elements: list[dict[str, Any]],
    chunk_size: int,
    overlap: int,
    embedding_client: Any | None = None,
    vlm_client: Any | None = None,
    row_chunk_size: int = 20,
    column_group_size: int = 9,
) -> list[dict[str, Any]]:
    """Chunk ordered text/image/table elements."""
    chunks: list[dict[str, Any]] = []
    current_section = ""
    text_buffer: list[str] = []
    buffer_page: int | None = None
    buffer_section = ""
    active_vlm_client = vlm_client

    def get_vlm_client() -> Any:
        nonlocal active_vlm_client
        if active_vlm_client is None:
            active_vlm_client = QwenVLMClient()
        return active_vlm_client

    def flush_text_buffer() -> None:
        nonlocal text_buffer, buffer_page, buffer_section
        if not text_buffer:
            return
        for text in semantic_text_chunks(text_buffer, chunk_size, overlap, embedding_client):
            if not text.strip():
                continue
            chunks.append(
                make_chunk_payload(
                    paper_id=paper_id,
                    chunk_index=len(chunks),
                    page_num=buffer_page or 1,
                    section_title=buffer_section,
                    text=text,
                    chunk_type="text",
                )
            )
        text_buffer = []
        buffer_page = None
        buffer_section = ""

    for element in sorted(elements, key=lambda item: int(item.get("order", 0) or 0)):
        element_type = str(element.get("type") or "text").lower()
        page_num = safe_int(element.get("page_num"), 1)

        if element_type == "table":
            flush_text_buffer()
            for table_chunk in table_element_chunks(
                paper_id=paper_id,
                chunk_index_start=len(chunks),
                element=element,
                section_title=current_section,
                row_chunk_size=row_chunk_size,
                column_group_size=column_group_size,
            ):
                chunks.append(table_chunk)
            continue

        if element_type == "image":
            flush_text_buffer()
            bind_image_to_chunks(
                chunks=chunks,
                paper_id=paper_id,
                image_element=element,
                current_section=current_section,
                chunk_size=chunk_size,
                vlm_client=get_vlm_client(),
            )
            continue

        for paragraph in split_page_paragraphs(str(element.get("text") or "")):
            section_title = detect_section_title(paragraph)
            if section_title:
                flush_text_buffer()
                current_section = section_title
                continue

            section_title, body = split_section_prefix(paragraph)
            if section_title:
                flush_text_buffer()
                current_section = section_title
                paragraph = body
                if not paragraph:
                    continue

            if buffer_page is not None and (page_num != buffer_page or current_section != buffer_section):
                flush_text_buffer()
            if buffer_page is None:
                buffer_page = page_num
                buffer_section = current_section
            text_buffer.append(paragraph)

    flush_text_buffer()
    return [renumber_chunk(chunk, index, paper_id) for index, chunk in enumerate(chunks) if chunk["text"].strip()]


def elements_from_pages(pages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Build text-only elements from page records."""
    elements: list[dict[str, Any]] = []
    for order, page in enumerate(pages):
        text = str(page.get("text", "")).strip()
        if text:
            elements.append(
                {
                    "type": "text",
                    "order": order,
                    "page_num": safe_int(page.get("page_num"), 1),
                    "text": text,
                }
            )
    return elements


def semantic_text_chunks(
    paragraphs: list[str],
    chunk_size: int,
    overlap: int,
    embedding_client: Any | None = None,
) -> list[str]:
    """Split text by sentence embedding similarity and size-pack the results."""
    sentences = [sentence for paragraph in paragraphs for sentence in split_sentences(paragraph) if sentence]
    if not sentences:
        return []
    if len(sentences) == 1:
        return pack_sentence_segments([[sentences[0]]], chunk_size, overlap)

    client = embedding_client or EmbeddingClient()
    embeddings = client.embed(sentences)
    if len(embeddings) != len(sentences):
        raise ValueError("embedding count does not match sentence count")

    similarities = [
        cosine_similarity(embeddings[index], embeddings[index + 1])
        for index in range(len(embeddings) - 1)
    ]
    if not similarities:
        return pack_sentence_segments([sentences], chunk_size, overlap)

    mean = sum(similarities) / len(similarities)
    variance = sum((value - mean) ** 2 for value in similarities) / len(similarities)
    threshold = mean - math.sqrt(variance)

    segments: list[list[str]] = []
    current = [sentences[0]]
    for index, similarity in enumerate(similarities):
        if similarity < threshold:
            segments.append(current)
            current = [sentences[index + 1]]
        else:
            current.append(sentences[index + 1])
    if current:
        segments.append(current)

    return pack_sentence_segments(segments, chunk_size, overlap)


def pack_sentence_segments(
    segments: list[list[str]],
    chunk_size: int,
    overlap: int,
) -> list[str]:
    """Pack semantic sentence segments without crossing semantic boundaries."""
    chunks: list[str] = []
    current = ""
    previous = ""

    def flush_current() -> None:
        nonlocal current, previous
        if current.strip():
            chunks.append(current.strip())
            previous = current.strip()
            current = ""

    def start_text(piece: str) -> str:
        prefix = tail_overlap(previous, overlap)
        candidate = f"{prefix} {piece}".strip() if prefix else piece
        return candidate if len(candidate) <= chunk_size else piece

    for segment in segments:
        if current:
            flush_current()
        for sentence in segment:
            for piece in split_by_words(sentence, chunk_size):
                if not current:
                    current = start_text(piece)
                    continue
                candidate = f"{current} {piece}".strip()
                if len(candidate) <= chunk_size:
                    current = candidate
                else:
                    flush_current()
                    current = start_text(piece)
        flush_current()

    return chunks


def tail_overlap(text: str, overlap: int) -> str:
    """Return a sentence-aware overlap tail."""
    if overlap <= 0 or not text:
        return ""
    selected: list[str] = []
    total = 0
    for sentence in reversed(split_sentences(text)):
        extra = len(sentence) + (1 if selected else 0)
        if total + extra > overlap:
            continue
        selected.append(sentence)
        total += extra
    if selected:
        return " ".join(reversed(selected)).strip()
    return text[-overlap:].strip()


def cosine_similarity(left: list[float], right: list[float]) -> float:
    """Return cosine similarity for two embedding vectors."""
    if not left or not right or len(left) != len(right):
        return 0.0
    dot = sum(float(a) * float(b) for a, b in zip(left, right))
    left_norm = math.sqrt(sum(float(value) * float(value) for value in left))
    right_norm = math.sqrt(sum(float(value) * float(value) for value in right))
    if left_norm <= 0 or right_norm <= 0:
        return 0.0
    return dot / (left_norm * right_norm)


def bind_image_to_chunks(
    chunks: list[dict[str, Any]],
    paper_id: str,
    image_element: dict[str, Any],
    current_section: str,
    chunk_size: int,
    vlm_client: Any,
) -> None:
    """Bind an image description and metadata to the current chunk."""
    image_payload = image_metadata(image_element)
    description = image_description(image_payload, vlm_client)

    if chunks and chunks[-1].get("chunk_type") != "table":
        candidate = f"{chunks[-1]['text']}\n\n{description}".strip()
        if len(candidate) <= chunk_size:
            chunks[-1]["text"] = candidate
            chunks[-1]["chunk_type"] = "multimodal"
            images = list(chunks[-1].get("images") or [])
            images.append(image_payload)
            chunks[-1]["images"] = images
            chunks[-1]["images_json"] = json.dumps(images, ensure_ascii=False)
            return

    chunks.append(
        make_chunk_payload(
            paper_id=paper_id,
            chunk_index=len(chunks),
            page_num=safe_int(image_element.get("page_num"), 1),
            section_title=current_section,
            text=description,
            chunk_type="multimodal",
            images=[image_payload],
        )
    )


def image_metadata(element: dict[str, Any]) -> dict[str, Any]:
    """Return stable image metadata for chunk storage."""
    return {
        "kind": str(element.get("kind") or "image"),
        "label": str(element.get("label") or ""),
        "caption": str(element.get("caption") or ""),
        "alt_text": str(element.get("alt_text") or ""),
        "path": str(element.get("path") or ""),
        "mime_type": str(element.get("mime_type") or ""),
        "page_num": safe_int(element.get("page_num"), 1),
        "bbox": element.get("bbox") or [],
        "visual_id": str(element.get("visual_id") or ""),
        "source_paths": list(element.get("source_paths") or []),
    }


def image_description(image: dict[str, Any], vlm_client: Any) -> str:
    """Build the real VLM-backed image description used for retrieval."""
    try:
        vlm_description = " ".join(str(vlm_client.describe(image)).split())
    except Exception as exc:
        vlm_description = fallback_image_description(image)
        image["vlm_error"] = " ".join(str(exc).split())
        logger.warning(
            "VLM image description failed; using metadata fallback. "
            "kind=%s page=%s path=%s sources=%s error=%s",
            image.get("kind") or "image",
            image.get("page_num") or "",
            image.get("path") or "",
            image.get("source_paths") or [],
            exc,
        )
    image["vlm_description"] = vlm_description
    parts = [
        f"kind={image.get('kind') or 'image'}",
        f"caption={image.get('caption') or ''}",
        f"alt={image.get('alt_text') or ''}",
        f"page={image.get('page_num') or ''}",
        f"path={image.get('path') or ''}",
    ]
    bbox = image.get("bbox")
    if bbox:
        parts.append(f"bbox={bbox}")
    parts.append(f"vlm={vlm_description}")
    if image.get("vlm_error"):
        parts.append(f"vlm_error={image.get('vlm_error')}")
    return "[图片: " + "; ".join(parts) + "]"


def fallback_image_description(image: dict[str, Any]) -> str:
    """Build a metadata-only description when the VLM call cannot be used."""
    parts = [
        f"metadata-only {image.get('kind') or 'image'}",
        f"caption={image.get('caption') or ''}",
        f"alt={image.get('alt_text') or ''}",
        f"label={image.get('label') or ''}",
        f"page={image.get('page_num') or ''}",
        f"path={image.get('path') or ''}",
    ]
    source_paths = image.get("source_paths") or []
    if source_paths:
        parts.append(f"sources={source_paths}")
    bbox = image.get("bbox")
    if bbox:
        parts.append(f"bbox={bbox}")
    return "; ".join(parts)


def table_element_chunks(
    paper_id: str,
    chunk_index_start: int,
    element: dict[str, Any],
    section_title: str,
    row_chunk_size: int,
    column_group_size: int,
) -> list[dict[str, Any]]:
    """Build one or more table chunks for a table element."""
    rows = parse_table_rows(str(element.get("table_body") or ""))
    caption = str(element.get("caption") or "")
    if not rows:
        text = "\n".join(part for part in [f"Table: {caption}" if caption else "", str(element.get("table_body") or "")] if part)
        return [
            make_table_chunk(
                paper_id,
                chunk_index_start,
                element,
                section_title,
                text or "Table",
                {"mode": "raw", "caption": caption},
            )
        ]

    header, data_rows = split_header_rows(rows)
    row_count = len(data_rows)
    column_count = max((len(row) for row in rows), default=0)
    full_text = table_text(caption, header, data_rows, {"mode": "small", "total_rows": row_count})

    if column_count > 10:
        return wide_table_chunks(
            paper_id,
            chunk_index_start,
            element,
            section_title,
            caption,
            header,
            data_rows,
            column_group_size,
        )
    if row_count <= 20 and count_tokens(full_text) < 500:
        return [
            make_table_chunk(
                paper_id,
                chunk_index_start,
                element,
                section_title,
                full_text,
                {
                    "mode": "small",
                    "caption": caption,
                    "total_rows": row_count,
                    "total_columns": column_count,
                },
            )
        ]
    if row_count <= 100:
        category_index = find_category_column(header, data_rows)
        if category_index is not None:
            return grouped_table_chunks(
                paper_id,
                chunk_index_start,
                element,
                section_title,
                caption,
                header,
                data_rows,
                category_index,
            )
        return row_range_table_chunks(
            paper_id,
            chunk_index_start,
            element,
            section_title,
            caption,
            header,
            data_rows,
            row_chunk_size,
            "medium_rows",
        )

    return row_range_table_chunks(
        paper_id,
        chunk_index_start,
        element,
        section_title,
        caption,
        header,
        data_rows,
        row_chunk_size,
        "large_rows",
    )


def split_header_rows(rows: list[list[str]]) -> tuple[list[str], list[list[str]]]:
    """Return table header and data rows."""
    if not rows:
        return [], []
    header = rows[0]
    data_rows = rows[1:] if len(rows) > 1 else rows
    return header, data_rows


def wide_table_chunks(
    paper_id: str,
    chunk_index_start: int,
    element: dict[str, Any],
    section_title: str,
    caption: str,
    header: list[str],
    data_rows: list[list[str]],
    column_group_size: int,
) -> list[dict[str, Any]]:
    """Split wide tables into primary-key plus related-column chunks."""
    chunks: list[dict[str, Any]] = []
    total_columns = len(header)
    primary = 0
    related_starts = list(range(1, max(1, total_columns), column_group_size))
    if not related_starts:
        related_starts = [0]

    for start in related_starts:
        related_indices = list(range(start, min(total_columns, start + column_group_size)))
        indices = [primary, *related_indices] if primary not in related_indices else related_indices
        selected_header = select_columns(header, indices)
        selected_rows = [select_columns(row, indices) for row in data_rows]
        metadata = {
            "mode": "wide_columns",
            "caption": caption,
            "total_rows": len(data_rows),
            "total_columns": total_columns,
            "column_range": f"{min(related_indices, default=0) + 1}-{max(related_indices, default=0) + 1}",
            "primary_key": header[primary] if header else "",
        }
        text = table_text(caption, selected_header, selected_rows, metadata)
        chunks.append(
            make_table_chunk(
                paper_id,
                chunk_index_start + len(chunks),
                element,
                section_title,
                text,
                metadata,
            )
        )
    return chunks


def grouped_table_chunks(
    paper_id: str,
    chunk_index_start: int,
    element: dict[str, Any],
    section_title: str,
    caption: str,
    header: list[str],
    data_rows: list[list[str]],
    category_index: int,
) -> list[dict[str, Any]]:
    """Split medium tables by a low-cardinality category column."""
    grouped: dict[str, list[list[str]]] = defaultdict(list)
    for row in data_rows:
        key = row[category_index].strip() if category_index < len(row) else ""
        grouped[key or "unknown"].append(row)

    chunks: list[dict[str, Any]] = []
    category_name = header[category_index] if category_index < len(header) else f"column_{category_index + 1}"
    for key, rows in grouped.items():
        metadata = {
            "mode": "medium_group",
            "caption": caption,
            "group_column": category_name,
            "group_key": key,
            "group_rows": len(rows),
            "total_rows": len(data_rows),
            "total_columns": len(header),
        }
        text = table_text(caption, header, rows, metadata)
        chunks.append(
            make_table_chunk(
                paper_id,
                chunk_index_start + len(chunks),
                element,
                section_title,
                text,
                metadata,
            )
        )
    return chunks


def row_range_table_chunks(
    paper_id: str,
    chunk_index_start: int,
    element: dict[str, Any],
    section_title: str,
    caption: str,
    header: list[str],
    data_rows: list[list[str]],
    row_chunk_size: int,
    mode: str,
) -> list[dict[str, Any]]:
    """Split a table into fixed row ranges."""
    chunks: list[dict[str, Any]] = []
    for start in range(0, len(data_rows), row_chunk_size):
        rows = data_rows[start : start + row_chunk_size]
        end = start + len(rows)
        metadata = {
            "mode": mode,
            "caption": caption,
            "row_range": f"{start + 1}-{end}",
            "total_rows": len(data_rows),
            "total_columns": len(header),
            "summary": table_summary(caption, header, len(data_rows)),
        }
        text = table_text(caption, header, rows, metadata)
        chunks.append(
            make_table_chunk(
                paper_id,
                chunk_index_start + len(chunks),
                element,
                section_title,
                text,
                metadata,
            )
        )
    return chunks


def find_category_column(header: list[str], data_rows: list[list[str]]) -> int | None:
    """Return a low-cardinality column index for medium-table grouping."""
    row_count = len(data_rows)
    if row_count <= 0:
        return None
    for index, _name in enumerate(header):
        values = [
            row[index].strip()
            for row in data_rows
            if index < len(row) and row[index].strip()
        ]
        if not values:
            continue
        unique_values = set(values)
        if len(unique_values) <= 1:
            continue
        if len(unique_values) <= min(20, max(2, row_count // 2)) and len(unique_values) / row_count <= 0.35:
            if all(len(value) <= 80 for value in unique_values):
                return index
    return None


def table_text(
    caption: str,
    header: list[str],
    rows: list[list[str]],
    metadata: dict[str, Any],
) -> str:
    """Render a table chunk as searchable text."""
    parts: list[str] = []
    if caption:
        parts.append(f"Table title: {caption}")
    if metadata:
        parts.append(
            "Table metadata: "
            + "; ".join(f"{key}={value}" for key, value in metadata.items() if value not in ("", None))
        )
    parts.append(markdown_table(header, rows))
    return "\n".join(part for part in parts if part).strip()


def table_summary(caption: str, header: list[str], row_count: int) -> str:
    """Return a compact table summary for large-table chunks."""
    title = caption or "untitled table"
    return f"{title}; rows={row_count}; columns={len(header)}; headers={', '.join(header[:12])}"


def markdown_table(header: list[str], rows: list[list[str]]) -> str:
    """Render table rows as Markdown."""
    if not header and not rows:
        return ""
    width = max(len(header), *(len(row) for row in rows), 1)
    safe_header = pad_row(header, width)
    lines = [
        "| " + " | ".join(escape_table_cell(cell) for cell in safe_header) + " |",
        "| " + " | ".join("---" for _ in range(width)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(escape_table_cell(cell) for cell in pad_row(row, width)) + " |")
    return "\n".join(lines)


def parse_table_rows(table_body: str) -> list[list[str]]:
    """Parse HTML tables with the standard library."""
    if not table_body.strip():
        return []
    parser = SimpleHTMLTableParser()
    try:
        parser.feed(table_body)
    except Exception:
        return []
    rows = [[cell.strip() for cell in row] for row in parser.rows if any(cell.strip() for cell in row)]
    if rows:
        return rows
    plain_rows = []
    for line in table_body.splitlines():
        cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
        if len(cells) > 1:
            plain_rows.append(cells)
    return plain_rows


class SimpleHTMLTableParser(HTMLParser):
    """Tiny HTML table parser for MinerU table_body fragments."""

    def __init__(self) -> None:
        super().__init__()
        self.rows: list[list[str]] = []
        self._current_row: list[str] | None = None
        self._current_cell: list[str] | None = None

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag.lower() == "tr":
            self._current_row = []
        elif tag.lower() in {"td", "th"}:
            self._current_cell = []

    def handle_data(self, data: str) -> None:
        if self._current_cell is not None:
            self._current_cell.append(data)

    def handle_endtag(self, tag: str) -> None:
        if tag.lower() in {"td", "th"} and self._current_cell is not None:
            if self._current_row is not None:
                self._current_row.append(" ".join("".join(self._current_cell).split()))
            self._current_cell = None
        elif tag.lower() == "tr" and self._current_row is not None:
            self.rows.append(self._current_row)
            self._current_row = None


def make_table_chunk(
    paper_id: str,
    chunk_index: int,
    element: dict[str, Any],
    section_title: str,
    text: str,
    table_metadata: dict[str, Any],
) -> dict[str, Any]:
    """Create a table chunk payload."""
    table_payload = {
        **table_metadata,
        "page_num": safe_int(element.get("page_num"), 1),
        "bbox": element.get("bbox") or [],
        "visual_id": str(element.get("visual_id") or ""),
        "label": str(element.get("label") or ""),
        "path": str(element.get("path") or ""),
        "source_paths": list(element.get("source_paths") or []),
    }
    return make_chunk_payload(
        paper_id=paper_id,
        chunk_index=chunk_index,
        page_num=safe_int(element.get("page_num"), 1),
        section_title=section_title,
        text=text,
        chunk_type="table",
        tables=[table_payload],
    )


def make_chunk_payload(
    paper_id: str,
    chunk_index: int,
    page_num: int,
    section_title: str,
    text: str,
    chunk_type: str = "text",
    images: list[dict[str, Any]] | None = None,
    tables: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Create a chunk payload compatible with storage and retrieval."""
    image_payloads = images or []
    table_payloads = tables or []
    return {
        "paper_id": paper_id,
        "chunk_id": f"{paper_id}_chunk_{chunk_index:04d}",
        "chunk_index": chunk_index,
        "page_num": page_num,
        "section_title": section_title,
        "text": text.strip(),
        "chunk_type": chunk_type,
        "images": image_payloads,
        "tables": table_payloads,
        "images_json": json.dumps(image_payloads, ensure_ascii=False),
        "tables_json": json.dumps(table_payloads, ensure_ascii=False),
    }


def renumber_chunk(chunk: dict[str, Any], index: int, paper_id: str) -> dict[str, Any]:
    """Keep chunk ids sequential after filtering empty text."""
    item = dict(chunk)
    item["chunk_index"] = index
    item["chunk_id"] = f"{paper_id}_chunk_{index:04d}"
    return item


def count_tokens(text: str) -> int:
    """Approximate token count without adding a tokenizer dependency."""
    return len(re.findall(r"\w+|[^\s\w]", text, flags=re.UNICODE))


def select_columns(row: list[str], indices: list[int]) -> list[str]:
    return [row[index] if index < len(row) else "" for index in indices]


def pad_row(row: list[str], width: int) -> list[str]:
    return [*(str(cell) for cell in row), *([""] * max(0, width - len(row)))]


def escape_table_cell(value: Any) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ").strip()


def safe_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def chunk_text(
    text: str,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    overlap: int = DEFAULT_OVERLAP,
    embedding_client: Any | None = None,
) -> list[str]:
    """Split plain text into chunk text strings."""
    chunks = chunk_pages(
        paper_id="",
        pages=[{"page_num": 1, "text": text}],
        chunk_size=chunk_size,
        overlap=overlap,
        embedding_client=embedding_client,
    )
    return [chunk["text"] for chunk in chunks]
