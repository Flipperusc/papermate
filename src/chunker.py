"""Text chunking utilities."""

from __future__ import annotations

import re
from typing import Any


CHUNKER_VERSION = "section-detection-v3-mineru-markdown"

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


def split_long_paragraph(text: str, max_size: int) -> list[str]:
    """Split an oversized paragraph by sentence first, then words."""
    if len(text) <= max_size:
        return [text]

    chunks: list[str] = []
    current: list[str] = []

    for sentence in split_sentences(text):
        if len(sentence) > max_size:
            if current:
                chunks.append(" ".join(current))
                current = []
            chunks.extend(split_by_words(sentence, max_size))
            continue

        candidate = " ".join([*current, sentence]) if current else sentence
        if len(candidate) <= max_size:
            current.append(sentence)
        else:
            chunks.append(" ".join(current))
            current = [sentence]

    if current:
        chunks.append(" ".join(current))

    return [chunk.strip() for chunk in chunks if chunk.strip()]


def joined_length(units: list[dict[str, Any]]) -> int:
    """Return the rendered chunk length for a list of paragraph units."""
    return len("\n\n".join(unit["text"] for unit in units))


def make_overlap_units(units: list[dict[str, Any]], overlap: int) -> list[dict[str, Any]]:
    """Build paragraph-aware overlap units from the end of a chunk."""
    if overlap <= 0:
        return []

    selected: list[dict[str, Any]] = []
    remaining = overlap

    for unit in reversed(units):
        text = unit["text"]
        separator_len = 2 if selected else 0

        if len(text) + separator_len <= remaining:
            selected.append(unit)
            remaining -= len(text) + separator_len
            continue

        available = remaining - separator_len
        if available <= 0:
            break

        tail_candidates = split_sentences(text)
        tail_parts: list[str] = []
        tail_len = 0
        for sentence in reversed(tail_candidates):
            candidate_len = len(sentence) + (1 if tail_parts else 0)
            if tail_len + candidate_len > available:
                continue
            tail_parts.append(sentence)
            tail_len += candidate_len

        if tail_parts:
            selected.append({**unit, "text": " ".join(reversed(tail_parts))})
        else:
            word_tail = split_by_words(text, available)
            if word_tail:
                selected.append({**unit, "text": word_tail[-1]})
        break

    return list(reversed(selected))


def make_chunk(
    paper_id: str,
    chunk_index: int,
    units: list[dict[str, Any]],
) -> dict[str, Any]:
    """Create a chunk payload from paragraph units."""
    text = "\n\n".join(unit["text"] for unit in units).strip()
    first_unit = units[0]
    section_title = next(
        (unit["section_title"] for unit in reversed(units) if unit["section_title"]),
        "",
    )

    return {
        "paper_id": paper_id,
        "chunk_id": f"{paper_id}_chunk_{chunk_index:04d}",
        "chunk_index": chunk_index,
        "page_num": first_unit["page_num"],
        "section_title": section_title,
        "text": text,
    }


def chunk_pages(
    paper_id: str,
    pages: list[dict[str, Any]],
    chunk_size: int = 1000,
    overlap: int = 150,
) -> list[dict[str, Any]]:
    """Split parsed PDF pages into paragraph-aware chunks."""
    chunk_size = max(1, chunk_size)
    overlap = max(0, min(overlap, chunk_size // 2))

    paragraph_units: list[dict[str, Any]] = []
    current_section = ""
    next_unit_starts_section = False

    for page in pages:
        page_num = int(page.get("page_num", 0) or 0)
        page_text = str(page.get("text", "")).strip()
        if not page_text:
            continue

        for paragraph in split_page_paragraphs(page_text):
            section_title = detect_section_title(paragraph)
            if section_title:
                current_section = section_title
                next_unit_starts_section = True
                continue

            section_title, body = split_section_prefix(paragraph)
            if section_title:
                current_section = section_title
                next_unit_starts_section = True
                paragraph = body
                if not paragraph:
                    continue

            for text_part in split_long_paragraph(paragraph, chunk_size):
                paragraph_units.append(
                    {
                        "page_num": page_num,
                        "section_title": current_section,
                        # Force a chunk boundary at real section starts so
                        # retrieval snippets keep paper structure visible.
                        "section_start": next_unit_starts_section,
                        "text": text_part,
                    }
                )
                next_unit_starts_section = False

    if not paragraph_units:
        return []

    chunks: list[dict[str, Any]] = []
    current_units: list[dict[str, Any]] = []

    for unit in paragraph_units:
        if not current_units:
            current_units.append(unit)
            continue

        if unit.get("section_start"):
            chunks.append(make_chunk(paper_id, len(chunks), current_units))
            current_units = [unit]
            continue

        projected = joined_length([*current_units, unit])
        if projected <= chunk_size:
            current_units.append(unit)
            continue

        chunks.append(make_chunk(paper_id, len(chunks), current_units))
        overlap_units = make_overlap_units(current_units, overlap)

        if overlap_units and joined_length([*overlap_units, unit]) <= chunk_size:
            current_units = [*overlap_units, unit]
        else:
            current_units = [unit]

    if current_units:
        chunks.append(make_chunk(paper_id, len(chunks), current_units))

    return [chunk for chunk in chunks if chunk["text"].strip()]


def chunk_text(text: str, chunk_size: int = 1000, overlap: int = 150) -> list[str]:
    """Split plain text into chunk text strings."""
    chunks = chunk_pages(
        paper_id="",
        pages=[{"page_num": 1, "text": text}],
        chunk_size=chunk_size,
        overlap=overlap,
    )
    return [chunk["text"] for chunk in chunks]
