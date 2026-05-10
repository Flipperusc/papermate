"""Rule-based alignment for interleaved bilingual Markdown reading."""

from __future__ import annotations

import hashlib
import re
from typing import Any


HEADING_RE = re.compile(r"^\s{0,3}(#{1,6})\s+(.+?)\s*$")
SECTION_HEADING_RE = re.compile(r"^\s{0,3}(#{1,4})\s+(.+?)\s*$")
LIST_RE = re.compile(r"^\s{0,3}(?:[-*+]\s+|\d+[.)]\s+)")
FENCE_RE = re.compile(r"^\s*(```|~~~)")


def align_markdown_bilingual(
    source_markdown: str,
    translated_markdown: str,
    mode: str = "section",
) -> list[dict[str, Any]]:
    """Align source and translated Markdown blocks without calling an LLM."""
    requested_mode = normalize_align_mode(mode)

    if requested_mode == "section":
        source_sections = split_markdown_sections(source_markdown)
        target_sections = split_markdown_sections(translated_markdown)
        if sections_are_close(source_sections, target_sections):
            return align_blocks_by_order(source_sections, target_sections)

        aligned = align_blocks_by_order(
            split_markdown_blocks(source_markdown),
            split_markdown_blocks(translated_markdown),
        )
        if aligned:
            aligned[0]["alignment_warning"] = (
                "Section counts differ; fell back to paragraph-order alignment."
            )
            aligned[0]["fallback_mode"] = "paragraph"
        return aligned

    return align_blocks_by_order(
        split_markdown_blocks(source_markdown),
        split_markdown_blocks(translated_markdown),
    )


def split_markdown_sections(markdown_text: str) -> list[dict[str, Any]]:
    """Split Markdown into heading-led sections."""
    sections: list[str] = []
    current: list[str] = []

    for line in (markdown_text or "").splitlines():
        if SECTION_HEADING_RE.match(line) and current:
            sections.append("\n".join(current).strip())
            current = [line]
        else:
            current.append(line)

    if current:
        sections.append("\n".join(current).strip())

    return [
        build_block(section, index)
        for index, section in enumerate(sections, start=1)
        if section.strip()
    ]


def split_markdown_blocks(markdown_text: str) -> list[dict[str, Any]]:
    """Split Markdown into renderable blocks while preserving protected structures."""
    lines = (markdown_text or "").splitlines()
    blocks: list[str] = []
    buffer: list[str] = []
    index = 0

    def flush() -> None:
        nonlocal buffer
        content = "\n".join(buffer).strip()
        if content:
            blocks.append(content)
        buffer = []

    while index < len(lines):
        line = lines[index]
        stripped = line.strip()

        if not stripped:
            flush()
            index += 1
            continue

        fence_match = FENCE_RE.match(line)
        if fence_match:
            flush()
            fence = fence_match.group(1)
            block = [line]
            index += 1
            while index < len(lines):
                block.append(lines[index])
                if lines[index].strip().startswith(fence):
                    index += 1
                    break
                index += 1
            blocks.append("\n".join(block).strip())
            continue

        if stripped == "$$":
            flush()
            block = [line]
            index += 1
            while index < len(lines):
                block.append(lines[index])
                if lines[index].strip() == "$$":
                    index += 1
                    break
                index += 1
            blocks.append("\n".join(block).strip())
            continue

        if is_table_line(line):
            flush()
            block = [line]
            index += 1
            while index < len(lines) and is_table_line(lines[index]):
                block.append(lines[index])
                index += 1
            blocks.append("\n".join(block).strip())
            continue

        if HEADING_RE.match(line):
            flush()
            blocks.append(line.strip())
            index += 1
            continue

        buffer.append(line)
        index += 1

    flush()
    return [
        build_block(block, index)
        for index, block in enumerate(blocks, start=1)
        if block.strip()
    ]


def detect_markdown_block_type(block: str) -> str:
    """Return a coarse Markdown block type."""
    stripped = (block or "").strip()
    if not stripped:
        return "other"
    if HEADING_RE.match(stripped.splitlines()[0]):
        return "heading"
    if stripped.startswith("```") or stripped.startswith("~~~"):
        return "code"
    if stripped.startswith("$$") and stripped.endswith("$$"):
        return "formula"
    if all(is_table_line(line) for line in stripped.splitlines() if line.strip()):
        return "table"
    if re.match(r"^\s*!\[[^\]]*\]\([^)]+\)", stripped) or re.match(
        r"^\s*\[[^\]]*(?:图|image|figure)[^\]]*\]\(data:image",
        stripped,
        flags=re.IGNORECASE,
    ):
        return "image"
    if LIST_RE.match(stripped.splitlines()[0]):
        return "list"
    return "paragraph"


def slugify_heading(text: str) -> str:
    """Create a stable anchor-friendly slug from a heading."""
    original = text or ""
    cleaned = re.sub(r"^\s{0,3}#{1,6}\s*", "", original.strip())
    cleaned = re.sub(r"!\[[^\]]*\]\([^)]+\)", "", cleaned)
    cleaned = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", cleaned)
    cleaned = re.sub(r"[`*_~$\\]", "", cleaned)
    cleaned = re.sub(r"\s+", "-", cleaned.lower())
    cleaned = re.sub(r"[^\w\u4e00-\u9fff-]+", "", cleaned)
    cleaned = re.sub(r"-{2,}", "-", cleaned).strip("-_")
    if cleaned:
        return cleaned[:96]
    return hashlib.sha1(original.encode("utf-8")).hexdigest()[:10]


def align_blocks_by_order(
    source_blocks: list[dict[str, Any]],
    target_blocks: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Zip source and target blocks by order, preserving unmatched trailing blocks."""
    aligned: list[dict[str, Any]] = []
    total = max(len(source_blocks), len(target_blocks))
    for index in range(total):
        source = source_blocks[index] if index < len(source_blocks) else {}
        target = target_blocks[index] if index < len(target_blocks) else {}
        block_type = source.get("type") or target.get("type") or "other"
        anchor = source.get("anchor") or target.get("anchor") or f"block-{index + 1:03d}"
        level = source.get("level") or target.get("level")
        aligned.append(
            {
                "id": f"block_{index + 1:03d}",
                "type": block_type,
                "source": source.get("content", ""),
                "target": target.get("content", ""),
                "level": level,
                "anchor": anchor,
                "source_index": source.get("index"),
                "target_index": target.get("index"),
            }
        )
    return aligned


def build_block(content: str, index: int) -> dict[str, Any]:
    """Build a normalized block dict from Markdown content."""
    block_type = detect_markdown_block_type(content)
    level: int | None = None
    anchor = f"block-{index:03d}"
    first_line = content.strip().splitlines()[0] if content.strip() else ""
    heading_match = HEADING_RE.match(first_line)
    if heading_match:
        level = len(heading_match.group(1))
        anchor = slugify_heading(first_line)

    return {
        "index": index,
        "type": block_type,
        "content": content.strip(),
        "level": level,
        "anchor": anchor,
    }


def sections_are_close(
    source_sections: list[dict[str, Any]],
    target_sections: list[dict[str, Any]],
) -> bool:
    """Return whether section counts are close enough for order alignment."""
    source_count = len(source_sections)
    target_count = len(target_sections)
    if source_count <= 1 or target_count <= 1:
        return False
    diff = abs(source_count - target_count)
    tolerance = max(2, int(max(source_count, target_count) * 0.35))
    return diff <= tolerance


def is_table_line(line: str) -> bool:
    stripped = line.strip()
    return stripped.startswith("|") and stripped.endswith("|") and stripped.count("|") >= 2


def normalize_align_mode(mode: str) -> str:
    normalized = (mode or "section").strip().lower()
    if normalized in {"paragraph", "段落对齐", "paragraphs", "block"}:
        return "paragraph"
    return "section"
