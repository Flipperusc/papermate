"""PDF parsing utilities."""

from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

from config import settings
from src.errors import ErrorCode, PdfParseError
from src.mineru_client import MinerUClient


def clean_page_text(text: str) -> str:
    """Apply basic cleanup to extracted page text."""
    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    normalized = re.sub(r"[ \t\f\v]+", " ", normalized)
    normalized = "\n".join(line.strip() for line in normalized.splitlines())
    normalized = re.sub(r"\n{3,}", "\n\n", normalized)
    return normalized.strip()


def get_pdf_page_count(file_path: str | Path) -> int:
    """Return the PDF page count when PyMuPDF is available."""
    try:
        import fitz

        with fitz.open(file_path) as document:
            return int(document.page_count)
    except Exception:
        return 0


def parse_pdf(file_path: str | Path, paper_id: str, include_images: bool = False) -> dict[str, Any]:
    """Parse a PDF with the configured provider.

    Default provider is MinerU, which converts the PDF to Markdown first.
    Set PAPERMATE_PDF_PARSE_PROVIDER=pymupdf to use local PyMuPDF extraction.
    """
    provider = settings.pdf_parse_provider.lower()
    if provider == "mineru":
        return parse_pdf_with_mineru(file_path, paper_id, include_images=include_images)
    if provider == "pymupdf":
        return parse_pdf_with_pymupdf(file_path, paper_id)

    raise PdfParseError(
        ErrorCode.PDF_PARSE_FAILED,
        detail=f"不支持的 PDF 解析方式：{settings.pdf_parse_provider}",
    )


def parse_pdf_with_mineru(file_path: str | Path, paper_id: str, include_images: bool = False) -> dict[str, Any]:
    """Convert PDF to Markdown with MinerU and build page text for downstream chunks."""
    path = Path(file_path)
    mineru_result = MinerUClient().pdf_to_markdown(
        path,
        paper_id,
        file_name=path.name,
        include_images=include_images,
    )

    content_list = mineru_result.get("content_list")
    # MinerU's content_list preserves page-level structure; if it is missing,
    # fall back to a single Markdown-derived page so indexing can still proceed.
    images = mineru_result.get("images", [])
    elements = elements_from_content_list(content_list, images) if content_list else []
    pages = pages_from_content_list(content_list) if content_list else []
    if not pages:
        pages = pages_from_markdown(mineru_result["markdown"])
    if not elements:
        elements = elements_from_pages(pages)

    total_text = "".join(page["text"] for page in pages)
    if not total_text.strip():
        raise PdfParseError(ErrorCode.PDF_NO_TEXT)

    page_count = get_pdf_page_count(path) or max((page["page_num"] for page in pages), default=0)

    return {
        "paper_id": paper_id,
        "page_count": page_count,
        "pages": pages,
        "parser": "mineru",
        "markdown": mineru_result["markdown"],
        "markdown_path": mineru_result["markdown_path"],
        "content_list_path": mineru_result.get("content_list_path"),
        "images": images,
        "elements": elements,
        "mineru_batch_id": mineru_result["batch_id"],
    }


def parse_pdf_with_pymupdf(file_path: str | Path, paper_id: str) -> dict[str, Any]:
    """Extract cleaned text from a PDF file with PyMuPDF."""
    path = Path(file_path)

    try:
        import fitz
    except ImportError as exc:
        raise PdfParseError(
            ErrorCode.PDF_PARSE_FAILED,
            detail="缺少 PyMuPDF 依赖，请先安装 requirements.txt。",
        ) from exc

    pages: list[dict[str, Any]] = []

    try:
        with fitz.open(path) as document:
            page_count = document.page_count

            for page_index, page in enumerate(document, start=1):
                text = clean_page_text(page.get_text("text"))
                if text:
                    pages.append({"page_num": page_index, "text": text})
    except Exception as exc:
        raise PdfParseError(ErrorCode.PDF_PARSE_FAILED, detail=str(exc)) from exc

    total_text = "".join(page["text"] for page in pages)
    if not total_text.strip():
        raise PdfParseError(ErrorCode.PDF_NO_TEXT)

    return {
        "paper_id": paper_id,
        "page_count": page_count,
        "pages": pages,
        "elements": elements_from_pages(pages),
        "parser": "pymupdf",
        "images": [],
    }


def elements_from_pages(pages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Build text-only chunking elements from page records."""
    elements: list[dict[str, Any]] = []
    for order, page in enumerate(pages):
        text = str(page.get("text", "")).strip()
        if not text:
            continue
        elements.append(
            {
                "type": "text",
                "order": order,
                "page_num": int(page.get("page_num", 1) or 1),
                "text": text,
            }
        )
    return elements


def elements_from_content_list(
    content_list: Any,
    images: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """Build ordered text/image/table elements from MinerU content_list."""
    if not isinstance(content_list, list):
        return []

    image_lookup = build_image_lookup(images or [])
    elements: list[dict[str, Any]] = []
    for order, item in enumerate(content_list):
        if not isinstance(item, dict):
            continue

        item_type = str(item.get("type", "")).strip().lower()
        page_num = content_item_page_num(item)
        if item_type == "table":
            elements.append(build_table_element(item, image_lookup, order, page_num))
            continue
        if item_type in {"image", "equation"}:
            elements.append(build_image_element(item, image_lookup, order, page_num))
            continue

        text = content_item_to_text(item)
        if text:
            elements.append(
                {
                    "type": "text",
                    "order": order,
                    "page_num": page_num,
                    "text": text,
                    "raw_type": item_type,
                    "bbox": item.get("bbox"),
                }
            )

    return elements


def build_image_lookup(images: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Index normalized MinerU images by archive path, filename, and path."""
    lookup: dict[str, dict[str, Any]] = {}
    for image in images:
        for key in ("archive_name", "file_name", "path"):
            value = image.get(key)
            if isinstance(value, str) and value.strip():
                for candidate in normalized_path_keys(value):
                    lookup[candidate] = image
        for source_path in image.get("source_paths") or []:
            if isinstance(source_path, str) and source_path.strip():
                for candidate in normalized_path_keys(source_path):
                    lookup[candidate] = image
    return lookup


def build_image_element(
    item: dict[str, Any],
    image_lookup: dict[str, dict[str, Any]],
    order: int,
    page_num: int,
) -> dict[str, Any]:
    """Return one image-like element with normalized image metadata when available."""
    image = lookup_content_image(item, image_lookup) or {}
    caption = first_text_value(item, ("image_caption", "caption", "alt_text", "content"))
    return {
        "type": "image",
        "order": order,
        "page_num": page_num,
        "kind": str(item.get("type") or image.get("kind") or "image").lower(),
        "path": image.get("path") or first_text_value(item, ("img_path", "image_path", "path")),
        "mime_type": image.get("mime_type") or first_text_value(item, ("mime_type",)),
        "caption": image.get("caption") or caption,
        "alt_text": first_text_value(item, ("alt_text", "alt", "content")),
        "bbox": image.get("bbox") or item.get("bbox"),
        "visual_id": image.get("visual_id") or "",
        "label": image.get("label") or "",
        "source_paths": image.get("source_paths") or item_source_paths(item),
    }


def build_table_element(
    item: dict[str, Any],
    image_lookup: dict[str, dict[str, Any]],
    order: int,
    page_num: int,
) -> dict[str, Any]:
    """Return one table element with table body and visual metadata."""
    image = lookup_content_image(item, image_lookup) or {}
    return {
        "type": "table",
        "order": order,
        "page_num": page_num,
        "caption": image.get("caption") or first_text_value(item, ("table_caption", "caption")),
        "table_body": image.get("table_body") or first_text_value(item, ("table_body", "table_html", "html", "content")),
        "path": image.get("path") or first_text_value(item, ("img_path", "image_path", "path")),
        "bbox": image.get("bbox") or item.get("bbox"),
        "visual_id": image.get("visual_id") or "",
        "label": image.get("label") or "",
        "source_paths": image.get("source_paths") or item_source_paths(item),
    }


def lookup_content_image(
    item: dict[str, Any],
    image_lookup: dict[str, dict[str, Any]],
) -> dict[str, Any] | None:
    """Find a normalized MinerU visual record for a content_list item."""
    for source_path in item_source_paths(item):
        for candidate in normalized_path_keys(source_path):
            image = image_lookup.get(candidate)
            if image:
                return image
    return None


def item_source_paths(item: dict[str, Any]) -> list[str]:
    """Return path-like references from a MinerU content item."""
    sources: list[str] = []
    for key in ("img_path", "image_path", "path"):
        value = item.get(key)
        if isinstance(value, str) and value.strip():
            sources.append(value.strip())
    return sources


def normalized_path_keys(value: str) -> list[str]:
    """Return stable lookup keys for local and archive image paths."""
    normalized = str(value or "").replace("\\", "/").strip()
    if not normalized:
        return []
    return list(dict.fromkeys([normalized, Path(normalized).name]))


def pages_from_content_list(content_list: Any) -> list[dict[str, Any]]:
    """Build page text from MinerU content_list entries when available."""
    if not isinstance(content_list, list):
        return []

    page_blocks: dict[int, list[str]] = defaultdict(list)
    for item in content_list:
        if not isinstance(item, dict):
            continue

        text = content_item_to_text(item)
        if not text:
            continue

        page_num = content_item_page_num(item)
        page_blocks[page_num].append(text)

    pages = []
    for page_num in sorted(page_blocks):
        text = clean_page_text("\n\n".join(page_blocks[page_num]))
        if text:
            pages.append({"page_num": page_num, "text": text})

    return pages


def content_item_page_num(item: dict[str, Any]) -> int:
    """Return a 1-based page number for a MinerU content item."""
    if "page_idx" in item:
        return safe_int(item.get("page_idx"), default=0) + 1
    if "page_num" in item:
        return max(1, safe_int(item.get("page_num"), default=1))
    if "page" in item:
        return safe_int(item.get("page"), default=0) + 1
    return 1


def content_item_to_text(item: dict[str, Any]) -> str:
    """Convert one MinerU content_list item into Markdown-like text."""
    item_type = str(item.get("type", "")).lower()
    text = first_text_value(
        item,
        (
            "text",
            "content",
            "list_items",
            "code_body",
            "table_body",
            "table_caption",
            "img_caption",
            "image_caption",
            "caption",
            "latex",
        ),
    )

    if not text:
        return ""

    if item_type == "title":
        return f"# {text.strip()}"
    if "equation" in item_type and not text.strip().startswith("$$"):
        return f"$$\n{text.strip()}\n$$"

    return text.strip()


def first_text_value(item: dict[str, Any], keys: tuple[str, ...]) -> str:
    """Return the first non-empty textual value from an item."""
    for key in keys:
        value = item.get(key)
        if value is None:
            continue
        if isinstance(value, str) and value.strip():
            return value
        if isinstance(value, list):
            joined = " ".join(str(part).strip() for part in value if str(part).strip())
            if joined:
                return joined
        if isinstance(value, dict):
            serialized = json.dumps(value, ensure_ascii=False)
            if serialized.strip():
                return serialized
    return ""


def pages_from_markdown(markdown: str) -> list[dict[str, Any]]:
    """Fallback: use the full Markdown as one logical page."""
    text = clean_markdown_for_index(markdown)
    return [{"page_num": 1, "text": text}] if text else []


def clean_markdown_for_index(markdown: str) -> str:
    """Remove Markdown-only artifacts that are not useful for retrieval."""
    text = re.sub(r"!\[[^\]]*\]\([^)]+\)", "", markdown)
    text = re.sub(r"<img\b[^>]*>", "", text, flags=re.IGNORECASE)
    text = re.sub(r"<!--.*?-->", "", text, flags=re.DOTALL)
    return clean_page_text(text)


def safe_int(value: Any, default: int) -> int:
    """Coerce a value to int with a default."""
    try:
        return int(value)
    except (TypeError, ValueError):
        return default
