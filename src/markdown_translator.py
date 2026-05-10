"""Translate Markdown papers into Simplified Chinese."""

from __future__ import annotations

import hashlib
import json
import re
import time
from pathlib import Path
from typing import Any, Callable

from config import settings
from src.llm_client import DEEPSEEK_CALL_FAILED_MESSAGE, LLMClient
from src.logger import get_logger


logger = get_logger(__name__)

TRANSLATION_SYSTEM_PROMPT = (
    "你是一名专业的学术论文翻译助手。你只输出翻译后的 Markdown，不输出额外说明。"
)

TRANSLATION_PROMPT_TEMPLATE = """你是一名专业的学术论文翻译助手。请将下面的 Markdown 学术论文内容翻译成简体中文。

要求：
1. 保留原始 Markdown 结构，包括标题层级、列表、表格、图片链接、代码块和公式。
2. 不要翻译图片路径、URL、代码、变量名、函数名、数学公式。
3. 学术术语要准确、自然，符合中文论文阅读习惯。
4. 不要省略任何内容。
5. 不要添加原文中不存在的解释。
6. 如果遇到参考文献、作者名、机构名，可以保留英文。
7. 输出必须仍然是合法 Markdown。
8. 只输出翻译后的 Markdown，不要输出额外说明。

待翻译内容如下：
---
{chunk}
---"""

PROTECTED_TOKEN_TEMPLATE = "[[PM_PROTECTED_{index:04d}]]"
DOC_PROTECTED_TOKEN_TEMPLATE = "[[PM_DOC_PROTECTED_{index:06d}]]"
MIN_MERGE_TRANSLATION_CHARS = 1000
TranslationProgressCallback = Callable[[int, int, str], None]


def translate_markdown_to_chinese(
    input_md_path: str,
    output_md_path: str,
    model: str | None = None,
    chunk_size: int = 3500,
    force: bool = False,
    progress_callback: TranslationProgressCallback | None = None,
    timeout: int | None = None,
) -> str:
    """Translate a Markdown file into Simplified Chinese and return output path."""
    input_path = Path(input_md_path)
    output_path = Path(output_md_path)
    if output_path.exists() and not force:
        notify_progress(progress_callback, 1, 1, "exists")
        return str(output_path.resolve())

    if not input_path.exists() or not input_path.is_file():
        raise FileNotFoundError(f"Markdown 文件不存在：{input_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    markdown = input_path.read_text(encoding="utf-8")
    compact_markdown, document_protected = protect_document_fragments(markdown)
    chunks = split_markdown(compact_markdown, max(800, int(chunk_size or settings.translation_chunk_size)))
    cache_dir = Path(f"{output_path}.cache")
    cache_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = cache_dir / "manifest.json"

    selected_model = model or settings.translation_model
    total_chunks = len(chunks)
    request_timeout = timeout if timeout is not None else settings.translation_timeout
    client = LLMClient(
        model=selected_model,
        system_prompt=TRANSLATION_SYSTEM_PROMPT,
        timeout=request_timeout,
    )
    translated_chunks: list[str] = []
    manifest: list[dict[str, Any]] = []
    logger.info(
        "Markdown translation started. input=%s output=%s chunks=%s protected_fragments=%s chunk_size=%s model=%s timeout=%s",
        input_path,
        output_path,
        total_chunks,
        len(document_protected),
        chunk_size,
        selected_model,
        request_timeout,
    )
    notify_progress(progress_callback, 0, total_chunks, "start")

    for index, chunk in enumerate(chunks, start=1):
        chunk_hash = stable_hash(
            json.dumps(
                {
                    "model": selected_model,
                    "provider": settings.translation_provider,
                    "chunk": chunk,
                },
                ensure_ascii=False,
            )
        )
        cache_path = cache_dir / f"{index:04d}_{chunk_hash}.md"
        if cache_path.exists() and not force:
            notify_progress(progress_callback, index - 1, total_chunks, "cached")
            translated = cache_path.read_text(encoding="utf-8")
            status = "cached"
            logger.info(
                "Markdown translation chunk reused from cache. chunk=%s/%s chars=%s",
                index,
                total_chunks,
                len(chunk),
            )
        else:
            notify_progress(progress_callback, index - 1, total_chunks, "translating")
            logger.info(
                "Markdown translation chunk started. chunk=%s/%s chars=%s",
                index,
                total_chunks,
                len(chunk),
            )
            translated = translate_chunk_with_retry(client, chunk, selected_model, index)
            cache_path.write_text(translated, encoding="utf-8")
            status = "translated"

        translated_chunks.append(translated.rstrip())
        manifest.append(
            {
                "index": index,
                "hash": chunk_hash,
                "cache_path": str(cache_path.resolve()),
                "chars": len(chunk),
            }
        )
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        notify_progress(progress_callback, index, total_chunks, status)
        logger.info(
            "Markdown translation chunk finished. chunk=%s/%s status=%s",
            index,
            total_chunks,
            status,
        )

    output_markdown = restore_document_fragments("\n\n".join(translated_chunks), document_protected)
    output_path.write_text(output_markdown.strip() + "\n", encoding="utf-8")
    notify_progress(progress_callback, total_chunks, total_chunks, "done")
    logger.info("Markdown translation finished. output=%s chunks=%s", output_path, total_chunks)
    return str(output_path.resolve())


def notify_progress(
    progress_callback: TranslationProgressCallback | None,
    completed: int,
    total: int,
    status: str,
) -> None:
    """Report translation progress without letting UI callback errors break translation."""
    if progress_callback is None:
        return
    try:
        progress_callback(completed, total, status)
    except Exception:
        logger.debug("Markdown translation progress callback failed.", exc_info=True)


def protect_document_fragments(markdown: str) -> tuple[str, dict[str, str]]:
    """Compact large non-translatable document fragments before chunking."""
    protected: dict[str, str] = {}

    def store(match: re.Match[str]) -> str:
        token = DOC_PROTECTED_TOKEN_TEMPLATE.format(index=len(protected))
        protected[token] = match.group(0)
        return token

    compacted = re.sub(
        r"data:image/[A-Za-z0-9.+-]+;base64,[A-Za-z0-9+/=]+",
        store,
        markdown,
        flags=re.IGNORECASE,
    )
    return compacted, protected


def restore_document_fragments(markdown: str, protected: dict[str, str]) -> str:
    """Restore document-level protected fragments after all chunks are translated."""
    restored = markdown
    for token, value in protected.items():
        restored = restored.replace(token, value)
    return restored


def split_markdown(markdown: str, chunk_size: int) -> list[str]:
    """Split Markdown by headings first, then by protected block-aware units."""
    sections: list[str] = []
    current: list[str] = []

    for line in markdown.splitlines(keepends=True):
        if is_heading(line) and current:
            sections.append("".join(current))
            current = [line]
        else:
            current.append(line)
    if current:
        sections.append("".join(current))

    chunks: list[str] = []
    for section in sections:
        if len(section) <= chunk_size:
            chunks.append(section)
        else:
            chunks.extend(split_large_section(section, chunk_size))
    return merge_short_chunks_forward(chunks, MIN_MERGE_TRANSLATION_CHARS)


def merge_short_chunks_forward(chunks: list[str], min_chars: int) -> list[str]:
    """Merge short translation chunks into the following chunk without re-splitting."""
    cleaned_chunks = [chunk.strip() for chunk in chunks if chunk.strip()]
    if len(cleaned_chunks) <= 1:
        return cleaned_chunks

    merged: list[str] = []
    pending: list[str] = []
    last_index = len(cleaned_chunks) - 1

    for index, chunk in enumerate(cleaned_chunks):
        if len(chunk) < min_chars and index < last_index:
            pending.append(chunk)
            continue

        if pending:
            merged.append("\n\n".join([*pending, chunk]).strip())
            pending = []
        else:
            merged.append(chunk)

    if pending:
        merged.append("\n\n".join(pending).strip())
    return merged


def split_large_section(section: str, chunk_size: int) -> list[str]:
    """Split an oversized section while avoiding code/math/table splits."""
    units = markdown_units(section)
    chunks: list[str] = []
    current: list[str] = []
    current_len = 0

    for unit in units:
        unit_len = len(unit)
        if current and current_len + unit_len > chunk_size:
            chunks.append("".join(current))
            current = []
            current_len = 0

        if unit_len > chunk_size and not is_protected_block(unit):
            chunks.extend(split_plain_text(unit, chunk_size))
            continue

        current.append(unit)
        current_len += unit_len

    if current:
        chunks.append("".join(current))
    return chunks


def markdown_units(markdown: str) -> list[str]:
    """Return block-aware units for packing into translation chunks."""
    lines = markdown.splitlines(keepends=True)
    units: list[str] = []
    buffer: list[str] = []
    index = 0

    while index < len(lines):
        line = lines[index]
        stripped = line.strip()

        if stripped.startswith("```"):
            flush_buffer(buffer, units)
            block = [line]
            index += 1
            while index < len(lines):
                block.append(lines[index])
                if lines[index].strip().startswith("```"):
                    index += 1
                    break
                index += 1
            units.append("".join(block))
            continue

        if stripped == "$$":
            flush_buffer(buffer, units)
            block = [line]
            index += 1
            while index < len(lines):
                block.append(lines[index])
                if lines[index].strip() == "$$":
                    index += 1
                    break
                index += 1
            units.append("".join(block))
            continue

        if is_table_line(line):
            flush_buffer(buffer, units)
            block = [line]
            index += 1
            while index < len(lines) and is_table_line(lines[index]):
                block.append(lines[index])
                index += 1
            units.append("".join(block))
            continue

        buffer.append(line)
        if not stripped:
            flush_buffer(buffer, units)
        index += 1

    flush_buffer(buffer, units)
    return units


def protect_markdown_fragments(chunk: str) -> tuple[str, dict[str, str]]:
    """Replace fragments that should not be translated with stable tokens."""
    protected: dict[str, str] = {}

    def store(value: str) -> str:
        token = PROTECTED_TOKEN_TEMPLATE.format(index=len(protected))
        protected[token] = value
        return token

    protected_chunk = chunk
    patterns = [
        r"\[\[PM_DOC_PROTECTED_\d{6}\]\]",
        r"```[\s\S]*?```",
        r"\$\$[\s\S]*?\$\$",
        r"!\[[^\]]*\]\([^)]+\)",
        r"`[^`\n]+`",
        r"(?<!\$)\$(?!\$)(?:\\.|[^$\n])+\$(?!\$)",
    ]
    for pattern in patterns:
        protected_chunk = re.sub(pattern, lambda match: store(match.group(0)), protected_chunk)

    protected_chunk = re.sub(
        r"(\[[^\]]+\]\()([^)]+)(\))",
        lambda match: f"{match.group(1)}{store(match.group(2))}{match.group(3)}",
        protected_chunk,
    )
    protected_chunk = re.sub(
        r"https?://[^\s)>\]]+",
        lambda match: store(match.group(0)),
        protected_chunk,
    )
    return protected_chunk, protected


def restore_markdown_fragments(text: str, protected: dict[str, str]) -> str:
    """Restore protected Markdown fragments after translation."""
    restored = text
    for token, value in protected.items():
        restored = restored.replace(token, value)
    return restored


def translate_chunk_with_retry(
    client: LLMClient,
    chunk: str,
    model: str,
    index: int,
    retries: int = 2,
) -> str:
    """Translate one chunk, falling back to original text after retries."""
    protected_chunk, protected = protect_markdown_fragments(chunk)
    prompt = TRANSLATION_PROMPT_TEMPLATE.format(chunk=protected_chunk)
    max_tokens = max(2000, min(8000, int(len(protected_chunk) * 1.8)))

    last_error: Exception | None = None
    for attempt in range(1, retries + 2):
        try:
            translated = client.generate(prompt, temperature=0.1, max_tokens=max_tokens)
            if not translated.strip() or translated == DEEPSEEK_CALL_FAILED_MESSAGE:
                raise RuntimeError(DEEPSEEK_CALL_FAILED_MESSAGE)
            return restore_markdown_fragments(clean_translation_output(translated), protected)
        except Exception as exc:
            last_error = exc
            logger.warning(
                "Markdown translation chunk failed. chunk=%s attempt=%s model=%s error=%s",
                index,
                attempt,
                model,
                exc,
            )
            time.sleep(min(2, attempt))

    logger.error(
        "Markdown translation chunk permanently failed; keeping source text. chunk=%s error=%s",
        index,
        last_error,
    )
    return chunk


def clean_translation_output(text: str) -> str:
    """Remove common wrapper noise while keeping valid Markdown content."""
    cleaned = text.strip()
    if cleaned.startswith("```markdown") and cleaned.endswith("```"):
        cleaned = cleaned[len("```markdown") : -3].strip()
    elif cleaned.startswith("```") and cleaned.endswith("```"):
        cleaned = cleaned[3:-3].strip()
    return cleaned


def split_plain_text(text: str, chunk_size: int) -> list[str]:
    """Split long plain text into smaller chunks."""
    parts: list[str] = []
    start = 0
    while start < len(text):
        end = min(len(text), start + chunk_size)
        if end < len(text):
            boundary = max(text.rfind("\n", start, end), text.rfind(". ", start, end))
            if boundary > start + chunk_size // 2:
                end = boundary + 1
        parts.append(text[start:end])
        start = end
    return parts


def is_heading(line: str) -> bool:
    return bool(re.match(r"^\s{0,3}#{1,6}\s+\S", line))


def is_table_line(line: str) -> bool:
    stripped = line.strip()
    return stripped.startswith("|") and stripped.endswith("|") and stripped.count("|") >= 2


def is_protected_block(text: str) -> bool:
    stripped = text.strip()
    return stripped.startswith("```") or stripped.startswith("$$") or is_table_line(stripped.splitlines()[0])


def flush_buffer(buffer: list[str], units: list[str]) -> None:
    if buffer:
        units.append("".join(buffer))
        buffer.clear()


def stable_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]
