"""Literature card generation pipeline."""

from __future__ import annotations

from typing import Any

from src.db import get_paper_chunks
from src.llm_client import LLMClient


UNKNOWN_VALUE = "原文未明确说明"
CARD_FIELDS = [
    "论文标题",
    "作者",
    "年份",
    "研究领域",
    "研究问题",
    "方法概述",
    "实验数据集",
]
PRIORITY_SECTIONS = [
    "Abstract",
    "Introduction",
    "Related Work",
    "Method",
    "Experiments",
    "Results",
    "Conclusion",
    "Limitations",
]


def empty_literature_card() -> str:
    """Return a markdown card with all required fields marked unknown."""
    lines = ["# 文献卡片", ""]
    for field in CARD_FIELDS:
        lines.extend([f"## {field}", UNKNOWN_VALUE, ""])
    return "\n".join(lines).strip()


def select_card_context(chunks: list[dict[str, Any]], max_chars: int = 18000) -> str:
    """Select high-signal chunks for literature-card generation."""
    if not chunks:
        return ""

    selected: list[dict[str, Any]] = []
    seen_chunk_ids: set[str] = set()

    def add_chunk(chunk: dict[str, Any]) -> None:
        chunk_id = str(chunk["chunk_id"])
        if chunk_id not in seen_chunk_ids:
            selected.append(chunk)
            seen_chunk_ids.add(chunk_id)

    for chunk in chunks[:5]:
        add_chunk(chunk)

    for section in PRIORITY_SECTIONS:
        matches = [
            chunk
            for chunk in chunks
            if section.lower() in str(chunk.get("section_title") or "").lower()
        ]
        for chunk in matches[:4]:
            add_chunk(chunk)

    for chunk in chunks:
        if len(render_context_chunks(selected)) >= max_chars:
            break
        add_chunk(chunk)

    context = render_context_chunks(selected)
    return context[:max_chars]


def render_context_chunks(chunks: list[dict[str, Any]]) -> str:
    """Render chunks into a compact prompt context."""
    rendered: list[str] = []
    for chunk in chunks:
        section_title = chunk.get("section_title") or "未识别章节"
        rendered.append(
            "\n".join(
                [
                    f"chunk_id: {chunk['chunk_id']}",
                    f"page_num: {chunk['page_num']}",
                    f"section_title: {section_title}",
                    "text:",
                    str(chunk["text"]),
                ]
            )
        )
    return "\n\n---\n\n".join(rendered)


def build_card_prompt(context: str) -> str:
    """Build a strict prompt for markdown literature-card generation."""
    field_list = "\n".join(f"- {field}" for field in CARD_FIELDS)
    return f"""请基于下面的论文原文片段生成 Markdown 格式的文献卡片。

严格要求：
1. 只能使用给定论文片段中的信息，不允许补充常识或猜测。
2. 没有依据的信息必须写“{UNKNOWN_VALUE}”。
3. 输出必须是 Markdown。
4. 必须包含且只包含以下字段，字段名使用二级标题：
{field_list}
5. 不要编造作者、年份、数据集、实验结论或页码。
6. 内容要简洁、具体，适合后续复习和管理。

请按以下格式输出：

# 文献卡片

## 论文标题
...

## 作者
...

## 年份
...

## 研究领域
...

## 研究问题
...

## 方法概述
...

## 实验数据集
...

论文原文片段：
{context}
"""


def ensure_required_card_fields(markdown: str) -> str:
    """Append missing card fields defensively."""
    content = markdown.strip()
    if not content:
        return empty_literature_card()

    if not content.startswith("# 文献卡片"):
        content = f"# 文献卡片\n\n{content}"

    for field in CARD_FIELDS:
        if f"## {field}" not in content:
            content = f"{content.rstrip()}\n\n## {field}\n{UNKNOWN_VALUE}"

    return content.strip()


class CardPipeline:
    """Generate literature cards from persisted paper chunks."""

    def __init__(self, llm_client: LLMClient | None = None) -> None:
        self.llm_client = llm_client or LLMClient()

    def generate_literature_card(self, paper_id: str) -> str:
        """Generate one markdown literature card for a paper."""
        chunks = get_paper_chunks(paper_id)
        context = select_card_context(chunks)
        if not context:
            return empty_literature_card()

        prompt = build_card_prompt(context)
        markdown = self.llm_client.generate(prompt, temperature=0.2, max_tokens=1600)
        return ensure_required_card_fields(markdown)

    def create_cards(self, paper_text: str) -> list[dict[str, str]]:
        """Create a simple card from raw text for backward compatibility."""
        prompt = build_card_prompt(paper_text[:18000])
        markdown = self.llm_client.generate(prompt, temperature=0.2, max_tokens=1600)
        return [{"type": "literature_card", "markdown": ensure_required_card_fields(markdown)}]


def generate_literature_card(paper_id: str) -> str:
    """Generate a markdown literature card for a paper."""
    return CardPipeline().generate_literature_card(paper_id)
