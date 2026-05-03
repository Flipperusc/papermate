"""RAG question-answering pipeline."""

from __future__ import annotations

from typing import Any

from src.llm_client import LLMClient
from src.vector_store import VectorStore


REFUSAL_ANSWER = "论文原文中没有找到足够依据回答这个问题"


def build_rag_prompt(question: str, chunks: list[dict[str, Any]]) -> str:
    """Build a strict RAG prompt from retrieved chunks."""
    references: list[str] = []
    for index, chunk in enumerate(chunks, start=1):
        source_id = f"S{index}"
        section_title = chunk.get("section_title") or "未识别章节"
        references.append(
            "\n".join(
                [
                    f"[{source_id}]",
                    f"chunk_id: {chunk['chunk_id']}",
                    f"page_num: {chunk['page_num']}",
                    f"section_title: {section_title}",
                    "text:",
                    str(chunk["text"]),
                ]
            )
        )

    return f"""请根据下面的论文参考片段回答问题。

规则：
1. 只能根据参考片段回答，不允许使用片段之外的信息。
2. 如果参考片段不足以回答，必须原样回答：{REFUSAL_ANSWER}
3. 每个关键结论尽量标注来源，使用片段编号，例如 [S1]、[S2]。
4. 不要编造页码、章节名或引用来源。

问题：{question}

参考片段：
{chr(10).join(references)}
"""


def build_citations(chunks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Build citations strictly from retrieved chunk metadata."""
    citations: list[dict[str, Any]] = []
    seen_chunk_ids: set[str] = set()

    for index, chunk in enumerate(chunks, start=1):
        chunk_id = str(chunk["chunk_id"])
        if chunk_id in seen_chunk_ids:
            continue

        seen_chunk_ids.add(chunk_id)
        citations.append(
            {
                "source_id": f"S{index}",
                "chunk_id": chunk_id,
                "paper_id": chunk.get("paper_id", ""),
                "page_num": chunk.get("page_num", 0),
                "section_title": chunk.get("section_title") or "未识别章节",
            }
        )

    return citations


def build_source_chunks(chunks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return source snippets for display without trusting model-generated citations."""
    source_chunks: list[dict[str, Any]] = []
    for index, chunk in enumerate(chunks, start=1):
        source_chunks.append(
            {
                "source_id": f"S{index}",
                "chunk_id": chunk["chunk_id"],
                "page_num": chunk.get("page_num", 0),
                "section_title": chunk.get("section_title") or "未识别章节",
                "text": chunk.get("text", ""),
            }
        )
    return source_chunks


class RAGPipeline:
    """Coordinate retrieval and answer generation."""

    def __init__(
        self,
        vector_store: VectorStore | None = None,
        llm_client: LLMClient | None = None,
        top_k: int = 5,
    ) -> None:
        self.vector_store = vector_store or VectorStore()
        self.llm_client = llm_client or LLMClient()
        self.top_k = top_k

    def answer_question(self, paper_id: str, question: str) -> dict[str, Any]:
        """Answer a question using retrieved paper context."""
        clean_question = question.strip()
        if not clean_question:
            return {"answer": REFUSAL_ANSWER, "citations": [], "source_chunks": []}

        chunks = self.vector_store.search(paper_id, clean_question, top_k=self.top_k)
        if not chunks:
            return {"answer": REFUSAL_ANSWER, "citations": [], "source_chunks": []}

        prompt = build_rag_prompt(clean_question, chunks)
        answer = self.llm_client.generate(prompt)

        return {
            "answer": answer,
            "citations": build_citations(chunks),
            "source_chunks": build_source_chunks(chunks),
        }

    def answer(self, question: str) -> str:
        """Backward-compatible placeholder method."""
        raise ValueError("RAGPipeline.answer requires a paper_id; use answer_question instead.")


def answer_question(paper_id: str, question: str) -> dict[str, Any]:
    """Answer a question for a paper using the default RAG pipeline."""
    return RAGPipeline().answer_question(paper_id, question)
