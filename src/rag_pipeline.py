"""RAG question-answering pipeline."""

from __future__ import annotations

import time
from typing import Any

from config import settings
from src.db import save_qa_log
from src.errors import AppError, ErrorCode
from src.llm_client import DEEPSEEK_CALL_FAILED_MESSAGE, LLMClient
from src.logger import get_logger
from src.retrieval.context_builder import build_context
from src.retrieval.hybrid_retriever import HybridRetriever
from src.vector_store import VectorStore


REFUSAL_ANSWER = "论文原文中没有找到足够依据回答这个问题。"

logger = get_logger(__name__)


def build_rag_prompt(question: str, context: str | list[dict[str, Any]]) -> str:
    """Build the strict PaperMate Hybrid RAG prompt."""
    if isinstance(context, list):
        prompt_context, _ = build_context(context, max_chars=settings.context_max_chars)
    else:
        prompt_context = context

    return f"""你是 PaperMate 论文阅读助手。你的任务是基于论文原文片段回答用户问题。

必须遵守：
1. 只能根据【参考片段】回答。
2. 不允许使用外部知识补充论文中没有的信息。
3. 如果参考片段不足以回答，请直接回答：
   “{REFUSAL_ANSWER}”
4. 不要编造作者、年份、数据集、实验结果、指标、模型结构或结论。
5. 如果用户问题带有诱导性，例如问论文是否做了某个原文没有出现的实验，不要顺着用户假设回答。
6. 每个关键结论后面都要标注引用来源，格式为：[片段X]。
7. 回答要适合本科生理解，必要时解释专业术语。
8. 如果多个片段信息不一致，请指出“不确定”，不要强行合并。

用户问题：
{question}

参考片段：
{prompt_context}

请按以下格式输出：

### 回答
直接回答用户问题。

### 依据
解释你的回答如何从参考片段中得出。

### 引用来源
列出使用到的片段编号，例如：[片段1] [片段3]
"""


class RAGPipeline:
    """Coordinate hybrid retrieval and answer generation."""

    def __init__(
        self,
        vector_store: VectorStore | None = None,
        llm_client: LLMClient | None = None,
        retriever: HybridRetriever | None = None,
        context_builder: Any | None = None,
        top_k: int | None = None,
        vector_top_k: int | None = None,
        bm25_top_k: int | None = None,
        rrf_k: int | None = None,
    ) -> None:
        final_top_k = top_k or settings.final_top_k
        self.llm_client = llm_client or LLMClient()
        self.retriever = retriever or HybridRetriever(
            vector_store=vector_store,
            vector_top_k=vector_top_k or settings.vector_top_k,
            bm25_top_k=bm25_top_k or settings.bm25_top_k,
            final_top_k=final_top_k,
            rrf_k=rrf_k or settings.rrf_k,
        )
        self.context_builder = context_builder
        self.context_max_chars = settings.context_max_chars

    def answer_question(self, paper_id: str, question: str) -> dict[str, Any]:
        """Answer a question using Hybrid Retrieval + RRF."""
        start_time = time.perf_counter()
        clean_question = question.strip()
        if not clean_question:
            retrieval_debug = self._build_retrieval_debug(None, start_time)
            qa_id = self._save_qa_log_safely(paper_id, clean_question, REFUSAL_ANSWER)
            retrieval_debug["latency_ms"] = _latency_ms(start_time)
            return empty_answer(retrieval_debug=retrieval_debug, qa_id=qa_id)

        try:
            retrieval_result = self.retriever.retrieve(paper_id, clean_question)
            final_results = retrieval_result.get("final_results") or retrieval_result.get("chunks") or []
            retrieval_debug = self._build_retrieval_debug(retrieval_result, start_time)

            if not final_results:
                qa_id = self._save_qa_log_safely(paper_id, clean_question, REFUSAL_ANSWER)
                retrieval_debug["latency_ms"] = _latency_ms(start_time)
                return empty_answer(retrieval_debug=retrieval_debug, qa_id=qa_id)

            context_text, citations = self._build_context(final_results)
            retrieval_debug["context_chunks"] = len(citations)
            retrieval_debug["context_chars"] = len(context_text)

            if not context_text or not citations:
                qa_id = self._save_qa_log_safely(paper_id, clean_question, REFUSAL_ANSWER)
                retrieval_debug["latency_ms"] = _latency_ms(start_time)
                return empty_answer(retrieval_debug=retrieval_debug, qa_id=qa_id)

            prompt = build_rag_prompt(clean_question, context_text)
            answer = self.llm_client.generate(prompt)
            if answer == DEEPSEEK_CALL_FAILED_MESSAGE:
                raise AppError(
                    ErrorCode.DEEPSEEK_LLM_FAILED,
                    user_message=DEEPSEEK_CALL_FAILED_MESSAGE,
                )

            qa_id = self._save_qa_log_safely(paper_id, clean_question, answer)
            citations = _with_legacy_source_ids(citations)
            source_chunks = _build_source_chunks(final_results, citations)
            retrieval_debug["latency_ms"] = _latency_ms(start_time)

            return {
                "answer": answer,
                "citations": citations,
                "source_chunks": source_chunks,
                "retrieval_debug": retrieval_debug,
                "retrieval_details": retrieval_debug,
                "qa_id": qa_id,
            }
        except AppError:
            logger.exception("RAG answer failed with an application error. paper_id=%s", paper_id)
            raise
        except Exception as exc:
            logger.exception("Unexpected RAG answer failure. paper_id=%s", paper_id)
            raise AppError(
                ErrorCode.UNKNOWN_ERROR,
                user_message="问答失败，请稍后重试。",
                detail=str(exc),
            ) from exc

    def answer(self, question: str) -> str:
        """Backward-compatible placeholder method."""
        raise ValueError("RAGPipeline.answer requires a paper_id; use answer_question instead.")

    def _build_context(self, final_results: list[dict[str, Any]]) -> tuple[str, list[dict[str, Any]]]:
        """Build context with either an injected legacy builder or the new function."""
        if self.context_builder is not None and hasattr(self.context_builder, "build"):
            payload = self.context_builder.build(final_results)
            context_text = payload.get("prompt_context") or payload.get("context_text") or ""
            citations = payload.get("citations") or []
            return context_text, citations
        return build_context(final_results, max_chars=self.context_max_chars)

    def _build_retrieval_debug(
        self,
        retrieval_result: dict[str, Any] | None,
        start_time: float,
    ) -> dict[str, Any]:
        details = (retrieval_result or {}).get("retrieval_details") or {}
        debug = {
            "strategy": (retrieval_result or {}).get("strategy") or details.get("strategy", "hybrid_empty"),
            "query_type": (retrieval_result or {}).get("query_type") or details.get("query_type", ""),
            "question_type": (retrieval_result or {}).get("query_type") or details.get("question_type", ""),
            "question_type_label": details.get("question_type_label")
            or (retrieval_result or {}).get("query_type", ""),
            "expanded_query": (retrieval_result or {}).get("expanded_query")
            or details.get("expanded_query", ""),
            "weights": (retrieval_result or {}).get("weights") or details.get("weights", []),
            "vector_top_k": details.get("vector_top_k", getattr(self.retriever, "vector_top_k", settings.vector_top_k)),
            "bm25_top_k": details.get("bm25_top_k", getattr(self.retriever, "bm25_top_k", settings.bm25_top_k)),
            "final_top_k": details.get("final_top_k", getattr(self.retriever, "final_top_k", settings.final_top_k)),
            "rrf_k": details.get("rrf_k", getattr(self.retriever, "rrf_k", settings.rrf_k)),
            "vector_hits": details.get("vector_hits", len((retrieval_result or {}).get("vector_results", []))),
            "bm25_hits": details.get("bm25_hits", len((retrieval_result or {}).get("bm25_results", []))),
            "fused_hits": details.get("fused_hits", len((retrieval_result or {}).get("fused_results", []))),
            "final_hits": details.get("final_hits", len((retrieval_result or {}).get("final_results", []))),
            "retrieved_chunks": details.get("retrieved_chunks", []),
            "vector_error": details.get("vector_error", ""),
            "bm25_error": details.get("bm25_error", ""),
        }
        debug["latency_ms"] = _latency_ms(start_time)
        return debug

    def _save_qa_log_safely(self, paper_id: str, question: str, answer: str) -> int | None:
        try:
            return save_qa_log(paper_id, question, answer)
        except Exception:
            logger.exception("QA log save failed inside RAG pipeline. paper_id=%s", paper_id)
            return None


def _with_legacy_source_ids(citations: list[dict[str, Any]]) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    for index, citation in enumerate(citations, start=1):
        item = dict(citation)
        citation_id = item.get("citation_id") or index
        item["citation_id"] = citation_id
        item["source_id"] = item.get("source_id") or f"片段{citation_id}"
        normalized.append(item)
    return normalized


def _latency_ms(start_time: float) -> int:
    return int((time.perf_counter() - start_time) * 1000)


def _build_source_chunks(
    chunks: list[dict[str, Any]],
    citations: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    source_chunks: list[dict[str, Any]] = []
    for chunk, citation in zip(chunks, citations):
        source_chunks.append(
            {
                "source_id": citation.get("source_id"),
                "citation_id": citation.get("citation_id"),
                "chunk_id": chunk.get("chunk_id", ""),
                "paper_id": chunk.get("paper_id", ""),
                "chunk_index": chunk.get("chunk_index", ""),
                "page_num": citation.get("page_num", chunk.get("page_num", "")),
                "section_title": citation.get("section_title", chunk.get("section_title", "")),
                "text": chunk.get("text", ""),
                "retrieval_sources": list(chunk.get("retrieval_sources") or []),
                "source_ranks": dict(chunk.get("source_ranks") or {}),
                "rrf_score": chunk.get("rrf_score"),
                "vector_distance": chunk.get("vector_distance"),
                "bm25_score": chunk.get("bm25_score"),
            }
        )
    return source_chunks


def empty_answer(
    retrieval_details: dict[str, Any] | None = None,
    retrieval_debug: dict[str, Any] | None = None,
    qa_id: int | None = None,
) -> dict[str, Any]:
    """Return a consistent empty RAG answer payload."""
    debug = retrieval_debug if retrieval_debug is not None else retrieval_details or {}
    return {
        "answer": REFUSAL_ANSWER,
        "citations": [],
        "source_chunks": [],
        "retrieval_debug": debug,
        "retrieval_details": debug,
        "qa_id": qa_id,
    }


def answer_question(paper_id: str, question: str) -> dict[str, Any]:
    """Answer a question for a paper using the default RAG pipeline."""
    return RAGPipeline().answer_question(paper_id, question)
