"""Hybrid vector + BM25 retrieval with RRF fusion."""

from __future__ import annotations

from typing import Any

from config import settings
from src.errors import AppError, ErrorCode
from src.logger import get_logger
from src.retrieval.bm25_store import BM25Store
from src.retrieval.query_processor import classify_query, expand_query, get_rrf_weights
from src.retrieval.rrf import reciprocal_rank_fusion
from src.retrieval.vector_retriever import VectorRetriever


logger = get_logger(__name__)


class HybridRetriever:
    """Combine Chroma vector retrieval and BM25 retrieval with RRF."""

    def __init__(
        self,
        vector_retriever=None,
        bm25_store=None,
        vector_top_k: int = 20,
        bm25_top_k: int = 20,
        final_top_k: int = 6,
        rrf_k: int = 60,
        vector_store=None,
        bm25_searcher=None,
    ) -> None:
        # vector_store and bm25_searcher are accepted for compatibility with
        # earlier local code; the new public API is vector_retriever/bm25_store.
        self.vector_retriever = vector_retriever or VectorRetriever(vector_store=vector_store)
        self.bm25_store = bm25_store or bm25_searcher or BM25Store()
        self.vector_top_k = max(1, int(vector_top_k or settings.vector_top_k))
        self.bm25_top_k = max(1, int(bm25_top_k or settings.bm25_top_k))
        self.final_top_k = max(1, int(final_top_k or settings.final_top_k))
        self.rrf_k = max(1, int(rrf_k or settings.rrf_k))

    def retrieve(self, paper_id: str, question: str) -> dict[str, Any]:
        """Retrieve and fuse context chunks for one paper question."""
        query_type = classify_query(question)
        expanded_query = expand_query(question)
        weights = get_rrf_weights(query_type)

        # Search failures are captured per channel so vector-only or BM25-only
        # retrieval can still answer when the other index is unavailable.
        vector_results, vector_error = self._safe_vector_search(paper_id, question)
        bm25_results, bm25_error = self._safe_bm25_search(paper_id, expanded_query)

        available_lists: list[list[dict]] = []
        available_weights: list[float] = []
        if vector_results:
            available_lists.append(vector_results)
            available_weights.append(weights[0])
        if bm25_results:
            available_lists.append(bm25_results)
            available_weights.append(weights[1])

        fused_results = reciprocal_rank_fusion(
            available_lists,
            rrf_k=self.rrf_k,
            weights=available_weights if available_lists else None,
        )
        final_results = fused_results[: self.final_top_k]
        strategy = self._strategy(vector_results, bm25_results, vector_error, bm25_error)

        retrieval_details = self._build_details(
            query_type=query_type,
            expanded_query=expanded_query,
            weights=weights,
            vector_results=vector_results,
            bm25_results=bm25_results,
            fused_results=fused_results,
            final_results=final_results,
            strategy=strategy,
            vector_error=vector_error,
            bm25_error=bm25_error,
        )

        return {
            "strategy": strategy,
            "query_type": query_type,
            "expanded_query": expanded_query,
            "weights": weights,
            "vector_results": vector_results,
            "bm25_results": bm25_results,
            "fused_results": fused_results,
            "final_results": final_results,
            # Backward-compatible aliases used by the current RAG pipeline UI.
            "chunks": final_results,
            "retrieval_details": retrieval_details,
        }

    def _safe_vector_search(self, paper_id: str, question: str) -> tuple[list[dict], Exception | None]:
        try:
            return self.vector_retriever.search(paper_id, question, self.vector_top_k), None
        except Exception as exc:
            logger.warning(
                "Vector search failed; hybrid retrieval may fallback. paper_id=%s error=%s",
                paper_id,
                exc,
            )
            return [], exc

    def _safe_bm25_search(self, paper_id: str, expanded_query: str) -> tuple[list[dict], Exception | None]:
        try:
            return self.bm25_store.search(paper_id, expanded_query, self.bm25_top_k), None
        except AppError as exc:
            if exc.code == ErrorCode.BM25_INDEX_MISSING:
                logger.warning("BM25 index missing; falling back to vector search. paper_id=%s", paper_id)
            else:
                logger.warning(
                    "BM25 search failed; hybrid retrieval may fallback. paper_id=%s error=%s",
                    paper_id,
                    exc,
                )
            return [], exc
        except Exception as exc:
            logger.warning(
                "BM25 search failed; hybrid retrieval may fallback. paper_id=%s error=%s",
                paper_id,
                exc,
            )
            return [], exc

    @staticmethod
    def _strategy(
        vector_results: list[dict],
        bm25_results: list[dict],
        vector_error: Exception | None,
        bm25_error: Exception | None,
    ) -> str:
        if vector_results and bm25_results:
            return "hybrid_rrf"
        if bm25_results and (vector_error or not vector_results):
            return "bm25_fallback"
        if vector_results and (bm25_error or not bm25_results):
            return "vector_fallback"
        return "hybrid_empty"

    def _build_details(
        self,
        query_type: str,
        expanded_query: str,
        weights: list[float],
        vector_results: list[dict],
        bm25_results: list[dict],
        fused_results: list[dict],
        final_results: list[dict],
        strategy: str,
        vector_error: Exception | None,
        bm25_error: Exception | None,
    ) -> dict[str, Any]:
        return {
            "strategy": strategy,
            "query_type": query_type,
            "question_type": query_type,
            "question_type_label": query_type,
            "expanded_query": expanded_query,
            "weights": weights,
            "vector_top_k": self.vector_top_k,
            "bm25_top_k": self.bm25_top_k,
            "final_top_k": self.final_top_k,
            "rrf_k": self.rrf_k,
            "vector_hits": len(vector_results),
            "bm25_hits": len(bm25_results),
            "fused_hits": len(fused_results),
            "final_hits": len(final_results),
            "vector_error": type(vector_error).__name__ if vector_error else "",
            "bm25_error": type(bm25_error).__name__ if bm25_error else "",
            "retrieved_chunks": [
                {
                    "rank": index,
                    "chunk_id": chunk.get("chunk_id"),
                    "page_num": chunk.get("page_num", ""),
                    "section_title": chunk.get("section_title", ""),
                    "retrieval_sources": chunk.get("retrieval_sources", []),
                    "source_ranks": chunk.get("source_ranks", {}),
                    "rrf_score": chunk.get("rrf_score"),
                    "vector_rank": chunk.get("source_ranks", {}).get("vector"),
                    "bm25_rank": chunk.get("source_ranks", {}).get("bm25"),
                    "bm25_score": chunk.get("bm25_score"),
                    "vector_distance": chunk.get("vector_distance"),
                }
                for index, chunk in enumerate(final_results, start=1)
            ],
        }
