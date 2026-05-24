"""BM25 keyword search over chunks stored in SQLite."""

from __future__ import annotations

import math
from collections import Counter
from functools import lru_cache
from pathlib import Path
from typing import Any

from src.db import get_db_connection, get_paper_chunks
from src.errors import AppError, ErrorCode
from src.logger import get_logger
from src.retrieval.query_processor import QueryPlan
from src.retrieval.tokenization import tokenize


logger = get_logger(__name__)


class BM25Searcher:
    """Build an in-memory BM25 index from SQLite chunks for one paper."""

    def __init__(self, db_path: str | Path | None = None) -> None:
        self.db_path = db_path

    def search(self, paper_id: str, query: QueryPlan | str, top_k: int = 20) -> list[dict[str, Any]]:
        """Return top BM25 chunk matches for one paper."""
        try:
            BM25Okapi = load_bm25_okapi()
            chunks = self._get_chunks(paper_id)
            if not chunks:
                return []

            corpus_tokens = [self._chunk_tokens(chunk) for chunk in chunks]
            if not any(corpus_tokens):
                return []

            query_tokens = self._query_tokens(query)
            if not query_tokens:
                return []

            bm25 = BM25Okapi(corpus_tokens)
            scores = bm25.get_scores(query_tokens)
            ranked_indices = sorted(
                range(len(chunks)),
                key=lambda index: float(scores[index]),
                reverse=True,
            )

            matches: list[dict[str, Any]] = []
            for rank, index in enumerate(ranked_indices, start=1):
                score = float(scores[index])
                if score <= 0:
                    continue

                chunk = dict(chunks[index])
                chunk["bm25_score"] = score
                chunk["bm25_rank"] = rank
                chunk["retrieval_sources"] = ["bm25"]
                matches.append(chunk)
                if len(matches) >= max(1, top_k):
                    break

            return matches
        except AppError:
            raise
        except Exception as exc:
            logger.exception("BM25 search failed. paper_id=%s", paper_id)
            raise AppError(
                ErrorCode.BM25_SEARCH_FAILED,
                user_message="关键词检索失败，请稍后重试。",
                detail=str(exc),
            ) from exc

    @staticmethod
    def _query_tokens(query: QueryPlan | str) -> list[str]:
        if isinstance(query, QueryPlan):
            return query.expanded_terms or tokenize(query.normalized_query)
        return tokenize(str(query))

    def _get_chunks(self, paper_id: str) -> list[dict[str, Any]]:
        if self.db_path is None:
            return get_paper_chunks(paper_id)

        connection = get_db_connection(self.db_path)
        try:
            rows = connection.execute(
                """
                SELECT
                    chunk_id,
                    paper_id,
                    chunk_index,
                    page_num,
                    section_title,
                    text,
                    chunk_type,
                    images_json,
                    tables_json
                FROM chunks
                WHERE paper_id = ?
                ORDER BY chunk_index ASC
                """,
                (paper_id,),
            ).fetchall()
        finally:
            connection.close()
        return [dict(row) for row in rows]

    @staticmethod
    def _chunk_tokens(chunk: dict[str, Any]) -> list[str]:
        section_title = str(chunk.get("section_title") or "")
        text = str(chunk.get("text") or "")
        section_tokens = tokenize(section_title)
        return [*section_tokens, *section_tokens, *tokenize(text)]


@lru_cache(maxsize=1)
def load_bm25_okapi():
    """Load rank_bm25 when available, otherwise use the local implementation."""
    try:
        from rank_bm25 import BM25Okapi
    except ImportError:
        logger.warning("rank_bm25 is not installed; using built-in BM25 fallback.")
        return SimpleBM25Okapi
    return BM25Okapi


class SimpleBM25Okapi:
    """Small BM25Okapi-compatible fallback used when rank_bm25 is unavailable."""

    def __init__(self, corpus: list[list[str]], k1: float = 1.5, b: float = 0.75) -> None:
        self.corpus = corpus
        self.k1 = k1
        self.b = b
        self.doc_freqs = [Counter(document) for document in corpus]
        self.doc_len = [len(document) for document in corpus]
        self.corpus_size = len(corpus)
        self.avgdl = sum(self.doc_len) / self.corpus_size if self.corpus_size else 0.0
        self.idf = self._build_idf()

    def get_scores(self, query: list[str]) -> list[float]:
        """Return BM25 scores for all documents."""
        if not self.corpus or self.avgdl <= 0:
            return [0.0 for _ in self.corpus]

        scores: list[float] = []
        for frequencies, doc_len in zip(self.doc_freqs, self.doc_len):
            score = 0.0
            for term in query:
                term_frequency = frequencies.get(term, 0)
                if term_frequency <= 0:
                    continue
                denominator = term_frequency + self.k1 * (
                    1.0 - self.b + self.b * doc_len / self.avgdl
                )
                score += self.idf.get(term, 0.0) * term_frequency * (self.k1 + 1.0) / denominator
            scores.append(score)
        return scores

    def _build_idf(self) -> dict[str, float]:
        document_frequency: Counter[str] = Counter()
        for frequencies in self.doc_freqs:
            document_frequency.update(frequencies.keys())

        return {
            term: math.log(1.0 + (self.corpus_size - freq + 0.5) / (freq + 0.5))
            for term, freq in document_frequency.items()
        }
