"""Persistent BM25 keyword index for paper chunks."""

from __future__ import annotations

import json
import math
import pickle
from collections import Counter
from pathlib import Path
from typing import Any

from src.errors import AppError, ErrorCode
from src.logger import get_logger
from src.retrieval.tokenizer import tokenize_text


logger = get_logger(__name__)


class BM25Store:
    """Build and search persistent BM25 indexes for paper chunks."""

    def __init__(self, index_dir: str = "data/bm25") -> None:
        self.index_dir = Path(index_dir)

    def build_index(self, paper_id: str, chunks: list[dict]) -> dict:
        """Build and persist a BM25 index for one paper."""
        try:
            from rank_bm25 import BM25Okapi
        except ImportError:
            BM25Okapi = SimpleBM25Okapi

        try:
            clean_paper_id = str(paper_id).strip()
            if not clean_paper_id:
                raise ValueError("paper_id is required")
            if not chunks:
                raise ValueError("chunks is empty")

            self.index_dir.mkdir(parents=True, exist_ok=True)
            payloads = [self._build_payload(chunk) for chunk in chunks]
            tokenized_corpus = [tokenize_text(payload["text"]) for payload in payloads]
            bm25 = BM25Okapi(tokenized_corpus)

            index_path = self._index_path(clean_paper_id)
            payload_path = self._payload_path(clean_paper_id)

            # The pickle stores BM25 statistics; the JSON payload keeps chunk
            # metadata readable and independent from the BM25 implementation.
            with index_path.open("wb") as file:
                pickle.dump(bm25, file)
            with payload_path.open("w", encoding="utf-8") as file:
                json.dump(payloads, file, ensure_ascii=False, indent=2)

            return {
                "paper_id": clean_paper_id,
                "chunk_count": len(payloads),
                "index_path": str(index_path),
                "payload_path": str(payload_path),
            }
        except AppError:
            raise
        except Exception as exc:
            logger.exception("BM25 index build failed. paper_id=%s", paper_id)
            raise AppError(
                code=ErrorCode.BM25_INDEX_FAILED,
                user_message="关键词索引构建失败，请重新构建索引或查看日志。",
                detail=str(exc),
            ) from exc

    def search(self, paper_id: str, query: str, top_k: int = 20) -> list[dict]:
        """Search a persisted BM25 index for one paper."""
        index_path = self._index_path(paper_id)
        payload_path = self._payload_path(paper_id)
        if not index_path.exists() or not payload_path.exists():
            raise AppError(
                code=ErrorCode.BM25_INDEX_MISSING,
                user_message="当前论文关键词索引不存在，请重新构建论文索引。",
            )

        try:
            with index_path.open("rb") as file:
                bm25 = pickle.load(file)
            with payload_path.open("r", encoding="utf-8") as file:
                payloads = json.load(file)

            query_tokens = tokenize_text(query)
            if not query_tokens:
                return []

            scores = bm25.get_scores(query_tokens)
            ranked_indices = sorted(
                range(len(payloads)),
                key=lambda index: float(scores[index]),
                reverse=True,
            )

            results: list[dict] = []
            for index in ranked_indices:
                score = float(scores[index])
                if score <= 0:
                    continue

                payload = dict(payloads[index])
                results.append(
                    {
                        "chunk_id": payload["chunk_id"],
                        "paper_id": payload["paper_id"],
                        "chunk_index": payload["chunk_index"],
                        "section_title": payload.get("section_title", ""),
                        "page_num": payload.get("page_num", ""),
                        "text": payload.get("text", ""),
                        "rank": len(results) + 1,
                        "bm25_score": score,
                        "retrieval_source": "bm25",
                    }
                )
                if len(results) >= max(1, top_k):
                    break

            return results
        except AppError:
            raise
        except Exception as exc:
            logger.exception("BM25 search failed. paper_id=%s", paper_id)
            raise AppError(
                code=ErrorCode.BM25_SEARCH_FAILED,
                user_message="关键词检索失败，请重新构建索引或查看日志。",
                detail=str(exc),
            ) from exc

    def _index_path(self, paper_id: str) -> Path:
        return self.index_dir / f"{paper_id}_bm25.pkl"

    def _payload_path(self, paper_id: str) -> Path:
        return self.index_dir / f"{paper_id}_payloads.json"

    @staticmethod
    def _build_payload(chunk: dict) -> dict:
        return {
            "chunk_id": str(chunk["chunk_id"]),
            "paper_id": str(chunk["paper_id"]),
            "chunk_index": int(chunk["chunk_index"]),
            "section_title": str(chunk.get("section_title") or ""),
            "page_num": chunk.get("page_num", ""),
            "text": str(chunk["text"]),
        }


class SimpleBM25Okapi:
    """Small BM25Okapi-compatible fallback for local environments missing rank_bm25."""

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
