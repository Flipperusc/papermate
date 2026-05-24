"""Vector retrieval adapter over the existing Chroma VectorStore."""

from __future__ import annotations

import re
from typing import Any

from src.errors import AppError, ErrorCode
from src.logger import get_logger
from src.vector_store import VectorStore


logger = get_logger(__name__)


class VectorRetriever:
    """Normalize VectorStore.search results for hybrid retrieval."""

    def __init__(self, vector_store=None) -> None:
        self.vector_store = vector_store

    def search(self, paper_id: str, query: str, top_k: int = 20) -> list[dict]:
        """Search Chroma via VectorStore and return a common result shape."""
        try:
            vector_store = self.vector_store or VectorStore()
            matches = vector_store.search(paper_id=paper_id, query=query, top_k=top_k)
            return [self._normalize_match(match, rank, paper_id) for rank, match in enumerate(matches, start=1)]
        except Exception as exc:
            logger.exception("Vector retrieval failed. paper_id=%s", paper_id)
            detail = str(exc)
            if "index_version" in detail or "rebuild index" in detail:
                user_message = "当前论文向量索引版本过旧，请重新构建论文索引。"
            else:
                user_message = "向量检索失败，请检查该论文是否已完成索引。"
            raise AppError(
                code=ErrorCode.VECTOR_SEARCH_FAILED,
                user_message=user_message,
                detail=detail,
            ) from exc

    @staticmethod
    def _normalize_match(match: dict[str, Any], rank: int, paper_id: str) -> dict:
        chunk_id = str(match.get("chunk_id") or "")
        return {
            "chunk_id": chunk_id,
            "paper_id": match.get("paper_id") or paper_id,
            "chunk_index": get_chunk_index(match, rank),
            "section_title": match.get("section_title", ""),
            "page_num": match.get("page_num", ""),
            "chunk_type": match.get("chunk_type", "text"),
            "text": match.get("text", ""),
            "search_text": match.get("search_text", ""),
            "images_json": match.get("images_json", "[]"),
            "tables_json": match.get("tables_json", "[]"),
            "index_version": match.get("index_version", ""),
            "rank": rank,
            "vector_distance": match.get("vector_distance", match.get("distance")),
            "retrieval_source": "vector",
        }


def get_chunk_index(match: dict[str, Any], fallback_rank: int) -> int:
    """Return chunk_index from metadata, chunk_id, or rank fallback."""
    raw_index = match.get("chunk_index")
    if raw_index is not None:
        try:
            return int(raw_index)
        except (TypeError, ValueError):
            pass

    chunk_id = str(match.get("chunk_id") or "")
    match_obj = re.search(r"_chunk_(\d+)$", chunk_id)
    if match_obj:
        return int(match_obj.group(1))

    return fallback_rank
