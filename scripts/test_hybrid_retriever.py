"""Smoke tests for HybridRetriever."""

from __future__ import annotations

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.errors import AppError, ErrorCode
from src.retrieval.hybrid_retriever import HybridRetriever


class FakeVectorRetriever:
    def __init__(self, should_fail: bool = False) -> None:
        self.should_fail = should_fail

    def search(self, paper_id: str, query: str, top_k: int = 20) -> list[dict]:
        if self.should_fail:
            raise AppError(
                ErrorCode.VECTOR_SEARCH_FAILED,
                user_message="向量检索失败，请检查该论文是否已完成索引。",
            )
        return [
            {
                "chunk_id": "chunk_A",
                "paper_id": paper_id,
                "chunk_index": 1,
                "section_title": "Method",
                "page_num": 1,
                "text": "vector text",
                "rank": 1,
                "vector_distance": 0.1,
                "retrieval_source": "vector",
            }
        ][:top_k]


class FakeBM25Store:
    def __init__(self, should_fail: bool = False, missing: bool = False) -> None:
        self.should_fail = should_fail
        self.missing = missing

    def search(self, paper_id: str, query: str, top_k: int = 20) -> list[dict]:
        if self.missing:
            raise AppError(
                ErrorCode.BM25_INDEX_MISSING,
                user_message="当前论文关键词索引不存在，请重新构建论文索引。",
            )
        if self.should_fail:
            raise AppError(
                ErrorCode.BM25_SEARCH_FAILED,
                user_message="关键词检索失败，请重新构建索引或查看日志。",
            )
        return [
            {
                "chunk_id": "chunk_B",
                "paper_id": paper_id,
                "chunk_index": 2,
                "section_title": "Experiments",
                "page_num": 2,
                "text": "bm25 text dataset",
                "rank": 1,
                "bm25_score": 3.2,
                "retrieval_source": "bm25",
            }
        ][:top_k]


def main() -> None:
    hybrid = HybridRetriever(
        vector_retriever=FakeVectorRetriever(),
        bm25_store=FakeBM25Store(),
    )
    result = hybrid.retrieve("paper_1", "本文用了哪些数据集？")
    assert isinstance(result, dict)
    assert "final_results" in result
    assert result["final_results"]
    assert result["strategy"] == "hybrid_rrf"

    vector_fallback = HybridRetriever(
        vector_retriever=FakeVectorRetriever(),
        bm25_store=FakeBM25Store(missing=True),
    ).retrieve("paper_1", "dataset")
    assert vector_fallback["strategy"] == "vector_fallback"
    assert vector_fallback["final_results"][0]["retrieval_sources"] == ["vector"]
    assert "source_ranks" in vector_fallback["final_results"][0]

    bm25_fallback = HybridRetriever(
        vector_retriever=FakeVectorRetriever(should_fail=True),
        bm25_store=FakeBM25Store(),
    ).retrieve("paper_1", "dataset")
    assert bm25_fallback["strategy"] == "bm25_fallback"
    assert bm25_fallback["final_results"][0]["retrieval_sources"] == ["bm25"]
    assert "source_ranks" in bm25_fallback["final_results"][0]

    print("hybrid retriever tests passed")


if __name__ == "__main__":
    main()
