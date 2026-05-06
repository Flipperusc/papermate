"""Smoke tests for Reciprocal Rank Fusion."""

from __future__ import annotations

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.retrieval.rrf import reciprocal_rank_fusion


def make_result(chunk_id: str, rank: int, source: str, **extra) -> dict:
    """Create one test retrieval result."""
    return {
        "chunk_id": chunk_id,
        "rank": rank,
        "paper_id": "paper_1",
        "chunk_index": rank,
        "text": f"text for {chunk_id}",
        "retrieval_source": source,
        **extra,
    }


def main() -> None:
    """Run lightweight assertions without requiring pytest."""
    vector_results = [
        make_result("chunk_A", 1, "vector", vector_distance=0.1),
        make_result("chunk_B", 2, "vector", vector_distance=0.2),
        make_result("chunk_D", 3, "vector", vector_distance=0.3),
    ]
    bm25_results = [
        make_result("chunk_C", 1, "bm25", bm25_score=9.0),
        make_result("chunk_A", 2, "bm25", bm25_score=8.0),
        make_result("chunk_E", 3, "bm25", bm25_score=7.0),
    ]

    fused = reciprocal_rank_fusion([vector_results, bm25_results])
    chunk_a = next(item for item in fused if item["chunk_id"] == "chunk_A")
    assert set(chunk_a["retrieval_sources"]) == {"vector", "bm25"}
    assert chunk_a["source_ranks"]["vector"] == 1
    assert chunk_a["source_ranks"]["bm25"] == 2

    scores = [item["rrf_score"] for item in fused]
    assert scores == sorted(scores, reverse=True)
    assert len({item["chunk_id"] for item in fused}) == len(fused)

    weighted = reciprocal_rank_fusion([vector_results, bm25_results], weights=[1.3, 0.8])
    assert weighted
    assert weighted[0]["rrf_score"] >= weighted[-1]["rrf_score"]

    print("rrf tests passed")


if __name__ == "__main__":
    main()
