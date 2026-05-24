"""Smoke tests for context builder."""

from __future__ import annotations

from copy import deepcopy
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.retrieval.context_builder import build_context


def main() -> None:
    chunks = [
        {
            "chunk_id": "chunk_A",
            "paper_id": "paper_1",
            "chunk_index": 1,
            "page_num": 5,
            "section_title": "Experiments",
            "text": "This paper evaluates ImageNet and CIFAR-10 with accuracy and F1.",
            "rrf_score": 0.032,
            "retrieval_sources": ["vector", "bm25"],
            "source_ranks": {"vector": 2, "bm25": 5},
            "vector_distance": 0.123,
            "bm25_score": 8.9,
        },
        {
            "chunk_id": "chunk_B",
            "paper_id": "paper_1",
            "chunk_index": 2,
            "section_title": "Method",
            "text": "The framework uses a two-stage encoder architecture.",
            "rrf_score": 0.021,
            "retrieval_sources": ["vector"],
            "source_ranks": {"vector": 1},
            "vector_distance": 0.08,
        },
        {
            "chunk_id": "chunk_C",
            "paper_id": "paper_1",
            "chunk_index": 3,
            "page_num": "第7页",
            "text": "Ablation results are reported in Table 2.",
            "rrf_score": 0.019,
            "retrieval_sources": ["bm25"],
            "source_ranks": {"bm25": 1},
            "bm25_score": 6.4,
        },
    ]
    original = deepcopy(chunks)

    context_text, citations = build_context(chunks, max_chars=2000)
    assert "[片段1" in context_text
    assert "页码=第5页" in context_text
    assert "页码=未知页" in context_text
    assert "章节=未知章节" in context_text
    assert len(citations) == 3
    assert citations[1]["page_num"] == "未知页"
    assert citations[2]["section_title"] == "未知章节"
    assert citations[0]["text_preview"] == chunks[0]["text"][:300]
    assert citations[0]["retrieval_sources"] == ["vector", "bm25"]
    assert citations[0]["source_ranks"] == {"vector": 2, "bm25": 5}
    assert chunks == original

    first_context, _ = build_context([chunks[0]], max_chars=2000)
    limited_text, limited_citations = build_context(chunks, max_chars=len(first_context))
    assert len(limited_text) <= len(first_context)
    assert len(limited_citations) == 1

    neighbor = {
        "chunk_id": "chunk_neighbor",
        "paper_id": "paper_1",
        "chunk_index": 2,
        "page_num": 5,
        "section_title": "Experiments",
        "text": "neighbor " * 200,
        "expanded_neighbor": True,
        "parent_chunk_id": "chunk_A",
    }
    priority_text, priority_citations = build_context([chunks[0], neighbor], max_chars=len(first_context) + 20)
    assert "chunk_A" in priority_text
    assert "chunk_neighbor" not in priority_text
    assert len(priority_citations) == 1

    empty_text, empty_citations = build_context(chunks, max_chars=10)
    assert empty_text == ""
    assert empty_citations == []

    print("context builder tests passed")


if __name__ == "__main__":
    main()
