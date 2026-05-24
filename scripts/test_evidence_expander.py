"""Smoke tests for neighbor evidence expansion."""

from __future__ import annotations

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.retrieval.evidence_expander import EvidenceExpander


def fake_loader(paper_id: str) -> list[dict]:
    return [
        {"chunk_id": "p1_c0", "paper_id": paper_id, "chunk_index": 0, "text": "before"},
        {"chunk_id": "p1_c1", "paper_id": paper_id, "chunk_index": 1, "text": "core one"},
        {"chunk_id": "p1_c2", "paper_id": paper_id, "chunk_index": 2, "text": "core two"},
        {"chunk_id": "p1_c3", "paper_id": paper_id, "chunk_index": 3, "text": "after"},
        {"chunk_id": "other_c2", "paper_id": "other", "chunk_index": 2, "text": "wrong paper"},
    ]


def main() -> None:
    core_chunks = [
        {
            "chunk_id": "p1_c1",
            "paper_id": "paper_1",
            "chunk_index": 1,
            "text": "core one",
            "final_score": 0.9,
            "retrieval_sources": ["vector", "bm25"],
            "source_ranks": {"vector": 1},
        },
        {
            "chunk_id": "p1_c2",
            "paper_id": "paper_1",
            "chunk_index": 2,
            "text": "core two",
            "final_score": 0.8,
            "retrieval_sources": ["bm25"],
            "source_ranks": {"bm25": 1},
        },
    ]

    expanded = EvidenceExpander(window=1, chunk_loader=fake_loader).expand("paper_1", core_chunks)
    ids = [chunk["chunk_id"] for chunk in expanded]
    assert ids[:2] == ["p1_c1", "p1_c2"]
    assert ids == ["p1_c1", "p1_c2", "p1_c0", "p1_c3"]
    assert "other_c2" not in ids
    assert expanded[2]["expanded_neighbor"] is True
    assert expanded[2]["parent_chunk_id"] == "p1_c1"
    assert expanded[2]["retrieval_sources"] == ["neighbor"]

    no_expand = EvidenceExpander(window=0, chunk_loader=fake_loader).expand("paper_1", core_chunks)
    assert [chunk["chunk_id"] for chunk in no_expand] == ["p1_c1", "p1_c2"]
    print("evidence expander tests passed")


if __name__ == "__main__":
    main()
