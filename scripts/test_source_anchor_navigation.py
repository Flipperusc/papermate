"""Smoke tests for source anchor generation used by QA citations."""

from __future__ import annotations

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app import add_chunk_anchors_to_markdown, chunk_anchor_id


def assert_anchor_present(markdown: str, chunk_id: str) -> None:
    anchor = f'id="{chunk_anchor_id(chunk_id)}"'
    assert anchor in markdown, f"missing anchor for {chunk_id}"


def main() -> None:
    markdown = """
# Sample Paper

## Abstract

Alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu.

## Method

The method uses retrieval augmented generation with trusted source citations and page metadata.
The implementation stores chunks in order and renders the original Markdown in the reader.

## Results

The answer cites evidence and should jump back to the matching paragraph in the source reader.
""".strip()

    chunks = [
        {
            "chunk_id": "paper_chunk_0000",
            "chunk_index": 0,
            "page_num": 1,
            "section_title": "Abstract",
            "text": "Alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu.",
        },
        {
            "chunk_id": "paper_chunk_0001",
            "chunk_index": 1,
            "page_num": 1,
            "section_title": "Abstract",
            "text": "This generated multimodal description is not present in the source Markdown.",
        },
        {
            "chunk_id": "paper_chunk_0002",
            "chunk_index": 2,
            "page_num": 2,
            "section_title": "Method",
            "text": "The method uses retrieval augmented generation with trusted source citations and page metadata.",
        },
        {
            "chunk_id": "paper_chunk_0003",
            "chunk_index": 3,
            "page_num": 3,
            "section_title": "Results",
            "text": "The answer cites evidence and should jump back to the matching paragraph in the source reader.",
        },
    ]

    anchored, exact_missing = add_chunk_anchors_to_markdown(markdown, chunks)
    assert [chunk["chunk_id"] for chunk in exact_missing] == ["paper_chunk_0001"]
    for chunk in chunks:
        assert_anchor_present(anchored, chunk["chunk_id"])

    first_position = anchored.index(chunk_anchor_id("paper_chunk_0000"))
    fallback_position = anchored.index(chunk_anchor_id("paper_chunk_0001"))
    method_position = anchored.index(chunk_anchor_id("paper_chunk_0002"))
    assert first_position <= fallback_position < method_position

    print("source anchor navigation smoke test passed")


if __name__ == "__main__":
    main()
