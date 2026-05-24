"""Smoke tests for BM25 indexing and index-version checks."""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import settings
from src.errors import AppError, ErrorCode
from src.retrieval.bm25_store import BM25Store
from src.retrieval.constants import RETRIEVAL_INDEX_VERSION


def main() -> None:
    assert BM25Store().index_dir == settings.bm25_dir

    chunks = [
        {
            "chunk_id": "chunk_A",
            "paper_id": "paper_1",
            "chunk_index": 1,
            "page_num": 2,
            "section_title": "Experiments",
            "text": "Table 2 reports CIFAR-10 accuracy and F1.",
        },
        {
            "chunk_id": "chunk_B",
            "paper_id": "paper_1",
            "chunk_index": 2,
            "page_num": 3,
            "section_title": "Method",
            "text": "The encoder has two attention layers.",
        },
    ]

    with tempfile.TemporaryDirectory() as tmp_dir:
        store = BM25Store(index_dir=tmp_dir)
        build_result = store.build_index("paper_1", chunks)
        assert build_result["index_version"] == RETRIEVAL_INDEX_VERSION

        results = store.search("paper_1", "Table 2 CIFAR-10 metric", top_k=2)
        assert results[0]["chunk_id"] == "chunk_A"
        assert results[0]["index_version"] == RETRIEVAL_INDEX_VERSION
        assert "search_text" in results[0]

        payload_path = Path(build_result["payload_path"])
        payloads = json.loads(payload_path.read_text(encoding="utf-8"))
        payloads[0].pop("index_version", None)
        payload_path.write_text(json.dumps(payloads, ensure_ascii=False), encoding="utf-8")
        try:
            store.search("paper_1", "Table 2", top_k=1)
        except AppError as exc:
            assert exc.code == ErrorCode.BM25_INDEX_MISSING
        else:
            raise AssertionError("stale BM25 payload should request reindex")

    print("bm25 store tests passed")


if __name__ == "__main__":
    main()
