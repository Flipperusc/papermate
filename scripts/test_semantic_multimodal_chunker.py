"""Offline tests for semantic multimodal chunking."""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import settings
from src.chunker import chunk_pages
from src.db import get_paper_chunks, init_db, save_paper_and_chunks
from src.pdf_parser import elements_from_content_list
from src.retrieval.bm25_store import BM25Store
from src.vector_store import VectorStore


class FakeEmbeddingClient:
    """Return deterministic vectors that create one semantic breakpoint."""

    def embed(self, texts: list[str]) -> list[list[float]]:
        vectors: list[list[float]] = []
        for text in texts:
            lowered = text.lower()
            if "gamma" in lowered or "delta" in lowered:
                vectors.append([0.0, 1.0])
            else:
                vectors.append([1.0, 0.0])
        return vectors


class FailingEmbeddingClient:
    def embed(self, texts: list[str]) -> list[list[float]]:
        raise RuntimeError("embedding failed")


class FakeVLMClient:
    def describe(self, image: dict) -> str:
        return f"VLM description for {image.get('caption') or image.get('path')}"


def main() -> None:
    test_semantic_breakpoint_and_overlap()
    test_image_binding()
    test_table_modes()
    test_legacy_pages()
    test_embedding_failure()
    test_multimodal_chain_metadata_roundtrip()
    print("semantic multimodal chunker tests passed")


def test_semantic_breakpoint_and_overlap() -> None:
    elements = [
        {
            "type": "text",
            "order": 0,
            "page_num": 1,
            "text": "1 Introduction\nAlpha stays close. Alpha remains close. Gamma changes topic. Delta continues.",
        }
    ]
    chunks = chunk_pages(
        "paper",
        pages=[],
        elements=elements,
        chunk_size=512,
        overlap=100,
        embedding_client=FakeEmbeddingClient(),
    )
    assert len(chunks) >= 2
    assert all(len(chunk["text"]) <= 512 for chunk in chunks if chunk["chunk_type"] == "text")
    assert chunks[0]["section_title"] == "Introduction"
    assert "Alpha stays close" in chunks[0]["text"]
    assert "Gamma changes topic" in chunks[1]["text"]


def test_image_binding() -> None:
    elements = [
        {"type": "text", "order": 0, "page_num": 1, "text": "Alpha context."},
        {
            "type": "image",
            "order": 1,
            "page_num": 1,
            "caption": "Architecture diagram",
            "alt_text": "model blocks",
            "path": "images/arch.png",
            "bbox": [1, 2, 3, 4],
        },
    ]
    chunks = chunk_pages(
        "paper",
        pages=[],
        elements=elements,
        chunk_size=512,
        overlap=100,
        embedding_client=FakeEmbeddingClient(),
        vlm_client=FakeVLMClient(),
    )
    assert chunks[-1]["chunk_type"] == "multimodal"
    assert chunks[-1]["images"]
    assert "[图片:" in chunks[-1]["text"]
    assert "VLM description" in chunks[-1]["text"]
    assert chunks[-1]["images"][0]["vlm_description"]


def test_table_modes() -> None:
    small = table_html(["Name", "Score"], [["A", "1"], ["B", "2"]])
    medium_rows = [[f"group-{index % 3}", str(index)] for index in range(30)]
    medium = table_html(["Category", "Value"], medium_rows)
    large_rows = [[f"row-{index}", str(index)] for index in range(105)]
    large = table_html(["Name", "Value"], large_rows)
    wide_header = ["ID", *[f"C{index}" for index in range(1, 13)]]
    wide_rows = [["row-1", *[str(index) for index in range(1, 13)]]]
    wide = table_html(wide_header, wide_rows)

    elements = [
        {"type": "table", "order": 0, "page_num": 1, "caption": "Small", "table_body": small},
        {"type": "table", "order": 1, "page_num": 1, "caption": "Medium", "table_body": medium},
        {"type": "table", "order": 2, "page_num": 1, "caption": "Large", "table_body": large},
        {"type": "table", "order": 3, "page_num": 1, "caption": "Wide", "table_body": wide},
    ]
    chunks = chunk_pages(
        "paper",
        pages=[],
        elements=elements,
        chunk_size=512,
        overlap=100,
        embedding_client=FakeEmbeddingClient(),
        table_large_row_chunk_size=20,
        table_wide_column_group_size=9,
    )
    table_chunks = [chunk for chunk in chunks if chunk["chunk_type"] == "table"]
    modes = [chunk["tables"][0]["mode"] for chunk in table_chunks]
    assert "small" in modes
    assert "medium_group" in modes
    assert "large_rows" in modes
    assert "wide_columns" in modes
    assert any("row_range" in chunk["tables"][0] for chunk in table_chunks if chunk["tables"][0]["mode"] == "large_rows")
    assert any("column_range" in chunk["tables"][0] for chunk in table_chunks if chunk["tables"][0]["mode"] == "wide_columns")


def test_legacy_pages() -> None:
    chunks = chunk_pages(
        "legacy",
        pages=[{"page_num": 1, "text": "Only one sentence."}],
        chunk_size=512,
        overlap=100,
        embedding_client=FakeEmbeddingClient(),
    )
    assert len(chunks) == 1
    assert chunks[0]["chunk_type"] == "text"


def test_embedding_failure() -> None:
    try:
        chunk_pages(
            "paper",
            pages=[{"page_num": 1, "text": "Alpha one. Alpha two."}],
            chunk_size=512,
            overlap=100,
            embedding_client=FailingEmbeddingClient(),
        )
    except RuntimeError:
        return
    raise AssertionError("embedding failure did not propagate")


def test_multimodal_chain_metadata_roundtrip() -> None:
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        old_paths = {
            "db_path": settings.db_path,
            "upload_dir": settings.upload_dir,
            "chroma_dir": settings.chroma_dir,
            "mineru_output_dir": settings.mineru_output_dir,
            "bm25_dir": settings.bm25_dir,
        }
        try:
            for name, value in {
                "db_path": tmp_path / "papermate.db",
                "upload_dir": tmp_path / "uploads",
                "chroma_dir": tmp_path / "chroma",
                "mineru_output_dir": tmp_path / "markdown",
                "bm25_dir": tmp_path / "bm25",
            }.items():
                object.__setattr__(settings, name, value)

            elements = elements_from_content_list(
                [
                    {
                        "type": "text",
                        "page_idx": 0,
                        "text": "1 Introduction\nAlpha architecture context.",
                    },
                    {
                        "type": "image",
                        "page_idx": 0,
                        "img_path": "images/fig1.png",
                        "image_caption": "Figure 1 architecture",
                        "bbox": [0, 0, 10, 10],
                    },
                    {
                        "type": "table",
                        "page_idx": 0,
                        "table_caption": "Table 1 metrics",
                        "table_body": table_html(["Metric", "Value"], [["Accuracy", "0.90"]]),
                    },
                ],
                [
                    {
                        "archive_name": "images/fig1.png",
                        "file_name": "fig1.png",
                        "path": str(tmp_path / "fig1.png"),
                        "mime_type": "image/png",
                        "caption": "Figure 1 architecture",
                    }
                ],
            )
            chunks = chunk_pages(
                "chain-paper",
                pages=[],
                elements=elements,
                chunk_size=512,
                overlap=100,
                embedding_client=FakeEmbeddingClient(),
                vlm_client=FakeVLMClient(),
            )
            assert any(chunk["chunk_type"] == "multimodal" for chunk in chunks)
            assert any(chunk["chunk_type"] == "table" for chunk in chunks)

            init_db()
            save_paper_and_chunks(
                {
                    "paper_id": "chain-paper",
                    "file_name": "chain.pdf",
                    "file_size_bytes": 123,
                    "save_path": str(tmp_path / "chain.pdf"),
                    "parser": "test",
                    "images": [{"path": str(tmp_path / "fig1.png")}],
                    "page_count": 1,
                    "total_chars": sum(len(chunk["text"]) for chunk in chunks),
                },
                chunks,
            )
            saved_chunks = get_paper_chunks("chain-paper")
            multimodal = next(chunk for chunk in saved_chunks if chunk["chunk_type"] == "multimodal")
            assert multimodal["images"][0]["vlm_description"].startswith("VLM description")
            table_chunk = next(chunk for chunk in saved_chunks if chunk["chunk_type"] == "table")
            assert table_chunk["tables"][0]["caption"] == "Table 1 metrics"

            bm25 = BM25Store(index_dir=tmp_path / "bm25")
            bm25_result = bm25.build_index("chain-paper", saved_chunks)
            bm25_hits = bm25.search("chain-paper", "VLM description architecture", top_k=3)
            assert bm25_result["chunk_count"] == len(saved_chunks)
            assert any(json.loads(hit["images_json"]) for hit in bm25_hits)

            fake_collection = FakeCollection()
            vector_store = object.__new__(VectorStore)
            vector_store.embedding_client = FakeEmbeddingClient()
            vector_store.collection = fake_collection
            assert vector_store.add_chunks(saved_chunks) == len(saved_chunks)
            vector_metadata = fake_collection.metadatas[0]
            assert "chunk_type" in vector_metadata
            assert "images_json" in vector_metadata
            assert "tables_json" in vector_metadata
            assert any("VLM description" in document for document in fake_collection.documents)
            assert json.loads(multimodal["images_json"])[0]["vlm_description"]
        finally:
            for name, value in old_paths.items():
                object.__setattr__(settings, name, value)


class FakeCollection:
    def __init__(self) -> None:
        self.documents: list[str] = []
        self.metadatas: list[dict] = []

    def upsert(self, ids: list[str], documents: list[str], embeddings: list[list[float]], metadatas: list[dict]) -> None:
        assert len(ids) == len(documents) == len(embeddings) == len(metadatas)
        self.documents = documents
        self.metadatas = metadatas


def table_html(header: list[str], rows: list[list[str]]) -> str:
    all_rows = [header, *rows]
    rendered = ["<table>"]
    for row in all_rows:
        rendered.append("<tr>" + "".join(f"<td>{cell}</td>" for cell in row) + "</tr>")
    rendered.append("</table>")
    return "".join(rendered)


if __name__ == "__main__":
    main()
