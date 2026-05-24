"""Chroma vector store integration."""

from __future__ import annotations

import re
from typing import Any

from config import settings
from src.chunk_metadata import metadata_json
from src.embedding_client import EmbeddingClient, ZHIPU_PROVIDERS
from src.errors import ErrorCode, VectorStoreError
from src.retrieval.constants import RETRIEVAL_INDEX_VERSION
from src.retrieval.index_text import build_enriched_chunk_text


COLLECTION_NAME = "papermate_chunks"


class VectorStore:
    """Persistent Chroma vector store for paper chunks."""

    def __init__(
        self,
        embedding_client: EmbeddingClient | None = None,
        collection_name: str | None = None,
    ) -> None:
        self.embedding_client = embedding_client or EmbeddingClient()
        self.collection_name = collection_name or collection_name_for_embedding(self.embedding_client)

        try:
            import chromadb
        except ImportError as exc:
            raise VectorStoreError(
                ErrorCode.VECTOR_STORE_WRITE_FAILED,
                detail="缺少 chromadb 依赖，请先安装 requirements.txt。",
            ) from exc

        try:
            settings.chroma_dir.mkdir(parents=True, exist_ok=True)
            self.client = chromadb.PersistentClient(path=str(settings.chroma_dir))
            self.collection = self.client.get_or_create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"},
            )
        except Exception as exc:
            raise VectorStoreError(ErrorCode.VECTOR_STORE_WRITE_FAILED, detail=str(exc)) from exc

    def add_chunks(self, chunks: list[dict[str, Any]]) -> int:
        """Embed and upsert chunks into Chroma."""
        if not chunks:
            return 0

        ids = [str(chunk["chunk_id"]) for chunk in chunks]
        documents = [build_enriched_chunk_text(chunk) for chunk in chunks]
        metadatas = [
            {
                "chunk_id": str(chunk["chunk_id"]),
                "paper_id": str(chunk["paper_id"]),
                "chunk_index": int(chunk["chunk_index"]),
                "page_num": int(chunk["page_num"]),
                "section_title": str(chunk.get("section_title") or ""),
                "chunk_type": str(chunk.get("chunk_type") or "text"),
                # Keep text in metadata as well as documents so citation display
                # remains stable across Chroma include/document behavior changes.
                "text": str(chunk["text"]),
                "search_text": documents[index],
                "images_json": metadata_json(chunk, "images"),
                "tables_json": metadata_json(chunk, "tables"),
                "index_version": RETRIEVAL_INDEX_VERSION,
            }
            for index, chunk in enumerate(chunks)
        ]

        embeddings = self.embedding_client.embed(documents)

        try:
            self.collection.upsert(
                ids=ids,
                documents=documents,
                embeddings=embeddings,
                metadatas=metadatas,
            )
        except Exception as exc:
            raise VectorStoreError(ErrorCode.VECTOR_STORE_WRITE_FAILED, detail=str(exc)) from exc

        return len(chunks)

    def add_documents(self, chunks: list[str], embeddings: list[list[float]]) -> None:
        """Store plain documents and embeddings.

        Kept for compatibility with the initial project skeleton.
        """
        ids = [f"document_{index:04d}" for index, _ in enumerate(chunks)]
        metadatas = [
            {
                "chunk_id": ids[index],
                "paper_id": "",
                "page_num": 0,
                "section_title": "",
                "chunk_type": "text",
                "text": text,
                "chunk_index": index,
                "search_text": text,
                "images_json": "[]",
                "tables_json": "[]",
                "index_version": RETRIEVAL_INDEX_VERSION,
            }
            for index, text in enumerate(chunks)
        ]

        try:
            self.collection.upsert(
                ids=ids,
                documents=chunks,
                embeddings=embeddings,
                metadatas=metadatas,
            )
        except Exception as exc:
            raise VectorStoreError(ErrorCode.VECTOR_STORE_WRITE_FAILED, detail=str(exc)) from exc

    def search(self, paper_id: str, query: str, top_k: int = 5) -> list[dict[str, Any]]:
        """Search for relevant chunks."""
        query_embedding = self.embedding_client.embed([query])[0]

        try:
            result = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=max(1, top_k),
                where={"paper_id": paper_id},
                include=["documents", "metadatas", "distances"],
            )
        except Exception as exc:
            raise VectorStoreError(ErrorCode.VECTOR_STORE_SEARCH_FAILED, detail=str(exc)) from exc

        ids = result.get("ids", [[]])[0]
        documents = result.get("documents", [[]])[0]
        metadatas = result.get("metadatas", [[]])[0]
        distances = result.get("distances", [[]])[0]

        matches: list[dict[str, Any]] = []
        for index, chunk_id in enumerate(ids):
            metadata = metadatas[index] or {}
            if metadata.get("index_version") != RETRIEVAL_INDEX_VERSION:
                raise VectorStoreError(
                    ErrorCode.VECTOR_STORE_SEARCH_FAILED,
                    detail=f"Chroma index_version is missing or stale; rebuild index with {RETRIEVAL_INDEX_VERSION}.",
                )
            text = metadata.get("text") or documents[index]
            matches.append(
                {
                    "chunk_id": chunk_id,
                    "paper_id": metadata.get("paper_id", paper_id),
                    "chunk_index": metadata.get("chunk_index"),
                    "page_num": metadata.get("page_num", 0),
                    "section_title": metadata.get("section_title", ""),
                    "chunk_type": metadata.get("chunk_type", "text"),
                    "text": text,
                    "search_text": metadata.get("search_text") or documents[index],
                    "images_json": metadata.get("images_json", "[]"),
                    "tables_json": metadata.get("tables_json", "[]"),
                    "index_version": metadata.get("index_version", ""),
                    "distance": distances[index],
                }
            )

        return matches


def collection_name_for_embedding(embedding_client: EmbeddingClient) -> str:
    """Return a Chroma collection name that avoids cross-provider dimension conflicts."""
    version_suffix = re.sub(r"[^A-Za-z0-9._-]+", "_", RETRIEVAL_INDEX_VERSION).strip("._-")
    if embedding_client.provider not in ZHIPU_PROVIDERS:
        return f"{COLLECTION_NAME}_{version_suffix}"[:63]

    suffix = re.sub(r"[^A-Za-z0-9._-]+", "_", embedding_client.identity()).strip("._-")
    return f"{COLLECTION_NAME}_{suffix}_{version_suffix}"[:63]

