"""Chroma vector store integration."""

from __future__ import annotations

from typing import Any

from config import settings
from src.embedding_client import EmbeddingClient
from src.errors import ErrorCode, VectorStoreError


COLLECTION_NAME = "papermate_chunks"


class VectorStore:
    """Persistent Chroma vector store for paper chunks."""

    def __init__(self, embedding_client: EmbeddingClient | None = None) -> None:
        self.embedding_client = embedding_client or EmbeddingClient()

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
                name=COLLECTION_NAME,
                metadata={"hnsw:space": "cosine"},
            )
        except Exception as exc:
            raise VectorStoreError(ErrorCode.VECTOR_STORE_WRITE_FAILED, detail=str(exc)) from exc

    def add_chunks(self, chunks: list[dict[str, Any]]) -> int:
        """Embed and upsert chunks into Chroma."""
        if not chunks:
            return 0

        ids = [str(chunk["chunk_id"]) for chunk in chunks]
        documents = [str(chunk["text"]) for chunk in chunks]
        metadatas = [
            {
                "chunk_id": str(chunk["chunk_id"]),
                "paper_id": str(chunk["paper_id"]),
                "page_num": int(chunk["page_num"]),
                "section_title": str(chunk.get("section_title") or ""),
                "text": str(chunk["text"]),
            }
            for chunk in chunks
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
                "text": text,
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
            text = metadata.get("text") or documents[index]
            matches.append(
                {
                    "chunk_id": chunk_id,
                    "paper_id": metadata.get("paper_id", paper_id),
                    "page_num": metadata.get("page_num", 0),
                    "section_title": metadata.get("section_title", ""),
                    "text": text,
                    "distance": distances[index],
                }
            )

        return matches
