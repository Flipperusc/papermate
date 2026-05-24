"""Neighbor evidence expansion for retrieved chunks."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from src.db import get_paper_chunks
from src.logger import get_logger


logger = get_logger(__name__)


class EvidenceExpander:
    """Add nearby chunks as lower-priority context evidence."""

    def __init__(
        self,
        window: int = 1,
        chunk_loader: Callable[[str], list[dict[str, Any]]] = get_paper_chunks,
    ) -> None:
        self.window = max(0, int(window or 0))
        self.chunk_loader = chunk_loader

    def expand(self, paper_id: str, core_chunks: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Return core chunks followed by unique neighbor chunks."""
        if not core_chunks:
            return []

        core_results: list[dict[str, Any]] = []
        seen_keys: set[tuple[str, int | str]] = set()
        for chunk in core_chunks:
            item = dict(chunk)
            item["expanded_neighbor"] = bool(item.get("expanded_neighbor", False))
            key = _chunk_key(item)
            if key in seen_keys:
                continue
            seen_keys.add(key)
            core_results.append(item)

        if self.window <= 0:
            return core_results

        try:
            all_chunks = self.chunk_loader(paper_id)
        except Exception as exc:
            logger.warning("Neighbor expansion failed to load chunks. paper_id=%s error=%s", paper_id, exc)
            return core_results

        chunks_by_index: dict[int, dict[str, Any]] = {}
        for chunk in all_chunks:
            if str(chunk.get("paper_id") or "") != str(paper_id):
                continue
            chunk_index = _safe_int(chunk.get("chunk_index"))
            if chunk_index is not None:
                chunks_by_index[chunk_index] = chunk

        neighbor_results: list[dict[str, Any]] = []
        for parent in core_results:
            parent_index = _safe_int(parent.get("chunk_index"))
            if parent_index is None:
                continue
            for offset in range(-self.window, self.window + 1):
                if offset == 0:
                    continue
                neighbor = chunks_by_index.get(parent_index + offset)
                if not neighbor:
                    continue
                key = _chunk_key(neighbor)
                if key in seen_keys:
                    continue
                seen_keys.add(key)
                neighbor_results.append(_build_neighbor(parent, neighbor, offset))

        return core_results + neighbor_results


def _build_neighbor(parent: dict[str, Any], neighbor: dict[str, Any], offset: int) -> dict[str, Any]:
    item = dict(neighbor)
    item.update(
        {
            "expanded_neighbor": True,
            "parent_chunk_id": parent.get("chunk_id", ""),
            "neighbor_offset": offset,
            "retrieval_sources": ["neighbor"],
            "source_ranks": dict(parent.get("source_ranks") or {}),
            "rrf_score": parent.get("rrf_score"),
            "rerank_score": parent.get("rerank_score"),
            "final_score": max(0.0, _safe_float(parent.get("final_score")) - 0.02 * abs(offset)),
            "section_boost": parent.get("section_boost", 0.0),
            "exact_overlap": 0.0,
            "index_version": parent.get("index_version", ""),
        }
    )
    return item


def _chunk_key(chunk: dict[str, Any]) -> tuple[str, int | str]:
    chunk_id = str(chunk.get("chunk_id") or "")
    if chunk_id:
        return ("id", chunk_id)
    chunk_index = _safe_int(chunk.get("chunk_index"))
    if chunk_index is not None:
        return (str(chunk.get("paper_id") or ""), chunk_index)
    return ("text", str(chunk.get("text") or "")[:120])


def _safe_int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _safe_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0
