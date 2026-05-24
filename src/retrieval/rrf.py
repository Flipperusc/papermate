"""Reciprocal Rank Fusion utilities."""

from __future__ import annotations

import math
from typing import Any

from src.errors import AppError, ErrorCode
from src.logger import get_logger


logger = get_logger(__name__)
DEFAULT_SOURCE_NAMES = ["vector", "bm25"]


def reciprocal_rank_fusion(
    result_lists: list[list[dict]] | None = None,
    rrf_k: int = 60,
    weights: list[float] | None = None,
    **legacy_kwargs: Any,
) -> list[dict]:
    """Fuse ranked retrieval results with Reciprocal Rank Fusion."""
    try:
        normalized_lists, top_k = _normalize_inputs(result_lists, legacy_kwargs)
        if not normalized_lists:
            return []

        safe_rrf_k = max(1, int(rrf_k))
        safe_weights = _normalize_weights(weights, len(normalized_lists))
        fused: dict[str, dict[str, Any]] = {}

        for list_index, results in enumerate(normalized_lists):
            source_name = _default_source_name(list_index)
            weight = safe_weights[list_index]
            for fallback_rank, result in enumerate(results, start=1):
                chunk_id = str(result.get("chunk_id") or "").strip()
                if not chunk_id:
                    logger.warning("RRF skipped result without chunk_id: %s", result)
                    continue

                source = _extract_source(result, source_name)
                rank = _extract_rank(result, source, fallback_rank)
                entry = fused.setdefault(chunk_id, _base_entry(result, chunk_id))
                # The same chunk may appear in multiple retrieval lists; merge
                # its best metadata while accumulating source-specific ranks.
                _merge_metadata(entry, result)

                if source not in entry["retrieval_sources"]:
                    entry["retrieval_sources"].append(source)
                entry["source_ranks"][source] = rank
                entry["rrf_score"] += weight * (1.0 / (safe_rrf_k + rank))

                if source == "vector":
                    vector_distance = result.get("vector_distance", result.get("distance"))
                    if vector_distance is not None:
                        entry["vector_distance"] = vector_distance
                if source == "bm25" and result.get("bm25_score") is not None:
                    entry["bm25_score"] = result.get("bm25_score")

                # Backward-compatible rank fields used by existing display code.
                entry[f"{source}_rank"] = rank

        ranked_results = sorted(
            fused.values(),
            key=lambda item: (
                -float(item.get("rrf_score", 0.0)),
                min(item.get("source_ranks", {}).values(), default=10**9),
                str(item.get("chunk_id") or ""),
            ),
        )
        if top_k is not None:
            return ranked_results[: max(0, int(top_k))]
        return ranked_results
    except AppError:
        raise
    except Exception as exc:
        logger.exception("RRF fusion failed.")
        raise AppError(
            code=ErrorCode.RRF_FUSION_FAILED,
            user_message="检索结果融合失败，请查看系统日志。",
            detail=str(exc),
        ) from exc


def _normalize_inputs(
    result_lists: list[list[dict]] | None,
    legacy_kwargs: dict[str, Any],
) -> tuple[list[list[dict]], int | None]:
    """Support the new result_lists API and the previous keyword API."""
    top_k = legacy_kwargs.get("final_top_k")
    if result_lists is not None:
        return result_lists, top_k

    vector_results = legacy_kwargs.get("vector_results")
    bm25_results = legacy_kwargs.get("bm25_results")
    if vector_results is None and bm25_results is None:
        return [], top_k
    return [vector_results or [], bm25_results or []], top_k


def _normalize_weights(weights: list[float] | None, list_count: int) -> list[float]:
    """Return valid weights, defaulting to equal weights on mismatch."""
    if weights is None or len(weights) != list_count:
        return [1.0 for _ in range(list_count)]
    try:
        return [float(weight) for weight in weights]
    except (TypeError, ValueError):
        return [1.0 for _ in range(list_count)]


def _default_source_name(list_index: int) -> str:
    if list_index < len(DEFAULT_SOURCE_NAMES):
        return DEFAULT_SOURCE_NAMES[list_index]
    return f"source_{list_index + 1}"


def _extract_source(result: dict[str, Any], fallback: str) -> str:
    source = str(result.get("retrieval_source") or "").strip()
    if source:
        return source

    sources = result.get("retrieval_sources")
    if isinstance(sources, list) and sources:
        return str(sources[0])

    return fallback


def _extract_rank(result: dict[str, Any], source: str, fallback_rank: int) -> int:
    candidates = [
        result.get("rank"),
        result.get(f"{source}_rank"),
        result.get("vector_rank") if source == "vector" else None,
        result.get("bm25_rank") if source == "bm25" else None,
        fallback_rank,
    ]
    for candidate in candidates:
        if candidate is None:
            continue
        try:
            rank = int(candidate)
        except (TypeError, ValueError):
            continue
        if rank > 0:
            return rank
    return fallback_rank


def _base_entry(result: dict[str, Any], chunk_id: str) -> dict[str, Any]:
    return {
        "chunk_id": chunk_id,
        "paper_id": result.get("paper_id", ""),
        "chunk_index": result.get("chunk_index"),
        "text": result.get("text", ""),
        "search_text": result.get("search_text", ""),
        "section_title": result.get("section_title", ""),
        "page_num": result.get("page_num", ""),
        "chunk_type": result.get("chunk_type", "text"),
        "images_json": result.get("images_json", "[]"),
        "tables_json": result.get("tables_json", "[]"),
        "index_version": result.get("index_version", ""),
        "rrf_score": 0.0,
        "retrieval_sources": [],
        "source_ranks": {},
    }


def _merge_metadata(entry: dict[str, Any], result: dict[str, Any]) -> None:
    for key in (
        "paper_id",
        "chunk_index",
        "text",
        "search_text",
        "section_title",
        "page_num",
        "chunk_type",
        "images_json",
        "tables_json",
        "index_version",
    ):
        if _is_missing(entry.get(key)) and not _is_missing(result.get(key)):
            entry[key] = result.get(key)

    vector_distance = result.get("vector_distance", result.get("distance"))
    if vector_distance is not None:
        existing = entry.get("vector_distance")
        if existing is None or _safe_float(vector_distance) < _safe_float(existing):
            entry["vector_distance"] = vector_distance

    if result.get("bm25_score") is not None:
        existing_score = entry.get("bm25_score")
        if existing_score is None or _safe_float(result["bm25_score"]) > _safe_float(existing_score):
            entry["bm25_score"] = result["bm25_score"]


def _is_missing(value: Any) -> bool:
    return value is None or value == ""


def _safe_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return math.inf
