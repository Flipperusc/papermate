"""Shared helpers for chunk image/table metadata serialization."""

from __future__ import annotations

import json
from typing import Any


def metadata_json(chunk: dict[str, Any], key: str) -> str:
    """Return JSON text for chunk image/table metadata."""
    json_key = f"{key}_json"
    raw_json = chunk.get(json_key)
    if isinstance(raw_json, str) and raw_json.strip():
        return raw_json
    value = chunk.get(key) or []
    return json.dumps(value, ensure_ascii=False)


def hydrate_chunk_metadata(chunk: dict[str, Any]) -> dict[str, Any]:
    """Add parsed image/table metadata lists to a persisted chunk."""
    chunk["chunk_type"] = chunk.get("chunk_type") or "text"
    chunk["images_json"] = chunk.get("images_json") or "[]"
    chunk["tables_json"] = chunk.get("tables_json") or "[]"
    chunk["images"] = parse_metadata_json(chunk["images_json"])
    chunk["tables"] = parse_metadata_json(chunk["tables_json"])
    return chunk


def parse_metadata_json(value: Any) -> list[dict[str, Any]]:
    """Parse chunk metadata JSON safely."""
    if isinstance(value, list):
        return [dict(item) for item in value if isinstance(item, dict)]
    if not value:
        return []
    try:
        parsed = json.loads(str(value))
    except (TypeError, ValueError, json.JSONDecodeError):
        return []
    if not isinstance(parsed, list):
        return []
    return [dict(item) for item in parsed if isinstance(item, dict)]
