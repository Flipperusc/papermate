"""Tokenizer for BM25 keyword retrieval."""

from __future__ import annotations

import re


STOPWORDS = {
    "the",
    "a",
    "an",
    "of",
    "and",
    "or",
    "to",
    "in",
    "for",
    "本文",
    "论文",
    "作者",
    "什么",
    "哪些",
    "是否",
    "有没有",
    "请",
}

TOKEN_PATTERN = re.compile(
    r"[A-Za-z]+[A-Za-z0-9]*(?:[-_.][A-Za-z0-9]+)*\.?"
    r"|[A-Za-z]*\d+(?:[-_.][A-Za-z0-9]+)*"
    r"|\d+(?:\.\d+)*"
    r"|[\u4e00-\u9fff]+"
)

EN_NUMBERED_ENTITY_PATTERN = re.compile(
    r"\b(table|tab|figure|fig|eq|equation)\.?\s*(\d+(?:\.\d+)*)\b",
    flags=re.IGNORECASE,
)
ZH_NUMBERED_ENTITY_PATTERN = re.compile(r"(表|图|公式)\s*(\d+(?:\.\d+)*)")


def tokenize_text(text: str) -> list[str]:
    """Tokenize paper text for BM25 while preserving common scientific entities."""
    normalized_text = str(text or "")
    tokens: list[str] = []

    for match in EN_NUMBERED_ENTITY_PATTERN.finditer(normalized_text):
        label = _canonical_numbered_label(match.group(1).lower().rstrip("."))
        number = match.group(2)
        tokens.append(f"{label} {number}")
        tokens.append(f"{label}-{number}")

    for match in ZH_NUMBERED_ENTITY_PATTERN.finditer(normalized_text):
        label = match.group(1)
        number = match.group(2)
        tokens.append(f"{label} {number}")
        tokens.append(f"{label}-{number}")
        tokens.append(f"{_canonical_numbered_label(label)} {number}")

    for match in TOKEN_PATTERN.finditer(normalized_text):
        token = normalize_token(match.group(0))
        if not token or token in STOPWORDS:
            continue

        if is_cjk(token):
            tokens.extend(tokenize_cjk(token))
        else:
            tokens.append(token)

    return unique_preserve_order(tokens)


def normalize_token(token: str) -> str:
    """Normalize one token without removing internal paper-entity symbols."""
    normalized = token.lower().strip()
    normalized = normalized.strip("'\"“”‘’()[]{}<>:,;!?/\\")
    normalized = normalized.strip("-_")
    if normalized.endswith(".") and normalized not in {"eq.", "fig."}:
        normalized = normalized.rstrip(".")
    return normalized


def _canonical_numbered_label(label: str) -> str:
    mapping = {
        "tab": "table",
        "fig": "figure",
        "eq": "equation",
        "表": "table",
        "图": "figure",
        "公式": "equation",
    }
    return mapping.get(label, label)


def tokenize_cjk(token: str) -> list[str]:
    """Tokenize continuous CJK text into whole-span and n-gram tokens."""
    if token in STOPWORDS:
        return []
    if len(token) <= 1:
        return [token]

    tokens = [token]
    tokens.extend(token[index : index + 2] for index in range(0, len(token) - 1))
    if len(token) >= 3:
        tokens.extend(token[index : index + 3] for index in range(0, len(token) - 2))
    return [item for item in tokens if item not in STOPWORDS]


def is_cjk(token: str) -> bool:
    """Return whether a token is made only of CJK characters."""
    return all("\u4e00" <= char <= "\u9fff" for char in token)


def unique_preserve_order(tokens: list[str]) -> list[str]:
    """Deduplicate tokens while keeping stable order."""
    seen: set[str] = set()
    unique_tokens: list[str] = []
    for token in tokens:
        if token in seen:
            continue
        seen.add(token)
        unique_tokens.append(token)
    return unique_tokens
