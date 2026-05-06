"""Tokenization helpers for keyword retrieval."""

from __future__ import annotations

import re


TOKEN_PATTERN = re.compile(
    r"[A-Za-z][A-Za-z0-9_+\-.]{1,}|[0-9]+(?:\.[0-9]+)?|[\u4e00-\u9fff]+"
)

ENGLISH_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "been",
    "by",
    "can",
    "could",
    "did",
    "do",
    "does",
    "for",
    "from",
    "had",
    "has",
    "have",
    "how",
    "in",
    "into",
    "is",
    "it",
    "its",
    "of",
    "on",
    "or",
    "that",
    "the",
    "their",
    "this",
    "to",
    "was",
    "were",
    "what",
    "when",
    "where",
    "which",
    "who",
    "why",
    "with",
}

CJK_STOPWORDS = {
    "一个",
    "一下",
    "什么",
    "以及",
    "他们",
    "如何",
    "它的",
    "是否",
    "这个",
    "这些",
    "论文",
    "请问",
}


def normalize_text(text: str) -> str:
    """Collapse whitespace for retrieval queries and documents."""
    return re.sub(r"\s+", " ", str(text or "")).strip()


def unique_preserve_order(items: list[str]) -> list[str]:
    """Return items without duplicates while preserving first occurrence."""
    seen: set[str] = set()
    unique_items: list[str] = []
    for item in items:
        if not item or item in seen:
            continue
        seen.add(item)
        unique_items.append(item)
    return unique_items


def tokenize(text: str) -> list[str]:
    """Tokenize English, numbers, and CJK text for BM25 retrieval."""
    tokens: list[str] = []
    for match in TOKEN_PATTERN.finditer(normalize_text(text)):
        raw_token = match.group(0)
        if _is_cjk(raw_token):
            tokens.extend(_tokenize_cjk(raw_token))
        else:
            tokens.extend(_tokenize_ascii(raw_token))
    return unique_preserve_order(tokens)


def _tokenize_ascii(token: str) -> list[str]:
    normalized = token.lower().strip("._-+")
    if not normalized:
        return []

    candidates = [normalized]
    candidates.extend(
        part
        for part in re.split(r"[_+\-./]+", normalized)
        if part and part != normalized
    )

    return [
        candidate
        for candidate in candidates
        if len(candidate) > 1 and candidate not in ENGLISH_STOPWORDS
    ]


def _tokenize_cjk(text: str) -> list[str]:
    if len(text) <= 2:
        return [] if text in CJK_STOPWORDS else [text]

    tokens: list[str] = []
    if len(text) <= 8 and text not in CJK_STOPWORDS:
        tokens.append(text)

    tokens.extend(
        text[index : index + 2]
        for index in range(0, len(text) - 1)
        if text[index : index + 2] not in CJK_STOPWORDS
    )
    return tokens


def _is_cjk(token: str) -> bool:
    return all("\u4e00" <= char <= "\u9fff" for char in token)
