"""Embedding client for Zhipu and OpenAI-compatible APIs."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any
from urllib.parse import urlsplit, urlunsplit

import requests

from config import settings
from src.errors import EmbeddingError, ErrorCode
from src.logger import get_logger


OPENAI_COMPATIBLE_PROVIDERS = {"openai", "openai-compatible", "openai_compatible", "compatible"}
ZHIPU_PROVIDERS = {"zhipu", "zhipuai", "zhipu_ai", "bigmodel", "bigmodel_cn"}
ZHIPU_DEFAULT_BASE_URL = "https://open.bigmodel.cn/api/paas/v4"
ZHIPU_EMBEDDING_3_DIMENSIONS = {256, 512, 1024, 2048}

logger = get_logger(__name__)


class EmbeddingClient:
    """Client responsible for creating text embeddings."""

    def __init__(
        self,
        provider: str | None = None,
        model: str | None = None,
        api_key: str | None = None,
        base_url: str | None = None,
        batch_size: int = 64,
        dimensions: int | None = None,
        session: requests.Session | None = None,
    ) -> None:
        self.provider = normalize_provider(provider or settings.embedding_provider)
        self.model = model or settings.embedding_model
        self.api_key = api_key if api_key is not None else settings.embedding_api_key
        raw_base_url = base_url if base_url is not None else settings.embedding_base_url
        self.base_url = self._normalize_base_url(raw_base_url)
        self.batch_size = max(1, batch_size)
        self.dimensions = dimensions if dimensions is not None else settings.embedding_dimensions
        self.session = session or requests.Session()

    def _create_openai_client(self):
        """Create an OpenAI SDK client configured for compatible APIs."""
        if not self.api_key:
            raise EmbeddingError(ErrorCode.EMBEDDING_API_KEY_MISSING)

        try:
            from openai import OpenAI
        except ImportError as exc:
            logger.exception("OpenAI SDK is not installed for embedding calls.")
            raise EmbeddingError(
                ErrorCode.EMBEDDING_CALL_FAILED,
                detail="缺少 openai 依赖，请先安装 requirements.txt。",
            ) from exc

        client_kwargs = {"api_key": self.api_key}
        if self.base_url:
            client_kwargs["base_url"] = self.base_url
        try:
            return OpenAI(**client_kwargs)
        except Exception as exc:
            logger.exception("Failed to create OpenAI-compatible embedding client.")
            raise EmbeddingError(ErrorCode.EMBEDDING_CALL_FAILED, detail=str(exc)) from exc

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Create embeddings for a list of texts."""
        if not texts:
            return []

        if self.provider in ZHIPU_PROVIDERS:
            return self._embed_zhipu(texts)
        if self.provider in OPENAI_COMPATIBLE_PROVIDERS:
            return self._embed_openai_compatible(texts)

        logger.error("Unsupported embedding provider: %s", self.provider)
        raise EmbeddingError(
            ErrorCode.EMBEDDING_CALL_FAILED,
            detail=f"暂不支持的 embedding provider: {self.provider}",
        )

    def _embed_openai_compatible(self, texts: list[str]) -> list[list[float]]:
        """Create embeddings through an OpenAI-compatible SDK endpoint."""
        client = self._create_openai_client()
        embeddings: list[list[float]] = []

        try:
            for batch in batched(texts, self.batch_size):
                response = client.embeddings.create(
                    model=self.model,
                    input=[text or "" for text in batch],
                )
                response_data = sorted(response.data, key=lambda item: item.index)
                embeddings.extend([list(item.embedding) for item in response_data])
        except Exception as exc:
            logger.exception(
                "Embedding call failed. provider=%s model=%s base_url=%s batch_size=%s",
                self.provider,
                self.model,
                redact_url(self.base_url),
                self.batch_size,
            )
            raise EmbeddingError(ErrorCode.EMBEDDING_CALL_FAILED, detail=str(exc)) from exc

        if len(embeddings) != len(texts):
            logger.error(
                "Embedding count mismatch: expected=%s actual=%s",
                len(texts),
                len(embeddings),
            )
            raise EmbeddingError(
                ErrorCode.EMBEDDING_CALL_FAILED,
                detail="返回的 embedding 数量与输入文本数量不一致。",
            )

        return embeddings

    def _embed_zhipu(self, texts: list[str]) -> list[list[float]]:
        """Create embeddings through Zhipu BigModel's native embeddings API."""
        if not self.api_key:
            raise EmbeddingError(ErrorCode.EMBEDDING_API_KEY_MISSING)
        self._validate_zhipu_dimensions()

        embeddings: list[list[float]] = []
        try:
            for batch in batched(texts, min(self.batch_size, 64)):
                payload: dict[str, Any] = {
                    "model": self.model,
                    "input": [text or "" for text in batch],
                }
                if self.dimensions > 0:
                    payload["dimensions"] = self.dimensions

                response = self.session.post(
                    zhipu_embeddings_url(self.base_url),
                    headers={
                        "Authorization": f"Bearer {self.api_key}",
                        "Content-Type": "application/json",
                    },
                    json=payload,
                    timeout=120,
                )
                if response.status_code >= 400:
                    logger.error(
                        "Zhipu embedding call failed. status=%s body=%s",
                        response.status_code,
                        response.text[:1000],
                    )
                    raise EmbeddingError(
                        ErrorCode.EMBEDDING_CALL_FAILED,
                        detail=f"智谱 Embedding 接口返回 HTTP {response.status_code}。",
                    )

                data = response.json()
                response_data = sorted(
                    data.get("data") or [],
                    key=lambda item: int(item.get("index", 0)),
                )
                for item in response_data:
                    embedding = item.get("embedding")
                    if not isinstance(embedding, list):
                        raise ValueError("Zhipu response item has no embedding list")
                    embeddings.append([float(value) for value in embedding])
        except EmbeddingError:
            raise
        except (requests.RequestException, ValueError, TypeError) as exc:
            logger.exception(
                "Zhipu embedding call failed. model=%s base_url=%s batch_size=%s dimensions=%s",
                self.model,
                redact_url(self.base_url),
                min(self.batch_size, 64),
                self.dimensions,
            )
            raise EmbeddingError(ErrorCode.EMBEDDING_CALL_FAILED, detail=str(exc)) from exc

        if len(embeddings) != len(texts):
            logger.error(
                "Zhipu embedding count mismatch: expected=%s actual=%s",
                len(texts),
                len(embeddings),
            )
            raise EmbeddingError(
                ErrorCode.EMBEDDING_CALL_FAILED,
                detail="智谱返回的 embedding 数量与输入文本数量不一致。",
            )

        return embeddings

    def _normalize_base_url(self, base_url: str) -> str:
        if self.provider in ZHIPU_PROVIDERS:
            return normalize_zhipu_base_url(base_url)
        return normalize_openai_base_url(base_url)

    def _validate_zhipu_dimensions(self) -> None:
        if self.model != "embedding-3" or self.dimensions <= 0:
            return
        if self.dimensions in ZHIPU_EMBEDDING_3_DIMENSIONS:
            return
        raise EmbeddingError(
            ErrorCode.EMBEDDING_CALL_FAILED,
            detail="embedding-3 的 EMBEDDING_DIMENSIONS 只支持 256、512、1024 或 2048。",
        )

    def identity(self) -> str:
        """Return a stable embedding-backend identity for vector collections."""
        parts = [self.provider, self.model]
        if self.provider in ZHIPU_PROVIDERS and self.dimensions > 0:
            parts.append(str(self.dimensions))
        return "_".join(parts)


def batched(items: Sequence[str], batch_size: int) -> list[Sequence[str]]:
    """Split items into fixed-size batches."""
    return [items[index : index + batch_size] for index in range(0, len(items), batch_size)]


def normalize_provider(provider: str) -> str:
    """Normalize provider names from env values."""
    return str(provider or "").strip().lower().replace("-", "_")


def normalize_openai_base_url(base_url: str) -> str:
    """Normalize common OpenAI-compatible base URL mistakes."""
    normalized = base_url.strip().rstrip("/")
    if not normalized:
        return ""

    if normalized.endswith("/embeddings"):
        normalized = normalized[: -len("/embeddings")]

    if normalized in {"https://api.openai.com", "http://api.openai.com"}:
        normalized = f"{normalized}/v1"

    return normalized


def normalize_zhipu_base_url(base_url: str) -> str:
    """Normalize Zhipu BigModel base URLs to the v4 API root."""
    normalized = str(base_url or "").strip().rstrip("/")
    if not normalized:
        return ZHIPU_DEFAULT_BASE_URL

    if normalized.endswith("/embeddings"):
        normalized = normalized[: -len("/embeddings")]

    if normalized in {
        "https://api.openai.com",
        "http://api.openai.com",
        "https://api.openai.com/v1",
        "http://api.openai.com/v1",
    }:
        return ZHIPU_DEFAULT_BASE_URL

    if normalized in {"https://open.bigmodel.cn", "http://open.bigmodel.cn"}:
        normalized = f"{normalized}/api/paas/v4"

    return normalized


def zhipu_embeddings_url(base_url: str) -> str:
    """Return the Zhipu embeddings endpoint URL."""
    return f"{normalize_zhipu_base_url(base_url)}/embeddings"


def redact_url(url: str) -> str:
    """Return a URL safe for logs."""
    if not url:
        return ""
    parsed = urlsplit(url)
    return urlunsplit((parsed.scheme, parsed.netloc, parsed.path, "", ""))
