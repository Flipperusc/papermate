"""Embedding client for OpenAI-compatible APIs."""

from __future__ import annotations

from collections.abc import Sequence
from urllib.parse import urlsplit, urlunsplit

from config import settings
from src.errors import EmbeddingError, ErrorCode
from src.logger import get_logger


OPENAI_COMPATIBLE_PROVIDERS = {"openai", "openai-compatible", "openai_compatible", "compatible"}

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
    ) -> None:
        self.provider = (provider or settings.embedding_provider).strip().lower()
        self.model = model or settings.embedding_model
        self.api_key = api_key if api_key is not None else settings.embedding_api_key
        self.base_url = normalize_openai_base_url(
            base_url if base_url is not None else settings.embedding_base_url
        )
        self.batch_size = max(1, batch_size)

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

        if self.provider not in OPENAI_COMPATIBLE_PROVIDERS:
            logger.error("Unsupported embedding provider: %s", self.provider)
            raise EmbeddingError(
                ErrorCode.EMBEDDING_CALL_FAILED,
                detail=f"暂不支持的 embedding provider: {self.provider}",
            )

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


def batched(items: Sequence[str], batch_size: int) -> list[Sequence[str]]:
    """Split items into fixed-size batches."""
    return [items[index : index + batch_size] for index in range(0, len(items), batch_size)]


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


def redact_url(url: str) -> str:
    """Return a URL safe for logs."""
    if not url:
        return ""
    parsed = urlsplit(url)
    return urlunsplit((parsed.scheme, parsed.netloc, parsed.path, "", ""))
