"""Centralized application configuration for PaperMate."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


PROJECT_ROOT = Path(__file__).resolve().parent

try:
    from dotenv import load_dotenv
except ImportError:
    pass
else:
    load_dotenv(PROJECT_ROOT / ".env")


def env_value(name: str, default: str = "") -> str:
    """Read a non-empty environment value with a default."""
    value = os.getenv(name)
    if value is None or value.strip() == "":
        return default
    return value.strip()


def first_env(names: Iterable[str], default: str = "") -> str:
    """Read the first non-empty value from a list of environment names."""
    for name in names:
        value = env_value(name)
        if value:
            return value
    return default


def env_bool(name: str, default: bool = False) -> bool:
    """Read a boolean environment value."""
    value = env_value(name)
    if not value:
        return default
    return value.lower() in {"1", "true", "yes", "y", "on"}


def env_int(name: str, default: int) -> int:
    """Read an integer environment value."""
    value = env_value(name)
    if not value:
        return default
    try:
        return int(value)
    except ValueError:
        return default


def env_float(name: str, default: float) -> float:
    """Read a float environment value."""
    value = env_value(name)
    if not value:
        return default
    try:
        return float(value)
    except ValueError:
        return default


def env_path(names: Iterable[str], default: str) -> Path:
    """Read a path and resolve relative values from the project root."""
    value = first_env(names, default)
    path = Path(value)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def normalize_provider(value: str) -> str:
    """Normalize provider names for internal comparisons."""
    return value.strip().lower().replace("-", "_")


@dataclass(frozen=True)
class Settings:
    """Runtime settings loaded from environment variables."""

    # Application
    app_name: str = env_value("PAPERMATE_APP_NAME", "PaperMate")
    app_env: str = env_value("APP_ENV", "local")
    app_password: str = env_value("PAPERMATE_APP_PASSWORD")
    host_port: int = env_int("PAPERMATE_HOST_PORT", 8501)
    log_level: str = first_env(("LOG_LEVEL", "PAPERMATE_LOG_LEVEL"), "INFO")

    # Paths
    data_dir: Path = env_path(("DATA_DIR",), "data")
    upload_dir: Path = env_path(("UPLOAD_DIR", "PAPERMATE_UPLOAD_DIR"), "data/uploads")
    markdown_dir: Path = env_path(("MARKDOWN_DIR",), "data/markdown")
    chroma_dir: Path = env_path(("CHROMA_DIR", "PAPERMATE_CHROMA_DIR"), "data/chroma_db")
    bm25_dir: Path = env_path(("BM25_DIR",), "data/bm25")
    log_dir: Path = env_path(("LOG_DIR",), "logs")
    db_path: Path = env_path(("DB_PATH", "PAPERMATE_DB_PATH"), "data/papermate.db")

    # OpenAI-compatible fallback fields kept for existing local setups.
    openai_api_key: str = env_value("OPENAI_API_KEY")
    openai_base_url: str = env_value("OPENAI_BASE_URL")
    openai_model: str = env_value("OPENAI_MODEL", "gpt-4o-mini")

    # Vision-language model for multimodal chunk image descriptions.
    vlm_api_key: str = first_env(("VLM_API_KEY", "DASHSCOPE_API_KEY"))
    vlm_base_url: str = env_value(
        "VLM_BASE_URL",
        "https://dashscope.aliyuncs.com/compatible-mode/v1",
    )
    vlm_model: str = env_value("VLM_MODEL", "qwen3.6-plus")
    vlm_timeout: float = env_float("VLM_TIMEOUT", 90.0)
    vlm_temperature: float = env_float("VLM_TEMPERATURE", 0.1)
    vlm_max_tokens: int = env_int("VLM_MAX_TOKENS", 512)
    vlm_enabled: bool = env_bool("VLM_ENABLED", True)
    vlm_parse_timeout: float = env_float("VLM_PARSE_TIMEOUT", 20.0)
    vlm_max_images_per_paper: int = env_int("VLM_MAX_IMAGES_PER_PAPER", 8)
    vlm_max_failures_per_paper: int = env_int("VLM_MAX_FAILURES_PER_PAPER", 2)

    # DeepSeek LLM
    deepseek_api_key: str = env_value("DEEPSEEK_API_KEY")
    deepseek_base_url: str = env_value("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
    deepseek_model: str = env_value("DEEPSEEK_MODEL", "deepseek-v4-pro")

    # Markdown translation
    translation_enabled: bool = env_bool("TRANSLATION_ENABLED", True)
    translation_provider: str = normalize_provider(env_value("TRANSLATION_PROVIDER", "deepseek"))
    translation_model: str = env_value("TRANSLATION_MODEL", "deepseek-chat")
    translation_chunk_size: int = env_int("TRANSLATION_CHUNK_SIZE", 3500)
    translation_timeout: int = env_int("TRANSLATION_TIMEOUT", 60)

    # PDF-to-Markdown
    pdf_parse_provider: str = normalize_provider(
        first_env(("PDF_PARSE_PROVIDER", "PAPERMATE_PDF_PARSE_PROVIDER"), "mineru")
    )
    mineru_api_token: str = first_env(("MINERU_API_TOKEN", "MINERU_TOKEN"))
    mineru_base_url: str = env_value("MINERU_BASE_URL", "https://mineru.net")
    mineru_model_version: str = env_value("MINERU_MODEL_VERSION", "vlm")
    mineru_is_ocr: bool = env_bool("MINERU_IS_OCR", True)
    mineru_enable_formula: bool = env_bool("MINERU_ENABLE_FORMULA", True)
    mineru_enable_table: bool = env_bool("MINERU_ENABLE_TABLE", True)
    mineru_language: str = env_value("MINERU_LANGUAGE", "en")
    mineru_poll_interval: float = env_float("MINERU_POLL_INTERVAL", 3.0)
    mineru_poll_timeout: int = env_int("MINERU_POLL_TIMEOUT", 600)
    mineru_output_dir: Path = env_path(("MINERU_OUTPUT_DIR", "MARKDOWN_DIR"), "data/markdown")

    # Embedding
    embedding_provider: str = normalize_provider(
        env_value("EMBEDDING_PROVIDER", "zhipu")
    )
    embedding_api_key: str = first_env(
        (
            "EMBEDDING_API_KEY",
            "ZHIPU_API_KEY",
            "ZHIPUAI_API_KEY",
            "BIGMODEL_API_KEY",
            "OPENAI_API_KEY",
        )
    )
    embedding_base_url: str = first_env(
        (
            "EMBEDDING_BASE_URL",
            "ZHIPU_BASE_URL",
            "BIGMODEL_BASE_URL",
            "OPENAI_BASE_URL",
        ),
        "https://open.bigmodel.cn/api/paas/v4",
    )
    embedding_model: str = env_value("EMBEDDING_MODEL", "embedding-3")
    embedding_dimensions: int = env_int("EMBEDDING_DIMENSIONS", 2048)

    # RAG chunking
    rag_chunk_strategy: str = env_value("RAG_CHUNK_STRATEGY", "semantic_multimodal")
    rag_chunk_size: int = env_int("RAG_CHUNK_SIZE", 512)
    rag_chunk_overlap: int = env_int("RAG_CHUNK_OVERLAP", 100)
    table_large_row_chunk_size: int = env_int("TABLE_LARGE_ROW_CHUNK_SIZE", 20)
    table_wide_column_group_size: int = env_int("TABLE_WIDE_COLUMN_GROUP_SIZE", 9)

    # RAG retrieval
    vector_top_k: int = env_int("VECTOR_TOP_K", 40)
    bm25_top_k: int = env_int("BM25_TOP_K", 40)
    final_top_k: int = env_int("FINAL_TOP_K", 8)
    rrf_k: int = env_int("RRF_K", 60)
    context_max_chars: int = env_int("CONTEXT_MAX_CHARS", 9000)
    context_expand_window: int = env_int("CONTEXT_EXPAND_WINDOW", 1)
    rerank_enabled: bool = env_bool("RERANK_ENABLED", True)
    rerank_top_k: int = env_int("RERANK_TOP_K", 30)
    rerank_batch_size: int = env_int("RERANK_BATCH_SIZE", 8)

    # Shared external API retry policy
    external_api_max_attempts: int = env_int("EXTERNAL_API_MAX_ATTEMPTS", 2)
    external_api_retry_base_seconds: float = env_float("EXTERNAL_API_RETRY_BASE_SECONDS", 1.0)
    external_api_retry_max_seconds: float = env_float("EXTERNAL_API_RETRY_MAX_SECONDS", 8.0)


settings = Settings()
LOG_DIR = settings.log_dir
