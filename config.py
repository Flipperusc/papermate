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

    # DeepSeek LLM
    deepseek_api_key: str = env_value("DEEPSEEK_API_KEY")
    deepseek_base_url: str = env_value("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
    deepseek_model: str = env_value("DEEPSEEK_MODEL", "deepseek-v4-pro")

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
        env_value("EMBEDDING_PROVIDER", "openai_compatible")
    )
    embedding_api_key: str = env_value("EMBEDDING_API_KEY", env_value("OPENAI_API_KEY"))
    embedding_base_url: str = env_value("EMBEDDING_BASE_URL", env_value("OPENAI_BASE_URL"))
    embedding_model: str = env_value("EMBEDDING_MODEL", "text-embedding-3-small")

    # RAG retrieval
    vector_top_k: int = env_int("VECTOR_TOP_K", 20)
    bm25_top_k: int = env_int("BM25_TOP_K", 20)
    final_top_k: int = env_int("FINAL_TOP_K", 6)
    rrf_k: int = env_int("RRF_K", 60)
    context_max_chars: int = env_int("CONTEXT_MAX_CHARS", 6000)
    context_expand_window: int = env_int("CONTEXT_EXPAND_WINDOW", 0)


settings = Settings()
LOG_DIR = settings.log_dir
