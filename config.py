"""Application configuration for PaperMate."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
LOG_DIR = PROJECT_ROOT / "logs"


def env_value(name: str, default: str = "") -> str:
    """Read a non-empty environment value with a default."""
    value = os.getenv(name)
    if value is None or value.strip() == "":
        return default
    return value.strip()


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

try:
    from dotenv import load_dotenv
except ImportError:
    pass
else:
    load_dotenv(PROJECT_ROOT / ".env")


@dataclass(frozen=True)
class Settings:
    """Runtime settings loaded from environment variables."""

    app_name: str = os.getenv("PAPERMATE_APP_NAME", "PaperMate")
    upload_dir: Path = PROJECT_ROOT / os.getenv("PAPERMATE_UPLOAD_DIR", "data/uploads")
    chroma_dir: Path = PROJECT_ROOT / os.getenv("PAPERMATE_CHROMA_DIR", "data/chroma_db")
    db_path: Path = PROJECT_ROOT / os.getenv("PAPERMATE_DB_PATH", "data/papermate.db")
    log_level: str = os.getenv("PAPERMATE_LOG_LEVEL", "INFO")
    app_password: str = env_value("PAPERMATE_APP_PASSWORD")

    openai_api_key: str = os.getenv("OPENAI_API_KEY", "")
    openai_base_url: str = os.getenv("OPENAI_BASE_URL", "")
    openai_model: str = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

    deepseek_api_key: str = env_value("DEEPSEEK_API_KEY")
    deepseek_base_url: str = env_value("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
    deepseek_model: str = env_value("DEEPSEEK_MODEL", "deepseek-v4-pro")

    embedding_provider: str = env_value("EMBEDDING_PROVIDER", "openai-compatible")
    embedding_model: str = env_value("EMBEDDING_MODEL", "text-embedding-3-small")
    embedding_api_key: str = env_value("EMBEDDING_API_KEY", env_value("OPENAI_API_KEY"))
    embedding_base_url: str = env_value("EMBEDDING_BASE_URL", env_value("OPENAI_BASE_URL"))

    pdf_parse_provider: str = env_value("PAPERMATE_PDF_PARSE_PROVIDER", "mineru").lower()

    mineru_api_token: str = env_value("MINERU_API_TOKEN", env_value("MINERU_TOKEN"))
    mineru_base_url: str = env_value("MINERU_BASE_URL", "https://mineru.net")
    mineru_model_version: str = env_value("MINERU_MODEL_VERSION", "vlm")
    mineru_is_ocr: bool = env_bool("MINERU_IS_OCR", True)
    mineru_enable_formula: bool = env_bool("MINERU_ENABLE_FORMULA", True)
    mineru_enable_table: bool = env_bool("MINERU_ENABLE_TABLE", True)
    mineru_language: str = env_value("MINERU_LANGUAGE", "en")
    mineru_poll_interval: float = env_float("MINERU_POLL_INTERVAL", 3.0)
    mineru_poll_timeout: int = env_int("MINERU_POLL_TIMEOUT", 600)
    mineru_output_dir: Path = PROJECT_ROOT / env_value("MINERU_OUTPUT_DIR", "data/mineru_outputs")


settings = Settings()
