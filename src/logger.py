"""Logging helpers for PaperMate."""

from __future__ import annotations

import logging
import os
import re
import sys
from pathlib import Path

from config import LOG_DIR, settings


SECRET_ENV_NAMES = (
    "DEEPSEEK_API_KEY",
    "EMBEDDING_API_KEY",
    "OPENAI_API_KEY",
    "MINERU_API_TOKEN",
    "MINERU_TOKEN",
    "PAPERMATE_APP_PASSWORD",
)
SECRET_PATTERN = re.compile(
    r"(?i)\b(api[_-]?key|token|authorization|password)\b\s*[:=]\s*['\"]?[^'\"\s,;]+"
)


class RedactingFormatter(logging.Formatter):
    """Formatter that redacts known secrets from the final log line."""

    def format(self, record: logging.LogRecord) -> str:
        rendered = super().format(record)
        return redact_secrets(rendered)


def redact_secrets(text: str) -> str:
    """Remove API keys and tokens from log text."""
    redacted = text
    for name in SECRET_ENV_NAMES:
        secret = os.getenv(name, "")
        if len(secret) >= 4:
            redacted = redacted.replace(secret, "[REDACTED]")
    return SECRET_PATTERN.sub(lambda match: f"{match.group(1)}=[REDACTED]", redacted)


def get_logger(name: str) -> logging.Logger:
    """Return a configured logger with file and console handlers."""
    log_dir = Path(LOG_DIR)
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / "app.log"

    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, settings.log_level.upper(), logging.INFO))
    logger.propagate = False

    if not has_papermate_handler(logger, "file"):
        file_handler = logging.FileHandler(log_path, encoding="utf-8")
        file_handler.setLevel(logger.level)
        file_handler.setFormatter(
            RedactingFormatter("%(asctime)s %(levelname)s [%(name)s] %(message)s")
        )
        file_handler._papermate_handler = "file"  # type: ignore[attr-defined]
        logger.addHandler(file_handler)

    if not has_papermate_handler(logger, "console"):
        console_handler = logging.StreamHandler(sys.stderr)
        console_handler.setLevel(logger.level)
        console_handler.setFormatter(RedactingFormatter("%(levelname)s [%(name)s] %(message)s"))
        console_handler._papermate_handler = "console"  # type: ignore[attr-defined]
        logger.addHandler(console_handler)

    return logger


def has_papermate_handler(logger: logging.Logger, handler_type: str) -> bool:
    """Return whether a logger already has a PaperMate handler of this type."""
    return any(
        getattr(handler, "_papermate_handler", None) == handler_type
        for handler in logger.handlers
    )
