"""Logging helpers for PaperMate."""

from __future__ import annotations

import logging
from pathlib import Path

from config import LOG_DIR, settings


def get_logger(name: str) -> logging.Logger:
    """Return a configured logger."""
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_path = Path(LOG_DIR) / "app.log"

    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, settings.log_level.upper(), logging.INFO))
    logger.propagate = False

    if not logger.handlers:
        formatter = logging.Formatter("%(asctime)s %(levelname)s [%(name)s] %(message)s")

        file_handler = logging.FileHandler(log_path, encoding="utf-8")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger
