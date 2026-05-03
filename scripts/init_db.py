"""Initialize the PaperMate SQLite database."""

from __future__ import annotations

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import settings
from src.db import init_db


def main() -> None:
    """Create the SQLite database and required tables."""
    init_db()
    print(f"SQLite database initialized: {settings.db_path}")


if __name__ == "__main__":
    main()
