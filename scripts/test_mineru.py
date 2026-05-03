"""Smoke test for MinerU PDF-to-Markdown conversion.

Usage:
    python scripts/test_mineru.py path/to/paper.pdf
"""

from __future__ import annotations

import sys
from pathlib import Path
from uuid import uuid4

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.mineru_client import MinerUClient  # noqa: E402


def main() -> int:
    """Convert one PDF through MinerU and print the Markdown path."""
    if len(sys.argv) != 2:
        print("Usage: python scripts/test_mineru.py path/to/paper.pdf")
        return 2

    pdf_path = Path(sys.argv[1]).resolve()
    if not pdf_path.exists() or pdf_path.suffix.lower() != ".pdf":
        print(f"PDF file not found or not a PDF: {pdf_path}")
        return 2

    paper_id = f"mineru_test_{uuid4().hex[:12]}"
    result = MinerUClient().pdf_to_markdown(pdf_path, paper_id, file_name=pdf_path.name)

    print("MinerU conversion succeeded.")
    print(f"paper_id: {paper_id}")
    print(f"markdown_path: {result['markdown_path']}")
    print(f"markdown_chars: {len(result['markdown'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
