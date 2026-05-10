"""Audit and clean PaperMate runtime files that SQLite no longer references."""

from __future__ import annotations

import argparse
import shutil
import sqlite3
import sys
from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import settings


CACHE_EXCLUDED_DIRS = {
    ".git",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".venv",
    "data",
    "env",
    "htmlcov",
    "logs",
    "node_modules",
    "venv",
}


@dataclass(frozen=True)
class CleanupItem:
    """A runtime file or directory that can be removed safely."""

    path: Path
    kind: str
    reason: str
    size_bytes: int


def format_bytes(size_bytes: int) -> str:
    """Return a compact human-readable size."""
    units = ("B", "KB", "MB", "GB")
    size = float(size_bytes)
    for unit in units:
        if size < 1024 or unit == units[-1]:
            return f"{size:.2f} {unit}"
        size /= 1024
    return f"{size_bytes} B"


def path_size(path: Path) -> int:
    """Calculate file or directory size, ignoring files deleted during the walk."""
    if path.is_file():
        return path.stat().st_size
    total = 0
    for child in path.rglob("*"):
        if child.is_file():
            try:
                total += child.stat().st_size
            except FileNotFoundError:
                continue
    return total


def ensure_inside_project(path: Path) -> Path:
    """Resolve a path and reject anything outside this repository."""
    resolved = path.resolve()
    resolved.relative_to(PROJECT_ROOT.resolve())
    return resolved


def load_paper_ids(db_path: Path) -> set[str]:
    """Read paper ids from SQLite without creating a new database."""
    if not db_path.exists():
        return set()

    with sqlite3.connect(db_path) as connection:
        try:
            rows = connection.execute("SELECT paper_id FROM papers").fetchall()
        except sqlite3.OperationalError:
            return set()
    return {str(row[0]) for row in rows}


def upload_paper_id(path: Path) -> str:
    """Extract the paper id prefix from uploaded PDF filenames."""
    return path.stem.split("_", 1)[0]


def bm25_paper_id(path: Path) -> str:
    """Extract the paper id prefix from BM25 index filenames."""
    return path.name.split("_", 1)[0]


def collect_orphan_uploads(paper_ids: set[str]) -> list[CleanupItem]:
    """Find uploaded files whose generated paper id is absent from SQLite."""
    upload_dir = settings.upload_dir
    if not upload_dir.exists():
        return []

    items: list[CleanupItem] = []
    for path in upload_dir.iterdir():
        if path.is_file() and upload_paper_id(path) not in paper_ids:
            items.append(
                CleanupItem(
                    path=path,
                    kind="upload",
                    reason="file prefix is not present in papers.paper_id",
                    size_bytes=path_size(path),
                )
            )
    return items


def collect_orphan_output_dirs(paper_ids: set[str]) -> list[CleanupItem]:
    """Find parsed Markdown/MinerU directories not referenced by SQLite."""
    items: list[CleanupItem] = []
    seen_dirs: set[Path] = set()
    candidates = (
        ("markdown", settings.markdown_dir),
        ("mineru", settings.mineru_output_dir),
    )

    for kind, base_dir in candidates:
        if not base_dir.exists():
            continue
        resolved_base = base_dir.resolve()
        if resolved_base in seen_dirs:
            continue
        seen_dirs.add(resolved_base)
        for path in base_dir.iterdir():
            if path.is_dir() and path.name not in paper_ids:
                items.append(
                    CleanupItem(
                        path=path,
                        kind=kind,
                        reason="directory name is not present in papers.paper_id",
                        size_bytes=path_size(path),
                    )
                )
    return items


def collect_orphan_bm25_files(paper_ids: set[str]) -> list[CleanupItem]:
    """Find per-paper BM25 files not referenced by SQLite."""
    bm25_dir = settings.bm25_dir
    if not bm25_dir.exists():
        return []

    items: list[CleanupItem] = []
    for path in bm25_dir.iterdir():
        if path.is_file() and bm25_paper_id(path) not in paper_ids:
            items.append(
                CleanupItem(
                    path=path,
                    kind="bm25",
                    reason="file prefix is not present in papers.paper_id",
                    size_bytes=path_size(path),
                )
            )
    return items


def collect_python_cache() -> list[CleanupItem]:
    """Find Python bytecode caches under the repository."""
    items: list[CleanupItem] = []
    for path in PROJECT_ROOT.rglob("__pycache__"):
        if path.is_dir() and not is_excluded_cache_path(path):
            items.append(
                CleanupItem(
                    path=path,
                    kind="cache",
                    reason="Python bytecode cache",
                    size_bytes=path_size(path),
                )
            )
    for pattern in ("*.pyc", "*.pyo"):
        for path in PROJECT_ROOT.rglob(pattern):
            if (
                path.is_file()
                and "__pycache__" not in path.parts
                and not is_excluded_cache_path(path)
            ):
                items.append(
                    CleanupItem(
                        path=path,
                        kind="cache",
                        reason="Python bytecode cache",
                        size_bytes=path_size(path),
                    )
                )
    return items


def is_excluded_cache_path(path: Path) -> bool:
    """Return True for cache paths under virtualenvs or runtime stores."""
    relative_parts = path.resolve().relative_to(PROJECT_ROOT.resolve()).parts
    return any(part in CACHE_EXCLUDED_DIRS for part in relative_parts)


def collect_cleanup_items(include_cache: bool) -> list[CleanupItem]:
    """Collect all safe cleanup candidates."""
    paper_ids = load_paper_ids(settings.db_path)
    items = [
        *collect_orphan_uploads(paper_ids),
        *collect_orphan_output_dirs(paper_ids),
        *collect_orphan_bm25_files(paper_ids),
    ]
    if include_cache:
        items.extend(collect_python_cache())

    deduped: dict[Path, CleanupItem] = {}
    for item in items:
        resolved = ensure_inside_project(item.path)
        deduped[resolved] = item
    return sorted(deduped.values(), key=lambda item: str(item.path).lower())


def remove_item(path: Path) -> None:
    """Remove one file or directory."""
    resolved = ensure_inside_project(path)
    if resolved.is_dir():
        shutil.rmtree(resolved)
    elif resolved.exists():
        resolved.unlink()


def print_report(items: list[CleanupItem], apply: bool) -> None:
    """Print a dry-run or applied cleanup report."""
    mode = "APPLY" if apply else "DRY-RUN"
    total_size = sum(item.size_bytes for item in items)
    print(f"{mode}: {len(items)} cleanup candidate(s), {format_bytes(total_size)} total")
    for item in items:
        relative = item.path.resolve().relative_to(PROJECT_ROOT.resolve())
        print(f"- [{item.kind}] {relative} ({format_bytes(item.size_bytes)})")
        print(f"  reason: {item.reason}")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Clean PaperMate runtime files that are not referenced by SQLite."
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="delete the reported files; default is dry-run only",
    )
    parser.add_argument(
        "--include-cache",
        action="store_true",
        help="also remove Python __pycache__ directories and bytecode files",
    )
    return parser.parse_args()


def main() -> int:
    """Run the cleanup audit and optional deletion."""
    args = parse_args()
    items = collect_cleanup_items(include_cache=args.include_cache)
    print_report(items, apply=args.apply)

    if args.apply:
        for item in items:
            remove_item(item.path)
        print("Cleanup complete.")
    else:
        print("No files were deleted. Re-run with --apply to delete these items.")

    print("Chroma files are intentionally not cleaned per paper; rebuild the vector store if needed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
