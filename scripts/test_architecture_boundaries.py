"""Static architecture boundary checks for PaperMate backend layers."""

from __future__ import annotations

import ast
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
APP_PATH = PROJECT_ROOT / "app.py"


FORBIDDEN_SRC_IMPORT_ROOTS = {"streamlit"}
FORBIDDEN_APPLICATION_IMPORT_ROOTS = {"app", "streamlit"}
FORBIDDEN_APP_DIRECT_IMPORTS = {
    ("src.db", "save_qa_log"),
    ("src.db", "save_paper_and_chunks"),
    ("src.job_service", "enqueue_job"),
    ("src.rag_pipeline", "answer_question"),
    ("src.literature_card_service", "save_literature_card"),
    ("src.pdf_parser", "parse_pdf"),
    ("src.chunker", "chunk_pages"),
}
FORBIDDEN_APP_DEFINITIONS = {
    "process_uploaded_pdf",
}


def main() -> None:
    failures: list[str] = []
    failures.extend(check_src_has_no_streamlit_imports())
    failures.extend(check_application_layer_imports())
    failures.extend(check_app_direct_imports())
    failures.extend(check_app_removed_legacy_workflows())
    if failures:
        for failure in failures:
            print(f"architecture boundary violation: {failure}", file=sys.stderr)
        raise SystemExit(1)
    print("architecture boundary tests passed")


def check_src_has_no_streamlit_imports() -> list[str]:
    failures: list[str] = []
    for path in python_files(SRC_DIR):
        for imported_name in imported_roots(path):
            if imported_name in FORBIDDEN_SRC_IMPORT_ROOTS:
                failures.append(f"{relative(path)} imports {imported_name}")
    return failures


def check_application_layer_imports() -> list[str]:
    failures: list[str] = []
    application_dir = SRC_DIR / "application"
    for path in python_files(application_dir):
        for imported_name in imported_roots(path):
            if imported_name in FORBIDDEN_APPLICATION_IMPORT_ROOTS:
                failures.append(f"{relative(path)} imports UI module {imported_name}")
    return failures


def check_app_direct_imports() -> list[str]:
    failures: list[str] = []
    tree = parse_file(APP_PATH)
    for node in ast.walk(tree):
        if not isinstance(node, ast.ImportFrom):
            continue
        module = node.module or ""
        for alias in node.names:
            imported = (module, alias.name)
            if imported in FORBIDDEN_APP_DIRECT_IMPORTS:
                failures.append(f"app.py directly imports {module}.{alias.name}")
    return failures


def check_app_removed_legacy_workflows() -> list[str]:
    failures: list[str] = []
    tree = parse_file(APP_PATH)
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name in FORBIDDEN_APP_DEFINITIONS:
            failures.append(f"app.py still defines legacy workflow {node.name}()")
    return failures


def python_files(root: Path) -> list[Path]:
    if not root.exists():
        return []
    return sorted(path for path in root.rglob("*.py") if "__pycache__" not in path.parts)


def imported_roots(path: Path) -> set[str]:
    roots: set[str] = set()
    tree = parse_file(path)
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                roots.add(alias.name.split(".", 1)[0])
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                roots.add(node.module.split(".", 1)[0])
    return roots


def parse_file(path: Path) -> ast.Module:
    return ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


def relative(path: Path) -> str:
    return str(path.relative_to(PROJECT_ROOT)).replace("\\", "/")


if __name__ == "__main__":
    main()

