"""Run PaperMate backend verification suites.

The default suite is intentionally local and non-networked. It checks the
application service layer, job runtime, external-call retry policy, schema
migrations, and UI non-blocking guards.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


SMOKE_COMMANDS: list[list[str]] = [
    ["-m", "compileall", "app.py", "config.py", "src", "scripts"],
    ["scripts/test_architecture_boundaries.py"],
    ["scripts/test_team_schema_migration.py"],
    ["scripts/test_team_permissions_jobs.py"],
    ["scripts/test_worker_runtime.py"],
    ["scripts/test_paper_workflow_service.py"],
    ["scripts/test_study_workflow_service.py"],
    ["scripts/test_external_call_policy.py"],
    ["scripts/test_backend_diagnostics.py"],
    ["scripts/test_ui_nonblocking_guards.py"],
]

RETRIEVAL_COMMANDS: list[list[str]] = [
    ["scripts/test_query_processor.py"],
    ["scripts/test_query_planner.py"],
    ["scripts/test_rrf.py"],
    ["scripts/test_context_builder.py"],
    ["scripts/test_evidence_expander.py"],
    ["scripts/test_reranker.py"],
    ["scripts/test_bm25_store.py"],
    ["scripts/test_hybrid_retriever.py"],
]

DEEP_LOCAL_COMMANDS: list[list[str]] = [
    ["scripts/test_semantic_multimodal_chunker.py"],
    ["scripts/test_mineru_visual_normalization.py"],
    ["scripts/test_source_anchor_navigation.py"],
]


def main() -> int:
    args = parse_args()
    commands = commands_for_suite(args.suite)
    started_at = time.perf_counter()
    print(f"Running PaperMate backend verification suite: {args.suite}")
    for index, command in enumerate(commands, start=1):
        label = " ".join([sys.executable, *command])
        print(f"\n[{index}/{len(commands)}] {label}", flush=True)
        result = subprocess.run(
            [sys.executable, *command],
            cwd=PROJECT_ROOT,
            check=False,
        )
        if result.returncode != 0:
            print(f"\nFAILED: {label}", file=sys.stderr)
            return result.returncode

    elapsed = time.perf_counter() - started_at
    print(f"\nBackend verification passed: {args.suite} ({elapsed:.1f}s)")
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--suite",
        choices=("smoke", "retrieval", "all"),
        default="smoke",
        help="smoke checks backend architecture/runtime; retrieval checks RAG local logic; all runs both plus deeper local parsers",
    )
    return parser.parse_args()


def commands_for_suite(suite: str) -> list[list[str]]:
    if suite == "smoke":
        return list(SMOKE_COMMANDS)
    if suite == "retrieval":
        return list(RETRIEVAL_COMMANDS)
    if suite == "all":
        return [*SMOKE_COMMANDS, *RETRIEVAL_COMMANDS, *DEEP_LOCAL_COMMANDS]
    raise ValueError(f"unsupported suite: {suite}")


if __name__ == "__main__":
    raise SystemExit(main())
