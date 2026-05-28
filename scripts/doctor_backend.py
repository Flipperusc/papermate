"""Run a safe local backend diagnostic report for PaperMate."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.diagnostics import CHECK_ERROR, CHECK_WARNING, collect_backend_diagnostics


def main() -> int:
    args = parse_args()
    report = collect_backend_diagnostics()
    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2))
    else:
        print_report(report)
    return exit_code_for_report(report, strict=args.strict)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true", help="print machine-readable JSON")
    parser.add_argument(
        "--strict",
        action="store_true",
        help="return a non-zero exit code for warnings as well as errors",
    )
    return parser.parse_args()


def print_report(report: dict[str, Any]) -> None:
    """Print a compact human-readable diagnostic report."""
    print(f"PaperMate backend doctor: {report['status']}")
    summary = report.get("summary", {})
    print(
        "summary: "
        f"ok={summary.get('ok', 0)} "
        f"warning={summary.get('warning', 0)} "
        f"error={summary.get('error', 0)}"
    )
    for check in report.get("checks", []):
        status = str(check.get("status", "unknown"))
        name = str(check.get("name", "unknown"))
        message = str(check.get("message", ""))
        print(f"[{status}] {name}: {message}")
        details = check.get("details") or {}
        if details:
            print(f"  details: {json.dumps(details, ensure_ascii=False, sort_keys=True)}")


def exit_code_for_report(report: dict[str, Any], *, strict: bool = False) -> int:
    """Return process exit code for a diagnostic report."""
    status = str(report.get("status") or "")
    if status == CHECK_ERROR:
        return 1
    if strict and status == CHECK_WARNING:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
