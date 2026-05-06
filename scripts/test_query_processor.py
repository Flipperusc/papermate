"""Smoke tests for the PaperMate query processor."""

from __future__ import annotations

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.retrieval.query_processor import classify_query, expand_query, get_rrf_weights


def main() -> None:
    """Run lightweight assertions without requiring pytest."""
    assert classify_query("本文用了哪些数据集？") == "exact"
    assert classify_query("这篇论文的核心创新点是什么？") == "semantic"
    assert classify_query("请解释本文方法流程") == "semantic"
    assert classify_query("论文是否提到 MIMIC-III？") == "exact"

    expanded = expand_query("本文用了哪些评价指标？")
    for expected in ("metric", "accuracy", "f1", "auc"):
        assert expected in expanded.lower(), f"missing expansion: {expected}"

    assert get_rrf_weights("exact") == [0.8, 1.3]
    print("query processor tests passed")


if __name__ == "__main__":
    main()
