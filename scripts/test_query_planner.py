"""Smoke tests for enhanced retrieval query planning."""

from __future__ import annotations

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.retrieval.query_planner import plan_query


def main() -> None:
    dataset_plan = plan_query("本文用了哪些数据集？")
    assert dataset_plan.question_type == "exact"
    assert "dataset" in dataset_plan.bm25_query.lower()
    assert "Experiments" in dataset_plan.section_targets

    contribution_plan = plan_query("这篇论文的核心创新点是什么？")
    assert contribution_plan.question_type == "semantic"
    assert "contribution" in contribution_plan.bm25_query.lower()

    table_plan = plan_query("Table 2 里比较了哪些指标？")
    assert table_plan.question_type == "exact"
    assert any(entity.lower() == "table 2" for entity in table_plan.entities)
    assert "metric" in table_plan.bm25_query.lower()

    formula_plan = plan_query("请解释 Eq. 4 的损失函数。")
    assert formula_plan.question_type == "exact"
    assert any(entity.lower() == "equation 4" for entity in formula_plan.entities)
    assert formula_plan.vector_queries == [formula_plan.normalized_query]

    print("query planner tests passed")


if __name__ == "__main__":
    main()
