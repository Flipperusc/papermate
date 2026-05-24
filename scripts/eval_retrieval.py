"""Evaluate baseline and optimized Hybrid RAG retrieval on a JSONL seed set."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.errors import AppError
from src.retrieval.bm25_store import BM25Store
from src.retrieval.hybrid_retriever import HybridRetriever
from src.retrieval.query_processor import classify_query, expand_query, get_rrf_weights
from src.retrieval.reranker import LocalReranker
from src.retrieval.rrf import reciprocal_rank_fusion
from src.retrieval.vector_retriever import VectorRetriever


class BaselineRRFOnlyRetriever:
    """Pre-rerank baseline: vector + BM25 + RRF direct topK."""

    def __init__(
        self,
        vector_top_k: int = 20,
        bm25_top_k: int = 20,
        final_top_k: int = 8,
        rrf_k: int = 60,
    ) -> None:
        self.vector_retriever = VectorRetriever()
        self.bm25_store = BM25Store()
        self.vector_top_k = vector_top_k
        self.bm25_top_k = bm25_top_k
        self.final_top_k = final_top_k
        self.rrf_k = rrf_k

    def retrieve(self, paper_id: str, question: str) -> dict[str, Any]:
        query_type = classify_query(question)
        expanded_query = expand_query(question)
        weights = get_rrf_weights(query_type)
        vector_results = _safe_search(
            lambda: self.vector_retriever.search(paper_id, question, self.vector_top_k)
        )
        bm25_results = _safe_search(
            lambda: self.bm25_store.search(paper_id, expanded_query, self.bm25_top_k)
        )
        result_lists: list[list[dict[str, Any]]] = []
        result_weights: list[float] = []
        if vector_results:
            result_lists.append(vector_results)
            result_weights.append(weights[0])
        if bm25_results:
            result_lists.append(bm25_results)
            result_weights.append(weights[1])
        fused = reciprocal_rank_fusion(
            result_lists,
            rrf_k=self.rrf_k,
            weights=result_weights if result_lists else None,
        )
        return {
            "strategy": "baseline_rrf",
            "core_results": fused[: self.final_top_k],
            "final_results": fused[: self.final_top_k],
        }


def main() -> None:
    args = parse_args()
    rows = load_jsonl(args.input)
    if args.limit:
        rows = rows[: args.limit]

    optimized_reranker = LocalReranker() if args.disable_llm_rerank else None
    optimized = HybridRetriever(
        final_top_k=args.final_top_k,
        rerank_enabled=not args.disable_llm_rerank,
        reranker=optimized_reranker,
    )
    baseline = BaselineRRFOnlyRetriever(final_top_k=args.final_top_k)

    report = {
        "input": str(args.input),
        "query_count": len(rows),
        "baseline": evaluate_retriever("baseline_rrf", baseline, rows),
        "optimized": evaluate_retriever("optimized_hybrid_rag", optimized, rows),
    }
    report["relative_lift"] = {
        "recall_at_8": relative_lift(
            report["baseline"]["metrics"]["recall_at_8"],
            report["optimized"]["metrics"]["recall_at_8"],
        ),
        "mrr_at_8": relative_lift(
            report["baseline"]["metrics"]["mrr_at_8"],
            report["optimized"]["metrics"]["mrr_at_8"],
        ),
    }

    print(json.dumps(report, ensure_ascii=False, indent=2))
    if args.output:
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="JSONL with paper_id, question, expected_chunk_ids")
    parser.add_argument("--output", type=Path, help="Optional JSON report path")
    parser.add_argument("--limit", type=int, default=0, help="Evaluate only the first N rows")
    parser.add_argument("--final-top-k", type=int, default=8)
    parser.add_argument(
        "--disable-llm-rerank",
        action="store_true",
        help="Use local rerank for optimized retrieval to avoid LLM calls.",
    )
    return parser.parse_args()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            item = json.loads(stripped)
            expected = item.get("expected_chunk_ids", item.get("expected_chunk_id"))
            if isinstance(expected, str):
                expected_ids = [expected]
            else:
                expected_ids = [str(value) for value in (expected or [])]
            rows.append(
                {
                    "line_number": line_number,
                    "paper_id": str(item["paper_id"]),
                    "question": str(item["question"]),
                    "expected_chunk_ids": expected_ids,
                    "query_type": item.get("query_type", ""),
                }
            )
    return rows


def evaluate_retriever(name: str, retriever: Any, rows: list[dict[str, Any]]) -> dict[str, Any]:
    examples: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    ranks: list[int | None] = []

    for row in rows:
        expected_ids = set(row["expected_chunk_ids"])
        try:
            retrieval = retriever.retrieve(row["paper_id"], row["question"])
            results = retrieval.get("core_results") or retrieval.get("final_results") or retrieval.get("chunks") or []
        except AppError as exc:
            retrieval = {"strategy": "error", "error": exc.code.value}
            results = []
        except Exception as exc:
            retrieval = {"strategy": "error", "error": type(exc).__name__}
            results = []

        got_ids = [str(chunk.get("chunk_id") or "") for chunk in results]
        first_rank = first_hit_rank(got_ids, expected_ids, max_rank=8)
        ranks.append(first_rank)
        example = {
            "line_number": row["line_number"],
            "paper_id": row["paper_id"],
            "question": row["question"],
            "query_type": row["query_type"],
            "expected_chunk_ids": row["expected_chunk_ids"],
            "hit_rank": first_rank,
            "strategy": retrieval.get("strategy", name),
            "retrieved_chunk_ids": got_ids[:8],
        }
        examples.append(example)
        if first_rank is None:
            failures.append(example)

    return {
        "metrics": build_metrics(ranks),
        "failures": failures[:20],
        "examples": examples,
    }


def build_metrics(ranks: list[int | None]) -> dict[str, float]:
    total = max(1, len(ranks))
    return {
        "recall_at_3": sum(1 for rank in ranks if rank is not None and rank <= 3) / total,
        "recall_at_5": sum(1 for rank in ranks if rank is not None and rank <= 5) / total,
        "recall_at_8": sum(1 for rank in ranks if rank is not None and rank <= 8) / total,
        "mrr_at_8": sum((1.0 / rank) for rank in ranks if rank is not None and rank <= 8) / total,
    }


def first_hit_rank(got_ids: list[str], expected_ids: set[str], max_rank: int) -> int | None:
    for index, chunk_id in enumerate(got_ids[:max_rank], start=1):
        if chunk_id in expected_ids:
            return index
    return None


def relative_lift(baseline_value: float, optimized_value: float) -> float | None:
    if baseline_value <= 0:
        return None
    return (optimized_value - baseline_value) / baseline_value


def _safe_search(callback) -> list[dict[str, Any]]:
    try:
        return callback()
    except Exception:
        return []


if __name__ == "__main__":
    main()
