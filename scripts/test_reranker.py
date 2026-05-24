"""Smoke tests for retrieval rerankers."""

from __future__ import annotations

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.retrieval.query_planner import plan_query
from src.retrieval.reranker import LLMReranker, LocalReranker, parse_rerank_response


class FakeLLMClient:
    def generate(self, prompt: str, **kwargs) -> str:
        del prompt, kwargs
        return """
        [
          {"chunk_id":"chunk_low","relevance_score":1,"reason":"weak"},
          {"chunk_id":"chunk_high","relevance_score":4,"reason":"direct"}
        ]
        """


class BadLLMClient:
    def generate(self, prompt: str, **kwargs) -> str:
        del prompt, kwargs
        return "not json"


def main() -> None:
    query_plan = plan_query("Which dataset is used in Table 2?")
    candidates = [
        {
            "chunk_id": "chunk_low",
            "section_title": "Introduction",
            "text": "The paper discusses motivation.",
            "rrf_score": 0.04,
            "retrieval_sources": ["vector"],
        },
        {
            "chunk_id": "chunk_high",
            "section_title": "Experiments",
            "text": "Table 2 reports CIFAR-10 and ImageNet results.",
            "rrf_score": 0.02,
            "retrieval_sources": ["bm25"],
        },
    ]

    parsed = parse_rerank_response('```json\n[{"chunk_id":"a","relevance_score":3,"reason":"ok"}]\n```')
    assert parsed[0]["chunk_id"] == "a"

    llm_results = LLMReranker(llm_client=FakeLLMClient(), batch_size=2).rerank(query_plan, candidates)
    assert llm_results[0]["chunk_id"] == "chunk_high"
    assert llm_results[0]["rerank_source"] == "deepseek"
    assert llm_results[0]["final_score"] > llm_results[1]["final_score"]

    fallback_results = LLMReranker(llm_client=BadLLMClient(), batch_size=2).rerank(query_plan, candidates)
    assert fallback_results
    assert all(item["rerank_source"] == "local" for item in fallback_results)

    local_results = LocalReranker().rerank(query_plan, candidates)
    assert local_results
    assert "exact_overlap" in local_results[0]
    print("reranker tests passed")


if __name__ == "__main__":
    main()
