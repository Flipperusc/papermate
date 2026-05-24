"""Candidate reranking for Hybrid RAG retrieval."""

from __future__ import annotations

import json
from typing import Any

from src.llm_client import DEEPSEEK_CALL_FAILED_MESSAGE, LLMClient
from src.logger import get_logger
from src.retrieval.query_planner import QueryPlan
from src.retrieval.tokenizer import tokenize_text


logger = get_logger(__name__)


class LocalReranker:
    """Feature-based reranker used directly or as the LLM fallback."""

    def rerank(
        self,
        query_plan: QueryPlan,
        candidates: list[dict[str, Any]],
        top_k: int | None = None,
    ) -> list[dict[str, Any]]:
        """Return candidates sorted by local retrieval features."""
        pool = list(candidates[: max(1, top_k or len(candidates))])
        rrf_norms = _normalize_rrf_scores(pool)

        reranked: list[dict[str, Any]] = []
        for candidate in pool:
            chunk_id = str(candidate.get("chunk_id") or "")
            rrf_score_norm = rrf_norms.get(chunk_id, 0.0)
            exact_overlap = calculate_exact_overlap(query_plan, candidate)
            section_boost = calculate_section_boost(query_plan, candidate)
            source_diversity = calculate_source_diversity(candidate)
            final_score = (
                0.55 * rrf_score_norm
                + 0.25 * exact_overlap
                + 0.15 * section_boost
                + 0.05 * source_diversity
            )
            item = dict(candidate)
            item.update(
                {
                    "rerank_score": final_score,
                    "llm_relevance_score": None,
                    "rrf_score_norm": rrf_score_norm,
                    "exact_overlap": exact_overlap,
                    "section_boost": section_boost,
                    "source_diversity": source_diversity,
                    "final_score": final_score,
                    "rerank_source": "local",
                    "rerank_reason": "local feature fallback",
                }
            )
            reranked.append(item)

        return _sort_reranked(reranked)


class LLMReranker:
    """DeepSeek reranker with deterministic local fallback."""

    def __init__(
        self,
        enabled: bool = True,
        top_k: int = 30,
        batch_size: int = 8,
        llm_client: LLMClient | None = None,
        fallback: LocalReranker | None = None,
    ) -> None:
        self.enabled = bool(enabled)
        self.top_k = max(1, int(top_k or 30))
        self.batch_size = max(1, int(batch_size or 8))
        self.llm_client = llm_client or LLMClient()
        self.fallback = fallback or LocalReranker()

    def rerank(
        self,
        query_plan: QueryPlan,
        candidates: list[dict[str, Any]],
        top_k: int | None = None,
    ) -> list[dict[str, Any]]:
        """Rerank candidates by answerability, falling back on local features."""
        if not candidates:
            return []
        limit = max(1, int(top_k or self.top_k))
        pool = list(candidates[:limit])
        if not self.enabled:
            return self.fallback.rerank(query_plan, pool, top_k=len(pool))

        try:
            llm_scores = self._score_with_llm(query_plan, pool)
            if not llm_scores:
                raise ValueError("LLM reranker returned no scores")
        except Exception as exc:
            logger.warning("LLM rerank failed; using local fallback. error=%s", exc)
            return self.fallback.rerank(query_plan, pool, top_k=len(pool))

        rrf_norms = _normalize_rrf_scores(pool)
        reranked: list[dict[str, Any]] = []
        for candidate in pool:
            chunk_id = str(candidate.get("chunk_id") or "")
            llm_payload = llm_scores.get(chunk_id, {})
            llm_raw_score = _clamp_float(llm_payload.get("relevance_score"), 0.0, 4.0)
            llm_score_norm = llm_raw_score / 4.0
            rrf_score_norm = rrf_norms.get(chunk_id, 0.0)
            exact_overlap = calculate_exact_overlap(query_plan, candidate)
            section_boost = calculate_section_boost(query_plan, candidate)
            source_diversity = calculate_source_diversity(candidate)
            final_score = (
                0.65 * llm_score_norm
                + 0.20 * rrf_score_norm
                + 0.10 * section_boost
                + 0.05 * exact_overlap
            )
            item = dict(candidate)
            item.update(
                {
                    "rerank_score": llm_score_norm,
                    "llm_relevance_score": llm_raw_score,
                    "rrf_score_norm": rrf_score_norm,
                    "exact_overlap": exact_overlap,
                    "section_boost": section_boost,
                    "source_diversity": source_diversity,
                    "final_score": final_score,
                    "rerank_source": "deepseek",
                    "rerank_reason": str(llm_payload.get("reason") or "")[:240],
                }
            )
            reranked.append(item)

        return _sort_reranked(reranked)

    def _score_with_llm(
        self,
        query_plan: QueryPlan,
        candidates: list[dict[str, Any]],
    ) -> dict[str, dict[str, Any]]:
        scores: dict[str, dict[str, Any]] = {}
        for start in range(0, len(candidates), self.batch_size):
            batch = candidates[start : start + self.batch_size]
            prompt = _build_rerank_prompt(query_plan, batch)
            response = self.llm_client.generate(prompt, temperature=0.0, max_tokens=900)
            if response == DEEPSEEK_CALL_FAILED_MESSAGE:
                raise RuntimeError("DeepSeek call failed")
            for item in parse_rerank_response(response):
                chunk_id = str(item.get("chunk_id") or "").strip()
                if not chunk_id:
                    continue
                scores[chunk_id] = {
                    "relevance_score": _clamp_float(item.get("relevance_score"), 0.0, 4.0),
                    "reason": str(item.get("reason") or ""),
                }
        return scores


def parse_rerank_response(response: str) -> list[dict[str, Any]]:
    """Parse the JSON array returned by the LLM reranker."""
    text = str(response or "").strip()
    if text.startswith("```"):
        text = text.strip("`").strip()
        if text.lower().startswith("json"):
            text = text[4:].strip()

    start = text.find("[")
    end = text.rfind("]")
    if start < 0 or end < start:
        raise ValueError("rerank response does not contain a JSON array")

    payload = json.loads(text[start : end + 1])
    if not isinstance(payload, list):
        raise ValueError("rerank response is not a JSON array")
    return [item for item in payload if isinstance(item, dict)]


def calculate_exact_overlap(query_plan: QueryPlan, candidate: dict[str, Any]) -> float:
    """Return a 0..1 overlap score for query entities/terms in a candidate."""
    terms = query_plan.exact_terms or query_plan.keywords or tokenize_text(query_plan.normalized_query)
    terms = [term for term in terms if len(str(term).strip()) > 1]
    if not terms:
        return 0.0

    candidate_text = " ".join(
        str(candidate.get(key) or "")
        for key in ("section_title", "text", "search_text")
    ).lower()
    candidate_tokens = set(tokenize_text(candidate_text))

    matched = 0
    for term in terms:
        clean_term = str(term).strip().lower()
        if not clean_term:
            continue
        term_tokens = tokenize_text(clean_term)
        if clean_term in candidate_text or (term_tokens and set(term_tokens).issubset(candidate_tokens)):
            matched += 1

    return min(1.0, matched / max(1, len(terms)))


def calculate_section_boost(query_plan: QueryPlan, candidate: dict[str, Any]) -> float:
    """Return a 0..1 boost when the candidate matches intended sections."""
    if not query_plan.section_targets:
        return 0.0
    section_title = str(candidate.get("section_title") or "").lower()
    text = str(candidate.get("text") or "").lower()

    best = 0.0
    for target in query_plan.section_targets:
        target_text = str(target or "").lower()
        if not target_text:
            continue
        if target_text in section_title:
            best = max(best, 1.0)
        elif target_text in text:
            best = max(best, 0.5)
    return best


def calculate_source_diversity(candidate: dict[str, Any]) -> float:
    """Return a small boost for chunks found by multiple retrieval sources."""
    sources = candidate.get("retrieval_sources")
    if not isinstance(sources, list):
        sources = [candidate.get("retrieval_source")] if candidate.get("retrieval_source") else []
    return min(1.0, len({str(source) for source in sources if source}) / 2.0)


def _normalize_rrf_scores(candidates: list[dict[str, Any]]) -> dict[str, float]:
    scores = [float(candidate.get("rrf_score") or 0.0) for candidate in candidates]
    max_score = max(scores, default=0.0)
    if max_score <= 0:
        return {str(candidate.get("chunk_id") or ""): 0.0 for candidate in candidates}
    return {
        str(candidate.get("chunk_id") or ""): max(0.0, float(candidate.get("rrf_score") or 0.0) / max_score)
        for candidate in candidates
    }


def _build_rerank_prompt(query_plan: QueryPlan, candidates: list[dict[str, Any]]) -> str:
    candidate_blocks: list[str] = []
    for candidate in candidates:
        text = str(candidate.get("text") or "")
        candidate_blocks.append(
            "\n".join(
                [
                    f"chunk_id: {candidate.get('chunk_id')}",
                    f"section: {candidate.get('section_title', '')}",
                    f"page: {candidate.get('page_num', '')}",
                    f"text: {text[:900]}",
                ]
            )
        )

    return (
        "你是论文检索重排序器。请判断每个片段能否直接回答用户问题。\n"
        "只返回 JSON 数组，不要返回解释性正文。数组元素格式："
        '{"chunk_id":"...","relevance_score":0-4,"reason":"..."}。\n'
        "评分标准：4=可直接回答，3=强相关但需要少量上下文，2=部分相关，1=弱相关，0=无关。\n\n"
        f"问题：{query_plan.original_query}\n"
        f"问题类型：{query_plan.question_type}\n"
        f"精确实体：{', '.join(query_plan.entities)}\n\n"
        "候选片段：\n"
        + "\n\n---\n\n".join(candidate_blocks)
    )


def _sort_reranked(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(
        items,
        key=lambda item: (
            -float(item.get("final_score") or 0.0),
            -float(item.get("rerank_score") or 0.0),
            -float(item.get("rrf_score") or 0.0),
            str(item.get("chunk_id") or ""),
        ),
    )


def _clamp_float(value: Any, minimum: float, maximum: float) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        number = minimum
    return min(maximum, max(minimum, number))
