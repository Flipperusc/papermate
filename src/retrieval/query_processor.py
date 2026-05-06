"""Query classification and keyword expansion for hybrid retrieval."""

from __future__ import annotations

from dataclasses import dataclass

from src.retrieval.tokenizer import tokenize_text


@dataclass(frozen=True)
class QueryPlan:
    """Processed query information used by vector and BM25 retrieval."""

    original_query: str
    normalized_query: str
    question_type: str
    question_type_label: str
    keywords: list[str]
    expanded_terms: list[str]
    vector_query: str
    bm25_query: str


EXACT_KEYWORDS = [
    "是否",
    "有没有",
    "有无",
    "是否提到",
    "是否使用",
    "数据集",
    "dataset",
    "benchmark",
    "corpus",
    "imagenet",
    "cifar",
    "mimic",
    "指标",
    "评价指标",
    "metric",
    "accuracy",
    "precision",
    "recall",
    "f1",
    "auc",
    "bleu",
    "rouge",
    "表格",
    "table",
    "tab.",
    "图片",
    "figure",
    "fig.",
    "公式",
    "equation",
    "eq.",
    "lora",
    "bert",
    "gpt",
    "transformer",
    "resnet",
    "lstm",
]

SEMANTIC_KEYWORDS = [
    "核心创新点",
    "主要贡献",
    "解决什么问题",
    "方法流程",
    "如何理解",
    "为什么这样设计",
    "传统方法",
    "有什么区别",
    "局限性",
    "总结这篇论文",
    "研究背景",
    "创新点",
    "贡献是什么",
    "流程是什么",
    "解释本文方法",
    "理解本文方法",
    "why",
    "summarize",
    "summary",
    "contribution",
    "contributions",
    "limitation",
    "limitations",
    "background",
]

EXPANSION_RULES = [
    (("数据集",), ["dataset", "datasets", "benchmark", "corpus", "data"]),
    (
        ("评价指标", "指标"),
        ["metric", "metrics", "accuracy", "precision", "recall", "f1", "auc", "bleu", "rouge"],
    ),
    (("实验",), ["experiment", "experiments", "evaluation", "results"]),
    (("方法",), ["method", "approach", "framework", "model", "architecture"]),
    (("局限",), ["limitation", "limitations", "future work", "discussion"]),
    (("贡献", "创新"), ["contribution", "contributions", "novel", "propose", "proposed"]),
    (("消融",), ["ablation", "ablation study"]),
    (("表格",), ["table", "tab."]),
    (("公式",), ["equation", "eq."]),
    (("图", "图片"), ["figure", "fig."]),
]

QUERY_TYPE_LABELS = {
    "exact": "精确匹配类",
    "semantic": "语义理解类",
    "default": "默认类",
}


def classify_query(query: str) -> str:
    """Classify a user query as exact, semantic, or default."""
    normalized_query = normalize_query(query)
    if not normalized_query:
        return "default"

    if any(keyword in normalized_query for keyword in EXACT_KEYWORDS):
        return "exact"

    if any(keyword in normalized_query for keyword in SEMANTIC_KEYWORDS):
        return "semantic"

    return "default"


def expand_query(query: str) -> str:
    """Append a small set of paper-domain English keywords without using an LLM."""
    original_query = str(query or "").strip()
    if not original_query:
        return ""

    expansions: list[str] = []
    for triggers, terms in EXPANSION_RULES:
        if any(trigger in original_query for trigger in triggers):
            expansions.extend(terms)

    unique_expansions = unique_preserve_order(expansions)
    if not unique_expansions:
        return original_query
    return f"{original_query} {' '.join(unique_expansions)}"


def get_rrf_weights(query_type: str) -> list[float]:
    """Return [vector_weight, bm25_weight] for a query type."""
    normalized_type = str(query_type or "").strip().lower()
    if normalized_type == "exact":
        return [0.8, 1.3]
    if normalized_type == "semantic":
        return [1.3, 0.8]
    return [1.0, 1.0]


def process_query(question: str) -> QueryPlan:
    """Return a backward-compatible query plan for the existing retrieval code."""
    normalized_query = " ".join(str(question or "").split())
    question_type = classify_query(normalized_query)
    expanded_query = expand_query(normalized_query)
    keywords = tokenize_text(normalized_query)
    expanded_terms = tokenize_text(expanded_query)

    return QueryPlan(
        original_query=question,
        normalized_query=normalized_query,
        question_type=question_type,
        question_type_label=QUERY_TYPE_LABELS.get(question_type, QUERY_TYPE_LABELS["default"]),
        keywords=keywords,
        expanded_terms=expanded_terms,
        vector_query=normalized_query,
        bm25_query=expanded_query,
    )


def detect_question_type(normalized_query: str, keywords: list[str] | None = None) -> str:
    """Backward-compatible alias for older retrieval code."""
    del keywords
    return classify_query(normalized_query)


def normalize_query(query: str) -> str:
    """Normalize query text for rule matching."""
    return " ".join(str(query or "").lower().split())


def unique_preserve_order(items: list[str]) -> list[str]:
    """Deduplicate items while keeping stable order."""
    seen: set[str] = set()
    unique_items: list[str] = []
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        unique_items.append(item)
    return unique_items
