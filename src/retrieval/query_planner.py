"""Enhanced query planning for Hybrid RAG retrieval."""

from __future__ import annotations

import re
from dataclasses import dataclass

from src.retrieval.tokenizer import tokenize_text


@dataclass(frozen=True)
class QueryPlan:
    """Processed query information used by vector, BM25, and reranking."""

    original_query: str
    normalized_query: str
    question_type: str
    question_type_label: str
    keywords: list[str]
    expanded_terms: list[str]
    vector_query: str
    bm25_query: str
    entities: list[str]
    section_targets: list[str]
    exact_terms: list[str]
    vector_queries: list[str]


EXACT_KEYWORDS = [
    "是否",
    "有没有",
    "有无",
    "是否提到",
    "是否使用",
    "哪一页",
    "第几页",
    "数据集",
    "dataset",
    "datasets",
    "benchmark",
    "corpus",
    "imagenet",
    "cifar",
    "mimic",
    "指标",
    "评价指标",
    "metric",
    "metrics",
    "accuracy",
    "precision",
    "recall",
    "f1",
    "auc",
    "bleu",
    "rouge",
    "表格",
    "表 ",
    "table",
    "tab.",
    "图片",
    "图 ",
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
    (("数据集", "dataset", "benchmark", "corpus"), ["dataset", "datasets", "benchmark", "corpus", "data"]),
    (
        ("评价指标", "指标", "metric"),
        ["metric", "metrics", "accuracy", "precision", "recall", "f1", "auc", "bleu", "rouge"],
    ),
    (("实验", "结果", "性能"), ["experiment", "experiments", "evaluation", "results", "performance"]),
    (("方法", "模型", "框架"), ["method", "approach", "framework", "model", "architecture"]),
    (("局限", "不足", "未来"), ["limitation", "limitations", "future work", "discussion"]),
    (("贡献", "创新"), ["contribution", "contributions", "novel", "propose", "proposed"]),
    (("消融",), ["ablation", "ablation study"]),
    (("表格", "表 "), ["table", "tab."]),
    (("公式",), ["equation", "eq."]),
    (("图", "图片"), ["figure", "fig."]),
]

SECTION_RULES = [
    (("摘要", "abstract", "总结", "overview"), ["Abstract"]),
    (("背景", "相关工作", "related", "background"), ["Introduction", "Related Work"]),
    (("方法", "模型", "框架", "流程", "method", "approach", "architecture"), ["Method"]),
    (("实验", "数据集", "指标", "结果", "性能", "experiment", "evaluation", "dataset", "metric"), ["Experiments", "Results"]),
    (("贡献", "创新", "contribution", "novel"), ["Abstract", "Introduction", "Conclusion"]),
    (("局限", "不足", "未来", "limitation", "future"), ["Results", "Conclusion"]),
    (("表格", "table", "图", "figure", "公式", "equation"), ["Experiments", "Results", "Method"]),
]

QUERY_TYPE_LABELS = {
    "exact": "精确匹配类",
    "semantic": "语义理解类",
    "default": "默认类",
}

NUMBERED_ENTITY_PATTERN = re.compile(
    r"\b(?:table|tab\.?|figure|fig\.?|equation|eq\.?)\s*\d+(?:\.\d+)*\b"
    r"|(?:表|图|公式)\s*\d+(?:\.\d+)*",
    flags=re.IGNORECASE,
)
SCIENTIFIC_ENTITY_PATTERN = re.compile(
    r"\b[A-Za-z][A-Za-z0-9]*(?:[-_.][A-Za-z0-9]+)+\b"
    r"|\b[A-Z][A-Za-z]*\d+[A-Za-z0-9_.-]*\b"
    r"|\b(?:BERT|GPT|LoRA|ResNet|LSTM|CNN|RNN|Transformer|ImageNet|CIFAR-?\d*|MIMIC-?[A-Z0-9]*)\b",
    flags=re.IGNORECASE,
)


def plan_query(question: str) -> QueryPlan:
    """Build a retrieval plan from the user question."""
    normalized_query = " ".join(str(question or "").split())
    question_type = classify_query(normalized_query)
    entities = extract_entities(normalized_query)
    section_targets = detect_section_targets(normalized_query)
    expanded_query = expand_query(normalized_query)
    keywords = tokenize_text(normalized_query)
    expanded_terms = tokenize_text(expanded_query)
    exact_terms = unique_preserve_order([*entities, *keywords])
    bm25_query = " ".join(
        unique_preserve_order(
            [
                normalized_query,
                expanded_query,
                *entities,
                *section_targets,
            ]
        )
    )
    vector_queries = unique_preserve_order([normalized_query])

    return QueryPlan(
        original_query=str(question or ""),
        normalized_query=normalized_query,
        question_type=question_type,
        question_type_label=QUERY_TYPE_LABELS.get(question_type, QUERY_TYPE_LABELS["default"]),
        keywords=keywords,
        expanded_terms=expanded_terms,
        vector_query=normalized_query,
        bm25_query=bm25_query,
        entities=entities,
        section_targets=section_targets,
        exact_terms=exact_terms,
        vector_queries=vector_queries,
    )


def classify_query(query: str) -> str:
    """Classify a user query as exact, semantic, or default."""
    normalized_query = normalize_query(query)
    if not normalized_query:
        return "default"
    if any(keyword in normalized_query for keyword in EXACT_KEYWORDS):
        return "exact"
    if NUMBERED_ENTITY_PATTERN.search(query or ""):
        return "exact"
    if any(keyword in normalized_query for keyword in SEMANTIC_KEYWORDS):
        return "semantic"
    return "default"


def expand_query(query: str) -> str:
    """Append paper-domain English keywords without using an LLM."""
    original_query = str(query or "").strip()
    if not original_query:
        return ""

    normalized_query = original_query.lower()
    expansions: list[str] = []
    for triggers, terms in EXPANSION_RULES:
        if any(trigger.lower() in normalized_query for trigger in triggers):
            expansions.extend(terms)

    entities = extract_entities(original_query)
    unique_expansions = unique_preserve_order([*expansions, *entities])
    if not unique_expansions:
        return original_query
    return f"{original_query} {' '.join(unique_expansions)}"


def detect_section_targets(query: str) -> list[str]:
    """Return likely paper sections for the query intent."""
    normalized_query = normalize_query(query)
    targets: list[str] = []
    for triggers, sections in SECTION_RULES:
        if any(trigger.lower() in normalized_query for trigger in triggers):
            targets.extend(sections)
    return unique_preserve_order(targets)


def extract_entities(query: str) -> list[str]:
    """Extract exact-match entities such as table numbers, datasets, and model names."""
    text = str(query or "")
    entities: list[str] = []
    for pattern in (NUMBERED_ENTITY_PATTERN, SCIENTIFIC_ENTITY_PATTERN):
        for match in pattern.finditer(text):
            entity = " ".join(match.group(0).split()).strip(" ,;:()[]{}")
            if entity:
                entities.append(entity)
                normalized = normalize_numbered_entity(entity)
                if normalized and normalized != entity.lower():
                    entities.append(normalized)
    return unique_preserve_order(entities)


def normalize_numbered_entity(entity: str) -> str:
    """Normalize figure/table/equation references for BM25 matching."""
    text = " ".join(str(entity or "").lower().replace(".", "").split())
    compact_match = re.match(r"^(table|tab|figure|fig|equation|eq)(\d+(?:\.\d+)*)$", text)
    if compact_match:
        prefix = compact_match.group(1)
        number = compact_match.group(2)
        replacement = {
            "tab": "table",
            "fig": "figure",
            "eq": "equation",
        }.get(prefix, prefix)
        return f"{replacement} {number}"
    zh_match = re.match(r"^(表|图|公式)(\d+(?:\.\d+)*)$", text)
    if zh_match:
        replacement = {
            "表": "table",
            "图": "figure",
            "公式": "equation",
        }[zh_match.group(1)]
        return f"{replacement} {zh_match.group(2)}"
    replacements = {
        "tab ": "table ",
        "fig ": "figure ",
        "eq ": "equation ",
        "表 ": "table ",
        "图 ": "figure ",
        "公式 ": "equation ",
    }
    for prefix, replacement in replacements.items():
        if text.startswith(prefix):
            return f"{replacement}{text[len(prefix):].strip()}"
    return text


def get_rrf_weights(query_type: str) -> list[float]:
    """Return [vector_weight, bm25_weight] for a query type."""
    normalized_type = str(query_type or "").strip().lower()
    if normalized_type == "exact":
        return [0.8, 1.3]
    if normalized_type == "semantic":
        return [1.3, 0.8]
    return [1.0, 1.0]


def normalize_query(query: str) -> str:
    """Normalize query text for rule matching."""
    return " ".join(str(query or "").lower().split())


def unique_preserve_order(items: list[str]) -> list[str]:
    """Deduplicate items while keeping stable order."""
    seen: set[str] = set()
    unique_items: list[str] = []
    for item in items:
        clean_item = str(item or "").strip()
        key = clean_item.lower()
        if not clean_item or key in seen:
            continue
        seen.add(key)
        unique_items.append(clean_item)
    return unique_items
