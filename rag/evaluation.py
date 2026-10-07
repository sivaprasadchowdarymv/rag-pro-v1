"""
Retrieval / answer quality metrics (lexical heuristics, no extra LLM calls).

Old RAG.py equivalents: `_STOP`, `_words()`, `metric_faithfulness()`,
`metric_coverage()`, `metric_diversity()`, `metric_avg_score()`,
`metric_completeness()`, `compute_metrics()` — formulas unchanged.
Latency is now included in the returned dict.

These are cheap heuristics, not ground truth:
  faithfulness   share of answer words that also appear in the excerpts
  coverage       share of question words found in the excerpts
  diversity      number of source types used / 5
  retrieval      average top-1 hybrid score across engines
  completeness   answer length heuristic (0 if NOT FOUND)
  confidence     weighted blend of the above
"""
from __future__ import annotations

import re
from typing import Dict, List

from rag.models import RetrievedItem, SourceRef

_STOP = {
    "the", "a", "an", "is", "are", "was", "were", "of", "in", "to", "and", "or",
    "for", "on", "at", "by", "with", "this", "that", "it", "be", "as", "what",
    "how", "does", "do",
}  # fmt: skip


def _words(text: str) -> set:
    return set(re.findall(r"[a-z0-9]+", (text or "").lower())) - _STOP


def metric_faithfulness(answer: str, chunks: List[str]) -> float:
    ctx = set().union(*[_words(c) for c in chunks])
    ans = _words(answer)
    return round(len(ans & ctx) / max(len(ans), 1), 3)


def metric_coverage(chunks: List[str], query: str) -> float:
    qw = _words(query)
    ctw = set().union(*[_words(c) for c in chunks])
    return round(len(qw & ctw) / max(len(qw), 1), 3)


def metric_diversity(sources: List[SourceRef]) -> float:
    return round(len({s.type for s in sources}) / 5.0, 3)


def metric_avg_score(engine_results: Dict[str, List[RetrievedItem]]) -> float:
    top = [items[0].score for items in engine_results.values() if items]
    return round(sum(top) / len(top), 2) if top else 0.0


def metric_completeness(answer: str) -> float:
    if not answer or "NOT FOUND" in answer.upper():
        return 0.0
    return round(min(1.0, len(answer.split()) / 80.0), 3)


def compute_metrics(
    answer: str,
    sources: List[SourceRef],
    query: str,
    engine_results: Dict[str, List[RetrievedItem]],
    latency: float,
) -> Dict[str, float]:
    chunks = [s.full_text for s in sources]
    fa = metric_faithfulness(answer, chunks)
    cov = metric_coverage(chunks, query)
    div = metric_diversity(sources)
    avg = metric_avg_score(engine_results)
    comp = metric_completeness(answer)
    conf = round((fa * 0.35 + cov * 0.25 + min(avg / 100, 1) * 0.25 + comp * 0.15) * 100, 1)
    return {
        "faithfulness": fa,
        "context_coverage": cov,
        "source_diversity": div,
        "avg_retrieval_score": avg,
        "answer_completeness": comp,
        "confidence": conf,
        "latency_s": round(latency, 2),
    }
