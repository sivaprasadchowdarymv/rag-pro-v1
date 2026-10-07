"""
RAG metrics for one answer (live), plus the composite RAG health score.

Keeps every original metric (rag/evaluation.py) and adds:
  answer_relevance     cosine(question, answer) with the local embedding model
  citation_precision / citation_coverage / numerical_score / equation_score  (verifier)
  stage latency (ms), tokens, LLM calls

Retrieval Precision@K / Recall@K / MRR / Hit rate need ground truth and
are computed in benchmark mode (metrics/benchmark.py).

The health score is an APPLICATION-LEVEL COMPOSITE, not a scientific
measure of correctness. Weights are configurable (HEALTH_WEIGHTS).
"""
from __future__ import annotations

from typing import Dict, Optional, Sequence

import numpy as np

from config.settings import Settings, get_logger
from rag.evaluation import (metric_completeness, metric_coverage, metric_diversity,
                            metric_faithfulness)
from rag.models import SourceRef

log = get_logger("metrics")


def answer_relevance(question: str, answer: str, settings: Settings) -> float:
    """Embedding similarity (0..1); falls back to word overlap if embeddings fail."""
    try:
        from rag.embeddings import embed_query, embed_texts
        from storage.vector_store import normalize_rows

        q = embed_query(question, settings)
        a = normalize_rows(np.asarray(embed_texts([answer[:2000]], settings, settings.embed_model), dtype=np.float32))[0]
        return round(float(max(0.0, min(1.0, float(q @ a)))), 3)
    except Exception:
        return metric_coverage([answer], question)


def health_score(m: Dict[str, float], weights: Sequence[float], feedback_ratio: Optional[float]) -> float:
    """S = w1*faithfulness + w2*relevance + w3*citation accuracy + w4*grounding + w5*user feedback."""
    citation = (m.get("citation_precision", 0) * m.get("citation_coverage", 0)
                * m.get("citation_accuracy", 1.0) * m.get("citation_support", 1.0))
    grounding = (m.get("numerical_score", 0) + m.get("equation_score", 0)) / 2
    fb = 0.5 if feedback_ratio is None else feedback_ratio  # neutral until users rate answers
    parts = (m.get("faithfulness", 0), m.get("answer_relevance", 0), citation, grounding, fb)
    return round(100 * sum(w * p for w, p in zip(weights, parts)), 1)


def compute(question: str, answer: str, sources: Sequence[SourceRef], verification: Dict, settings: Settings,
            stage_latency: Dict[str, float], tokens: int, llm_calls: int) -> Dict[str, float]:
    chunks = [s.full_text for s in sources]
    m: Dict[str, float] = {
        "faithfulness": metric_faithfulness(answer, chunks),
        "context_coverage": metric_coverage(chunks, question),
        "source_diversity": metric_diversity(sources),
        "answer_completeness": metric_completeness(answer),
        "answer_relevance": answer_relevance(question, answer, settings),
        "citation_precision": verification.get("citation_precision", 0.0),
        "citation_coverage": verification.get("citation_coverage", 0.0),
        "citation_accuracy": verification.get("citation_accuracy", 0.0),
        "citation_support": verification.get("citation_support", 0.0),
        "numerical_score": verification.get("numerical_score", 0.0),
        "equation_score": verification.get("equation_score", 0.0),
        "grounded": 1.0 if verification.get("grounded") else 0.0,
        "tokens": tokens,
        "llm_calls": llm_calls,
        "latency_s": round(sum(stage_latency.values()) / 1000, 2),
    }
    m["health_score"] = health_score(m, settings.health_weights, None)
    return m
