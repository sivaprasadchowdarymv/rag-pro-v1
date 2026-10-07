"""
Evidence fusion: merge specialist results, remove duplicates, rank, detect
conflicts, and keep only the best evidence within the token budget.

Ranking = Reciprocal Rank Fusion over the retrievers that found an item
(hybrid, BM25, specialist-specific), then the cross-encoder reranker when
available. Source references (doc, page, section, EQ/TBL/FIG ids) are kept.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

from rag.models import Evidence

RRF_K = 60
_WORDS = re.compile(r"[a-z0-9]+")
_ASSIGN = re.compile(r"\b([A-Z][A-Za-z]{0,3}[A-Z0-9]{0,4})\s*=\s*(-?\d+(?:\.\d+)?)\s*([a-zA-Zµ°Ω]*)")


@dataclass
class FusionResult:
    evidence: List[Evidence]
    conflicts: List[str] = field(default_factory=list)
    dropped_duplicates: int = 0
    candidates: int = 0


def rrf(evidence: Sequence[Evidence]) -> None:
    """Set .score from the per-retriever ranks (Reciprocal Rank Fusion)."""
    for ev in evidence:
        ev.score = sum(1.0 / (RRF_K + r) for r in ev.ranks.values())


def merge(groups: Sequence[Sequence[Evidence]]) -> List[Evidence]:
    """Same item found by several agents/retrievers -> one item with all ranks."""
    by_key: Dict[str, Evidence] = {}
    for group in groups:
        for ev in group:
            if ev.key in by_key:
                for name, r in ev.ranks.items():
                    by_key[ev.key].ranks[name] = min(r, by_key[ev.key].ranks.get(name, r))
            else:
                by_key[ev.key] = ev
    return list(by_key.values())


def _jaccard(a: str, b: str) -> float:
    wa, wb = set(_WORDS.findall(a.lower())), set(_WORDS.findall(b.lower()))
    return len(wa & wb) / max(len(wa | wb), 1)


def _near_duplicate(ev: Evidence, kept: List[Evidence]) -> bool:
    body = ev.excerpt.strip().lower()
    for k in kept:
        if k.doc_id != ev.doc_id:
            continue
        other = k.excerpt.strip().lower()
        if body and body in other:  # e.g. a table row already inside a selected table
            return True
        if _jaccard(ev.excerpt, k.excerpt) >= 0.8:  # overlapping text chunks
            return True
    return False


def detect_conflicts(evidence: Sequence[Evidence]) -> List[str]:
    """Same symbol, same section, different values -> worth stating conditions.
    (Different sections, e.g. test conditions vs absolute maximum ratings, are normal.)"""
    seen: Dict[tuple, tuple] = {}
    notes: List[str] = []
    for ev in evidence:
        for sym, val, unit in _ASSIGN.findall(ev.excerpt):
            key = (ev.doc_id, ev.section.lower(), sym)
            if key in seen and seen[key][0] != val:
                notes.append(f"{sym} has different values in '{ev.section}': {seen[key][0]}{seen[key][1]} "
                             f"(p{seen[key][2]}) and {val}{unit} (p{ev.page}); state the conditions for each.")
            seen.setdefault(key, (val, unit, ev.page))
    return list(dict.fromkeys(notes))[:5]


def fuse(groups: Sequence[Sequence[Evidence]], max_items: int, max_chars: int,
         rerank_scores: Optional[Dict[str, float]] = None) -> FusionResult:
    items = merge(groups)
    rrf(items)
    if rerank_scores:  # reranker decides the order; RRF breaks ties
        items.sort(key=lambda e: (rerank_scores.get(e.key, -1e9), e.score), reverse=True)
    else:
        items.sort(key=lambda e: e.score, reverse=True)

    kept: List[Evidence] = []
    used, dropped = 0, 0
    for ev in items:
        if len(kept) >= max_items:
            break
        if _near_duplicate(ev, kept):
            dropped += 1
            continue
        if kept and used + len(ev.content) > max_chars:
            continue
        kept.append(ev)
        used += len(ev.content)

    # Confidence: rank-based and relative to this answer (1.0 = best evidence).
    for i, ev in enumerate(kept):
        ev.confidence = round(max(0.05, 1.0 - 0.12 * i), 2)
    return FusionResult(kept, detect_conflicts(kept), dropped, len(items))
