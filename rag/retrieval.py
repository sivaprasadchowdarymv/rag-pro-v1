"""
Hybrid retrieval: semantic + fuzzy + bonuses - depth penalty.

Old RAG.py equivalents: `_fuzzy()`, `semantic_search()`, `text_engine()`,
`table_engine()`, `row_engine()`, `equation_engine()`,
`figure_caption_engine()`, and the threshold filter / de-duplication inside
`answer_query()`.

The scoring formula is unchanged:

    score = max(semantic*100, fuzzy)
            + 15  if the node's section title appears in the query
            + 10  if a query word (>3 chars) appears in the parent context
            +  8  per metadata field that contains a query word
            -  2 * chunk depth

What changed (same results, less work):
  * The query is embedded ONCE per question. RAG.py embedded it once per
    engine, i.e. 4 identical Ollama calls per question.
  * Cosine similarity for all nodes is one NumPy matrix-vector product.
  * No thread pool: the work is now a few milliseconds.
  * Hits are ordered by score (RAG.py used thread completion order, so the
    order of sources — and which ones fit in the LLM context — could change
    between identical runs).

Every function here is pure (no Ollama, no Streamlit), so it can be unit-tested.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import numpy as np
from rapidfuzz import fuzz

from config.settings import SCORE_THRESHOLDS
from rag.models import DocumentIndex, Node, RetrievedItem

# engine name -> node types it searches (unchanged from RAG.py)
ENGINES: Dict[str, Sequence[str]] = {
    "text": ("text",),
    "table": ("table",),
    "row": ("row", "pin"),
    "equation": ("equation",),
}


def fuzzy_score(query: str, text: str) -> float:
    return fuzz.token_set_ratio(query.lower(), text.lower())


def semantic_scores(index: DocumentIndex, query_vec: Optional[np.ndarray]) -> np.ndarray:
    """Cosine similarity * 100 for every node (0 where no embedding exists)."""
    n = len(index.nodes)
    if (
        query_vec is None
        or index.embeddings.ndim != 2
        or index.embeddings.shape[0] != n
        or index.embeddings.shape[1] != query_vec.shape[0]
    ):
        return np.zeros(n, dtype=np.float32)
    sims = index.embeddings @ query_vec.astype(np.float32)
    sims[~index.has_embedding] = 0.0
    return sims * 100.0


def _bonuses(query_lower: str, query_words: List[str], long_words: List[str], node: Node) -> float:
    title_bonus = 15 if node.section and node.section.lower() in query_lower else 0

    ctx = (node.parent_ctx or "").lower()
    ctx_bonus = 10 if ctx and any(w in ctx for w in long_words) else 0

    meta_bonus = 0
    for values in (node.meta or {}).values():
        for v in values if isinstance(values, list) else [values]:
            v_lower = ("" if v is None else str(v)).lower()
            if any(w in v_lower for w in query_words):
                meta_bonus += 8
                break

    return title_bonus + ctx_bonus + meta_bonus - node.depth * 2


def retrieve(
    query: str,
    index: DocumentIndex,
    top_k: int,
    node_types: Optional[Sequence[str]] = None,
    query_vec: Optional[np.ndarray] = None,
    semantic: Optional[np.ndarray] = None,
    engine: str = "",
) -> List[RetrievedItem]:
    """
    Score nodes of the given types and return the best `top_k`, best first.

    `query_vec` is the normalised query embedding (None = fuzzy only).
    `semantic` lets callers reuse precomputed semantic scores across engines.
    """
    if semantic is None:
        semantic = semantic_scores(index, query_vec)

    if node_types:
        candidates: List[int] = []
        for t in node_types:
            candidates.extend(index.indices_by_type.get(t, []))
        candidates.sort()  # document order, so ties break exactly as in RAG.py
    else:
        candidates = list(range(len(index.nodes)))

    query_lower = query.lower()
    words = query_lower.split()
    long_words = [w for w in words if len(w) > 3]

    scored: List[RetrievedItem] = []
    for i in candidates:
        node = index.nodes[i]
        fz = fuzz.token_set_ratio(query_lower, index.lower_content[i])
        score = max(float(semantic[i]), fz) + _bonuses(query_lower, words, long_words, node)
        scored.append(RetrievedItem(score=score, node=node, engine=engine or node.type, node_index=i))

    scored.sort(key=lambda r: r.score, reverse=True)
    return scored[:top_k]


def run_engines(
    query: str, index: DocumentIndex, top_k: int, query_vec: Optional[np.ndarray]
) -> Dict[str, List[RetrievedItem]]:
    """Run the text / table / row / equation engines with shared semantic scores."""
    semantic = semantic_scores(index, query_vec)
    return {
        name: retrieve(query, index, top_k, types, semantic=semantic, engine=name)
        for name, types in ENGINES.items()
    }


def gather_hits(engine_results: Dict[str, List[RetrievedItem]]) -> List[RetrievedItem]:
    """Keep hits above each engine's threshold, drop duplicates, best first."""
    hits = [
        item
        for name, items in engine_results.items()
        for item in items
        if item.score >= SCORE_THRESHOLDS.get(name, 52)
    ]
    hits.sort(key=lambda r: r.score, reverse=True)

    unique: List[RetrievedItem] = []
    seen = set()
    for item in hits:
        key = (item.node.page, item.node.raw_content[:60])  # same key as RAG.py
        if key not in seen:
            seen.add(key)
            unique.append(item)
    return unique


def search_figures(query: str, figures: List[Node], top_k: int = 3) -> List[RetrievedItem]:
    """Fuzzy match the question against figure captions (unchanged)."""
    scored = [
        RetrievedItem(score=fuzzy_score(query, f.caption or f.content), node=f, engine="figure", node_index=i)
        for i, f in enumerate(figures)
    ]
    scored.sort(key=lambda r: r.score, reverse=True)
    return scored[:top_k]
