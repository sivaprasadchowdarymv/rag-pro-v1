"""
Run the planned specialist agents over the selected documents, then rerank
and fuse their evidence. Every stage is timed; every agent's calls,
results, failures and latency are recorded for the metrics page.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence

from agents.orchestrator import Plan
from agents.specialists import SPECIALISTS, QueryContext
from concepts.expand import expand
from concepts.graph import get_graph
from config.settings import Settings, get_logger
from rag.models import DocumentIndex, Evidence
from retrieval.evidence_fusion import FusionResult, fuse, merge, rrf
from retrieval.reranker import rerank

log = get_logger("retrieval_stage")
RERANK_POOL = 20


@dataclass
class RetrievalOutcome:
    fusion: FusionResult
    agent_stats: Dict[str, Dict[str, float]] = field(default_factory=dict)
    latency_ms: Dict[str, float] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)
    reranked: bool = False


def run_agents(agents: Sequence[str], query: str, indexes: Sequence[DocumentIndex], settings: Settings,
               describe: Optional[Callable] = None, ctx: Optional[QueryContext] = None):
    ctx = ctx or QueryContext(query, settings)
    groups: List[List[Evidence]] = []
    stats: Dict[str, Dict[str, float]] = {}
    for name in agents:
        fn = SPECIALISTS[name]
        st = stats.setdefault(name, {"calls": 0, "results": 0, "failures": 0, "latency_ms": 0.0})
        for index in indexes:
            t0 = time.perf_counter()
            st["calls"] += 1
            try:
                found = fn(index, ctx, describe) if name == "figure_agent" else fn(index, ctx)
                groups.append(found)
                st["results"] += len(found)
            except Exception:  # one failing agent must never break the answer
                log.exception("%s failed on %s", name, index.filename)
                st["failures"] += 1
            st["latency_ms"] += (time.perf_counter() - t0) * 1000
    return groups, stats, ctx


def run_retrieval(plan: Plan, indexes: Sequence[DocumentIndex], settings: Settings,
                  describe: Optional[Callable] = None, chooser: Optional[Callable] = None) -> RetrievalOutcome:
    t0 = time.perf_counter()
    groups, stats, ctx = run_agents(plan.agents, plan.retrieval_query, indexes, settings, describe)
    t_retrieval = (time.perf_counter() - t0) * 1000

    # Concept/entity resolution -> selective cross-modal expansion (deterministic, cached graph).
    t_concepts = 0.0
    if getattr(settings, "concept_linking", False) and indexes:
        try:
            if chooser is not None:
                for ix in indexes:
                    get_graph(ix, chooser)
            existing = {e.key for g in groups for e in g}
            linked, cstats = expand(plan.retrieval_query, plan.intents, indexes, existing,
                                    settings.concept_expand_max, settings.concept_min_conf,
                                    settings.max_chars_per_chunk)
            t_concepts = cstats["latency_ms"]
            if linked:
                groups.append(linked)
            stats["concept_linker"] = {"calls": 1, "results": cstats["added"], "failures": 0,
                                       "latency_ms": t_concepts, "concepts": cstats["concepts"]}
        except Exception:  # linking is an enhancement: never break retrieval
            log.exception("concept linking failed")
            stats["concept_linker"] = {"calls": 1, "results": 0, "failures": 1, "latency_ms": 0.0}

    # Rerank the best candidates by RRF with the cross-encoder.
    pool = merge(groups)
    rrf(pool)
    pool.sort(key=lambda e: e.score, reverse=True)
    pool = pool[:RERANK_POOL]
    scores, t_rerank, warn = rerank(plan.retrieval_query, [e.content for e in pool], settings)
    rerank_scores = {e.key: s for e, s in zip(pool, scores)} if scores else None

    t1 = time.perf_counter()
    budget_items = settings.max_context_chunks + (2 if len(indexes) > 1 else 0)
    fused = fuse(groups, budget_items, settings.max_context_chars, rerank_scores)
    t_fusion = (time.perf_counter() - t1) * 1000

    if "concept_linker" in stats:  # what actually reached the LLM context because of linking
        kept = [e for e in fused.evidence if set(e.ranks) == {"concept"}]
        stats["concept_linker"].update(kept=len(kept), extra_tokens=round(sum(len(e.content) for e in kept) / 4))
    warnings = list(dict.fromkeys(ctx.warnings + ([warn] if warn else [])))
    lat = {"retrieval": t_retrieval, "rerank": t_rerank, "fusion": t_fusion}
    if t_concepts:
        lat["concepts"] = t_concepts
    return RetrievalOutcome(fused, stats, lat, warnings, reranked=bool(scores))
