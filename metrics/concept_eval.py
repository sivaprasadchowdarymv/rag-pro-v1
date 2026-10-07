"""Cross-modal linking evaluation: entity-linking P/R, relation accuracy, cross-modal recall,
false-link rate, latency and extra tokens/cost.   python -m metrics.concept_eval"""
from __future__ import annotations

import json
import tempfile
import time
from pathlib import Path
from typing import Dict

from concepts.expand import expand
from concepts.golden import FORBIDDEN_LINKS, GOLD_LINKS, GOLD_QUERIES, golden_index
from concepts.graph import build_graph, clear_cache


def evaluate(cost_per_1k: float = 0.0) -> Dict:
    clear_cache()
    with tempfile.TemporaryDirectory() as tmp:
        index = golden_index(Path(tmp))
        g = build_graph(index)
        pfx = index.doc_id[:10] + ":"
        pred = {(m.concept_id[len(pfx):], m.ref): m.relation for m in g.mentions if not m.ambiguous}
        gold = {(c, r): rel for c, r, rel in GOLD_LINKS}
        # only score predictions for concepts that the gold set covers
        scored = {k: v for k, v in pred.items() if k[0] in {c for c, _ in gold}}
        tp = [k for k in scored if k in gold]
        rel_ok = sum(1 for k in tp if scored[k] == gold[k])
        false_links = [k for k in FORBIDDEN_LINKS if k in pred]
        recall_hits, recall_total, extra_chars, lat = 0, 0, 0, []
        for q, initial, expected, forbidden in GOLD_QUERIES:
            keys = {f"{index.doc_id}#{r}" for r in initial}
            t0 = time.perf_counter()
            added, stats = expand(q, [], [index], keys)
            lat.append((time.perf_counter() - t0) * 1000)
            got = {e.key.split("#")[1] for e in added}
            recall_hits += len(expected & got)
            recall_total += len(expected)
            false_links += [(q, r) for r in forbidden & got]
            extra_chars += stats["added_chars"]
    n = len(GOLD_QUERIES)
    extra_tokens = extra_chars / 4 / n
    return {"entity_linking_precision": round(len(tp) / max(len(scored), 1), 3),
            "entity_linking_recall": round(len(tp) / len(gold), 3),
            "relation_accuracy": round(rel_ok / max(len(tp), 1), 3),
            "cross_modal_retrieval_recall": round(recall_hits / recall_total, 3),
            "false_link_rate": round(len(false_links) / (len(FORBIDDEN_LINKS) + sum(len(x[3]) for x in GOLD_QUERIES)), 3),
            "false_links": [list(map(str, f)) for f in false_links],
            "graph_build_ms": g.build_ms, "expansion_latency_ms_avg": round(sum(lat) / n, 2),
            "additional_tokens_per_query_max": round(extra_tokens, 1),
            "additional_cost_per_query": round(extra_tokens / 1000 * cost_per_1k, 6),
            "llm_calls_for_linking": 0}


if __name__ == "__main__":
    print(json.dumps(evaluate(), indent=2))
