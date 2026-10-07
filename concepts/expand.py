"""
Selective cross-modal expansion, inserted between initial retrieval and reranking.

  query -> initial evidence -> resolve concepts (query names them) -> pick ONLY the
  modalities the query asks for -> add linked items not already retrieved (capped)
  -> the existing reranker + fusion decide what reaches the LLM (dedup + budget).

Security: candidates come only from the graphs of the indexes passed in (the user's
active, authorised workspace documents). A mention whose doc_id is not in that set
is dropped, so linking can never surface another tenant's or version's content.
"""
from __future__ import annotations

import re
import time
from typing import Dict, List, Sequence, Set, Tuple

from concepts.graph import ConceptGraph, Mention, get_graph, query_concepts
from rag.models import DocumentIndex, Evidence

_WANT = {
    "equation": re.compile(r"\b(formula|equation|expression|calculat\w*|comput\w*|derive|how (?:do|to) (?:i )?"
                           r"(?:get|find|calc))\b", re.I),
    "figure": re.compile(r"\b(fig(?:ure)?|graph|plot|curve|diagram|chart|shown|image|picture)\b", re.I),
    "table": re.compile(r"\b(table|tbl|value|values|max(?:imum)?|min(?:imum)?|typ(?:ical)?|rating|spec\w*|"
                        r"limit|range)\b", re.I),
    "definition": re.compile(r"\b(what is|what are|define|definition|meaning|stands? for|explain)\b", re.I),
}
_MODALITY = {"equation": {"equation"}, "figure": {"figure"}, "table": {"table", "row", "pin"}}
_INTENT_TO_WANT = {"equation": "equation", "calculation": "equation", "figure": "figure", "table": "table"}
_REL_RANK = {"SAME_CONCEPT": 0, "DEFINES": 1, "QUANTIFIES": 2, "VISUALIZES": 2, "REFERENCES": 3, "RELATED_TO": 4}


def wanted_modalities(query: str, intents: Sequence[str] = ()) -> Set[str]:
    want = {k for k, rx in _WANT.items() if rx.search(query)}
    want |= {_INTENT_TO_WANT[i] for i in intents if i in _INTENT_TO_WANT}
    if "equation" in want:
        want.add("definition")  # "formula" -> equation + its definition
    if "figure" in want and "table" in want:
        want.add("text")        # "compare Figure 4 and Table 2" -> figure + table + relevant text
    return want


def _to_evidence(index: DocumentIndex, m: Mention, rank: int, max_chars: int) -> Evidence:
    from agents.specialists import _evidence, doc_tools  # reuse the existing evidence builder
    meta = {"concept_id": m.concept_id, "relation": m.relation, "link_confidence": m.confidence, **m.repr}
    if m.ref.startswith("f"):
        j = int(m.ref[1:])
        fig = index.figures[j]
        ctx = (fig.meta or {}).get("context", "")
        return Evidence(key=f"{index.doc_id}#f{j}", doc_id=index.doc_id, doc_name=index.filename, type="figure",
                        page=fig.page, section=fig.caption or "Figure",
                        content=f"FIG_{j + 1} (page {fig.page}): {fig.caption or 'no caption'}\nNearby text: {ctx or '-'}"
                        [:max_chars], excerpt=f"{fig.caption or 'Figure'}\n{ctx}".strip(), agent="concept_linker",
                        item_id=f"FIG_{j + 1}", meta=meta, figure_file=fig.figure_file or "",
                        ranks={"concept": rank})
    i = int(m.ref[1:])
    tools = doc_tools(index)
    item = tools.eq_ids.get(i) or tools.tbl_ids.get(i, "")
    return _evidence(index, i, "concept_linker", {"concept": rank}, index.nodes[i].content[:max_chars], item,
                     meta={**(index.nodes[i].meta or {}), **meta})


def expand(query: str, intents: Sequence[str], indexes: Sequence[DocumentIndex], existing_keys: Set[str],
           max_items: int = 4, min_conf: float = 0.6, max_chars: int = 900) -> Tuple[List[Evidence], Dict]:
    t0 = time.perf_counter()
    stats: Dict = {"concepts": 0, "added": 0, "added_chars": 0, "skipped_unauthorised": 0}
    want = wanted_modalities(query, intents)
    allowed = {ix.doc_id for ix in indexes}
    if not want or max_items <= 0:
        stats["latency_ms"] = (time.perf_counter() - t0) * 1000
        return [], stats
    cands: List[Tuple[Tuple, DocumentIndex, Mention]] = []
    for index in indexes:
        g: ConceptGraph = get_graph(index)
        qc = query_concepts(query, g)
        stats["concepts"] += len(qc)
        if not qc:
            continue
        by_c = g.by_concept()
        for cid, qconf in qc.items():
            for m in by_c.get(cid, []):
                if m.doc_id not in allowed or m.doc_id != index.doc_id:
                    stats["skipped_unauthorised"] += 1
                    continue
                if m.ambiguous or m.confidence < min_conf:
                    continue
                ok = (m.modality in _MODALITY.get("equation", set()) and "equation" in want) \
                    or (m.modality == "figure" and "figure" in want) \
                    or (m.modality in _MODALITY["table"] and "table" in want) \
                    or (m.modality == "text" and m.relation == "DEFINES" and ("definition" in want or "text" in want)) \
                    or (m.modality == "text" and "text" in want and m.relation == "REFERENCES")
                if not ok:
                    continue
                # tables: prefer the specific row/pin that quantifies over the whole table
                pri = (_REL_RANK.get(m.relation, 5), m.modality == "table", -m.confidence * qconf, m.page)
                cands.append((pri, index, m))
    cands.sort(key=lambda x: x[0])
    out: List[Evidence] = []
    seen = set(existing_keys)
    per_modality: Dict[str, int] = {}
    for _, index, m in cands:
        key = f"{index.doc_id}#{m.ref}"
        mod = "table" if m.modality in _MODALITY["table"] else m.modality
        if key in seen or per_modality.get(mod, 0) >= (1 if mod == "table" else 2):
            continue
        ev = _to_evidence(index, m, len(out) + 1, max_chars)
        seen.add(key)
        per_modality[mod] = per_modality.get(mod, 0) + 1
        out.append(ev)
        if len(out) >= max_items:
            break
    stats.update(added=len(out), added_chars=sum(len(e.content) for e in out),
                 latency_ms=(time.perf_counter() - t0) * 1000)
    return out, stats
