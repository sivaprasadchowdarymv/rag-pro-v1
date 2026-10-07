"""
Specialist retrieval agents. Deterministic, local and fast; no LLM calls
except the optional vision description of a figure.

Each agent searches ONE kind of element and returns structured Evidence:
    TextAgent       text chunks (with section breadcrumbs)
    TableAgent      whole tables + individual rows / pins          (TBL_n)
    EquationAgent   formulas + variables, units, surrounding text  (EQ_n)
    FigureAgent     figure captions + nearby text (+ vision)       (FIG_n)
    DocumentAgent   title, author, pages, sections, counts

Candidates come from two retrievers, each recorded as a rank:
    "hybrid"  the original semantic + fuzzy + bonus scorer (rag/retrieval.py)
    "bm25"    the new lexical index (exact symbols, part numbers)
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np

from config.settings import SCORE_THRESHOLDS, Settings, get_logger
from rag.embeddings import EmbeddingError, embed_query
from rag.models import DocumentIndex, Evidence, Node
from rag.retrieval import retrieve, semantic_scores
from retrieval.bm25 import BM25Index

log = get_logger("specialists")

_VAR = re.compile(r"\b[A-Z][A-Za-z]{0,3}[A-Z0-9]{0,5}\b")
_UNIT = re.compile(r"\b(?:mV|V|µA|uA|mA|A|mW|W|kΩ|MΩ|Ω|ohms?|°C|K|Hz|kHz|MHz|nF|µF|uF|pF|watts?|volts?|amperes?)\b")
_NOT_VARS = {"V", "A", "W", "K", "THE", "AND", "FOR", "WITH", "WHERE", "IS"}
_FIG_NUM = re.compile(r"\bfig(?:ure)?\.?\s*(\d+)", re.I)


# ---------------------------------------------------------------------------
# Per-document caches (built lazily, reused across questions)
# ---------------------------------------------------------------------------
@dataclass
class DocTools:
    bm25: BM25Index
    eq_ids: Dict[int, str]
    tbl_ids: Dict[int, str]  # table AND row/pin node index -> TBL_n
    page_lines: Dict[int, List[str]]
    fig_bm25: Optional[BM25Index] = None


_DOC_CACHE: Dict[str, DocTools] = {}


def doc_tools(index: DocumentIndex) -> DocTools:
    key = f"{index.doc_id}:{len(index.nodes)}"
    if key in _DOC_CACHE:
        return _DOC_CACHE[key]
    eq_ids, tbl_ids, page_lines = {}, {}, {}
    current_tbl = ""
    for i, n in enumerate(index.nodes):
        if n.type == "equation":
            eq_ids[i] = f"EQ_{len(eq_ids) + 1}"
        elif n.type == "table":
            current_tbl = f"TBL_{len([v for v in set(tbl_ids.values())]) + 1}"
            tbl_ids[i] = current_tbl
        elif n.type in ("row", "pin"):
            tbl_ids[i] = current_tbl
        elif n.type == "text":
            page_lines.setdefault(n.page, []).extend(ln for ln in n.raw_content.split("\n") if ln.strip())
    fig_texts = [f"{f.caption or ''} {(f.meta or {}).get('context', '')}" for f in index.figures]
    tools = DocTools(BM25Index([n.content for n in index.nodes]), eq_ids, tbl_ids, page_lines,
                     BM25Index(fig_texts) if fig_texts else None)
    _DOC_CACHE[key] = tools
    return tools


@dataclass
class QueryContext:
    """Per-question state shared by all agents (embedding computed once per document)."""

    query: str
    settings: Settings
    semantic: Dict[str, np.ndarray] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)

    def semantic_for(self, index: DocumentIndex) -> np.ndarray:
        if index.doc_id not in self.semantic:
            qvec = None
            if index.has_embedding.any():
                try:
                    qvec = embed_query(self.query, self.settings, index.embed_model)
                except EmbeddingError as exc:
                    self.warnings.append(exc.user_message)
            self.semantic[index.doc_id] = semantic_scores(index, qvec)
        return self.semantic[index.doc_id]


def _candidates(index: DocumentIndex, ctx: QueryContext, types: Sequence[str], engine: str,
                k: int = 8) -> List[tuple]:
    """(node_index, ranks) from hybrid + BM25 for the given node types."""
    hybrid = retrieve(ctx.query, index, k, types, semantic=ctx.semantic_for(index), engine=engine)
    floor = SCORE_THRESHOLDS.get(engine, 52) - 10  # looser than legacy: the reranker filters
    out: Dict[int, Dict[str, int]] = {}
    for rank, item in enumerate(i for i in hybrid if i.score >= floor):
        out.setdefault(item.node_index, {})["hybrid"] = rank + 1
    pool = sorted(i for t in types for i in index.indices_by_type.get(t, []))
    if pool:
        scores = doc_tools(index).bm25.scores(ctx.query, pool)
        order = np.argsort(-scores)[:k]
        for rank, j in enumerate(order):
            if scores[j] > 0:
                out.setdefault(pool[j], {})["bm25"] = rank + 1
    return list(out.items())


def _evidence(index: DocumentIndex, i: int, agent: str, ranks: Dict[str, int], content: str,
              item_id: str = "", meta: Optional[dict] = None) -> Evidence:
    n: Node = index.nodes[i]
    return Evidence(key=f"{index.doc_id}#n{i}", doc_id=index.doc_id, doc_name=index.filename, type=n.type,
                    page=n.page, section=n.section, content=content, excerpt=n.raw_content, agent=agent,
                    item_id=item_id, parent_ctx=n.parent_ctx, meta=meta if meta is not None else (n.meta or {}),
                    node_index=i, depth=n.depth, chunk_idx=n.chunk_idx, ranks=dict(ranks))


def _cap(text: str, settings: Settings) -> str:
    return text[: settings.max_chars_per_chunk]


# ---------------------------------------------------------------------------
# Agents
# ---------------------------------------------------------------------------
def text_agent(index: DocumentIndex, ctx: QueryContext) -> List[Evidence]:
    return [_evidence(index, i, "text_agent", r, _cap(index.nodes[i].content, ctx.settings))
            for i, r in _candidates(index, ctx, ("text",), "text")]


def table_agent(index: DocumentIndex, ctx: QueryContext) -> List[Evidence]:
    tools, out = doc_tools(index), []
    for types, engine in ((("table",), "table"), (("row", "pin"), "row")):
        for i, r in _candidates(index, ctx, types, engine, k=6):
            n = index.nodes[i]
            tid = tools.tbl_ids.get(i, "")
            cap = f" ({n.caption})" if n.caption else ""
            label = f"Table {tid}{cap}" if n.type == "table" else f"Row of {tid}"
            out.append(_evidence(index, i, "table_agent", r, _cap(f"{label}:\n{n.raw_content}", ctx.settings), tid))
    return out


def analyse_equation(line: str, page_lines: Sequence[str]) -> dict:
    lhs, _, rhs = line.partition("=")
    variables = [v for v in dict.fromkeys(_VAR.findall(line)) if len(v) >= 2 and v.upper() not in _NOT_VARS]
    # A formula relates symbols with an operator (PD = (VIN - VOUT) x IOUT). A list of
    # assignments ("VIN = 10V, IO = 500mA, TJ = 25°C") or a value ("VOUT = 5.0V") is not one.
    is_assignment_list = line.count("=") > 1 and "," in line
    has_operator = bool(re.search(r"[-+*/×÷^()]|\sx\s", rhs))
    is_formula = (bool(rhs) and bool(_VAR.search(rhs)) and has_operator and not is_assignment_list
                  and not re.fullmatch(r"\s*[-\d.,\s]+\s*[A-Za-zµ°Ω]*\s*", rhs))
    context = ""
    for j, ln in enumerate(page_lines):
        if line.strip() and line.strip() in ln:
            context = " ".join(page_lines[max(0, j - 1): j + 3])
            break
    return {"variables": variables, "units": list(dict.fromkeys(_UNIT.findall(f"{line} {context}"))),
            "is_formula": is_formula, "context": context[:400]}


def equation_agent(index: DocumentIndex, ctx: QueryContext) -> List[Evidence]:
    tools, out = doc_tools(index), []
    q_symbols = {s.upper() for s in _VAR.findall(ctx.query) if len(s) >= 2}
    for i, r in _candidates(index, ctx, ("equation",), "equation", k=8):
        n = index.nodes[i]
        info = analyse_equation(n.raw_content, tools.page_lines.get(n.page, []))
        ranks = dict(r)
        lhs_vars = {v.upper() for v in _VAR.findall(n.raw_content.partition("=")[0])}
        if q_symbols & lhs_vars:
            ranks["variable_match"] = 1  # the equation DEFINES the asked symbol
        elif q_symbols & {v.upper() for v in info["variables"]}:
            ranks["variable_match"] = 3  # the equation only uses it
        if info["is_formula"]:
            ranks["formula"] = 1  # real formulas outrank spec lines like "VOUT = 5.0V"
        eid = tools.eq_ids.get(i, "")
        content = (f"{eid} [{n.section}]: {n.raw_content}\n"
                   f"Variables: {', '.join(info['variables']) or '-'}; units: {', '.join(info['units']) or '-'}\n"
                   f"Context: {info['context'] or '-'}")
        out.append(_evidence(index, i, "equation_agent", ranks, _cap(content, ctx.settings), eid,
                             meta={**(n.meta or {}), **info}))
    return out


def figure_agent(index: DocumentIndex, ctx: QueryContext, describe=None) -> List[Evidence]:
    """`describe(fig, path) -> str` optionally adds a vision description (top figure only)."""
    tools = doc_tools(index)
    if not index.figures or tools.fig_bm25 is None:
        return []
    order = list(range(len(index.figures)))
    scores = tools.fig_bm25.scores(ctx.query, order)
    wanted = _FIG_NUM.search(ctx.query)
    out: List[Evidence] = []
    ranked = sorted(order, key=lambda j: -scores[j])
    for rank, j in enumerate(ranked[:4]):
        fig = index.figures[j]
        ranks = {"bm25": rank + 1} if scores[j] > 0 else {}
        if wanted and re.search(rf"\bfig(?:ure)?\.?\s*{wanted.group(1)}\b", fig.caption or "", re.I):
            ranks["figure_number"] = 1
        if not ranks:
            continue
        fid = f"FIG_{j + 1}"
        context = (fig.meta or {}).get("context", "")
        content = f"{fid} (page {fig.page}): {fig.caption or 'no caption'}\nNearby text: {context or '-'}"
        if describe and not out:
            path = index.figure_path(fig)
            desc = describe(fig, path) if path else ""
            if desc:
                content += f"\nVisual description: {desc}"
        out.append(Evidence(key=f"{index.doc_id}#f{j}", doc_id=index.doc_id, doc_name=index.filename, type="figure",
                            page=fig.page, section=fig.caption or "Figure", content=_cap(content, ctx.settings),
                            excerpt=f"{fig.caption or 'Figure'}\n{context}".strip(), agent="figure_agent",
                            item_id=fid, meta={"caption": fig.caption}, figure_file=fig.figure_file or "",
                            ranks=ranks))
    return out


def document_agent(index: DocumentIndex, ctx: QueryContext) -> List[Evidence]:
    m, s = index.doc_meta or {}, index.stats()
    sections = ", ".join(f"{t} (p{p})" for t, p in (m.get("sections") or [])[:25])
    lines = [f"Document: {index.filename}", f"Title: {m.get('title') or '-'}"]
    if m.get("author"):
        lines.append(f"Author: {m['author']}")
    lines += [f"Pages: {s['pages']}; text chunks: {s['text']}; tables: {s['table']}; equations: {s['equation']}; "
              f"figures: {s['figure']}", f"Sections: {sections or '-'}"]
    if m.get("scanned_pages"):
        lines.append(f"Image-only (scanned) pages without text: {m['scanned_pages'][:10]}")
    text = "\n".join(lines)
    return [Evidence(key=f"{index.doc_id}#meta", doc_id=index.doc_id, doc_name=index.filename, type="metadata",
                     page=1, section="Document information", content=_cap(text, ctx.settings), excerpt=text,
                     agent="document_agent", ranks={"document": 1})]


SPECIALISTS = {"text_agent": text_agent, "table_agent": table_agent, "equation_agent": equation_agent,
               "figure_agent": figure_agent, "document_agent": document_agent}
