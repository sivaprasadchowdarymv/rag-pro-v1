"""
Build a per-document concept graph from an existing DocumentIndex (no re-indexing).

Concepts come only from evidence of identity, never from plain words:
  * acronym/symbol definitions   "Power Dissipation (PD)", "Re (Reynolds number)", "where Re is the ..."
  * equation left-hand symbols   "PD = (VIN - VOUT) x IOUT"
  * numbered objects             "Figure 4", "Table 2" (captions) and their references "see Fig. 4"
An abbreviation with two different expansions becomes two concepts (senses);
each mention is assigned by context, else marked ambiguous and never expanded.

Mentions keep modality-specific representations (equation raw/normalized/LaTeX,
table headers/cells/units/caption, figure caption/context/vision text) plus
doc_id (= content-hash version), page, section and node reference.
"""
from __future__ import annotations

import hashlib
from itertools import permutations
import json
import re
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

from rag.models import DocumentIndex, Node

GRAPH_VERSION = 1
RELATIONS = ("SAME_CONCEPT", "ALIAS_OF", "DEFINES", "QUANTIFIES", "VISUALIZES", "REFERENCES", "RELATED_TO")

_TOKEN = re.compile(r"[A-Za-z0-9_µθΘΩ]+(?:-[a-z]+)*")  # "Re-check" is one word, not the symbol Re
_ACR_AFTER = re.compile(r"\b([A-Za-z][A-Za-z\-]+(?:\s+[A-Za-z][A-Za-z\-]+){0,6})\s*\(\s*([A-Z][A-Za-z0-9_]{0,9})\s*\)")
_ACR_BEFORE = re.compile(r"\b([A-Z][A-Za-z0-9_]{0,9})\s*\(\s*(?:the\s+)?([a-z][a-z\-]+(?:\s+[a-z][a-z\-]+){0,5})\s*\)")
_WHERE = re.compile(r"\b(?:where|and)\s+([A-Za-z][A-Za-z0-9_]{0,7})\s+(?:is|denotes|represents)\s+(?:the\s+)?"
                    r"([a-z][a-z\-]+(?:\s+(?!and\b|or\b|is\b)[a-z][a-z\-]+){0,5})")
_EQ_LHS = re.compile(r"^\s*([A-Za-z][A-Za-z0-9_]{0,7})\s*=\s*(.+)$")
_OBJ = re.compile(r"\b(fig(?:ure)?|table|tbl)\.?\s*(\d{1,3})\b", re.I)
_NUM = re.compile(r"\d")
_STOP = {"the", "and", "for", "with", "this", "that", "from", "see", "note", "max", "min", "typ", "a", "an", "of"}
_UNITS = re.compile(r"(?<![A-Za-z])(mV|V|µA|uA|mA|A|mW|W|kΩ|Ω|°C|Hz|kHz|MHz|nF|µF|pF|s|ms|µs)(?![A-Za-z])")


# ---------------------------------------------------------------------------
# Normalisation (OCR-tolerant)
# ---------------------------------------------------------------------------
_OCR = str.maketrans({"0": "o", "1": "l", "5": "s"})


def ocr_norm(tok: str) -> str:
    """Fix one OCR digit inside a word: 'V0UT'->'VOUT', 'Reyno1ds'->'Reynolds'. Part numbers stay."""
    digits = len(_NUM.findall(tok))
    if digits == 1 and len(tok) >= 3 and not tok[-1].isdigit() and not tok[0].isdigit():
        fixed = tok.translate(_OCR)
        return fixed.upper() if tok.isupper() or tok.replace("0", "").replace("1", "").replace("5", "").isupper() else fixed
    return tok


def norm_tokens(text: str) -> List[str]:
    return [ocr_norm(t) for t in _TOKEN.findall(text or "")]


def slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", " ".join(ocr_norm(t) for t in text.split()).lower()).strip("-")


def _in_order(letters: List[str], words: List[str]) -> bool:
    if not words or letters[0] != words[0][0]:
        return False
    joined, pos = "".join(words), 0
    for c in letters:  # every abbreviation letter appears in order in the long form
        pos = joined.find(c, pos)
        if pos < 0:
            return False
        pos += 1
    return True


def _initials_match(abbr: str, long: str) -> bool:
    """'PD'~'power dissipation'; symbol style 'VOUT'~'output voltage' (V + OUT, words swapped)."""
    letters = [c.lower() for c in abbr if c.isalpha()]
    words = [w for w in re.split(r"[\s\-]+", long.lower()) if w and w not in _STOP]
    if not letters or not words or len(words) > 6:
        return False
    if _in_order(letters, words):
        return True
    return len(words) <= 3 and any(_in_order(letters, list(p)) for p in permutations(words) if list(p) != words)


def _trim_long(long: str, abbr: str) -> str:
    """'The device power dissipation' (PD) -> 'power dissipation' (keep as many words as letters)."""
    words = long.split()
    n = max(1, sum(c.isupper() for c in abbr) or len(abbr))
    for k in range(1, min(len(words), n + 1) + 1):  # shortest matching tail wins
        cand = " ".join(words[-k:])
        if _initials_match(abbr, cand) and cand.split()[0].lower() not in _STOP:
            return cand
    return long


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------
@dataclass
class Concept:
    concept_id: str
    canonical: str            # long form when known, else the symbol
    kind: str                 # acronym | symbol | figure | table
    aliases: List[str] = field(default_factory=list)
    sense: str = ""           # expansion that distinguishes homonyms
    defined_on: List[int] = field(default_factory=list)  # pages


@dataclass
class Mention:
    concept_id: str
    doc_id: str
    ref: str                  # "n12" (node) or "f3" (figure)
    modality: str             # text | table | row | pin | equation | figure
    page: int
    section: str
    surface: str
    relation: str             # one of RELATIONS
    confidence: float
    ambiguous: bool = False
    repr: Dict = field(default_factory=dict)


@dataclass
class ConceptGraph:
    doc_id: str
    signature: str
    concepts: Dict[str, Concept] = field(default_factory=dict)
    mentions: List[Mention] = field(default_factory=list)
    relations: List[Tuple[str, str, str, float]] = field(default_factory=list)  # (src, rel, dst, conf)
    build_ms: float = 0.0

    def by_concept(self) -> Dict[str, List[Mention]]:
        out: Dict[str, List[Mention]] = {}
        for m in self.mentions:
            out.setdefault(m.concept_id, []).append(m)
        return out

    def to_json(self) -> str:
        return json.dumps({"version": GRAPH_VERSION, "doc_id": self.doc_id, "signature": self.signature,
                           "concepts": [asdict(c) for c in self.concepts.values()],
                           "mentions": [asdict(m) for m in self.mentions], "relations": self.relations,
                           "build_ms": self.build_ms})

    @classmethod
    def from_json(cls, raw: str) -> "ConceptGraph":
        d = json.loads(raw)
        if d.get("version") != GRAPH_VERSION:
            raise ValueError("old concept graph")
        g = cls(d["doc_id"], d["signature"], build_ms=d.get("build_ms", 0.0))
        g.concepts = {c["concept_id"]: Concept(**c) for c in d["concepts"]}
        g.mentions = [Mention(**m) for m in d["mentions"]]
        g.relations = [tuple(r) for r in d["relations"]]
        return g


# ---------------------------------------------------------------------------
# Modality-specific representations
# ---------------------------------------------------------------------------
def to_latex(expr: str) -> str:
    s = expr.replace("×", r"\times ").replace(" x ", r" \times ").replace("÷", r"\div ")
    s = re.sub(r"\b([A-Z])([A-Z]{2,5}|[a-z]{1,4}|\d)\b", r"\1_{\2}", s)  # VOUT -> V_{OUT}
    return re.sub(r"\(([^()]+)\)\s*/\s*\(([^()]+)\)", r"\\frac{\1}{\2}", s)


def _repr(index: DocumentIndex, node: Node, table_of: Dict[int, int], i: Optional[int]) -> Dict:
    if node.type == "equation":
        raw = node.raw_content.strip()
        return {"raw": raw, "normalized": re.sub(r"\s+", "", raw.replace("×", "*").replace(" x ", "*")),
                "latex": to_latex(raw), "mathml": None}
    if node.type in ("table", "row", "pin"):
        lines = node.raw_content.split("\n")
        header = lines[0] if lines else ""
        split = (lambda s: [c.strip() for c in re.split(r"\s*\|\s*|\t|\s{2,}", s) if c.strip()])
        parent = index.nodes[table_of[i]] if i is not None and i in table_of else node
        cells = split(lines[1]) if node.type != "table" and len(lines) > 1 else []
        return {"caption": parent.caption, "headers": split(header), "row_label": cells[0] if cells else "",
                "cells": cells, "units": sorted(set(_UNITS.findall(node.raw_content)))}
    if node.type == "figure":
        return {"caption": node.caption, "context": (node.meta or {}).get("context", ""),
                "ocr": None, "visual_description": node.vision_text, "image_embedding": None,
                "figure_file": node.figure_file}
    return {}


# ---------------------------------------------------------------------------
# Build
# ---------------------------------------------------------------------------
def _signature(index: DocumentIndex) -> str:
    h = hashlib.sha256(f"{GRAPH_VERSION}|{index.doc_id}|{len(index.nodes)}|{len(index.figures)}".encode())
    return h.hexdigest()[:16]


def _items(index: DocumentIndex):
    for i, n in enumerate(index.nodes):
        yield f"n{i}", i, n
    for j, f in enumerate(index.figures):
        yield f"f{j}", None, f


def _text_of(n: Node) -> str:
    if n.type == "figure":
        return f"{n.caption or ''}\n{(n.meta or {}).get('context', '')}"
    if n.type == "table" and n.caption:
        return f"{n.caption}\n{n.raw_content}"
    return n.raw_content or n.content


def build_graph(index: DocumentIndex, chooser: Optional[Callable[[str, List[str]], Optional[int]]] = None,
                max_llm: int = 20) -> ConceptGraph:
    """`chooser(context, senses) -> sense index` resolves ambiguous mentions (optional, e.g. a small LLM)."""
    t0 = time.perf_counter()
    g = ConceptGraph(index.doc_id, _signature(index))
    pfx = index.doc_id[:10]
    table_of: Dict[int, int] = {}
    cur = None
    for i, n in enumerate(index.nodes):
        if n.type == "table":
            cur = i
        elif n.type in ("row", "pin") and cur is not None:
            table_of[i] = cur

    # 1. Definitions -> concepts (one per (abbr, sense))
    defs: List[Tuple[str, str, str, int, str]] = []  # abbr, long, ref, page, section
    for ref, i, n in _items(index):
        text = _text_of(n)
        for m in _ACR_AFTER.finditer(text):
            long, abbr = m.group(1).strip(), m.group(2)
            if _initials_match(abbr, long):
                defs.append((abbr, _trim_long(long, abbr), ref, n.page, n.section))
        for rx in (_ACR_BEFORE, _WHERE):
            for m in rx.finditer(text):
                abbr, long = m.group(1), m.group(2).strip()
                if abbr.lower() not in _STOP and len(long) > 3:
                    defs.append((abbr, long, ref, n.page, n.section))
    senses: Dict[str, Dict[str, Concept]] = {}
    for abbr, long, ref, page, _ in defs:
        key = ocr_norm(abbr)
        s = slug(long)
        if not s:
            continue
        c = senses.setdefault(key, {}).get(s)
        if c is None:
            c = Concept(f"{pfx}:{key}~{s}", long.lower(), "acronym", [abbr, long.lower()], sense=s)
            senses[key][s] = c
            g.concepts[c.concept_id] = c
            g.relations.append((f"{c.concept_id}#{abbr}", "ALIAS_OF", f"{c.concept_id}#{long.lower()}", 0.95))
        if page not in c.defined_on:
            c.defined_on.append(page)

    # 2. Equation LHS symbols without a definition -> symbol concepts
    for i, n in enumerate(index.nodes):
        if n.type == "equation":
            m = _EQ_LHS.match(n.raw_content.strip().split("\n")[0])
            if m and ocr_norm(m.group(1)) not in senses and len(m.group(1)) >= 2:
                key = ocr_norm(m.group(1))
                c = Concept(f"{pfx}:{key}", key, "symbol", [m.group(1)])
                senses[key] = {"": c}
                g.concepts[c.concept_id] = c

    # 3. Numbered objects (figure N / table N) from captions
    objects: Dict[str, str] = {}
    for ref, i, n in _items(index):
        cap = n.caption or ""
        m = _OBJ.match(cap.strip()) if cap else None
        if m and n.type in ("figure", "table"):
            kind = "figure" if m.group(1).lower().startswith("fig") else "table"
            cid = f"{pfx}:{kind}-{m.group(2)}"
            objects[f"{kind}-{m.group(2)}"] = cid
            if cid not in g.concepts:
                g.concepts[cid] = Concept(cid, f"{kind} {m.group(2)}", kind, [f"{kind.title()} {m.group(2)}"])
            g.mentions.append(Mention(cid, index.doc_id, ref, n.type, n.page, n.section, cap[:40], "SAME_CONCEPT",
                                      1.0, repr=_repr(index, n, table_of, i)))

    # 4. Mentions of every concept in every modality
    section_text: Dict[str, str] = {}
    for _, _, n in _items(index):
        section_text[n.section] = section_text.get(n.section, "") + " " + " ".join(norm_tokens(_text_of(n))).lower().replace("-", " ")
    llm_budget = max_llm
    for ref, i, n in _items(index):
        text = _text_of(n)
        toks = norm_tokens(text)
        tokset, low = set(toks), " " + " ".join(toks).lower().replace("-", " ") + " "
        rep = None
        for key, by_sense in senses.items():
            sym_hit = key in tokset
            long_hits = [c for c in by_sense.values() if c.sense and f" {c.sense.replace('-', ' ')} " in low]
            if not sym_hit and not long_hits:
                continue
            # which sense?
            if long_hits:
                chosen, conf, amb = long_hits[0], 0.9, False
            elif len(by_sense) == 1:
                chosen, conf, amb = next(iter(by_sense.values())), 0.85, False
            else:
                sec = section_text.get(n.section, "")
                ctx_hits = [c for c in by_sense.values() if c.sense.replace("-", " ") in sec]
                page_hits = [c for c in by_sense.values() if n.page in c.defined_on]
                if len(ctx_hits) == 1:
                    chosen, conf, amb = ctx_hits[0], 0.75, False
                elif len(page_hits) == 1:
                    chosen, conf, amb = page_hits[0], 0.6, False
                else:
                    chosen, conf, amb = list(by_sense.values())[0], 0.3, True
                    if chooser and llm_budget > 0:
                        llm_budget -= 1
                        opts = list(by_sense.values())
                        pick = chooser(text[:400], [c.canonical for c in opts])
                        if pick is not None and 0 <= pick < len(opts):
                            chosen, conf, amb = opts[pick], 0.7, False
            rel, conf = _relation(n, key, chosen, text, conf)
            rep = rep if rep is not None else _repr(index, n, table_of, i)
            g.mentions.append(Mention(chosen.concept_id, index.doc_id, ref, n.type, n.page, n.section,
                                      key if sym_hit else chosen.canonical, rel, round(conf, 2), amb, rep))
        for m in _OBJ.finditer(text):  # cross-page references "see Figure 4"
            kind = "figure" if m.group(1).lower().startswith("fig") else "table"
            cid = objects.get(f"{kind}-{m.group(2)}")
            if cid and not (n.caption or "").strip().lower().startswith(m.group(0).lower()[:3]):
                g.mentions.append(Mention(cid, index.doc_id, ref, n.type, n.page, n.section, m.group(0),
                                          "REFERENCES", 0.9, repr=rep if rep is not None else _repr(index, n, table_of, i)))
    g.build_ms = round((time.perf_counter() - t0) * 1000, 2)
    return g


def _relation(n: Node, key: str, c: Concept, text: str, conf: float) -> Tuple[str, float]:
    if n.type == "equation":
        m = _EQ_LHS.match(text.strip().split("\n")[0])
        return ("DEFINES", max(conf, 0.95)) if m and ocr_norm(m.group(1)) == key else ("RELATED_TO", conf * 0.8)
    if n.type in ("row", "pin"):
        return ("QUANTIFIES", conf) if _NUM.search(text.split("\n")[-1]) else ("REFERENCES", conf * 0.8)
    if n.type == "table":
        return "QUANTIFIES", conf * 0.9
    if n.type == "figure":
        return ("VISUALIZES", conf) if key in norm_tokens(n.caption or "") or c.canonical in (n.caption or "").lower() \
            else ("RELATED_TO", conf * 0.6)
    defines = any(rx.search(text) and (key in rx.search(text).group(0)) for rx in (_ACR_AFTER, _ACR_BEFORE, _WHERE)) \
        or re.search(rf"\b{re.escape(key)}\b[^.]{{0,40}}\b(is defined as|is given by|denotes|is the)\b", text)
    return ("DEFINES", max(conf, 0.85)) if defines else ("REFERENCES", conf * 0.9)


# ---------------------------------------------------------------------------
# Cache (memory + JSON next to the document; synced with the workspace)
# ---------------------------------------------------------------------------
_GRAPHS: Dict[str, ConceptGraph] = {}


def get_graph(index: DocumentIndex, chooser=None) -> ConceptGraph:
    sig = _signature(index)
    g = _GRAPHS.get(index.doc_id)
    if g is not None and g.signature == sig:
        return g
    path = Path(index.figures_dir).parent / "concepts.json"
    try:
        g = ConceptGraph.from_json(path.read_text(encoding="utf-8"))
        if g.signature != sig:
            g = None
    except (OSError, ValueError, KeyError, TypeError):
        g = None
    if g is None:
        g = build_graph(index, chooser)
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(g.to_json(), encoding="utf-8")
        except OSError:
            pass
    _GRAPHS[index.doc_id] = g
    return g


def clear_cache() -> None:
    _GRAPHS.clear()


def query_concepts(query: str, graph: ConceptGraph) -> Dict[str, float]:
    """Resolve concepts named in the query (symbols case-sensitive, long forms case-insensitive)."""
    toks = norm_tokens(query)
    tokset, low = set(toks), " " + " ".join(toks).lower().replace("-", " ") + " "
    out: Dict[str, float] = {}
    for c in graph.concepts.values():
        if c.kind in ("figure", "table"):
            continue
        sym = c.concept_id.split(":", 1)[1].split("~")[0]
        if c.sense and f" {c.sense.replace('-', ' ')} " in low:
            out[c.concept_id] = 1.0
        elif sym in tokset:
            out.setdefault(c.concept_id, 0.8)
    for m in _OBJ.finditer(query):
        kind = "figure" if m.group(1).lower().startswith("fig") else "table"
        cid = f"{graph.doc_id[:10]}:{kind}-{m.group(2)}"
        if cid in graph.concepts:
            out[cid] = 1.0
    # an abbreviation with several senses and no long form in the query stays ambiguous: drop it
    by_sym: Dict[str, List[str]] = {}
    for cid in out:
        by_sym.setdefault(cid.split("~")[0], []).append(cid)
    for sym, cids in by_sym.items():
        if len(cids) > 1:
            named = [c for c in cids if out[c] >= 1.0]
            for c in cids:  # keep only the sense the query names; if none is named it stays ambiguous
                if c not in named:
                    out.pop(c)
    return out
