"""
Data classes shared by the whole pipeline.

Old RAG.py equivalent: `class Node` (+ the plain dicts used for sources and
results). The Node fields are unchanged except:
  * `embedding` moved out of Node into one NumPy matrix on DocumentIndex
    (much smaller cache files and vectorised cosine similarity);
  * `figure_path` became `figure_file` (a file name, not an absolute server
    path) so cached indexes are portable and server paths are never exposed.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from functools import cached_property
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np


@dataclass
class Node:
    type: str  # text | table | row | pin | equation | figure
    section: str  # nearest heading
    content: str  # text sent to the embedder + LLM (has parent-context prefix)
    raw_content: str  # original un-prefixed text (shown to the user)
    page: int
    chunk_idx: int = 0  # sibling index within the section
    depth: int = 0  # recursion depth when split
    parent_ctx: str = ""  # injected ancestor heading chain (breadcrumb)
    meta: Dict[str, Any] = field(default_factory=dict)
    figure_file: Optional[str] = None  # figures only: file name in figures dir
    caption: Optional[str] = None  # figures only
    vision_text: Optional[str] = None  # figures only: cached LLaVA description

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": self.type,
            "section": self.section,
            "content": self.content,
            "raw_content": self.raw_content,
            "page": self.page,
            "chunk_idx": self.chunk_idx,
            "depth": self.depth,
            "parent_ctx": self.parent_ctx,
            "meta": self.meta,
            "figure_file": self.figure_file,
            "caption": self.caption,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Node":
        allowed = {k: v for k, v in d.items() if k in cls.__dataclass_fields__}
        return cls(**allowed)


@dataclass
class DocumentIndex:
    """Everything needed to answer questions about one PDF."""

    doc_id: str
    filename: str
    page_count: int
    nodes: List[Node]  # text / table / row / pin / equation nodes
    figures: List[Node]  # figure nodes (kept separate, as in RAG.py)
    figures_dir: Path
    embed_model: str = ""
    doc_meta: Dict[str, Any] = field(default_factory=dict)  # title, author, sections, scanned_pages
    # Row i belongs to nodes[i]. Rows are L2-normalised; rows without an
    # embedding are all zeros and flagged False in `has_embedding`.
    embeddings: np.ndarray = field(
        default_factory=lambda: np.zeros((0, 0), dtype=np.float32), repr=False
    )
    has_embedding: np.ndarray = field(
        default_factory=lambda: np.zeros((0,), dtype=bool), repr=False
    )

    # --- Derived lookups (computed once, used by retrieval) --------------
    @cached_property
    def lower_content(self) -> List[str]:
        return [n.content.lower() for n in self.nodes]

    @cached_property
    def indices_by_type(self) -> Dict[str, List[int]]:
        out: Dict[str, List[int]] = {}
        for i, n in enumerate(self.nodes):
            out.setdefault(n.type, []).append(i)
        return out

    @property
    def missing_embeddings(self) -> int:
        if len(self.has_embedding) != len(self.nodes):
            return len(self.nodes)
        return int((~self.has_embedding).sum())

    def figure_path(self, fig: Node) -> Optional[Path]:
        if not fig.figure_file:
            return None
        path = (self.figures_dir / fig.figure_file).resolve()
        # Never follow a name outside this document's figure folder.
        if path.parent != self.figures_dir.resolve() or not path.exists():
            return None
        return path

    def stats(self) -> Dict[str, Any]:
        counts: Dict[str, int] = {}
        for n in self.nodes:
            counts[n.type] = counts.get(n.type, 0) + 1
        text_depths = [n.depth for n in self.nodes if n.type == "text"]
        return {
            "pages": self.page_count,
            "text": counts.get("text", 0),
            "table": counts.get("table", 0),
            "row_pin": counts.get("row", 0) + counts.get("pin", 0),
            "equation": counts.get("equation", 0),
            "figure": len(self.figures),
            "total": len(self.nodes),
            "avg_depth": round(sum(text_depths) / max(len(text_depths), 1), 2),
        }


@dataclass
class RetrievedItem:
    """One retrieval hit: the node, its hybrid score and the engine that found it."""

    score: float
    node: Node
    engine: str  # text | table | row | equation | figure
    node_index: int = -1


@dataclass
class SourceRef:
    """A cited source, as shown to the user and to the LLM."""

    ref_num: int
    label: str  # e.g. "[REF-1: TABLE p5]"
    tag: str  # TEXT | TABLE | ROW | EQUATION | FIGURE
    page: Any
    section: str
    parent_ctx: str
    snippet: str  # preview for the UI (raw_content[:MAX_PREVIEW])
    full_text: str  # what the LLM saw for this reference
    type: str
    meta: Dict[str, Any]
    score: Optional[float]
    depth: int
    chunk_idx: int
    doc_name: str = ""
    item_id: str = ""  # EQ_3 / TBL_1 / FIG_2 when the source is a specific element
    agent: str = ""
    figure_file: str = ""  # figures: file name inside the document's figure folder
    doc_id: str = ""


@dataclass
class Evidence:
    """Structured evidence returned by every specialist agent."""

    key: str  # unique: doc_id + node/page/element identity
    doc_id: str
    doc_name: str
    type: str  # text | table | row | pin | equation | figure | metadata | page
    page: int
    section: str
    content: str  # compact text given to the master agent
    excerpt: str  # what the user sees
    agent: str  # which specialist found it
    item_id: str = ""
    score: float = 0.0  # final ranking score
    confidence: float = 0.0  # 0..1, relative within this answer
    parent_ctx: str = ""
    meta: Dict[str, Any] = field(default_factory=dict)
    node_index: int = -1
    depth: int = 0
    chunk_idx: int = 0
    figure_file: str = ""
    ranks: Dict[str, int] = field(default_factory=dict)  # rank per retriever (hybrid, bm25, ...)


@dataclass
class AgentStep:
    """One ReAct step, shown in the "How the agent worked" panel."""

    kind: str  # "thought" | "action" | "observation" | "answer" | "note"
    text: str
    tool: str = ""
    args: Dict[str, Any] = field(default_factory=dict)


@dataclass
class QueryResult:
    kind: str  # answer | not_found | feedback_override
    answer: str
    sources: List[SourceRef] = field(default_factory=list)
    metrics: Dict[str, Any] = field(default_factory=dict)
    latency: float = 0.0
    warnings: List[str] = field(default_factory=list)
    used_llm: bool = False
    mode: str = "agent"
    steps: List[AgentStep] = field(default_factory=list)
    llm_calls: int = 0
    tokens: int = 0
    # --- RAG∞ Pro additions (empty in legacy mode) ---------------------------
    provider: str = ""
    model: str = ""
    stage_latency: Dict[str, float] = field(default_factory=dict)  # ms per stage
    plan: Dict[str, Any] = field(default_factory=dict)
    verification: Dict[str, Any] = field(default_factory=dict)
    agent_stats: Dict[str, Dict[str, float]] = field(default_factory=dict)
    conflicts: List[str] = field(default_factory=list)
    regenerated: bool = False
    cached: bool = False
    question: str = ""
