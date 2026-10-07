"""
Tools the ReAct agent can call, plus the registry that numbers sources.

    search_datasheet(query, source)  hybrid retrieval (semantic + fuzzy + bonuses)
    read_page(page)                  full text + tables of one page
    calculate(expression)            safe arithmetic (no eval / no code execution)

Every piece of evidence gets a stable label such as [REF-3: TABLE p5], the
same citation format as the original app, so citations stay consistent
across agent steps and the UI can show the exact excerpt.
"""
from __future__ import annotations

import ast
import json
import math
import operator
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from config.settings import MAX_PREVIEW, Settings, get_logger
from rag.embeddings import EmbeddingError, embed_query
from rag.models import DocumentIndex, Node, RetrievedItem, SourceRef
from rag.retrieval import gather_hits, run_engines

log = get_logger("tools")


# ---------------------------------------------------------------------------
# Source registry
# ---------------------------------------------------------------------------
class SourceRegistry:
    """Numbers evidence once; the same node always keeps the same REF number."""

    def __init__(self, settings: Settings):
        self.settings = settings
        self.refs: List[SourceRef] = []
        self._by_key: Dict[Any, SourceRef] = {}

    def _add(self, key, tag: str, page, section: str, parent_ctx: str, text: str,
             node: Optional[Node], score: Optional[float]) -> SourceRef:
        if key in self._by_key:
            return self._by_key[key]
        n = len(self.refs) + 1
        ref = SourceRef(
            ref_num=n, label=f"[REF-{n}: {tag} p{page}]", tag=tag, page=page,
            section=section, parent_ctx=parent_ctx,
            snippet=(node.raw_content if node else text)[:MAX_PREVIEW],
            full_text=text, type=node.type if node else tag.lower(),
            meta=(node.meta if node else {}) or {}, score=score,
            depth=node.depth if node else 0, chunk_idx=node.chunk_idx if node else 0,
        )
        self.refs.append(ref)
        self._by_key[key] = ref
        return ref

    def add_hit(self, item: RetrievedItem) -> SourceRef:
        node = item.node
        text = node.content[: self.settings.max_chars_per_chunk]
        return self._add(("node", item.node_index), item.engine.upper(), node.page,
                         node.section, node.parent_ctx, text, node, round(item.score, 1))

    def add_page(self, page: int, text: str) -> SourceRef:
        return self._add(("page", page), "PAGE", page, f"Page {page}", "", text, None, None)

    def by_number(self, n: int) -> Optional[SourceRef]:
        return self.refs[n - 1] if 0 < n <= len(self.refs) else None


def format_hits(registry: SourceRegistry, items: List[RetrievedItem], budget: int,
                max_items: int) -> Tuple[str, int]:
    """Register hits and render them as an observation within a character budget."""
    lines: List[str] = []
    used = 0
    for item in items[:max_items]:
        ref = registry.add_hit(item)
        header = f"{ref.label} section: {ref.section}"
        body = ref.full_text.strip()
        room = budget - used - len(header) - 2
        if room < 120:
            break
        body = body[:room]
        lines.append(f"{header}\n{body}")
        used += len(header) + len(body) + 2
    return "\n\n".join(lines), len(lines)


# ---------------------------------------------------------------------------
# Safe calculator
# ---------------------------------------------------------------------------
_BIN_OPS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
            ast.Div: operator.truediv, ast.Pow: operator.pow, ast.Mod: operator.mod}
_UNARY = {ast.UAdd: operator.pos, ast.USub: operator.neg}
_FUNCS: Dict[str, Callable] = {
    "sqrt": math.sqrt, "log": math.log, "log10": math.log10, "log2": math.log2,
    "exp": math.exp, "sin": math.sin, "cos": math.cos, "tan": math.tan,
    "atan": math.atan, "abs": abs, "min": min, "max": max, "round": round,
}
_CONSTS = {"pi": math.pi, "e": math.e}


def safe_calculate(expression: str) -> float:
    expr = (expression or "").replace("^", "**").replace("×", "*").replace("÷", "/").strip()
    if not expr or len(expr) > 200:
        raise ValueError("expression is empty or too long")
    tree = ast.parse(expr, mode="eval")

    def ev(node):
        if isinstance(node, ast.Expression):
            return ev(node.body)
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return node.value
        if isinstance(node, ast.BinOp) and type(node.op) in _BIN_OPS:
            left, right = ev(node.left), ev(node.right)
            if isinstance(node.op, ast.Pow) and abs(right) > 100:
                raise ValueError("exponent too large")
            return _BIN_OPS[type(node.op)](left, right)
        if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY:
            return _UNARY[type(node.op)](ev(node.operand))
        if isinstance(node, ast.Name) and node.id in _CONSTS:
            return _CONSTS[node.id]
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                and node.func.id in _FUNCS and not node.keywords):
            return _FUNCS[node.func.id](*[ev(a) for a in node.args])
        raise ValueError("only numbers, + - * / ** %, parentheses and math functions are allowed")

    return float(ev(tree))


# ---------------------------------------------------------------------------
# Tool schemas (OpenAI/Groq function-calling format)
# ---------------------------------------------------------------------------
TOOL_SCHEMAS: List[Dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "search_datasheet",
            "description": (
                "Search the uploaded datasheet. Use specific engineering keywords and symbols "
                "(e.g. 'VOUT output voltage', 'quiescent current IQ', 'pin 3 function'). "
                "Returns numbered excerpts to cite."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "What to look for."},
                    "source": {
                        "type": "string",
                        "enum": ["any", "text", "table", "row", "equation"],
                        "description": "Restrict to a source type: 'row' = individual table rows/pins.",
                    },
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_page",
            "description": "Read the full text and tables of one page when excerpts are incomplete.",
            "parameters": {
                "type": "object",
                "properties": {"page": {"type": "integer", "description": "1-based page number."}},
                "required": ["page"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "calculate",
            "description": (
                "Evaluate arithmetic exactly, e.g. '(12-5)*1.5' or 'sqrt(2)*3.3'. "
                "Use plain numbers (convert mA to A etc. yourself)."
            ),
            "parameters": {
                "type": "object",
                "properties": {"expression": {"type": "string"}},
                "required": ["expression"],
            },
        },
    },
]

_SOURCE_TO_ENGINES = {
    "any": ("text", "table", "row", "equation"),
    "text": ("text",), "table": ("table",), "row": ("row",), "equation": ("equation",),
}


class ToolBox:
    """Executes tool calls against one document index."""

    def __init__(self, index: DocumentIndex, settings: Settings, registry: SourceRegistry):
        self.index, self.settings, self.registry = index, settings, registry
        self._qvec_cache: Dict[str, Optional[np.ndarray]] = {}
        self.embedding_warning = ""

    def query_vector(self, query: str) -> Optional[np.ndarray]:
        if not self.index.has_embedding.any():
            return None
        if query not in self._qvec_cache:
            try:
                self._qvec_cache[query] = embed_query(query, self.settings, self.index.embed_model)
            except EmbeddingError as exc:
                self.embedding_warning = exc.user_message
                self._qvec_cache[query] = None
        return self._qvec_cache[query]

    # --- tools -------------------------------------------------------------
    def search_datasheet(self, query: str, source: str = "any") -> str:
        query = (query or "").strip()[:300]
        if not query:
            return "Error: empty query."
        engines = _SOURCE_TO_ENGINES.get(source, _SOURCE_TO_ENGINES["any"])
        results = run_engines(query, self.index, max(self.settings.top_k, 4), self.query_vector(query))
        results = {k: v for k, v in results.items() if k in engines}
        hits = gather_hits(results)
        weak = False
        if not hits:  # nothing above threshold: offer the best two, flagged as weak
            pool = sorted((i for v in results.values() for i in v), key=lambda r: r.score, reverse=True)
            hits, weak = pool[:2], True
        if not hits:
            return "No matching excerpts. Try different keywords or read_page."
        text, count = format_hits(self.registry, hits, self.settings.observation_chars, 4)
        prefix = "Weak matches only (may be irrelevant):\n\n" if weak else ""
        return prefix + text if count else "No matching excerpts."

    def read_page(self, page: Any) -> str:
        try:
            page = int(page)
        except (TypeError, ValueError):
            return "Error: page must be a number."
        if not 1 <= page <= self.index.page_count:
            return f"Error: page must be between 1 and {self.index.page_count}."
        seen, parts = set(), []
        for node in self.index.nodes:
            if node.page != page or node.type not in ("text", "table"):
                continue
            body = node.raw_content.strip()
            if body and body not in seen:
                seen.add(body)
                parts.append(body)
        text = "\n".join(parts)[: int(self.settings.observation_chars * 1.5)]
        if not text:
            return f"Page {page} has no extractable text."
        ref = self.registry.add_page(page, text)
        return f"{ref.label}\n{text}"

    def calculate(self, expression: str) -> str:
        try:
            value = safe_calculate(expression)
        except Exception as exc:
            return f"Error: {exc}"
        return f"{expression} = {value:.6g}"

    def execute(self, name: str, arguments: str) -> Tuple[Dict[str, Any], str]:
        try:
            args = json.loads(arguments or "{}")
            if not isinstance(args, dict):
                raise ValueError
        except ValueError:
            return {}, "Error: arguments were not valid JSON."
        if name == "search_datasheet":
            return args, self.search_datasheet(str(args.get("query", "")), str(args.get("source", "any")))
        if name == "read_page":
            return args, self.read_page(args.get("page"))
        if name == "calculate":
            return args, self.calculate(str(args.get("expression", "")))
        return args, f"Error: unknown tool '{name}'."


def initial_evidence(query: str, toolbox: ToolBox) -> Tuple[str, Dict[str, List[RetrievedItem]], int]:
    """Hybrid retrieval for the question itself (the agent's first observation)."""
    settings = toolbox.settings
    engine_results = run_engines(query, toolbox.index, settings.top_k, toolbox.query_vector(query))
    hits = gather_hits(engine_results)
    text, count = format_hits(toolbox.registry, hits, settings.max_context_chars, settings.max_context_chunks)
    return text, engine_results, count

