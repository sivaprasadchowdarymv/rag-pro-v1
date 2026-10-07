"""
Equation lane: regex scan for equation / specification lines.

Old RAG.py equivalents: `_EQ_RE`, `is_equation_line()` and the "EQUATION
lane" inside `build_graph()`. Same regex, same per-page de-duplication, same
"[section]\\nexpression" content prefix.

One fix: RAG.py labelled *every* equation on a page with the *last* heading
on that page. Here each equation gets the heading it actually sits under
(using the same heading detector as the text lane), which gives the LLM and
the source cards the correct section.
"""
from __future__ import annotations

import re
from typing import List

from rag.chunking import _HEADING_RE
from rag.models import Node

_EQ_RE = re.compile(
    r"(?:"
    r"[A-Za-z_]\w*\s*=\s*[\w\.\+\-\*/\(\)\^µ]+|"  # A = expr
    r"[A-Za-z_]\w*\s*[<>≤≥]\s*[\d\.]+\s*\w*|"  # V < 5V
    r"(?:min|max|typ)\s*[\=\:]\s*[\d\.]+|"  # typ = 3.3
    r"[\d\.]+\s*(?:V|A|Ω|W|Hz|s|F|H|mA|µA|kHz|MHz)\b"
    r")",
    re.IGNORECASE,
)


def is_equation_line(line: str) -> bool:
    s = line.strip()
    return 4 < len(s) < 200 and bool(_EQ_RE.search(s))


def extract_equations(page_text: str, page_num: int) -> List[Node]:
    nodes: List[Node] = []
    seen: set = set()
    current_section = "General"
    for line in (page_text or "").split("\n"):
        stripped = line.strip()
        if 4 < len(stripped) < 100 and _HEADING_RE.match(stripped):
            current_section = stripped
        if not is_equation_line(line) or stripped in seen:
            continue
        seen.add(stripped)
        nodes.append(
            Node(
                type="equation",
                section=current_section,
                content=f"[{current_section}]\n{stripped}",
                raw_content=stripped,
                page=page_num,
            )
        )
    return nodes
