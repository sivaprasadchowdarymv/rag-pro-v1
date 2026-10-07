"""
Structural + recursive text chunking with parent-context injection and overlap.

Old RAG.py equivalents (behaviour preserved):
  * `_SPLIT_PATTERNS`, `_split_text()`, `recursive_chunk()`
  * `_HEADING_RE`, `detect_sections()`
  * the "TEXT lane" + breadcrumb logic that lived inside `build_graph()`

Only change: chunk size / overlap are function parameters instead of globals
that the sidebar mutated, so different users can't affect each other.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

from config.settings import AVG_CHARS_PER_TOK
from rag.metadata import extract_metadata
from rag.models import Node

# Ordered from coarsest to finest split boundary.
_SPLIT_PATTERNS: List[Tuple[str, "re.Pattern[str]"]] = [
    ("heading_caps", re.compile(r"\n(?=[A-Z][A-Z0-9 \-\/\.]{4,}\n)")),  # ALL-CAPS
    ("heading_title", re.compile(r"\n(?=[A-Z][a-z]+(?:\s[A-Z][a-z]+){1,6}\n)")),
    ("heading_num", re.compile(r"\n(?=\d+[\.\)]\s+[A-Z])")),  # "1. Section"
    ("paragraph", re.compile(r"\n{2,}")),  # blank lines
    ("newline", re.compile(r"\n")),  # single newline
    ("sentence", re.compile(r"(?<=[.!?])\s+")),  # sentence boundary
]

_HEADING_RE = re.compile(
    r"^("
    r"[A-Z][A-Z0-9 \-\(\)\/\.]{4,}|"  # ALL CAPS
    r"\d+[\.\)]\s+[A-Z].{3,60}|"  # 1. Title
    r"[A-Z][a-z]+(?:\s[A-Z][a-z]+){0,7}"  # Title Case
    r")$"
)


def tok(text: str) -> int:
    """Fast token-count approximation (1 token ~= 4 chars)."""
    return max(1, len(text) // AVG_CHARS_PER_TOK)


def _split_text(text: str, level: int, max_tokens: int) -> List[str]:
    """Split `text` using the pattern at `level`; return non-empty parts."""
    if level >= len(_SPLIT_PATTERNS):
        size = max_tokens * AVG_CHARS_PER_TOK  # hard split at char boundary
        return [text[i : i + size] for i in range(0, len(text), size)]
    _, pattern = _SPLIT_PATTERNS[level]
    return [p for p in pattern.split(text) if p.strip()]


def recursive_chunk(
    text: str,
    section_title: str,
    parent_ctx: str,
    page: int,
    max_tokens: int,
    overlap_tokens: int,
    depth: int = 0,
) -> List[Node]:
    """
    Recursively split `text` until every chunk is <= max_tokens.

    Parent context (ancestor headings) is prepended to the embedded content so
    the embedding captures both the local text AND its location in the
    document. Overlap is added between adjacent sibling chunks.
    """
    if tok(text) <= max_tokens:
        prefix = f"[{parent_ctx}]\n" if parent_ctx else ""
        return [
            Node(
                type="text",
                section=section_title,
                content=f"{prefix}{text.strip()}",
                raw_content=text.strip(),
                page=page,
                depth=depth,
                parent_ctx=parent_ctx,
                meta=extract_metadata(text),
            )
        ]

    parts = _split_text(text, depth, max_tokens)
    if len(parts) <= 1:  # can't split at this level -> go one level finer
        return recursive_chunk(
            text, section_title, parent_ctx, page, max_tokens, overlap_tokens, depth + 1
        )

    # Overlap: each part gets the tail of the previous part prepended.
    overlap_chars = overlap_tokens * AVG_CHARS_PER_TOK
    with_overlap: List[str] = []
    for i, part in enumerate(parts):
        if i == 0 or overlap_chars <= 0:
            with_overlap.append(part)
        else:
            with_overlap.append(parts[i - 1][-overlap_chars:] + "\n" + part)

    nodes: List[Node] = []
    for idx, part in enumerate(with_overlap):
        children = recursive_chunk(
            part, section_title, parent_ctx, page, max_tokens, overlap_tokens, depth + 1
        )
        for child in children:
            child.chunk_idx = idx
        nodes.extend(children)
    return nodes


def detect_sections(text: str) -> List[Dict[str, str]]:
    """
    First-pass section splitter: returns [{"title", "content"}, ...].
    Content may still be large; recursive_chunk() handles oversized sections.
    """
    sections: List[Dict[str, str]] = []
    current = {"title": "General", "content": ""}
    for line in (text or "").split("\n"):
        stripped = line.strip()
        if 4 < len(stripped) < 100 and _HEADING_RE.match(stripped):
            if current["content"].strip():
                sections.append(current)
            current = {"title": stripped, "content": ""}
        else:
            current["content"] += line + "\n"
    if current["content"].strip():
        sections.append(current)
    return sections


@dataclass
class PageChunks:
    nodes: List[Node] = field(default_factory=list)
    sections: List[Dict[str, str]] = field(default_factory=list)  # non-empty only
    headings: List[str] = field(default_factory=list)  # in reading order

    @property
    def last_heading(self) -> str:
        return self.headings[-1] if self.headings else ""


def chunk_page_text(
    text: str, page: int, max_tokens: int, overlap_tokens: int
) -> PageChunks:
    """TEXT lane for one page: sections -> recursive chunks with a breadcrumb."""
    result = PageChunks()
    for sec in detect_sections(text):
        if not sec["content"].strip():
            continue
        result.sections.append(sec)
        result.headings.append(sec["title"])
        breadcrumb = " > ".join(result.headings[-2:])  # last 2 headings
        result.nodes.extend(
            recursive_chunk(
                text=sec["content"],
                section_title=sec["title"],
                parent_ctx=breadcrumb,
                page=page,
                max_tokens=max_tokens,
                overlap_tokens=overlap_tokens,
            )
        )
    return result
