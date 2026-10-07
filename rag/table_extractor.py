"""
Table lane: PyMuPDF find_tables() -> one full-table node + one row/pin node
per data row (header always injected so every row is self-contained).

Old RAG.py equivalents: `extract_tables()`, `table_to_nodes()` — unchanged,
except that failures are now logged instead of silently swallowed.
"""
from __future__ import annotations

from typing import Any, Dict, List

from config.settings import get_logger
from rag.models import Node

log = get_logger("tables")

_PIN_KEYWORDS = ("pin", "port", "signal", "gpio")


def _table_caption(page, table) -> str:
    """Nearest text block above the table (e.g. 'Table 3. Electrical characteristics')."""
    try:
        top = table.bbox[1]
        best, best_gap = "", 80.0
        for b in page.get_text("blocks") or []:
            text = str(b[4] or "").strip()
            gap = top - b[3]
            if text and 0 <= gap < best_gap:
                best, best_gap = text, gap
        return best.split("\n")[0][:150]
    except Exception:
        return ""


def extract_tables(page) -> List[Dict[str, Any]]:
    tables: List[Dict[str, Any]] = []
    try:
        found = page.find_tables().tables
    except Exception as exc:  # find_tables can fail on malformed pages
        log.debug("find_tables failed on page %s: %s", page.number + 1, exc)
        return tables

    for table in found:
        try:
            df = table.to_pandas().fillna("").astype(str)
        except Exception as exc:
            log.debug("Could not convert a table on page %s: %s", page.number + 1, exc)
            continue
        caption = _table_caption(page, table)
        headers = [str(h) for h in df.columns]
        header_line = " | ".join(headers)
        rows = [" | ".join(r) for r in df.values.tolist()]
        tables.append(
            {
                "headers": headers,
                "content": header_line + "\n" + "\n".join(rows),
                "rows": rows,
                "header_line": header_line,
                "caption": caption,
            }
        )
    return tables


def table_to_nodes(tbl: Dict[str, Any], page_num: int, section_title: str) -> List[Node]:
    nodes: List[Node] = []
    header = tbl["header_line"]
    content = tbl["content"]

    nodes.append(  # full-table node (semantic table search)
        Node(
            type="table",
            section=section_title,
            content=content,
            raw_content=content,
            page=page_num,
            caption=tbl.get("caption") or None,
        )
    )

    row_type = "pin" if any(k in content.lower() for k in _PIN_KEYWORDS) else "row"
    for idx, row in enumerate(tbl["rows"]):
        if not row.strip():
            continue
        raw = f"{header}\n{row}"
        nodes.append(
            Node(
                type=row_type,
                section=section_title,
                content=raw,
                raw_content=raw,
                page=page_num,
                chunk_idx=idx,
            )
        )
    return nodes
