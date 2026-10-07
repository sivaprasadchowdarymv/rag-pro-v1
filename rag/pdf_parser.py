"""
PDF -> nodes. Opens the PDF once and runs the four extraction lanes per page:

    PDF ─ PyMuPDF ─┬─ TEXT     chunking.chunk_page_text()    (sections, recursion,
                   │                                          overlap, breadcrumb)
                   ├─ TABLE    table_extractor               (table + row/pin nodes)
                   ├─ EQUATION equation_extractor            (atomic equation nodes)
                   └─ FIGURE   figure_extractor              (images for LLaVA)

Old RAG.py equivalent: the extraction half of `build_graph()`. Differences:
  * Upload validation (size, PDF signature, encryption, page limit) with
    user-friendly errors — never a stack trace.
  * No Streamlit calls in here; progress is reported through a callback,
    so this module can be used and tested without the UI.
  * `import pymupdf` (the `fitz` alias is deprecated in current PyMuPDF).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional

import pymupdf

from config.settings import Settings, get_logger
from rag.chunking import chunk_page_text
from rag.equation_extractor import extract_equations
from rag.figure_extractor import extract_figures
from rag.models import Node
from rag.table_extractor import extract_tables, table_to_nodes

log = get_logger("pdf")

ProgressFn = Optional[Callable[[float, str], None]]


class PdfProcessingError(Exception):
    """`user_message` is safe to show in the UI."""

    def __init__(self, user_message: str):
        super().__init__(user_message)
        self.user_message = user_message


INVALID_PDF = "Unable to process this PDF.\nPlease verify that the file is a valid PDF."


def validate_pdf_bytes(pdf_bytes: bytes, settings: Settings) -> None:
    """Cheap checks before PyMuPDF touches the file."""
    size_mb = len(pdf_bytes) / (1024 * 1024)
    if not pdf_bytes:
        raise PdfProcessingError("The uploaded file is empty.")
    if size_mb > settings.max_upload_mb:
        raise PdfProcessingError(
            f"This file is {size_mb:.1f} MB. The limit is {settings.max_upload_mb} MB."
        )
    if b"%PDF-" not in pdf_bytes[:1024]:
        raise PdfProcessingError(INVALID_PDF)


@dataclass
class ParsedPdf:
    nodes: List[Node]
    figures: List[Node]
    page_count: int
    doc_meta: Dict = field(default_factory=dict)


def parse_pdf(
    pdf_bytes: bytes, settings: Settings, figures_dir: Path, progress: ProgressFn = None
) -> ParsedPdf:
    validate_pdf_bytes(pdf_bytes, settings)
    try:
        doc = pymupdf.open(stream=pdf_bytes, filetype="pdf")
    except Exception as exc:
        log.warning("PyMuPDF could not open the upload: %s", exc)
        raise PdfProcessingError(INVALID_PDF) from exc

    try:
        if doc.needs_pass:
            raise PdfProcessingError(
                "This PDF is password-protected. Please upload an unprotected copy."
            )
        total = doc.page_count
        if total == 0:
            raise PdfProcessingError(INVALID_PDF)
        if total > settings.max_pages:
            raise PdfProcessingError(
                f"This PDF has {total} pages. The limit is {settings.max_pages} pages."
            )
        log.info("PDF loaded: %d pages", total)
        figures_dir.mkdir(parents=True, exist_ok=True)

        nodes: List[Node] = []
        figures: List[Node] = []
        written_images: Dict[int, str] = {}
        sections: List[List] = []
        scanned: List[int] = []
        first_heading = ""
        pdf_meta = dict(doc.metadata or {})

        for i, page in enumerate(doc):
            page_num = i + 1
            if progress:
                progress(i / total, f"Reading page {page_num} of {total}")
            text = page.get_text() or ""

            # 1. TEXT lane
            page_chunks = chunk_page_text(
                text, page_num, settings.max_chunk_tokens, settings.overlap_tokens
            )
            nodes.extend(page_chunks.nodes)
            for heading in page_chunks.headings:
                if heading != "General" and len(sections) < 300:
                    sections.append([heading, page_num])
            if page_num == 1:  # title = first meaningful line of page 1
                first_heading = next((ln.strip() for ln in text.split("\n") if 3 < len(ln.strip()) < 120), "")
            if len(text.strip()) < 25 and page.get_images():
                scanned.append(page_num)  # image-only page: needs OCR (not available)

            # 2. TABLE lane (nearest heading on the page, as in RAG.py)
            table_section = page_chunks.last_heading or "Table"
            for tbl in extract_tables(page):
                nodes.extend(table_to_nodes(tbl, page_num, table_section))

            # 3. EQUATION lane
            nodes.extend(extract_equations(text, page_num))

            # 4. FIGURE lane
            figures.extend(
                extract_figures(
                    doc, page, page_num, figures_dir, settings.min_figure_px, written_images
                )
            )
    except PdfProcessingError:
        raise
    except Exception as exc:
        log.exception("PDF processing failed")
        raise PdfProcessingError(INVALID_PDF) from exc
    finally:
        doc.close()

    counts: Dict[str, int] = {}
    for n in nodes:
        counts[n.type] = counts.get(n.type, 0) + 1
    log.info("Extracted %d pages", total)
    log.info("Created %d text chunks", counts.get("text", 0))
    log.info(
        "Extracted %d tables (%d row/pin nodes)",
        counts.get("table", 0),
        counts.get("row", 0) + counts.get("pin", 0),
    )
    log.info("Extracted %d equations", counts.get("equation", 0))
    log.info("Extracted %d figures", len(figures))
    if scanned:
        log.info("Scanned/image-only pages (no OCR available): %s", scanned[:20])
    doc_meta = {
        "title": (pdf_meta.get("title") or "").strip() or first_heading,
        "author": (pdf_meta.get("author") or "").strip(),
        "subject": (pdf_meta.get("subject") or "").strip(),
        "sections": sections,
        "scanned_pages": scanned,
    }
    return ParsedPdf(nodes=nodes, figures=figures, page_count=total, doc_meta=doc_meta)
