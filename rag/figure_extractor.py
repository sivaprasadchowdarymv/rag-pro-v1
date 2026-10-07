"""
Figure lane: extract embedded images so LLaVA can look at them on demand.

Old RAG.py equivalent: `extract_figures()`. Caption detection is unchanged.
Fixes / optimisations:
  * Images are saved in a per-document folder. RAG.py wrote every document's
    images to `.rag_cache/fig_p{page}_{idx}` so a second PDF overwrote the
    first PDF's figures while its cached index still pointed at them.
  * An image reused on many pages (e.g. a logo) is written to disk once.
  * Tiny images (< MIN_FIGURE_PX on either side: bullets, icons) are skipped.
    Set MIN_FIGURE_PX=0 to keep everything, like RAG.py.
  * JPX / JBIG2 / CMYK images are converted to PNG, because Ollama vision
    models only accept PNG/JPEG.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, List

import pymupdf

from config.settings import get_logger
from rag.models import Node

log = get_logger("figures")

_CAPTION_RE = re.compile(
    r"\b(fig|figure|graph|chart|plot|diagram|waveform)\b", re.IGNORECASE
)
_VISION_READY_EXT = {"png", "jpeg", "jpg"}


def _find_caption(page) -> str:
    try:
        blocks = page.get_text("blocks") or []
    except Exception:
        return ""
    for block in blocks:
        text = "" if block[4] is None else str(block[4])
        if _CAPTION_RE.search(text):
            return text.strip()[:200]
    return ""


def _blocks(page):
    try:
        return [(pymupdf.Rect(b[:4]), str(b[4] or "").strip()) for b in page.get_text("blocks") or [] if b[6] == 0]
    except Exception:
        return []


def _caption_and_context(page, rect) -> tuple:
    """Caption = caption-like text block closest to the image (below preferred);
    context = other text blocks within ~150 pt of the image."""
    if rect is None:
        return _find_caption(page), ""
    best, best_dist = "", 1e9
    context = []
    for brect, text in _blocks(page):
        if not text:
            continue
        below = brect.y0 - rect.y1
        above = rect.y0 - brect.y1
        dist = below if below >= -5 else (above + 20 if above >= -5 else 1e9)
        if _CAPTION_RE.search(text) and 0 <= dist < best_dist and dist < 120:
            best, best_dist = text, dist
        elif min(abs(below), abs(above)) < 150:
            context.append(text)
    caption = best or _find_caption(page)
    ctx = " ".join(t for t in context if t != caption)
    return caption[:200], ctx[:500]


def _save_image(doc, xref: int, figures_dir: Path) -> str:
    """Write image `xref` to disk (once) and return its file name."""
    info = doc.extract_image(xref)
    ext = (info.get("ext") or "png").lower()
    data = info["image"]
    if ext not in _VISION_READY_EXT:
        pix = pymupdf.Pixmap(doc, xref)
        if pix.n - pix.alpha >= 4:  # CMYK -> RGB
            pix = pymupdf.Pixmap(pymupdf.csRGB, pix)
        data, ext = pix.tobytes("png"), "png"
    name = f"img_{xref}.{ext}"
    path = figures_dir / name
    if not path.exists():
        path.write_bytes(data)
    return name


def extract_figures(
    doc,
    page,
    page_num: int,
    figures_dir: Path,
    min_px: int,
    written: Dict[int, str],
) -> List[Node]:
    """`written` maps xref -> file name and is shared across pages of one PDF."""
    nodes: List[Node] = []

    for idx, img in enumerate(page.get_images(full=True) or []):
        xref, width, height = img[0], img[2], img[3]
        if min_px and (width < min_px or height < min_px):
            continue
        try:
            if xref not in written:
                written[xref] = _save_image(doc, xref, figures_dir)
        except Exception as exc:
            log.debug("Skipping image xref=%s on page %s: %s", xref, page_num, exc)
            continue

        try:
            rects = page.get_image_rects(xref)
        except Exception:
            rects = []
        rect = rects[0] if rects else None
        caption, context = _caption_and_context(page, rect)
        meta = {"context": context}
        if rect is not None:
            meta["bbox"] = [round(v, 1) for v in rect]
        nodes.append(
            Node(
                type="figure",
                section="Figure",
                content=caption or f"Figure {idx + 1} on page {page_num}",
                raw_content=caption or "",
                page=page_num,
                chunk_idx=idx,
                figure_file=written[xref],
                caption=caption,
                meta=meta,
            )
        )
    return nodes
