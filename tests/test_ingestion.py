"""Phases 3-4: document ingestion (figures, tables, equations, metadata)."""
import pymupdf

from rag.pdf_parser import parse_pdf
from tests.conftest import _png


def test_doc_metadata(indexed):
    index, _ = indexed
    assert index.doc_meta["title"] == "LM7805X VOLTAGE REGULATOR"
    assert ["POWER DISSIPATION", 3] in index.doc_meta["sections"]
    assert index.doc_meta["scanned_pages"] == []


def test_tables_have_captions(indexed):
    index, _ = indexed
    caps = [n.caption for n in index.nodes if n.type == "table"]
    assert caps == ["ELECTRICAL CHARACTERISTICS", "PIN CONFIGURATION"]


def test_each_image_gets_its_own_caption(tmp_path, make_settings):
    doc = pymupdf.open(); p = doc.new_page()
    p.insert_image(pymupdf.Rect(50, 50, 250, 170), stream=_png(200, 120))
    p.insert_text((50, 185), "Figure 1. Dropout voltage vs output current", fontsize=9)
    p.insert_image(pymupdf.Rect(50, 300, 250, 420), stream=_png(201, 121))
    p.insert_text((50, 435), "Figure 2. Ripple rejection vs frequency", fontsize=9)
    parsed = parse_pdf(doc.tobytes(), make_settings(), tmp_path / "figs")
    caps = [f.caption for f in parsed.figures]
    assert caps == ["Figure 1. Dropout voltage vs output current", "Figure 2. Ripple rejection vs frequency"]
    assert all("bbox" in f.meta for f in parsed.figures)


def test_scanned_page_is_detected(tmp_path, make_settings):
    doc = pymupdf.open(); p = doc.new_page()
    p.insert_image(pymupdf.Rect(0, 0, 595, 842), stream=_png(300, 400))
    parsed = parse_pdf(doc.tobytes(), make_settings(), tmp_path / "figs")
    assert parsed.doc_meta["scanned_pages"] == [1]
