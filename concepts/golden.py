"""Synthetic golden document for cross-modal linking evaluation (no PDF, no models needed)."""
from __future__ import annotations

from pathlib import Path

from rag.models import DocumentIndex, Node


def _n(type_, section, text, page, caption=None, meta=None):
    return Node(type=type_, section=section, content=f"[{section}] {text}", raw_content=text, page=page,
                caption=caption, meta=meta or {})


def golden_index(tmp: Path, doc_id: str = "a" * 64, vout_max: str = "5.25", filename: str = "LDO_v1.pdf",
                 ocr_noise: bool = True) -> DocumentIndex:
    nodes = [
        _n("text", "Thermal", "The device power dissipation (PD) must stay below the package limit. "
           "See Figure 4 for the derating curve.", 1),
        _n("equation", "Thermal", "PD = (VIN - VOUT) x IOUT", 1),
        _n("text", "Thermal", "where VIN is the input voltage and IOUT is the output current.", 1),
        _n("table", "Electrical", "Parameter | Symbol | Min | Typ | Max | Unit\nOutput voltage | VOUT | 4.75 | 5.0 | "
           f"{vout_max} | V", 2, caption="Table 2. Electrical characteristics"),
        _n("row", "Electrical", "Parameter | Symbol | Min | Typ | Max | Unit\nOutput voltage | VOUT | 4.75 | 5.0 | "
           f"{vout_max} | V", 2),
        _n("row", "Electrical", "Parameter | Symbol | Min | Typ | Max | Unit\nPower dissipation | PD | - | - | 2.0 | W", 2),
        _n("text", "Electrical", "The output voltage (VOUT) is regulated over the full load range.", 2),
        _n("text", "Sensor", "The photodiode (PD) converts light into a small current.", 3),
        _n("text", "Sensor", "Connect the PD cathode to the bias pin.", 3),
        _n("text", "Fluid cooling", "The Reynolds number (Re) characterises the coolant flow regime.", 4),
        _n("equation", "Fluid cooling", "Re = (rho x v x L) / mu", 4),
        _n("text", "Fluid cooling", ("The Reyno1ds number above 4000 means turbulent flow; " if ocr_noise else
                                     "The Reynolds number above 4000 means turbulent flow; ")
           + "compare with the values in Table 2.", 4),
        _n("text", "Notes", "The output current is limited internally.", 5),
        _n("text", "Optics", "Re-check the lens alignment; the light PD reading should be stable.", 5),
    ]
    figures = [_n("figure", "Figure", "", 3, caption="Figure 4. Power dissipation (PD) derating versus temperature",
                  meta={"context": "PD falls linearly above 25 °C."})]
    figures[0].figure_file = "fig_p3_0.png"
    return DocumentIndex(doc_id=doc_id, filename=filename, page_count=5, nodes=nodes, figures=figures,
                         figures_dir=tmp / doc_id[:12] / "figures")


# Expected links: (concept suffix after "<doc>:", node ref, relation)
GOLD_LINKS = [
    ("PD~power-dissipation", "n0", "DEFINES"), ("PD~power-dissipation", "n1", "DEFINES"),
    ("PD~power-dissipation", "n5", "QUANTIFIES"), ("PD~power-dissipation", "f0", "VISUALIZES"),
    ("PD~photodiode", "n7", "DEFINES"), ("PD~photodiode", "n8", "REFERENCES"),
    ("VOUT~output-voltage", "n1", "RELATED_TO"), ("VOUT~output-voltage", "n4", "QUANTIFIES"),
    ("VOUT~output-voltage", "n6", "DEFINES"),
    ("VOUT~output-voltage", "n3", "QUANTIFIES"), ("VIN~input-voltage", "n1", "RELATED_TO"),
    ("IOUT~output-current", "n1", "RELATED_TO"), ("VIN~input-voltage", "n2", "DEFINES"), ("IOUT~output-current", "n2", "DEFINES"),
    ("Re~reynolds-number", "n9", "DEFINES"), ("Re~reynolds-number", "n10", "DEFINES"),
    ("Re~reynolds-number", "n11", "REFERENCES"), ("IOUT~output-current", "n12", "REFERENCES"),
    ("figure-4", "f0", "SAME_CONCEPT"), ("figure-4", "n0", "REFERENCES"),
    ("table-2", "n3", "SAME_CONCEPT"), ("table-2", "n11", "REFERENCES"),
]
# Mentions that must NOT be linked: (concept suffix, ref)
FORBIDDEN_LINKS = [("PD~photodiode", "n0"), ("PD~photodiode", "n1"), ("PD~power-dissipation", "n7"),
                   ("PD~power-dissipation", "n8"), ("Re~reynolds-number", "n13"),
                   ("PD~power-dissipation", "n13")]  # "Re-check" is a word, not Re; PD here is unresolved

# Cross-modal retrieval cases: query, initial evidence keys, expected linked refs, refs that must not be added
GOLD_QUERIES = [
    ("What is the Reynolds number formula?", {"n9"}, {"n10"}, {"n11", "f0", "n3"}),
    ("How do I calculate the power dissipation PD?", set(), {"n1", "n0"}, {"n7", "n8"}),
    ("Compare the values shown in Figure 4 and Table 2.", set(), {"f0", "n3"}, {"n7"}),
    ("What is the maximum output voltage VOUT?", set(), {"n4"}, {"f0"}),
    ("What does the PD curve in the figure show for power dissipation?", set(), {"f0"}, {"n7"}),
]
