"""Cross-modal concept linking: modalities, aliases, homonyms, references, ambiguity, OCR, versions, security."""
import pytest

from concepts.expand import expand, wanted_modalities
from concepts.golden import golden_index
from concepts.graph import Mention, build_graph, clear_cache, get_graph, query_concepts
from rag.models import Node


@pytest.fixture(autouse=True)
def _fresh():
    clear_cache()
    yield
    clear_cache()


@pytest.fixture
def gi(tmp_path):
    ix = golden_index(tmp_path)
    return ix, build_graph(ix)


def cid(ix, suffix):
    return f"{ix.doc_id[:10]}:{suffix}"


def refs(g, concept_id, relation=None):
    return {m.ref for m in g.mentions if m.concept_id == concept_id and not m.ambiguous
            and (relation is None or m.relation == relation)}


def keys(added):
    return {e.key.split("#")[1] for e in added}


def test_text_table_same_concept(gi):
    ix, g = gi
    vout = cid(ix, "VOUT~output-voltage")
    assert {"n6", "n4"} <= refs(g, vout)
    assert "n4" in refs(g, vout, "QUANTIFIES") and "n6" in refs(g, vout, "DEFINES")


def test_text_equation(gi):
    ix, g = gi
    assert {"n0", "n1"} <= refs(g, cid(ix, "PD~power-dissipation"), "DEFINES")


def test_text_image(gi):
    ix, g = gi
    pd = cid(ix, "PD~power-dissipation")
    assert "f0" in refs(g, pd, "VISUALIZES") and "n0" in refs(g, pd)


def test_all_modalities(gi):
    ix, g = gi
    mods = {m.modality for m in g.mentions if m.concept_id == cid(ix, "PD~power-dissipation")}
    assert {"text", "equation", "row", "figure"} <= mods


def test_aliases_and_acronyms(gi):
    ix, g = gi
    c = g.concepts[cid(ix, "PD~power-dissipation")]
    assert "PD" in c.aliases and "power dissipation" in c.aliases
    assert any(r[1] == "ALIAS_OF" and r[0].startswith(c.concept_id) for r in g.relations)
    assert c.concept_id in query_concepts("explain the power dissipation", g)


def test_same_word_different_meaning(gi):
    ix, g = gi
    power, photo = cid(ix, "PD~power-dissipation"), cid(ix, "PD~photodiode")
    assert power != photo
    assert "n7" in refs(g, photo) and "n7" not in refs(g, power) and "n0" not in refs(g, photo)
    assert "n13" not in refs(g, cid(ix, "Re~reynolds-number"))  # "Re-check" is a word, not the symbol


def test_cross_page_references(gi):
    ix, g = gi
    fig4, tbl2 = cid(ix, "figure-4"), cid(ix, "table-2")
    assert "n0" in refs(g, fig4, "REFERENCES") and "f0" in refs(g, fig4, "SAME_CONCEPT")   # p1 -> p3
    assert "n11" in refs(g, tbl2, "REFERENCES") and "n3" in refs(g, tbl2, "SAME_CONCEPT")  # p4 -> p2


def test_ambiguous_concept_not_expanded_and_llm_resolver(tmp_path, gi):
    ix, g = gi
    amb = [m for m in g.mentions if m.ref == "n13" and m.surface == "PD"]
    assert amb and all(m.ambiguous and m.confidence < 0.5 for m in amb)
    assert not any(c.startswith(cid(ix, "PD")) for c in query_concepts("What is the PD value in the table?", g))
    calls = []
    g2 = build_graph(ix, chooser=lambda ctx, senses: calls.append(senses) or senses.index("photodiode"))
    m = [m for m in g2.mentions if m.ref == "n13" and m.surface == "PD"][0]
    assert calls and not m.ambiguous and m.concept_id == cid(ix, "PD~photodiode")


def test_ocr_errors(tmp_path):
    ix = golden_index(tmp_path)
    ix.nodes.append(Node(type="text", section="Electrical", content="x", raw_content="Keep V0UT within limits.", page=2))
    g = build_graph(ix)
    assert "n11" in refs(g, cid(ix, "Re~reynolds-number"))         # "Reyno1ds number"
    assert f"n{len(ix.nodes) - 1}" in refs(g, cid(ix, "VOUT~output-voltage"))  # "V0UT"


def test_multiple_versions_stay_separate(tmp_path):
    v1 = golden_index(tmp_path, doc_id="1" * 64, vout_max="5.25", filename="LDO.pdf")
    v2 = golden_index(tmp_path, doc_id="2" * 64, vout_max="5.30", filename="LDO.pdf")
    g1, g2 = get_graph(v1), get_graph(v2)
    assert not ({c for c in g1.concepts} & {c for c in g2.concepts})
    added, _ = expand("What is the maximum output voltage VOUT?", [], [v2], set())
    assert added and all(e.doc_id == v2.doc_id and "5.30" in e.excerpt for e in added if e.type == "row")


def test_unauthorised_linked_content_is_never_returned(tmp_path):
    mine = golden_index(tmp_path, doc_id="m" * 64)
    other = golden_index(tmp_path, doc_id="o" * 64)
    get_graph(other)                      # other tenant's graph is in the process cache
    g = get_graph(mine)
    leaked = Mention(cid(mine, "VOUT~output-voltage"), other.doc_id, "n4", "row", 2, "Electrical", "VOUT",
                     "QUANTIFIES", 1.0)
    g.mentions.insert(0, leaked)          # even a corrupted/poisoned link must not cross documents
    added, stats = expand("What is the maximum output voltage VOUT?", [], [mine], set())
    assert added and all(e.doc_id == mine.doc_id for e in added) and stats["skipped_unauthorised"] >= 1


def test_selective_modalities(gi):
    ix, _ = gi
    assert wanted_modalities("What is the Reynolds number formula?") >= {"equation", "definition"}
    assert "figure" not in wanted_modalities("What is the Reynolds number formula?")
    assert {"figure", "table", "text"} <= wanted_modalities("Compare the values shown in Figure 4 and Table 2.")
    formula, _ = expand("What is the Reynolds number formula?", [], [ix], {f"{ix.doc_id}#n9"})
    assert keys(formula) == {"n10"}
    compare, _ = expand("Compare the values shown in Figure 4 and Table 2.", [], [ix], set())
    assert {"f0", "n3"} <= keys(compare) and "n7" not in keys(compare)
    assert expand("Hello there", [], [ix], set())[0] == []  # no cue -> no expansion


def test_representations_preserved(gi):
    ix, g = gi
    eq = next(m for m in g.mentions if m.ref == "n1")
    assert eq.repr["normalized"] == "PD=(VIN-VOUT)*IOUT" and "V_{OUT}" in eq.repr["latex"]
    row = next(m for m in g.mentions if m.ref == "n4")
    assert row.repr["headers"][:2] == ["Parameter", "Symbol"] and row.repr["cells"][1] == "VOUT"
    assert row.repr["units"] == ["V"] and row.repr["caption"].startswith("Table 2")
    fig = next(m for m in g.mentions if m.ref == "f0")
    assert fig.repr["caption"].startswith("Figure 4") and "25 °C" in fig.repr["context"]
    assert {"ocr", "visual_description", "image_embedding"} <= set(fig.repr)
    assert all(m.page and m.section and m.doc_id == ix.doc_id for m in g.mentions)


def test_graph_cached_on_disk(gi):
    ix, _ = gi
    g = get_graph(ix)
    path = ix.figures_dir.parent / "concepts.json"
    assert path.exists()
    clear_cache()
    assert get_graph(ix).to_json() == g.to_json()


def test_evaluation_thresholds():
    from metrics.concept_eval import evaluate
    r = evaluate()
    assert r["entity_linking_precision"] >= 0.9 and r["entity_linking_recall"] >= 0.9
    assert r["relation_accuracy"] >= 0.9 and r["cross_modal_retrieval_recall"] >= 0.9
    assert r["false_link_rate"] <= 0.05 and r["llm_calls_for_linking"] == 0


def test_real_pipeline_integration(indexed):
    """Held-out: the synthetic PDF used by the other tests (not tuned for linking)."""
    from agents.orchestrator import plan
    from agents.retrieval_stage import run_retrieval
    index, s = indexed
    g = get_graph(index)
    assert g.concepts  # builds on a parsed PDF
    out = run_retrieval(plan("How do I calculate the power dissipation?"), [index], s)
    assert "concept_linker" in out.agent_stats and out.agent_stats["concept_linker"]["failures"] == 0
    assert any(e.type == "equation" for e in out.fusion.evidence)
    from dataclasses import replace
    off = run_retrieval(plan("How do I calculate the power dissipation?"), [index], replace(s, concept_linking=False))
    assert "concept_linker" not in off.agent_stats
