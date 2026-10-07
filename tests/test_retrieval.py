"""Phase 5-8: BM25, reranking, specialists, orchestrator, evidence fusion."""
from agents.orchestrator import plan
from agents.retrieval_stage import run_retrieval
from agents.specialists import QueryContext, analyse_equation, equation_agent, figure_agent, table_agent
from rag.models import Evidence
from retrieval.bm25 import BM25Index
from retrieval.evidence_fusion import detect_conflicts, fuse


def test_bm25_prefers_exact_symbol():
    idx = BM25Index(["thermal resistance RthJA junction to ambient", "output voltage tolerance", "junction"])
    s = idx.scores("RthJA", [0, 1, 2])
    assert s[0] > 0 and s[1] == 0 and s[2] == 0


def test_router_matches_brief_example():
    p = plan("Explain the controller shown in Figure 5 and derive the steering equation.")
    assert {"figure", "equation"} <= set(p.intents)
    assert {"figure_agent", "equation_agent"} <= set(p.agents) and p.requires_master_reasoning
    assert "document_agent" in p.skipped_agents  # irrelevant agents are not invoked


def test_simple_lookup_skips_tools_and_irrelevant_agents():
    p = plan("What is the maximum output current?")
    assert not p.use_tools and "figure_agent" not in p.agents and "equation_agent" not in p.agents


def test_follow_up_is_rewritten():
    p = plan("and at 85 C?", previous_question="What is the quiescent current?")
    assert p.is_follow_up and p.retrieval_query.startswith("What is the quiescent current?")


def test_standalone_short_question_is_not_a_follow_up():
    assert not plan("What is the peak output current?", previous_question="How do I calculate PD?").is_follow_up
    assert plan("Is it short-circuit protected?", previous_question="Tell me about the LM7805").is_follow_up


def test_equation_analysis():
    info = analyse_equation("PD = (VIN - VOUT) x IOUT", ["PD = (VIN - VOUT) x IOUT", "where PD is the power dissipation in watts"])
    assert info["is_formula"] and info["variables"] == ["PD", "VIN", "VOUT", "IOUT"] and "watts" in info["units"]
    assert not analyse_equation("Output voltage VOUT = 5.0V", [])["is_formula"]
    assert not analyse_equation("Conditions: VIN = 10V, IO = 500mA, TJ = 25°C", [])["is_formula"]
    assert analyse_equation("Junction temperature TJ = TA + PD x RthJA", [])["is_formula"]


def test_equation_agent_ranks_formulas(indexed):
    index, s = indexed
    ev = equation_agent(index, QueryContext("power dissipation equation PD", s))
    top = sorted(ev, key=lambda e: -sum(1 / (60 + r) for r in e.ranks.values()))[0]
    assert top.item_id.startswith("EQ_") and "PD = (VIN - VOUT) x IOUT" in top.content and "Variables: PD" in top.content


def test_table_agent_ids_and_captions(indexed):
    index, s = indexed
    ev = table_agent(index, QueryContext("pin configuration GND", s))
    assert any(e.item_id == "TBL_2" and "PIN CONFIGURATION" in e.content for e in ev)


def test_figure_agent_by_number(indexed):
    index, s = indexed
    ev = figure_agent(index, QueryContext("explain figure 2", s))
    assert ev and ev[0].item_id == "FIG_2" and "block diagram" in ev[0].content.lower() and ev[0].figure_file


def _ev(key, excerpt, section="S", page=1, ranks=None):
    return Evidence(key=key, doc_id="d", doc_name="x.pdf", type="text", page=page, section=section,
                    content=excerpt, excerpt=excerpt, agent="text_agent", ranks=ranks or {"hybrid": 1})


def test_fusion_dedupes_and_respects_budget():
    a = _ev("a", "Parameter | Min | Typ | Max\nPeak output current | | 2.2 | A", ranks={"hybrid": 1})
    row = _ev("b", "Peak output current | | 2.2 | A", ranks={"bm25": 1})  # contained in a
    c = _ev("c", "unrelated text " * 50, ranks={"hybrid": 2})
    out = fuse([[a, c], [row, a]], max_items=5, max_chars=200)
    assert [e.key for e in out.evidence] == ["a"] and out.dropped_duplicates == 1


def test_conflicts_only_within_same_section():
    e1, e2 = _ev("1", "VIN = 10V", "Conditions"), _ev("2", "VIN = 35V", "Absolute Maximum Ratings")
    assert detect_conflicts([e1, e2]) == []
    e3 = _ev("3", "VIN = 12V", "Conditions", page=3)
    assert detect_conflicts([e1, e3])


def test_full_retrieval_stage(indexed):
    index, s = indexed
    out = run_retrieval(plan("What is the peak output current?"), [index], s)
    assert out.reranked and any("2.2" in e.excerpt for e in out.fusion.evidence[:3])
    assert {"retrieval", "rerank", "fusion"} <= set(out.latency_ms) and out.agent_stats["table_agent"]["calls"] == 1
