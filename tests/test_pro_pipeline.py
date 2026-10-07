"""Phases 9-10: master agent, verification agent, full RAG∞ Pro pipeline."""
from agents.rag_pro import answer_pro
from agents.verifier import canonicalize, check
from llm.model_router import ModelRouter
from tests.conftest import server_log


def _run(q, indexed, **kw):
    index, s = indexed
    return answer_pro(q, [index], s, ModelRouter(s), **kw)


def test_simple_question_one_call_verified(server, indexed):
    r = _run("What is the peak output current?", indexed)
    assert r.kind == "answer" and r.llm_calls == 1 and r.provider == "groq"
    assert r.verification["grounded"] and r.verification["citation_precision"] == 1.0
    assert all(src.label in r.answer for src in r.sources)
    assert {"router", "retrieval", "rerank", "fusion", "master", "verifier", "total"} <= set(r.stage_latency)
    assert server_log(server)["errors"] == []


def test_trace_has_actions_not_reasoning(server, indexed):
    r = _run("How do I calculate the power dissipation at 12 V input?", indexed)
    kinds = {s.kind for s in r.steps}
    assert "thought" not in kinds and "action" in kinds  # no chain-of-thought is stored
    assert any(s.tool == "calculate" for s in r.steps) and r.llm_calls == 2
    assert r.verification["numerical_score"] == 1.0  # 10.5 comes from the calculator: supported


def test_unsupported_number_triggers_one_regeneration(server, indexed):
    r = _run("unsupported number please: what is the output current", indexed)
    assert r.regenerated and r.verification["grounded"] and "9.87" not in r.answer


def test_invented_citation_flagged(server, indexed):
    index, s = indexed
    r = answer_pro("invent a citation", [index], s.__class__(**{**s.__dict__, "max_regenerations": 0}), ModelRouter(s))
    assert not r.verification["grounded"] and any("REF-99" in w for w in r.warnings)


def test_not_found(server, indexed):
    assert _run("nothing relevant here", indexed).kind == "not_found"


def test_follow_up_uses_memory(server, indexed):
    r = _run("and the typical value?", indexed, memory="Q: What is the quiescent current?\nA: 5 mA",
             previous_question="What is the quiescent current?")
    assert r.plan["intents"] and r.kind == "answer"


def test_no_provider_falls_back_to_evidence(indexed, make_settings, sample_pdf):
    index, _ = indexed
    s = make_settings(groq_api_key="")
    r = answer_pro("What is the peak output current?", [index], s, ModelRouter(s))
    assert not r.used_llm and r.sources and any("No AI provider" in w for w in r.warnings)


def test_multi_document(server, indexed, make_settings, tmp_path):
    from rag.pipeline import get_index
    import pymupdf
    index, s = indexed
    doc = pymupdf.open(); p = doc.new_page(); p.insert_text((50, 60), "LM317 ADJUSTABLE REGULATOR", fontsize=10)
    p.insert_text((50, 80), "Output current up to 1.5 A, dropout voltage VDO = 2.5V", fontsize=10)
    other, _ = get_index(doc.tobytes(), "LM317.pdf", s)
    r = answer_pro("Compare the dropout voltage of both documents", [index, other], s, ModelRouter(s))
    assert "multi_document" in r.plan["intents"] and r.kind == "answer"
    assert {"LM7805X.pdf", "LM317.pdf"} & {src.doc_name for src in r.sources}


def test_verifier_unit_scaling_and_equations():
    refs = {1: "IO = 500mA PD VIN IOUT"}
    rep = check("The current is 0.5 A [REF-1: TEXT p1].\n\n$$P_D = V_{IN} \\times I_{OUT}$$ [REF-1: TEXT p1]",
                refs, [], "q", [])
    assert rep.numerical_score == 1.0 and rep.equation_score == 1.0 and rep.grounded and rep.verdict == "verified"
    bad = check("It is 7.77 V [REF-1: TEXT p1].", {1: "VOUT = 5.0V"}, [], "q", [])
    assert not bad.grounded and bad.unsupported_claims and bad.verdict == "unverified"


def test_verifier_catches_wrong_citation():
    refs = {1: "Quiescent current typ 5 mA", 2: "Peak output current 2.2 A"}
    rep = check("The peak output current is 2.2 A [REF-1: ROW p2].", refs, [], "q", [])
    assert not rep.grounded and rep.miscited == ["2.2 is in REF-2, not REF-1"] and rep.citation_accuracy == 0.0
    ok = check("The peak output current is 2.2 A [REF-2: ROW p2].", refs, [], "q", [])
    assert ok.grounded and ok.citation_accuracy == 1.0


def test_calculated_sentences_skip_support_check():
    weak = lambda sent, texts: [-5.0 for _ in texts]
    rep = check("Using the calculator: P = (12-5) x 1.5 = 10.5 W [REF-1: EQUATION p3]",
                {1: "PD = (VIN - VOUT) x IOUT"}, ["(12-5)*1.5 = 10.5"], "power at 12 V", [], support_fn=weak)
    assert rep.verdict == "verified" and not rep.weak_support


def test_unit_scaling_only_with_units():
    # "500" without a unit must not match 0.5 by silent rescaling
    assert not check("The value is 500 [REF-1: TEXT p1].", {1: "limit 0.5"}, [], "q", []).grounded
    assert check("The value is 500 mA [REF-1: TEXT p1].", {1: "limit 0.5 A"}, [], "q", []).grounded


def test_weak_support_gives_partial_verdict():
    refs = {1: "Thermal shutdown protects the device 2.2"}
    score = lambda sent, texts: [-5.0 for _ in texts]  # scorer says: does not support
    rep = check("The peak output current is 2.2 A [REF-1: TEXT p1].", refs, [], "q", [], support_fn=score)
    assert rep.grounded and rep.verdict == "partial" and rep.weak_support


def test_canonicalize():
    class R:  # minimal ref
        def __init__(self, n): self.label = f"[REF-{n}: ROW p2]"
    refs = {1: R(1)}
    text, cited, bad = canonicalize("x [REF-1: TABLE p9] y [REF-7: TEXT p1]", refs.get)
    assert "[REF-1: ROW p2]" in text and len(cited) == 1 and bad == ["[REF-7: TEXT p1]"]
