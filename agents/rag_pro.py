"""
RAG∞ Pro pipeline: one question → verified, cited answer.

    orchestrator (plan) → specialists → rerank → fusion → master → verifier
    (→ one bounded regeneration if verification fails) → metrics

Every stage is timed. The trace records stages and tool ACTIONS only.
"""
from __future__ import annotations

import re
import time
from typing import Callable, Dict, List, Optional, Sequence

from agents.master import EvidenceRegistry, ProToolBox, regenerate, run_master
from agents.orchestrator import plan as make_plan
from agents.retrieval_stage import run_retrieval
from agents.verifier import canonicalize, check, feedback_text, llm_check
from config.settings import Settings, get_logger
from llm.model_router import ModelRouter
from llm.providers import LLMError
from metrics import rag_metrics
from ops import answer_cache
from rag.agent import NOT_FOUND
from rag.models import AgentStep, DocumentIndex, QueryResult
from rag.pipeline import normalize_answer
from retrieval.reranker import rerank
from storage.cache import DocumentPaths, VisionCache

log = get_logger("rag_pro")
StepFn = Optional[Callable[[AgentStep], None]]
_AGENT_NAMES = {"text_agent": "Text retrieval", "table_agent": "Table retrieval", "equation_agent": "Equation retrieval",
                "figure_agent": "Figure retrieval", "document_agent": "Document metadata"}


def _vision_describer(router: ModelRouter, settings: Settings):
    if not router.has_role("vision"):
        return None

    def describe(fig, path) -> str:
        cache = VisionCache(DocumentPaths(settings.data_dir, path.parent.parent.name).vision_file)
        cached = cache.get(fig.figure_file or "")
        if cached:
            return cached
        prompt = ("Describe this figure from a technical document precisely: axes, units, curves, blocks and "
                  "connections, key values. No speculation.")
        try:
            text = router.chat("vision", [{"role": "user", "content": prompt}], images=[path.read_bytes()],
                               max_tokens=500).result.content
        except LLMError as exc:
            log.info("Vision skipped: %s", exc.kind)
            return ""
        if text:
            cache.set(fig.figure_file or "", text)
        return text

    return describe


def _concept_chooser(router: ModelRouter, settings: Settings):
    """Small-LLM resolver for ambiguous abbreviations only (off by default; results cached in the graph)."""
    if not settings.concept_llm_resolve or not router.has_role("fast"):
        return None

    def choose(context: str, senses: List[str]):
        prompt = ("Which meaning does the abbreviation have in this text? Reply with the number only.\n"
                  + "\n".join(f"{k + 1}. {s}" for k, s in enumerate(senses)) + f"\n\nText: {context}")
        try:
            res = router.chat("fast", [{"role": "user", "content": prompt}], max_tokens=5)
            m = re.search(r"\d+", res.result.content or "")
            return int(m.group()) - 1 if m else None
        except LLMError:
            return None
    return choose


def answer_pro(question: str, indexes: Sequence[DocumentIndex], settings: Settings, router: ModelRouter,
               memory: str = "", previous_question: Optional[str] = None, preferences: str = "",
               on_step: StepFn = None) -> QueryResult:
    t_start = time.perf_counter()
    question = (question or "").strip()
    steps: List[AgentStep] = []
    warnings: List[str] = []
    latency: Dict[str, float] = {}

    def stage(text: str) -> None:
        step = AgentStep("stage", text)
        steps.append(step)
        if on_step:
            try:
                on_step(step)
            except Exception:
                pass

    # 1. Plan
    plan = make_plan(question, previous_question, len(indexes), settings.agent_mode)
    latency["router"] = plan.latency_ms
    stage(f"Query classified: {', '.join(plan.intents)}" + (" (follow-up)" if plan.is_follow_up else ""))

    cache = answer_cache.answers(settings.answer_cache_size, settings.answer_cache_ttl)
    ckey = None
    if not plan.is_follow_up:
        ckey = answer_cache.answer_key(
            question, [i.doc_id for i in indexes], [getattr(i, "embed_model", "") for i in indexes],
            settings.agent_mode, settings.rerank, settings.max_context_chars, settings.llm_providers,
            settings.master_model, preferences)
        hit = cache.get(ckey)
        if hit is not None:
            hit.cached, hit.steps = True, [AgentStep("stage", "Answered from cache (verified earlier)")]
            hit.latency, hit.llm_calls, hit.tokens = round(time.perf_counter() - t_start, 3), 0, 0
            if on_step:
                on_step(hit.steps[0])
            return hit

    # 2-4. Specialists → rerank → fusion
    retrieval = run_retrieval(plan, indexes, settings, _vision_describer(router, settings),
                              _concept_chooser(router, settings))
    latency.update(retrieval.latency_ms)
    warnings += retrieval.warnings
    for name in plan.agents:
        st = retrieval.agent_stats.get(name, {})
        stage(f"{_AGENT_NAMES[name]}: {int(st.get('results', 0))} candidates"
              + (f" ({int(st['failures'])} failed)" if st.get("failures") else ""))
    cl = retrieval.agent_stats.get("concept_linker")
    if cl and cl.get("concepts"):
        stage(f"Concept linker: {int(cl['concepts'])} concept(s) resolved, {int(cl['results'])} linked item(s) added")
    if retrieval.reranked:
        stage("Evidence reranked")
    fused = retrieval.fusion
    stage(f"Evidence fused: {len(fused.evidence)} items kept, {fused.dropped_duplicates} duplicates removed")

    base = dict(question=question, mode=settings.agent_mode, plan=plan.as_dict(), agent_stats=retrieval.agent_stats,
                conflicts=fused.conflicts)
    use_tools = plan.use_tools or (settings.agent_mode == "agent" and not fused.evidence)
    if not fused.evidence and not use_tools:
        latency["total"] = (time.perf_counter() - t_start) * 1000
        return QueryResult(kind="not_found", answer=NOT_FOUND, steps=steps, warnings=warnings,
                           stage_latency=latency, latency=round(latency["total"] / 1000, 2), **base)

    registry = EvidenceRegistry()
    evidence_text = registry.render(fused.evidence, settings.max_context_chars)
    toolbox = ProToolBox(indexes, settings, registry)

    # 5. Master synthesis
    answer, calls, tokens, provider, model, regenerated = "", 0, 0, "", "", False
    outcome = None
    try:
        outcome = run_master(question, evidence_text, use_tools, router, settings, toolbox, memory, preferences,
                             fused.conflicts, on_step=lambda s: (steps.append(s), on_step and on_step(s)))
        answer, calls, tokens = outcome.answer, outcome.llm_calls, outcome.tokens
        provider, model = outcome.provider, outcome.model
        latency["master"] = outcome.llm_ms
    except LLMError as exc:
        warnings.append(f"Showing the best matching evidence without an AI answer.\n\n{exc.user_message}")

    if not answer:
        answer = "\n\n".join(f"{r.label}\n{r.full_text[:300]}" for r in registry.refs[:3]) or NOT_FOUND
    answer = normalize_answer(answer)

    # 6. Verification (+ one bounded regeneration)
    def support(sentence: str, texts: List[str]) -> List[float]:
        scores, _, _ = rerank(sentence, texts, settings)
        if scores is None:
            raise RuntimeError("support scoring unavailable")
        return scores

    def verify(text: str):
        text, cited, bad = canonicalize(text, registry.by_number)
        ref_texts = {r.ref_num: r.full_text for r in registry.refs}
        all_text = "\n".join(ref_texts.values())
        rep = check(text, ref_texts, toolbox.observations, question, bad,
                    support_fn=support if settings.rerank else None, support_threshold=settings.support_threshold)
        if settings.verify_with_llm and outcome is not None and rep.grounded and router.has_role("verifier"):
            rep = llm_check(rep, text, all_text, router)
        return text, cited, rep

    answer, cited, report = verify(answer)
    latency["verifier"] = report.latency_ms
    if outcome is not None and report.needs_regeneration and settings.max_regenerations > 0:
        stage("Verification failed: regenerating once")
        try:
            outcome = regenerate(outcome, feedback_text(report), router)
            candidate, c_cited, c_report = verify(normalize_answer(outcome.answer))
            latency["master"], latency["verifier"] = outcome.llm_ms, latency["verifier"] + c_report.latency_ms
            calls, tokens = outcome.llm_calls, outcome.tokens
            answer, cited, report, regenerated = candidate, c_cited, c_report, True
        except LLMError as exc:
            warnings.append(f"Regeneration skipped: {exc.user_message}")
    if outcome is not None:
        stage({"verified": "Verification passed", "partial": "Verification passed (some citations weakly supported)"}
              .get(report.verdict, "Verification: issues found (see warnings)"))
    if report.invalid_citations:
        warnings.append(f"The answer cites {', '.join(report.invalid_citations)}, which are not retrieved sources.")
    if report.miscited and not report.grounded:
        warnings.append("Some values are cited to the wrong source: " + "; ".join(report.miscited[:3]))
    if report.weak_support and report.grounded:
        warnings.append("Some cited sources only weakly support their sentence. Check: "
                        + " | ".join(report.weak_support[:2]))
    if report.unsupported_claims and not report.grounded:
        warnings.append("Some values could not be found in the evidence. Check them against the sources: "
                        + " | ".join(report.unsupported_claims[:3]))
    if outcome is not None and not cited and NOT_FOUND not in answer.upper():
        warnings.append("The answer has no valid citations. Verify it against the sources below.")

    not_found = NOT_FOUND in answer.upper() and len(answer) < 80
    sources = [] if not_found else (cited or list(registry.refs))
    latency["total"] = (time.perf_counter() - t_start) * 1000
    metrics = {} if not_found else rag_metrics.compute(question, answer, sources, report.as_dict(), settings,
                                                        {k: v for k, v in latency.items() if k != "total"},
                                                        tokens, calls)
    if registry.injection_lines:
        warnings.append(f"{registry.injection_lines} instruction-like line(s) in the documents were treated as data "
                        "(prompt-injection guard).")
    result = QueryResult(
        kind="not_found" if not_found else "answer", answer=answer, sources=sources, metrics=metrics,
        latency=round(latency["total"] / 1000, 2), warnings=list(dict.fromkeys(warnings)),
        used_llm=outcome is not None and bool(outcome.answer), steps=steps, llm_calls=calls, tokens=tokens,
        provider=provider, model=model, stage_latency={k: round(v, 1) for k, v in latency.items()},
        verification=report.as_dict(), regenerated=regenerated, **base)
    if (ckey and result.kind == "answer" and result.used_llm and report.verdict in ("verified", "partial")
            and not report.invalid_citations):
        cache.set(ckey, result)
    return result
