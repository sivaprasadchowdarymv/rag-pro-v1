"""
Master RAG Agent: final reasoning and synthesis over fused evidence.

ReAct with tools when the plan asks for it (calculations, comparisons,
equations), otherwise a single call. Inputs: question, controlled
conversation memory, the user's preference profile, fused evidence and
conflict notes. Only ACTIONS are recorded for the UI; the model's hidden
reasoning is never requested, stored or shown.
"""
from __future__ import annotations

from ops.guard import UNTRUSTED_NOTE, sanitize_evidence
import json
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence

from agents.retrieval_stage import run_agents
from config.settings import MAX_PREVIEW, Settings, get_logger
from llm.model_router import ModelRouter
from llm.providers import LLMError
from rag.agent import _ANSWER_FORMAT, NOT_FOUND
from rag.models import AgentStep, DocumentIndex, Evidence, SourceRef
from rag.tools import TOOL_SCHEMAS, safe_calculate

log = get_logger("master")
StepFn = Optional[Callable[[AgentStep], None]]
_TYPE_LABEL = {"metadata": "DOC", "pin": "PIN", "row": "ROW"}
_SOURCE_AGENTS = {"any": ("text_agent", "table_agent", "equation_agent"), "text": ("text_agent",),
                  "table": ("table_agent",), "row": ("table_agent",), "equation": ("equation_agent",),
                  "figure": ("figure_agent",)}


class EvidenceRegistry:
    """Gives every evidence item a stable [REF-n: TYPE pX] label."""

    def __init__(self) -> None:
        self.injection_lines = 0
        self.refs: List[SourceRef] = []
        self.evidence: Dict[int, Evidence] = {}
        self._by_key: Dict[str, SourceRef] = {}

    def add(self, ev: Evidence) -> SourceRef:
        if ev.key in self._by_key:
            return self._by_key[ev.key]
        n = len(self.refs) + 1
        tag = _TYPE_LABEL.get(ev.type, ev.type.upper())
        ref = SourceRef(ref_num=n, label=f"[REF-{n}: {tag} p{ev.page}]", tag=tag, page=ev.page,
                        section=ev.section, parent_ctx=ev.parent_ctx, snippet=ev.excerpt[:MAX_PREVIEW],
                        full_text=ev.content, type=ev.type, meta=ev.meta or {}, score=round(ev.confidence, 2),
                        depth=ev.depth, chunk_idx=ev.chunk_idx, doc_name=ev.doc_name, item_id=ev.item_id,
                        agent=ev.agent, figure_file=ev.figure_file, doc_id=ev.doc_id)
        self.refs.append(ref)
        self.evidence[n] = ev
        self._by_key[ev.key] = ref
        return ref

    def by_number(self, n: int) -> Optional[SourceRef]:
        return self.refs[n - 1] if 0 < n <= len(self.refs) else None

    def render(self, items: Sequence[Evidence], budget: int) -> str:
        out, used = [], 0
        for ev in items:
            ref = self.add(ev)
            extra = ", ".join(x for x in (ev.doc_name, ev.item_id) if x)
            content, hits = sanitize_evidence(ev.content.strip())
            self.injection_lines += hits
            block = f"{ref.label} ({extra}) section: {ev.section}\n{content}"
            if out and used + len(block) > budget:
                break
            out.append(block[: max(200, budget - used)])
            used += len(block)
        return "\n\n".join(out)


class ProToolBox:
    """search_datasheet / read_page / calculate over the active documents."""

    def __init__(self, indexes: Sequence[DocumentIndex], settings: Settings, registry: EvidenceRegistry):
        self.indexes, self.settings, self.registry = list(indexes), settings, registry
        self.observations: List[str] = []

    def execute(self, name: str, arguments: str):
        try:
            args = json.loads(arguments or "{}")
            assert isinstance(args, dict)
        except (ValueError, AssertionError):
            return {}, "Error: arguments were not valid JSON."
        if name == "search_datasheet":
            agents = _SOURCE_AGENTS.get(str(args.get("source", "any")), _SOURCE_AGENTS["any"])
            groups, _, _ = run_agents(agents, str(args.get("query", ""))[:300], self.indexes, self.settings)
            found = sorted((e for g in groups for e in g), key=lambda e: -sum(1 / (60 + r) for r in e.ranks.values()))
            obs = self.registry.render(found[:4], self.settings.observation_chars) or "No matching excerpts."
        elif name == "read_page":
            obs = self._read_page(args.get("page"), str(args.get("document", "")))
        elif name == "calculate":
            try:
                obs = f"{args.get('expression')} = {safe_calculate(str(args.get('expression', ''))):.6g}"
            except Exception as exc:
                obs = f"Error: {exc}"
        else:
            obs = f"Error: unknown tool '{name}'."
        self.observations.append(obs)
        return args, obs

    def _read_page(self, page: Any, document: str) -> str:
        try:
            page = int(page)
        except (TypeError, ValueError):
            return "Error: page must be a number."
        index = next((i for i in self.indexes if document and document.lower() in i.filename.lower()), self.indexes[0])
        if not 1 <= page <= index.page_count:
            return f"Error: page must be between 1 and {index.page_count}."
        seen, parts = set(), []
        for n in index.nodes:
            body = n.raw_content.strip()
            if n.page == page and n.type in ("text", "table") and body and body not in seen:
                seen.add(body)
                parts.append(body)
        text = "\n".join(parts)[: int(self.settings.observation_chars * 1.5)]
        ev = Evidence(key=f"{index.doc_id}#page{page}", doc_id=index.doc_id, doc_name=index.filename, type="page",
                      page=page, section=f"Page {page}", content=text, excerpt=text, agent="read_page")
        return self.registry.render([ev], len(text) + 200) if text else f"Page {page} has no extractable text."


@dataclass
class MasterOutcome:
    answer: str
    messages: List[Dict[str, Any]]
    steps: List[AgentStep] = field(default_factory=list)
    llm_calls: int = 0
    tokens: int = 0
    provider: str = ""
    model: str = ""
    llm_ms: float = 0.0


def _system_prompt(use_tools: bool, preferences: str) -> str:
    tools = ("You work in ReAct style: decide what evidence you still need, call a tool, read the observation, "
             "then answer. Tools: search_datasheet(query, source) to search again with other keywords or symbols; "
             "read_page(page) when an excerpt is cut off; calculate(expression) for ALL arithmetic. "
             "If the evidence already answers the question, answer immediately.\n\n") if use_tools else ""
    prefs = f"\n\nUser preferences (follow them unless they conflict with accuracy):\n{preferences}" if preferences else ""
    return ("You are an expert electronics engineer answering questions about component datasheets and "
            "technical documents, using ONLY the numbered evidence provided.\n\n" + tools + _ANSWER_FORMAT
            + "\n- If evidence comes from several documents, say which document each fact comes from."
            + "\n- If the evidence is insufficient for part of the question, say so explicitly."
            + "\n- " + UNTRUSTED_NOTE + prefs)


def run_master(question: str, evidence_text: str, use_tools: bool, router: ModelRouter, settings: Settings,
               toolbox: ProToolBox, memory: str = "", preferences: str = "", conflicts: Sequence[str] = (),
               on_step: StepFn = None) -> MasterOutcome:
    t0 = time.perf_counter()
    user = ""
    if memory:
        user += f"Conversation so far (for context only):\n{memory}\n\n"
    user += f"Question: {question}\n\nEvidence:\n\n{evidence_text or '(No matching evidence was found.)'}"
    if conflicts:
        user += "\n\nNotes on the evidence:\n" + "\n".join(f"- {c}" for c in conflicts)
    messages: List[Dict[str, Any]] = [{"role": "system", "content": _system_prompt(use_tools, preferences)},
                                      {"role": "user", "content": user}]
    out = MasterOutcome(answer="", messages=messages)

    def add(step: AgentStep) -> None:
        out.steps.append(step)
        if on_step:
            try:
                on_step(step)
            except Exception:
                pass

    rounds = settings.agent_max_steps if use_tools else 0
    for step in range(rounds + 1):
        must_answer = step == rounds
        tools = TOOL_SCHEMAS if use_tools else None
        try:
            routed = router.chat("master", messages, tools=tools, tool_choice="none" if must_answer else "auto")
        except LLMError as exc:
            if exc.kind != "tool_use_failed":
                raise
            add(AgentStep("note", "Invalid tool call; asked for a direct answer."))
            routed = router.chat("master", messages + [{"role": "user", "content": "Answer now using the evidence you have."}])
        res = routed.result
        out.llm_calls += 1
        out.tokens += res.tokens
        out.provider, out.model = res.provider, res.model
        if res.tool_calls and not must_answer:
            messages.append({"role": "assistant", "content": res.content or "", "tool_calls": [
                {"id": c.id, "type": "function", "function": {"name": c.name, "arguments": c.arguments}}
                for c in res.tool_calls]})
            for c in res.tool_calls[:3]:
                args, obs = toolbox.execute(c.name, c.arguments)
                add(AgentStep("action", "", tool=c.name, args=args))
                add(AgentStep("observation", obs[:600], tool=c.name))
                messages.append({"role": "tool", "tool_call_id": c.id, "content": obs})
            for c in res.tool_calls[3:]:
                messages.append({"role": "tool", "tool_call_id": c.id, "content": "Skipped: too many tool calls."})
            continue
        out.answer = res.content
        messages.append({"role": "assistant", "content": res.content})
        break
    out.llm_ms = (time.perf_counter() - t0) * 1000
    add(AgentStep("answer", f"Master synthesis ({out.provider}: {out.model})"))
    return out


def regenerate(outcome: MasterOutcome, feedback: str, router: ModelRouter) -> MasterOutcome:
    """One bounded retry after verification failed (no tools, explicit feedback)."""
    t0 = time.perf_counter()
    messages = outcome.messages + [{"role": "user", "content": (
        f"VERIFICATION FEEDBACK: {feedback}\nRewrite the complete answer using ONLY the evidence above. "
        f"Remove or correct every unsupported statement and cite every fact. If the evidence does not "
        f"contain the answer, reply exactly: {NOT_FOUND}")}]
    routed = router.chat("master", messages)
    res = routed.result
    return MasterOutcome(res.content, messages + [{"role": "assistant", "content": res.content}], outcome.steps,
                         outcome.llm_calls + 1, outcome.tokens + res.tokens, res.provider, res.model,
                         outcome.llm_ms + (time.perf_counter() - t0) * 1000)
