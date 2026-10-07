"""
Reusable display widgets. Answers are rendered as Markdown + LaTeX (never raw
HTML from the model); every custom HTML snippet escapes its inputs.
"""
from __future__ import annotations

import html
import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import streamlit as st

from config.settings import Settings
from rag.models import AgentStep, QueryResult, SourceRef
from storage.cache import DocumentPaths

ICONS = {"TEXT": "📄", "PAGE": "📄", "TABLE": "📊", "ROW": "📊", "PIN": "📌", "EQUATION": "📐",
         "FIGURE": "🖼", "DOC": "🗂"}
_CITE = re.compile(r"\[REF-(\d+):\s*([A-Z]+)\s*p(\d+)\]")
_STAGE_ICON = {"stage": "✓", "action": "🔧", "observation": "👁", "note": "ℹ️", "answer": "✓"}


def esc(v) -> str:
    return html.escape("" if v is None else str(v))


def brand() -> None:
    st.markdown('<div class="brand"><div class="logo">∞</div><div><div class="brand-title">RAG∞ Pro</div>'
                '<div class="brand-sub">Verified answers from your documents</div></div></div>', unsafe_allow_html=True)


def hero(title: str, subtitle: str, steps: Optional[List[tuple]] = None) -> None:
    html_steps = ""
    if steps:
        html_steps = '<div class="steps">' + "".join(
            f'<div class="step"><span class="step-n">{i + 1}</span><span class="step-t">{esc(t)}</span>'
            f'<div class="step-d">{esc(d)}</div></div>' for i, (t, d) in enumerate(steps)) + "</div>"
    st.markdown(f'<div class="hero"><div class="hero-title">{esc(title)}</div>'
                f'<div class="hero-sub">{esc(subtitle)}</div>{html_steps}</div>', unsafe_allow_html=True)


def section(label: str) -> None:
    st.markdown(f'<div class="section-label">{esc(label)}</div>', unsafe_allow_html=True)


def status_line(ok: Optional[bool], label: str, value: str) -> str:
    cls = "ok" if ok else ("warn" if ok is None else "bad")
    return f'<div class="status-line"><span class="dot {cls}"></span>{esc(label)}: {esc(value)}</div>'


def format_answer_markdown(answer: str) -> str:
    return _CITE.sub(lambda m: f":blue-background[REF-{m.group(1)} · {m.group(2)} p{m.group(3)}]", answer or "")


def evidence_chip(src: SourceRef) -> str:
    what = f"{src.item_id} p{src.page}" if src.item_id else f"Page {src.page}"
    return f'<span class="chip">{ICONS.get(src.tag, "📄")} {esc(what)}</span>'


def figure_path(settings: Settings, src: SourceRef) -> Optional[Path]:
    if not (src.figure_file and src.doc_id):
        return None
    try:
        p = (DocumentPaths(settings.data_dir, src.doc_id).figures_dir / src.figure_file).resolve()
    except ValueError:
        return None
    return p if p.exists() else None





def verification_pill(result: QueryResult) -> str:
    v = result.verification or {}
    if not v or result.kind != "answer":
        return ""
    verdict = v.get("verdict") or ("verified" if v.get("grounded") else "unverified")
    txt, cls = {"verified": ("✓ Verified", "ok"), "partial": ("≈ Partly verified", "warn"),
                "unverified": ("⚠ Check sources", "bad")}.get(verdict, ("", ""))
    if not txt:
        return ""
    if result.regenerated:
        txt += " · corrected once"
    return f'<span class="badge {cls}">{esc(txt)}</span>'


def render_result(result: QueryResult, settings: Settings, show_evidence: bool = True) -> None:
    for w in dict.fromkeys(result.warnings):
        st.warning(w)
    st.markdown(format_answer_markdown(result.answer))
    if result.sources:
        st.markdown("".join(dict.fromkeys(evidence_chip(s) for s in result.sources)), unsafe_allow_html=True)
    meta = [f"{result.latency} s"]
    if result.provider:
        meta.append(f"{result.provider}: {result.model}")
    if result.llm_calls:
        meta.append(f"{result.llm_calls} LLM call{'s' if result.llm_calls != 1 else ''}")
    if result.tokens:
        meta.append(f"~{result.tokens:,} tokens")
    st.markdown(f'{verification_pill(result)}<span class="meta">{esc(" · ".join(meta))}</span>',
                unsafe_allow_html=True)
    if show_evidence and result.sources:
        with st.expander(f"Evidence ({len(result.sources)})"):
            render_sources(result.sources, settings)
    trace = [s for s in result.steps if s.kind in _STAGE_ICON]
    if trace:
        with st.expander("Agent activity"):
            render_trace(trace)


def render_sources(sources: Iterable[SourceRef], settings: Settings) -> None:
    for src in sources:
        head = f"{ICONS.get(src.tag, '📄')} **REF-{src.ref_num}** · {src.tag} · page {src.page}"
        if src.item_id:
            head += f" · {src.item_id}"
        st.markdown(head)
        details = [x for x in (src.doc_name, f"Section: {src.section}" if src.section else "",
                               f"Path: {src.parent_ctx}" if src.parent_ctx else "",
                               f"Found by: {src.agent.replace('_', ' ')}" if src.agent else "",
                               f"Confidence: {src.score:.2f}" if isinstance(src.score, (int, float)) else "") if x]
        st.markdown(f'<div class="src-meta">{"<br>".join(esc(d) for d in details)}</div>', unsafe_allow_html=True)
        fig = figure_path(settings, src)
        if fig is not None:
            st.image(str(fig), width=360)
        st.markdown(f'<div class="src-body">{esc(src.snippet)}</div>', unsafe_allow_html=True)
        st.divider()


def describe_step(step: AgentStep) -> str:
    if step.kind == "action":
        if step.tool == "search_datasheet":
            where = step.args.get("source", "any")
            return f'Search {"" if where == "any" else where + " "}for "{step.args.get("query", "")}"'
        if step.tool == "read_page":
            return f"Read page {step.args.get('page')}"
        if step.tool == "calculate":
            return f"Calculate {step.args.get('expression', '')}"
        return f"{step.tool}"
    if step.kind == "observation":
        return f"{step.tool} result: {step.text[:160]}"
    return step.text


def render_trace(steps: List[AgentStep]) -> None:
    """Actions and stages only; no model reasoning (chain-of-thought) is ever shown."""
    for s in steps:
        if s.kind == "thought":  # legacy results may contain these; never display them
            continue
        st.markdown(f'<div class="trace-line">{_STAGE_ICON.get(s.kind, "•")} {esc(describe_step(s))}</div>',
                    unsafe_allow_html=True)


def _color(pct: float) -> str:
    return "var(--ok)" if pct >= 70 else "var(--warn)" if pct >= 40 else "var(--bad)"


def metric_cards(values: Dict[str, Optional[float]]) -> None:
    """values: label -> 0..1 (shown as %), or None to skip."""
    cells = []
    for label, v in values.items():
        if v is None:
            continue
        pct = max(0, min(100, int(round(v * 100))))
        cells.append(f'<div class="metric"><div class="metric-name">{esc(label)}</div>'
                     f'<div class="metric-val" style="color:{_color(pct)}">{pct}%</div>'
                     f'<div class="meter"><span style="width:{pct}%;background:{_color(pct)}"></span></div></div>')
    if cells:
        st.markdown(f'<div class="metrics">{"".join(cells)}</div>', unsafe_allow_html=True)


def doc_card(name: str, stats: Dict, meta: Dict, active: bool) -> str:
    scanned = meta.get("scanned_pages") or []
    badges = '<span class="badge ok">✓ Indexed</span>' + ('<span class="badge info">In chat</span>' if active else "")
    if scanned:
        badges += f'<span class="badge warn">{len(scanned)} scanned page(s), no OCR</span>'
    title = meta.get("title") or ""
    stats_html = "".join(f'<span class="stat"><b>{stats[k]}</b> {label}</span>' for k, label in
                         (("pages", "pages"), ("table", "tables"), ("equation", "equations"), ("figure", "figures")))
    return (f'<div class="card"><div class="card-title">📄 {esc(name)}</div>'
            f'<div class="card-sub">{esc(title) if title else "&nbsp;"}</div>'
            f'<div class="stat-row">{stats_html}</div><div style="margin-top:.6rem">{badges}</div></div>')


def missing_card(name: str) -> str:
    return (f'<div class="card"><div class="card-title">📄 {esc(name)}</div>'
            '<div class="card-sub">The saved index is not available (storage reset or settings changed). '
            'Upload the PDF again to restore it.</div><span class="badge warn">Needs re-upload</span></div>')
