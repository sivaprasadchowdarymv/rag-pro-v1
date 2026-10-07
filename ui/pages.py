"""Pages: Chat · Library · Validate · Insights · Settings."""
from __future__ import annotations

import json
import statistics
import time
from typing import Dict, List

import pandas as pd
import streamlit as st

from agents.rag_pro import answer_pro
from chat.conversations import result_from_dict
from learning.dpo_dataset import build_pairs
from learning.feedback import REASONS
from learning.preferences import DEPTHS, LENGTHS, Preferences
from llm.model_router import DEFAULT_MODELS
from metrics import validation
from metrics.rag_metrics import health_score
from rag.models import AgentStep, QueryResult
from rag.pdf_parser import PdfProcessingError
from rag.pipeline import answer_query, get_index
from ui import state, widgets

MAX_QUESTION_CHARS = 1000
GENERIC_SUGGESTIONS = ["Summarize this document", "What are the absolute maximum ratings?",
                       "What is the pin configuration?", "How do I calculate the power dissipation?"]


# =============================================================================
# Shared helpers
# =============================================================================
def ask(question: str, conv, on_step) -> QueryResult:
    s, indexes = state.settings(), state.active_indexes()
    if s.pipeline_mode == "legacy":
        result = answer_query(question, indexes[0], s, on_step=on_step)
        if len(indexes) > 1:
            result.warnings.append("Legacy mode answers from the first active document only.")
        return result
    return answer_pro(question, indexes, s, state.router(), memory=conv.memory(s.memory_turns),
                      previous_question=conv.last_question(), preferences=state.preferences().load().to_prompt(),
                      on_step=on_step)


def _run_with_status(question: str, conv) -> QueryResult:
    with st.status("Thinking…", expanded=True) as box:
        def on_step(step: AgentStep) -> None:
            if step.kind in ("stage", "action", "note"):
                box.write(("✓ " if step.kind == "stage" else "🔧 ") + widgets.describe_step(step))
        result = ask(question, conv, on_step)
        box.update(label=f"Answered in {result.latency} s", state="complete", expanded=False)
    st.session_state["last_result"] = result
    return result


def _upload_box(key: str, label: str = "Drop PDF files here") -> None:
    s = state.settings()
    uploads = st.file_uploader(label, type=["pdf"], accept_multiple_files=True, key=key,
                               help=f"Up to {s.max_upload_mb} MB and {s.max_pages} pages each.")
    for up in uploads or []:
        marker = f"done_{key}_{up.name}_{up.size}"
        if st.session_state.get(marker):
            continue
        bar = st.progress(0.0, text=f"Reading {up.name}…")
        try:
            index, warnings = get_index(up.getvalue(), up.name, s,
                                        progress=lambda f, m: bar.progress(min(max(f, 0.0), 1.0), text=m))
            state.add_doc(index)
            st.session_state[marker] = True
            st.toast(f"{index.filename} is ready.", icon="✅")
            for w in warnings:
                st.warning(w)
        except PdfProcessingError as exc:
            st.error(f"{up.name}: {exc.user_message}")
        finally:
            bar.empty()


def _suggestions() -> List[str]:
    if "suggestions" not in st.session_state:
        qs: List[str] = []
        for index in state.active_indexes()[:2]:
            qs += [q["question"] for q in validation.generate_questions(index, limit=6)]
        picked = []
        for q in qs:  # mix question types: values, pins, equations
            if q not in picked and len(picked) < 3:
                picked.append(q)
        st.session_state["suggestions"] = (picked + [g for g in GENERIC_SUGGESTIONS if g not in picked])[:4]
    return st.session_state["suggestions"]


# =============================================================================
# Sidebar (all pages)
# =============================================================================
def sidebar() -> None:
    sync = state.sync_manager().status()
    with st.sidebar:
        st.caption("Verified answers from your documents")
        if st.button("✏️  New chat", use_container_width=True, type="primary"):
            conv = state.conversations().create([i.doc_id for i in state.active_indexes()])
            st.session_state["conv_id"] = conv.id
            st.session_state.pop("last_result", None)
            st.switch_page(st.session_state["pages"]["chat"])
        current_id = st.session_state.get("conv_id")
        convs = [c for c in state.conversations().list() if c.messages or c.id == current_id][:25]
        if any(c.messages for c in convs):
            widgets.section("Recent chats")
            current = st.session_state.get("conv_id")
            for c in convs:
                label = ("● " if c.id == current else "") + c.title
                if st.button(label, key=f"conv_{c.id}", use_container_width=True):
                    st.session_state["conv_id"] = c.id
                    st.session_state.pop("last_result", None)
                    st.switch_page(st.session_state["pages"]["chat"])
        widgets.section("Status")
        lines = []
        configured = [p for p in state.provider_status() if p.configured]
        if not configured:
            lines.append(widgets.status_line(False, "AI", "no provider configured"))
        for p in configured:
            lines.append(widgets.status_line(p.connected, p.label, "connected" if p.connected else "unavailable"))
        if sync.enabled:
            ok = not sync.last_error
            lines.append(widgets.status_line(ok, "Storage", "saved to Hugging Face" if ok else sync.last_error))
        else:
            lines.append(widgets.status_line(None, "Storage", "temporary (resets on restart)"))
        st.markdown("".join(lines), unsafe_allow_html=True)


# =============================================================================
# Chat
# =============================================================================
def _feedback_widget(conv, msg, question: str) -> None:
    if msg.feedback:
        st.caption("Thanks for your feedback " + ("👍" if msg.feedback["rating"] > 0 else "👎"))
        return
    rating = st.feedback("thumbs", key=f"fb_{msg.id}")
    if rating is None:
        return
    reasons, comment = [], ""
    if rating == 0:
        reasons = st.pills("What went wrong?", REASONS, selection_mode="multi", key=f"fbr_{msg.id}") or []
        comment = st.text_input("Anything else? (optional)", key=f"fbc_{msg.id}", max_chars=500)
    if st.button("Send feedback", key=f"fbs_{msg.id}", type="primary"):
        r = msg.result or {}
        value = 1 if rating == 1 else -1
        state.feedback().record(conversation_id=conv.id, message_id=msg.id, question=question, answer=msg.content,
                                rating=value, reasons=list(reasons), comment=comment, provider=r.get("provider", ""),
                                model=r.get("model", ""), metrics=r.get("metrics", {}))
        msg.feedback = {"rating": value, "reasons": list(reasons), "comment": comment}
        state.conversations().save(conv)
        changes = state.preferences().apply_feedback(value, list(reasons))
        st.toast("Feedback saved." + (" I'll adapt: " + "; ".join(changes) if changes else ""), icon="🙏")
        st.rerun()


def chat_page() -> None:
    s = state.settings()
    conv = state.current_conversation()
    indexes = state.active_indexes()

    if not indexes:
        widgets.hero("Ask your documents anything",
                     "Upload a datasheet or technical PDF. Every answer cites the exact page, table or equation, "
                     "and is checked before you see it.",
                     [("Upload", "Add one or more PDFs below."), ("Ask", "Type a question in plain English."),
                      ("Verify", "Open the evidence behind every answer.")])
        _upload_box("chat_upload")
        if state.active_indexes():
            st.rerun()
        return

    chips = "".join(f'<span class="chip">📄 {widgets.esc(i.filename)}</span>' for i in indexes)
    st.markdown(f'<div class="meta" style="margin-bottom:.4rem">Answering from</div>{chips}', unsafe_allow_html=True)

    pending = st.session_state.pop("pending_question", None)
    if not conv.messages and not pending:
        widgets.hero("What would you like to know?",
                     "Pick a suggestion or type your own question. Follow-up questions keep the context.")
        cols = st.columns(2)
        for i, q in enumerate(_suggestions()):
            if cols[i % 2].button(q, key=f"sugg_{i}", use_container_width=True):
                st.session_state["pending_question"] = q
                st.rerun()

    last_q = ""
    assistants = [m for m in conv.messages if m.role == "assistant"]
    for msg in conv.messages:
        with st.chat_message(msg.role, avatar="🧑" if msg.role == "user" else "♾️"):
            if msg.role == "user":
                last_q = msg.content
                st.markdown(msg.content)
                continue
            result = result_from_dict(msg.result) if msg.result else QueryResult("answer", msg.content)
            widgets.render_result(result, s)
            c1, c2, _ = st.columns([1, 1.3, 4])
            with c1.popover("Copy"):
                st.code(msg.content, language="markdown")
            if msg is assistants[-1] and c2.button("↻ Regenerate", key=f"regen_{msg.id}"):
                new = _run_with_status(last_q, conv)
                state.conversations().replace_answer(conv, msg.id, new)
                st.rerun()
            _feedback_widget(conv, msg, last_q)

    question = st.chat_input("Ask about your documents…", max_chars=MAX_QUESTION_CHARS) or pending
    if question and question.strip():
        with st.chat_message("user", avatar="🧑"):
            st.markdown(question)
        with st.chat_message("assistant", avatar="♾️"):
            result = _run_with_status(question.strip(), conv)
        state.conversations().add_exchange(conv, question.strip(), result)
        st.rerun()


# =============================================================================
# Library
# =============================================================================
def library_page() -> None:
    st.title("📚 Library")
    st.caption("Documents in your workspace. They are reopened automatically next time"
               + (" (saved to Hugging Face)." if state.sync_manager().enabled else "; storage is temporary on this server."))
    _upload_box("library_upload", "Add PDFs")
    docs = state.docs()
    if not docs:
        st.info("Your library is empty. Add a PDF above.")
        return
    cols = st.columns(2)
    for i, (doc_id, d) in enumerate(list(docs.items())):
        with cols[i % 2]:
            index = d["index"]
            if index is None:
                st.markdown(widgets.missing_card(d["filename"]), unsafe_allow_html=True)
                if st.button("Remove", key=f"rm_{doc_id}"):
                    state.remove_doc(doc_id)
                    st.rerun()
                continue
            st.markdown(widgets.doc_card(index.filename, index.stats(), index.doc_meta or {}, d["active"]),
                        unsafe_allow_html=True)
            a, b, c = st.columns([1.3, 1, 1])
            active = a.toggle("Use in chat", value=d["active"], key=f"act_{doc_id}")
            state.set_active(doc_id, active)
            if b.button("Ask →", key=f"ask_{doc_id}"):
                for other in docs:
                    state.set_active(other, other == doc_id)
                st.switch_page(st.session_state["pages"]["chat"])
            if c.button("Remove", key=f"rm_{doc_id}"):
                state.remove_doc(doc_id)
                st.rerun()
            sections = (index.doc_meta or {}).get("sections") or []
            if sections:
                with st.expander("Sections"):
                    st.markdown("\n".join(f"- {t} · p{p}" for t, p in sections[:60]))
            st.write("")


# =============================================================================
# Validate
# =============================================================================
def validate_page() -> None:
    st.title("✅ Validate")
    st.caption("Check accuracy on YOUR documents. Questions are generated from your tables and equations with "
               "known answers; review them, then run the test.")
    indexes = state.active_indexes()
    if not indexes:
        st.info("Add a document in the Library first.")
        return
    ws = state.workspace()
    rows = st.session_state.get("val_rows") or ws.load_validation()
    c1, c2 = st.columns([2, 1])
    by_id = {i.doc_id: i for i in indexes}  # options must be plain values, not index objects
    target = by_id[c1.selectbox("Document", list(by_id), format_func=lambda d: by_id[d].filename)]
    count = c2.number_input("Questions", 3, 30, 10)
    if st.button("✨ Generate test questions", type="primary") or (not rows and st.session_state.get("val_auto") is None):
        st.session_state["val_auto"] = True
        rows = validation.generate_questions(target, limit=int(count))
        st.session_state["val_rows"] = rows
    if not rows:
        st.warning("No questions could be generated (no tables or formulas found). Add your own rows below.")
        rows = [{"question": "", "expected_pages": [1], "expected_answer_contains": [""]}]

    table = pd.DataFrame([{"question": r["question"],
                           "expected pages": ", ".join(map(str, r.get("expected_pages", []))),
                           "expected answer contains": ", ".join(map(str, r.get("expected_answer_contains", [])))}
                          for r in rows])
    edited = st.data_editor(table, num_rows="dynamic", use_container_width=True, hide_index=True, key="val_editor")
    questions = []
    for _, r in edited.iterrows():
        if str(r["question"]).strip():
            pages = [int(p) for p in str(r["expected pages"]).replace(" ", "").split(",") if p.isdigit()]
            expect = [x.strip() for x in str(r["expected answer contains"]).split(",") if x.strip()]
            questions.append({"question": str(r["question"]).strip(), "expected_pages": pages,
                              "expected_answer_contains": expect})
    if st.button("💾 Save this question set"):
        ws.save_validation(questions)
        st.toast("Saved to your workspace.")

    a, b = st.columns(2)
    run_retrieval = a.button("Check retrieval (free, no AI calls)", use_container_width=True)
    run_full = b.button(f"Full test ({len(questions)} AI answers)", use_container_width=True,
                        disabled=not state.router().enabled)
    if (run_retrieval or run_full) and questions:
        bar = st.progress(0.0)
        rep = validation.evaluate(questions, indexes, state.settings(), state.router(), with_answers=run_full,
                                  pause_s=2.0 if run_full else 0.0,
                                  progress=lambda f, m: bar.progress(min(max(f, 0.0), 1.0), text=m))
        bar.empty()
        st.session_state["val_report"] = rep
    rep = st.session_state.get("val_report")
    if rep:
        s = rep["summary"]
        widgets.metric_cards({"Retrieval hit rate": s["retrieval_hit_rate"], "MRR": s["mrr"],
                              "Answer accuracy": s["answer_accuracy"], "Cites correct page": s["cited_correct_page"],
                              "Verified": s["verified_rate"]})
        view = pd.DataFrame([{"question": r["question"], "expected": r["expected"],
                              "retrieval": "✅" if r["retrieval_hit"] else "❌",
                              "answer": "-" if "answer_correct" not in r else ("✅" if r["answer_correct"] else "❌"),
                              "verdict": r.get("verdict", "-")} for r in rep["rows"]])
        st.dataframe(view, use_container_width=True, hide_index=True)
        st.download_button("Download report", validation.to_markdown(rep), "VALIDATION_REPORT.md")


# =============================================================================
# Insights
# =============================================================================
def _all_results() -> List[Dict]:
    return [{"conversation": c.title, **m.result} for c in state.conversations().list()
            for m in c.messages if m.role == "assistant" and m.result and m.result.get("metrics")]


def insights_page() -> None:
    st.title("📊 Insights")
    tab1, tab2, tab3 = st.tabs(["Quality", "Agent activity", "Feedback"])
    s = state.settings()
    results = _all_results()
    fb = state.feedback().stats()
    with tab1:
        if not results:
            st.info("Ask a few questions; quality metrics appear here.")
        else:
            df = pd.DataFrame([{**r["metrics"], "question": r.get("question", "")} for r in results])
            mean = {k: float(df[k].mean()) for k in df.columns if df[k].dtype.kind in "fi"}
            st.metric("RAG health score", f"{health_score(mean, s.health_weights, fb['ratio'])} / 100",
                      help="Application-level composite of faithfulness, relevance, citation accuracy, grounding and "
                           "user feedback. A monitoring aid, not a scientific measure of correctness.")
            widgets.metric_cards({"Faithfulness": mean.get("faithfulness"), "Relevance": mean.get("answer_relevance"),
                                  "Citation precision": mean.get("citation_precision"),
                                  "Citation accuracy": mean.get("citation_accuracy"),
                                  "Citation support": mean.get("citation_support"),
                                  "Numbers grounded": mean.get("numerical_score"),
                                  "Grounded answers": mean.get("grounded"), "User 👍": fb["ratio"]})
            c1, c2, c3 = st.columns(3)
            c1.metric("Answers", len(results))
            c2.metric("Avg latency", f"{mean.get('latency_s', 0):.2f} s")
            c3.metric("Avg tokens", f"{mean.get('tokens', 0):.0f}")
            stages: Dict[str, List[float]] = {}
            for r in results:
                for k, v in (r.get("stage_latency") or {}).items():
                    if k != "total":
                        stages.setdefault(k, []).append(v)
            if stages:
                widgets.section("Average latency by stage (ms)")
                st.bar_chart(pd.Series({k: statistics.mean(v) for k, v in stages.items()}, name="ms"))
            with st.expander("All answers"):
                cols = [c for c in ("question", "health_score", "faithfulness", "citation_accuracy", "citation_support",
                                    "numerical_score", "grounded", "latency_s", "tokens") if c in df.columns]
                st.dataframe(df[cols], hide_index=True, use_container_width=True)
    with tab2:
        result = st.session_state.get("last_result")
        if result is None:
            conv = state.current_conversation()
            last = next((m for m in reversed(conv.messages) if m.role == "assistant" and m.result), None)
            result = result_from_dict(last.result) if last else None
        if result is None:
            st.info("Ask a question to see how the agents handled it.")
        else:
            st.markdown(f"**Question:** {result.question or '(legacy mode)'}")
            if result.plan:
                st.markdown(" ".join(f'<span class="chip">{widgets.esc(i)}</span>' for i in result.plan.get("intents", [])),
                            unsafe_allow_html=True)
                for name in ("text_agent", "table_agent", "equation_agent", "figure_agent", "document_agent"):
                    ran = name in result.plan.get("agents", [])
                    st_ = (result.agent_stats or {}).get(name, {})
                    st.markdown(f"{'✅' if ran else '⏭️'} **{name.replace('_', ' ').title()}**"
                                + (f": {int(st_.get('results', 0))} candidates · {st_.get('latency_ms', 0):.1f} ms"
                                   if ran else " · skipped (not needed)"))
            if result.stage_latency:
                widgets.section("Latency by stage")
                st.dataframe(pd.DataFrame([{"stage": k, "ms": round(v, 1)} for k, v in result.stage_latency.items()]),
                             hide_index=True)
            widgets.section("Steps (actions only)")
            widgets.render_trace(result.steps)
            if result.verification:
                with st.expander("Verification report"):
                    st.json(result.verification)
    with tab3:
        c1, c2, c3 = st.columns(3)
        c1.metric("👍 Helpful", fb["up"])
        c2.metric("👎 Not helpful", fb["down"])
        c3.metric("Helpful rate", "-" if fb["ratio"] is None else f"{fb['ratio'] * 100:.0f}%")
        if fb["reasons"]:
            st.bar_chart(pd.Series(fb["reasons"], name="count"))


# =============================================================================
# Settings
# =============================================================================
def settings_page() -> None:
    st.title("⚙️ Settings")
    t1, t2, t3, t4 = st.tabs(["Workspace & storage", "AI models", "Answer style", "Advanced"])
    s = state.settings()
    with t1:
        ws = state.workspace()
        st.markdown("**Your private workspace link.** Bookmark it to come back to your chats and documents. "
                    "Anyone with this link can open the workspace, so keep it private.")
        st.code(f"?ws={ws.id}", language=None)
        st.caption("Add it to the end of the app address, e.g. https://your-app.streamlit.app/?ws=…")
        with st.form("switch_ws"):
            code = st.text_input("Open another workspace (paste its code)")
            if st.form_submit_button("Open"):
                if state.switch_workspace(code.strip().split("ws=")[-1]):
                    st.rerun()
                st.error("That doesn't look like a workspace code.")
        sync = state.sync_manager().status()
        widgets.section("Storage")
        if sync.enabled:
            ago = f"{int(time.time() - sync.last_push)} s ago" if sync.last_push else "not yet"
            st.markdown(f'<div class="store">☁️ <b>Saved to your private Hugging Face dataset</b><br>'
                        f'Last upload: {ago} · pending changes: {sync.pending}'
                        f'{"<br>⚠ " + widgets.esc(sync.last_error) if sync.last_error else ""}</div>',
                        unsafe_allow_html=True)
            if st.button("Save now"):
                ok = state.sync_manager().flush("manual")
                st.toast("Saved." if ok else "Upload failed; it will be retried automatically.")
        else:
            st.markdown('<div class="store">⚠️ <b>Temporary storage.</b> Chats, feedback and indexes are lost when the '
                        'server restarts. Add HF_TOKEN and HF_DATASET_REPO to the secrets to keep them (see '
                        'DEPLOYMENT.md).</div>', unsafe_allow_html=True)
        conv = state.current_conversation()
        st.download_button("Export current chat (Markdown)", state.conversations().export_markdown(conv),
                           f"chat_{conv.id}.md")
        if st.button("Delete current chat"):
            state.conversations().delete(conv.id)
            st.session_state.pop("conv_id", None)
            st.rerun()
    with t2:
        st.caption("Only providers with a key are ever called, in this order. Use free-plan keys: the app cannot "
                   "see your billing plan and never enables a provider by itself.")
        statuses = state.provider_status()
        st.markdown("".join(widgets.status_line(p.connected if p.configured else None, p.label,
                                                ("connected" if p.connected else p.message or "unavailable")
                                                if p.configured else "not configured") for p in statuses),
                    unsafe_allow_html=True)
        active = [p for p in statuses if p.configured and p.connected]
        if active:
            first = active[0]
            default = s.master_model or DEFAULT_MODELS.get(first.name, {}).get("master", "")
            model = st.selectbox(f"Answer model ({first.label})", list(dict.fromkeys([default] + first.models)))
            if model != default:
                state.set_override(master_model=model)
        mode = st.radio("Answer mode", ["agent", "quick"], index=0 if s.agent_mode == "agent" else 1, horizontal=True,
                        format_func=lambda x: "🧠 Thorough (agent uses tools when needed)" if x == "agent"
                        else "⚡ Quick (one AI call, saves quota)")
        state.set_override(agent_mode=mode)
    with t3:
        store = state.preferences()
        p = store.load()
        with st.form("prefs"):
            length = st.select_slider("Answer length", LENGTHS, value=p.response_length)
            depth = st.select_slider("Technical depth", DEPTHS, value=p.technical_depth)
            c = st.columns(4)
            cit, eq = c[0].toggle("Citations", p.citations), c[1].toggle("Equations", p.equations)
            fig, tab = c[2].toggle("Figures", p.figures), c[3].toggle("Tables", p.tables)
            if st.form_submit_button("Save", type="primary"):
                store.save(Preferences(length, depth, cit, eq, fig, tab, p.history))
                st.toast("Saved.")
        if p.history:
            with st.expander(f"Learned from your feedback ({len(p.history)})"):
                for h in reversed(p.history[-20:]):
                    st.write("• " + "; ".join(h["changes"]))
        pairs = build_pairs(state.feedback().all())
        st.download_button(f"Download training pairs (DPO, {len(pairs)})",
                           "\n".join(json.dumps(x, ensure_ascii=False) for x in pairs), "dpo_dataset.jsonl",
                           disabled=not pairs)
    with t4:
        c1, c2 = st.columns(2)
        pipe = c1.radio("Pipeline", ["pro", "legacy"], index=0 if s.pipeline_mode == "pro" else 1,
                        format_func=lambda x: "RAG∞ Pro" if x == "pro" else "Legacy (previous version)")
        rerank = c2.toggle("Cross-encoder reranking", s.rerank)
        c3, c4, c5 = st.columns(3)
        vllm = c3.toggle("Extra AI verification (uses tokens)", s.verify_with_llm)
        regen = c4.number_input("Max corrections", 0, 2, s.max_regenerations)
        steps = c5.slider("Agent tool rounds", 1, 6, s.agent_max_steps)
        c6, c7 = st.columns(2)
        memory = c6.slider("Conversation memory (turns)", 0, 10, s.memory_turns)
        top_k = c7.slider("Top-K per retriever", 1, 10, s.top_k)
        widgets.section("Health score weights")
        names = ["Faithfulness", "Relevance", "Citations", "Grounding", "Feedback"]
        wc = st.columns(5)
        weights = [wc[i].number_input(n, 0.0, 1.0, float(round(s.health_weights[i], 2)), 0.05) for i, n in enumerate(names)]
        total = sum(weights) or 1.0
        state.set_override(pipeline_mode=pipe, rerank=rerank, verify_with_llm=vllm, max_regenerations=int(regen),
                           agent_max_steps=steps, memory_turns=memory, top_k=top_k,
                           health_weights=tuple(w / total for w in weights))
