# Architecture Audit: gr_RAG (before the RAG∞ Pro upgrade)

Audited commit: `d62a94a` ("Datasheet RAG Agent: ReAct + Groq"), 3,155 lines of Python in 26 files.

## 1. Current architecture

```text
app.py (Streamlit, single page)
  │
  ├─ get_index(pdf) ─────────────► rag/pdf_parser.py ─┬─ rag/chunking.py           (sections → recursive chunks)
  │   (indexed once, cached)                          ├─ rag/table_extractor.py    (table + row/pin nodes)
  │                                                   ├─ rag/equation_extractor.py (regex "equation" lines)
  │                                                   └─ rag/figure_extractor.py   (images + page caption)
  │                               rag/embeddings.py (fastembed nomic-v1.5, in-app)
  │                               storage/vector_store.py (nodes JSON + embedding .npz)
  │
  └─ answer_query(question) ─────► rag/tools.initial_evidence  (hybrid retrieval, rag/retrieval.py)
                                   rag/agent.run_agent / run_quick (ReAct on Groq via rag/llm.py)
                                   citation canonicalisation + check (rag/pipeline.py)
                                   rag/evaluation.compute_metrics
```

## 2. Execution flow (one question)

1. `initial_evidence`: the query is embedded once; four engines (text, table, row/pin, equation) score nodes with `max(semantic, fuzzy) + bonuses − depth penalty`, threshold, de-duplicate.
2. The top excerpts (≤6, ≤5,000 chars) become the agent's first observation.
3. ReAct loop on Groq (`openai/gpt-oss-120b`): tools `search_datasheet`, `read_page`, `calculate`; at most 3 tool rounds.
4. Citations are rewritten to their true labels; invalid or missing citations are flagged.
5. Lexical metrics are computed and shown with sources and the agent trace.

## 3. Files and responsibilities

| File | Responsibility | Verdict |
|---|---|---|
| `rag/pdf_parser.py`, `chunking.py`, `table_extractor.py`, `equation_extractor.py`, `figure_extractor.py`, `metadata.py` | Local document intelligence | Keep (verified identical to the original RAG.py in earlier parity tests) |
| `rag/retrieval.py` | Hybrid scoring, engines, thresholds | Keep; becomes the candidate generator for reranking |
| `rag/embeddings.py` | In-app nomic embeddings with task prefixes | Keep |
| `rag/llm.py` | Groq client + friendly errors | Generalise into a provider layer |
| `rag/agent.py` | ReAct loop + answer-format prompt | Becomes the Master Agent |
| `rag/tools.py` | Tools + source registry | Keep; registry generalised to multi-document |
| `rag/pipeline.py` | Indexing + question answering | Indexing kept; answering kept as **LEGACY_MODE** |
| `rag/evaluation.py` | Lexical metrics | Keep and extend |
| `storage/*` | Cache, vector store, saved answers | Keep |
| `ui/*`, `app.py` | Single-page UI | Replace with a multi-page workspace |

## 4. Strengths (must not be broken)

* Deterministic local ingestion; parity-tested against the original RAG.py.
* Hybrid retrieval with section, context and metadata bonuses; embeddings cached per document hash.
* Retrieval-first ReAct: most questions take 1–2 LLM calls.
* Citation canonicalisation: displayed labels always match real sources.
* Graceful failure everywhere (bad key, rate limit, malformed tool call, bad PDF).
* Safe calculator (AST whitelist; no eval).

## 5. Weaknesses found

| # | Weakness | Impact |
|---|---|---|
| W1 | Agent panel shows the model's reasoning text ("Thought") | Exposes chain-of-thought (the brief forbids it) |
| W2 | Only one provider (Groq), hard-coded in `rag/llm.py` | No Ollama Cloud, no fallback |
| W3 | No reranking; lexical side is fuzzy `token_set_ratio` only (no BM25) | Exact symbols and part numbers can rank below vaguer chunks |
| W4 | One document at a time | No multi-document reasoning or comparison |
| W5 | One question at a time; no conversation memory | Follow-ups ("and at 85 °C?") lose context |
| W6 | Equation "agent" is a regex that also catches spec lines | Equations lack variables, units and surrounding explanation |
| W7 | Every image on a page gets the same caption; no surrounding text | Weak figure retrieval |
| W8 | No document-level agent (title, sections, counts) | "What is this document / list the sections" goes to text search |
| W9 | Verification only checks that citation numbers exist | Unsupported numbers can pass |
| W10 | Metrics are single-answer lexical heuristics; no latency per stage, no agent metrics, no feedback | Hard to tune or compare |
| W11 | No thumbs up/down, no preferences, no training-data export | No adaptation |
| W12 | No tests | Regressions go unnoticed |
| W13 | No benchmark | "Better" can't be measured |

## 6. Performance bottlenecks

| Stage | Cost | Note |
|---|---|---|
| First embedding-model load | ~130 MB download, once per container | Mitigated by background warm-up |
| Indexing | CPU embedding of every chunk | Cached per document hash |
| Retrieval | Fuzzy scoring over all nodes, 4 engines | Milliseconds for a datasheet |
| LLM | 1–4 Groq calls; free plan ~8K tokens/min | The real bottleneck; evidence must stay compact |

## 7. Migration plan (phases)

| Phase | Work | Status in this upgrade |
|---|---|---|
| 1 | This audit | ✅ |
| 2 | Provider layer: Groq, Ollama Cloud, local Ollama; role-based model router; explicit-opt-in fallback | ✅ |
| 3–4 | Ingestion upgrades: equation variables/units/context, per-image figure context, document metadata | ✅ |
| 5 | BM25 + cross-encoder reranker + evidence fusion | ✅ |
| 6–8 | Deterministic query orchestrator, specialist agents, evidence fusion | ✅ |
| 9 | Master Agent (ReAct) with fused evidence, preferences, conversation memory | ✅ |
| 10 | Verification agent (deterministic, optional LLM) with one bounded regeneration | ✅ |
| 11–12 | Ollama Cloud provider + fallback chain | ✅ |
| 13 | Chat: multiple conversations, memory, regenerate, export | ✅ |
| 14 | Metrics: stage latency, agent metrics, citation precision/coverage, health score | ✅ |
| 15 | Feedback, preference profile, DPO export | ✅ |
| 16 | Multi-page workspace UI (light/dark) | ✅ |
| 17 | pytest suite: 43 offline tests (fake providers + stand-in models) | ✅ |
| 18 | Benchmark: legacy vs new | ✅ (on a synthetic datasheet; rerun on your own PDFs) |
| 19 | README, ARCHITECTURE, DEPLOYMENT | ✅ |

Deliberately not done:
* **Real RLHF/DPO training.** Only the dataset export is built, as the brief specifies.
* **OCR of scanned PDFs.** This needs Tesseract, which Streamlit Cloud's Python-only build doesn't provide. Scanned pages are detected and reported instead.
* **A vision model by default.** None is on Groq's free plan. Configure `VISION_MODEL` if your provider has one.
