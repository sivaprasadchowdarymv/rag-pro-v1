# Architecture

## Levels (hierarchical, not "every agent talks to every agent")

| Level | Work | Cost |
|---|---|---|
| 1 Local deterministic | PDF parsing, chunking, tables, equations, figures, metadata, embeddings, BM25, indexing | CPU only; done once per document (cached by SHA-256) |
| 2 Lightweight | Orchestrator (rules), specialist retrieval, reranking, evidence fusion | Milliseconds; no LLM |
| 3 Master reasoning | One master model writes the answer; ReAct tools only when the plan needs them | 1 LLM call (simple), 2–4 (calculations, comparisons) |
| 4 Verification | Deterministic checks always; optional verifier model | ~1 ms; optional extra call |

## Request flow

```text
question
  │ agents/orchestrator.py       intents → agents to run (others skipped), follow-up rewrite, tools yes/no
  ▼
agents/retrieval_stage.py        run only the planned specialists over the active documents
  │   agents/specialists.py      text · table (TBL_n) · equation (EQ_n) · figure (FIG_n) · document
  │       candidates per agent:  "hybrid" (rag/retrieval.py: semantic + fuzzy + bonuses)
  │                              "bm25"   (retrieval/bm25.py: exact symbols, part numbers)
  │   retrieval/reranker.py      cross-encoder over the top 20 by RRF
  │   retrieval/evidence_fusion.py  merge, de-duplicate, rank, conflict notes, token budget
  ▼
agents/master.py                 EvidenceRegistry assigns [REF-n: TYPE pX]; ReAct loop via the router
  │   llm/model_router.py        role → (provider, model) chain; free-first fallback
  │   llm/providers.py           Groq · Ollama Cloud · local Ollama (one normalised interface)
  ▼
agents/verifier.py               canonical citations; citation precision/coverage; numeric and
  │                              equation grounding; one bounded regeneration with feedback
  ▼
metrics/rag_metrics.py           per-answer metrics + health score; stage latency recorded throughout
```

`agents/rag_pro.py` orchestrates the whole flow and returns a `QueryResult` with the answer, sources, plan, agent statistics, stage latency, verification report and metrics.

## Evidence record

Every specialist returns `rag.models.Evidence`:

```text
key, doc_id, doc_name, type, page, section, content (for the LLM), excerpt (for the user),
agent, item_id (EQ_n / TBL_n / FIG_n), confidence, ranks {hybrid, bm25, variable_match, ...}
```

## Design decisions

* **The router is rule-based, not an LLM.** It's predictable, testable, ~1 ms, and spends no tokens. The router is where speed is won or lost.
* **Retrieval comes first, before the first LLM call.** The master starts with strong evidence, so simple questions need one call.
* **Verification is deterministic by default.** It catches invented numbers, missing citations and foreign equation symbols at zero cost. The LLM verifier is opt-in because free tiers are token-limited.
* **No chain-of-thought is stored.** Providers are called with hidden reasoning disabled where supported, and the trace shows stages and tool actions only.
* **Legacy pipeline preserved.** `rag/pipeline.answer_query` is the previous system; the benchmark compares the two.
* **The vector store is NumPy, not a vector database.** Hybrid scoring touches every node anyway, and a datasheet has only thousands of nodes.

## Storage (`DATA_DIR`, default `./data`)

```text
data/documents/<doc_id>/   meta.json, figures/
data/cache/<doc_id>/       nodes + embeddings (rebuildable), vision.json
data/conversations/        one JSON per conversation
data/learning/             feedback.jsonl, preferences.json
data/models/               downloaded embedding/reranker models
```

Uploaded PDFs themselves are not kept after indexing.

## Module map

| Package | Purpose |
|---|---|
| `config/` | Settings from env / `.env` / Streamlit secrets |
| `rag/` | Ingestion, embeddings, original hybrid retrieval, legacy agent and pipeline |
| `retrieval/` | BM25, reranker, evidence fusion |
| `agents/` | Orchestrator, specialists, retrieval stage, master, verifier, Pro pipeline |
| `llm/` | Providers and model router |
| `chat/` | Conversations and memory |
| `learning/` | Feedback, preferences, DPO export |
| `metrics/` | Live metrics, health score, benchmark |
| `storage/` | Cache, vector store, saved answers |
| `ui/` | Pages, widgets, state, styles |
| `tests/` | Offline test suite with fake providers and models |

## Workspaces and persistent storage

```text
browser ── ?ws=<24-char random key> ──► storage/workspace.py   data/users/<ws>/  (chats, feedback, preferences,
                                                               document list, validation sets) → private per visitor
storage/remote.py  SyncManager ── batched commits (every SYNC_INTERVAL s or "Save now") ──► private HF dataset
                   restore on open: users/<ws>/**, cache/<doc>/**, documents/<doc>/**
rag/pipeline.open_index(doc_id)  reopens a document from its saved index: no PDF needed
```

## Verification (claim level)

For each sentence of the answer, the verifier:
* checks every number against the sources **that sentence cites**;
* reports values that only exist in another source as mis-cited, and names the correct source in the correction prompt;
* accepts calculator results as supported by the calculation itself;
* asks the local cross-encoder whether each cited excerpt supports its sentence;
* checks that equation symbols exist in the sources.

Verdicts are ✓ verified / ≈ partly verified / ⚠ check sources, with one bounded correction.
`metrics/validation.py` measures all of this on your own documents.
