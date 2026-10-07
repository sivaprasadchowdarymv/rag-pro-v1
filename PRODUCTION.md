# RAG∞ Pro v3: Production Report

v3 is an incremental hardening of v2. Retrieval, agents, verification and the UI behave exactly as before; the before and after retrieval scores are identical. Everything added is a deterministic guard or a cache, so v3 makes **no new LLM calls**.

## 1. Audit results and changes (P0 → P1 → P2)

| # | Pri | CHANGE | WHY | IMPACT | FILES MODIFIED | TEST RESULT |
|---|---|---|---|---|---|---|
| 1 | P0 | Prompt-injection guard: instruction-like lines in PDFs are quoted as data, and the system prompt marks the evidence as untrusted | PDF text reached the LLM with no protection | A PDF can no longer redirect the model; the user sees a warning when the guard fires | `ops/guard.py`, `agents/master.py`, `agents/rag_pro.py` | `test_injection_*`, `test_registry_counts_injection` ✅ |
| 2 | P0 | Rate limit: sliding 60-second window, per session (8/min) and per server (25/min) | One visitor could use up the free Groq quota (~30 RPM) | The quota is protected, and users get a friendly "wait N s" message | `ops/guard.py`, `ui/pages.py`, `config/settings.py` | `test_rate_limiter_*` ✅ |
| 3 | P0 | Dockerfile (non-root user, health check), `.dockerignore`, `docker-compose.yml` | The checklist requires a Docker-ready deployment, and v2 had none | Runs on any VM; `/_stcore/health` is monitored | new files | `docker build` not run here (no Docker in the sandbox); health endpoint checked live ✅ |
| 4 | P1 | Answer cache (TTL + LRU) for **standalone** questions whose answer passed verification | Repeated questions re-ran the full pipeline and LLM calls | Repeats are instant and use 0 tokens; unverified answers are never cached | `ops/answer_cache.py`, `agents/rag_pro.py`, `rag/models.py` | `test_ttl_cache_and_key`, `test_pro_answer_cache_hit` ✅ |
| 5 | P1 | Query-embedding LRU across questions | Each question re-embedded the same text | Saves about 20–80 ms per repeated query on CPU | `rag/embeddings.py` | covered by the pipeline tests ✅ |
| 6 | P1 | System telemetry: P50/P95 latency, tokens, cost per request, cache hit rate, error rate, rate-limited count; Insights → **System** tab; JSONL log | The checklist asks for system metrics, and v2 had none | Observable in the UI and in `data/telemetry/requests.jsonl` (no question text stored) | `ops/telemetry.py`, `ui/pages.py` | `test_telemetry_summary` ✅, browser check of the System tab ✅ |
| 7 | P1 | NDCG@K added to the benchmark and Validate | NDCG was missing from the evaluation set | Measures ranking quality, not just whether there was a hit | `metrics/benchmark.py`, `metrics/validation.py` | `test_ndcg`, `test_retrieval_scores` ✅ |
| – | P2 | **Not done, on purpose:** pgvector, Redis, async ingestion, LangChain | One datasheet is a few thousand nodes; NumPy brute-force search takes milliseconds. Redis or Postgres would add cost and moving parts without measurable gain on the free tier. | Revisit above about 50 documents per workspace or more than 1 server replica | – | – |

**Tests:** 63 pass offline (56 from v2, unchanged, plus 7 new).

## 2. Checklist status (28 items)

✅ = in place · ➕ = added in v3 · ◻ = deliberately deferred

| | Item | Where |
|---|---|---|
| ✅ | 1 Smart chunking · 2 Metadata-rich chunks | `rag/chunking.py` (sections, parent breadcrumb, overlap), `Node` metadata, `doc_meta` |
| ✅ | 3 Content hash + dedup | `doc_id` = SHA-256 of the PDF (the same file is never indexed twice); near-duplicate evidence removed at fusion |
| ✅ | 4 Incremental ingestion · 5 Embedding caching | Index cached per `doc_id` + embedding signature; only missing embeddings are computed; ➕ query-vector LRU |
| ✅ | 6 Hybrid BM25 + vector · 7 RRF · 8 Reranker | `retrieval/bm25.py`, `evidence_fusion.rrf` (k=60), `retrieval/reranker.py` (MiniLM cross-encoder) |
| ✅ | 9 Query rewrite only when needed | Rule-based: follow-ups only (`orchestrator.plan`), no LLM |
| ✅ | 10 Context compression | Dedup + rerank + `MAX_CONTEXT_CHARS` budget + per-chunk truncation |
| ✅ | 11 Citations/pages · 12 Insufficient evidence · 13 Grounding | `[REF-n: TYPE pX]`, `NOT FOUND IN DOCUMENT` without calling the LLM when evidence is empty, claim-level verifier + 1 bounded regeneration |
| ✅ | 14 Memory · 15 Multi-turn | Bounded memory (`MEMORY_TURNS`), follow-up detection |
| ✅ | 16 Tenant isolation | Unguessable workspace ID (`?ws=`), per-workspace storage, optional `APP_PASSWORD` |
| ➕ | 17 Prompt injection | `ops/guard.py` |
| ✅ | 18 Input validation | PDF signature/size/pages/password checks, 1000-character question limit, filenames never used as paths |
| ➕ | 19 Rate limiting · 20 Retrieval/LLM caching | `ops/guard.py`, `ops/answer_cache.py` |
| ◻ | 21 Async ingestion | Streamlit runs indexing in the session with a progress bar; a worker queue only pays off with many concurrent uploads |
| ✅ | 22 Retry + timeout + fallback | SDK retries (2), `LLM_TIMEOUT`, provider fallback chain, evidence-only answer if every provider fails |
| ✅➕ | 23 Logging/tracing · 25 Latency/token/cost | `config.settings` logger, per-stage latency, actions-only trace, ➕ telemetry |
| ✅➕ | 24 Evaluation · 26 Health checks · 27 Docker · 28 Env config | Validate page + benchmark (➕ NDCG), `/_stcore/health`, ➕ Dockerfile, `.env.example` / Streamlit secrets |

## 3. Production architecture

```text
Browser ─HTTPS─► Streamlit (Community Cloud or Docker VM)
  ├─ guard: password · workspace isolation · rate limit · input validation
  ├─ INGEST: PDF → PyMuPDF parse → clean → metadata → smart chunks → SHA-256 doc_id (dedup)
  │          → fastembed nomic (cached) → NumPy index  ──sync──► private HF dataset
  └─ QUERY:  question → answer cache? ─hit─► answer
             → rule-based plan (follow-up rewrite only) → specialists (text/table/equation/figure/doc)
             → hybrid (cosine + fuzzy + BM25) → RRF → cross-encoder rerank → dedup + budget
             → injection guard → master LLM (Groq → Ollama Cloud fallback)
             → citation + number verifier (+1 regeneration) → answer → telemetry
```

## 4. Database / vector schema (file-based, no database server)

```text
data/
  users/<ws_id>/workspace.json          docs [{doc_id, filename, active}], validation set
  users/<ws_id>/conversations/*.json    chats (results incl. sources, metrics)
  users/<ws_id>/learning/feedback.jsonl ratings + reasons ; preferences.json
  cache/<doc_id>/nodes.json             Node: id, type, page, section, parent_ctx, content, metadata
  cache/<doc_id>/embeddings.npy         float32 [n_nodes × 768], L2-normalised; meta: format v3, embed signature
  documents/<doc_id>/figures/*.png      extracted figures ; vision.json (cached descriptions)
  telemetry/requests.jsonl              ts, q-hash, latency_s, tokens, llm_calls, provider, cached, error, cost
```
Hugging Face sync mirrors `users/`, `cache/` and `documents/` to a private dataset repo.

## 5. Evaluation dataset and metrics

* **Golden set:** `benchmark/sample_questions.json` (question, `expected_pages`, `expected_answer_contains`). Use **Validate** to auto-generate one per PDF, edit it in the table, and save it per workspace.
* **Retrieval:** Precision@K, Recall@K, Hit rate, MRR, ➕ NDCG@K.
* **Generation:** faithfulness, context coverage, answer relevance, citation precision/coverage/accuracy/support.
* **System:** ➕ P50/P95 latency, tokens, cost per request, cache hit rate, error rate.

```bash
python -m metrics.benchmark --pdf your.pdf --questions benchmark/sample_questions.json --answers
```

## 6. Benchmark, before vs after

Offline harness: synthetic datasheet, 4 questions, fake LLM server. **Latency here is not representative of real Groq latency.**

| | v2 (before) | v3 (after) |
|---|---|---|
| Retrieval P@5 / R@5 / Hit / MRR | 0.20 / 0.75 / 0.75 / 0.312 | **identical** (functionality preserved) |
| NDCG@5 | – | 0.423 |
| First-time questions: LLM calls / tokens | 6 / 5,312 | 6 / 5,487 (+3%: the injection note in the prompt) |
| **Repeated questions: LLM calls / tokens** | 6 / 5,312 | **2 / 1,678 (−68%)**; the 2 remaining were not verified, so not cached |
| Repeated question, pipeline time | 9.1 ms | 2.3 ms; with real Groq this saves 1–5 s per cached answer |

Rerun on your own PDFs from **Validate** to get real numbers.

## 7. Security checklist

- [x] Keys only in `.env` or Streamlit secrets; `.gitignore` and `.dockerignore` block `.env`, `secrets.toml`, `data/` and `*.pdf`
- [x] Optional `APP_PASSWORD` (constant-time compare); unguessable workspace IDs; private HF dataset
- [x] Prompt-injection guard; the evidence is marked untrusted; no chain-of-thought is shown
- [x] Rate limiting per session and per server; free-first providers only, never silent paid usage
- [x] Upload validation (size, signature, pages, encrypted files); HTML escaping; `showErrorDetails=false`
- [x] Docker: non-root user, bound to `127.0.0.1` in compose, health check, log rotation
- [ ] Not included: SSO/RBAC roles, encryption at rest (HF private repo only), WAF. Add these for confidential documents.

## 8. Monitoring

* **Insights → System:** P50/P95, cache hit rate, error rate, tokens, cost, rate-limited requests.
* **Insights → Quality / Agent activity / Feedback:** health score, per-stage latency, verifier verdicts.
* **Logs:** Streamlit Cloud → *Manage app → Logs*; Docker: `docker compose logs -f app`.
* **Uptime:** point a free uptime monitor (e.g. UptimeRobot) at `https://<app>/_stcore/health`. A visit every few hours also keeps a Streamlit Cloud app from sleeping.

## 9. New environment variables

| Variable | Default | Meaning |
|---|---|---|
| `RATE_LIMIT_PER_MIN` | 8 | Questions per browser session per minute (0 = off) |
| `GLOBAL_RATE_LIMIT_PER_MIN` | 25 | Questions per minute for the whole server |
| `ANSWER_CACHE_SIZE` / `ANSWER_CACHE_TTL` | 256 / 3600 | Verified-answer cache (TTL 0 = off) |
| `COST_PER_1K_TOKENS` | 0 | Only for estimating paid usage |

## 10. Production-readiness score

**82 / 100** for a free-tier single-instance deployment (v2 was about 70).

| | Score |
|---|---|
| Retrieval and grounding | 18/20 |
| Evaluation | 13/15 |
| Security | 13/20 (no SSO/RBAC or encryption at rest) |
| Reliability | 12/15 (single instance, free-tier sleep) |
| Observability | 13/15 |
| Deployability | 13/15 |

The remaining points need paid or heavier infrastructure (SSO, multi-replica, managed DB), which isn't justified at this scale.
