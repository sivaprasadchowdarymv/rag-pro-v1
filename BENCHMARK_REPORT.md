# Benchmark: legacy vs RAG∞ Pro

Document: `synthetic_LM7805X.pdf`, 8 questions, K = 5, reranking on.

> Smoke benchmark generated in a sandbox: synthetic 4-page datasheet, OFFLINE stand-ins for the embedding model, the cross-encoder and the LLM (tests/fakes). It proves the harness and pipelines run end to end; the numbers are NOT representative of real quality. Rerun with your own PDFs and real models: `python -m metrics.benchmark --pdf your.pdf --questions your_questions.json --answers`.

| Metric | Legacy | RAG∞ Pro |
|---|---|---|
| Precision@K | 0.64 | 0.675 |
| Recall@K | 0.875 | 1.0 |
| Hit rate | 0.875 | 1.0 |
| MRR | 0.792 | 1.0 |
| Retrieval latency (ms) | 0.263 | 0.791 |
| Answer correctness | 0.125 | 0.125 |
| Grounded | - | 0.625 |
| Citation precision | - | 1.0 |
| LLM calls / question | 1.25 | 1.625 |
| Tokens / question | 950.375 | 1420.375 |
| Answer latency (s) | 0.01 | 0.007 |

## Per question

| Question | Legacy MRR | Pro MRR |
|---|---|---|
| What is the peak output current? | 0.33 | 1.00 |
| What is the typical quiescent current? | 1.00 | 1.00 |
| What is the dropout voltage VDO? | 1.00 | 1.00 |
| Which equation gives the power dissipation PD? | 1.00 | 1.00 |
| How is the junction temperature TJ calculated? | 1.00 | 1.00 |
| What is the function of pin 2? | 1.00 | 1.00 |
| What does Figure 2 show? | 0.00 | 1.00 |
| What is the absolute maximum input voltage? | 1.00 | 1.00 |
