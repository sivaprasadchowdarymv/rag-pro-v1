"""
Benchmark: LEGACY pipeline vs RAG∞ PRO on questions with known answers.

    python -m metrics.benchmark --pdf my_datasheet.pdf --questions benchmark/questions.json \
        [--answers] [--out BENCHMARK_REPORT.md]

questions.json: [{"question": "...", "expected_pages": [2], "expected_answer_contains": ["2.2"]}, ...]

Retrieval metrics (no LLM needed): Precision@K, Recall@K, Hit rate, MRR, by page.
With --answers (uses your configured provider): answer correctness (all expected
strings present), grounding, citation precision, LLM calls, tokens, latency.
"""
from __future__ import annotations

import argparse
import json
import statistics
import math
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence

from agents.orchestrator import plan
from agents.rag_pro import answer_pro
from agents.retrieval_stage import run_retrieval
from config.settings import Settings, load_settings
from llm.model_router import ModelRouter
from rag.pipeline import answer_query, get_index
from rag.tools import SourceRegistry, ToolBox, initial_evidence


def retrieval_scores(ranked_pages: Sequence[int], expected: Sequence[int], k: int) -> Dict[str, float]:
    exp = set(expected)
    top = list(ranked_pages)[:k]
    hits = [p for p in top if p in exp]
    rr = next((1.0 / (i + 1) for i, p in enumerate(ranked_pages) if p in exp), 0.0)
    # NDCG@K with binary relevance; a page counts once (duplicates are not rewarded)
    seen, dcg = set(), 0.0
    for i, pg in enumerate(top):
        if pg in exp and pg not in seen:
            seen.add(pg)
            dcg += 1.0 / math.log2(i + 2)
    idcg = sum(1.0 / math.log2(i + 2) for i in range(min(len(exp), k)))
    return {"precision@k": len(hits) / max(len(top), 1), "recall@k": len(set(hits)) / max(len(exp), 1),
            "hit": 1.0 if hits else 0.0, "mrr": rr, "ndcg@k": dcg / idcg if idcg else 0.0}


def _legacy_pages(q: str, index, settings: Settings) -> List[int]:
    reg = SourceRegistry(settings)
    initial_evidence(q, ToolBox(index, settings, reg))
    return [r.page for r in reg.refs]


def _pro_pages(q: str, index, settings: Settings) -> List[int]:
    return [e.page for e in run_retrieval(plan(q), [index], settings).fusion.evidence]


def run(pdf_bytes: bytes, filename: str, questions: List[Dict], settings: Settings, k: int = 5,
        with_answers: bool = False, router: Optional[ModelRouter] = None) -> Dict:
    index, _ = get_index(pdf_bytes, filename, settings)
    rows = []
    for item in questions:
        q, pages = item["question"], item.get("expected_pages", [])
        row: Dict = {"question": q}
        for name, fn in (("legacy", _legacy_pages), ("pro", _pro_pages)):
            t0 = time.perf_counter()
            ranked = fn(q, index, settings)
            row[name] = {**retrieval_scores(ranked, pages, k), "retrieval_ms": (time.perf_counter() - t0) * 1000}
        if with_answers and router is not None:
            expect = [s.lower() for s in item.get("expected_answer_contains", [])]
            t0 = time.perf_counter()
            old = answer_query(q, index, settings)
            row["legacy"].update(correct=float(all(s in old.answer.lower() for s in expect)),
                                 llm_calls=old.llm_calls, tokens=old.tokens, answer_s=time.perf_counter() - t0)
            t0 = time.perf_counter()
            new = answer_pro(q, [index], settings, router)
            row["pro"].update(correct=float(all(s in new.answer.lower() for s in expect)),
                              grounded=float(bool(new.verification.get("grounded"))),
                              citation_precision=new.verification.get("citation_precision", 0.0),
                              llm_calls=new.llm_calls, tokens=new.tokens, answer_s=time.perf_counter() - t0)
        rows.append(row)

    def mean(name: str, key: str):
        vals = [r[name][key] for r in rows if key in r[name]]
        return round(statistics.mean(vals), 3) if vals else None

    keys = ["precision@k", "recall@k", "hit", "mrr", "ndcg@k", "retrieval_ms", "correct", "grounded",
            "citation_precision", "llm_calls", "tokens", "answer_s"]
    summary = {name: {kk: mean(name, kk) for kk in keys} for name in ("legacy", "pro")}
    return {"document": filename, "k": k, "questions": len(rows), "summary": summary, "rows": rows,
            "with_answers": with_answers, "rerank": settings.rerank}


def to_markdown(rep: Dict, note: str = "") -> str:
    lines = ["# Benchmark: legacy vs RAG∞ Pro", "", f"Document: `{rep['document']}`, {rep['questions']} questions, "
             f"K = {rep['k']}, reranking {'on' if rep['rerank'] else 'off'}.", ""]
    if note:
        lines += [f"> {note}", ""]
    lines += ["| Metric | Legacy | RAG∞ Pro |", "|---|---|---|"]
    names = {"precision@k": "Precision@K", "recall@k": "Recall@K", "hit": "Hit rate", "mrr": "MRR", "ndcg@k": "NDCG@K",
             "retrieval_ms": "Retrieval latency (ms)", "correct": "Answer correctness", "grounded": "Grounded",
             "citation_precision": "Citation precision", "llm_calls": "LLM calls / question",
             "tokens": "Tokens / question", "answer_s": "Answer latency (s)"}
    for key, label in names.items():
        a, b = rep["summary"]["legacy"].get(key), rep["summary"]["pro"].get(key)
        if a is None and b is None:
            continue
        lines.append(f"| {label} | {'-' if a is None else a} | {'-' if b is None else b} |")
    lines += ["", "## Per question", "", "| Question | Legacy MRR | Pro MRR |", "|---|---|---|"]
    for r in rep["rows"]:
        lines.append(f"| {r['question']} | {r['legacy']['mrr']:.2f} | {r['pro']['mrr']:.2f} |")
    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pdf", required=True)
    ap.add_argument("--questions", required=True)
    ap.add_argument("--answers", action="store_true", help="also generate answers (uses your LLM quota)")
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--out", default="BENCHMARK_REPORT.md")
    a = ap.parse_args()
    settings = load_settings()
    rep = run(Path(a.pdf).read_bytes(), Path(a.pdf).name, json.loads(Path(a.questions).read_text()), settings,
              a.k, a.answers, ModelRouter(settings) if a.answers else None)
    Path(a.out).write_text(to_markdown(rep))
    print(json.dumps(rep["summary"], indent=2))


if __name__ == "__main__":
    main()
