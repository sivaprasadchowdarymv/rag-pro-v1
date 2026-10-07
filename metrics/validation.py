"""
Validate the system on YOUR documents.

1. generate_questions(): build a test set from the document itself, with known
   answers: table rows with Min/Typ/Max values, pin functions, and the
   equations that define a symbol. Every question records its source page and
   expected value, so results can be checked automatically. Review and edit
   the set before trusting it.
2. evaluate(): run each question through retrieval (free) and optionally the
   full pipeline (uses LLM quota) and score:
     retrieval hit / MRR      the expected page is among the retrieved evidence
     answer correct           the expected value appears in the answer
     cited correct page       at least one citation points to the expected page
     verified                 the verifier's verdict is "verified"
"""
from __future__ import annotations

from metrics.benchmark import retrieval_scores
import re
import statistics
import time
from typing import Callable, Dict, List, Optional, Sequence

from agents.orchestrator import plan
from agents.rag_pro import answer_pro
from agents.retrieval_stage import run_retrieval
from agents.specialists import _VAR, analyse_equation
from config.settings import Settings
from llm.model_router import ModelRouter
from rag.models import DocumentIndex

_LEVELS = (("max", "maximum"), ("typ", "typical"), ("min", "minimum"))
_NUMBER = re.compile(r"-?\d+(?:\.\d+)?")


def _cells(line: str) -> List[str]:
    return [c.strip() for c in line.split(" | ")]


def generate_questions(index: DocumentIndex, limit: int = 12) -> List[Dict]:
    out: List[Dict] = []
    seen = set()

    def add(q: str, page: int, expect: str, source: str) -> None:
        key = q.lower()
        if key not in seen and len(out) < limit:
            seen.add(key)
            out.append({"question": q, "expected_pages": [page], "expected_answer_contains": [expect],
                        "source": source, "document": index.filename})

    for n in index.nodes:
        if n.type not in ("row", "pin"):
            continue
        lines = n.raw_content.split("\n")
        if len(lines) < 2:
            continue
        header, row = _cells(lines[0]), _cells(lines[1])
        if len(row) != len(header) or not row[0] or _NUMBER.fullmatch(row[0]):
            if n.type == "pin" and len(row) >= 3 and row[0].isdigit():
                add(f"What is the function of pin {row[0]}?", n.page, row[1] if row[1] else row[2], "pin table")
            continue
        param = row[0]
        cols = [(i, word) for key, word in _LEVELS for i, h in enumerate(header) if h.lower().startswith(key)]
        for col, word in cols:  # prefer max, then typ, then min
            match = _NUMBER.search(row[col]) if col < len(row) else None
            if match:
                add(f"What is the {word} {param.lower()}?", n.page, match.group(0), f"table row ({header[col]})")
                break
    for n in index.nodes:
        if n.type != "equation":
            continue
        info = analyse_equation(n.raw_content, [])
        lhs = [v for v in _VAR.findall(n.raw_content.partition("=")[0]) if len(v) >= 2]
        if info["is_formula"] and lhs:
            add(f"Which equation is used to calculate {lhs[0]}?", n.page, lhs[0], "equation")
    return out


def _contains(answer: str, expected: Sequence[str]) -> bool:
    low = answer.lower()
    for exp in expected:
        e = str(exp).strip().lower()
        if not e:
            continue
        if _NUMBER.fullmatch(e):  # numeric: compare as numbers (2.2 == 2.20)
            if not any(abs(float(x) - float(e)) < 1e-9 for x in _NUMBER.findall(low)):
                return False
        elif e not in low:
            return False
    return True


def evaluate(questions: Sequence[Dict], indexes: Sequence[DocumentIndex], settings: Settings,
             router: Optional[ModelRouter] = None, with_answers: bool = False, pause_s: float = 0.0,
             progress: Optional[Callable[[float, str], None]] = None) -> Dict:
    rows = []
    for i, q in enumerate(questions):
        if progress:
            progress(i / max(len(questions), 1), f"Question {i + 1}/{len(questions)}")
        pages = set(int(p) for p in q.get("expected_pages", []))
        t0 = time.perf_counter()
        found = [e.page for e in run_retrieval(plan(q["question"]), indexes, settings).fusion.evidence]
        rank = next((k + 1 for k, p in enumerate(found) if p in pages), None)
        row = {"question": q["question"], "expected": ", ".join(map(str, q.get("expected_answer_contains", []))),
               "retrieval_hit": rank is not None, "mrr": round(1 / rank, 3) if rank else 0.0,
               "ndcg@5": round(retrieval_scores(found, sorted(pages), 5)["ndcg@k"], 3),
               "retrieval_ms": round((time.perf_counter() - t0) * 1000, 1)}
        if with_answers and router is not None:
            r = answer_pro(q["question"], indexes, settings, router)
            row.update(answer_correct=_contains(r.answer, q.get("expected_answer_contains", [])),
                       cited_correct_page=any(s.page in pages for s in r.sources),
                       verdict=r.verification.get("verdict", "n/a"), latency_s=r.latency, tokens=r.tokens,
                       answer=r.answer[:300])
            if pause_s:
                time.sleep(pause_s)  # stay under free-tier per-minute limits
        rows.append(row)
    if progress:
        progress(1.0, "Done")

    def rate(key: str) -> Optional[float]:
        vals = [r[key] for r in rows if key in r]
        return round(sum(1 for v in vals if v is True or v == "verified") / len(vals), 3) if vals else None

    summary = {"questions": len(rows), "retrieval_hit_rate": rate("retrieval_hit"),
               "mrr": round(statistics.mean(r["mrr"] for r in rows), 3) if rows else None,
               "ndcg@5": round(statistics.mean(r["ndcg@5"] for r in rows), 3) if rows else None,
               "answer_accuracy": rate("answer_correct"), "cited_correct_page": rate("cited_correct_page"),
               "verified_rate": rate("verdict")}
    return {"summary": summary, "rows": rows}


def to_markdown(report: Dict, title: str = "Validation report") -> str:
    s = report["summary"]
    pct = lambda v: "-" if v is None else f"{v * 100:.0f}%"  # noqa: E731
    lines = [f"# {title}", "", f"{s['questions']} questions", "",
             "| Metric | Value |", "|---|---|",
             f"| Retrieval hit rate | {pct(s['retrieval_hit_rate'])} |", f"| MRR | {s['mrr']} |", f"| NDCG@5 | {s['ndcg@5']} |",
             f"| Answer accuracy | {pct(s['answer_accuracy'])} |",
             f"| Cites the correct page | {pct(s['cited_correct_page'])} |",
             f"| Verified | {pct(s['verified_rate'])} |", "", "| Question | Hit | Correct | Verdict |", "|---|---|---|---|"]
    for r in report["rows"]:
        lines.append(f"| {r['question']} | {'✅' if r['retrieval_hit'] else '❌'} | "
                     f"{'-' if 'answer_correct' not in r else ('✅' if r['answer_correct'] else '❌')} | {r.get('verdict', '-')} |")
    return "\n".join(lines) + "\n"
