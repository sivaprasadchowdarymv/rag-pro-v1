"""Phase 18: benchmark harness."""
import json
from pathlib import Path

from metrics.benchmark import retrieval_scores, run, to_markdown


def test_retrieval_scores():
    s = retrieval_scores([3, 2, 2, 1], [2], k=3)
    assert s == {"precision@k": 2 / 3, "recall@k": 1.0, "hit": 1.0, "mrr": 0.5}
    assert retrieval_scores([1], [4], 5)["mrr"] == 0.0


def test_benchmark_runs(sample_pdf, make_settings):
    qs = json.loads((Path(__file__).parent.parent / "benchmark" / "sample_questions.json").read_text())
    rep = run(sample_pdf, "LM7805X.pdf", qs, make_settings())
    assert rep["questions"] == 8 and 0 <= rep["summary"]["pro"]["mrr"] <= 1
    assert "| MRR |" in to_markdown(rep)
