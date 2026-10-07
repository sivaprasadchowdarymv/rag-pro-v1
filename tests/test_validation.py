"""Fix #3: validation on the user's own documents."""
from llm.model_router import ModelRouter
from metrics.validation import _contains, evaluate, generate_questions, to_markdown


def test_generates_answerable_questions(indexed):
    index, _ = indexed
    qs = generate_questions(index)
    texts = [q["question"] for q in qs]
    assert "What is the maximum output voltage?" in texts
    assert any(q["question"] == "What is the typical peak output current?" and q["expected_answer_contains"] == ["2.2"]
               and q["expected_pages"] == [2] for q in qs)
    assert any(t.startswith("What is the function of pin 2") for t in texts)
    assert "Which equation is used to calculate PD?" in texts
    assert "Which equation is used to calculate VIN?" not in texts  # a conditions line is not a formula


def test_numeric_matching():
    assert _contains("It is 2.20 A", ["2.2"]) and not _contains("It is 22 A", ["2.2"])
    assert _contains("Pin 2 is GND (ground)", ["GND"])


def test_evaluate_retrieval_and_answers(server, indexed):
    index, s = indexed
    qs = generate_questions(index, limit=4)
    rep = evaluate(qs, [index], s, ModelRouter(s), with_answers=True)
    assert rep["summary"]["questions"] == 4 and rep["summary"]["retrieval_hit_rate"] is not None
    assert all("answer_correct" in r and "verdict" in r for r in rep["rows"])
    assert "| Retrieval hit rate |" in to_markdown(rep)
