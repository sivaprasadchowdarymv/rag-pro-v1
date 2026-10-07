"""The previous pipeline must keep working unchanged (LEGACY_MODE)."""
from rag.pipeline import answer_query


def test_index_contents(indexed):
    index, _ = indexed
    s = index.stats()
    assert s["pages"] == 4 and s["text"] > 0 and s["table"] == 2 and s["row_pin"] >= 6 and s["figure"] == 2
    assert index.missing_embeddings == 0


def test_legacy_agent_answer(server, indexed):
    index, settings = indexed
    r = answer_query("What is the peak output current?", index, settings)
    assert r.kind == "answer" and r.used_llm and r.sources
    assert all(src.label in r.answer for src in r.sources)  # canonical citations


def test_legacy_quick_and_not_found(server, indexed):
    index, settings = indexed
    assert answer_query("What is the peak output current?", index, settings, mode="quick").llm_calls == 1
    assert answer_query("nothing relevant here", index, settings).kind == "not_found"
