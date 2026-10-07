"""Phases 13 + 15: conversations, controlled memory, feedback, preferences, DPO export."""
import json

from agents.rag_pro import answer_pro
from chat.conversations import ConversationStore, result_from_dict
from learning.dpo_dataset import build_pairs, export
from learning.feedback import FeedbackStore
from learning.preferences import PreferenceStore
from llm.model_router import ModelRouter


def test_conversation_roundtrip_and_memory(server, indexed, tmp_path):
    index, s = indexed
    store = ConversationStore(tmp_path)
    conv = store.create([index.doc_id])
    for q in ["What is the peak output current?", "What is the pin configuration?",
              "What is the quiescent current?", "What is the dropout voltage?"]:
        r = answer_pro(q, [index], s, ModelRouter(s), memory=conv.memory(s.memory_turns), previous_question=conv.last_question())
        store.add_exchange(conv, q, r)
    again = store.load(conv.id)
    assert again.title == "What is the peak output current?" and len(again.messages) == 8
    mem = again.memory(3)
    assert mem.count("Q:") == 3 and "Q: What is the peak output current?" not in mem and "[REF-" not in mem  # bounded
    restored = result_from_dict(again.messages[1].result)
    assert restored.sources and restored.sources[0].label.startswith("[REF-")
    assert "# What is the peak output current?" in store.export_markdown(again)


def test_regenerate_keeps_variant(server, indexed, tmp_path):
    index, s = indexed
    store = ConversationStore(tmp_path); conv = store.create()
    r = answer_pro("What is the peak output current?", [index], s, ModelRouter(s))
    msg = store.add_exchange(conv, "What is the peak output current?", r)
    store.replace_answer(conv, msg.id, r)
    assert len(store.load(conv.id).messages[1].variants) == 1


def test_feedback_preferences_and_dpo(tmp_path):
    fb, prefs = FeedbackStore(tmp_path), PreferenceStore(tmp_path)
    fb.record(conversation_id="c", message_id="m1", question="Max current?", answer="Long rambling answer", rating=-1,
              reasons=["too long", "not-a-real-reason"])
    fb.record(conversation_id="c", message_id="m2", question="max current", answer="2.2 A [REF-1: ROW p2]", rating=1)
    st = fb.stats()
    assert st == {"total": 2, "up": 1, "down": 1, "ratio": 0.5, "reasons": {"too long": 1}}
    assert prefs.apply_feedback(-1, ["too long"]) == ["response length: medium → short (answer was too long)"]
    p = prefs.load()
    assert p.response_length == "short" and "brief" in p.to_prompt() and p.history
    pairs = build_pairs(fb.all())
    assert pairs == [{"prompt": "max current", "chosen": "2.2 A [REF-1: ROW p2]",
                      "rejected": "Long rambling answer", "feedback_reason": "too long"}]
    n = export(fb.path, tmp_path / "dpo.jsonl")
    assert n == 1 and json.loads((tmp_path / "dpo.jsonl").read_text().splitlines()[0])["chosen"].startswith("2.2")
