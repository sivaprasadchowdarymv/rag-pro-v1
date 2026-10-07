"""Phase 2: provider layer, role routing, free-first fallback."""
import pytest

from llm.model_router import ModelRouter
from llm.providers import LLMError
from tests.conftest import control, server_log

MSG = [{"role": "user", "content": "Question: What is the peak output current?\n\n[REF-1: TEXT p2] peak 2.2 A"}]


def test_only_configured_providers_are_used(make_settings):
    s = make_settings(llm_providers=("groq", "ollama_cloud", "ollama_local"), ollama_api_key="",
                      enable_ollama_local=False)
    assert [p.name for p in ModelRouter(s).providers] == ["groq"]  # no key / not enabled => never called


def test_no_provider_gives_clear_error(make_settings):
    with pytest.raises(LLMError) as err:
        ModelRouter(make_settings(groq_api_key="")).chat("master", MSG)
    assert err.value.kind == "no_provider"


def test_role_models(make_settings):
    r = ModelRouter(make_settings(llm_providers=("groq", "ollama_cloud"), ollama_api_key="ollama_test"))
    assert [(p.name, m) for p, m in r.chain("master")] == [("groq", "openai/gpt-oss-120b"), ("ollama_cloud", "gpt-oss:120b")]
    assert r.chain("fast")[0][1] == "openai/gpt-oss-20b"
    assert not r.has_role("vision")  # no free vision model configured by default


def test_groq_chat(server, make_settings):
    out = ModelRouter(make_settings()).chat("master", MSG)
    assert "2.2 A" in out.result.content and out.result.provider == "groq" and out.result.tokens > 0


def test_fallback_groq_to_ollama_cloud(server, make_settings):
    control(server, "groq")  # Groq returns 429
    r = ModelRouter(make_settings(llm_providers=("groq", "ollama_cloud"), ollama_api_key="ollama_test"))
    out = r.chat("master", MSG)
    assert out.result.provider == "ollama_cloud"
    assert [a[2] for a in out.attempts] == ["rate_limit", "ok"]


def test_all_providers_failing_raises(server, make_settings):
    control(server, "groq")
    with pytest.raises(LLMError) as err:
        ModelRouter(make_settings()).chat("master", MSG)
    assert err.value.kind == "rate_limit"


def test_wrong_key_is_reported_not_leaked(server, make_settings):
    with pytest.raises(LLMError) as err:
        ModelRouter(make_settings(groq_api_key="gsk_WRONG")).chat("master", MSG)
    assert err.value.kind == "auth" and "gsk_WRONG" not in err.value.user_message


def test_ollama_tool_calling_roundtrip(server, make_settings):
    r = ModelRouter(make_settings(llm_providers=("ollama_cloud",), ollama_api_key="ollama_test"))
    q = [{"role": "user", "content": "Question: calculate the dissipation\n\n[REF-1: TEXT p3] PD"}]
    from rag.tools import TOOL_SCHEMAS
    first = r.chat("master", q, tools=TOOL_SCHEMAS).result
    assert [c.name for c in first.tool_calls] == ["search_datasheet", "calculate"]
    msgs = q + [{"role": "assistant", "tool_calls": [{"id": c.id, "type": "function",
                 "function": {"name": c.name, "arguments": c.arguments}} for c in first.tool_calls]}]
    msgs += [{"role": "tool", "tool_call_id": c.id, "content": "obs [REF-1: TEXT p3]"} for c in first.tool_calls]
    second = r.chat("master", msgs, tools=TOOL_SCHEMAS).result
    assert "P_D" in second.content
    assert server_log(server)["errors"] == []  # tool_name set, no tool_choice sent to Ollama


def test_status_lists_chat_models_only(server, make_settings):
    st = ModelRouter(make_settings()).status()[0]
    assert st.connected and "openai/gpt-oss-120b" in st.models and not any("whisper" in m for m in st.models)
