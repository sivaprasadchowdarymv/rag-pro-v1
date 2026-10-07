"""
LLM providers behind one small interface.

    GroqProvider          Groq API (OpenAI-compatible, tool calling)
    OllamaProvider        Ollama Cloud (https://ollama.com + API key) or a local Ollama server

Every provider returns the same ChatResult and raises LLMError with a
user-safe message. The model's hidden reasoning is never stored or returned:
the UI shows actions, not chain-of-thought.

Messages and tools use the OpenAI format everywhere; OllamaProvider converts.
"""
from __future__ import annotations

import json
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any, Dict, List, Optional

from config.settings import get_logger

log = get_logger("providers")

# Errors after which the router may try the next enabled provider.
FALLBACK_KINDS = {"rate_limit", "connection", "timeout", "model", "api", "auth", "quota"}


class LLMError(Exception):
    """`user_message` is safe to show in the UI (never contains keys)."""

    def __init__(self, user_message: str, kind: str = "error", provider: str = ""):
        super().__init__(user_message)
        self.user_message = user_message
        self.kind = kind
        self.provider = provider


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: str  # JSON text


@dataclass
class ChatResult:
    content: str
    tool_calls: List[ToolCall] = field(default_factory=list)
    tokens: int = 0
    provider: str = ""
    model: str = ""


class Provider(ABC):
    name: str = "provider"
    label: str = "Provider"
    supports_tool_choice: bool = True

    @abstractmethod
    def configured(self) -> bool: ...

    @abstractmethod
    def list_models(self) -> List[str]: ...

    @abstractmethod
    def chat(self, model: str, messages: List[Dict[str, Any]], tools: Optional[List[Dict]] = None,
             tool_choice: str = "auto", temperature: float = 0.1, max_tokens: int = 1500,
             images: Optional[List[bytes]] = None) -> ChatResult: ...


# =============================================================================
# Groq
# =============================================================================
_NON_CHAT_HINTS = ("whisper", "orpheus", "guard", "tts", "playai", "distil")


@lru_cache(maxsize=4)
def _groq_client(api_key: str, base_url: str, timeout: float):
    import groq

    kwargs: Dict[str, Any] = {"api_key": api_key, "timeout": timeout, "max_retries": 2}
    if base_url:
        kwargs["base_url"] = base_url
    return groq.Groq(**kwargs)


class GroqProvider(Provider):
    name, label = "groq", "Groq"

    def __init__(self, api_key: str, base_url: str = "", timeout: float = 60.0):
        self.api_key, self.base_url, self.timeout = api_key, base_url, timeout

    def configured(self) -> bool:
        return bool(self.api_key)

    def _client(self):
        if not self.api_key:
            raise LLMError("No Groq API key is set (GROQ_API_KEY).", "auth", self.name)
        return _groq_client(self.api_key, self.base_url, self.timeout)

    def _translate(self, exc: Exception, model: str) -> LLMError:
        import groq

        if isinstance(exc, LLMError):
            return exc
        if isinstance(exc, (groq.AuthenticationError, groq.PermissionDeniedError)):
            return LLMError("Groq rejected the API key. Create a new one at console.groq.com/keys.", "auth", self.name)
        if isinstance(exc, groq.RateLimitError):
            return LLMError("Groq's free rate limit was reached. Wait a minute or use Quick mode.", "rate_limit", self.name)
        if isinstance(exc, groq.NotFoundError):
            return LLMError(f"Model '{model}' is not available on Groq.", "model", self.name)
        if isinstance(exc, groq.BadRequestError):
            body = str(getattr(exc, "body", "") or exc)
            if "tool_use_failed" in body or "tool call" in body.lower():
                return LLMError("The model produced an invalid tool call.", "tool_use_failed", self.name)
            log.warning("Groq rejected a request: %s", body[:300])
            return LLMError("Groq rejected the request.", "bad_request", self.name)
        if isinstance(exc, groq.APITimeoutError):
            return LLMError("Groq took too long to respond.", "timeout", self.name)
        if isinstance(exc, groq.APIConnectionError):
            return LLMError("Cannot reach Groq right now.", "connection", self.name)
        if isinstance(exc, groq.APIStatusError):
            return LLMError("Groq returned an error.", "api", self.name)
        log.exception("Unexpected Groq failure")
        return LLMError("Unexpected error while talking to Groq.", "api", self.name)

    def list_models(self) -> List[str]:
        try:
            resp = self._client().models.list()
        except Exception as exc:
            raise self._translate(exc, "") from exc
        return sorted(m.id for m in (getattr(resp, "data", None) or [])
                      if not any(h in m.id.lower() for h in _NON_CHAT_HINTS))

    def chat(self, model, messages, tools=None, tool_choice="auto", temperature=0.1,
             max_tokens=1500, images=None) -> ChatResult:
        if images:  # attach images to the last user message (OpenAI vision format)
            import base64

            messages = [dict(m) for m in messages]
            last = messages[-1]
            parts = [{"type": "text", "text": last.get("content", "")}]
            parts += [{"type": "image_url", "image_url": {"url": "data:image/png;base64,"
                       + base64.b64encode(b).decode()}} for b in images]
            last["content"] = parts
        kwargs: Dict[str, Any] = {"model": model, "messages": messages, "temperature": temperature,
                                  "max_completion_tokens": max_tokens}
        if model.startswith("openai/gpt-oss"):
            kwargs.update(reasoning_effort="low", include_reasoning=False)  # hidden reasoning not returned
        if tools:
            kwargs.update(tools=tools, tool_choice=tool_choice)
        try:
            resp = self._client().chat.completions.create(**kwargs)
        except Exception as exc:
            raise self._translate(exc, model) from exc
        msg = resp.choices[0].message
        calls = [ToolCall(tc.id, tc.function.name, tc.function.arguments or "{}") for tc in (msg.tool_calls or [])]
        usage = getattr(resp, "usage", None)
        return ChatResult((msg.content or "").strip(), calls, int(getattr(usage, "total_tokens", 0) or 0),
                          self.name, model)


# =============================================================================
# Ollama (cloud or local)
# =============================================================================
@lru_cache(maxsize=4)
def _ollama_client(host: str, api_key: str, timeout: float):
    import ollama

    headers = {"Authorization": f"Bearer {api_key}"} if api_key else None
    return ollama.Client(host=host, timeout=timeout, headers=headers)


class OllamaProvider(Provider):
    supports_tool_choice = False  # Ollama has no tool_choice; "none" = send no tools

    def __init__(self, name: str, host: str, api_key: str = "", enabled: bool = True, timeout: float = 120.0):
        self.name = name
        self.label = "Ollama Cloud" if name == "ollama_cloud" else "Ollama (local)"
        self.host, self.api_key, self.enabled, self.timeout = host, api_key, enabled, timeout

    def configured(self) -> bool:
        return self.enabled and (bool(self.api_key) if self.name == "ollama_cloud" else True)

    def _client(self):
        return _ollama_client(self.host, self.api_key, self.timeout)

    def _translate(self, exc: Exception, model: str) -> LLMError:
        import httpx
        import ollama

        if isinstance(exc, LLMError):
            return exc
        if isinstance(exc, ollama.ResponseError):
            code = exc.status_code
            if code in (401, 403):
                return LLMError(f"{self.label} rejected the API key.", "auth", self.name)
            if code == 402:
                return LLMError(f"'{model}' needs a paid {self.label} plan; it was not used.", "quota", self.name)
            if code == 429:
                return LLMError(f"{self.label} usage limit reached.", "rate_limit", self.name)
            if code == 404 or "not found" in str(exc).lower():
                return LLMError(f"Model '{model}' is not available on {self.label}.", "model", self.name)
            return LLMError(f"{self.label} returned an error.", "api", self.name)
        if isinstance(exc, httpx.TimeoutException):
            return LLMError(f"{self.label} took too long to respond.", "timeout", self.name)
        if isinstance(exc, (ConnectionError, httpx.HTTPError, OSError)):
            return LLMError(f"Cannot reach {self.label}.", "connection", self.name)
        log.exception("Unexpected Ollama failure")
        return LLMError(f"Unexpected error while talking to {self.label}.", "api", self.name)

    def list_models(self) -> List[str]:
        try:
            resp = self._client().list()
        except Exception as exc:
            raise self._translate(exc, "") from exc
        return sorted(str(getattr(m, "model", "") or "") for m in (getattr(resp, "models", None) or []) if getattr(m, "model", ""))

    @staticmethod
    def _convert(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        names: Dict[str, str] = {}
        out: List[Dict[str, Any]] = []
        for m in messages:
            if m["role"] == "assistant" and m.get("tool_calls"):
                calls = []
                for tc in m["tool_calls"]:
                    names[tc["id"]] = tc["function"]["name"]
                    try:
                        args = json.loads(tc["function"].get("arguments") or "{}")
                    except ValueError:
                        args = {}
                    calls.append({"function": {"name": tc["function"]["name"], "arguments": args}})
                out.append({"role": "assistant", "content": m.get("content") or "", "tool_calls": calls})
            elif m["role"] == "tool":
                out.append({"role": "tool", "content": m["content"], "tool_name": names.get(m.get("tool_call_id", ""), "")})
            else:
                out.append({"role": m["role"], "content": m.get("content") or ""})
        return out

    def chat(self, model, messages, tools=None, tool_choice="auto", temperature=0.1,
             max_tokens=1500, images=None) -> ChatResult:
        msgs = self._convert(messages)
        if images:
            msgs[-1]["images"] = images
        kwargs: Dict[str, Any] = {"model": model, "messages": msgs,
                                  "options": {"temperature": temperature, "num_predict": max_tokens}}
        if self.name == "ollama_local":
            kwargs["options"]["num_ctx"] = 8192
        if tools and tool_choice != "none":
            kwargs["tools"] = tools
        if "gpt-oss" in model:
            kwargs["think"] = "low"
        try:
            resp = self._client().chat(**kwargs)
        except Exception as exc:
            raise self._translate(exc, model) from exc
        msg = resp.message
        calls = [ToolCall(f"call_{uuid.uuid4().hex[:10]}", tc.function.name, json.dumps(tc.function.arguments or {}))
                 for tc in (msg.tool_calls or [])]
        tokens = int(getattr(resp, "prompt_eval_count", 0) or 0) + int(getattr(resp, "eval_count", 0) or 0)
        return ChatResult((msg.content or "").strip(), calls, tokens, self.name, model)
