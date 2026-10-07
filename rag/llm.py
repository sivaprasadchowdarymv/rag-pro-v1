"""
The single place that talks to Groq.

* Uses the official `groq` SDK (OpenAI-compatible chat + tool calling).
* Turns every failure into an LLMError with a short, user-safe message,
  so no stack trace or API key ever reaches the browser.
* The SDK already retries rate-limit (429) errors using Groq's
  `retry-after` header; if it still fails we report it plainly.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any, Dict, List, Optional

import groq

from config.settings import Settings, get_logger

log = get_logger("llm")

# Groq lists non-chat models too (speech, safety classifiers); hide them in the UI.
_NON_CHAT_HINTS = ("whisper", "orpheus", "guard", "tts", "playai", "distil")


class LLMError(Exception):
    """`user_message` is safe to show in the UI."""

    def __init__(self, user_message: str, kind: str = "error"):
        super().__init__(user_message)
        self.user_message = user_message
        self.kind = kind


def _missing_key() -> LLMError:
    return LLMError(
        "No Groq API key is set.\n\nAdd GROQ_API_KEY to the app's secrets "
        "(create a free key at console.groq.com/keys).",
        "no_key",
    )


@lru_cache(maxsize=4)
def _client(api_key: str, base_url: str, timeout: float) -> groq.Groq:
    kwargs: Dict[str, Any] = {"api_key": api_key, "timeout": timeout, "max_retries": 2}
    if base_url:
        kwargs["base_url"] = base_url
    return groq.Groq(**kwargs)


def get_client(settings: Settings) -> groq.Groq:
    if not settings.groq_api_key:
        raise _missing_key()
    return _client(settings.groq_api_key, settings.groq_base_url, settings.llm_timeout)


def translate_error(exc: Exception, model: str) -> LLMError:
    if isinstance(exc, LLMError):
        return exc
    if isinstance(exc, groq.AuthenticationError) or isinstance(exc, groq.PermissionDeniedError):
        return LLMError(
            "Groq rejected the API key.\n\nCreate a new key at console.groq.com/keys "
            "and update GROQ_API_KEY.",
            "auth",
        )
    if isinstance(exc, groq.RateLimitError):
        return LLMError(
            "The free Groq rate limit was reached (requests or tokens per minute/day).\n\n"
            "Wait a minute and ask again, or switch to Quick mode, which uses fewer tokens.",
            "rate_limit",
        )
    if isinstance(exc, groq.NotFoundError):
        return LLMError(
            f"Model '{model}' is not available on Groq.\n\nPick another model in "
            "⚙ Configuration or set GROQ_MODEL (see console.groq.com/docs/models).",
            "model",
        )
    if isinstance(exc, groq.BadRequestError):
        body = str(getattr(exc, "body", "") or exc)
        if "tool_use_failed" in body or "tool call" in body.lower():
            return LLMError("The model produced an invalid tool call.", "tool_use_failed")
        if "context" in body.lower() and "length" in body.lower():
            return LLMError("The request was too long for this model.", "too_long")
        log.warning("Groq rejected a request: %s", body[:300])
        return LLMError("Groq rejected the request.", "bad_request")
    if isinstance(exc, groq.APITimeoutError):
        return LLMError("Groq took too long to respond. Please try again.", "timeout")
    if isinstance(exc, groq.APIConnectionError):
        return LLMError("Cannot reach Groq right now. Check the connection and try again.", "connection")
    if isinstance(exc, groq.APIStatusError):
        log.warning("Groq API error %s", getattr(exc, "status_code", "?"))
        return LLMError("Groq returned an error. Please try again in a moment.", "api")
    log.exception("Unexpected LLM failure")
    return LLMError("Unexpected error while talking to Groq.", "unexpected")


# ---------------------------------------------------------------------------
# Status (for the sidebar)
# ---------------------------------------------------------------------------
@dataclass
class LLMStatus:
    has_key: bool
    connected: bool
    models: List[str] = field(default_factory=list)
    message: str = ""

    def has(self, model: str) -> bool:
        return self.connected and model in self.models


def check_status(settings: Settings) -> LLMStatus:
    if not settings.groq_api_key:
        return LLMStatus(False, False, message=_missing_key().user_message)
    try:
        resp = get_client(settings).models.list()
    except Exception as exc:
        err = translate_error(exc, settings.groq_model)
        return LLMStatus(True, False, message=err.user_message)
    ids = sorted(
        m.id for m in getattr(resp, "data", []) or []
        if getattr(m, "id", None) and not any(h in m.id.lower() for h in _NON_CHAT_HINTS)
    )
    return LLMStatus(True, True, models=ids)


# ---------------------------------------------------------------------------
# Chat
# ---------------------------------------------------------------------------
@dataclass
class ChatResult:
    content: str
    reasoning: str
    tool_calls: List[Any]
    usage_tokens: int
    raw_message: Any = None


def _model_kwargs(model: str) -> Dict[str, Any]:
    """Model-family specific options (only sent where Groq supports them)."""
    if model.startswith("openai/gpt-oss"):
        # Low reasoning effort = faster and fewer tokens; the tools supply the facts.
        return {"reasoning_effort": "low", "include_reasoning": True}
    return {}


def chat(
    settings: Settings,
    messages: List[Dict[str, Any]],
    tools: Optional[List[Dict[str, Any]]] = None,
    tool_choice: str = "auto",
    model: Optional[str] = None,
) -> ChatResult:
    model = model or settings.groq_model
    kwargs: Dict[str, Any] = {
        "model": model,
        "messages": messages,
        "temperature": settings.llm_temperature,
        "max_completion_tokens": settings.max_output_tokens,
        **_model_kwargs(model),
    }
    if tools:
        kwargs["tools"] = tools
        kwargs["tool_choice"] = tool_choice
    try:
        resp = get_client(settings).chat.completions.create(**kwargs)
    except Exception as exc:
        raise translate_error(exc, model) from exc

    msg = resp.choices[0].message
    usage = getattr(resp, "usage", None)
    return ChatResult(
        content=(msg.content or "").strip(),
        reasoning=(getattr(msg, "reasoning", None) or "").strip(),
        tool_calls=list(msg.tool_calls or []),
        usage_tokens=int(getattr(usage, "total_tokens", 0) or 0),
        raw_message=msg,
    )
