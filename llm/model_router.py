"""
Role-based model routing with a free-first fallback chain.

Roles:
    master    final reasoning and synthesis (largest allowed model)
    fast      light tasks (optional LLM verification, titles)
    verifier  grounding check when VERIFY_WITH_LLM=true
    vision    figures (only if a vision model is configured)

Free-first / zero-billing-safety rules:
  * Only providers listed in LLM_PROVIDERS **and** configured (key present, or
    ENABLE_OLLAMA_LOCAL=true) are ever called. Nothing is enabled implicitly.
  * Fallback walks that list in order, only for recoverable errors (rate limit,
    quota, unavailable model, network). A "needs a paid plan" error (HTTP 402)
    is never retried on the same provider.
  * Each answer records which provider/model produced it, so usage is visible.
  The app cannot see your billing plan. Use free-plan keys and no paid model
  can be called (see DEPLOYMENT.md).
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from config.settings import Settings, _lookup, get_logger
from llm.providers import (FALLBACK_KINDS, ChatResult, GroqProvider, LLMError, OllamaProvider,
                           Provider)

log = get_logger("router")

ROLES = ("master", "fast", "verifier", "vision")

DEFAULT_MODELS: Dict[str, Dict[str, str]] = {
    "groq": {"master": "openai/gpt-oss-120b", "fast": "openai/gpt-oss-20b",
             "verifier": "openai/gpt-oss-20b", "vision": ""},
    "ollama_cloud": {"master": "gpt-oss:120b", "fast": "gpt-oss:20b",
                     "verifier": "gpt-oss:20b", "vision": ""},
    "ollama_local": {"master": "mistral", "fast": "mistral", "verifier": "mistral", "vision": "llava:7b"},
}


@dataclass
class ProviderStatus:
    name: str
    label: str
    configured: bool
    connected: bool = False
    models: List[str] = field(default_factory=list)
    message: str = ""


@dataclass
class RoutedResult:
    result: ChatResult
    attempts: List[Tuple[str, str, str]]  # (provider, model, outcome)
    latency_ms: float


class ModelRouter:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.providers: List[Provider] = []
        for name in settings.llm_providers:
            p = self._build(name)
            if p is not None and p.configured():
                self.providers.append(p)

    def _build(self, name: str) -> Optional[Provider]:
        s = self.settings
        if name == "groq":
            return GroqProvider(s.groq_api_key, s.groq_base_url, s.llm_timeout)
        if name == "ollama_cloud":
            return OllamaProvider("ollama_cloud", s.ollama_cloud_host, s.ollama_api_key, enabled=True,
                                  timeout=max(s.llm_timeout, 60))
        if name == "ollama_local":
            return OllamaProvider("ollama_local", s.ollama_local_host, "", enabled=s.enable_ollama_local,
                                  timeout=max(s.llm_timeout, 300))
        return None

    # ------------------------------------------------------------------ models
    def model_for(self, provider: Provider, role: str) -> str:
        """Per-provider override > generic override (first provider only) > default."""
        specific = (_lookup(f"{provider.name.upper()}_{role.upper()}_MODEL") or "").strip()
        if specific:
            return specific
        generic = getattr(self.settings, f"{role}_model", "") if role != "master" else self.settings.master_model
        if generic and self.providers and provider is self.providers[0]:
            return generic
        if role == "master" and provider.name == "groq" and self.settings.groq_model:
            return self.settings.groq_model  # backward compatible GROQ_MODEL
        return DEFAULT_MODELS.get(provider.name, {}).get(role, "")

    def chain(self, role: str) -> List[Tuple[Provider, str]]:
        return [(p, m) for p in self.providers if (m := self.model_for(p, role))]

    def has_role(self, role: str) -> bool:
        return bool(self.chain(role))

    @property
    def enabled(self) -> bool:
        return bool(self.providers)

    # -------------------------------------------------------------------- chat
    def chat(self, role: str, messages: List[Dict[str, Any]], tools: Optional[List[Dict]] = None,
             tool_choice: str = "auto", images: Optional[List[bytes]] = None,
             max_tokens: Optional[int] = None) -> RoutedResult:
        t0 = time.perf_counter()
        attempts: List[Tuple[str, str, str]] = []
        chain = self.chain(role)
        if not chain:
            raise LLMError(
                "No AI provider is configured. Add GROQ_API_KEY or OLLAMA_API_KEY to the app's secrets.",
                "no_provider")
        last: Optional[LLMError] = None
        for provider, model in chain:
            try:
                result = provider.chat(model, messages, tools=tools, tool_choice=tool_choice,
                                       temperature=self.settings.llm_temperature,
                                       max_tokens=max_tokens or self.settings.max_output_tokens, images=images)
                attempts.append((provider.name, model, "ok"))
                return RoutedResult(result, attempts, (time.perf_counter() - t0) * 1000)
            except LLMError as exc:
                attempts.append((provider.name, model, exc.kind))
                last = exc
                if exc.kind not in FALLBACK_KINDS:
                    raise
                log.warning("%s/%s failed (%s); trying next provider", provider.name, model, exc.kind)
        assert last is not None
        last.user_message += " No other enabled provider could answer."
        raise last

    # ------------------------------------------------------------------ status
    def status(self) -> List[ProviderStatus]:
        out: List[ProviderStatus] = []
        for name in self.settings.llm_providers:
            p = self._build(name)
            if p is None:
                continue
            st = ProviderStatus(p.name, p.label, p.configured())
            if st.configured:
                try:
                    st.models, st.connected = p.list_models(), True
                except LLMError as exc:
                    st.message = exc.user_message
            out.append(st)
        return out
