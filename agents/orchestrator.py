"""
Query orchestrator: classify intents and pick ONLY the agents needed.

Deterministic (regex rules, ~1 ms, no LLM call): routing is the cheapest
place to save time and tokens, and rules are predictable and testable.
Follow-up questions ("and at 85 °C?") are rewritten with the previous
question so retrieval keeps the context.
"""
from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from typing import List, Optional

_RULES = {
    "equation": r"\b(equation|formula|derive|derivation|expression|variable|symbol|relationship between)\b",
    "calculation": r"\b(calculate|compute|how much|estimate|what (?:is|will be) the .* (?:at|if|when|for)|\d+(?:\.\d+)?\s*(?:v|a|ma|w|mw|°c|c|ohm|ω|hz|khz|mhz)\b)",
    "figure": r"\b(figure|fig\.?\s*\d+|graph|plot|curve|chart|diagram|waveform|block diagram|image|picture|schematic)\b",
    "table": r"\b(table|pin|pinout|pins|row|column|min(?:imum)?|max(?:imum)?|typ(?:ical)?|rating|ratings|specification|spec|characteristics|current|voltage|temperature|power|frequency|resistance|capacitance)\b",
    "metadata": r"\b(title|author|document|datasheet|manufacturer|sections?|pages?|contents|what is this|which part)\b",
    "summarization": r"\b(summar\w*|overview|key features|main features|describe the (?:device|part|document))\b",
    "comparison": r"\b(compare|comparison|difference|differences|versus|vs\.|better|between .* and)\b",
}
_FOLLOW_UP = re.compile(r"^(and|what about|how about|also|then|same|but)\b", re.I)
_PRONOUN = re.compile(r"\b(it|its|that|this|those|these|they|them|the same)\b", re.I)

AGENTS = ("text_agent", "table_agent", "equation_agent", "figure_agent", "document_agent")


@dataclass
class Plan:
    intents: List[str]
    agents: List[str]
    use_tools: bool  # True = ReAct loop with tools; False = one master call
    requires_master_reasoning: bool
    retrieval_query: str
    skipped_agents: List[str] = field(default_factory=list)
    latency_ms: float = 0.0
    is_follow_up: bool = False

    def as_dict(self) -> dict:
        return {"intents": self.intents, "agents": self.agents, "use_tools": self.use_tools,
                "requires_master_reasoning": self.requires_master_reasoning}


def classify(question: str) -> List[str]:
    q = question.lower()
    intents = [name for name, pattern in _RULES.items() if re.search(pattern, q)]
    if not intents or not set(intents) <= {"metadata", "summarization"}:
        intents.insert(0, "text")
    return list(dict.fromkeys(intents))


def plan(question: str, previous_question: Optional[str] = None, n_documents: int = 1,
         agent_mode: str = "agent") -> Plan:
    t0 = time.perf_counter()
    question = question.strip()
    # Follow-up only on explicit cues: an opener ("and…", "what about…") or a pronoun in a short question.
    follow_up = bool(previous_question) and (bool(_FOLLOW_UP.match(question))
                                             or (len(question.split()) <= 10 and bool(_PRONOUN.search(question))))
    retrieval_query = f"{previous_question} {question}" if follow_up else question
    intents = classify(retrieval_query)
    if n_documents > 1 and re.search(r"\b(both|each|all|documents|datasheets|compare)\b", question.lower()):
        intents.append("multi_document")

    agents = []
    if "text" in intents or {"comparison", "summarization"} & set(intents):
        agents.append("text_agent")
    if {"table", "comparison", "calculation"} & set(intents):
        agents.append("table_agent")
    if {"equation", "calculation"} & set(intents):
        agents.append("equation_agent")
    if "figure" in intents:
        agents.append("figure_agent")
    if {"metadata", "summarization", "multi_document"} & set(intents):
        agents.append("document_agent")
    if not agents:
        agents = ["text_agent"]

    use_tools = agent_mode == "agent" and bool({"calculation", "comparison", "multi_document", "equation"} & set(intents))
    return Plan(
        intents=intents, agents=agents, use_tools=use_tools, requires_master_reasoning=True,
        retrieval_query=retrieval_query, skipped_agents=[a for a in AGENTS if a not in agents],
        latency_ms=(time.perf_counter() - t0) * 1000, is_follow_up=follow_up,
    )
