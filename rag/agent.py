"""
ReAct agent (Reason + Act) over one datasheet, using Groq tool calling.

    Thought ──► Action (tool call) ──► Observation ──► … ──► Final answer

Design for speed and accuracy on Groq's free tier (8K tokens/min):
  * Retrieval-first: the app's own hybrid retrieval runs BEFORE the first
    LLM call, and its excerpts are the first observation. Most questions
    are then answered in one or two Groq calls instead of four or five.
  * At most AGENT_MAX_STEPS tool rounds; the last round forces an answer.
  * Tool observations are capped (OBSERVATION_CHARS) to keep tokens low.
  * Low reasoning effort on gpt-oss models (faster); the tools supply facts.
  * Arithmetic goes through the `calculate` tool, never mental math.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from config.settings import Settings, get_logger
from rag import llm
from rag.models import AgentStep
from rag.tools import TOOL_SCHEMAS, ToolBox

log = get_logger("agent")

NOT_FOUND = "NOT FOUND IN DOCUMENT"

_ANSWER_FORMAT = f"""Write the final answer in Markdown with these sections:

**Answer:** the direct answer in 1-3 sentences: value, unit, and conditions (min/typ/max, temperature, test conditions), with citations.

**Explanation:** what it means in plain engineering language: what the parameter is, why it matters, the conditions it applies to, and any caveats, with citations.

**Equations:** ONLY if an equation or formula is relevant. Write each one in LaTeX display math on its own line as $$ ... $$, then list what every symbol means with its unit, with citations. If you computed a value, show the substituted equation and the result. Omit this whole section when no equation applies.

Rules:
- Use ONLY facts from the numbered excerpts / tool observations. Never invent values, symbols or equations.
- Cite every factual statement by copying the exact label printed before the excerpt you used (format [REF-n: TYPE pX]). Never make up a label.
- Use $...$ for inline math and $$...$$ for display math (not \\( \\) or \\[ \\]).
- If the datasheet does not contain the answer, reply exactly: {NOT_FOUND}"""

AGENT_SYSTEM_PROMPT = f"""You are an expert electronics engineer answering questions about ONE component datasheet. You work in ReAct style: think about what evidence you need, call a tool, read the observation, repeat, then answer.

Tools:
- search_datasheet(query, source): search again with different keywords, symbols (VOUT, IQ, TJ, RθJA) or synonyms, optionally restricted to text/table/row/equation.
- read_page(page): read a whole page when an excerpt is cut off or a table continues.
- calculate(expression): do ALL arithmetic with this tool.

You already have initial excerpts from the datasheet's hybrid search. If they fully answer the question, answer immediately without calling tools. Otherwise use tools, but be efficient: at most a few calls.

{_ANSWER_FORMAT}"""

QUICK_SYSTEM_PROMPT = f"""You are an expert electronics engineer answering questions about ONE component datasheet using only the numbered excerpts provided.

{_ANSWER_FORMAT}"""


StepFn = Optional[Callable[[AgentStep], None]]


@dataclass
class AgentOutcome:
    answer: str
    steps: List[AgentStep] = field(default_factory=list)
    llm_calls: int = 0
    tokens: int = 0
    on_step: StepFn = None

    def add(self, step: AgentStep) -> None:
        self.steps.append(step)
        if self.on_step:
            try:
                self.on_step(step)
            except Exception:  # a UI problem must never break the agent
                pass


def _user_message(query: str, evidence: str) -> str:
    evidence = evidence or "(No strong matches were found for the question wording.)"
    return f"Question: {query}\n\nInitial excerpts from the datasheet:\n\n{evidence}"


def _assistant_tool_message(result: llm.ChatResult) -> Dict[str, Any]:
    msg: Dict[str, Any] = {
        "role": "assistant",
        "tool_calls": [
            {"id": tc.id, "type": "function",
             "function": {"name": tc.function.name, "arguments": tc.function.arguments or "{}"}}
            for tc in result.tool_calls
        ],
    }
    if result.content:
        msg["content"] = result.content
    return msg


def _short(text: str, limit: int) -> str:
    text = " ".join((text or "").split())
    return text if len(text) <= limit else text[: limit - 1] + "…"


def run_quick(query: str, evidence: str, settings: Settings) -> AgentOutcome:
    """One LLM call with the retrieved excerpts (fewest tokens)."""
    messages = [
        {"role": "system", "content": QUICK_SYSTEM_PROMPT},
        {"role": "user", "content": _user_message(query, evidence)},
    ]
    result = llm.chat(settings, messages)
    steps = [AgentStep("thought", _short(result.reasoning, 500))] if result.reasoning else []
    return AgentOutcome(result.content, steps, llm_calls=1, tokens=result.usage_tokens)


def run_agent(query: str, evidence: str, toolbox: ToolBox, settings: Settings,
              on_step: StepFn = None) -> AgentOutcome:
    """ReAct loop with tool calling. Raises llm.LLMError on Groq failures."""
    messages: List[Dict[str, Any]] = [
        {"role": "system", "content": AGENT_SYSTEM_PROMPT},
        {"role": "user", "content": _user_message(query, evidence)},
    ]
    out = AgentOutcome(answer="", on_step=on_step)

    for step in range(settings.agent_max_steps + 1):
        must_answer = step == settings.agent_max_steps
        try:
            result = llm.chat(settings, messages, tools=TOOL_SCHEMAS,
                              tool_choice="none" if must_answer else "auto")
        except llm.LLMError as exc:
            if exc.kind != "tool_use_failed":
                raise
            # The model emitted a malformed tool call: ask for the answer instead.
            out.add(AgentStep("note", "Invalid tool call; asking for a direct answer."))
            result = llm.chat(settings, messages + [
                {"role": "user", "content": "Answer now using the excerpts you already have."}
            ])
        out.llm_calls += 1
        out.tokens += result.usage_tokens
        if result.reasoning:
            out.add(AgentStep("thought", _short(result.reasoning, 500)))

        if result.tool_calls and not must_answer:
            messages.append(_assistant_tool_message(result))
            for tc in result.tool_calls[:3]:  # cap parallel calls per round
                args, observation = toolbox.execute(tc.function.name, tc.function.arguments)
                out.add(AgentStep("action", "", tool=tc.function.name, args=args))
                out.add(AgentStep("observation", _short(observation, 600), tool=tc.function.name))
                messages.append({"role": "tool", "tool_call_id": tc.id, "content": observation})
            # Tool calls beyond the cap still need a reply so the conversation stays valid.
            for tc in result.tool_calls[3:]:
                messages.append({"role": "tool", "tool_call_id": tc.id,
                                 "content": "Skipped: too many tool calls in one step."})
            continue

        out.answer = result.content
        break

    if out.answer:
        out.add(AgentStep("answer", "Final answer written."))
    log.info("Agent finished: %d LLM calls, %d tokens", out.llm_calls, out.tokens)
    return out
