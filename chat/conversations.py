"""
Persistent multi-conversation chat with controlled memory.

Conversations are JSON files under data/conversations/. Only the last
MEMORY_TURNS exchanges (questions + shortened answers, citations removed)
are given to the model: memory stays bounded however long the chat gets.
"""
from __future__ import annotations

import re
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from rag.models import AgentStep, QueryResult, SourceRef
from storage.cache import read_json, write_json_atomic

_CITE = re.compile(r"\[REF-\d+[^\]]*\]")
_ID = re.compile(r"^[0-9a-f]{12}$")


@dataclass
class ChatMessage:
    id: str
    role: str  # user | assistant
    content: str
    created: float
    result: Optional[Dict[str, Any]] = None  # serialized QueryResult (assistant)
    feedback: Optional[Dict[str, Any]] = None  # {"rating": 1|-1, "reasons": [...], "comment": ""}
    variants: List[Dict[str, Any]] = field(default_factory=list)  # earlier answers replaced by "regenerate"


@dataclass
class Conversation:
    id: str
    title: str
    created: float
    updated: float
    doc_ids: List[str] = field(default_factory=list)
    messages: List[ChatMessage] = field(default_factory=list)

    def last_question(self) -> Optional[str]:
        return next((m.content for m in reversed(self.messages) if m.role == "user"), None)

    def memory(self, turns: int, chars: int = 500) -> str:
        """Last `turns` Q&A pairs, answers shortened and citations stripped."""
        pairs, q = [], None
        for m in self.messages:
            if m.role == "user":
                q = m.content
            elif q is not None:
                pairs.append((q, _CITE.sub("", m.content).strip()[:chars]))
                q = None
        return "\n".join(f"Q: {a}\nA: {b}" for a, b in pairs[-turns:]) if turns > 0 else ""


def result_to_dict(r: QueryResult) -> Dict[str, Any]:
    return asdict(r)


def result_from_dict(d: Dict[str, Any]) -> QueryResult:
    d = dict(d)
    d["sources"] = [SourceRef(**s) for s in d.get("sources", [])]
    d["steps"] = [AgentStep(**s) for s in d.get("steps", [])]
    return QueryResult(**{k: v for k, v in d.items() if k in QueryResult.__dataclass_fields__})


class ConversationStore:
    def __init__(self, data_dir: Path):
        self.dir = Path(data_dir) / "conversations"
        self.dir.mkdir(parents=True, exist_ok=True)

    def _path(self, cid: str) -> Path:
        if not _ID.match(cid):
            raise ValueError("invalid conversation id")
        return self.dir / f"{cid}.json"

    def create(self, doc_ids: Optional[List[str]] = None) -> Conversation:
        now = time.time()
        conv = Conversation(uuid.uuid4().hex[:12], "New chat", now, now, list(doc_ids or []))
        self.save(conv)
        return conv

    def save(self, conv: Conversation) -> None:
        conv.updated = time.time()
        write_json_atomic(self._path(conv.id), asdict(conv))

    def load(self, cid: str) -> Optional[Conversation]:
        raw = read_json(self._path(cid))
        if not raw:
            return None
        raw["messages"] = [ChatMessage(**m) for m in raw.get("messages", [])]
        return Conversation(**raw)

    def list(self) -> List[Conversation]:
        convs = [c for c in (self.load(p.stem) for p in self.dir.glob("*.json") if _ID.match(p.stem)) if c]
        return sorted(convs, key=lambda c: c.updated, reverse=True)

    def delete(self, cid: str) -> None:
        self._path(cid).unlink(missing_ok=True)

    def add_exchange(self, conv: Conversation, question: str, result: QueryResult) -> ChatMessage:
        now = time.time()
        conv.messages.append(ChatMessage(uuid.uuid4().hex[:12], "user", question, now))
        reply = ChatMessage(uuid.uuid4().hex[:12], "assistant", result.answer, now, result_to_dict(result))
        conv.messages.append(reply)
        if conv.title == "New chat":
            conv.title = (question[:48] + "…") if len(question) > 48 else question
        self.save(conv)
        return reply

    def replace_answer(self, conv: Conversation, message_id: str, result: QueryResult) -> None:
        """Regenerate: keep the previous answer as a variant (useful for DPO pairs)."""
        for m in conv.messages:
            if m.id == message_id and m.role == "assistant":
                m.variants.append({"content": m.content, "result": m.result, "feedback": m.feedback})
                m.content, m.result, m.feedback = result.answer, result_to_dict(result), None
        self.save(conv)

    @staticmethod
    def export_markdown(conv: Conversation) -> str:
        lines = [f"# {conv.title}", ""]
        for m in conv.messages:
            lines += [f"**{'You' if m.role == 'user' else 'Assistant'}:**", "", m.content, ""]
            if m.role == "assistant" and m.result and m.result.get("sources"):
                lines.append("Sources: " + "; ".join(
                    f"{s['label']} {s.get('doc_name', '')} – {s['section']}" for s in m.result["sources"]))
                lines.append("")
        return "\n".join(lines)
