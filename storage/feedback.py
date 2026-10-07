"""
Human feedback store ("saved answers").

Old RAG.py equivalent: `load_feedback()`, `store_feedback()`,
`check_feedback()`. Preserved, with three changes needed for a public
deployment:
  * Saved answers are per document. RAG.py kept one global feedback.json,
    so an answer saved for datasheet A was returned for datasheet B.
  * Matching is exact (after normalising case and spaces). RAG.py matched
    substrings, so a saved question "vcc" overrode every question containing
    "vcc".
  * The feature is off unless ENABLE_FEEDBACK=true. On a public server,
    anyone could otherwise overwrite the answers other people see.
"""
from __future__ import annotations

import re
import threading
import time
from typing import Dict, List, Optional

from storage.cache import DocumentPaths, read_json, write_json_atomic

_lock = threading.Lock()


def _normalize(question: str) -> str:
    return re.sub(r"\s+", " ", (question or "").strip().lower()).rstrip("?. ")


def load_feedback(paths: DocumentPaths) -> List[Dict]:
    data = read_json(paths.feedback_file, [])
    return data if isinstance(data, list) else []


def store_feedback(paths: DocumentPaths, question: str, answer: str) -> None:
    with _lock:
        data = load_feedback(paths)
        data.append({"q": _normalize(question), "a": answer.strip(), "ts": time.time()})
        write_json_atomic(paths.feedback_file, data)


def check_feedback(paths: DocumentPaths, question: str) -> Optional[str]:
    q = _normalize(question)
    if not q:
        return None
    for entry in reversed(load_feedback(paths)):  # newest wins
        if entry.get("q") == q:
            return entry.get("a")
    return None
