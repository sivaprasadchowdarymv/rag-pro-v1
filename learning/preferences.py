"""
User preference profile, adapted from explicit feedback.

This is PREFERENCE ADAPTATION, not RLHF: simple, transparent rules update
a small profile that is injected into the master agent's prompt. Every
automatic change is logged so the user can see and undo it.
"""
from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List

from storage.cache import read_json, write_json_atomic

LENGTHS = ("short", "medium", "long")
DEPTHS = ("basic", "medium", "high")


@dataclass
class Preferences:
    response_length: str = "medium"
    technical_depth: str = "high"
    citations: bool = True
    equations: bool = True
    figures: bool = True
    tables: bool = True
    history: List[Dict] = field(default_factory=list)  # automatic changes, newest last

    def to_prompt(self) -> str:
        length = {"short": "Keep answers brief: 2-4 sentences plus essential equations.",
                  "medium": "Use a moderate length.",
                  "long": "Give thorough answers with detailed explanations."}[self.response_length]
        depth = {"basic": "Explain in simple terms for a non-specialist.",
                 "medium": "Assume general engineering knowledge.",
                 "high": "Use precise, expert-level technical language."}[self.technical_depth]
        extra = []
        if self.equations:
            extra.append("Include relevant equations in LaTeX when the evidence contains them.")
        if self.tables:
            extra.append("Present comparisons of several values as a Markdown table.")
        if not self.figures:
            extra.append("Mention figures only if essential.")
        return " ".join([length, depth, *extra])


class PreferenceStore:
    def __init__(self, data_dir: Path):
        self.path = Path(data_dir) / "learning" / "preferences.json"

    def load(self) -> Preferences:
        raw = read_json(self.path, {}) or {}
        return Preferences(**{k: v for k, v in raw.items() if k in Preferences.__dataclass_fields__})

    def save(self, prefs: Preferences) -> None:
        write_json_atomic(self.path, asdict(prefs))

    def apply_feedback(self, rating: int, reasons: List[str]) -> List[str]:
        """Update the profile from one piece of feedback; returns human-readable changes."""
        prefs, changes = self.load(), []

        def shift(attr: str, scale, step: int, why: str) -> None:
            cur = scale.index(getattr(prefs, attr))
            new = scale[max(0, min(len(scale) - 1, cur + step))]
            if new != getattr(prefs, attr):
                changes.append(f"{attr.replace('_', ' ')}: {getattr(prefs, attr)} → {new} ({why})")
                setattr(prefs, attr, new)

        if rating < 0:
            if "too long" in reasons:
                shift("response_length", LENGTHS, -1, "answer was too long")
            if "too short" in reasons or "incomplete" in reasons:
                shift("response_length", LENGTHS, +1, "answer was too short or incomplete")
            if "missing equation" in reasons and not prefs.equations:
                prefs.equations = True
                changes.append("equations: off → on (missing equation)")
            if "missing citation" in reasons and not prefs.citations:
                prefs.citations = True
                changes.append("citations: off → on (missing citation)")
        if changes:
            prefs.history = (prefs.history + [{"ts": time.time(), "changes": changes}])[-50:]
            self.save(prefs)
        return changes
