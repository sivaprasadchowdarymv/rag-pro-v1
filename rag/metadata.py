"""
Electrical-parameter metadata extraction.

Old RAG.py equivalent: `extract_metadata()` — logic unchanged, patterns are
now compiled once instead of on every call.
"""
from __future__ import annotations

import re
from typing import Any, Dict

_PATTERNS = {
    "voltage": re.compile(r"V[A-Z]{1,4}\s*=\s*[\d\.\-\+]+\s*V", re.IGNORECASE),
    "temperature": re.compile(r"T[A-Z]{1,3}\s*=\s*[\d\.\-\+]+\s*°?C", re.IGNORECASE),
    "current": re.compile(
        r"I[A-Z]{1,4}\s*=\s*[\d\.\-\+]+\s*(?:mA|µA|uA|A|nA)", re.IGNORECASE
    ),
    "frequency": re.compile(r"f\s*=\s*[\d\.]+\s*(?:MHz|kHz|GHz|Hz)", re.IGNORECASE),
    "power": re.compile(r"P[A-Z]{0,3}\s*=\s*[\d\.]+\s*(?:mW|W|µW|uW)", re.IGNORECASE),
    "resistance": re.compile(
        r"R[A-Z]{0,3}\s*=\s*[\d\.]+\s*(?:kΩ|MΩ|Ω|ohm)", re.IGNORECASE
    ),
    "capacitance": re.compile(
        r"C[A-Z]{0,3}\s*=\s*[\d\.]+\s*(?:pF|nF|µF|uF|mF|F)", re.IGNORECASE
    ),
}
_PART_NUMBER = re.compile(r"\b[A-Z]{2,5}\d{2,8}[A-Z0-9\-]*\b")


def extract_metadata(text: str) -> Dict[str, Any]:
    """Return e.g. {"voltage": ["VCC = 3.3V"], "part_numbers": ["LM358"]}."""
    meta: Dict[str, Any] = {}
    text = text or ""
    for key, pattern in _PATTERNS.items():
        found = pattern.findall(text)
        if found:
            meta[key] = list(dict.fromkeys(found))[:8]  # de-duplicated
    part_numbers = _PART_NUMBER.findall(text)
    if part_numbers:
        meta["part_numbers"] = list(dict.fromkeys(part_numbers))[:10]
    return meta
