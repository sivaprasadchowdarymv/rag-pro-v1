"""Offline stand-in cross-encoder: score = word overlap (tests only)."""
import re

_STOP = {"what", "is", "the", "a", "an", "of", "in", "to", "and", "or", "for", "on", "at", "how", "do", "i", "explain"}


class TextCrossEncoder:
    def __init__(self, model_name="", cache_dir=None, **kw):
        self.model_name = model_name

    def rerank(self, query, documents, batch_size=64, **kw):
        q = set(re.findall(r"[a-z0-9]+", query.lower())) - _STOP
        for d in documents:
            words = set(re.findall(r"[a-z0-9]+", d.lower()))
            yield float(len(q & words)) - 0.001 * len(words)
