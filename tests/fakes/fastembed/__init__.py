"""Offline stand-in for fastembed used ONLY by the test suite (deterministic hashing)."""
import hashlib
import numpy as np


def _vec(text, dim=768):
    v = np.zeros(dim, np.float32)
    for w in text.lower().replace("search_query: ", "").replace("search_document: ", "").split():
        v[int(hashlib.md5(w.encode()).hexdigest(), 16) % dim] += 1
    return v


class TextEmbedding:
    def __init__(self, model_name="", cache_dir=None, **kw):
        self.model_name = model_name

    def embed(self, documents, batch_size=256, **kw):
        docs = [documents] if isinstance(documents, str) else list(documents)
        for d in docs:
            yield _vec(d)
