"""
Lightweight BM25 (Okapi) lexical index, pure Python/NumPy (no new dependency).

Complements the existing fuzzy matcher: BM25 rewards exact, rare terms such as
symbols (RthJA, VDO), pin names and part numbers, which fuzzy token-set
matching can rank below vaguer chunks.
"""
from __future__ import annotations

import math
import re
from collections import Counter
from typing import Dict, List, Sequence

import numpy as np

_TOKEN = re.compile(r"[a-z0-9µΩ°]+(?:\.[0-9]+)?")


def tokenize(text: str) -> List[str]:
    return _TOKEN.findall((text or "").lower())


class BM25Index:
    def __init__(self, docs: Sequence[str], k1: float = 1.4, b: float = 0.75):
        self.k1, self.b = k1, b
        self.tfs: List[Counter] = [Counter(tokenize(d)) for d in docs]
        self.lengths = np.array([sum(tf.values()) for tf in self.tfs], dtype=np.float32)
        self.avg_len = float(self.lengths.mean()) if len(self.lengths) else 0.0
        df: Counter = Counter()
        for tf in self.tfs:
            df.update(tf.keys())
        n = len(self.tfs)
        self.idf: Dict[str, float] = {t: math.log(1 + (n - c + 0.5) / (c + 0.5)) for t, c in df.items()}

    def scores(self, query: str, candidates: Sequence[int]) -> np.ndarray:
        terms = [t for t in set(tokenize(query)) if t in self.idf]
        out = np.zeros(len(candidates), dtype=np.float32)
        if not terms or self.avg_len == 0:
            return out
        for j, i in enumerate(candidates):
            tf, length = self.tfs[i], self.lengths[i]
            norm = self.k1 * (1 - self.b + self.b * length / self.avg_len)
            out[j] = sum(self.idf[t] * tf[t] * (self.k1 + 1) / (tf[t] + norm) for t in terms if t in tf)
        return out
