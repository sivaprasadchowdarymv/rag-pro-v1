"""
Cross-encoder reranking (fastembed TextCrossEncoder, ONNX on CPU, ~80 MB).

The cross-encoder reads the question and each candidate together, which ranks
evidence far more precisely than embedding similarity alone. If the model
cannot be loaded, reranking is skipped (the fused order is kept) and a
warning is reported; the app keeps working.
"""
from __future__ import annotations

import threading
import time
from functools import lru_cache
from typing import List, Optional, Sequence, Tuple

from config.settings import Settings, get_logger

log = get_logger("reranker")
_lock = threading.Lock()


@lru_cache(maxsize=2)
def _load(model: str, cache_dir: str):
    from fastembed.rerank.cross_encoder import TextCrossEncoder

    log.info("Loading reranker %s", model)
    return TextCrossEncoder(model_name=model, cache_dir=cache_dir)


def rerank(query: str, texts: Sequence[str], settings: Settings) -> Tuple[Optional[List[float]], float, str]:
    """Return (scores or None, latency_ms, warning)."""
    if not settings.rerank or not texts:
        return None, 0.0, ""
    t0 = time.perf_counter()
    try:
        with _lock:
            model = _load(settings.rerank_model, str(settings.data_dir / "models"))
        scores = [float(s) for s in model.rerank(query, [t[:1500] for t in texts])]
        return scores, (time.perf_counter() - t0) * 1000, ""
    except Exception as exc:
        log.warning("Reranking skipped: %s", exc)
        return None, (time.perf_counter() - t0) * 1000, "Reranking was skipped (model unavailable); results use hybrid ranking."
