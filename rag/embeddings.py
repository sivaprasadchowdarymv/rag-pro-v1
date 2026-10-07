"""
In-app embeddings (fastembed + ONNX on CPU), no API needed.

Model: nomic-embed-text-v1.5 (quantised, ~130 MB, 768 dims), the same model
family the original app used through Ollama. It downloads once on first use.

Vectors are stored L2-normalised in one matrix, so cosine similarity for all
nodes is a single matrix-vector product (see retrieval.py).
"""
from __future__ import annotations

import gc
import os
import threading
from collections import OrderedDict
from functools import lru_cache
from typing import Callable, List, Optional

import numpy as np

from config.settings import Settings, get_logger
from rag.models import DocumentIndex
from storage.vector_store import normalize_rows

log = get_logger("embeddings")

EMBED_CHAR_LIMIT = 2000  # unchanged from the original RAG
# Small batches + capped threads keep peak RAM low on free hosts (Streamlit Cloud ~2.7 GB);
# a 10-page paper can produce 800+ chunks. Override with EMBED_BATCH_SIZE / EMBED_THREADS.
EMBED_BATCH_SIZE = max(1, int(os.getenv("EMBED_BATCH_SIZE", "8") or 8))
EMBED_THREADS = max(1, int(os.getenv("EMBED_THREADS", "2") or 2))

ProgressFn = Optional[Callable[[float, str], None]]


class EmbeddingError(Exception):
    def __init__(self, model: str):
        self.user_message = (
            f"The embedding model '{model.split('/')[-1]}' could not be loaded "
            "(it downloads once, about 130 MB). Semantic search is off; keyword "
            "search still works. Please try again in a minute."
        )
        super().__init__(self.user_message)


# Nomic models are trained with task prefixes; using them improves retrieval.
DOC_PREFIX = "search_document: "
QUERY_PREFIX = "search_query: "

_load_lock = threading.Lock()


@lru_cache(maxsize=2)
def _model_cached(model_name: str, cache_dir: str):
    from fastembed import TextEmbedding  # heavy import: only when needed

    log.info("Loading embedding model %s", model_name)
    # kSameAsRequested stops onnxruntime's memory arena from doubling on every growth
    providers = [("CPUExecutionProvider", {"arena_extend_strategy": "kSameAsRequested"})]
    try:
        return TextEmbedding(model_name=model_name, cache_dir=cache_dir, threads=EMBED_THREADS, providers=providers)
    except Exception as exc:  # option not supported by this fastembed/onnxruntime: fall back to defaults
        log.warning("Low-memory embedding options unavailable (%s); using defaults", type(exc).__name__)
        return TextEmbedding(model_name=model_name, cache_dir=cache_dir)


def _model(model_name: str, cache_dir: str):
    with _load_lock:  # two users at once must not download the model twice
        return _model_cached(model_name, cache_dir)


def _uses_prefixes(model: str) -> bool:
    return "nomic" in model.lower()


def embedding_signature(settings: Settings, model: Optional[str] = None) -> str:
    """Identifies how vectors were made; part of the embedding cache key."""
    model = model or settings.embed_model
    return f"fastembed:{model}" + ("|prefix-v1" if _uses_prefixes(model) else "")


def warm_up(settings: Settings) -> None:
    """Start loading the embedding model in the background when the app opens."""

    def _load() -> None:
        try:
            _model(settings.embed_model, str(settings.data_dir / "models"))
        except Exception as exc:  # reported properly when actually used
            log.warning("Background model load failed: %s", exc)

    threading.Thread(target=_load, name="embed-warmup", daemon=True).start()


def embed_texts(texts: List[str], settings: Settings, model: str, kind: str = "document") -> List[List[float]]:
    if _uses_prefixes(model):
        prefix = QUERY_PREFIX if kind == "query" else DOC_PREFIX
        texts = [prefix + t for t in texts]
    try:
        engine = _model(model, str(settings.data_dir / "models"))
        return [np.asarray(v, dtype=np.float32).tolist() for v in engine.embed(texts, batch_size=EMBED_BATCH_SIZE)]
    except Exception as exc:
        log.exception("Embedding failed")
        raise EmbeddingError(model) from exc


_QVEC: "OrderedDict[tuple, np.ndarray]" = OrderedDict()
_QVEC_MAX = 512
QVEC_STATS = {"hits": 0, "misses": 0}


def embed_query(query: str, settings: Settings, model: Optional[str] = None) -> np.ndarray:
    model = model or settings.embed_model
    key = (model, query[:EMBED_CHAR_LIMIT])
    cached = _QVEC.get(key)
    if cached is not None:
        _QVEC.move_to_end(key)
        QVEC_STATS["hits"] += 1
        return cached
    QVEC_STATS["misses"] += 1
    vec = embed_texts([query[:EMBED_CHAR_LIMIT]], settings, model, kind="query")[0]
    out = normalize_rows(np.asarray([vec], dtype=np.float32))[0]
    out.setflags(write=False)
    _QVEC[key] = out
    while len(_QVEC) > _QVEC_MAX:
        _QVEC.popitem(last=False)
    return out


def fill_missing_embeddings(index: DocumentIndex, settings: Settings, progress: ProgressFn = None) -> int:
    """Embed nodes that have no embedding yet (in place). Returns how many."""
    n = len(index.nodes)
    if len(index.has_embedding) != n:
        index.has_embedding = np.zeros(n, dtype=bool)
    todo = [i for i in range(n) if not index.has_embedding[i] and index.nodes[i].content]
    if not todo:
        return 0
    done = 0
    for start in range(0, len(todo), EMBED_BATCH_SIZE):
        batch = todo[start : start + EMBED_BATCH_SIZE]
        vectors = embed_texts([index.nodes[i].content[:EMBED_CHAR_LIMIT] for i in batch], settings, index.embed_model)
        matrix = normalize_rows(np.asarray(vectors, dtype=np.float32))
        if index.embeddings.shape != (n, matrix.shape[1]):
            index.embeddings = np.zeros((n, matrix.shape[1]), dtype=np.float32)
            index.has_embedding[:] = False
        index.embeddings[batch] = matrix
        index.has_embedding[batch] = True
        done += len(batch)
        del vectors, matrix
        if done % (EMBED_BATCH_SIZE * 16) == 0:
            gc.collect()
        if progress:
            progress(done / len(todo), f"Embedding {done}/{len(todo)} chunks")
    log.info("Embedding completed: %d nodes", done)
    return done
