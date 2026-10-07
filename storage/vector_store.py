"""
Tiny local vector store: nodes in JSON + one NumPy embedding matrix per
(document, chunking settings, embedding model).

Why not ChromaDB?
  * Retrieval here is *hybrid*: every node also gets a RapidFuzz score plus
    section / parent-context / metadata bonuses. That needs a pass over all
    nodes anyway, so an approximate-nearest-neighbour index would not make
    retrieval faster.
  * A datasheet has hundreds to a few thousand nodes. One matrix-vector
    product over that is well under a millisecond with NumPy.
  * ChromaDB would add a large dependency tree (onnxruntime, grpc, ...) and a
    second storage format to back up — more to break on a small VM.
If you later index thousands of PDFs *together*, revisit this choice.

Cache keys:
  parse_key = hash(chunk size, overlap, min figure size, format version)
  embed_key = hash(parse_key, embedding model)
So changing the embedding model re-embeds without re-parsing the PDF, and
changing the chunk size re-parses. RAG.py keyed its cache on the PDF hash
only, so changing either setting silently kept using the old index (and a
different embedding model could crash cosine similarity on mismatched
vector sizes).
"""
from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

from config.settings import Settings, get_logger
from rag.models import Node
from storage.cache import DocumentPaths, read_json, short_hash, write_json_atomic

log = get_logger("vector_store")

INDEX_FORMAT_VERSION = 3  # bump when the node format / extraction changes


def parse_key(settings: Settings) -> str:
    return short_hash(INDEX_FORMAT_VERSION, *settings.parse_signature)


def embed_key(p_key: str, embed_model: str) -> str:
    return short_hash(p_key, embed_model)


# ---------------------------------------------------------------------------
# Nodes
# ---------------------------------------------------------------------------
def save_nodes(
    paths: DocumentPaths,
    p_key: str,
    nodes: List[Node],
    figures: List[Node],
    page_count: int,
    doc_meta: Optional[dict] = None,
) -> None:
    write_json_atomic(
        paths.nodes_file(p_key),
        {
            "version": INDEX_FORMAT_VERSION,
            "page_count": page_count,
            "nodes": [n.to_dict() for n in nodes],
            "figures": [f.to_dict() for f in figures],
            "doc_meta": doc_meta or {},
        },
    )


def load_nodes(
    paths: DocumentPaths, p_key: str
) -> Optional[Tuple[List[Node], List[Node], int, dict]]:
    raw = read_json(paths.nodes_file(p_key))
    if not raw or raw.get("version") != INDEX_FORMAT_VERSION:
        return None
    try:
        nodes = [Node.from_dict(n) for n in raw["nodes"]]
        figures = [Node.from_dict(f) for f in raw["figures"]]
        return nodes, figures, int(raw["page_count"]), dict(raw.get("doc_meta") or {})
    except (KeyError, TypeError, ValueError) as exc:
        log.warning("Discarding malformed node cache (%s)", exc)
        return None


# ---------------------------------------------------------------------------
# Embeddings
# ---------------------------------------------------------------------------
def normalize_rows(matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=np.float32)
    if matrix.size == 0:
        return matrix
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return matrix / norms


def save_embeddings(
    paths: DocumentPaths, e_key: str, matrix: np.ndarray, mask: np.ndarray
) -> None:
    target = paths.embeddings_file(e_key)
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=target.parent, suffix=".tmp")
    try:
        with os.fdopen(fd, "wb") as fh:
            np.savez(fh, embeddings=matrix.astype(np.float32), mask=mask.astype(bool))
        os.replace(tmp, target)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise


def load_embeddings(
    paths: DocumentPaths, e_key: str, expected_rows: int
) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    path = paths.embeddings_file(e_key)
    if not path.exists():
        return None
    try:
        with np.load(path, allow_pickle=False) as data:
            matrix, mask = data["embeddings"], data["mask"]
    except Exception as exc:
        log.warning("Discarding unreadable embedding cache (%s)", exc)
        return None
    if matrix.shape[0] != expected_rows or mask.shape[0] != expected_rows:
        return None
    return matrix, mask
