"""
The two things the UI needs:

    get_index(pdf_bytes, ...)   PDF -> DocumentIndex   (indexed once, then cached)
    answer_query(query, index)  question -> QueryResult (ReAct agent answer + sources)

Old RAG.py equivalents: `build_graph()` (caching half) and `answer_query()`.

Caching layers, fastest first:
  1. in-process LRU (`_registry`) — shared by all browser sessions;
  2. disk cache under data/cache/<doc_id>/ — survives restarts;
  3. full parse + embed — only for a new PDF or new chunking/embedding settings.
A lock makes sure only one PDF is indexed at a time, which keeps memory use
predictable on a small VM and stops two users indexing the same file twice.
"""
from __future__ import annotations

import re
import threading
import time
from collections import OrderedDict
from typing import Callable, List, Optional, Tuple

import numpy as np

from config.settings import Settings, get_logger
from rag import llm
from rag.agent import NOT_FOUND, run_agent, run_quick
from rag.embeddings import EmbeddingError, embedding_signature, fill_missing_embeddings
from rag.evaluation import compute_metrics
from rag.models import AgentStep, DocumentIndex, QueryResult
from rag.pdf_parser import parse_pdf, validate_pdf_bytes
from rag.tools import SourceRegistry, ToolBox, initial_evidence
from storage import feedback as feedback_store
from storage import vector_store
from storage.cache import (
    DocumentPaths,
    clear_document_cache,
    compute_doc_id,
    read_json,
    sanitize_filename,
    save_document_meta,
)

log = get_logger("pipeline")

ProgressFn = Optional[Callable[[float, str], None]]

_REGISTRY_SIZE = 6
_registry: "OrderedDict[Tuple[str, str, str], DocumentIndex]" = OrderedDict()
_registry_lock = threading.Lock()
_build_lock = threading.Lock()


# ---------------------------------------------------------------------------
# Indexing
# ---------------------------------------------------------------------------
def _registry_get(key) -> Optional[DocumentIndex]:
    with _registry_lock:
        index = _registry.get(key)
        if index is not None:
            _registry.move_to_end(key)
        return index


def _registry_put(key, index: DocumentIndex) -> None:
    with _registry_lock:
        _registry[key] = index
        _registry.move_to_end(key)
        while len(_registry) > _REGISTRY_SIZE:
            _registry.popitem(last=False)


def _load_or_build(pdf_bytes: Optional[bytes], filename: str, settings: Settings, paths: DocumentPaths,
                   p_key: str, e_key: str, progress: ProgressFn) -> Optional[DocumentIndex]:
    cached = vector_store.load_nodes(paths, p_key)
    if cached is not None:
        nodes, figures, page_count, doc_meta = cached
        log.info("Loaded index for %s from disk cache", paths.doc_id)
    elif pdf_bytes is None:
        return None  # nothing cached and no PDF to rebuild from
    else:
        paths.ensure()
        parsed = parse_pdf(
            pdf_bytes, settings, paths.figures_dir,
            progress=(lambda f, m: progress(f * 0.7, m)) if progress else None,
        )
        nodes, figures, page_count, doc_meta = parsed.nodes, parsed.figures, parsed.page_count, parsed.doc_meta
        vector_store.save_nodes(paths, p_key, nodes, figures, page_count, doc_meta)
        save_document_meta(paths, filename, page_count)

    meta = read_json(paths.meta_file, {}) or {}
    index = DocumentIndex(
        doc_id=paths.doc_id,
        filename=meta.get("filename", filename),
        page_count=page_count,
        nodes=nodes,
        figures=figures,
        figures_dir=paths.figures_dir,
        embed_model=settings.embed_model,
        doc_meta=doc_meta,
        has_embedding=np.zeros(len(nodes), dtype=bool),
    )
    loaded = vector_store.load_embeddings(paths, e_key, len(nodes))
    if loaded is not None:
        index.embeddings, index.has_embedding = loaded

    return index


def get_index(
    pdf_bytes: bytes,
    filename: str,
    settings: Settings,
    progress: ProgressFn = None,
) -> Tuple[DocumentIndex, List[str]]:
    """
    Return the index for this PDF plus user-facing warnings.
    Raises PdfProcessingError (with a friendly message) for bad files.
    """
    validate_pdf_bytes(pdf_bytes, settings)
    filename = sanitize_filename(filename)
    doc_id = compute_doc_id(pdf_bytes)
    paths = DocumentPaths(settings.data_dir, doc_id)
    p_key = vector_store.parse_key(settings)
    e_key = vector_store.embed_key(p_key, embedding_signature(settings))
    key = (doc_id, p_key, e_key)

    index = _registry_get(key)
    if index is None:
        with _build_lock:
            index = _registry_get(key)
            if index is None:
                log.info("Document hash: %s (%s)", doc_id, filename)
                index = _load_or_build(pdf_bytes, filename, settings, paths, p_key, e_key, progress)
                _registry_put(key, index)

    warnings: List[str] = []
    if index.missing_embeddings:
        reason = ""
        with _build_lock:
            try:
                wrapped = (lambda f, m: progress(0.7 + f * 0.3, m)) if progress else None
                if fill_missing_embeddings(index, settings, wrapped):
                    vector_store.save_embeddings(paths, e_key, index.embeddings, index.has_embedding)
            except EmbeddingError as exc:
                reason = exc.user_message
        if index.missing_embeddings:
            warnings.append(
                "Semantic search is off for part of this document, so answers use "
                f"keyword matching only.\n\n{reason}".strip()
            )
    return index, warnings


def open_index(doc_id: str, filename: str, settings: Settings) -> Optional[DocumentIndex]:
    """Re-open an already indexed document from its cache (no PDF needed).
    Returns None if the cache is missing or was built with different chunk settings."""
    try:
        paths = DocumentPaths(settings.data_dir, doc_id)
    except ValueError:
        return None
    p_key = vector_store.parse_key(settings)
    e_key = vector_store.embed_key(p_key, embedding_signature(settings))
    key = (doc_id, p_key, e_key)
    index = _registry_get(key)
    if index is None:
        with _build_lock:
            index = _registry_get(key) or _load_or_build(None, sanitize_filename(filename), settings, paths,
                                                         p_key, e_key, None)
            if index is not None:
                _registry_put(key, index)
    if index is not None and index.missing_embeddings:
        try:
            with _build_lock:
                if fill_missing_embeddings(index, settings):
                    vector_store.save_embeddings(paths, e_key, index.embeddings, index.has_embedding)
        except EmbeddingError:
            pass  # keyword search still works
    return index


def clear_cache(doc_id: str, settings: Settings) -> None:
    with _registry_lock:
        for key in [k for k in _registry if k[0] == doc_id]:
            del _registry[key]
    clear_document_cache(DocumentPaths(settings.data_dir, doc_id))


def document_paths(index: DocumentIndex, settings: Settings) -> DocumentPaths:
    return DocumentPaths(settings.data_dir, index.doc_id)


# ---------------------------------------------------------------------------
# Question answering
# ---------------------------------------------------------------------------
_IMAGE_MD = re.compile(r"!\[[^\]]*\]\([^)]*\)")
_CITE_NUM = re.compile(r"\[REF-(\d+)[^\]]*\]")
StepFn = Optional[Callable[[AgentStep], None]]


def normalize_answer(text: str) -> str:
    """LaTeX delimiters Streamlit understands; drop remote-image markdown."""
    text = (text or "").strip()
    text = text.replace("\\[", "$$").replace("\\]", "$$")
    text = text.replace("\\(", "$").replace("\\)", "$")
    return _IMAGE_MD.sub("", text)


def _fallback(registry: SourceRegistry) -> str:
    return "\n\n".join(f"{r.label}\n{r.full_text[:300]}" for r in registry.refs[:3])


def answer_query(
    query: str,
    index: DocumentIndex,
    settings: Settings,
    mode: Optional[str] = None,
    on_step: StepFn = None,
) -> QueryResult:
    t0 = time.time()
    query = (query or "").strip()
    mode = mode or settings.agent_mode
    paths = document_paths(index, settings)
    warnings: List[str] = []

    # 0. Saved human answer (optional feature)
    if settings.enable_feedback:
        saved = feedback_store.check_feedback(paths, query)
        if saved:
            return QueryResult(kind="feedback_override", answer=saved, mode=mode)

    # 1. Local hybrid retrieval (free, fast) = the agent's first observation
    registry = SourceRegistry(settings)
    toolbox = ToolBox(index, settings, registry)
    evidence, engine_results, n_hits = initial_evidence(query, toolbox)
    if toolbox.embedding_warning:
        warnings.append(toolbox.embedding_warning)
    log.info("Retrieval completed: %d excerpts", n_hits)
    if on_step:
        on_step(AgentStep("observation", f"Found {n_hits} relevant excerpts", tool="hybrid_retrieval"))

    if mode == "quick" and n_hits == 0:
        return QueryResult(kind="not_found", answer=NOT_FOUND, mode=mode,
                           latency=round(time.time() - t0, 2), warnings=warnings)

    # 2. LLM: ReAct agent or one quick call
    steps: List[AgentStep] = []
    answer, calls, tokens, used_llm = "", 0, 0, False
    try:
        if mode == "quick":
            outcome = run_quick(query, evidence, settings)
        else:
            outcome = run_agent(query, evidence, toolbox, settings, on_step=on_step)
        answer, steps = outcome.answer, outcome.steps
        calls, tokens, used_llm = outcome.llm_calls, outcome.tokens, bool(outcome.answer)
    except llm.LLMError as exc:
        warnings.append(f"Showing the best matching excerpts without an AI answer.\n\n{exc.user_message}")
    if toolbox.embedding_warning and toolbox.embedding_warning not in warnings:
        warnings.append(toolbox.embedding_warning)

    if not answer:
        if not registry.refs:
            return QueryResult(kind="not_found", answer=NOT_FOUND, mode=mode, steps=steps,
                               latency=round(time.time() - t0, 2), warnings=warnings,
                               llm_calls=calls, tokens=tokens)
        answer = _fallback(registry)

    answer = normalize_answer(answer)
    not_found = NOT_FOUND in answer.upper() and len(answer) < 80

    # 3. Citation check: rewrite every citation to its true label (so the page
    #    and type shown are always correct), keep cited sources, flag problems.
    cited: List = []
    bad: List[str] = []

    def _canonical(match: "re.Match[str]") -> str:
        ref = registry.by_number(int(match.group(1)))
        if ref is None:
            bad.append(match.group(0))
            return match.group(0)
        if ref not in cited:
            cited.append(ref)
        return ref.label

    answer = _CITE_NUM.sub(_canonical, answer)
    for label in dict.fromkeys(bad):
        warnings.append(f"The answer cites {label}, which is not one of the retrieved sources. Treat that statement with caution.")
    if used_llm and not cited and not not_found:
        warnings.append("The answer has no valid citations. Verify it against the sources below.")
    sources = cited or list(registry.refs)

    latency = time.time() - t0
    log.info("Answered in %.2fs (%s mode, %d LLM calls, %d tokens)", latency, mode, calls, tokens)
    return QueryResult(
        kind="not_found" if not_found else "answer",
        answer=answer,
        sources=[] if not_found else sources,
        metrics={} if not_found else compute_metrics(answer, sources, query, engine_results, latency),
        latency=round(latency, 2),
        warnings=warnings,
        used_llm=used_llm,
        mode=mode,
        steps=steps,
        llm_calls=calls,
        tokens=tokens,
    )


def save_feedback(index: DocumentIndex, settings: Settings, question: str, answer: str) -> None:
    feedback_store.store_feedback(document_paths(index, settings), question, answer)
