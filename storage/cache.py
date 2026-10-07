"""
Where things are stored on disk.

    data/                              (DATA_DIR, a Docker volume in production)
    ├── documents/<doc_id>/            persistent document data
    │   ├── meta.json                  file name, pages, date indexed
    │   ├── figures/                   extracted images (LLaVA reads these)
    │   └── feedback.json              saved answers (only if ENABLE_FEEDBACK)
    └── cache/<doc_id>/                rebuildable cache, safe to delete
        ├── nodes_<parse_key>.json     chunks / tables / equations / figures
        ├── emb_<embed_key>.npz        embedding matrix
        └── vision.json                cached LLaVA figure descriptions

The uploaded PDF itself is NOT kept: once it is indexed, the index is all the
app needs. That is the safest way to "store uploaded documents" on a public
server. <doc_id> is the first 16 hex chars of the SHA-256 of the file, so the
same PDF uploaded twice is indexed once, and a user-supplied file name is
never used as a path.

Old RAG.py equivalent: `CACHE_DIR`, `pdf_hash()`, the JSON cache inside
`build_graph()` and the "Clear Cache" button.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import tempfile
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

from config.settings import get_logger

log = get_logger("storage")

_DOC_ID_RE = re.compile(r"^[0-9a-f]{16}$")
_KEY_RE = re.compile(r"^[0-9a-f]{12}$")


def compute_doc_id(pdf_bytes: bytes) -> str:
    return hashlib.sha256(pdf_bytes).hexdigest()[:16]


def short_hash(*parts: Any) -> str:
    return hashlib.sha256(repr(parts).encode("utf-8")).hexdigest()[:12]


def sanitize_filename(name: str, max_len: int = 100) -> str:
    """Display-safe file name: no directories, no odd characters."""
    base = os.path.basename((name or "").replace("\\", "/"))
    base = re.sub(r"[^A-Za-z0-9._\- ]+", "_", base).strip(" ._") or "document"
    stem, ext = os.path.splitext(base)
    if ext.lower() != ".pdf":
        stem, ext = base, ".pdf"
    return stem[: max_len - len(ext)] + ext


# ---------------------------------------------------------------------------
# Atomic JSON helpers (a crash mid-write never leaves a corrupt cache file)
# ---------------------------------------------------------------------------
def write_json_atomic(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(data, fh, ensure_ascii=False)
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise


def read_json(path: Path, default: Any = None) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return default
    except (OSError, ValueError) as exc:
        log.warning("Ignoring unreadable file %s (%s)", path.name, exc)
        return default


# ---------------------------------------------------------------------------
# Per-document paths
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class DocumentPaths:
    data_dir: Path
    doc_id: str

    def __post_init__(self) -> None:
        if not _DOC_ID_RE.match(self.doc_id):
            raise ValueError("invalid document id")

    @property
    def doc_dir(self) -> Path:
        return self.data_dir / "documents" / self.doc_id

    @property
    def figures_dir(self) -> Path:
        return self.doc_dir / "figures"

    @property
    def meta_file(self) -> Path:
        return self.doc_dir / "meta.json"

    @property
    def feedback_file(self) -> Path:
        return self.doc_dir / "feedback.json"

    @property
    def cache_dir(self) -> Path:
        return self.data_dir / "cache" / self.doc_id

    def nodes_file(self, parse_key: str) -> Path:
        assert _KEY_RE.match(parse_key)
        return self.cache_dir / f"nodes_{parse_key}.json"

    def embeddings_file(self, embed_key: str) -> Path:
        assert _KEY_RE.match(embed_key)
        return self.cache_dir / f"emb_{embed_key}.npz"

    @property
    def vision_file(self) -> Path:
        return self.cache_dir / "vision.json"

    def ensure(self) -> None:
        self.figures_dir.mkdir(parents=True, exist_ok=True)
        self.cache_dir.mkdir(parents=True, exist_ok=True)


def save_document_meta(paths: DocumentPaths, filename: str, page_count: int) -> None:
    write_json_atomic(
        paths.meta_file,
        {
            "doc_id": paths.doc_id,
            "filename": filename,
            "pages": page_count,
            "indexed_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        },
    )


def clear_document_cache(paths: DocumentPaths) -> None:
    """Delete everything rebuildable for one document (index, vision, figures).

    Figures are re-extracted on the next indexing run."""
    shutil.rmtree(paths.cache_dir, ignore_errors=True)
    shutil.rmtree(paths.figures_dir, ignore_errors=True)
    log.info("Cache cleared for document %s", paths.doc_id)


# ---------------------------------------------------------------------------
# LLaVA description cache (so a figure is only ever described once)
# ---------------------------------------------------------------------------
class VisionCache:
    _lock = threading.Lock()

    def __init__(self, path: Path):
        self.path = path

    def get(self, figure_file: str) -> Optional[str]:
        return (read_json(self.path, {}) or {}).get(figure_file)

    def set(self, figure_file: str, description: str) -> None:
        with self._lock:
            data: Dict[str, str] = read_json(self.path, {}) or {}
            data[figure_file] = description
            write_json_atomic(self.path, data)
