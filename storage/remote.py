"""
Persistent storage: mirror parts of DATA_DIR to a PRIVATE Hugging Face dataset.

Streamlit Community Cloud wipes the disk on every restart. With HF_TOKEN and
HF_DATASET_REPO set, these folders survive restarts:

    users/<workspace>/   conversations, feedback, preferences, document list, validation sets
    cache/<doc_id>/      document indexes (nodes + embeddings) -> no re-upload needed
    documents/<doc_id>/  document metadata + extracted figures

How it works
  * pull: when a workspace or document is opened, its folder is downloaded once.
  * push: changed files are found by modification time and uploaded together in
    ONE commit, at most every SYNC_INTERVAL seconds (default 45), from a
    background thread, or immediately with "Save now". Deleted files are
    deleted remotely too.
  * Hugging Face limits commit frequency, so changes are batched, not uploaded one by one.
  * If a push fails, the files stay pending and are retried; the app keeps working.
"""
from __future__ import annotations

import shutil
import threading
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set

from config.settings import get_logger

log = get_logger("remote")


class RemoteBackend(ABC):
    label = "remote"

    @abstractmethod
    def pull(self, patterns: Sequence[str], local_dir: Path) -> None: ...

    @abstractmethod
    def push(self, adds: Dict[str, Path], deletes: Sequence[str], message: str) -> None: ...


class HFDatasetBackend(RemoteBackend):
    """Private Hugging Face dataset repository (free)."""

    label = "Hugging Face"

    def __init__(self, repo_id: str, token: str):
        from huggingface_hub import HfApi

        self.repo_id, self.token = repo_id, token
        self.api = HfApi(token=token)
        # private=True: only you (the token owner) can read it
        self.api.create_repo(repo_id, repo_type="dataset", private=True, exist_ok=True)

    def pull(self, patterns: Sequence[str], local_dir: Path) -> None:
        from huggingface_hub import snapshot_download

        snapshot_download(self.repo_id, repo_type="dataset", allow_patterns=list(patterns),
                          local_dir=str(local_dir), token=self.token)

    def push(self, adds: Dict[str, Path], deletes: Sequence[str], message: str) -> None:
        from huggingface_hub import CommitOperationAdd, CommitOperationDelete

        ops: list = [CommitOperationAdd(path_in_repo=p, path_or_fileobj=str(f)) for p, f in adds.items()]
        ops += [CommitOperationDelete(path_in_repo=p) for p in deletes]
        if ops:
            self.api.create_commit(self.repo_id, operations=ops, commit_message=message, repo_type="dataset")


class DirBackend(RemoteBackend):
    """A plain folder acting as the remote (used by the tests)."""

    label = "folder"

    def __init__(self, root: Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.commits = 0

    def pull(self, patterns: Sequence[str], local_dir: Path) -> None:
        for pattern in patterns:
            # Match like huggingface_hub's allow_patterns: "dir/**" means every file below dir.
            matches = (self.root / pattern[:-3]).rglob("*") if pattern.endswith("/**") else self.root.glob(pattern)
            for src in matches:
                if src.is_file():
                    dst = Path(local_dir) / src.relative_to(self.root)
                    dst.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(src, dst)

    def push(self, adds: Dict[str, Path], deletes: Sequence[str], message: str) -> None:
        for rel, src in adds.items():
            dst = self.root / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
        for rel in deletes:
            (self.root / rel).unlink(missing_ok=True)
        self.commits += 1


@dataclass
class SyncStatus:
    enabled: bool
    backend: str = ""
    pending: int = 0
    last_push: float = 0.0
    last_error: str = ""
    pushes: int = 0


@dataclass
class SyncManager:
    backend: Optional[RemoteBackend]
    data_dir: Path
    interval: float = 45.0
    _prefixes: Set[str] = field(default_factory=set)
    _known: Dict[str, float] = field(default_factory=dict)  # repo path -> mtime last pushed / pulled
    _pulled: Set[str] = field(default_factory=set)
    _lock: threading.Lock = field(default_factory=threading.Lock)
    _status: SyncStatus = field(default_factory=lambda: SyncStatus(False))
    _thread: Optional[threading.Thread] = None

    def __post_init__(self) -> None:
        self.data_dir = Path(self.data_dir)
        self._status = SyncStatus(self.backend is not None, self.backend.label if self.backend else "")
        if self.backend is not None and self.interval > 0:
            self._thread = threading.Thread(target=self._loop, name="hf-sync", daemon=True)
            self._thread.start()

    @property
    def enabled(self) -> bool:
        return self.backend is not None

    # ------------------------------------------------------------------ pull
    def track(self, prefix: str, pull: bool = True) -> None:
        """Start syncing a folder (e.g. 'users/<ws>'); download it first if we haven't yet."""
        prefix = prefix.strip("/")
        with self._lock:
            self._prefixes.add(prefix)
            first = prefix not in self._pulled
            self._pulled.add(prefix)
        if self.backend is None or not pull or not first:
            return
        try:
            self.backend.pull([f"{prefix}/**"], self.data_dir)
        except Exception as exc:  # offline / no access: keep working locally
            log.warning("Could not restore %s: %s", prefix, type(exc).__name__)
            self._status.last_error = f"Restore failed ({type(exc).__name__})"
        with self._lock:  # pulled files are already remote: don't push them back
            for path, mtime in self._scan([prefix]).items():
                self._known[path] = mtime

    # ------------------------------------------------------------------ push
    def _scan(self, prefixes: Sequence[str]) -> Dict[str, float]:
        out: Dict[str, float] = {}
        for prefix in prefixes:
            base = self.data_dir / prefix
            if base.exists():
                for f in base.rglob("*"):
                    if f.is_file() and not f.name.endswith((".tmp", ".lock")):
                        out[f.relative_to(self.data_dir).as_posix()] = f.stat().st_mtime
        return out

    def _changes(self):
        current = self._scan(sorted(self._prefixes))
        adds = {p: self.data_dir / p for p, m in current.items() if self._known.get(p) != m}
        deletes = [p for p in self._known if p not in current
                   and any(p.startswith(pre + "/") for pre in self._prefixes)]
        return adds, deletes, current

    def pending(self) -> int:
        with self._lock:
            adds, deletes, _ = self._changes()
        return len(adds) + len(deletes)

    def flush(self, reason: str = "auto") -> bool:
        """Upload all changes in one commit. Returns True if nothing failed."""
        if self.backend is None:
            return True
        with self._lock:
            adds, deletes, current = self._changes()
            if not adds and not deletes:
                return True
            try:
                self.backend.push(adds, deletes, f"RAG∞ Pro sync ({reason}): {len(adds)} changed, {len(deletes)} deleted")
            except Exception as exc:
                log.warning("Sync failed: %s", exc)
                self._status.last_error = f"Upload failed ({type(exc).__name__}); will retry"
                return False
            for p in adds:
                self._known[p] = current[p]
            for p in deletes:
                self._known.pop(p, None)
            self._status.last_push, self._status.last_error = time.time(), ""
            self._status.pushes += 1
            return True

    def _loop(self) -> None:
        while True:
            time.sleep(self.interval)
            try:
                self.flush("auto")
            except Exception:  # never let the background thread die
                log.exception("Background sync error")

    def status(self) -> SyncStatus:
        s = self._status
        return SyncStatus(s.enabled, s.backend, self.pending() if self.enabled else 0, s.last_push, s.last_error, s.pushes)


def make_sync(data_dir: Path, token: str, repo_id: str, interval: float) -> SyncManager:
    backend: Optional[RemoteBackend] = None
    if repo_id.startswith("file://"):  # mirror to a folder, e.g. a persistent disk on your own server
        backend = DirBackend(Path(repo_id[len("file://"):]))
        log.info("Persistent storage: folder %s", repo_id)
    elif token and repo_id:
        try:
            backend = HFDatasetBackend(repo_id, token)
            log.info("Persistent storage: Hugging Face dataset %s", repo_id)
        except Exception as exc:
            log.warning("Hugging Face storage unavailable (%s); using temporary storage", type(exc).__name__)
    return SyncManager(backend, Path(data_dir), interval)


__all__: List[str] = ["SyncManager", "make_sync", "HFDatasetBackend", "DirBackend", "SyncStatus"]
