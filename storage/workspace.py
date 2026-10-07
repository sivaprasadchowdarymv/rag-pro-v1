"""
Private workspaces: every visitor's data lives under data/users/<workspace_id>/.

The id is a long random key (144 bits) kept in the browser link (?ws=...), so
bookmarking the link reopens your chats, documents and preferences, and no
one can list or guess another person's workspace.
"""
from __future__ import annotations

import re
import secrets
import time
from pathlib import Path
from typing import Dict, List, Optional

from storage.cache import read_json, write_json_atomic

_WS = re.compile(r"^[A-Za-z0-9_-]{20,40}$")


def new_workspace_id() -> str:
    return secrets.token_urlsafe(18)  # 24 characters, 144 random bits


def valid_workspace_id(value: Optional[str]) -> bool:
    return bool(value) and bool(_WS.match(value or ""))


class Workspace:
    def __init__(self, data_dir: Path, ws_id: str):
        if not valid_workspace_id(ws_id):
            raise ValueError("invalid workspace id")
        self.id = ws_id
        self.prefix = f"users/{ws_id}"
        self.dir = Path(data_dir) / self.prefix
        self.dir.mkdir(parents=True, exist_ok=True)
        self._file = self.dir / "workspace.json"

    # --- document list ---------------------------------------------------
    def _load(self) -> Dict:
        return read_json(self._file, {}) or {"created": time.time(), "docs": []}

    def docs(self) -> List[Dict]:
        return list(self._load().get("docs", []))

    def add_doc(self, doc_id: str, filename: str) -> None:
        data = self._load()
        if not any(d["doc_id"] == doc_id for d in data["docs"]):
            data["docs"].append({"doc_id": doc_id, "filename": filename, "added": time.time(), "active": True})
            write_json_atomic(self._file, data)

    def remove_doc(self, doc_id: str) -> None:
        data = self._load()
        data["docs"] = [d for d in data["docs"] if d["doc_id"] != doc_id]
        write_json_atomic(self._file, data)

    def set_active(self, doc_id: str, active: bool) -> None:
        data = self._load()
        changed = False
        for d in data["docs"]:
            if d["doc_id"] == doc_id and d.get("active", True) != active:
                d["active"], changed = active, True
        if changed:
            write_json_atomic(self._file, data)

    # --- validation question sets -----------------------------------------
    @property
    def validation_file(self) -> Path:
        return self.dir / "validation.json"

    def load_validation(self) -> List[Dict]:
        return read_json(self.validation_file, []) or []

    def save_validation(self, rows: List[Dict]) -> None:
        write_json_atomic(self.validation_file, rows)
