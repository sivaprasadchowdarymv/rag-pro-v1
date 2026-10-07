"""
Session state and shared resources.

* Workspace: taken from the link (?ws=...) or created; all personal data lives
  under data/users/<workspace>/ (private, isolated).
* Sync: one background SyncManager per process mirrors workspaces and document
  indexes to a private Hugging Face dataset when HF_TOKEN + HF_DATASET_REPO are set.
* Documents listed in the workspace are reopened automatically (no re-upload).
"""
from __future__ import annotations

import dataclasses
import hmac
from pathlib import Path
from typing import Dict, List, Optional

import streamlit as st

from chat.conversations import Conversation, ConversationStore
from config.settings import Settings, load_settings
from learning.feedback import FeedbackStore
from learning.preferences import PreferenceStore
from llm.model_router import ModelRouter, ProviderStatus
from rag.models import DocumentIndex
from rag.pipeline import open_index
from storage.remote import SyncManager, make_sync
from storage.workspace import Workspace, new_workspace_id, valid_workspace_id


@st.cache_resource(show_spinner=False)
def base_settings() -> Settings:
    return load_settings()


def settings() -> Settings:
    overrides = st.session_state.get("overrides", {})
    return dataclasses.replace(base_settings(), **overrides) if overrides else base_settings()


def set_override(**kw) -> None:
    st.session_state.setdefault("overrides", {}).update(kw)


# ------------------------------------------------------------------- access gate
def password_ok() -> bool:
    """Optional APP_PASSWORD protects the whole app."""
    secret = base_settings().app_password
    if not secret or st.session_state.get("auth_ok"):
        return True
    st.markdown('<div class="hero"><div class="hero-title">🔒 RAG∞ Pro</div>'
                '<div class="hero-sub">This workspace is password protected.</div></div>', unsafe_allow_html=True)
    with st.form("login"):
        pw = st.text_input("Password", type="password")
        if st.form_submit_button("Enter", type="primary"):
            if hmac.compare_digest(pw.encode(), secret.encode()):
                st.session_state["auth_ok"] = True
                st.rerun()
            st.error("Wrong password.")
    return False


# ------------------------------------------------------------- sync + workspace
@st.cache_resource(show_spinner=False)
def sync_manager() -> SyncManager:
    s = base_settings()
    return make_sync(s.data_dir, s.hf_token, s.hf_dataset_repo, s.sync_interval)


def workspace() -> Workspace:
    ws_id = st.session_state.get("ws_id")
    if not ws_id:
        candidate = st.query_params.get("ws")
        ws_id = candidate if valid_workspace_id(candidate) else new_workspace_id()
        st.session_state["ws_id"] = ws_id
        st.query_params["ws"] = ws_id  # keep it in the link so a bookmark reopens it
    ws = Workspace(settings().data_dir, ws_id)
    if not st.session_state.get("ws_synced") == ws_id:
        sync_manager().track(ws.prefix)  # restores the workspace from Hugging Face once
        st.session_state["ws_synced"] = ws_id
    return ws


def switch_workspace(ws_id: str) -> bool:
    if not valid_workspace_id(ws_id):
        return False
    for key in ("ws_id", "ws_synced", "docs", "docs_loaded", "conv_id", "last_result", "suggestions"):
        st.session_state.pop(key, None)
    st.query_params["ws"] = ws_id
    return True


@st.cache_resource(show_spinner=False)
def _stores(user_dir: str):
    d = Path(user_dir)
    return ConversationStore(d), FeedbackStore(d), PreferenceStore(d)


def conversations() -> ConversationStore:
    return _stores(str(workspace().dir))[0]


def feedback() -> FeedbackStore:
    return _stores(str(workspace().dir))[1]


def preferences() -> PreferenceStore:
    return _stores(str(workspace().dir))[2]


def router() -> ModelRouter:
    return ModelRouter(settings())


@st.cache_data(ttl=60, show_spinner=False)
def _status(_settings: Settings, key: str) -> List[ProviderStatus]:
    return ModelRouter(_settings).status()


def provider_status() -> List[ProviderStatus]:
    s = settings()
    key = "|".join(s.llm_providers) + f"|{bool(s.groq_api_key)}|{bool(s.ollama_api_key)}|{s.enable_ollama_local}"
    return _status(s, key)


# ---------------------------------------------------------------------- documents
def docs() -> Dict[str, Dict]:
    """doc_id -> {"index": DocumentIndex | None, "filename", "active", "missing"} for this workspace."""
    if not st.session_state.get("docs_loaded"):
        ws, s, sync = workspace(), settings(), sync_manager()
        loaded: Dict[str, Dict] = {}
        for d in ws.docs():
            for prefix in (f"cache/{d['doc_id']}", f"documents/{d['doc_id']}"):
                sync.track(prefix)
            index = open_index(d["doc_id"], d["filename"], s)
            loaded[d["doc_id"]] = {"index": index, "filename": d["filename"], "active": d.get("active", True),
                                   "missing": index is None}
        st.session_state["docs"] = loaded
        st.session_state["docs_loaded"] = True
    return st.session_state["docs"]


def add_doc(index: DocumentIndex) -> None:
    ws, sync = workspace(), sync_manager()
    ws.add_doc(index.doc_id, index.filename)
    for prefix in (f"cache/{index.doc_id}", f"documents/{index.doc_id}"):
        sync.track(prefix, pull=False)
    docs()[index.doc_id] = {"index": index, "filename": index.filename, "active": True, "missing": False}
    st.session_state.pop("suggestions", None)


def remove_doc(doc_id: str) -> None:
    workspace().remove_doc(doc_id)
    docs().pop(doc_id, None)
    st.session_state.pop("suggestions", None)


def set_active(doc_id: str, active: bool) -> None:
    d = docs().get(doc_id)
    if d and d["active"] != active:
        d["active"] = active
        workspace().set_active(doc_id, active)
        st.session_state.pop("suggestions", None)


def active_indexes() -> List[DocumentIndex]:
    return [d["index"] for d in docs().values() if d["active"] and d["index"] is not None]


# ---------------------------------------------------------------- conversation
def current_conversation() -> Conversation:
    store = conversations()
    cid = st.session_state.get("conv_id")
    conv: Optional[Conversation] = store.load(cid) if cid else None
    if conv is None:
        existing = store.list()
        conv = existing[0] if existing else store.create([i.doc_id for i in active_indexes()])
        st.session_state["conv_id"] = conv.id
    return conv
