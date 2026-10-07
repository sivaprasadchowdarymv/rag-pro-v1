"""Fixes #1 (private workspaces) and #2 (persistent storage that survives restarts)."""
import dataclasses

from chat.conversations import ConversationStore
from learning.feedback import FeedbackStore
from learning.preferences import PreferenceStore
from rag.pipeline import get_index, open_index
from storage.remote import DirBackend, SyncManager
from storage.workspace import Workspace, new_workspace_id


def test_workspaces_are_isolated(tmp_path):
    a, b = Workspace(tmp_path, new_workspace_id()), Workspace(tmp_path, new_workspace_id())
    ConversationStore(a.dir).create()
    FeedbackStore(a.dir).record(conversation_id="c", message_id="m", question="q", answer="x", rating=1)
    assert len(ConversationStore(a.dir).list()) == 1
    assert ConversationStore(b.dir).list() == [] and FeedbackStore(b.dir).all() == []  # B sees nothing of A
    assert a.dir != b.dir and a.dir.parent == b.dir.parent


def test_invalid_workspace_ids_rejected(tmp_path):
    for bad in ("../../etc", "short", "a/b" * 10, ""):
        try:
            Workspace(tmp_path, bad)
            assert False, bad
        except ValueError:
            pass


def test_everything_survives_a_restart(tmp_path, sample_pdf, make_settings):
    remote = DirBackend(tmp_path / "remote")
    # ---- first container -------------------------------------------------
    s1 = make_settings(data_dir=tmp_path / "container1")
    sync1 = SyncManager(remote, s1.data_dir, interval=0)
    ws = Workspace(s1.data_dir, new_workspace_id())
    sync1.track(ws.prefix)
    index, _ = get_index(sample_pdf, "LM7805X.pdf", s1)
    ws.add_doc(index.doc_id, index.filename)
    for pre in (f"cache/{index.doc_id}", f"documents/{index.doc_id}"):
        sync1.track(pre, pull=False)
    conv = ConversationStore(ws.dir).create([index.doc_id])
    FeedbackStore(ws.dir).record(conversation_id=conv.id, message_id="m", question="q", answer="a", rating=-1,
                                 reasons=["too long"])
    PreferenceStore(ws.dir).apply_feedback(-1, ["too long"])
    assert sync1.pending() > 0 and sync1.flush("test") and sync1.pending() == 0
    commits = remote.commits

    # ---- restart: brand-new empty disk -------------------------------------
    s2 = dataclasses.replace(s1, data_dir=tmp_path / "container2")
    sync2 = SyncManager(remote, s2.data_dir, interval=0)
    ws2 = Workspace(s2.data_dir, ws.id)
    sync2.track(ws2.prefix)
    docs = ws2.docs()
    assert [d["filename"] for d in docs] == ["LM7805X.pdf"]
    for pre in (f"cache/{docs[0]['doc_id']}", f"documents/{docs[0]['doc_id']}"):
        sync2.track(pre)
    from rag import pipeline
    pipeline._registry.clear()  # forget the in-memory copy: must come from disk
    again = open_index(docs[0]["doc_id"], docs[0]["filename"], s2)  # NO PDF needed
    assert again is not None and len(again.nodes) == len(index.nodes) and again.missing_embeddings == 0
    assert ConversationStore(ws2.dir).list()[0].id == conv.id
    assert FeedbackStore(ws2.dir).stats()["down"] == 1
    assert PreferenceStore(ws2.dir).load().response_length == "short"
    assert sync2.pending() == 0 and remote.commits == commits  # restored files are not re-uploaded


def test_deletions_are_synced(tmp_path):
    remote = DirBackend(tmp_path / "remote")
    sync = SyncManager(remote, tmp_path / "local", interval=0)
    ws = Workspace(tmp_path / "local", new_workspace_id())
    sync.track(ws.prefix)
    store = ConversationStore(ws.dir)
    conv = store.create()
    sync.flush()
    assert list((tmp_path / "remote").rglob(f"{conv.id}.json"))
    store.delete(conv.id)
    sync.flush()
    assert not list((tmp_path / "remote").rglob(f"{conv.id}.json"))


def test_failed_upload_is_retried(tmp_path):
    class Flaky(DirBackend):
        fail = True

        def push(self, adds, deletes, message):
            if self.fail:
                raise ConnectionError("offline")
            super().push(adds, deletes, message)

    remote = Flaky(tmp_path / "remote")
    sync = SyncManager(remote, tmp_path / "local", interval=0)
    ws = Workspace(tmp_path / "local", new_workspace_id())
    sync.track(ws.prefix)
    ConversationStore(ws.dir).create()
    assert not sync.flush() and sync.pending() == 1 and "retry" in sync.status().last_error
    remote.fail = False
    assert sync.flush() and sync.pending() == 0


def test_no_storage_configured_is_harmless(tmp_path):
    sync = SyncManager(None, tmp_path, interval=0)
    sync.track("users/x")
    assert sync.flush() and not sync.status().enabled
