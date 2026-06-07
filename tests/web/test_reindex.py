"""Tests for POST /api/documents/{doc_id}/reindex concurrency guard.

The FTS upsert path is plain INSERT (see docgraph/store/fts.py docstring),
so two concurrent indexing passes on the same doc duplicate every row.
The endpoint must refuse a reindex when the doc is already PROCESSING.
"""
from __future__ import annotations

from fastapi.testclient import TestClient

from docgraph.config import Config
from docgraph.models import DocumentRecord, DocumentStatus
from docgraph.web.app import create_app
from docgraph.web.deps import AppState


def _make_app(tmp_path):
    cfg = Config(data_dir=tmp_path)
    cfg.hybrid_enabled = True
    cfg.rerank_enabled = False
    state = AppState.create(cfg)
    app = create_app(cfg, state=state, mount_mcp=False)
    return app, state


def test_reindex_refuses_when_already_processing(tmp_path):
    app, state = _make_app(tmp_path)
    state.sqlite.insert_document(
        DocumentRecord(id="doc_busy", filename="f.pdf", folder="", tags=[])
    )
    # Default status after insert is PROCESSING — simulate an in-flight upload
    doc = state.sqlite.get_document("doc_busy")
    assert doc.status == DocumentStatus.PROCESSING

    with TestClient(app) as client:
        resp = client.post("/api/documents/doc_busy/reindex")
    assert resp.status_code == 409
    assert "already" in resp.json()["detail"].lower()


def test_reindex_accepts_when_ready(tmp_path):
    app, state = _make_app(tmp_path)
    state.sqlite.insert_document(
        DocumentRecord(id="doc_ready", filename="f.pdf", folder="", tags=[])
    )
    # Flip to READY so the claim can succeed
    state.sqlite.update_status("doc_ready", DocumentStatus.READY, chunk_count=1)

    # raise_server_exceptions=False: the BG task itself will fail (no real
    # file to reindex), but that is downstream of what we are testing —
    # we only care that the endpoint returned 202 and flipped status to
    # PROCESSING via claim_for_reindex.
    with TestClient(app, raise_server_exceptions=False) as client:
        resp = client.post("/api/documents/doc_ready/reindex")
    assert resp.status_code == 202
    # Either PROCESSING (claim landed, BG still running) or ERROR (BG ran
    # and failed because the source file does not exist). Both prove the
    # claim succeeded; only READY would mean the guard rejected us.
    final = state.sqlite.get_document("doc_ready").status
    assert final in (DocumentStatus.PROCESSING, DocumentStatus.ERROR)


def test_reindex_404_when_missing(tmp_path):
    app, _ = _make_app(tmp_path)
    with TestClient(app) as client:
        resp = client.post("/api/documents/doc_ghost/reindex")
    assert resp.status_code == 404


def test_claim_for_reindex_is_atomic(tmp_path):
    """Two consecutive claims on a READY doc — first wins, second returns False."""
    _, state = _make_app(tmp_path)
    state.sqlite.insert_document(
        DocumentRecord(id="doc_race", filename="f.pdf", folder="", tags=[])
    )
    state.sqlite.update_status("doc_race", DocumentStatus.READY, chunk_count=1)

    assert state.sqlite.claim_for_reindex("doc_race") is True
    assert state.sqlite.claim_for_reindex("doc_race") is False
    # Status is PROCESSING after first claim
    assert state.sqlite.get_document("doc_race").status == DocumentStatus.PROCESSING
