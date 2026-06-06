"""Integration tests for PATCH /api/documents/{doc_id}.

Verifies that metadata updates propagate to SQLite, Chroma, and FTS in lockstep.
Tests construct AppState directly to avoid needing a live Ollama embedder.
"""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from docgraph.config import Config
from docgraph.models import DocumentRecord
from docgraph.web.app import create_app
from docgraph.web.deps import AppState


def _make_app(tmp_path, *, hybrid_enabled: bool = True):
    cfg = Config(data_dir=tmp_path)
    cfg.hybrid_enabled = hybrid_enabled
    cfg.rerank_enabled = False
    state = AppState.create(cfg)
    app = create_app(cfg, state=state, mount_mcp=False)
    return app, state


def _seed_doc(state: AppState, doc_id: str, folder: str, tags: list[str]) -> None:
    """Insert a document into SQLite, Chroma, and (if enabled) FTS."""
    state.sqlite.insert_document(DocumentRecord(
        id=doc_id, filename="f.md", folder=folder, tags=tags,
    ))
    state.chroma.upsert_chunks([{
        "id": f"{doc_id}_0",
        "embedding": [0.1] * 768,
        "text": "hello world",
        "metadata": {
            "doc_id": doc_id,
            "filename": "f.md",
            "folder": folder,
            "tags": f'["{tags[0]}"]' if tags else "[]",
            "chunk_index": 0,
        },
    }])
    if state.fts is not None:
        import json
        state.fts.upsert_chunks([{
            "chunk_id": f"{doc_id}_0",
            "doc_id": doc_id,
            "folder": folder,
            "tags": json.dumps(tags),
            "chunk_index": 0,
            "text": "hello world",
            "filename": "f.md",
        }])


def test_patch_updates_sqlite_folder_and_tags(tmp_path):
    app, state = _make_app(tmp_path)
    _seed_doc(state, "doc_abc", "old_folder", ["orig"])

    with TestClient(app) as client:
        resp = client.patch(
            "/api/documents/doc_abc",
            data={"folder": "new_folder", "tags": "alpha,beta"},
        )
    assert resp.status_code == 200
    body = resp.json()
    assert body["folder"] == "new_folder"
    assert body["tags"] == ["alpha", "beta"]

    doc = state.sqlite.get_document("doc_abc")
    assert doc is not None
    assert doc.folder == "new_folder"
    assert doc.tags == ["alpha", "beta"]


def test_patch_updates_fts_metadata(tmp_path):
    """PATCH should update FTS rows so old-folder searches miss and new-folder hits."""
    app, state = _make_app(tmp_path, hybrid_enabled=True)
    assert state.fts is not None, "FTS must be wired when hybrid_enabled=True"
    _seed_doc(state, "doc_xyz", "old_folder", ["a"])

    # Confirm the chunk is found under old folder before PATCH
    pre_hits = state.fts.search("hello", folder="old_folder")
    assert len(pre_hits) == 1

    with TestClient(app) as client:
        resp = client.patch(
            "/api/documents/doc_xyz",
            data={"folder": "new_folder", "tags": "b,c"},
        )
    assert resp.status_code == 200

    # Old folder must no longer match
    assert state.fts.search("hello", folder="old_folder") == []
    # New folder must match
    new_hits = state.fts.search("hello", folder="new_folder")
    assert len(new_hits) >= 1
    # Tags must reflect the updated value
    for h in new_hits:
        assert h["tags"] == ["b", "c"]


def test_patch_updates_chroma_metadata(tmp_path):
    """PATCH should update Chroma metadata so old-folder vector search misses."""
    app, state = _make_app(tmp_path, hybrid_enabled=False)
    _seed_doc(state, "doc_chroma", "old_folder", ["x"])

    with TestClient(app) as client:
        resp = client.patch(
            "/api/documents/doc_chroma",
            data={"folder": "new_folder", "tags": "y"},
        )
    assert resp.status_code == 200

    # Old folder should return nothing
    old_hits = state.chroma.search([0.1] * 768, top_k=10, folder="old_folder")
    assert old_hits == []
    # New folder should find the chunk
    new_hits = state.chroma.search([0.1] * 768, top_k=10, folder="new_folder")
    assert len(new_hits) >= 1
    assert new_hits[0]["folder"] == "new_folder"
    assert new_hits[0]["tags"] == ["y"]


def test_patch_returns_404_for_unknown_doc(tmp_path):
    app, _ = _make_app(tmp_path)
    with TestClient(app) as client:
        resp = client.patch(
            "/api/documents/nonexistent",
            data={"folder": "x", "tags": ""},
        )
    assert resp.status_code == 404


def test_patch_fts_skipped_when_hybrid_disabled(tmp_path):
    """When hybrid_enabled=False, fts is None — PATCH must not crash."""
    app, state = _make_app(tmp_path, hybrid_enabled=False)
    assert state.fts is None
    _seed_doc(state, "doc_nofts", "old", ["t"])

    with TestClient(app) as client:
        resp = client.patch(
            "/api/documents/doc_nofts",
            data={"folder": "new", "tags": "t2"},
        )
    assert resp.status_code == 200
    assert resp.json()["folder"] == "new"
