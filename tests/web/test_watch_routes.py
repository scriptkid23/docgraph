from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from docgraph.config import Config
from docgraph.web.app import create_app
from docgraph.web.deps import AppState


@pytest.fixture
def client(tmp_path: Path):
    cfg = Config(data_dir=tmp_path / "data")
    cfg.rerank_prewarm = False  # don't load Rust model in tests
    cfg.ensure_dirs()
    state = AppState.create(cfg)
    app = create_app(cfg, state=state, mount_mcp=False)
    with TestClient(app) as c:
        yield c, state, tmp_path


def test_status_disabled_initial(client):
    c, _, _ = client
    r = c.get("/api/watch/status")
    assert r.status_code == 200
    body = r.json()
    assert body["enabled"] is False
    assert body["dirs_count"] == 0


def test_enable_then_disable(client):
    c, _, _ = client
    r = c.post("/api/watch/enable")
    assert r.status_code in (200, 202)
    body = r.json()
    assert body["enabled"] is True
    r = c.get("/api/watch/status")
    assert r.json()["enabled"] is True
    r = c.post("/api/watch/disable")
    assert r.status_code == 200
    assert r.json()["enabled"] is False


def test_enable_idempotent(client):
    c, _, _ = client
    c.post("/api/watch/enable")
    r = c.post("/api/watch/enable")
    assert r.status_code in (200, 202)
    c.post("/api/watch/disable")
