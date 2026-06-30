from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi.testclient import TestClient

from docgraph.config import Config
from docgraph.repo.codegraph_client import CodegraphNotInstalled
from docgraph.web.app import create_app
from docgraph.web.deps import AppState


@pytest.fixture
def client(tmp_data_dir):
    cfg = Config(data_dir=tmp_data_dir)
    # Skip the lifespan auto-rebuild + heavy reranker load by disabling hybrid + rerank.
    cfg.hybrid_enabled = False
    cfg.rerank_enabled = False
    state = AppState.create(cfg)
    state.codegraph.health_check = AsyncMock(return_value="codegraph 0.5.1-test")
    state.codegraph.init = AsyncMock()
    # Replace indexer with a stub so the background task doesn't pull in fastembed-rs.
    fake_indexer = MagicMock()
    fake_indexer.index_markdown = AsyncMock()
    state._indexer = fake_indexer
    app = create_app(cfg, state=state, mount_mcp=False)
    with TestClient(app) as c:
        yield c, state, cfg


def _populate_repo(repo_dir: Path) -> None:
    repo_dir.mkdir(parents=True, exist_ok=True)
    (repo_dir / ".git").mkdir()
    (repo_dir / "README.md").write_text("# Hi")


def test_create_repo_returns_202(client, tmp_data_dir):
    c, state, _ = client
    local = tmp_data_dir / "src_repo"
    _populate_repo(local)
    state.codegraph.init = AsyncMock()
    resp = c.post("/api/repos", json={"source": str(local)})
    assert resp.status_code == 202
    body = resp.json()
    assert body["repo_id"].startswith("repo_")


def test_list_get_delete_repo(client, tmp_data_dir):
    c, state, _ = client
    local = tmp_data_dir / "src_repo"
    _populate_repo(local)
    state.codegraph.init = AsyncMock()
    rid = c.post("/api/repos", json={"source": str(local)}).json()["repo_id"]
    assert c.get("/api/repos").status_code == 200
    assert c.get(f"/api/repos/{rid}").status_code == 200
    assert c.get("/api/repos/repo_unknown").status_code == 404
    d = c.delete(f"/api/repos/{rid}").json()
    assert d["deleted"] == rid


def test_create_repo_503_when_codegraph_missing(client, tmp_data_dir):
    c, state, _ = client
    state.codegraph.health_check = AsyncMock(
        side_effect=CodegraphNotInstalled("nope; run install.sh")
    )
    local = tmp_data_dir / "src_repo"
    _populate_repo(local)
    resp = c.post("/api/repos", json={"source": str(local)})
    assert resp.status_code == 503
    assert "install" in resp.json()["detail"].lower()


def test_health_reports_codegraph(client):
    c, _, _ = client
    body = c.get("/api/health").json()
    assert body["codegraph"]["ok"] is True
    assert "0.5.1" in body["codegraph"]["version"]
