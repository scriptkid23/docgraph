from __future__ import annotations

from fastapi.testclient import TestClient

from docgraph.config import Config
from docgraph.web.app import create_app
from docgraph.web.deps import AppState


def test_health_keeps_existing_fields(tmp_path):
    """Backward compat: pre-existing /api/health fields must still appear."""
    cfg = Config(data_dir=tmp_path)
    state = AppState.create(cfg)
    app = create_app(cfg, state=state, mount_mcp=False)
    with TestClient(app) as client:
        resp = client.get("/api/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert "ollama" in body
    assert body["embed_provider"] == cfg.embed_provider
    assert "mcp_sse_url" in body


def test_health_reports_hybrid_status_enabled(tmp_path):
    cfg = Config(data_dir=tmp_path)
    cfg.hybrid_enabled = True
    cfg.rerank_enabled = False
    state = AppState.create(cfg)
    app = create_app(cfg, state=state, mount_mcp=False)
    with TestClient(app) as client:
        resp = client.get("/api/health")
    body = resp.json()
    assert body["hybrid_enabled"] is True
    assert body["rerank_enabled"] is False
    assert body["rerank_status"] == "disabled"
    assert body["chroma_chunks"] == 0
    assert body["fts_chunks"] == 0  # fts is wired (hybrid_enabled=True), table empty
    assert body["fts_in_sync"] is True  # both zero → in sync


def test_health_reports_fts_null_when_hybrid_disabled(tmp_path):
    cfg = Config(data_dir=tmp_path)
    cfg.hybrid_enabled = False
    cfg.rerank_enabled = False
    state = AppState.create(cfg)
    app = create_app(cfg, state=state, mount_mcp=False)
    with TestClient(app) as client:
        resp = client.get("/api/health")
    body = resp.json()
    assert body["hybrid_enabled"] is False
    assert body["fts_chunks"] is None
    assert body["fts_in_sync"] is None


def test_health_reports_rerank_status_loading_before_init(tmp_path):
    cfg = Config(data_dir=tmp_path)
    cfg.rerank_enabled = True
    cfg.rerank_prewarm = False  # don't actually load anything
    cfg.hybrid_enabled = False
    state = AppState.create(cfg)
    # state.reranker exists but _initialized is False by default
    assert state.reranker is not None
    assert state.reranker._initialized is False
    app = create_app(cfg, state=state, mount_mcp=False)
    with TestClient(app) as client:
        resp = client.get("/api/health")
    assert resp.json()["rerank_status"] == "loading"


def test_health_reports_rerank_status_ready_after_init(tmp_path):
    cfg = Config(data_dir=tmp_path)
    cfg.rerank_enabled = True
    cfg.rerank_prewarm = False
    cfg.hybrid_enabled = False
    state = AppState.create(cfg)
    # Simulate reranker initialized (avoid actually loading Rust model)
    state.reranker._initialized = True
    app = create_app(cfg, state=state, mount_mcp=False)
    with TestClient(app) as client:
        resp = client.get("/api/health")
    assert resp.json()["rerank_status"] == "ready"


def test_health_fts_in_sync_false_when_drift_exceeds_threshold(tmp_path):
    """When chroma and fts diverge by > max(10, chroma*0.05), fts_in_sync is False.
    Both stores are populated to avoid triggering the lifespan auto-rebuild
    (which would equalize counts before the health check fires)."""
    cfg = Config(data_dir=tmp_path)
    cfg.hybrid_enabled = True
    cfg.rerank_enabled = False
    state = AppState.create(cfg)

    from docgraph.models import DocumentRecord
    state.sqlite.insert_document(DocumentRecord(
        id="doc_x", filename="x.md", folder="", tags=[]
    ))
    chunks_chroma = [{
        "id": f"doc_x_{i}", "embedding": [0.1] * 768, "text": f"t{i}",
        "metadata": {"doc_id": "doc_x", "filename": "x.md", "folder": "",
                     "tags": "[]", "chunk_index": i},
    } for i in range(200)]
    state.chroma.upsert_chunks(chunks_chroma)
    # Seed FTS with only 100 chunks — drift = 100, threshold = max(10, 200*0.05=10) = 10
    chunks_fts = [{
        "chunk_id": f"doc_x_{i}", "doc_id": "doc_x", "folder": "",
        "tags": "[]", "chunk_index": i, "text": f"t{i}", "filename": "x.md",
    } for i in range(100)]
    state.fts.upsert_chunks(chunks_fts)

    app = create_app(cfg, state=state, mount_mcp=False)
    with TestClient(app) as client:
        resp = client.get("/api/health")
    body = resp.json()
    assert body["chroma_chunks"] == 200
    assert body["fts_chunks"] == 100
    # diff = 100, threshold = 10, 100 > 10 → not in sync
    assert body["fts_in_sync"] is False


def test_health_fts_in_sync_true_at_boundary(tmp_path):
    """diff == threshold should still be in_sync (uses <=, not <)."""
    cfg = Config(data_dir=tmp_path)
    cfg.hybrid_enabled = True
    cfg.rerank_enabled = False
    state = AppState.create(cfg)

    from docgraph.models import DocumentRecord
    state.sqlite.insert_document(DocumentRecord(
        id="doc_x", filename="x.md", folder="", tags=[]
    ))
    # 100 chroma vs 90 fts → diff=10, threshold=max(10, 5)=10 → in_sync (boundary)
    chunks_chroma = [{
        "id": f"doc_x_{i}", "embedding": [0.1] * 768, "text": f"t{i}",
        "metadata": {"doc_id": "doc_x", "filename": "x.md", "folder": "",
                     "tags": "[]", "chunk_index": i},
    } for i in range(100)]
    state.chroma.upsert_chunks(chunks_chroma)
    chunks_fts = [{
        "chunk_id": f"doc_x_{i}", "doc_id": "doc_x", "folder": "",
        "tags": "[]", "chunk_index": i, "text": f"t{i}", "filename": "x.md",
    } for i in range(90)]
    state.fts.upsert_chunks(chunks_fts)

    app = create_app(cfg, state=state, mount_mcp=False)
    with TestClient(app) as client:
        resp = client.get("/api/health")
    body = resp.json()
    assert body["chroma_chunks"] == 100
    assert body["fts_chunks"] == 90
    assert body["fts_in_sync"] is True  # diff=10 <= threshold=10 (boundary)


def test_health_rerank_status_error_when_misconfigured(tmp_path):
    """cfg.rerank_enabled=True but state.reranker=None should report 'error'."""
    cfg = Config(data_dir=tmp_path)
    cfg.rerank_enabled = True
    cfg.rerank_prewarm = False
    cfg.hybrid_enabled = False
    state = AppState.create(cfg)
    # Forcibly remove reranker to simulate misconfig
    state.reranker = None
    app = create_app(cfg, state=state, mount_mcp=False)
    with TestClient(app) as client:
        resp = client.get("/api/health")
    assert resp.json()["rerank_status"] == "error"
