import io

import pytest
from httpx import ASGITransport, AsyncClient

from boostmcp.config import Config
from boostmcp.web.app import create_app


@pytest.fixture
def app(tmp_data_dir):
    cfg = Config(data_dir=tmp_data_dir)
    cfg.ensure_dirs()
    return create_app(cfg)


@pytest.mark.asyncio
async def test_health_endpoint(app):
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get("/api/health")
    assert resp.status_code == 200
    body = resp.json()
    assert "status" in body
    assert "mcp_sse_url" in body


def test_mcp_sse_mounted(app):
    mount_paths = [
        getattr(route, "path", "")
        for route in app.routes
        if getattr(route, "path", None)
    ]
    assert "/mcp" in mount_paths


@pytest.mark.asyncio
async def test_list_documents_empty(app):
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get("/api/documents")
    assert resp.status_code == 200
    assert resp.json() == []


@pytest.mark.asyncio
async def test_upload_document(app):
    transport = ASGITransport(app=app)
    content = b"# Hello\n\nWorld"
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.post(
            "/api/documents",
            files={"file": ("hello.md", io.BytesIO(content), "text/markdown")},
            data={"folder": "test", "tags": "demo,md"},
        )
    assert resp.status_code == 202
    body = resp.json()
    assert "doc_id" in body
    assert body["status"] == "processing"


@pytest.mark.asyncio
async def test_upload_rejects_oversized(app):
    app.state.boostmcp.cfg.max_file_size_mb = 0
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.post(
            "/api/documents",
            files={"file": ("big.md", io.BytesIO(b"x" * 100), "text/markdown")},
        )
    assert resp.status_code == 413
