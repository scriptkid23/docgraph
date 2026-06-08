import io

import pytest
import httpx
import respx
from httpx import ASGITransport, AsyncClient

from docgraph.config import Config
from docgraph.mcp.search import SearchService
from docgraph.web.app import create_app


@pytest.mark.asyncio
@respx.mock
async def test_upload_then_search_e2e(tmp_data_dir):
    cfg = Config(data_dir=tmp_data_dir)
    app = create_app(cfg)

    vec = [0.2] * 768
    respx.post(f"{cfg.ollama_url}/api/embed").mock(
        return_value=httpx.Response(200, json={"embeddings": [vec]})
    )

    content = b"# DocGraph v2\n\nDocument RAG with Ollama embedding."
    transport = ASGITransport(app=app)

    async with AsyncClient(transport=transport, base_url="http://test") as client:
        upload = await client.post(
            "/api/documents",
            files={"file": ("v2.md", io.BytesIO(content), "text/markdown")},
            data={"folder": "docs", "tags": "v2,rag"},
        )
        assert upload.status_code == 202
        doc_id = upload.json()["doc_id"]

        import asyncio
        doc = None
        for _ in range(30):
            docs = await client.get("/api/documents")
            doc = next(d for d in docs.json() if d["id"] == doc_id)
            if doc["status"] in ("ready", "error"):
                break
            await asyncio.sleep(0.1)

        assert doc["status"] == "ready"
        assert doc["chunk_count"] >= 1

    st = app.state.docgraph
    svc = SearchService(cfg, st.sqlite, st.chroma, st.embedder)
    results = await svc.search("Ollama embedding", folder="docs")
    assert len(results) >= 1
    assert "Ollama" in results[0].text


@pytest.mark.asyncio
async def test_hybrid_e2e_identifier_query(tmp_path):
    """Upload markdown with code identifier, query exact symbol, verify hit."""
    from docgraph.config import Config
    from docgraph.models import DocumentRecord
    from docgraph.web.deps import AppState

    cfg = Config(data_dir=tmp_path)
    cfg.hybrid_enabled = True
    cfg.rerank_enabled = False  # skip rerank to keep test fast + deterministic
    state = AppState.create(cfg)

    markdown = (
        "## Embedding\n\n"
        "Call `embed_query(text)` to embed user queries.\n\n"
        "## Vector storage\n\n"
        "ChromaDB stores embeddings with cosine similarity."
    )
    state.sqlite.insert_document(DocumentRecord(
        id="doc_e2e", filename="readme.md", folder="docs", tags=["test"]
    ))
    await state.indexer().index_markdown("doc_e2e", markdown)
    assert state.fts.count_chunks() > 0

    results = await state.search_service().search("embed_query", top_k=3)
    assert len(results) >= 1
    # Identifier should rank chunk 0 (the "embed_query" line) first or near top
    top_text = results[0].text
    assert "embed_query" in top_text
