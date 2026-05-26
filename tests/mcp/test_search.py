import pytest
import httpx
import respx

from boostmcp.config import Config
from boostmcp.embed.ollama import OllamaEmbedder
from boostmcp.mcp.search import SearchService
from boostmcp.models import DocumentRecord
from boostmcp.store import ChromaStore, SQLiteStore


@pytest.mark.asyncio
@respx.mock
async def test_search_returns_results(tmp_data_dir):
    cfg = Config(data_dir=tmp_data_dir)
    cfg.ensure_dirs()
    sqlite = SQLiteStore(cfg)
    sqlite.init_schema()
    chroma = ChromaStore(cfg)
    sqlite.insert_document(DocumentRecord(id="doc_1", filename="a.md", folder="F"))
    vec = [0.1] * 768
    chroma.upsert_chunks([{
        "id": "doc_1_0",
        "embedding": vec,
        "text": "embedding config",
        "metadata": {
            "doc_id": "doc_1",
            "filename": "a.md",
            "folder": "F",
            "tags": "x",
            "chunk_index": 0,
        },
    }])
    respx.post(f"{cfg.ollama_url}/api/embed").mock(
        return_value=httpx.Response(200, json={"embeddings": [vec]})
    )
    svc = SearchService(cfg, sqlite, chroma, OllamaEmbedder(cfg.ollama_url, cfg.ollama_model))
    results = await svc.search("how to configure embedding", top_k=5)
    assert len(results) == 1
    assert results[0].filename == "a.md"


@pytest.mark.asyncio
@respx.mock
async def test_search_empty(tmp_data_dir):
    cfg = Config(data_dir=tmp_data_dir)
    cfg.ensure_dirs()
    sqlite = SQLiteStore(cfg)
    sqlite.init_schema()
    chroma = ChromaStore(cfg)
    respx.post(f"{cfg.ollama_url}/api/embed").mock(
        return_value=httpx.Response(200, json={"embeddings": [[0.1] * 768]})
    )
    svc = SearchService(cfg, sqlite, chroma, OllamaEmbedder(cfg.ollama_url, cfg.ollama_model))
    results = await svc.search("nonexistent topic xyz", top_k=5)
    assert results == []
