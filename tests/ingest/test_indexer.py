import pytest
import httpx
import respx
import json as _json
from pathlib import Path

from docgraph.config import Config
from docgraph.embed.ollama import OllamaEmbedder
from docgraph.ingest.indexer import Indexer
from docgraph.models import DocumentRecord, DocumentStatus
from docgraph.store import ChromaStore, FileStore, SQLiteStore

FIXTURES = Path(__file__).parent.parent / "fixtures"


@pytest.mark.asyncio
@respx.mock
async def test_index_document_success(tmp_data_dir):
    cfg = Config(data_dir=tmp_data_dir)
    cfg.ensure_dirs()
    sqlite = SQLiteStore(cfg)
    sqlite.init_schema()
    files = FileStore(cfg)
    chroma = ChromaStore(cfg)
    embedder = OllamaEmbedder(cfg.ollama_url, cfg.ollama_model)

    respx.post(f"{cfg.ollama_url}/api/embed").mock(
        return_value=httpx.Response(200, json={"embeddings": [[0.1] * 768]})
    )

    doc = DocumentRecord(id="doc_1", filename="sample.md", folder="test", tags=["t1"])
    sqlite.insert_document(doc)
    orig = files.save_original("doc_1", "sample.md", (FIXTURES / "sample.md").read_bytes())

    indexer = Indexer(cfg, sqlite, files, chroma, embedder)
    await indexer.index_document("doc_1", orig)

    updated = sqlite.get_document("doc_1")
    assert updated.status == DocumentStatus.READY
    assert updated.chunk_count >= 1
    results = chroma.search(query_embedding=[0.1] * 768, top_k=1)
    assert len(results) == 1


@pytest.mark.asyncio
@respx.mock
async def test_index_document_routes_repomix_dump(tmp_data_dir):
    cfg = Config(data_dir=tmp_data_dir)
    cfg.ensure_dirs()
    sqlite = SQLiteStore(cfg)
    sqlite.init_schema()
    files = FileStore(cfg)
    chroma = ChromaStore(cfg)
    embedder = OllamaEmbedder(cfg.ollama_url, cfg.ollama_model)

    def _embed(request):
        n = len(_json.loads(request.content)["input"])
        return httpx.Response(200, json={"embeddings": [[0.1] * 768] * n})

    respx.post(f"{cfg.ollama_url}/api/embed").mock(side_effect=_embed)

    dump = (
        "================\nFile: src/a.py\n================\n"
        "def a():\n    return 1\n\n"
        "================\nFile: src/b.py\n================\n"
        "def b():\n    return 2\n"
    )
    doc = DocumentRecord(id="doc_cd", filename="repo.txt", folder="", tags=[])
    sqlite.insert_document(doc)
    orig = files.save_original("doc_cd", "repo.txt", dump.encode("utf-8"))

    indexer = Indexer(cfg, sqlite, files, chroma, embedder)
    await indexer.index_document("doc_cd", orig)

    updated = sqlite.get_document("doc_cd")
    assert updated.status == DocumentStatus.READY
    assert updated.chunk_count >= 2
    results = chroma.search(query_embedding=[0.1] * 768, top_k=5)
    file_paths = {r["file_path"] for r in results}
    assert {"src/a.py", "src/b.py"} <= file_paths
