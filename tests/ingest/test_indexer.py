import pytest
import httpx
import respx
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
