from __future__ import annotations

import pytest
import httpx
import respx
from pathlib import Path

from docgraph.config import Config
from docgraph.embed.ollama import OllamaEmbedder
from docgraph.ingest.indexer import Indexer
from docgraph.models import DocumentRecord, DocumentStatus
from docgraph.store import ChromaStore, FileStore, FtsStore, SQLiteStore

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


class FakeEmbedder:
    async def embed(self, texts, for_query=False):
        return [[0.1] * 768 for _ in texts]

    async def health_check(self):
        return None


@pytest.fixture
def pipeline(tmp_path):
    cfg = Config(data_dir=tmp_path)
    cfg.ensure_dirs()
    sqlite = SQLiteStore(cfg)
    sqlite.init_schema()
    files = FileStore(cfg)
    chroma = ChromaStore(cfg)
    fts = FtsStore(cfg)
    embedder = FakeEmbedder()
    indexer = Indexer(cfg, sqlite, files, chroma, embedder, fts=fts)
    return cfg, sqlite, chroma, fts, indexer


@pytest.mark.asyncio
async def test_index_markdown_writes_to_fts(pipeline):
    cfg, sqlite, chroma, fts, indexer = pipeline
    sqlite.insert_document(DocumentRecord(
        id="doc_test", filename="t.md", folder="x", tags=["v1"]
    ))
    await indexer.index_markdown(
        "doc_test", "## Section\n\nThis is text with embed_query identifier here."
    )
    assert chroma.count_chunks() > 0
    assert fts.count_chunks() == chroma.count_chunks()
    # Verify identifier searchable in FTS
    hits = fts.search("embed_query", top_k=5)
    assert len(hits) >= 1
    assert hits[0]["doc_id"] == "doc_test"


@pytest.mark.asyncio
async def test_reindex_clears_old_fts_rows(pipeline):
    cfg, sqlite, chroma, fts, indexer = pipeline
    sqlite.insert_document(DocumentRecord(id="doc_re", filename="r.md"))
    # First index
    await indexer.index_markdown("doc_re", "First version text about apples")
    assert fts.count_chunks() > 0
    # Delete from chroma + fts (simulating user re-index) and re-index with different content
    chroma.delete_by_doc_id("doc_re")
    fts.delete_by_doc_id("doc_re")
    sqlite.update_status("doc_re", DocumentStatus.PROCESSING)
    await indexer.index_markdown("doc_re", "Completely different content about oranges here")
    # Old chunks gone from FTS, new ones inserted
    hits = fts.search("apples", top_k=5)
    assert hits == []
    hits = fts.search("oranges", top_k=5)
    assert len(hits) >= 1
