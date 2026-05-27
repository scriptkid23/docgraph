from unittest.mock import AsyncMock

import pytest
import httpx
import respx

from docgraph.config import Config
from docgraph.embed.ollama import OllamaEmbedder
from docgraph.ingest.indexer import Indexer
from docgraph.models import DocumentRecord, DocumentStatus, SourceType
from docgraph.store import ChromaStore, FileStore, SQLiteStore


@pytest.mark.asyncio
@respx.mock
async def test_index_url_success(tmp_data_dir):
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

    doc = DocumentRecord(
        id="doc_url_1",
        filename="example.com_page",
        source_type=SourceType.URL,
        source_url="https://example.com/page",
    )
    sqlite.insert_document(doc)

    mock_crawler = AsyncMock()
    mock_crawler.crawl = AsyncMock(
        return_value=("# Hello\n\nWorld content for indexing.", "Hello Page")
    )

    indexer = Indexer(cfg, sqlite, files, chroma, embedder)
    await indexer._index_url_with_crawler(
        "doc_url_1", "https://example.com/page", mock_crawler
    )

    updated = sqlite.get_document("doc_url_1")
    assert updated.status == DocumentStatus.READY
    assert updated.filename == "Hello Page"
    assert updated.chunk_count >= 1
