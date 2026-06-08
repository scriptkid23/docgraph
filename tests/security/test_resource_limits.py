from __future__ import annotations

import asyncio
import time

import pytest

from docgraph.config import Config
from docgraph.mcp.search import SearchService
from docgraph.models import DocumentRecord
from docgraph.store import ChromaStore, FtsStore, SQLiteStore


class FakeEmbedder:
    async def embed(self, texts, for_query=False):
        return [[0.1] * 768 for _ in texts]


class SlowReranker:
    def __init__(self, delay):
        self.delay = delay
    async def rerank(self, q, p):
        await asyncio.sleep(self.delay)
        return [0.5] * len(p)


@pytest.fixture
def svc(tmp_path):
    cfg = Config(data_dir=tmp_path)
    cfg.hybrid_enabled = True
    cfg.rerank_enabled = False
    cfg.ensure_dirs()
    sqlite = SQLiteStore(cfg)
    sqlite.init_schema()
    chroma = ChromaStore(cfg)
    fts = FtsStore(cfg)
    for i in range(50):
        sqlite.insert_document(DocumentRecord(id=f"doc_{i}", filename=f"{i}.md"))
        chroma.upsert_chunks([{
            "id": f"doc_{i}_0", "embedding": [0.1] * 768,
            "text": f"document {i} content",
            "metadata": {"doc_id": f"doc_{i}", "filename": f"{i}.md",
                         "folder": "", "tags": "[]", "chunk_index": 0},
        }])
        fts.upsert_chunks([{
            "chunk_id": f"doc_{i}_0", "doc_id": f"doc_{i}", "folder": "", "tags": "[]",
            "chunk_index": 0, "text": f"document {i} content", "filename": f"{i}.md",
        }])
    return cfg, sqlite, chroma, fts


@pytest.mark.asyncio
async def test_huge_query_returns_in_time(svc):
    cfg, sqlite, chroma, fts = svc
    s = SearchService(cfg, sqlite, chroma, FakeEmbedder(), fts=fts, reranker=None)
    huge = "máy tính " * 10000  # ~80KB
    start = time.time()
    results = await asyncio.wait_for(s.search(huge), timeout=10.0)
    elapsed = time.time() - start
    assert elapsed < 5.0
    assert isinstance(results, list)


@pytest.mark.asyncio
async def test_rerank_timeout_bounds_total_time(svc):
    cfg, sqlite, chroma, fts = svc
    cfg.rerank_enabled = True
    cfg.rerank_score_gap_ratio = 0.99  # always rerank
    cfg.rerank_timeout_sec = 0.2
    s = SearchService(
        cfg, sqlite, chroma, FakeEmbedder(),
        fts=fts, reranker=SlowReranker(delay=5.0),
    )
    start = time.time()
    results = await s.search("document")
    elapsed = time.time() - start
    assert elapsed < 1.0
    assert len(results) > 0


@pytest.mark.asyncio
async def test_top_k_request_capped(svc):
    cfg, sqlite, chroma, fts = svc
    s = SearchService(cfg, sqlite, chroma, FakeEmbedder(), fts=fts, reranker=None)
    results = await s.search("document", top_k=10000)
    assert len(results) <= 100  # internal cap honored
