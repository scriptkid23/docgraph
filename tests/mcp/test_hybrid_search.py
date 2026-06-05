from __future__ import annotations

import asyncio

import pytest

from docgraph.config import Config
from docgraph.mcp.search import SearchService
from docgraph.models import DocumentRecord
from docgraph.store import ChromaStore, FtsStore, SQLiteStore


class FakeEmbedder:
    async def embed(self, texts, for_query=False):
        return [[0.1] * 768 for _ in texts]

    async def health_check(self):
        return None


class FakeReranker:
    def __init__(self, score_map=None):
        self.score_map = score_map or {}
        self.call_count = 0

    async def rerank(self, query, passages):
        self.call_count += 1
        return [self.score_map.get(p, 0.5) for p in passages]

    async def prewarm(self):
        return None


class BrokenReranker:
    async def rerank(self, query, passages):
        raise RuntimeError("rerank failed")

    async def prewarm(self):
        return None


class SlowReranker:
    def __init__(self, delay):
        self.delay = delay

    async def rerank(self, query, passages):
        await asyncio.sleep(self.delay)
        return [0.0] * len(passages)

    async def prewarm(self):
        return None


@pytest.fixture
def seeded(tmp_path):
    cfg = Config(data_dir=tmp_path)
    cfg.hybrid_enabled = True
    cfg.rerank_enabled = True
    cfg.rerank_timeout_sec = 1.0
    cfg.ensure_dirs()
    sqlite = SQLiteStore(cfg)
    sqlite.init_schema()
    chroma = ChromaStore(cfg)
    fts = FtsStore(cfg)
    docs = [
        ("doc_a", "DocGraph stores vectors in ChromaDB"),
        ("doc_b", "User interface built with React and Vite"),
        ("doc_c", "Indexer converts files to markdown"),
    ]
    for did, text in docs:
        sqlite.insert_document(DocumentRecord(id=did, filename=f"{did}.md", folder="x"))
        chroma.upsert_chunks([{
            "id": f"{did}_0", "embedding": [0.1] * 768, "text": text,
            "metadata": {"doc_id": did, "filename": f"{did}.md", "folder": "x",
                         "tags": "[]", "chunk_index": 0},
        }])
        fts.upsert_chunks([{
            "chunk_id": f"{did}_0", "doc_id": did, "folder": "x", "tags": "[]",
            "chunk_index": 0, "text": text, "filename": f"{did}.md",
        }])
    return cfg, sqlite, chroma, fts


class TestHybridPipeline:
    @pytest.mark.asyncio
    async def test_hybrid_returns_results(self, seeded):
        cfg, sqlite, chroma, fts = seeded
        svc = SearchService(cfg, sqlite, chroma, FakeEmbedder(), fts=fts, reranker=None)
        results = await svc.search("DocGraph")
        assert len(results) > 0
        assert any("DocGraph" in r.text or "vectors" in r.text for r in results)

    @pytest.mark.asyncio
    async def test_disable_hybrid_skips_fts(self, seeded, monkeypatch):
        cfg, sqlite, chroma, fts = seeded
        cfg.hybrid_enabled = False
        call_count = {"n": 0}
        orig_search = fts.search
        def tracked(*args, **kwargs):
            call_count["n"] += 1
            return orig_search(*args, **kwargs)
        monkeypatch.setattr(fts, "search", tracked)
        svc = SearchService(cfg, sqlite, chroma, FakeEmbedder(), fts=fts, reranker=None)
        await svc.search("test")
        assert call_count["n"] == 0

    @pytest.mark.asyncio
    async def test_rerank_invoked_changes_top_result(self, seeded):
        cfg, sqlite, chroma, fts = seeded
        cfg.rerank_score_gap_ratio = 0.99
        # Score doc_c highest via rerank so it surfaces above doc_a/doc_b.
        reranker = FakeReranker(score_map={
            "DocGraph stores vectors in ChromaDB": 0.1,
            "User interface built with React and Vite": 0.2,
            "Indexer converts files to markdown": 0.99,
        })
        svc = SearchService(cfg, sqlite, chroma, FakeEmbedder(), fts=fts, reranker=reranker)
        results = await svc.search("DocGraph React")
        assert reranker.call_count == 1
        assert results[0].doc_id == "doc_c"

    @pytest.mark.asyncio
    async def test_rerank_failure_falls_back_to_rrf(self, seeded):
        cfg, sqlite, chroma, fts = seeded
        cfg.rerank_score_gap_ratio = 0.99
        svc = SearchService(
            cfg, sqlite, chroma, FakeEmbedder(), fts=fts, reranker=BrokenReranker()
        )
        results = await svc.search("DocGraph React")
        assert len(results) > 0  # no exception propagated

    @pytest.mark.asyncio
    async def test_rerank_timeout_falls_back_to_rrf(self, seeded):
        cfg, sqlite, chroma, fts = seeded
        cfg.rerank_score_gap_ratio = 0.99
        cfg.rerank_timeout_sec = 0.1
        svc = SearchService(
            cfg, sqlite, chroma, FakeEmbedder(), fts=fts, reranker=SlowReranker(delay=2.0)
        )
        import time
        start = time.time()
        results = await svc.search("DocGraph React")
        elapsed = time.time() - start
        assert elapsed < 2.0  # bounded by timeout (5× margin under SlowReranker delay)
        assert len(results) > 0

    @pytest.mark.asyncio
    async def test_empty_text_hits_excluded_from_rerank(self, seeded):
        cfg, sqlite, chroma, fts = seeded
        cfg.rerank_score_gap_ratio = 0.99
        received_passages: list[list[str]] = []

        class CapturingReranker:
            async def rerank(self, query, passages):
                received_passages.append(list(passages))
                return [0.5] * len(passages)

            async def prewarm(self):
                return None

        svc = SearchService(
            cfg, sqlite, chroma, FakeEmbedder(), fts=fts, reranker=CapturingReranker()
        )
        # Inject a sparse-only chunk into FTS with no Chroma counterpart, so backfill
        # cannot find text for it.
        fts.upsert_chunks([{
            "chunk_id": "ghost_0", "doc_id": "ghost", "folder": "x", "tags": "[]",
            "chunk_index": 0, "text": "this text lives only in FTS5",
            "filename": "ghost.md",
        }])
        await svc.search("FTS5")
        assert received_passages, "reranker was never called"
        assert len(received_passages[0]) == 3, (
            f"expected 3 non-ghost passages, got {len(received_passages[0])}: "
            f"{received_passages[0]}"
        )
        assert all(p for p in received_passages[0]), (
            "reranker received an empty-string passage — sparse-only hits without "
            f"vector backfill should be excluded. Got: {received_passages[0]}"
        )
