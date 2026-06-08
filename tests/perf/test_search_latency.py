from __future__ import annotations

import pytest

from docgraph.config import Config
from docgraph.mcp.search import _rrf_fuse
from docgraph.store.fts import FtsStore, _sanitize_query
from docgraph.store.sqlite import SQLiteStore


@pytest.fixture
def fts_1000(tmp_path):
    cfg = Config(data_dir=tmp_path)
    cfg.ensure_dirs()
    sqlite = SQLiteStore(cfg)
    sqlite.init_schema()
    f = FtsStore(cfg)
    f.upsert_chunks([
        {"chunk_id": f"d_{i}", "doc_id": f"d{i // 10}", "folder": "x", "tags": "[]",
         "chunk_index": i % 10, "text": f"chunk text number {i} content sample",
         "filename": "f.md"}
        for i in range(1000)
    ])
    return f


@pytest.mark.benchmark
def test_fts_search_1000_chunks(benchmark, fts_1000):
    result = benchmark(lambda: fts_1000.search("chunk text", top_k=50))
    assert len(result) > 0
    median = benchmark.stats["median"]
    assert median < 0.100, f"FTS5 search median {median*1000:.1f}ms > budget 100ms"


@pytest.mark.benchmark
def test_rrf_fusion_30_each(benchmark):
    vec = [{
        "id": f"v{i}", "score": 1.0 / (i + 1), "text": "t", "doc_id": f"d{i}",
        "filename": "f", "folder": "", "tags": [], "chunk_index": i, "source_page": None,
    } for i in range(30)]
    sparse = [{
        "chunk_id": f"s{i}", "bm25_score": 1.0 / (i + 1), "doc_id": f"d{i}",
        "folder": "", "tags": [], "chunk_index": i,
    } for i in range(30)]
    result = benchmark(lambda: _rrf_fuse(vec, sparse, k_rrf=60))
    assert len(result) > 0
    assert benchmark.stats["median"] < 0.005


@pytest.mark.benchmark
def test_sanitize_long_query(benchmark):
    long_q = "máy tính " * 100
    benchmark(lambda: _sanitize_query(long_q))
    assert benchmark.stats["median"] < 0.005
