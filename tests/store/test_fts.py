from __future__ import annotations

import time

import pytest

from docgraph.config import Config
from docgraph.store.fts import FtsStore
from docgraph.store.sqlite import SQLiteStore


def _chunk(chunk_id, text, *, doc_id=None, folder="", filename="test.md", tags="[]"):
    if doc_id is None:
        doc_id = chunk_id.rsplit("_", 1)[0] if "_" in chunk_id else chunk_id
    chunk_index = int(chunk_id.rsplit("_", 1)[-1]) if "_" in chunk_id else 0
    return {
        "chunk_id": chunk_id,
        "doc_id": doc_id,
        "folder": folder,
        "tags": tags,
        "chunk_index": chunk_index,
        "text": text,
        "filename": filename,
    }


@pytest.fixture
def fts(tmp_path):
    cfg = Config(data_dir=tmp_path)
    cfg.ensure_dirs()
    sqlite = SQLiteStore(cfg)
    sqlite.init_schema()  # creates chunks_fts
    return FtsStore(cfg)


class TestFtsCRUD:
    def test_upsert_then_search_exact_token(self, fts):
        fts.upsert_chunks([_chunk("doc_abc_0", "DocGraph uses ChromaDB for vectors")])
        hits = fts.search("ChromaDB", top_k=10)
        assert len(hits) == 1
        assert hits[0]["chunk_id"] == "doc_abc_0"
        assert hits[0]["bm25_score"] > 0

    def test_diacritics_normalization(self, fts):
        fts.upsert_chunks([_chunk("c_0", "máy tính cá nhân")])
        hits = fts.search("may tinh", top_k=10)
        assert len(hits) == 1

    def test_identifier_with_underscore_preserved(self, fts):
        fts.upsert_chunks([
            _chunk("a_0", "Call embed_query() to embed user input"),
            _chunk("b_0", "The embed function takes text and returns vectors"),
        ])
        hits = fts.search("embed_query", top_k=10)
        assert len(hits) >= 1
        assert hits[0]["chunk_id"] == "a_0"

    def test_filename_weighted_higher(self, fts):
        fts.upsert_chunks([
            _chunk("a_0", "various config options here", filename="install.md"),
            _chunk("b_0", "various install options here", filename="config.md"),
        ])
        hits = fts.search("config", top_k=10)
        assert hits[0]["chunk_id"] == "b_0"

    def test_delete_by_doc_id_removes_all_chunks_of_doc(self, fts):
        fts.upsert_chunks(
            [_chunk(f"doc_abc_{i}", f"chunk {i} of doc_abc", doc_id="doc_abc") for i in range(5)]
        )
        fts.upsert_chunks([_chunk("doc_xyz_0", "other document content", doc_id="doc_xyz")])
        fts.delete_by_doc_id("doc_abc")
        hits = fts.search("chunk", top_k=20)
        assert all(h["doc_id"] != "doc_abc" for h in hits)
        hits = fts.search("other", top_k=20)
        assert len(hits) == 1
        assert hits[0]["doc_id"] == "doc_xyz"

    def test_folder_filter(self, fts):
        fts.upsert_chunks([
            _chunk("a_0", "shared content here", folder="docs"),
            _chunk("b_0", "shared content here", folder="code"),
        ])
        hits = fts.search("shared", top_k=10, folder="docs")
        assert len(hits) == 1
        assert hits[0]["chunk_id"] == "a_0"

    def test_empty_query_returns_empty(self, fts):
        fts.upsert_chunks([_chunk("a_0", "any text whatsoever")])
        assert fts.search("", top_k=10) == []
        assert fts.search("   ", top_k=10) == []
        assert fts.search("***", top_k=10) == []

    def test_batch_executemany_perf_1000_chunks(self, fts):
        chunks = [_chunk(f"d_{i}", f"chunk text number {i}") for i in range(1000)]
        start = time.time()
        fts.upsert_chunks(chunks)
        elapsed = time.time() - start
        assert elapsed < 2.0, f"batch insert of 1000 took {elapsed:.2f}s"
        assert fts.count_chunks() == 1000

    def test_clear_removes_everything(self, fts):
        fts.upsert_chunks([_chunk(f"d_{i}", "text") for i in range(10)])
        assert fts.count_chunks() == 10
        fts.clear()
        assert fts.count_chunks() == 0


import asyncio
import pytest

from docgraph.store.chroma import ChromaStore


@pytest.fixture
def chroma_and_sqlite(tmp_path):
    cfg = Config(data_dir=tmp_path)
    cfg.ensure_dirs()
    sqlite = SQLiteStore(cfg)
    sqlite.init_schema()
    chroma = ChromaStore(cfg)
    # Seed 3 docs (each 1 chunk) into SQLite + Chroma
    from docgraph.models import DocumentRecord
    for i, name in enumerate(["a", "b", "c"]):
        sqlite.insert_document(DocumentRecord(
            id=f"doc_{name}", filename=f"{name}.md", folder=f"f{i}", tags=[f"t{i}"]
        ))
    chroma.upsert_chunks([
        {"id": f"doc_{name}_0", "embedding": [0.1] * 768, "text": f"text {name}",
         "metadata": {"doc_id": f"doc_{name}", "filename": f"{name}.md",
                      "folder": f"f{i}", "tags": f'["t{i}"]', "chunk_index": 0}}
        for i, name in enumerate(["a", "b", "c"])
    ])
    return cfg, sqlite, chroma


@pytest.mark.asyncio
async def test_rebuild_from_chroma_populates_fts(chroma_and_sqlite):
    cfg, sqlite, chroma = chroma_and_sqlite
    fts = FtsStore(cfg)
    assert fts.count_chunks() == 0
    n = await fts.rebuild_from_chroma(chroma, sqlite)
    assert n == 3
    assert fts.count_chunks() == 3
    hits = fts.search("text", top_k=10)
    assert len(hits) == 3


@pytest.mark.asyncio
async def test_rebuild_clears_old_rows(chroma_and_sqlite):
    cfg, sqlite, chroma = chroma_and_sqlite
    fts = FtsStore(cfg)
    fts.upsert_chunks([_chunk("stale_0", "stale text", doc_id="stale")])
    assert fts.count_chunks() == 1
    await fts.rebuild_from_chroma(chroma, sqlite)
    hits = fts.search("stale", top_k=10)
    assert hits == []
    assert fts.count_chunks() == 3
