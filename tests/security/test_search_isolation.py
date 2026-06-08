from __future__ import annotations

import pytest

from docgraph.config import Config
from docgraph.mcp.search import SearchService
from docgraph.models import DocumentRecord
from docgraph.store import ChromaStore, FtsStore, SQLiteStore


class FakeEmbedder:
    async def embed(self, texts, for_query=False):
        return [[0.1] * 768 for _ in texts]


def _seed_two_folders(cfg):
    sqlite = SQLiteStore(cfg)
    sqlite.init_schema()
    chroma = ChromaStore(cfg)
    fts = FtsStore(cfg)
    for folder in ("public", "private"):
        did = f"doc_{folder}"
        sqlite.insert_document(DocumentRecord(id=did, filename=f"{folder}.md", folder=folder))
        chroma.upsert_chunks([{
            "id": f"{did}_0", "embedding": [0.1] * 768,
            "text": "shared content phrase here",
            "metadata": {"doc_id": did, "filename": f"{folder}.md",
                         "folder": folder, "tags": "[]", "chunk_index": 0},
        }])
        fts.upsert_chunks([{
            "chunk_id": f"{did}_0", "doc_id": did, "folder": folder, "tags": "[]",
            "chunk_index": 0, "text": "shared content phrase here", "filename": f"{folder}.md",
        }])
    return sqlite, chroma, fts


@pytest.mark.asyncio
async def test_folder_filter_isolates_results(tmp_path):
    cfg = Config(data_dir=tmp_path)
    cfg.hybrid_enabled = True
    cfg.rerank_enabled = False
    cfg.ensure_dirs()
    sqlite, chroma, fts = _seed_two_folders(cfg)
    svc = SearchService(cfg, sqlite, chroma, FakeEmbedder(), fts=fts, reranker=None)
    results = await svc.search("shared content", folder="public")
    assert len(results) >= 1
    assert all(r.folder == "public" for r in results)
    assert not any(r.folder == "private" for r in results)


@pytest.mark.asyncio
async def test_folder_filter_applies_to_both_branches(tmp_path):
    """Critical: if folder filter is only applied to one branch, fusion
    would leak the other folder's chunks."""
    cfg = Config(data_dir=tmp_path)
    cfg.hybrid_enabled = True
    cfg.rerank_enabled = False
    cfg.ensure_dirs()
    sqlite, chroma, fts = _seed_two_folders(cfg)
    svc = SearchService(cfg, sqlite, chroma, FakeEmbedder(), fts=fts, reranker=None)
    for folder in ("public", "private"):
        results = await svc.search("shared", folder=folder)
        assert all(r.folder == folder for r in results), \
            f"Filter leaked across folders for folder={folder}: {[r.folder for r in results]}"
