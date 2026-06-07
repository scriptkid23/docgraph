from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from docgraph.config import Config
from docgraph.models import SourceType, WatchedDirRecord
from docgraph.web.deps import AppState


@pytest.fixture
def state(tmp_path: Path) -> AppState:
    cfg = Config(data_dir=tmp_path / "data")
    cfg.ensure_dirs()
    return AppState.create(cfg)


@pytest.fixture
def wd(tmp_path: Path) -> WatchedDirRecord:
    src = tmp_path / "src"
    src.mkdir()
    return WatchedDirRecord(id="wd_t", path=str(src), created_at="2026-06-07T00:00:00Z")


@pytest.mark.asyncio
async def test_text_file_referenced_in_place(state: AppState, wd: WatchedDirRecord):
    p = Path(wd.path) / "note.md"
    p.write_text("# Note\n\nbody.")
    mtime = p.stat().st_mtime_ns
    indexer = state.indexer()
    indexer.index_text_direct = AsyncMock()  # type: ignore[method-assign]
    await indexer.index_watched_new("d_text", p, wd, False, mtime)
    doc = state.sqlite.get_document("d_text")
    assert doc.source_type == SourceType.WATCHED
    assert doc.watched_path == str(p)
    assert doc.materialize is False
    assert doc.original_path is None
    assert doc.mtime_ns == mtime
    indexer.index_text_direct.assert_awaited_once()


@pytest.mark.asyncio
async def test_binary_file_snapshot_copied(state: AppState, wd: WatchedDirRecord):
    p = Path(wd.path) / "doc.pdf"
    p.write_bytes(b"%PDF-1.4\nfake")
    mtime = p.stat().st_mtime_ns
    indexer = state.indexer()
    indexer.index_document = AsyncMock()  # type: ignore[method-assign]
    await indexer.index_watched_new("d_bin", p, wd, True, mtime)
    doc = state.sqlite.get_document("d_bin")
    assert doc.source_type == SourceType.WATCHED
    assert doc.materialize is True
    assert doc.original_path is not None
    assert Path(doc.original_path).exists()
    assert Path(doc.original_path).read_bytes() == b"%PDF-1.4\nfake"
    indexer.index_document.assert_awaited_once()


@pytest.mark.asyncio
async def test_reindex_watched_updates_mtime(state: AppState, wd: WatchedDirRecord):
    p = Path(wd.path) / "a.md"
    p.write_text("v1")
    mtime1 = p.stat().st_mtime_ns
    indexer = state.indexer()
    indexer.index_text_direct = AsyncMock()  # type: ignore[method-assign]
    await indexer.index_watched_new("d_r", p, wd, False, mtime1)
    p.write_text("v2 with more body")
    import os, time
    time.sleep(0.01)
    os.utime(p, None)
    mtime2 = p.stat().st_mtime_ns
    assert state.sqlite.claim_for_reindex("d_r") is True
    await indexer.reindex_watched("d_r", p, False, mtime2)
    doc = state.sqlite.get_document("d_r")
    assert doc.mtime_ns == mtime2
