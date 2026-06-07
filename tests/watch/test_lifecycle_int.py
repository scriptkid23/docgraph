import asyncio
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from docgraph.config import Config
from docgraph.models import WatchedDirRecord
from docgraph.watch.manager import WatcherManager
from docgraph.web.deps import AppState


def _stub_index_new(state, doc_id, p, wd, materialize, mtime_ns):
    from docgraph.models import DocumentRecord, DocumentStatus, SourceType
    doc = DocumentRecord(
        id=doc_id, filename=Path(p).name, folder=wd.folder, tags=list(wd.tags),
        source_type=SourceType.WATCHED, status=DocumentStatus.READY,
        watched_path=str(p), materialize=materialize, mtime_ns=mtime_ns,
    )
    state.sqlite.insert_document(doc)


def _stub_reindex(state, doc_id, mtime_ns):
    from docgraph.models import DocumentStatus
    state.sqlite.update_mtime_ns(doc_id, mtime_ns)
    state.sqlite.update_status(doc_id, DocumentStatus.READY)


@pytest.fixture
def setup(tmp_path: Path):
    cfg = Config(data_dir=tmp_path / "data")
    cfg.watch_debounce_sec = 0.1
    cfg.watch_workers = 2
    cfg.ensure_dirs()
    state = AppState.create(cfg)
    indexer = state.indexer()
    indexer.index_watched_new = AsyncMock(side_effect=lambda doc_id, p, wd, mat, mt: _stub_index_new(state, doc_id, p, wd, mat, mt))
    indexer.reindex_watched = AsyncMock(side_effect=lambda doc_id, p, mat, mt: _stub_reindex(state, doc_id, mt))
    watched = tmp_path / "src"
    watched.mkdir()
    state.sqlite.insert_watched_dir(WatchedDirRecord(
        id="wd_t", path=str(watched), created_at="2026-06-07T00:00:00Z",
    ))
    return state, watched


@pytest.mark.asyncio
async def test_enable_persists_across_restart(setup):
    state, watched = setup
    mgr = WatcherManager(state)
    await mgr.enable()
    assert state.sqlite.get_watcher_state("enabled") == "true"
    await mgr.disable()
    assert state.sqlite.get_watcher_state("enabled") == "false"


@pytest.mark.asyncio
async def test_stats_reset_on_enable(setup):
    state, watched = setup
    mgr = WatcherManager(state)
    await mgr.enable()
    mgr.stats.events_received = 100
    await mgr.disable()
    await mgr.enable()
    assert mgr.stats.events_received == 0
    await mgr.disable()
