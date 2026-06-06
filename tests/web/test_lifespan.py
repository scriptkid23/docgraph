from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from docgraph.config import Config
from docgraph.web.app import _lifespan


def _fake_app_with_state(state):
    app = MagicMock()
    app.state.docgraph = state
    return app


def _fake_state(cfg, *, fts_count, chroma_count, has_reranker=True):
    state = MagicMock()
    state.cfg = cfg
    fts = MagicMock()
    fts.count_chunks.return_value = fts_count
    fts.rebuild_from_chroma = AsyncMock(return_value=chroma_count)
    state.fts = fts
    chroma = MagicMock()
    chroma.count_chunks.return_value = chroma_count
    state.chroma = chroma
    state.sqlite = MagicMock()
    if has_reranker:
        rr = MagicMock()
        rr.prewarm = AsyncMock(return_value=None)
        state.reranker = rr
    else:
        state.reranker = None
    return state


@pytest.mark.asyncio
async def test_lifespan_prewarms_reranker_when_enabled(tmp_path):
    cfg = Config(data_dir=tmp_path)
    cfg.rerank_enabled = True
    cfg.rerank_prewarm = True
    cfg.hybrid_enabled = False
    state = _fake_state(cfg, fts_count=0, chroma_count=0)
    state.fts = None  # hybrid disabled

    async with _lifespan(_fake_app_with_state(state)):
        # Yield happens; let the create_task fire
        await asyncio.sleep(0.01)
    state.reranker.prewarm.assert_called_once()


@pytest.mark.asyncio
async def test_lifespan_skips_prewarm_when_prewarm_disabled(tmp_path):
    cfg = Config(data_dir=tmp_path)
    cfg.rerank_enabled = True
    cfg.rerank_prewarm = False
    cfg.hybrid_enabled = False
    state = _fake_state(cfg, fts_count=0, chroma_count=0)
    state.fts = None

    async with _lifespan(_fake_app_with_state(state)):
        await asyncio.sleep(0.01)
    state.reranker.prewarm.assert_not_called()


@pytest.mark.asyncio
async def test_lifespan_skips_prewarm_when_reranker_none(tmp_path):
    cfg = Config(data_dir=tmp_path)
    cfg.rerank_enabled = True
    cfg.rerank_prewarm = True
    cfg.hybrid_enabled = False
    state = _fake_state(cfg, fts_count=0, chroma_count=0, has_reranker=False)
    state.fts = None

    async with _lifespan(_fake_app_with_state(state)):
        await asyncio.sleep(0.01)
    # No reranker — no prewarm call to assert; just ensure no exception raised


@pytest.mark.asyncio
async def test_lifespan_auto_rebuilds_when_fts_empty_and_chroma_populated(tmp_path):
    cfg = Config(data_dir=tmp_path)
    cfg.hybrid_enabled = True
    cfg.rerank_enabled = False
    state = _fake_state(cfg, fts_count=0, chroma_count=42)

    async with _lifespan(_fake_app_with_state(state)):
        await asyncio.sleep(0.01)
    state.fts.rebuild_from_chroma.assert_called_once()


@pytest.mark.asyncio
async def test_lifespan_no_rebuild_when_both_empty(tmp_path):
    cfg = Config(data_dir=tmp_path)
    cfg.hybrid_enabled = True
    cfg.rerank_enabled = False
    state = _fake_state(cfg, fts_count=0, chroma_count=0)

    async with _lifespan(_fake_app_with_state(state)):
        await asyncio.sleep(0.01)
    state.fts.rebuild_from_chroma.assert_not_called()


@pytest.mark.asyncio
async def test_lifespan_no_rebuild_when_counts_match(tmp_path):
    cfg = Config(data_dir=tmp_path)
    cfg.hybrid_enabled = True
    cfg.rerank_enabled = False
    state = _fake_state(cfg, fts_count=100, chroma_count=100)

    async with _lifespan(_fake_app_with_state(state)):
        await asyncio.sleep(0.01)
    state.fts.rebuild_from_chroma.assert_not_called()


@pytest.mark.asyncio
async def test_lifespan_warns_on_drift_but_does_not_rebuild(tmp_path, caplog):
    cfg = Config(data_dir=tmp_path)
    cfg.hybrid_enabled = True
    cfg.rerank_enabled = False
    # 200 vs 50 — drift > max(10, 200*0.05=10) → 150 > 10 → warn
    state = _fake_state(cfg, fts_count=50, chroma_count=200)

    import logging
    with caplog.at_level(logging.WARNING, logger="docgraph.web.app"):
        async with _lifespan(_fake_app_with_state(state)):
            await asyncio.sleep(0.01)
    state.fts.rebuild_from_chroma.assert_not_called()
    assert "out of sync" in caplog.text.lower() or "rebuild-fts" in caplog.text


@pytest.mark.asyncio
async def test_lifespan_skips_fts_when_hybrid_disabled(tmp_path):
    cfg = Config(data_dir=tmp_path)
    cfg.hybrid_enabled = False
    cfg.rerank_enabled = False
    state = _fake_state(cfg, fts_count=0, chroma_count=10)
    state.fts = None  # hybrid disabled → state.fts is None

    async with _lifespan(_fake_app_with_state(state)):
        await asyncio.sleep(0.01)
    # No exception; chroma.count_chunks should not be called
    state.chroma.count_chunks.assert_not_called()
