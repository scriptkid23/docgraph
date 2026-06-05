from __future__ import annotations

import asyncio
import sys
from unittest.mock import MagicMock

import pytest

from docgraph.embed.rerank import Reranker


class TestRerankerWrapper:
    def _install_mock_module(self, monkeypatch, init_fn=None, rerank_fn=None, health_fn=None):
        mod = MagicMock()
        mod.rerank_init = init_fn or MagicMock(return_value=None)
        mod.rerank = rerank_fn or MagicMock(return_value=[0.9, 0.5, 0.1])
        mod.rerank_health_check = health_fn or MagicMock(return_value=None)
        monkeypatch.setitem(sys.modules, "docgraph_embed", mod)
        return mod

    @pytest.mark.asyncio
    async def test_rerank_returns_scores_in_order(self, tmp_path, monkeypatch):
        mod = self._install_mock_module(monkeypatch)
        r = Reranker(model="bge-reranker-v2-m3", cache_dir=tmp_path)
        scores = await r.rerank("query", ["passage a", "passage b", "passage c"])
        assert scores == [0.9, 0.5, 0.1]
        mod.rerank_init.assert_called_once()
        mod.rerank.assert_called_once_with("query", ["passage a", "passage b", "passage c"])

    @pytest.mark.asyncio
    async def test_rerank_empty_passages_returns_empty_without_init(self, tmp_path, monkeypatch):
        mod = self._install_mock_module(monkeypatch)
        r = Reranker(model="bge-reranker-v2-m3", cache_dir=tmp_path)
        scores = await r.rerank("query", [])
        assert scores == []
        # No native calls because we short-circuited
        mod.rerank_init.assert_not_called()
        mod.rerank.assert_not_called()

    @pytest.mark.asyncio
    async def test_concurrent_init_locks_once(self, tmp_path, monkeypatch):
        mod = self._install_mock_module(monkeypatch)
        r = Reranker(model="bge-reranker-v2-m3", cache_dir=tmp_path)
        await asyncio.gather(*[r.rerank("q", ["p"]) for _ in range(5)])
        # Init called exactly once across concurrent callers
        assert mod.rerank_init.call_count == 1

    @pytest.mark.asyncio
    async def test_prewarm_does_not_raise_on_failure(self, tmp_path, monkeypatch):
        # rerank raises but prewarm must NOT propagate
        mod = self._install_mock_module(
            monkeypatch,
            rerank_fn=MagicMock(side_effect=RuntimeError("model not ready")),
        )
        r = Reranker(model="bge-reranker-v2-m3", cache_dir=tmp_path)
        await r.prewarm()  # must not raise

    @pytest.mark.asyncio
    async def test_import_error_raises_helpful_message(self, tmp_path, monkeypatch):
        # Simulate missing native module
        monkeypatch.setitem(sys.modules, "docgraph_embed", None)
        # Force ImportError on import attempt
        original_import = __import__

        def fake_import(name, *args, **kwargs):
            if name == "docgraph_embed":
                raise ImportError("No module named 'docgraph_embed'")
            return original_import(name, *args, **kwargs)

        monkeypatch.setattr("builtins.__import__", fake_import)
        r = Reranker(model="bge-reranker-v2-m3", cache_dir=tmp_path)
        with pytest.raises(RuntimeError, match="maturin develop"):
            await r.rerank("q", ["p"])
