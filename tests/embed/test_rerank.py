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


@pytest.mark.rerank_model
class TestRerankerWithRealModel:
    """Downloads ~600MB BGE-reranker-v2-m3 ONNX model on first run.

    Opt-in: run with `poetry run pytest -m rerank_model`. Requires the Rust
    crate to be built (`cd crates/docgraph-embed && env -u CONDA_PREFIX \
    -u CONDA_DEFAULT_ENV -u CONDA_SHLVL -u CONDA_PROMPT_MODIFIER \
    maturin develop --release`).
    """

    @pytest.fixture(scope="class")
    def event_loop(self):
        import asyncio
        loop = asyncio.new_event_loop()
        yield loop
        loop.close()

    @pytest.fixture(scope="class")
    async def reranker(self, tmp_path_factory):
        from docgraph.embed.rerank import Reranker
        cache = tmp_path_factory.mktemp("models")
        r = Reranker(model="bge-reranker-v2-m3", cache_dir=cache)
        await r.prewarm()
        return r

    @pytest.mark.asyncio
    async def test_relevance_ordering(self, reranker):
        scores = await reranker.rerank(
            "What does DocGraph use for vector storage?",
            [
                "DocGraph stores vectors in ChromaDB with cosine similarity.",
                "The web UI is built with React and Vite.",
                "MarkItDown converts files to Markdown.",
            ],
        )
        assert len(scores) == 3
        assert scores[0] > scores[1]
        assert scores[0] > scores[2]

    @pytest.mark.asyncio
    async def test_multilingual_vn_en(self, reranker):
        scores = await reranker.rerank(
            "DocGraph dùng database nào để lưu vector?",
            [
                "DocGraph stores vectors in ChromaDB.",
                "User interface built with React.",
            ],
        )
        assert scores[0] > scores[1]

    @pytest.mark.asyncio
    async def test_concurrent_rerank_no_deadlock(self, reranker):
        import asyncio
        results = await asyncio.gather(*[
            reranker.rerank(f"query {i}", ["passage A", "passage B"]) for i in range(5)
        ])
        assert all(len(r) == 2 for r in results)

    @pytest.mark.asyncio
    async def test_empty_and_single_passage(self, reranker):
        assert await reranker.rerank("q", []) == []
        scores = await reranker.rerank("q", ["only one"])
        assert len(scores) == 1
