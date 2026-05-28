import sys
from types import ModuleType
from unittest.mock import MagicMock

import pytest

from docgraph.config import Config
from docgraph.embed.local import LocalEmbedder


def _fake_native():
    mod = ModuleType("docgraph_embed")
    mod._inited = False

    def init(model, cache_dir=None):
        mod._inited = True
        mod._model = model
        mod._cache = cache_dir

    def health_check():
        if not mod._inited:
            raise RuntimeError("not initialized")

    def embedding_dimension():
        return 768

    def embed(texts):
        return [[0.1, 0.2, 0.3] for _ in texts]

    mod.init = init
    mod.health_check = health_check
    mod.embedding_dimension = embedding_dimension
    mod.embed = embed
    return mod


@pytest.mark.asyncio
async def test_local_embedder_batches(tmp_data_dir, monkeypatch):
    fake = _fake_native()
    monkeypatch.setitem(sys.modules, "docgraph_embed", fake)

    cfg = Config(data_dir=tmp_data_dir)
    embedder = LocalEmbedder(cfg.local_model, cfg.local_model_dir)
    vecs = await embedder.embed(["a", "b", "c"])
    assert len(vecs) == 3
    assert len(vecs[0]) == 3
    assert fake._inited is True


@pytest.mark.asyncio
async def test_local_health_check(tmp_data_dir, monkeypatch):
    monkeypatch.setitem(sys.modules, "docgraph_embed", _fake_native())
    cfg = Config(data_dir=tmp_data_dir)
    embedder = LocalEmbedder(cfg.local_model, cfg.local_model_dir)
    await embedder.health_check()
