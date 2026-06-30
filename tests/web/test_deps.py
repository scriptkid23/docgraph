from __future__ import annotations

from docgraph.config import Config
from docgraph.web.deps import AppState


def test_appstate_includes_fts(tmp_path):
    cfg = Config(data_dir=tmp_path)
    cfg.hybrid_enabled = True
    state = AppState.create(cfg)
    assert state.fts is not None
    assert state.fts.count_chunks() == 0


def test_appstate_fts_none_when_hybrid_disabled(tmp_path):
    cfg = Config(data_dir=tmp_path)
    cfg.hybrid_enabled = False
    state = AppState.create(cfg)
    assert state.fts is None


def test_appstate_reranker_none_when_disabled(tmp_path):
    cfg = Config(data_dir=tmp_path)
    cfg.rerank_enabled = False
    state = AppState.create(cfg)
    assert state.reranker is None


def test_appstate_reranker_created_when_enabled(tmp_path):
    cfg = Config(data_dir=tmp_path)
    cfg.rerank_enabled = True
    state = AppState.create(cfg)
    assert state.reranker is not None


def test_appstate_has_codegraph_and_repos(tmp_data_dir):
    from docgraph.repo.codegraph_client import CodegraphClient
    from docgraph.repo.manager import RepoManager
    cfg = Config(data_dir=tmp_data_dir)
    state = AppState.create(cfg)
    assert isinstance(state.codegraph, CodegraphClient)
    assert isinstance(state.repos(), RepoManager)
