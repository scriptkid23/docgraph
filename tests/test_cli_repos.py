from unittest.mock import AsyncMock, MagicMock

from docgraph.cli_repos import run_repos_command
from docgraph.config import Config
from docgraph.web.deps import AppState


def test_list_repos_empty(tmp_data_dir, capsys):
    cfg = Config(data_dir=tmp_data_dir)
    cfg.hybrid_enabled = False
    cfg.rerank_enabled = False
    state = AppState.create(cfg)
    exit_code = run_repos_command(
        ["list-repos"], cfg, in_process=True, state=state,
    )
    assert exit_code == 0
    out = capsys.readouterr().out
    assert "0 repos" in out


def test_import_repo_local_in_process(tmp_data_dir):
    cfg = Config(data_dir=tmp_data_dir)
    cfg.hybrid_enabled = False
    cfg.rerank_enabled = False
    state = AppState.create(cfg)
    local = tmp_data_dir / "src_repo"
    local.mkdir(parents=True, exist_ok=True)
    (local / ".git").mkdir()
    (local / "README.md").write_text("# Hi")
    state.codegraph.init_and_index = AsyncMock()
    fake_indexer = MagicMock()
    fake_indexer.index_markdown = AsyncMock()
    state._indexer = fake_indexer
    code = run_repos_command(
        ["import-repo", str(local)], cfg, in_process=True, state=state,
    )
    assert code == 0
    assert len(state.sqlite.list_repos()) == 1


def test_delete_repo_missing(tmp_data_dir, capsys):
    cfg = Config(data_dir=tmp_data_dir)
    cfg.hybrid_enabled = False
    cfg.rerank_enabled = False
    state = AppState.create(cfg)
    code = run_repos_command(
        ["delete-repo", "repo_unknown"], cfg, in_process=True, state=state,
    )
    assert code == 1
    err = capsys.readouterr().err
    assert "not found" in err
