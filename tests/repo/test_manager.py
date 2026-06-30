from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from docgraph.config import Config
from docgraph.models import DocumentStatus, RepoRecord
from docgraph.repo.manager import RepoManager, _repo_slug
from docgraph.store import ChromaStore, FileStore, SQLiteStore


def test_repo_slug_from_https_url():
    assert _repo_slug("https://github.com/ethereum/go-ethereum") == "ethereum_go-ethereum"
    assert _repo_slug("https://github.com/ethereum/go-ethereum.git") == "ethereum_go-ethereum"


def test_repo_slug_from_git_ssh():
    assert _repo_slug("git@github.com:ethereum/go-ethereum.git") == "ethereum_go-ethereum"


def test_repo_slug_from_local_path(tmp_path):
    d = tmp_path / "myproj"
    d.mkdir()
    assert _repo_slug(str(d)) == "myproj"


def _populate_repo(repo_dir: Path) -> None:
    repo_dir.mkdir(parents=True, exist_ok=True)
    (repo_dir / ".git").mkdir()
    (repo_dir / "README.md").write_text("# Hello\n\nIntro to the project.")
    (repo_dir / "docs").mkdir()
    (repo_dir / "docs" / "design.md").write_text("# Design\n\nArchitecture notes.")
    (repo_dir / "node_modules").mkdir()
    (repo_dir / "node_modules" / "ignored.md").write_text("should be skipped")
    (repo_dir / "main.go").write_text("package main\n")


def _make_manager(cfg, sqlite, files, chroma):
    codegraph = MagicMock()
    codegraph.init = AsyncMock()
    indexer = MagicMock()
    indexer.index_markdown = AsyncMock()
    mgr = RepoManager(
        cfg=cfg, sqlite=sqlite, files=files, chroma=chroma,
        codegraph=codegraph, indexer_factory=lambda: indexer,
    )
    return mgr, codegraph, indexer


@pytest.mark.asyncio
async def test_import_repo_local_path(tmp_data_dir):
    cfg = Config(data_dir=tmp_data_dir)
    cfg.ensure_dirs()
    sqlite = SQLiteStore(cfg)
    sqlite.init_schema()
    chroma = ChromaStore(cfg)
    files = FileStore(cfg)

    local_repo = tmp_data_dir / "src_repo"
    _populate_repo(local_repo)
    mgr, codegraph, indexer = _make_manager(cfg, sqlite, files, chroma)

    repo_id = await mgr.import_repo(
        str(local_repo), folder="chains", tags=("evm",)
    )
    repo = sqlite.get_repo(repo_id)
    assert repo.status == DocumentStatus.READY
    assert repo.doc_count == 2  # README + docs/design.md ; node_modules skipped
    codegraph.init.assert_awaited_once()
    assert indexer.index_markdown.await_count == 2


@pytest.mark.asyncio
async def test_import_repo_rejects_duplicate_url(tmp_data_dir):
    cfg = Config(data_dir=tmp_data_dir)
    cfg.ensure_dirs()
    sqlite = SQLiteStore(cfg)
    sqlite.init_schema()
    chroma = ChromaStore(cfg)
    files = FileStore(cfg)
    mgr, _, _ = _make_manager(cfg, sqlite, files, chroma)

    local_repo = tmp_data_dir / "src_repo"
    _populate_repo(local_repo)
    await mgr.import_repo(str(local_repo))
    with pytest.raises(ValueError) as exc_info:
        await mgr.import_repo(str(local_repo))
    assert "already imported" in str(exc_info.value)


@pytest.mark.asyncio
async def test_delete_repo_cascades_docs(tmp_data_dir):
    cfg = Config(data_dir=tmp_data_dir)
    cfg.ensure_dirs()
    sqlite = SQLiteStore(cfg)
    sqlite.init_schema()
    chroma = ChromaStore(cfg)
    files = FileStore(cfg)
    mgr, _, _ = _make_manager(cfg, sqlite, files, chroma)

    local_repo = tmp_data_dir / "src_repo"
    _populate_repo(local_repo)
    repo_id = await mgr.import_repo(str(local_repo))
    assert len(sqlite.list_documents_by_repo(repo_id)) == 2

    cascaded = await mgr.delete_repo(repo_id)
    assert cascaded == 2
    assert sqlite.get_repo(repo_id) is None
    assert sqlite.list_documents_by_repo(repo_id) == []


def test_resolve_by_id_name_and_single(tmp_data_dir):
    cfg = Config(data_dir=tmp_data_dir)
    cfg.ensure_dirs()
    sqlite = SQLiteStore(cfg)
    sqlite.init_schema()
    chroma = ChromaStore(cfg)
    files = FileStore(cfg)
    mgr, _, _ = _make_manager(cfg, sqlite, files, chroma)

    sqlite.insert_repo(RepoRecord(
        id="repo_x", name="go-ethereum",
        local_path=str(tmp_data_dir / "x"),
        status=DocumentStatus.READY,
    ))
    assert mgr.resolve("repo_x").id == "repo_x"
    assert mgr.resolve("GO-ETHEREUM").id == "repo_x"
    assert mgr.resolve(None).id == "repo_x"


def test_watcher_ignores_codegraph_dir():
    from docgraph.watch.ignore import HARDCODED_IGNORE_DIRS
    assert ".codegraph" in HARDCODED_IGNORE_DIRS
