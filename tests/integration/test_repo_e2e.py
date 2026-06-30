import shutil
from pathlib import Path

import pytest

from docgraph.config import Config
from docgraph.web.deps import AppState


REQUIRES_CG = pytest.mark.skipif(
    shutil.which("codegraph") is None, reason="codegraph CLI not on PATH"
)
REQUIRES_GIT = pytest.mark.skipif(
    shutil.which("git") is None, reason="git not on PATH"
)


@pytest.mark.integration
@REQUIRES_CG
@REQUIRES_GIT
@pytest.mark.asyncio
async def test_import_real_repo_and_query(tmp_data_dir):
    cfg = Config(data_dir=tmp_data_dir)
    cfg.hybrid_enabled = False
    cfg.rerank_enabled = False
    state = AppState.create(cfg)
    repo_id = await state.repos().import_repo(
        "https://github.com/octocat/Hello-World"
    )
    repo = state.sqlite.get_repo(repo_id)
    assert repo.status.value == "ready"

    files = await state.codegraph.run("files", repo_path=Path(repo.local_path))
    assert files is not None
