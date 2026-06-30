import json
from unittest.mock import AsyncMock

import pytest

from docgraph.config import Config
from docgraph.mcp.server import create_mcp_server
from docgraph.models import DocumentStatus, RepoRecord
from docgraph.web.deps import AppState


def _make_state(tmp_data_dir):
    cfg = Config(data_dir=tmp_data_dir)
    cfg.hybrid_enabled = False
    cfg.rerank_enabled = False
    state = AppState.create(cfg)
    state.codegraph.health_check = AsyncMock(return_value="codegraph 0.5.1-test")
    return state


async def _call_tool(mcp, name: str, arguments: dict) -> str:
    result = await mcp.call_tool(name, arguments)
    # FastMCP.call_tool returns (content_list, structured) under mcp>=1.x.
    content = result[0] if isinstance(result, tuple) else result.content
    if hasattr(content[0], "text"):
        return content[0].text
    return str(content)


@pytest.mark.asyncio
async def test_mcp_has_code_tools(tmp_data_dir):
    state = _make_state(tmp_data_dir)
    mcp = create_mcp_server(state)
    tools = await mcp.list_tools()
    names = {t.name for t in tools}
    expected = {
        "list_repos", "import_repo",
        "code_search", "code_impact", "code_callers", "code_callees",
        "code_context", "code_files",
    }
    assert expected.issubset(names)


@pytest.mark.asyncio
async def test_code_search_uses_resolved_repo(tmp_data_dir):
    state = _make_state(tmp_data_dir)
    state.sqlite.insert_repo(RepoRecord(
        id="repo_x", name="go-ethereum",
        local_path=str(tmp_data_dir / "x"),
        status=DocumentStatus.READY,
    ))
    state.codegraph.run = AsyncMock(return_value={"results": [{"name": "Validator"}]})
    mcp = create_mcp_server(state)
    payload = await _call_tool(
        mcp, "code_search", {"query": "Validator", "repo": "go-ethereum"}
    )
    body = json.loads(payload)
    assert body["result"]["results"][0]["name"] == "Validator"
    args, kwargs = state.codegraph.run.call_args
    assert args[0] == "query"
    assert "Validator" in args
    assert "-j" in args
    assert str(kwargs["repo_path"]).endswith("x")


@pytest.mark.asyncio
async def test_code_search_returns_error_when_no_repo(tmp_data_dir):
    state = _make_state(tmp_data_dir)
    mcp = create_mcp_server(state)
    payload = await _call_tool(mcp, "code_search", {"query": "X"})
    body = json.loads(payload)
    assert "error" in body
    assert "specify repo" in body["error"]


@pytest.mark.asyncio
async def test_code_search_repo_not_ready(tmp_data_dir):
    state = _make_state(tmp_data_dir)
    state.sqlite.insert_repo(RepoRecord(
        id="repo_x", name="go-ethereum",
        local_path=str(tmp_data_dir / "x"),
        status=DocumentStatus.PROCESSING,
        progress_pct=42,
    ))
    mcp = create_mcp_server(state)
    payload = await _call_tool(
        mcp, "code_search", {"query": "Validator", "repo": "go-ethereum"}
    )
    body = json.loads(payload)
    assert body["status"] == "processing"
    assert body["progress_pct"] == 42
