import pytest

from boostmcp.config import Config
from boostmcp.mcp.server import create_mcp_server
from boostmcp.web.deps import AppState


@pytest.mark.asyncio
async def test_mcp_server_has_tools(tmp_data_dir):
    cfg = Config(data_dir=tmp_data_dir)
    state = AppState.create(cfg)
    mcp = create_mcp_server(state)
    tools = await mcp.list_tools()
    names = {t.name for t in tools}
    assert "search_documents" in names
    assert "list_documents" in names
    assert "get_document" in names
