import pytest

from docgraph.config import Config
from docgraph.mcp.server import create_mcp_server
from docgraph.web.deps import AppState


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
