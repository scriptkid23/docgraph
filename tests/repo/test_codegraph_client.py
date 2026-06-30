import asyncio
import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from docgraph.repo.codegraph_client import CodegraphClient, CodegraphNotInstalled


@pytest.fixture(autouse=True)
def _isolate_codegraph_lookup(monkeypatch):
    """Force the resolved binary to be the literal name (no .CMD shim on Windows)
    so we always take the create_subprocess_exec branch that tests patch."""
    monkeypatch.setattr("docgraph.repo.codegraph_client.shutil.which", lambda _: None)


def _fake_proc(stdout: bytes = b"", stderr: bytes = b"", returncode: int = 0):
    proc = MagicMock()
    proc.communicate = AsyncMock(return_value=(stdout, stderr))
    proc.returncode = returncode
    proc.terminate = MagicMock()
    proc.kill = MagicMock()
    proc.wait = AsyncMock(return_value=returncode)
    return proc


@pytest.mark.asyncio
async def test_health_check_returns_version():
    client = CodegraphClient(bin="codegraph")
    proc = _fake_proc(stdout=b"codegraph 0.5.1\n", returncode=0)
    with patch(
        "asyncio.create_subprocess_exec",
        AsyncMock(return_value=proc),
    ) as mocked:
        version = await client.health_check()
    assert "0.5.1" in version
    args, kwargs = mocked.call_args
    assert args[0] == "codegraph"
    assert "--version" in args


@pytest.mark.asyncio
async def test_health_check_missing_binary():
    client = CodegraphClient(bin="codegraph")
    with patch(
        "asyncio.create_subprocess_exec",
        AsyncMock(side_effect=FileNotFoundError("no such file")),
    ):
        with pytest.raises(CodegraphNotInstalled) as exc_info:
            await client.health_check()
    assert "install.sh" in str(exc_info.value)


@pytest.mark.asyncio
async def test_run_passes_json_flag_and_parses(tmp_path):
    client = CodegraphClient(bin="codegraph")
    payload = {"results": [{"name": "Validator"}]}
    proc = _fake_proc(stdout=json.dumps(payload).encode(), returncode=0)
    with patch(
        "asyncio.create_subprocess_exec",
        AsyncMock(return_value=proc),
    ) as mocked:
        result = await client.run("search", "Validator", repo_path=tmp_path)
    assert result == payload
    args, kwargs = mocked.call_args
    assert args == ("codegraph", "search", "Validator", "--json")
    assert kwargs["cwd"] == str(tmp_path)


@pytest.mark.asyncio
async def test_run_raises_on_nonzero(tmp_path):
    client = CodegraphClient(bin="codegraph")
    proc = _fake_proc(stdout=b"", stderr=b"boom", returncode=2)
    with patch(
        "asyncio.create_subprocess_exec",
        AsyncMock(return_value=proc),
    ):
        with pytest.raises(RuntimeError) as exc_info:
            await client.run("search", "X", repo_path=tmp_path)
    assert "boom" in str(exc_info.value)


@pytest.mark.asyncio
async def test_init_invokes_progress_cb(tmp_path):
    client = CodegraphClient(bin="codegraph", init_heartbeat_sec=0.05)
    proc = _fake_proc(stdout=b"", returncode=0)

    async def slow_communicate():
        await asyncio.sleep(0.2)
        return (b"", b"")

    proc.communicate = slow_communicate
    phases: list[str] = []
    with patch(
        "asyncio.create_subprocess_exec",
        AsyncMock(return_value=proc),
    ):
        await client.init(tmp_path, progress_cb=phases.append)
    assert len(phases) >= 1
    assert all("Building code index" in p for p in phases)
