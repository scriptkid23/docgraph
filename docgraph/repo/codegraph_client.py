from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Callable

logger = logging.getLogger(__name__)


class CodegraphNotInstalled(RuntimeError):
    """Raised when the codegraph CLI is missing or unusable."""


class CodegraphClient:
    INSTALL_HINT = (
        "codegraph CLI not found. Install with: "
        "curl -fsSL https://raw.githubusercontent.com/colbymchenry/codegraph/main/install.sh | sh"
    )

    def __init__(
        self,
        *,
        bin: str = "codegraph",
        query_timeout_sec: float = 30.0,
        init_timeout_sec: float = 600.0,
        init_heartbeat_sec: float = 5.0,
    ) -> None:
        self._bin = bin
        self._query_timeout = query_timeout_sec
        self._init_timeout = init_timeout_sec
        self._heartbeat = init_heartbeat_sec

    async def _spawn(self, *argv: str, cwd: str | None = None):
        # Resolve via PATH ourselves so we can detect Windows .cmd/.bat shims
        # (npm wraps `codegraph` as a .CMD file that create_subprocess_exec
        # cannot invoke directly — it requires shell semantics).
        resolved = shutil.which(self._bin) or self._bin
        try:
            if sys.platform == "win32" and resolved.lower().endswith((".cmd", ".bat")):
                cmdline = subprocess.list2cmdline([resolved, *argv])
                return await asyncio.create_subprocess_shell(
                    cmdline,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                    cwd=cwd,
                )
            return await asyncio.create_subprocess_exec(
                resolved, *argv,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=cwd,
            )
        except FileNotFoundError as exc:
            raise CodegraphNotInstalled(self.INSTALL_HINT) from exc

    async def health_check(self) -> str:
        proc = await self._spawn("--version")
        stdout, stderr = await proc.communicate()
        if proc.returncode != 0:
            raise CodegraphNotInstalled(
                f"{self.INSTALL_HINT} "
                f"(exit {proc.returncode}: {stderr.decode(errors='replace').strip()})"
            )
        return stdout.decode(errors="replace").strip()

    async def run(
        self,
        subcommand: str,
        *args: str,
        repo_path: Path,
        timeout: float | None = None,
    ) -> Any:
        argv = (subcommand, *args, "--json")
        proc = await self._spawn(*argv, cwd=str(repo_path))
        try:
            stdout, stderr = await asyncio.wait_for(
                proc.communicate(), timeout=timeout or self._query_timeout
            )
        except asyncio.TimeoutError as exc:
            proc.terminate()
            with contextlib.suppress(asyncio.TimeoutError):
                await asyncio.wait_for(proc.wait(), timeout=5)
            if proc.returncode is None:
                proc.kill()
                await proc.wait()
            raise TimeoutError(
                f"codegraph {subcommand} timed out after "
                f"{timeout or self._query_timeout}s"
            ) from exc
        if proc.returncode != 0:
            raise RuntimeError(
                f"codegraph {subcommand} failed (exit {proc.returncode}): "
                f"{stderr.decode(errors='replace').strip()}"
            )
        text = stdout.decode(errors="replace").strip()
        if not text:
            return None
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            return text

    async def init(
        self,
        repo_path: Path,
        *,
        progress_cb: Callable[[str], None] | None = None,
    ) -> None:
        proc = await self._spawn("init", cwd=str(repo_path))
        comm_task = asyncio.create_task(proc.communicate())
        elapsed = 0.0
        while not comm_task.done():
            try:
                await asyncio.wait_for(
                    asyncio.shield(comm_task), timeout=self._heartbeat
                )
            except asyncio.TimeoutError:
                elapsed += self._heartbeat
                if progress_cb:
                    progress_cb(
                        f"Building code index ({int(elapsed)}s elapsed)"
                    )
                if elapsed >= self._init_timeout:
                    proc.terminate()
                    with contextlib.suppress(asyncio.TimeoutError):
                        await asyncio.wait_for(comm_task, timeout=5)
                    if proc.returncode is None:
                        proc.kill()
                        await proc.wait()
                    raise TimeoutError(
                        f"codegraph init timed out after {self._init_timeout}s"
                    )
        stdout, stderr = comm_task.result()
        if proc.returncode != 0:
            raise RuntimeError(
                f"codegraph init failed (exit {proc.returncode}): "
                f"{stderr.decode(errors='replace').strip()}"
            )
        logger.debug("codegraph init done in %s", repo_path)

    async def aclose(self) -> None:
        return None
