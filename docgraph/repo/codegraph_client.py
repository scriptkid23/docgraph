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
        """Run an arbitrary codegraph subcommand and parse its stdout as JSON.

        Caller is responsible for passing the right JSON flag (`-j`,
        `--json`, or `--format json`) — different subcommands use different
        flag spellings and this wrapper does not second-guess them.
        """
        argv = (subcommand, *args)
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

    async def _run_long(
        self,
        subcommand: str,
        repo_path: Path,
        *extra_args: str,
        progress_cb: Callable[[str], None] | None = None,
        phase_label: str = "Building code index",
    ) -> None:
        """Run a long-lived codegraph subcommand (init / index) with heartbeat."""
        proc = await self._spawn(subcommand, *extra_args, cwd=str(repo_path))
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
                    progress_cb(f"{phase_label} ({int(elapsed)}s elapsed)")
                if elapsed >= self._init_timeout:
                    proc.terminate()
                    with contextlib.suppress(asyncio.TimeoutError):
                        await asyncio.wait_for(comm_task, timeout=5)
                    if proc.returncode is None:
                        proc.kill()
                        await proc.wait()
                    raise TimeoutError(
                        f"codegraph {subcommand} timed out after {self._init_timeout}s"
                    )
        stdout, stderr = comm_task.result()
        if proc.returncode != 0:
            raise RuntimeError(
                f"codegraph {subcommand} failed (exit {proc.returncode}): "
                f"{stderr.decode(errors='replace').strip()}"
            )
        logger.debug(
            "codegraph %s done in %s (stdout %dB)",
            subcommand, repo_path, len(stdout),
        )

    async def init(
        self,
        repo_path: Path,
        *,
        progress_cb: Callable[[str], None] | None = None,
    ) -> None:
        """Initialize CodeGraph in `repo_path` (creates `.codegraph/` skeleton).

        `init` alone does NOT populate the index — it just provisions the SQLite
        DB. Use `index()` afterwards (or call `init_and_index()`) so that the
        knowledge graph actually contains symbols and edges.
        """
        await self._run_long(
            "init", repo_path,
            progress_cb=progress_cb,
            phase_label="Initializing code index",
        )

    async def index(
        self,
        repo_path: Path,
        *,
        force: bool = False,
        progress_cb: Callable[[str], None] | None = None,
    ) -> None:
        """Index all files under `repo_path`. Use `force=True` to rebuild."""
        extra = ("--force",) if force else ()
        await self._run_long(
            "index", repo_path, *extra,
            progress_cb=progress_cb,
            phase_label="Building code index",
        )

    async def init_and_index(
        self,
        repo_path: Path,
        *,
        progress_cb: Callable[[str], None] | None = None,
    ) -> None:
        """Idempotent: provision `.codegraph/` then populate the index."""
        await self.init(repo_path, progress_cb=progress_cb)
        await self.index(repo_path, progress_cb=progress_cb)

    async def aclose(self) -> None:
        return None
