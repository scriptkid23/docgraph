from __future__ import annotations

import asyncio
import logging
import shutil
import uuid
from collections import defaultdict
from pathlib import Path
from typing import Callable, Iterable
from urllib.parse import urlparse

from docgraph.config import Config
from docgraph.ingest.urls import validate_url
from docgraph.models import DocumentRecord, DocumentStatus, RepoRecord
from docgraph.repo.codegraph_client import CodegraphClient
from docgraph.store.chroma import ChromaStore
from docgraph.store.files import FileStore
from docgraph.store.sqlite import SQLiteStore

logger = logging.getLogger(__name__)

MD_SKIP_DIRS = frozenset({
    ".git", "node_modules", "vendor", "dist", "build", "target",
    "__pycache__", ".venv", ".next", ".codegraph",
})


def _repo_slug(source: str) -> str:
    """Derive a filesystem-safe slug from a URL or local path."""
    if source.startswith(("http://", "https://", "git@")):
        if source.startswith("git@"):
            _, _, tail = source.partition(":")
            parts = tail.strip("/").split("/")
        else:
            parts = urlparse(source).path.strip("/").split("/")
        if len(parts) >= 2:
            owner, name = parts[-2], parts[-1]
        else:
            owner, name = "remote", parts[-1] if parts else "repo"
        if name.endswith(".git"):
            name = name[:-4]
        return f"{owner}_{name}"
    return Path(source).resolve().name


def _iter_markdown_files(root: Path) -> Iterable[Path]:
    for path in root.rglob("*.md"):
        if any(part in MD_SKIP_DIRS for part in path.relative_to(root).parts):
            continue
        yield path


class RepoManager:
    def __init__(
        self,
        *,
        cfg: Config,
        sqlite: SQLiteStore,
        files: FileStore,
        chroma: ChromaStore,
        codegraph: CodegraphClient,
        indexer_factory: Callable[[], object],
    ) -> None:
        self._cfg = cfg
        self._sqlite = sqlite
        self._files = files
        self._chroma = chroma
        self._codegraph = codegraph
        self._indexer_factory = indexer_factory
        self._locks: dict[str, asyncio.Lock] = defaultdict(asyncio.Lock)

    def list_repos(self) -> list[RepoRecord]:
        return self._sqlite.list_repos()

    def get_repo(self, repo_id: str) -> RepoRecord | None:
        return self._sqlite.get_repo(repo_id)

    def resolve(self, ref: str | None) -> RepoRecord | None:
        if ref:
            if r := self._sqlite.get_repo(ref):
                return r
            if r := self._sqlite.get_repo_by_name(ref):
                return r
            return None
        ready = [
            r for r in self._sqlite.list_repos()
            if r.status == DocumentStatus.READY
        ]
        return ready[0] if len(ready) == 1 else None

    def _is_url(self, source: str) -> bool:
        return source.startswith(("http://", "https://", "git@"))

    async def _clone(self, url: str, target: Path) -> None:
        proc = await asyncio.create_subprocess_exec(
            "git", "clone", "--depth", "1", url, str(target),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        _, stderr = await proc.communicate()
        if proc.returncode != 0:
            raise RuntimeError(
                f"git clone failed (exit {proc.returncode}): "
                f"{stderr.decode(errors='replace').strip()}"
            )

    async def import_repo(
        self,
        source: str,
        *,
        folder: str = "",
        tags: tuple[str, ...] = (),
        existing_repo_id: str | None = None,
    ) -> str:
        is_url = self._is_url(source)
        if is_url:
            validate_url(source)
            if existing_repo_id is None and self._sqlite.get_repo_by_source(source):
                raise ValueError(
                    f"repo already imported: {source}; use reindex to refresh"
                )
            slug = _repo_slug(source)
            target = self._cfg.repos_dir / slug
            source_url = source
        else:
            local = Path(source).resolve()
            if not local.is_dir():
                raise ValueError(
                    f"local path not found or not a directory: {source}"
                )
            if existing_repo_id is None and self._sqlite.get_repo_by_source(str(local)):
                raise ValueError(
                    f"repo already imported: {source}; use reindex to refresh"
                )
            slug = _repo_slug(str(local))
            target = local
            source_url = str(local)

        if existing_repo_id is None:
            repo_id = f"repo_{uuid.uuid4().hex[:12]}"
            name = slug.split("_", 1)[-1] if "_" in slug else slug
            self._sqlite.insert_repo(RepoRecord(
                id=repo_id, name=name,
                source_url=source_url, local_path=str(target),
                folder=folder, tags=list(tags),
            ))
            self._sqlite.update_repo_progress(repo_id, 0, "Queued (0%)")
        else:
            repo_id = existing_repo_id

        try:
            async with self._locks[repo_id]:
                await self._run_import(repo_id, target, is_url, source_url)
        except Exception as exc:
            self._sqlite.update_repo_status(
                repo_id, DocumentStatus.ERROR, error_message=str(exc)
            )
            logger.exception("repo import failed for repo_id=%s", repo_id)
            raise
        return repo_id

    async def _run_import(
        self, repo_id: str, target: Path, is_url: bool, source_url: str
    ) -> None:
        if is_url and not target.exists():
            self._sqlite.update_repo_progress(
                repo_id, 5, f"Cloning {source_url} (5%)"
            )
            await self._clone(source_url, target)
        self._sqlite.update_repo_progress(
            repo_id, 30, "Building code index (30%)"
        )

        def hb(phase: str) -> None:
            self._sqlite.update_repo_progress(repo_id, 50, phase)

        await self._codegraph.init(target, progress_cb=hb)

        md_files = list(_iter_markdown_files(target))
        total = len(md_files) or 1
        self._sqlite.update_repo_progress(
            repo_id, 80, f"Indexing docs (0/{total})"
        )
        indexer = self._indexer_factory()
        repo = self._sqlite.get_repo(repo_id)
        for idx, md in enumerate(md_files, start=1):
            content = md.read_text(encoding="utf-8", errors="replace")
            doc_id = f"doc_{uuid.uuid4().hex[:12]}"
            self._sqlite.insert_document(DocumentRecord(
                id=doc_id,
                filename=str(md.relative_to(target)),
                folder=repo.folder,
                tags=list(repo.tags),
                original_path=str(md),
                repo_id=repo_id,
            ))
            self._sqlite.update_progress(doc_id, 0, "Queued for indexing")
            try:
                await indexer.index_markdown(doc_id, content)
            except Exception as exc:
                logger.warning(
                    "md indexing failed in repo %s: %s — %s", repo_id, md, exc
                )
            pct = 80 + int(15 * idx / total)
            self._sqlite.update_repo_progress(
                repo_id, pct, f"Indexing docs ({idx}/{total})"
            )

        doc_count = len(self._sqlite.list_documents_by_repo(repo_id))
        self._sqlite.update_repo_progress(repo_id, 95, "Finalizing (95%)")
        self._sqlite.update_repo_status(
            repo_id, DocumentStatus.READY, doc_count=doc_count
        )

    async def reindex_repo(self, repo_id: str) -> None:
        repo = self._sqlite.get_repo(repo_id)
        if repo is None:
            raise ValueError(f"repo not found: {repo_id}")
        async with self._locks[repo_id]:
            for doc in self._sqlite.list_documents_by_repo(repo_id):
                self._chroma.delete_by_doc_id(doc.id)
                self._sqlite.delete_document(doc.id)
                self._files.delete_doc_files(doc.id)
            self._sqlite.update_repo_status(repo_id, DocumentStatus.PROCESSING)
            self._sqlite.update_repo_progress(repo_id, 0, "Starting re-index (0%)")
            target = Path(repo.local_path)
            try:
                await self._run_import(
                    repo_id, target, is_url=False, source_url=repo.source_url
                )
            except Exception as exc:
                self._sqlite.update_repo_status(
                    repo_id, DocumentStatus.ERROR, error_message=str(exc)
                )
                raise

    async def delete_repo(self, repo_id: str) -> int:
        repo = self._sqlite.get_repo(repo_id)
        if repo is None:
            return 0
        async with self._locks[repo_id]:
            docs = self._sqlite.list_documents_by_repo(repo_id)
            for doc in docs:
                self._chroma.delete_by_doc_id(doc.id)
                self._files.delete_doc_files(doc.id)
                self._sqlite.delete_document(doc.id)
            self._sqlite.delete_repo(repo_id)
            if repo.source_url.startswith(("http://", "https://", "git@")):
                target = Path(repo.local_path)
                try:
                    if target.is_dir() and target.is_relative_to(self._cfg.repos_dir):
                        shutil.rmtree(target, ignore_errors=True)
                except AttributeError:
                    # Python <3.9 lacks is_relative_to; we require >=3.10 globally
                    pass
        return len(docs)
