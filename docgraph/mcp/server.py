from __future__ import annotations

import json
from pathlib import Path

from mcp.server.fastmcp import FastMCP

from docgraph.models import DocumentStatus
from docgraph.repo.codegraph_client import CodegraphNotInstalled
from docgraph.web.deps import AppState


def create_mcp_server(state: AppState) -> FastMCP:
    mcp = FastMCP("docgraph")
    search_svc = state.search_service()
    repos_mgr = state.repos()

    @mcp.tool()
    async def search_documents(
        query: str,
        tags: list[str] | None = None,
        folder: str | None = None,
        top_k: int | None = None,
        repo: str | None = None,
    ) -> str:
        """Semantic search over uploaded documents. Returns relevant chunks with metadata.

        Args:
            query: The search query string.
            tags: Optional list of tags to filter results.
            folder: Optional folder path to restrict the search scope.
            top_k: Maximum number of results (1-100). Default: cfg.default_top_k (5).
            repo: Optional repo id or name to restrict results to one repository.
        """
        repo_id: str | None = None
        if repo:
            target = repos_mgr.resolve(repo)
            if target is None:
                available = [r.name for r in repos_mgr.list_repos()]
                return json.dumps({
                    "error": f"repo not found: {repo}",
                    "available": available,
                    "results": [],
                })
            repo_id = target.id
        try:
            results = await search_svc.search(
                query=query, top_k=top_k, folder=folder, tags=tags, repo_id=repo_id,
            )
        except RuntimeError as exc:
            return json.dumps({"error": str(exc), "results": []})
        if not results:
            return json.dumps({
                "message": "No documents matched. Upload at http://127.0.0.1:8088",
                "results": [],
            })
        payload = [
            {
                "text": r.text,
                "doc_id": r.doc_id,
                "filename": r.filename,
                "folder": r.folder,
                "tags": r.tags,
                "chunk_index": r.chunk_index,
                "score": round(r.score, 4),
                "rrf_score": round(r.rrf_score, 6),
                "source_page": r.source_page,
                "file_path": r.file_path,
                "heading_path": r.heading_path or [],
                "rerank_score": round(r.rerank_score, 4) if r.rerank_score is not None else None,
            }
            for r in results
        ]
        return json.dumps({"results": payload})

    @mcp.tool()
    async def list_documents(
        folder: str | None = None,
        tag: str | None = None,
        status: str | None = None,
    ) -> str:
        """List indexed documents with optional filters."""
        doc_status = DocumentStatus(status) if status else None
        docs = state.sqlite.list_documents(
            folder=folder, tag=tag, status=doc_status
        )
        return json.dumps([{
            "id": d.id,
            "filename": d.filename,
            "folder": d.folder,
            "tags": d.tags,
            "status": d.status.value,
            "chunk_count": d.chunk_count,
            "progress_pct": d.progress_pct,
            "progress_phase": d.progress_phase,
            "repo_id": d.repo_id,
        } for d in docs])

    @mcp.tool()
    async def get_document(doc_id: str) -> str:
        """Get full converted markdown for a document."""
        doc = state.sqlite.get_document(doc_id)
        if doc is None:
            return json.dumps({"error": f"document not found: {doc_id}"})
        try:
            md = state.files.read_markdown(doc_id)
        except FileNotFoundError:
            return json.dumps({"error": "markdown not available", "doc_id": doc_id})
        return json.dumps({
            "doc_id": doc_id,
            "filename": doc.filename,
            "folder": doc.folder,
            "tags": doc.tags,
            "markdown": md,
        })

    # --- repos / codegraph integration -------------------------------------

    @mcp.tool()
    async def list_repos() -> str:
        """List imported repositories."""
        return json.dumps([{
            "id": r.id, "name": r.name, "status": r.status.value,
            "progress_pct": r.progress_pct, "progress_phase": r.progress_phase,
            "doc_count": r.doc_count, "source_url": r.source_url,
        } for r in repos_mgr.list_repos()])

    @mcp.tool()
    async def import_repo(
        source: str,
        folder: str = "",
        tags: list[str] | None = None,
    ) -> str:
        """Clone (if URL) and index a repository via codegraph + DocGraph."""
        try:
            repo_id = await repos_mgr.import_repo(
                source, folder=folder, tags=tuple(tags or ()),
            )
        except (CodegraphNotInstalled, ValueError, RuntimeError) as exc:
            return json.dumps({"error": str(exc)})
        return json.dumps({"repo_id": repo_id, "status": "processing"})

    async def _run_codegraph(subcommand: str, repo: str | None, *args: str) -> str:
        target = repos_mgr.resolve(repo)
        if target is None:
            available = [r.name for r in repos_mgr.list_repos()]
            return json.dumps({
                "error": "specify repo (id or name)",
                "available": available,
            })
        if target.status != DocumentStatus.READY:
            return json.dumps({
                "error": "repo not ready",
                "status": target.status.value,
                "progress_pct": target.progress_pct,
            })
        try:
            result = await state.codegraph.run(
                subcommand, *args, repo_path=Path(target.local_path),
            )
        except CodegraphNotInstalled as exc:
            return json.dumps({
                "error": str(exc), "install_hint": state.codegraph.INSTALL_HINT,
            })
        except (TimeoutError, RuntimeError) as exc:
            return json.dumps({"error": str(exc)})
        return json.dumps({"repo": target.name, "result": result})

    @mcp.tool()
    async def code_search(query: str, repo: str | None = None) -> str:
        """Find a symbol by name/text in an imported repo."""
        return await _run_codegraph("search", repo, query)

    @mcp.tool()
    async def code_explore(symbols: list[str], repo: str | None = None) -> str:
        """Fetch source/context for multiple symbols in one call."""
        return await _run_codegraph("explore", repo, *symbols)

    @mcp.tool()
    async def code_callers(symbol: str, repo: str | None = None) -> str:
        """List callers of a symbol."""
        return await _run_codegraph("callers", repo, symbol)

    @mcp.tool()
    async def code_callees(symbol: str, repo: str | None = None) -> str:
        """List callees of a symbol."""
        return await _run_codegraph("callees", repo, symbol)

    @mcp.tool()
    async def code_trace(from_sym: str, to_sym: str, repo: str | None = None) -> str:
        """Trace a call path from one symbol to another."""
        return await _run_codegraph("trace", repo, from_sym, to_sym)

    @mcp.tool()
    async def code_context(query: str, repo: str | None = None) -> str:
        """Composed search + node + edges for a query."""
        return await _run_codegraph("context", repo, query)

    @mcp.tool()
    async def code_files(path: str = "", repo: str | None = None) -> str:
        """List files in an imported repo (optionally under a path)."""
        extra = (path,) if path else ()
        return await _run_codegraph("files", repo, *extra)

    return mcp
