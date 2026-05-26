from __future__ import annotations

import json

from mcp.server.fastmcp import FastMCP

from boostmcp.mcp.search import SearchService
from boostmcp.web.deps import AppState


def create_mcp_server(state: AppState) -> FastMCP:
    mcp = FastMCP("boostmcp")
    search_svc = SearchService(
        state.cfg, state.sqlite, state.chroma, state.embedder
    )

    @mcp.tool()
    async def search_documents(
        query: str,
        tags: list[str] | None = None,
        folder: str | None = None,
        top_k: int | None = None,
    ) -> str:
        """Semantic search over uploaded documents. Returns relevant chunks with metadata."""
        tag = tags[0] if tags else None
        try:
            results = await search_svc.search(
                query=query, top_k=top_k, folder=folder, tag=tag
            )
        except RuntimeError as exc:
            return json.dumps({"error": str(exc), "results": []})
        if not results:
            return json.dumps({
                "message": "No documents matched. Upload at http://127.0.0.1:8080",
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
                "source_page": r.source_page,
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
        from boostmcp.models import DocumentStatus

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

    return mcp
