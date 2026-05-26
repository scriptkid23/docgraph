# Plan v2-06 — MCP Server & Serve CLI

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** MCP stdio tools for document search and `boostmcp serve` monolith that runs MCP + FastAPI together.

**Architecture:** `boostmcp/mcp/server.py` registers three tools using MCP Python SDK. `boostmcp/cli.py` `serve` command starts uvicorn in a daemon thread then runs MCP stdio on main thread. Shared `AppState` from web deps.

**Tech Stack:** mcp Python SDK, uvicorn

**Depends on:** Plan v2-05  
**Blocks:** Plan v2-07

**Spec refs:** §3.3 Process Model, §5.3 MCP Tools, §8.2 Error Behavior

---

## File Structure

```
boostmcp/mcp/
├── __init__.py
└── server.py
boostmcp/cli.py          # modify serve command
tests/mcp/
└── test_tools.py
```

Add: `poetry add mcp`

---

### Task 1: MCP search service

**Files:**
- Create: `boostmcp/mcp/search.py`
- Create: `tests/mcp/test_search.py`

- [ ] **Step 1: Write failing test**

```python
# tests/mcp/test_search.py
import pytest
import respx
import httpx

from boostmcp.config import Config
from boostmcp.mcp.search import SearchService
from boostmcp.models import DocumentRecord
from boostmcp.store import ChromaStore, SQLiteStore
from boostmcp.embed.ollama import OllamaEmbedder


@pytest.mark.asyncio
@respx.mock
async def test_search_returns_results(tmp_data_dir):
    cfg = Config(data_dir=tmp_data_dir)
    cfg.ensure_dirs()
    sqlite = SQLiteStore(cfg)
    sqlite.init_schema()
    chroma = ChromaStore(cfg)
    sqlite.insert_document(DocumentRecord(id="doc_1", filename="a.md", folder="F"))
    vec = [0.1] * 768
    chroma.upsert_chunks([{
        "id": "doc_1_0",
        "embedding": vec,
        "text": "embedding config",
        "metadata": {"doc_id": "doc_1", "filename": "a.md", "folder": "F", "tags": "x", "chunk_index": 0},
    }])
    respx.post(f"{cfg.ollama_url}/api/embed").mock(
        return_value=httpx.Response(200, json={"embeddings": [vec]})
    )
    svc = SearchService(cfg, sqlite, chroma, OllamaEmbedder(cfg.ollama_url, cfg.ollama_model))
    results = await svc.search("how to configure embedding", top_k=5)
    assert len(results) == 1
    assert results[0].filename == "a.md"


@pytest.mark.asyncio
@respx.mock
async def test_search_empty_message(tmp_data_dir):
    cfg = Config(data_dir=tmp_data_dir)
    cfg.ensure_dirs()
    sqlite = SQLiteStore(cfg)
    sqlite.init_schema()
    chroma = ChromaStore(cfg)
    respx.post(f"{cfg.ollama_url}/api/embed").mock(
        return_value=httpx.Response(200, json={"embeddings": [[0.1] * 768]})
    )
    svc = SearchService(cfg, sqlite, chroma, OllamaEmbedder(cfg.ollama_url, cfg.ollama_model))
    results = await svc.search("nonexistent topic xyz", top_k=5)
    assert results == []
```

- [ ] **Step 2: Implement**

```python
# boostmcp/mcp/search.py
from __future__ import annotations

from typing import Optional

from boostmcp.config import Config
from boostmcp.embed.provider import EmbeddingProvider
from boostmcp.models import SearchResult
from boostmcp.store.chroma import ChromaStore
from boostmcp.store.sqlite import SQLiteStore


class SearchService:
    def __init__(
        self,
        cfg: Config,
        sqlite: SQLiteStore,
        chroma: ChromaStore,
        embedder: EmbeddingProvider,
    ) -> None:
        self._cfg = cfg
        self._sqlite = sqlite
        self._chroma = chroma
        self._embedder = embedder

    async def search(
        self,
        query: str,
        top_k: Optional[int] = None,
        folder: Optional[str] = None,
        tag: Optional[str] = None,
    ) -> list[SearchResult]:
        k = top_k or self._cfg.default_top_k
        vectors = await self._embedder.embed([query])
        raw = self._chroma.search(
            query_embedding=vectors[0],
            top_k=k,
            folder=folder,
            tag=tag,
        )
        results = [
            SearchResult(
                text=r["text"],
                doc_id=r["doc_id"],
                filename=r["filename"],
                folder=r["folder"],
                tags=r["tags"],
                chunk_index=r["chunk_index"],
                score=r["score"],
                source_page=r.get("source_page"),
            )
            for r in raw
            if r["score"] >= self._cfg.min_score
        ]
        return results
```

- [ ] **Step 3: Run tests — expect PASS**

- [ ] **Step 4: Commit**

```bash
git add boostmcp/mcp/search.py tests/mcp/test_search.py
git commit -m "feat: add search service for MCP tools"
```

---

### Task 2: MCP server with tools

**Files:**
- Create: `boostmcp/mcp/server.py`
- Create: `tests/mcp/test_tools.py`

- [ ] **Step 1: Implement MCP server**

```python
# boostmcp/mcp/server.py
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
        st = status
        doc_status = DocumentStatus(st) if st else None
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
```

- [ ] **Step 2: Write integration test**

```python
# tests/mcp/test_tools.py
import json
import pytest

from boostmcp.config import Config
from boostmcp.mcp.server import create_mcp_server
from boostmcp.web.deps import AppState


@pytest.mark.asyncio
async def test_list_documents_tool(tmp_data_dir):
    cfg = Config(data_dir=tmp_data_dir)
    state = AppState.create(cfg)
    mcp = create_mcp_server(state)
    tools = {t.name: t for t in await mcp.list_tools()}
    assert "search_documents" in tools
    assert "list_documents" in tools
    assert "get_document" in tools
```

- [ ] **Step 3: Run tests**

- [ ] **Step 4: Commit**

```bash
git add boostmcp/mcp/ tests/mcp/
git commit -m "feat: register MCP tools for search, list, and get document"
```

---

### Task 3: `boostmcp serve` monolith

**Files:**
- Modify: `boostmcp/cli.py`

- [ ] **Step 1: Implement serve command**

```python
# boostmcp/cli.py
import argparse
import logging
import sys
import threading

import uvicorn

from boostmcp.config import load_config
from boostmcp.web.app import create_app


def _start_web(cfg):
    app = create_app(cfg)
    uvicorn.run(
        app,
        host=cfg.web_host,
        port=cfg.web_port,
        log_level="warning",
    )


def _run_serve() -> None:
    cfg = load_config()
    cfg.ensure_dirs()
    logging.basicConfig(level=logging.INFO)
    log = logging.getLogger("boostmcp")

    # Background Web UI
    web_thread = threading.Thread(
        target=_start_web, args=(cfg,), daemon=True, name="boostmcp-web"
    )
    web_thread.start()
    log.info("Web UI at http://%s:%s", cfg.web_host, cfg.web_port)

    # Health check (non-blocking)
    from boostmcp.embed.factory import create_embedder
    embedder = create_embedder(cfg)
    import asyncio
    try:
        asyncio.run(embedder.health_check())
        log.info("Embedding provider OK (%s)", cfg.embed_provider)
    except Exception as exc:
        log.warning("Embedding health check failed: %s", exc)

    # MCP stdio on main thread
    from boostmcp.web.deps import AppState
    from boostmcp.mcp.server import create_mcp_server

    state = AppState.create(cfg)
    mcp = create_mcp_server(state)
    mcp.run(transport="stdio")


def main() -> None:
    parser = argparse.ArgumentParser(prog="boostmcp")
    sub = parser.add_subparsers(dest="command")
    sub.add_parser("serve", help="Start MCP + Web UI server")
    args = parser.parse_args()
    if args.command == "serve":
        _run_serve()
        return
    parser.print_help()
    sys.exit(1)
```

- [ ] **Step 2: Manual verify**

```bash
poetry run boostmcp serve
```

Expected stderr: Web UI URL + health log. Cursor can attach via stdio.

- [ ] **Step 3: Commit**

```bash
git add boostmcp/cli.py
git commit -m "feat: implement boostmcp serve monolith with MCP stdio and Web UI"
```

---

### Task 4: MCP package init

- [ ] **Step 1: Export**

```python
# boostmcp/mcp/__init__.py
from boostmcp.mcp.server import create_mcp_server

__all__ = ["create_mcp_server"]
```

- [ ] **Step 2: Run all MCP tests**

```bash
poetry run pytest tests/mcp/ -v
```

- [ ] **Step 3: Commit**

```bash
git add boostmcp/mcp/__init__.py
git commit -m "chore: export MCP server factory"
```
