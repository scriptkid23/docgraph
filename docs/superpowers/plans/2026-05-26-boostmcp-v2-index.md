# BoostMCP v2 — Implementation Plans Index

> **Spec:** [`docs/superpowers/specs/2026-05-26-boostmcp-v2-rag-design.md`](../specs/2026-05-26-boostmcp-v2-rag-design.md) (Approved)

**Goal:** Replace Go v1 code co-processor with Python document RAG — Web UI upload, MarkItDown conversion, Ollama embedding, ChromaDB search, MCP tools, `/document` Cursor skill.

**Strategy:** Seven small plans, each producing working, testable software. Execute in order; Plans 03 and 04 can run in parallel after Plan 02.

---

## Plan Map

```mermaid
flowchart LR
    P01["Plan 01<br/>Foundation"]
    P02["Plan 02<br/>Storage"]
    P03["Plan 03<br/>Embedding"]
    P04["Plan 04<br/>Ingest"]
    P05["Plan 05<br/>Web UI"]
    P06["Plan 06<br/>MCP & Serve"]
    P07["Plan 07<br/>E2E & Docs"]

    P01 --> P02
    P02 --> P03
    P02 --> P04
    P03 --> P04
    P04 --> P05
    P05 --> P06
    P06 --> P07
```

| # | Plan | Delivers | Est. tasks |
|---|---|---|---|
| 01 | [Foundation & Migration](./2026-05-26-plan-v2-01-foundation.md) | Poetry project, config, domain types, remove Go v1 | 5 |
| 02 | [Storage Layer](./2026-05-26-plan-v2-02-storage.md) | SQLite metadata, ChromaDB vectors, file store | 4 |
| 03 | [Embedding](./2026-05-26-plan-v2-03-embedding.md) | Ollama + OpenAI embedding providers | 3 |
| 04 | [Ingest Pipeline](./2026-05-26-plan-v2-04-ingest.md) | MarkItDown, chunker, indexer | 4 |
| 05 | [Web UI & API](./2026-05-26-plan-v2-05-web.md) | FastAPI upload/management + static UI | 4 |
| 06 | [MCP & Serve](./2026-05-26-plan-v2-06-mcp-serve.md) | MCP tools, `boostmcp serve` monolith | 4 |
| 07 | [Skill, E2E & Docs](./2026-05-26-plan-v2-07-e2e-docs.md) | `/document` skill, integration tests, README | 3 |

---

## Success Criteria (v2)

Mapped from spec §2, §5, §8, §13:

| Criterion | Plan |
|---|---|
| Go v1 removed; v1 docs archived | 01 |
| Config loads env + YAML with defaults | 01 |
| SQLite stores docs/tags/folders/status | 02 |
| ChromaDB stores chunk vectors + metadata | 02 |
| Ollama embed + health check | 03 |
| MarkItDown converts local files to markdown | 04 |
| Upload → index pipeline (processing/ready/error) | 04, 05 |
| Web UI drag-drop upload + doc list | 05 |
| MCP `search_documents`, `list_documents`, `get_document` | 06 |
| `boostmcp serve` runs MCP stdio + FastAPI :8080 | 06 |
| `/document` Cursor skill shipped | 07 |
| E2E: upload API → MCP search | 07 |
| README with Poetry + Cursor setup | 07 |

---

## Tech Stack (all plans)

- **Language:** Python 3.10+
- **Package manager:** Poetry
- **Web:** FastAPI + uvicorn + static HTML/JS
- **MCP:** `mcp` Python SDK (stdio)
- **Convert:** `markitdown[all]`
- **Vectors:** ChromaDB (local persist)
- **Metadata:** SQLite (`aiosqlite` or stdlib `sqlite3`)
- **Embedding (default):** Ollama `nomic-embed-text`
- **Testing:** pytest, pytest-asyncio, httpx

---

## Execution Options

After reviewing plans, choose:

1. **Subagent-Driven** — fresh subagent per plan/task, review between tasks
2. **Inline Execution** — implement plan-by-plan in this session with checkpoints

Start with **Plan 01** regardless of approach.
