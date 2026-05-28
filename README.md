# DocGraph

Local document RAG server for Cursor via MCP. Upload files or import URLs, index with a **local Rust embedder** (ONNX, no Ollama required), and query in Cursor via the `/document` skill.

## Prerequisites

- Python 3.10–3.13
- [Poetry](https://python-poetry.org/)
- [Rust toolchain](https://rustup.rs/) + [maturin](https://www.maturin.rs/) (for local embeddings)
- Optional: [Ollama](https://ollama.com/) if `DOCGRAPH_EMBED_PROVIDER=ollama`

## Install

```bash
poetry install
pip install maturin
cd crates/docgraph-embed && maturin develop --release && cd ../..
```

First run downloads the ONNX model (~100MB) to `~/.docgraph/models`.

Optional extras:

```bash
poetry install -E openai   # cloud embedding instead of local/ollama
poetry install -E crawl    # import URLs via crawl4ai
```

For URL import, install the crawl extra and set up the browser:

```bash
poetry install -E crawl
poetry run crawl4ai-setup
# or: python -m playwright install chromium
```

## Run

```bash
poetry run docgraph serve
```

- Web UI: http://127.0.0.1:8088 (React — upload files, import URLs, manage documents)

### Web UI development (React + Vite)

The UI uses a **Minimalist Monochrome** design system (Playfair Display, Source Serif 4, JetBrains Mono; pure black/white; sharp corners; line-based progress).

```bash
cd frontend
npm install
npm run dev          # http://127.0.0.1:5173 — proxies /api → :8088
npm run build        # output → docgraph/web/static/
```

Design tokens live in `frontend/src/styles/`. Run `npm run build` after UI changes.
- MCP SSE: http://127.0.0.1:8088/mcp/sse (connect Cursor via URL below)

Keep this terminal running while using Cursor.

## Cursor MCP Configuration

**Recommended — server chạy riêng, MCP trỏ HTTP** (chạy `docgraph serve` trước):

```json
{
  "mcpServers": {
    "docgraph": {
      "command": "npx",
      "args": ["-y", "mcp-remote@latest", "http://127.0.0.1:8088/mcp/sse"]
    }
  }
}
```

**Alternative — Cursor tự launch process (stdio):**

```json
{
  "mcpServers": {
    "docgraph": {
      "command": "poetry",
      "args": ["run", "docgraph", "serve", "--stdio"],
      "cwd": "C:/Users/hoan.do/Documents/project/DocGraph"
    }
  }
}
```

## `/document` Skill

Copy `skills/document/SKILL.md` to your Cursor skills directory:

- Global: `~/.cursor/skills/document/SKILL.md`
- Project: `.cursor/skills/document/SKILL.md`

Then in Cursor chat:

```
/document How do I configure the embedding provider?
/document --folder docs --tag v2 What is DocGraph v2?
```

## MCP Tools

| Tool | Description |
|------|-------------|
| `search_documents` | Semantic search over indexed documents |
| `list_documents` | List documents with optional filters |
| `get_document` | Get full converted markdown for a document |

## Configuration

| Env Variable | Default | Description |
|---|---|---|
| `DOCGRAPH_DATA_DIR` | `~/.docgraph` | Data directory |
| `DOCGRAPH_WEB_PORT` | `8088` | Web UI port |
| `DOCGRAPH_EMBED_PROVIDER` | `local` | `local`, `ollama`, or `openai` |
| `DOCGRAPH_LOCAL_MODEL` | `nomic-embed-text` | Local ONNX (English, 768-dim). For multilingual: `multilingual-e5-base` |
| `DOCGRAPH_OLLAMA_URL` | `http://127.0.0.1:11434` | Ollama endpoint when provider=ollama |
| `DOCGRAPH_OLLAMA_EMBED_MODEL` | `nomic-embed-text:latest` | Ollama embedding model |
| `DOCGRAPH_CRAWL_TIMEOUT_SEC` | `30` | Per-URL crawl timeout |
| `DOCGRAPH_MAX_URLS_PER_IMPORT` | `50` | Max URLs per import batch |
| `DOCGRAPH_MAX_CHUNKS_PER_DOC` | `5000` | Hard cap on chunks per document (oversize → ERROR) |

See [v2 design spec](docs/superpowers/specs/2026-05-26-docgraph-v2-rag-design.md) and [implementation plans](docs/superpowers/plans/2026-05-26-docgraph-v2-index.md).

## Test

```bash
poetry run pytest tests/ -v
poetry run pytest tests/ -m integration -v   # optional: live embedder / Ollama
```
