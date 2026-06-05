# DocGraph

Local document RAG server for Cursor via MCP. Upload files or import URLs, index them with a **local Rust embedder** (ONNX, no Ollama or cloud key required), and query them from Cursor through the `/document` skill.

- **Web UI** — upload files, import URLs, and manage documents
- **MCP server** — `search_documents`, `list_documents`, `get_document` exposed to Cursor
- **Local embeddings** — fast ONNX embedder written in Rust; nothing leaves your machine

---

## Quick start with Docker (recommended)

No need to install Python, Poetry, Rust, or Node — just run Docker.

```bash
docker compose up --build
```

Or with plain Docker:

```bash
docker build -t docgraph .
docker run -p 8088:8088 -v docgraph-data:/data docgraph
```

Then open:

- Web UI: http://127.0.0.1:8088
- MCP SSE: http://127.0.0.1:8088/mcp/sse

Notes:

- The **first build** is slow (it compiles the Rust embedder and downloads dependencies); later builds are cached.
- The **first run** downloads the ONNX model (~100MB) into the `/data` volume; later runs reuse it.
- All data (sqlite db, Chroma index, uploaded files, model cache) is persisted in the `docgraph-data` volume, so it survives restarts.
- For non-English documents, set `DOCGRAPH_LOCAL_MODEL=multilingual-e5-base` (uncomment it in `docker-compose.yml`) and re-index.

---

## Manual install (for development)

### Prerequisites

- Python 3.10–3.13
- [Poetry](https://python-poetry.org/)
- [Rust toolchain](https://rustup.rs/) + [maturin](https://www.maturin.rs/) (for local embeddings)
- Optional: [Ollama](https://ollama.com/) if `DOCGRAPH_EMBED_PROVIDER=ollama`

### Install

```bash
poetry install
pip install maturin
cd crates/docgraph-embed && maturin develop --release && cd ../..
```

Poetry-only setup (no global `pip install`):

```bash
poetry install
poetry add --group dev maturin
poetry run maturin develop --release -m crates/docgraph-embed/Cargo.toml
```

First run downloads the ONNX model (~100MB) to `~/.docgraph/models`.

#### Windows troubleshooting: `maturin develop` cannot overwrite module in use

If `maturin develop` fails with a message about `pip could not overwrite the installed extension module` and a leftover directory like `~ocgraph_embed`, another Python process is still holding the old compiled extension.

1. Stop all Python processes that may have imported `docgraph_embed` (for example, old `docgraph serve` terminals).
2. Delete the leftover directory in the Poetry venv site-packages (for example, `.../Lib/site-packages/~ocgraph_embed`).
3. Re-run:

```bash
poetry run maturin develop --release -m crates/docgraph-embed/Cargo.toml
```

### Optional extras

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

### Run

```bash
poetry run docgraph serve
```

- Web UI: http://127.0.0.1:8088 (React — upload files, import URLs, manage documents)
- MCP SSE: http://127.0.0.1:8088/mcp/sse

Keep this terminal running while using Cursor.

### Web UI development (React + Vite)

The UI uses a **Minimalist Monochrome** design system (Playfair Display, Source Serif 4, JetBrains Mono; pure black/white; sharp corners; line-based progress).

```bash
cd frontend
npm install
npm run dev          # http://127.0.0.1:5173 — proxies /api → :8088
npm run build        # output → docgraph/web/static/
```

Design tokens live in `frontend/src/styles/`. Run `npm run build` after UI changes.

---

## Cursor MCP configuration

Add to your Cursor MCP settings (`Settings → MCP` or `.cursor/mcp.json`).

**Recommended — run the server separately, connect via HTTP** (start the server, via Docker or `docgraph serve`, first):

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

**Alternative — let Cursor launch the process (stdio):**

```json
{
  "mcpServers": {
    "docgraph": {
      "command": "poetry",
      "args": ["run", "docgraph", "serve", "--stdio"],
      "cwd": "/path/to/docgraph"
    }
  }
}
```

Replace `/path/to/docgraph` with the absolute path to this repository on your machine.

---

## `/document` skill

Copy `skills/document/SKILL.md` to your Cursor skills directory:

- Global: `~/.cursor/skills/document/SKILL.md`
- Project: `.cursor/skills/document/SKILL.md`

Then in Cursor chat:

```
/document How do I configure the embedding provider?
/document --folder docs --tag v2 What is DocGraph v2?
```

---

## MCP tools

| Tool | Description |
|------|-------------|
| `search_documents` | Semantic search over indexed documents |
| `list_documents` | List documents with optional filters |
| `get_document` | Get full converted markdown for a document |

---

## Configuration

| Env Variable | Default | Description |
|---|---|---|
| `DOCGRAPH_DATA_DIR` | `~/.docgraph` (`/data` in Docker) | Data directory |
| `DOCGRAPH_WEB_HOST` | `127.0.0.1` (`0.0.0.0` in Docker) | Bind address for the Web UI / MCP server |
| `DOCGRAPH_WEB_PORT` | `8088` | Web UI port |
| `DOCGRAPH_EMBED_PROVIDER` | `local` | `local`, `ollama`, or `openai` |
| `DOCGRAPH_LOCAL_MODEL` | `nomic-embed-text` | Local ONNX (English, 768-dim). For multilingual: `multilingual-e5-base` |
| `DOCGRAPH_OLLAMA_URL` | `http://127.0.0.1:11434` | Ollama endpoint when provider=ollama |
| `DOCGRAPH_OLLAMA_EMBED_MODEL` | `nomic-embed-text:latest` | Ollama embedding model |
| `DOCGRAPH_CRAWL_TIMEOUT_SEC` | `30` | Per-URL crawl timeout |
| `DOCGRAPH_MAX_URLS_PER_IMPORT` | `50` | Max URLs per import batch |
| `DOCGRAPH_MAX_CHUNKS_PER_DOC` | `5000` | Hard cap on chunks per document (oversize → ERROR) |
| `DOCGRAPH_TOKENIZER_SOURCE` | `auto` | `auto`, `char-ratio`, or `tiktoken` |
| `DOCGRAPH_HEADING_PREFIX` | `true` | Prepend heading breadcrumb to markdown chunks |
| `DOCGRAPH_ATOMIC_BLOCKS` | `true` | Keep code fences and tables atomic when chunking |
| `DOCGRAPH_CODE_CHUNKER` | `regex` | `regex` or `treesitter` (requires `poetry install -E treesitter`) |
| `DOCGRAPH_DEDUP_ENABLED` | `true` | Skip duplicate chunk text within a document |
| `DOCGRAPH_MMR_LAMBDA` | `0.7` | MMR diversity (1.0 = pure relevance) |
| `DOCGRAPH_RERANK_ENABLED` | `false` | Cross-encoder rerank (`poetry install -E rerank`) |

Re-index after changing chunk settings:

```bash
poetry run docgraph reindex --all
# or inside Docker:
docker compose exec docgraph docgraph reindex --all
```

See the [chunker improvements spec](docs/superpowers/specs/2026-06-03-chunker-improvements-design.md).

---

## Test

```bash
poetry run pytest tests/ -v
poetry run pytest tests/ -m integration -v   # optional: live embedder / Ollama
```
