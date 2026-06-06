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

Add to your Cursor MCP settings (`Settings → MCP` or `.cursor/mcp.json`).

**Recommended — run the server separately, connect via HTTP** (start `docgraph serve` first):

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
| `DOCGRAPH_HYBRID_ENABLED` | `true` | Enable hybrid search (vector + BM25/FTS5) |
| `DOCGRAPH_RRF_K` | `60` | Reciprocal Rank Fusion constant |
| `DOCGRAPH_RERANK_ENABLED` | `true` | Enable cross-encoder reranker |
| `DOCGRAPH_RERANK_MODEL` | `bge-reranker-v2-m3` | Reranker model (fastembed) |
| `DOCGRAPH_RERANK_TOP_N` | `15` | Number of candidates passed to the reranker |
| `DOCGRAPH_RERANK_TIMEOUT_SEC` | `3.0` | Per-call rerank timeout (falls back to RRF order on timeout) |
| `DOCGRAPH_RERANK_PREWARM` | `true` | Warm up reranker model at server startup |
| `DOCGRAPH_RERANK_SCORE_GAP_RATIO` | `0.5` | Skip rerank when top-1 RRF dominates by this ratio |
| `DOCGRAPH_RERANK_MIN_FLOOR` | `0.015` | Skip rerank when top-1 RRF score is below this floor |

See [v2 design spec](docs/superpowers/specs/2026-05-26-docgraph-v2-rag-design.md) and [implementation plans](docs/superpowers/plans/2026-05-26-docgraph-v2-index.md).

## Troubleshooting

**"Hybrid search returns vector-only results"**
The FTS5 index hasn't been populated (e.g., after upgrading from a pre-hybrid version).
- Automatic: server startup detects an empty FTS index and rebuilds in the background.
- Manual: `poetry run docgraph rebuild-fts`.

**"Reranker reports disabled / error"**
Check `GET /api/health` field `rerank_status`:
- `error` — reranker construction failed. Rebuild the Rust crate inside the Poetry venv:
  ```bash
  env -u CONDA_PREFIX -u CONDA_DEFAULT_ENV -u CONDA_SHLVL -u CONDA_PROMPT_MODIFIER \
    poetry run maturin develop --release \
    --manifest-path crates/docgraph-embed/Cargo.toml
  ```
- `loading` — wait 5–10s; the BGE model is downloading on first use (~600MB).
- `disabled` — set `DOCGRAPH_RERANK_ENABLED=true` or enable it in the YAML config.

**"First search after startup is slow (~10s)"**
Reranker cold-start. `rerank_prewarm=true` (default) warms the model in the background at server start; if the first query is still slow, check the server log for prewarm errors.

## Test

```bash
poetry run pytest tests/ -v
poetry run pytest tests/ -m integration -v   # optional: live embedder / Ollama
```
