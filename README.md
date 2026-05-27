# DocGraph

Local document RAG server for Cursor via MCP. Upload files through a Web UI, index them with Ollama embeddings, and query your documents in Cursor using the `/document` skill.

## Prerequisites

- Python 3.10–3.13
- [Poetry](https://python-poetry.org/)
- [Ollama](https://ollama.com/) running locally
- Embedding model: `ollama pull nomic-embed-text`

## Install

```bash
poetry install
poetry install -E openai   # optional: cloud embedding
ollama pull nomic-embed-text
```

## Run

```bash
poetry run docgraph serve
```

- Web UI: http://127.0.0.1:8080 (React — upload and manage documents)

### Web UI development (React + Vite)

The UI uses a **Minimalist Monochrome** design system (Playfair Display, Source Serif 4, JetBrains Mono; pure black/white; sharp corners; line-based progress).

```bash
cd frontend
npm install
npm run dev          # http://127.0.0.1:5173 — proxies /api → :8080
npm run build        # output → docgraph/web/static/
```

Design tokens live in `frontend/src/styles/`. Run `npm run build` after UI changes.
- MCP SSE: http://127.0.0.1:8080/mcp/sse (connect Cursor via URL below)

Keep this terminal running while using Cursor.

## Cursor MCP Configuration

**Recommended — server chạy riêng, MCP trỏ HTTP** (chạy `docgraph serve` trước):

```json
{
  "mcpServers": {
    "docgraph": {
      "command": "npx",
      "args": ["-y", "mcp-remote@latest", "http://127.0.0.1:8080/mcp/sse"]
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
| `DOCGRAPH_WEB_PORT` | `8080` | Web UI port |
| `DOCGRAPH_EMBED_PROVIDER` | `ollama` | `ollama` or `openai` |
| `DOCGRAPH_OLLAMA_URL` | `http://127.0.0.1:11434` | Ollama endpoint (use `127.0.0.1` on Windows, not `localhost`) |
| `DOCGRAPH_OLLAMA_EMBED_MODEL` | `nomic-embed-text` | Embedding model |

See [v2 design spec](docs/superpowers/specs/2026-05-26-docgraph-v2-rag-design.md) and [implementation plans](docs/superpowers/plans/2026-05-26-docgraph-v2-index.md).

## Test

```bash
poetry run pytest tests/ -v
poetry run pytest tests/ -m integration -v   # requires Ollama
```
