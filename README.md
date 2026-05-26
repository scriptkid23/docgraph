# BoostMCP

Local AI code co-processor for Cursor via MCP. Generates multiple SLM candidates, narrows them with a rubric, and returns top-k for Cursor to pick.

## Prerequisites

- Go 1.22+
- [Ollama](https://ollama.com/) running locally
- A code model pulled, e.g. `ollama pull codellama:7b`

## Build

```bash
go build -o boostmcp ./cmd/boostmcp
```

## Cursor MCP Configuration

Add to Cursor MCP settings (`~/.cursor/mcp.json` or Cursor Settings → MCP):

```json
{
  "mcpServers": {
    "boostmcp": {
      "command": "boostmcp",
      "args": []
    }
  }
}
```

Use full path to binary if not on PATH.

## Tools

| Tool | Description |
|------|-------------|
| `generate_candidates` | Generate N code candidates from local SLM |
| `narrow_candidates` | Score and filter candidates using a rubric |

## Configuration

| Env Variable | Default | Description |
|---|---|---|
| `BOOSTMCP_MODEL` | `codellama:7b` | Default model |
| `BOOSTMCP_OLLAMA_URL` | `http://localhost:11434` | Ollama URL |
| `BOOSTMCP_TIMEOUT_MS` | `30000` | Per-call timeout |
| `BOOSTMCP_MAX_CANDIDATES` | `16` | Max N |
| `BOOSTMCP_DEFAULT_TOP_K` | `2` | Default top-k |

## Test

```bash
go test ./...
$env:OLLAMA_INTEGRATION="1"; go test -tags=integration ./internal/inference/ollama/... -v
go test -tags=e2e ./internal/mcp/... -v
```

## Architecture

See [design spec](docs/superpowers/specs/2026-05-26-boostmcp-v1-design.md) and [implementation plans](docs/superpowers/plans/2026-05-26-boostmcp-v1-index.md).
