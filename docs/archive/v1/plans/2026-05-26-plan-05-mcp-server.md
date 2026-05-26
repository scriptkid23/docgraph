# Plan 05 — MCP Server

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Wire MCP stdio server exposing `generate_candidates` and `narrow_candidates` tools.

**Architecture:** `cmd/boostmcp` loads config, creates inference provider, registers MCP tools via `mcp-go`. JSON args map to pipeline inputs; results serialized as JSON tool output.

**Tech Stack:** Go, `github.com/mark3labs/mcp-go`

**Depends on:** Plans [01](./2026-05-26-plan-01-foundation.md)–[04](./2026-05-26-plan-04-narrower.md)  
**Blocks:** [Plan 06](./2026-05-26-plan-06-e2e-docs.md)

**Spec refs:** §3 Architecture, §6 MCP Tool Contract, §9.3 Cursor config

---

## File Structure

```
cmd/boostmcp/
└── main.go
internal/mcp/
├── server.go          # NewServer, Run (stdio)
├── tools.go           # register tools
└── tools_test.go      # handler unit tests
```

---

### Task 0: Pin and spike mcp-go

The `mcp-go` API has changed shape across releases (tool registration, argument access, schema helpers). Before writing handlers, pin a version and confirm the API surface used below still exists.

- [ ] **Step 1: Pin a known-good version**

```bash
go get github.com/mark3labs/mcp-go@latest
go mod tidy
```

Record the resolved version in `go.mod` and verify the following symbols exist in the chosen version. If any are missing or renamed, **stop and update this plan** before continuing:

- `server.NewMCPServer(name, version string) *MCPServer`
- `server.ServeStdio(s *MCPServer) error`
- `mcp.NewTool(name string, opts ...ToolOption) Tool`
- `mcp.WithDescription`, `mcp.WithString`, `mcp.WithNumber`, `mcp.WithObject`, `mcp.WithArray`, `mcp.Required`
- Handler signature: `func(ctx context.Context, req mcp.CallToolRequest) (*mcp.CallToolResult, error)`
- Argument access: either `req.Params.Arguments` (raw `map[string]any`) **or** a helper like `req.GetArguments()` — confirm which and adjust `ParseGenerateArgs` / `ParseNarrowArgs` accordingly.
- `mcp.NewToolResultText`, `mcp.NewToolResultError`

- [ ] **Step 2: Smoke build**

```bash
go build ./...
```

Expected: builds (no consumers yet — just verifies the import resolves).

- [ ] **Step 3: Commit**

```bash
git add go.mod go.sum
git commit -m "chore: pin mcp-go for MCP server"
```

---

### Task 1: MCP server bootstrap

**Files:**
- Create: `internal/mcp/server.go`
- Create: `cmd/boostmcp/main.go`

- [ ] **Step 1: (mcp-go already pinned in Task 0)**

- [ ] **Step 2: Implement server skeleton**

```go
// internal/mcp/server.go
package mcp

import (
	"context"

	"github.com/mark3labs/mcp-go/server"
	"github.com/scriptkid23/boostmcp/internal/config"
	"github.com/scriptkid23/boostmcp/internal/inference"
	"github.com/scriptkid23/boostmcp/internal/pipeline/generator"
	"github.com/scriptkid23/boostmcp/internal/pipeline/narrower"
)

type Server struct {
	cfg       *config.Config
	provider  inference.InferenceProvider
	generator *generator.Generator
	narrower  *narrower.Narrower
}

func NewServer(cfg *config.Config) (*Server, error) {
	provider, err := inference.NewProvider(cfg)
	if err != nil {
		return nil, err
	}
	return &Server{
		cfg:       cfg,
		provider:  provider,
		generator: generator.New(provider, cfg),
		narrower:  narrower.New(provider, cfg),
	}, nil
}

func (s *Server) Run(ctx context.Context) error {
	srv := server.NewMCPServer("boostmcp", "1.0.0")
	s.registerTools(srv)
	return server.ServeStdio(srv)
}
```

```go
// cmd/boostmcp/main.go
package main

import (
	"context"
	"log"
	"os"

	"github.com/scriptkid23/boostmcp/internal/config"
	bmcp "github.com/scriptkid23/boostmcp/internal/mcp"
)

func main() {
	// CRITICAL: stdout is the MCP JSON-RPC channel. Nothing else may
	// write to it. Force the default logger to stderr so log.Printf
	// / log.Fatalf are safe.
	log.SetOutput(os.Stderr)

	cfg, err := config.Load("")
	if err != nil {
		log.Fatalf("config: %v", err)
	}
	srv, err := bmcp.NewServer(cfg)
	if err != nil {
		log.Fatalf("server: %v", err)
	}
	if err := srv.Run(context.Background()); err != nil {
		log.Fatalf("mcp: %v", err)
	}
	os.Exit(0)
}
```

> **stdout discipline:** never `fmt.Println` from any code path reachable in the stdio binary. Any stray byte on stdout breaks JSON-RPC framing and Cursor will silently disconnect.

- [ ] **Step 3: Verify build**

Run: `go build -o boostmcp.exe ./cmd/boostmcp`  
Expected: success

- [ ] **Step 4: Commit**

```bash
git add cmd/boostmcp/ internal/mcp/server.go go.mod go.sum
git commit -m "feat: add MCP server bootstrap and main entrypoint"
```

---

### Task 2: Register generate_candidates tool

**Files:**
- Create: `internal/mcp/tools.go`
- Create: `internal/mcp/tools_test.go`

- [ ] **Step 1: Write failing handler test**

```go
// internal/mcp/tools_test.go
package mcp_test

import (
	"context"
	"encoding/json"
	"testing"

	"github.com/scriptkid23/boostmcp/internal/config"
	bmcp "github.com/scriptkid23/boostmcp/internal/mcp"
)

func TestHandleGenerateCandidates(t *testing.T) {
	cfg := &config.Config{DefaultModel: "mock", MaxCandidates: 16, TimeoutMs: 5000, Provider: "ollama", OllamaURL: "http://127.0.0.1:1"}
	// Use mock by injecting via test helper — see handleGenerateCandidates exported for test
	// For v1: test JSON parsing separately
	var args map[string]any
	raw := `{"prompt":"hello","n_candidates":2}`
	if err := json.Unmarshal([]byte(raw), &args); err != nil {
		t.Fatal(err)
	}
	if args["prompt"] != "hello" {
		t.Fatal("parse failed")
	}
}
```

Better: export parse functions for testability.

Add to `tools.go`:

```go
type generateArgs struct {
	Prompt      string `json:"prompt"`
	Context     string `json:"context"`
	NCandidates int    `json:"n_candidates"`
	Model       string `json:"model"`
}

func parseGenerateArgs(raw map[string]any) (generateArgs, error) {
	b, _ := json.Marshal(raw)
	var a generateArgs
	if err := json.Unmarshal(b, &a); err != nil {
		return a, err
	}
	if a.Prompt == "" {
		return a, fmt.Errorf("prompt is required")
	}
	return a, nil
}
```

Test:

```go
func TestParseGenerateArgs(t *testing.T) {
	a, err := bmcp.ParseGenerateArgs(map[string]any{"prompt": "x", "n_candidates": float64(4)})
	if err != nil || a.Prompt != "x" || a.NCandidates != 4 {
		t.Fatalf("got %+v err=%v", a, err)
	}
	_, err = bmcp.ParseGenerateArgs(map[string]any{})
	if err == nil {
		t.Fatal("expected error")
	}
}
```

Export as `ParseGenerateArgs`.

- [ ] **Step 2: Implement tool registration**

```go
// internal/mcp/tools.go — add to registerTools method on Server
package mcp

import (
	"context"
	"encoding/json"
	"fmt"

	"github.com/mark3labs/mcp-go/mcp"
	mcpserver "github.com/mark3labs/mcp-go/server"
	"github.com/scriptkid23/boostmcp/internal/pipeline/generator"
)

func (s *Server) registerTools(srv *mcpserver.MCPServer) {
	srv.AddTool(mcp.NewTool("generate_candidates",
		mcp.WithDescription("Generate N diverse code candidates from local SLM"),
		mcp.WithString("prompt", mcp.Required(), mcp.Description("Task description")),
		mcp.WithString("context", mcp.Description("Optional file content or diff context")),
		mcp.WithNumber("n_candidates", mcp.Description("Number of candidates (1-16, default 4)")),
		mcp.WithString("model", mcp.Description("Optional model override")),
	), s.handleGenerateCandidates)

	// Full schema for narrow_candidates is wired in Task 3 (below) — the
	// description-only registration here is a placeholder so the tool name
	// exists for early stdio smoke tests. Task 3 replaces this entry.
	srv.AddTool(mcp.NewTool("narrow_candidates",
		mcp.WithDescription("Narrow candidates using rubric, return top-k scored (schema added in Task 3)"),
	), s.handleNarrowCandidates)
}

func ParseGenerateArgs(raw map[string]any) (generateArgs, error) {
	b, err := json.Marshal(raw)
	if err != nil {
		return generateArgs{}, err
	}
	var a generateArgs
	if err := json.Unmarshal(b, &a); err != nil {
		return generateArgs{}, err
	}
	if a.Prompt == "" {
		return generateArgs{}, fmt.Errorf("prompt is required")
	}
	return a, nil
}

type generateArgs struct {
	Prompt      string `json:"prompt"`
	Context     string `json:"context"`
	NCandidates int    `json:"n_candidates"`
	Model       string `json:"model"`
}

func (s *Server) handleGenerateCandidates(ctx context.Context, req mcp.CallToolRequest) (*mcp.CallToolResult, error) {
	args, err := ParseGenerateArgs(req.Params.Arguments)
	if err != nil {
		return mcp.NewToolResultError(err.Error()), nil
	}
	result, err := s.generator.Generate(ctx, generator.Input{
		Prompt: args.Prompt, Context: args.Context, NCandidates: args.NCandidates, Model: args.Model,
	})
	if err != nil {
		return mcp.NewToolResultError(err.Error()), nil
	}
	out, _ := json.Marshal(result)
	return mcp.NewToolResultText(string(out)), nil
}
```

Note: mcp-go API may vary slightly — adjust `req.Params.Arguments` to match actual SDK at implementation time.

- [ ] **Step 3: Run tests — expect PASS**

Run: `go test ./internal/mcp/... -v`  
Expected: PASS

- [ ] **Step 4: Commit**

```bash
git add internal/mcp/
git commit -m "feat: add generate_candidates MCP tool"
```

---

### Task 3: Register narrow_candidates tool

**Files:**
- Modify: `internal/mcp/tools.go`
- Modify: `internal/mcp/tools_test.go`

- [ ] **Step 1: Write parse test**

```go
func TestParseNarrowArgs(t *testing.T) {
	raw := map[string]any{
		"top_k": float64(2),
		"rubric": map[string]any{
			"criteria": []any{
				map[string]any{"name": "correctness", "weight": 1.0, "description": "ok"},
			},
		},
		"candidates": []any{
			map[string]any{"id": "c1", "content": "code", "metadata": map[string]any{}},
		},
	}
	a, err := bmcp.ParseNarrowArgs(raw)
	if err != nil || a.TopK != 2 || len(a.Candidates) != 1 {
		t.Fatalf("got %+v err=%v", a, err)
	}
}
```

- [ ] **Step 2: Replace the placeholder narrow registration with a fully-schemaed tool**

In `registerTools`, replace the placeholder `srv.AddTool(mcp.NewTool("narrow_candidates", ...))` block with the full schema below. This matches the input schema in spec §6.2 so Cursor's MCP client can validate args:

```go
srv.AddTool(mcp.NewTool("narrow_candidates",
	mcp.WithDescription("Narrow candidates using rubric, return top-k scored"),
	mcp.WithObject("rubric",
		mcp.Required(),
		mcp.Description("Rubric with weighted criteria and optional hard constraints"),
		mcp.Properties(map[string]any{
			"criteria": map[string]any{
				"type":        "array",
				"description": "Weighted scoring criteria (weights must sum to 1.0)",
				"items": map[string]any{
					"type": "object",
					"properties": map[string]any{
						"name":        map[string]any{"type": "string"},
						"weight":      map[string]any{"type": "number", "minimum": 0, "maximum": 1},
						"description": map[string]any{"type": "string"},
					},
					"required": []string{"name", "weight", "description"},
				},
			},
			"hard_constraints": map[string]any{
				"type":        "array",
				"description": "Rule-based pre-filter strings",
				"items":       map[string]any{"type": "string"},
			},
		}),
	),
	mcp.WithArray("candidates",
		mcp.Required(),
		mcp.Description("Candidates returned from generate_candidates"),
		mcp.Items(map[string]any{
			"type": "object",
			"properties": map[string]any{
				"id":       map[string]any{"type": "string"},
				"content":  map[string]any{"type": "string"},
				"metadata": map[string]any{"type": "object"},
			},
			"required": []string{"id", "content"},
		}),
	),
	mcp.WithNumber("top_k",
		mcp.Description("Number of top candidates to return (default from config)"),
	),
), s.handleNarrowCandidates)
```

> **mcp-go version note:** the exact helper names (`mcp.Properties`, `mcp.Items`) and option signatures depend on the version pinned in Task 0. If your version uses a different shape (e.g. `mcp.WithObjectProperties`, or accepting a `*jsonschema.Schema`), adapt to that — the *intent* is the JSON Schema fragment above. If the chosen version cannot express the nested rubric schema at all, fall back to declaring `rubric` as `mcp.WithObject("rubric", mcp.Required())` and document the runtime contract in the tool description.

- [ ] **Step 3: Implement narrow handler**

```go
type narrowArgs struct {
	Rubric     candidate.Rubric      `json:"rubric"`
	Candidates []candidate.Candidate `json:"candidates"`
	TopK       int                   `json:"top_k"`
}

func ParseNarrowArgs(raw map[string]any) (narrowArgs, error) {
	b, _ := json.Marshal(raw)
	var a narrowArgs
	if err := json.Unmarshal(b, &a); err != nil {
		return a, err
	}
	if err := a.Rubric.Validate(); err != nil {
		return a, fmt.Errorf("invalid rubric: %w", err)
	}
	if len(a.Candidates) == 0 {
		return a, fmt.Errorf("candidates is required")
	}
	return a, nil
}

func (s *Server) handleNarrowCandidates(ctx context.Context, req mcp.CallToolRequest) (*mcp.CallToolResult, error) {
	args, err := ParseNarrowArgs(req.Params.Arguments)
	if err != nil {
		return mcp.NewToolResultError(err.Error()), nil
	}
	result, err := s.narrower.Narrow(ctx, narrower.Input{
		Rubric: args.Rubric, Candidates: args.Candidates, TopK: args.TopK,
	})
	if err != nil {
		return mcp.NewToolResultError(err.Error()), nil
	}
	out, _ := json.Marshal(result)
	return mcp.NewToolResultText(string(out)), nil
}
```

- [ ] **Step 3: Run tests + build — expect PASS**

Run: `go test ./internal/mcp/... -v && go build ./cmd/boostmcp`  
Expected: PASS

- [ ] **Step 4: Commit**

```bash
git add internal/mcp/
git commit -m "feat: add narrow_candidates MCP tool"
```

---

### Task 4: Health check on startup (optional warning)

**Files:**
- Modify: `cmd/boostmcp/main.go`

- [ ] **Step 1: Log warning if Ollama unreachable (non-fatal)**

```go
// in main(), after NewServer:
if err := srv.ProviderHealth(context.Background()); err != nil {
	log.Printf("warning: %v", err)
}
```

Add to `Server`:

```go
func (s *Server) ProviderHealth(ctx context.Context) error {
	return s.provider.HealthCheck(ctx)
}
```

- [ ] **Step 2: Build and manual smoke**

Run: `go build -o boostmcp.exe ./cmd/boostmcp && ./boostmcp.exe`  
Expected: process starts, waits on stdin (Ctrl+C to exit); warning if Ollama down

- [ ] **Step 3: Commit**

```bash
git add cmd/boostmcp/main.go internal/mcp/server.go
git commit -m "feat: warn on startup when Ollama unreachable"
```

---

## v1 Limitations (documented, not blockers)

- **No tool-call deadline.** Inference timeout (`BOOSTMCP_TIMEOUT_MS`) is per-call. With N=16 candidates serialized worst-case, wall-clock can exceed common MCP client timeouts (~60s). v1 relies on Generator's parallelism + default N=4 to stay under that. Phase 2: add a per-tool overall deadline.
- **No graceful SIGINT.** `main` doesn't catch signals — Ctrl+C drops in-flight tool calls. Acceptable for solo-dev v1; Phase 2: cancel a root context on SIGINT/SIGTERM and let `ServeStdio` drain.
- **No request logging.** Errors go to stderr via `log`, but tool-call traces don't. Phase 2 ties into the observability roadmap (spec §11).

---

## Plan 05 Done When

- [ ] `boostmcp` binary builds and runs as stdio MCP server
- [ ] Both tools registered with schemas matching spec §6 (including the full `narrow_candidates` rubric/candidates/top_k schema)
- [ ] Invalid input returns MCP tool errors (not panics)
- [ ] Startup warns if Ollama unreachable
- [ ] No code path writes to stdout outside the MCP framing layer

**Next:** [Plan 06 — E2E & Docs](./2026-05-26-plan-06-e2e-docs.md)
