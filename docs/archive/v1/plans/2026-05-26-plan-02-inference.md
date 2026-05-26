# Plan 02 — Inference Layer

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement pluggable `InferenceProvider` interface with Ollama HTTP adapter and test mock.

**Architecture:** Interface in `internal/inference`. Ollama calls `/api/generate` and `/api/tags`. Factory selects provider from config. Mock used by Plans 03–04 unit tests.

**Tech Stack:** Go stdlib `net/http`, `context`

**Depends on:** [Plan 01](./2026-05-26-plan-01-foundation.md)  
**Blocks:** Plans 03, 04

**Spec refs:** §7 InferenceProvider, §8 Error Handling

---

## File Structure

```
internal/inference/
├── provider.go           # InferenceProvider interface + request/response types
├── factory.go            # NewProvider(cfg)
├── mock/
│   └── mock.go           # MockProvider for tests
└── ollama/
    ├── ollama.go         # OllamaProvider
    └── ollama_test.go    # httptest-based tests
```

---

### Task 1: InferenceProvider interface

**Files:**
- Create: `internal/inference/provider.go`

- [ ] **Step 1: Define interface and types**

```go
// internal/inference/provider.go
package inference

import "context"

type GenerateRequest struct {
	Prompt      string
	Model       string
	Temperature float64
}

type GenerateResponse struct {
	Text       string
	TokenCount int
	LatencyMs  int64
	Model      string
}

type ModelInfo struct {
	Name string
}

type InferenceProvider interface {
	Generate(ctx context.Context, req GenerateRequest) (*GenerateResponse, error)
	ListModels(ctx context.Context) ([]ModelInfo, error)
	HealthCheck(ctx context.Context) error
	Name() string
}
```

- [ ] **Step 2: Verify compiles**

Run: `go build ./internal/inference/...`  
Expected: success (no tests yet)

- [ ] **Step 3: Commit**

```bash
git add internal/inference/provider.go
git commit -m "feat: add InferenceProvider interface"
```

---

### Task 2: Mock provider

**Files:**
- Create: `internal/inference/mock/mock.go`
- Create: `internal/inference/mock/mock_test.go`

- [ ] **Step 1: Write failing test**

```go
// internal/inference/mock/mock_test.go
package mock_test

import (
	"context"
	"testing"

	"github.com/scriptkid23/boostmcp/internal/inference"
	"github.com/scriptkid23/boostmcp/internal/inference/mock"
)

func TestMockGenerate(t *testing.T) {
	m := mock.New([]string{"answer-a", "answer-b"})
	resp, err := m.Generate(context.Background(), inference.GenerateRequest{Prompt: "hi"})
	if err != nil {
		t.Fatal(err)
	}
	if resp.Text != "answer-a" {
		t.Fatalf("got %q, want answer-a", resp.Text)
	}
	resp2, _ := m.Generate(context.Background(), inference.GenerateRequest{Prompt: "hi"})
	if resp2.Text != "answer-b" {
		t.Fatalf("got %q, want answer-b", resp2.Text)
	}
}
```

- [ ] **Step 2: Run test — expect FAIL**

Run: `go test ./internal/inference/mock/... -v`  
Expected: FAIL

- [ ] **Step 3: Implement mock**

```go
// internal/inference/mock/mock.go
package mock

import (
	"context"
	"fmt"
	"sync/atomic"

	"github.com/scriptkid23/boostmcp/internal/inference"
)

type Provider struct {
	responses []string
	counter   atomic.Uint64
}

func New(responses []string) *Provider {
	return &Provider{responses: responses}
}

func (p *Provider) Generate(_ context.Context, req inference.GenerateRequest) (*inference.GenerateResponse, error) {
	if len(p.responses) == 0 {
		return nil, fmt.Errorf("mock: no responses configured")
	}
	i := p.counter.Add(1) - 1
	text := p.responses[int(i)%len(p.responses)]
	return &inference.GenerateResponse{
		Text:       text,
		TokenCount: len(text),
		LatencyMs:  1,
		Model:      req.Model,
	}, nil
}

func (p *Provider) ListModels(_ context.Context) ([]inference.ModelInfo, error) {
	return []inference.ModelInfo{{Name: "mock-model"}}, nil
}

func (p *Provider) HealthCheck(_ context.Context) error { return nil }

func (p *Provider) Name() string { return "mock" }
```

- [ ] **Step 4: Run test — expect PASS**

Run: `go test ./internal/inference/mock/... -v`  
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add internal/inference/mock/
git commit -m "feat: add mock InferenceProvider"
```

---

### Task 3: Ollama provider — Generate

**Files:**
- Create: `internal/inference/ollama/ollama.go`
- Create: `internal/inference/ollama/ollama_test.go`

- [ ] **Step 1: Write failing test with httptest**

```go
// internal/inference/ollama/ollama_test.go
package ollama_test

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/scriptkid23/boostmcp/internal/inference"
	"github.com/scriptkid23/boostmcp/internal/inference/ollama"
)

func TestGenerateSuccess(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/api/generate" {
			t.Fatalf("path: %s", r.URL.Path)
		}
		json.NewEncoder(w).Encode(map[string]any{
			"response":          "func main() {}",
			"eval_count":        10,
			"total_duration":    1_200_000_000,
		})
	}))
	defer srv.Close()

	p := ollama.New(srv.URL, "codellama:7b", 30000)
	resp, err := p.Generate(context.Background(), inference.GenerateRequest{
		Prompt:      "write main",
		Temperature: 0.7,
	})
	if err != nil {
		t.Fatal(err)
	}
	if resp.Text != "func main() {}" {
		t.Fatalf("got %q", resp.Text)
	}
	if resp.TokenCount != 10 {
		t.Fatalf("tokens: %d", resp.TokenCount)
	}
}
```

- [ ] **Step 2: Run test — expect FAIL**

Run: `go test ./internal/inference/ollama/... -v -run TestGenerateSuccess`  
Expected: FAIL

- [ ] **Step 3: Implement Ollama Generate**

```go
// internal/inference/ollama/ollama.go
package ollama

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"time"

	"github.com/scriptkid23/boostmcp/internal/inference"
)

type Provider struct {
	baseURL   string
	model     string
	timeoutMs int
	client    *http.Client
}

func New(baseURL, defaultModel string, timeoutMs int) *Provider {
	return &Provider{
		baseURL:   baseURL,
		model:     defaultModel,
		timeoutMs: timeoutMs,
		client:    &http.Client{Timeout: time.Duration(timeoutMs) * time.Millisecond},
	}
}

type generateBody struct {
	Model       string  `json:"model"`
	Prompt      string  `json:"prompt"`
	Stream      bool    `json:"stream"`
	Temperature float64 `json:"temperature,omitempty"`
}

type generateResp struct {
	Response      string `json:"response"`
	EvalCount     int    `json:"eval_count"`
	TotalDuration int64  `json:"total_duration"`
}

func (p *Provider) Generate(ctx context.Context, req inference.GenerateRequest) (*inference.GenerateResponse, error) {
	model := req.Model
	if model == "" {
		model = p.model
	}
	body, _ := json.Marshal(generateBody{
		Model: model, Prompt: req.Prompt, Stream: false, Temperature: req.Temperature,
	})
	httpReq, err := http.NewRequestWithContext(ctx, http.MethodPost, p.baseURL+"/api/generate", bytes.NewReader(body))
	if err != nil {
		return nil, err
	}
	httpReq.Header.Set("Content-Type", "application/json")

	start := time.Now()
	httpResp, err := p.client.Do(httpReq)
	if err != nil {
		return nil, fmt.Errorf("Ollama unreachable at %s. Start Ollama first: %w", p.baseURL, err)
	}
	defer httpResp.Body.Close()
	raw, _ := io.ReadAll(httpResp.Body)
	if httpResp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("ollama generate failed (%d): %s", httpResp.StatusCode, string(raw))
	}
	var gr generateResp
	if err := json.Unmarshal(raw, &gr); err != nil {
		return nil, fmt.Errorf("parse ollama response: %w", err)
	}
	latency := time.Since(start).Milliseconds()
	if gr.TotalDuration > 0 {
		latency = gr.TotalDuration / 1_000_000
	}
	return &inference.GenerateResponse{
		Text: gr.Response, TokenCount: gr.EvalCount, LatencyMs: latency, Model: model,
	}, nil
}

func (p *Provider) Name() string { return "ollama" }
```

- [ ] **Step 4: Run test — expect PASS**

Run: `go test ./internal/inference/ollama/... -v -run TestGenerateSuccess`  
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add internal/inference/ollama/
git commit -m "feat: add Ollama Generate implementation"
```

---

### Task 4: Ollama HealthCheck + ListModels

**Files:**
- Modify: `internal/inference/ollama/ollama.go`
- Modify: `internal/inference/ollama/ollama_test.go`

- [ ] **Step 1: Write failing tests**

```go
import "strings"

func TestHealthCheckSuccess(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		json.NewEncoder(w).Encode(map[string]any{"models": []any{}})
	}))
	defer srv.Close()
	p := ollama.New(srv.URL, "m", 5000)
	if err := p.HealthCheck(context.Background()); err != nil {
		t.Fatal(err)
	}
}

func TestHealthCheckUnreachable(t *testing.T) {
	// Port 1 is reserved and almost certainly closed — connection refused.
	p := ollama.New("http://127.0.0.1:1", "m", 1000)
	err := p.HealthCheck(context.Background())
	if err == nil {
		t.Fatal("expected error")
	}
	if !strings.Contains(err.Error(), "Ollama unreachable") {
		t.Fatalf("error must be actionable per spec §8; got: %v", err)
	}
}
```

- [ ] **Step 2: Implement HealthCheck + ListModels**

```go
type tagsResp struct {
	Models []struct {
		Name string `json:"name"`
	} `json:"models"`
}

func (p *Provider) HealthCheck(ctx context.Context) error {
	_, err := p.ListModels(ctx)
	if err != nil {
		return fmt.Errorf("Ollama unreachable at %s. Start Ollama first: %w", p.baseURL, err)
	}
	return nil
}

func (p *Provider) ListModels(ctx context.Context) ([]inference.ModelInfo, error) {
	httpReq, err := http.NewRequestWithContext(ctx, http.MethodGet, p.baseURL+"/api/tags", nil)
	if err != nil {
		return nil, err
	}
	httpResp, err := p.client.Do(httpReq)
	if err != nil {
		return nil, err
	}
	defer httpResp.Body.Close()
	raw, _ := io.ReadAll(httpResp.Body)
	if httpResp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("ollama tags failed (%d)", httpResp.StatusCode)
	}
	var tr tagsResp
	if err := json.Unmarshal(raw, &tr); err != nil {
		return nil, err
	}
	out := make([]inference.ModelInfo, len(tr.Models))
	for i, m := range tr.Models {
		out[i] = inference.ModelInfo{Name: m.Name}
	}
	return out, nil
}
```

- [ ] **Step 3: Run all ollama tests — expect PASS**

Run: `go test ./internal/inference/... -v`  
Expected: PASS

- [ ] **Step 4: Commit**

```bash
git add internal/inference/ollama/
git commit -m "feat: add Ollama HealthCheck and ListModels"
```

---

### Task 5: Provider factory

**Files:**
- Create: `internal/inference/factory.go`
- Create: `internal/inference/factory_test.go`

- [ ] **Step 1: Write failing test**

```go
// internal/inference/factory_test.go
package inference_test

import (
	"testing"

	"github.com/scriptkid23/boostmcp/internal/config"
	"github.com/scriptkid23/boostmcp/internal/inference"
)

func TestNewProviderOllama(t *testing.T) {
	cfg := &config.Config{Provider: "ollama", OllamaURL: "http://localhost:11434", DefaultModel: "m", TimeoutMs: 1000}
	p, err := inference.NewProvider(cfg)
	if err != nil {
		t.Fatal(err)
	}
	if p.Name() != "ollama" {
		t.Fatalf("got %q", p.Name())
	}
}

func TestNewProviderUnknown(t *testing.T) {
	cfg := &config.Config{Provider: "unknown"}
	_, err := inference.NewProvider(cfg)
	if err == nil {
		t.Fatal("expected error")
	}
}
```

- [ ] **Step 2: Implement factory**

```go
// internal/inference/factory.go
package inference

import (
	"fmt"

	"github.com/scriptkid23/boostmcp/internal/config"
	"github.com/scriptkid23/boostmcp/internal/inference/ollama"
)

func NewProvider(cfg *config.Config) (InferenceProvider, error) {
	switch cfg.Provider {
	case "ollama":
		return ollama.New(cfg.OllamaURL, cfg.DefaultModel, cfg.TimeoutMs), nil
	default:
		return nil, fmt.Errorf("unknown inference provider %q", cfg.Provider)
	}
}
```

- [ ] **Step 3: Run tests — expect PASS**

Run: `go test ./internal/inference/... -v`  
Expected: PASS

- [ ] **Step 4: Commit**

```bash
git add internal/inference/factory.go internal/inference/factory_test.go
git commit -m "feat: add inference provider factory"
```

---

## Plan 02 Done When

- [ ] `InferenceProvider` interface + Ollama + mock all tested
- [ ] Unreachable Ollama returns actionable error message
- [ ] Factory wires config → Ollama provider

**Next:** [Plan 03 — Generator](./2026-05-26-plan-03-generator.md) and [Plan 04 — Narrower](./2026-05-26-plan-04-narrower.md) (parallel OK)
