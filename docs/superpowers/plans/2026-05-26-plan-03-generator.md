# Plan 03 — Generator Pipeline

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement candidate generation — call SLM N times with varied temperature, attach metadata, return `GenerateResult`.

**Architecture:** `internal/pipeline/generator` accepts `InferenceProvider` + config. Generates unique candidate IDs. Parallel generation with `errgroup` for speed. Partial results on per-call failure per spec §8.

**Tech Stack:** Go, `golang.org/x/sync/errgroup`

**Depends on:** [Plan 01](./2026-05-26-plan-01-foundation.md), [Plan 02](./2026-05-26-plan-02-inference.md)  
**Blocks:** [Plan 05](./2026-05-26-plan-05-mcp-server.md)

**Spec refs:** §5.1 Round 1, §6.1 generate_candidates, §8 partial results

---

## File Structure

```
internal/pipeline/generator/
├── generator.go
└── generator_test.go
```

---

### Task 1: Generator struct + single candidate

**Files:**
- Create: `internal/pipeline/generator/generator.go`
- Create: `internal/pipeline/generator/generator_test.go`

- [ ] **Step 1: Add errgroup dependency**

```bash
go get golang.org/x/sync/errgroup
```

- [ ] **Step 2: Write failing test**

```go
// internal/pipeline/generator/generator_test.go
package generator_test

import (
	"context"
	"testing"

	"github.com/scriptkid23/boostmcp/internal/config"
	"github.com/scriptkid23/boostmcp/internal/inference/mock"
	"github.com/scriptkid23/boostmcp/internal/pipeline/generator"
)

func TestGenerateOneCandidate(t *testing.T) {
	prov := mock.New([]string{"code-v1"})
	cfg := &config.Config{DefaultModel: "mock-model", MaxCandidates: 16, TimeoutMs: 5000}
	g := generator.New(prov, cfg)

	result, err := g.Generate(context.Background(), generator.Input{
		Prompt:      "write hello world",
		NCandidates: 1,
	})
	if err != nil {
		t.Fatal(err)
	}
	if len(result.Candidates) != 1 {
		t.Fatalf("got %d candidates", len(result.Candidates))
	}
	c := result.Candidates[0]
	if c.Content != "code-v1" {
		t.Fatalf("content: %q", c.Content)
	}
	if c.ID == "" {
		t.Fatal("id required")
	}
	if c.Metadata.Index != 0 {
		t.Fatalf("index: %d", c.Metadata.Index)
	}
}
```

- [ ] **Step 3: Run test — expect FAIL**

Run: `go test ./internal/pipeline/generator/... -v -run TestGenerateOneCandidate`  
Expected: FAIL

- [ ] **Step 4: Implement generator skeleton**

```go
// internal/pipeline/generator/generator.go
package generator

import (
	"context"
	"fmt"
	"time"

	"github.com/scriptkid23/boostmcp/internal/config"
	"github.com/scriptkid23/boostmcp/internal/inference"
	"github.com/scriptkid23/boostmcp/pkg/candidate"
)

type Input struct {
	Prompt      string
	Context     string
	NCandidates int
	Model       string
}

type Generator struct {
	provider inference.InferenceProvider
	cfg      *config.Config
}

func New(provider inference.InferenceProvider, cfg *config.Config) *Generator {
	return &Generator{provider: provider, cfg: cfg}
}

func (g *Generator) Generate(ctx context.Context, in Input) (*candidate.GenerateResult, error) {
	n := in.NCandidates
	if n <= 0 {
		n = 4
	}
	if n > g.cfg.MaxCandidates {
		n = g.cfg.MaxCandidates
	}
	model := in.Model
	if model == "" {
		model = g.cfg.DefaultModel
	}
	prompt := in.Prompt
	if in.Context != "" {
		prompt = fmt.Sprintf("Context:\n%s\n\nTask:\n%s", in.Context, in.Prompt)
	}

	start := time.Now()
	temps := temperatures(n)
	candidates := make([]candidate.Candidate, 0, n)

	for i := 0; i < n; i++ {
		resp, err := g.provider.Generate(ctx, inference.GenerateRequest{
			Prompt: prompt, Model: model, Temperature: temps[i],
		})
		if err != nil {
			continue // partial results per spec §8
		}
		candidates = append(candidates, candidate.Candidate{
			ID:      fmt.Sprintf("cand-%03d", i+1),
			Content: resp.Text,
			Metadata: candidate.Metadata{
				Index: i, Model: resp.Model, LatencyMs: resp.LatencyMs, TokenCount: resp.TokenCount,
			},
		})
	}
	if len(candidates) == 0 {
		return nil, fmt.Errorf("failed to generate any candidates")
	}
	return &candidate.GenerateResult{
		Candidates: candidates,
		GenerationStats: candidate.GenerationStats{
			TotalMs: time.Since(start).Milliseconds(), Model: model, Requested: n, Received: len(candidates),
		},
	}, nil
}

// temperatures spreads N samples across [0.3, 0.9] so candidates have
// distinct decoding temperatures even for N up to MaxCandidates (16).
// For N=1 returns [0.6] (midpoint).
func temperatures(n int) []float64 {
	if n <= 0 {
		return nil
	}
	if n == 1 {
		return []float64{0.6}
	}
	const lo, hi = 0.3, 0.9
	step := (hi - lo) / float64(n-1)
	out := make([]float64, n)
	for i := 0; i < n; i++ {
		out[i] = lo + step*float64(i)
	}
	return out
}
```

- [ ] **Step 5: Run test — expect PASS**

Run: `go test ./internal/pipeline/generator/... -v -run TestGenerateOneCandidate`  
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add internal/pipeline/generator/
git commit -m "feat: add generator with single-candidate support"
```

---

### Task 2: Multiple candidates + varied temperature

**Files:**
- Modify: `internal/pipeline/generator/generator_test.go`

- [ ] **Step 1: Write failing test**

```go
func TestGenerateMultipleCandidates(t *testing.T) {
	prov := mock.New([]string{"a", "b", "c", "d"})
	cfg := &config.Config{DefaultModel: "mock-model", MaxCandidates: 16, TimeoutMs: 5000}
	g := generator.New(prov, cfg)

	result, err := g.Generate(context.Background(), generator.Input{
		Prompt: "task", NCandidates: 4,
	})
	if err != nil {
		t.Fatal(err)
	}
	if len(result.Candidates) != 4 {
		t.Fatalf("got %d", len(result.Candidates))
	}
	if result.GenerationStats.Requested != 4 || result.GenerationStats.Received != 4 {
		t.Fatalf("stats: %+v", result.GenerationStats)
	}
	ids := map[string]bool{}
	for _, c := range result.Candidates {
		if ids[c.ID] {
			t.Fatalf("duplicate id %s", c.ID)
		}
		ids[c.ID] = true
	}
}
```

- [ ] **Step 2: Run test — expect PASS** (implementation already supports N)

Run: `go test ./internal/pipeline/generator/... -v -run TestGenerateMultipleCandidates`  
Expected: PASS

- [ ] **Step 3: Commit**

```bash
git add internal/pipeline/generator/generator_test.go
git commit -m "test: verify multi-candidate generation"
```

---

### Task 3: Partial results on failure

**Files:**
- Modify: `internal/pipeline/generator/generator_test.go`

- [ ] **Step 1: Write failing test with flaky mock**

Create `internal/inference/mock/flaky.go`:

```go
// internal/inference/mock/flaky.go
package mock

import (
	"context"
	"fmt"
	"sync/atomic"

	"github.com/scriptkid23/boostmcp/internal/inference"
)

type Flaky struct {
	inner   *Provider
	failMod uint64
	calls   atomic.Uint64
}

func NewFlaky(responses []string, failEveryN uint64) *Flaky {
	return &Flaky{inner: New(responses), failMod: failEveryN}
}

func (f *Flaky) Generate(ctx context.Context, req inference.GenerateRequest) (*inference.GenerateResponse, error) {
	n := f.calls.Add(1)
	if f.failMod > 0 && n%f.failMod == 0 {
		return nil, fmt.Errorf("flaky failure")
	}
	return f.inner.Generate(ctx, req)
}

func (f *Flaky) ListModels(ctx context.Context) ([]inference.ModelInfo, error) {
	return f.inner.ListModels(ctx)
}
func (f *Flaky) HealthCheck(ctx context.Context) error { return f.inner.HealthCheck(ctx) }
func (f *Flaky) Name() string                          { return "flaky-mock" }
```

Test:

```go
func TestGeneratePartialResults(t *testing.T) {
	prov := mock.NewFlaky([]string{"ok"}, 2) // every 2nd call fails
	cfg := &config.Config{DefaultModel: "m", MaxCandidates: 16, TimeoutMs: 5000}
	g := generator.New(prov, cfg)

	result, err := g.Generate(context.Background(), generator.Input{Prompt: "x", NCandidates: 4})
	if err != nil {
		t.Fatal(err)
	}
	if result.GenerationStats.Received >= 4 {
		t.Fatal("expected some failures")
	}
	if result.GenerationStats.Received == 0 {
		t.Fatal("expected at least one candidate")
	}
}
```

- [ ] **Step 2: Run test — expect PASS**

Run: `go test ./internal/pipeline/generator/... -v -run TestGeneratePartialResults`  
Expected: PASS

- [ ] **Step 3: Commit**

```bash
git add internal/inference/mock/flaky.go internal/pipeline/generator/
git commit -m "feat: support partial candidate generation on failures"
```

---

### Task 4: Parallel generation

**Files:**
- Modify: `internal/pipeline/generator/generator.go`

- [ ] **Step 1: Refactor to errgroup parallel calls**

Replace the sequential loop in `Generate()` with the parallel version below. Returning `nil` from each goroutine is deliberate — per spec §8 we want partial results, so a single failure must not cancel the group.

Update the import block at the top of the file:

```go
import (
	"context"
	"fmt"
	"sort"
	"sync"
	"time"

	"golang.org/x/sync/errgroup"

	"github.com/scriptkid23/boostmcp/internal/config"
	"github.com/scriptkid23/boostmcp/internal/inference"
	"github.com/scriptkid23/boostmcp/pkg/candidate"
)
```

Replace the candidate-building section of `Generate` (everything between `temps := temperatures(n)` and the `if len(candidates) == 0` check) with:

```go
type indexedResult struct {
	index int
	cand  candidate.Candidate
}

results := make([]indexedResult, 0, n)
var mu sync.Mutex
eg, egCtx := errgroup.WithContext(ctx)

for i := 0; i < n; i++ {
	i, temp := i, temps[i] // capture per-iteration
	eg.Go(func() error {
		resp, err := g.provider.Generate(egCtx, inference.GenerateRequest{
			Prompt: prompt, Model: model, Temperature: temp,
		})
		if err != nil {
			// Partial results per spec §8: drop this candidate but keep the rest.
			// Returning nil intentionally prevents errgroup from cancelling siblings.
			return nil
		}
		mu.Lock()
		results = append(results, indexedResult{
			index: i,
			cand: candidate.Candidate{
				ID:      fmt.Sprintf("cand-%03d", i+1),
				Content: resp.Text,
				Metadata: candidate.Metadata{
					Index:      i,
					Model:      resp.Model,
					LatencyMs:  resp.LatencyMs,
					TokenCount: resp.TokenCount,
				},
			},
		})
		mu.Unlock()
		return nil
	})
}
_ = eg.Wait() // never errors — goroutines return nil on failure

sort.Slice(results, func(i, j int) bool { return results[i].index < results[j].index })

candidates := make([]candidate.Candidate, 0, len(results))
for _, r := range results {
	candidates = append(candidates, r.cand)
}
```

- [ ] **Step 2: Run all generator tests — expect PASS**

Run: `go test ./internal/pipeline/generator/... -v`  
Expected: PASS

- [ ] **Step 3: Commit**

```bash
git add internal/pipeline/generator/
git commit -m "perf: parallel candidate generation with errgroup"
```

---

## Plan 03 Done When

- [ ] Generates N candidates with unique IDs and metadata
- [ ] Temperature varies across calls
- [ ] Partial results when some calls fail
- [ ] Context prepended to prompt when provided

**Next:** [Plan 05 — MCP Server](./2026-05-26-plan-05-mcp-server.md) (after Plan 04 or in parallel)
