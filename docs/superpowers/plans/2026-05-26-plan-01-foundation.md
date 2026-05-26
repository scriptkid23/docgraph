# Plan 01 — Foundation

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Bootstrap Go module, domain types, and configuration loading.

**Architecture:** Single Go module `github.com/scriptkid23/boostmcp`. Domain types live in `pkg/candidate` (public). Config in `internal/config` reads env vars with sensible defaults per spec §9.

**Tech Stack:** Go 1.22+, `gopkg.in/yaml.v3`

**Depends on:** nothing  
**Blocks:** Plans 02–06

**Spec refs:** §4 Components, §9 Configuration

---

## File Structure

```
boostmcp/
├── go.mod
├── go.sum
├── pkg/candidate/
│   ├── candidate.go      # Candidate, Metadata, GenerationStats
│   ├── rubric.go         # Rubric, Criterion, Score, Dropped
│   └── candidate_test.go
└── internal/config/
    ├── config.go         # Config struct + Load()
    └── config_test.go
```

---

### Task 1: Initialize Go module

**Files:**
- Create: `go.mod`

- [ ] **Step 1: Create module**

From the repo root:

```bash
go mod init github.com/scriptkid23/boostmcp
```

> **Note:** confirm the module path matches the intended GitHub remote before running. If the repo lives elsewhere (e.g. `github.com/1hoodlabs/boostmcp`), substitute accordingly — and update every import path in subsequent plans.

- [ ] **Step 2: Verify**

Run: `go mod verify`  
Expected: no errors (empty module)

- [ ] **Step 3: Commit**

```bash
git add go.mod
git commit -m "chore: initialize Go module"
```

---

### Task 2: Domain types — Candidate

**Files:**
- Create: `pkg/candidate/candidate.go`
- Create: `pkg/candidate/candidate_test.go`

- [ ] **Step 1: Write failing test**

```go
// pkg/candidate/candidate_test.go
package candidate_test

import (
	"encoding/json"
	"strings"
	"testing"

	"github.com/scriptkid23/boostmcp/pkg/candidate"
)

func TestCandidateJSONRoundTrip(t *testing.T) {
	in := candidate.Candidate{
		ID:      "cand-001",
		Content: "func main() {}",
		Metadata: candidate.Metadata{
			Index:      0,
			Model:      "codellama:7b",
			LatencyMs:  1200,
			TokenCount: 256,
		},
	}
	raw, err := json.Marshal(in)
	if err != nil {
		t.Fatalf("marshal: %v", err)
	}
	var out candidate.Candidate
	if err := json.Unmarshal(raw, &out); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}
	if out.ID != in.ID || out.Content != in.Content {
		t.Fatalf("round-trip mismatch: %+v", out)
	}
	if out.Metadata != in.Metadata {
		t.Fatalf("metadata round-trip mismatch: %+v", out.Metadata)
	}
	// Verify JSON field names match spec §6.
	if !strings.Contains(string(raw), `"latency_ms"`) || !strings.Contains(string(raw), `"token_count"`) {
		t.Fatalf("snake_case JSON tags missing: %s", raw)
	}
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `go test ./pkg/candidate/... -v -run TestCandidateJSONRoundTrip`  
Expected: FAIL — package or types not defined

- [ ] **Step 3: Implement types**

```go
// pkg/candidate/candidate.go
package candidate

type Candidate struct {
	ID       string   `json:"id"`
	Content  string   `json:"content"`
	Metadata Metadata `json:"metadata"`
}

type Metadata struct {
	Index      int    `json:"index"`
	Model      string `json:"model"`
	LatencyMs  int64  `json:"latency_ms"`
	TokenCount int    `json:"token_count"`
	// DiffStats is reserved for Phase 2 — generator does not populate it in v1.
	// See spec §5.1 (mentions diff_stats) and Phase 2 roadmap §11.
	DiffStats *DiffStats `json:"diff_stats,omitempty"`
}

// DiffStats is a forward-compatible placeholder so JSON shape can evolve
// without breaking the v1 contract. v1 always emits null/omitted.
type DiffStats struct {
	Added   int `json:"added"`
	Removed int `json:"removed"`
}

type GenerationStats struct {
	TotalMs   int64  `json:"total_ms"`
	Model     string `json:"model"`
	Requested int    `json:"requested"`
	Received  int    `json:"received"`
}

type GenerateResult struct {
	Candidates      []Candidate     `json:"candidates"`
	GenerationStats GenerationStats `json:"generation_stats"`
}
```

- [ ] **Step 4: Run test — expect PASS**

Run: `go test ./pkg/candidate/... -v`  
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add pkg/candidate/
git commit -m "feat: add candidate domain types"
```

---

### Task 3: Domain types — Rubric

**Files:**
- Create: `pkg/candidate/rubric.go`
- Modify: `pkg/candidate/candidate_test.go`

- [ ] **Step 1: Write failing test**

```go
func TestRubricValidate(t *testing.T) {
	r := candidate.Rubric{
		Criteria: []candidate.Criterion{
			{Name: "correctness", Weight: 0.4, Description: "solves problem"},
			{Name: "minimal_diff", Weight: 0.6, Description: "small change"},
		},
		HardConstraints: []string{"must compile"},
	}
	if err := r.Validate(); err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
}

func TestRubricValidateRejectsEmptyCriteria(t *testing.T) {
	r := candidate.Rubric{Criteria: []candidate.Criterion{}}
	if err := r.Validate(); err == nil {
		t.Fatal("expected validation error for empty criteria")
	}
}
```

- [ ] **Step 2: Run test — expect FAIL**

Run: `go test ./pkg/candidate/... -v -run TestRubric`  
Expected: FAIL

- [ ] **Step 3: Implement rubric types + validation**

```go
// pkg/candidate/rubric.go
package candidate

import "fmt"

type Rubric struct {
	Criteria        []Criterion `json:"criteria"`
	HardConstraints []string    `json:"hard_constraints,omitempty"`
}

type Criterion struct {
	Name        string  `json:"name"`
	Weight      float64 `json:"weight"`
	Description string  `json:"description"`
}

type Score struct {
	CandidateID string             `json:"candidate_id"`
	TotalScore  float64            `json:"total_score"`
	Breakdown   map[string]float64 `json:"breakdown"`
}

type Dropped struct {
	CandidateID string `json:"candidate_id"`
	Reason      string `json:"reason"`
}

type NarrowResult struct {
	Narrowed []Candidate `json:"narrowed"`
	Scores   []Score     `json:"scores"`
	Dropped  []Dropped   `json:"dropped"`
}

func (r Rubric) Validate() error {
	if len(r.Criteria) == 0 {
		return fmt.Errorf("rubric must have at least one criterion")
	}
	var sum float64
	for _, c := range r.Criteria {
		if c.Name == "" {
			return fmt.Errorf("criterion name is required")
		}
		if c.Weight < 0 || c.Weight > 1 {
			return fmt.Errorf("criterion %q weight must be 0-1", c.Name)
		}
		sum += c.Weight
	}
	if sum < 0.99 || sum > 1.01 {
		return fmt.Errorf("criterion weights must sum to 1.0, got %.2f", sum)
	}
	return nil
}
```

- [ ] **Step 4: Run tests — expect PASS**

Run: `go test ./pkg/candidate/... -v`  
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add pkg/candidate/
git commit -m "feat: add rubric types with validation"
```

---

### Task 4: Configuration loader

**Files:**
- Create: `internal/config/config.go`
- Create: `internal/config/config_test.go`

- [ ] **Step 1: Add yaml dependency**

```bash
go get gopkg.in/yaml.v3
```

- [ ] **Step 2: Write failing test**

```go
// internal/config/config_test.go
package config_test

import (
	"testing"

	"github.com/scriptkid23/boostmcp/internal/config"
)

func TestLoadFromEnv(t *testing.T) {
	t.Setenv("BOOSTMCP_MODEL", "qwen2.5-coder:7b")
	t.Setenv("BOOSTMCP_OLLAMA_URL", "http://127.0.0.1:11434")
	t.Setenv("BOOSTMCP_TIMEOUT_MS", "60000")

	cfg, err := config.Load("")
	if err != nil {
		t.Fatal(err)
	}
	if cfg.DefaultModel != "qwen2.5-coder:7b" {
		t.Fatalf("model: got %q", cfg.DefaultModel)
	}
	if cfg.OllamaURL != "http://127.0.0.1:11434" {
		t.Fatalf("url: got %q", cfg.OllamaURL)
	}
	if cfg.TimeoutMs != 60000 {
		t.Fatalf("timeout: got %d", cfg.TimeoutMs)
	}
}

func TestLoadDefaults(t *testing.T) {
	// Clear ALL BOOSTMCP_* vars that affect defaults — missing any here lets
	// the developer's shell env leak into the test.
	for _, k := range []string{
		"BOOSTMCP_PROVIDER",
		"BOOSTMCP_MODEL",
		"BOOSTMCP_OLLAMA_URL",
		"BOOSTMCP_TIMEOUT_MS",
		"BOOSTMCP_MAX_CANDIDATES",
		"BOOSTMCP_DEFAULT_TOP_K",
	} {
		t.Setenv(k, "") // Setenv auto-restores; empty == "treat as unset" in applyEnv.
	}
	cfg, err := config.Load("")
	if err != nil {
		t.Fatal(err)
	}
	if cfg.DefaultModel != "codellama:7b" {
		t.Fatalf("default model: got %q", cfg.DefaultModel)
	}
	if cfg.MaxCandidates != 16 {
		t.Fatalf("max candidates: got %d", cfg.MaxCandidates)
	}
}
```

- [ ] **Step 3: Run test — expect FAIL**

Run: `go test ./internal/config/... -v`  
Expected: FAIL

- [ ] **Step 4: Implement config**

```go
// internal/config/config.go
package config

import (
	"fmt"
	"os"
	"strconv"

	"gopkg.in/yaml.v3"
)

type Config struct {
	Provider     string `yaml:"provider"`
	DefaultModel string `yaml:"default_model"`
	OllamaURL    string `yaml:"ollama_url"`
	TimeoutMs    int    `yaml:"timeout_ms"`
	MaxCandidates int   `yaml:"max_candidates"`
	DefaultTopK  int    `yaml:"default_top_k"`
}

func Load(yamlPath string) (*Config, error) {
	cfg := &Config{
		Provider:      "ollama",
		DefaultModel:  "codellama:7b",
		OllamaURL:     "http://localhost:11434",
		TimeoutMs:     30000,
		MaxCandidates: 16,
		DefaultTopK:   2,
	}
	if yamlPath != "" {
		data, err := os.ReadFile(yamlPath)
		if err != nil {
			return nil, fmt.Errorf("read config: %w", err)
		}
		if err := yaml.Unmarshal(data, cfg); err != nil {
			return nil, fmt.Errorf("parse config: %w", err)
		}
	}
	applyEnv(cfg)
	return cfg, nil
}

func applyEnv(cfg *Config) {
	if v := os.Getenv("BOOSTMCP_MODEL"); v != "" {
		cfg.DefaultModel = v
	}
	if v := os.Getenv("BOOSTMCP_OLLAMA_URL"); v != "" {
		cfg.OllamaURL = v
	}
	if v := os.Getenv("BOOSTMCP_TIMEOUT_MS"); v != "" {
		if n, err := strconv.Atoi(v); err == nil {
			cfg.TimeoutMs = n
		}
	}
	if v := os.Getenv("BOOSTMCP_MAX_CANDIDATES"); v != "" {
		if n, err := strconv.Atoi(v); err == nil {
			cfg.MaxCandidates = n
		}
	}
	if v := os.Getenv("BOOSTMCP_DEFAULT_TOP_K"); v != "" {
		if n, err := strconv.Atoi(v); err == nil {
			cfg.DefaultTopK = n
		}
	}
}
```

- [ ] **Step 5: Run tests — expect PASS**

Run: `go test ./... -v`  
Expected: all PASS

- [ ] **Step 6: Commit**

```bash
git add internal/config/ go.mod go.sum
git commit -m "feat: add config loader with env and yaml support"
```

---

## Plan 01 Done When

- [ ] `go test ./...` passes
- [ ] Domain types match spec §6 schemas
- [ ] Config defaults match spec §9.1

**Next:** [Plan 02 — Inference Layer](./2026-05-26-plan-02-inference.md)
