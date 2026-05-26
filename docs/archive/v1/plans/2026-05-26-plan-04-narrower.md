# Plan 04 — Narrower Pipeline

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement rubric-based narrowing — hard constraint pre-filter, SLM scoring, weighted top-k selection.

**Architecture:** `internal/pipeline/narrower` validates rubric, applies rule-based hard constraints, then one SLM call to score survivors. Returns `NarrowResult` with scores and dropped reasons.

**Tech Stack:** Go, inference provider for scoring prompt

**Depends on:** [Plan 01](./2026-05-26-plan-01-foundation.md), [Plan 02](./2026-05-26-plan-02-inference.md)  
**Blocks:** [Plan 05](./2026-05-26-plan-05-mcp-server.md)

**Spec refs:** §5.2 Round 2, §5.3 Rubric, §6.2 narrow_candidates, §8 zero candidates

---

## File Structure

```
internal/pipeline/narrower/
├── narrower.go
├── constraints.go
├── scorer.go
└── narrower_test.go
```

---

### Task 1: Rubric validation at entry

**Files:**
- Create: `internal/pipeline/narrower/narrower.go`
- Create: `internal/pipeline/narrower/narrower_test.go`

- [ ] **Step 1: Write failing test**

```go
// internal/pipeline/narrower/narrower_test.go
package narrower_test

import (
	"context"
	"testing"

	"github.com/scriptkid23/boostmcp/internal/config"
	"github.com/scriptkid23/boostmcp/internal/inference/mock"
	"github.com/scriptkid23/boostmcp/internal/pipeline/narrower"
	"github.com/scriptkid23/boostmcp/pkg/candidate"
)

func TestInvalidRubricRejected(t *testing.T) {
	n := narrower.New(mock.New([]string{}), &config.Config{DefaultModel: "m", TimeoutMs: 5000})
	_, err := n.Narrow(context.Background(), narrower.Input{
		Rubric:     candidate.Rubric{Criteria: []candidate.Criterion{}},
		Candidates: []candidate.Candidate{{ID: "c1", Content: "x"}},
		TopK:       2,
	})
	if err == nil {
		t.Fatal("expected validation error")
	}
}
```

- [ ] **Step 2: Run test — expect FAIL**

Run: `go test ./internal/pipeline/narrower/... -v -run TestInvalidRubricRejected`  
Expected: FAIL

- [ ] **Step 3: Implement narrower skeleton**

```go
// internal/pipeline/narrower/narrower.go
package narrower

import (
	"context"
	"fmt"

	"github.com/scriptkid23/boostmcp/internal/config"
	"github.com/scriptkid23/boostmcp/internal/inference"
	"github.com/scriptkid23/boostmcp/pkg/candidate"
)

type Input struct {
	Rubric     candidate.Rubric
	Candidates []candidate.Candidate
	TopK       int
}

type Narrower struct {
	provider inference.InferenceProvider
	cfg      *config.Config
}

func New(provider inference.InferenceProvider, cfg *config.Config) *Narrower {
	return &Narrower{provider: provider, cfg: cfg}
}

func (n *Narrower) Narrow(ctx context.Context, in Input) (*candidate.NarrowResult, error) {
	if err := in.Rubric.Validate(); err != nil {
		return nil, fmt.Errorf("invalid rubric: %w", err)
	}
	topK := in.TopK
	if topK <= 0 {
		topK = n.cfg.DefaultTopK
	}
	return &candidate.NarrowResult{}, nil // filled in later tasks
}
```

- [ ] **Step 4: Run test — expect PASS**

Run: `go test ./internal/pipeline/narrower/... -v -run TestInvalidRubricRejected`  
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add internal/pipeline/narrower/
git commit -m "feat: add narrower with rubric validation"
```

---

### Task 2: Hard constraint pre-filter

**Files:**
- Create: `internal/pipeline/narrower/constraints.go`
- Modify: `internal/pipeline/narrower/narrower_test.go`

- [ ] **Step 1: Write failing test**

```go
func TestHardConstraintDropsCandidate(t *testing.T) {
	candidates := []candidate.Candidate{
		{ID: "c1", Content: "import numpy\nfunc main() {}"},
		{ID: "c2", Content: "func main() {}"},
	}
	survivors, dropped := narrower.ApplyHardConstraints(candidates, []string{"no new dependencies"})
	if len(survivors) != 1 || survivors[0].ID != "c2" {
		t.Fatalf("survivors: %+v", survivors)
	}
	if len(dropped) != 1 || dropped[0].CandidateID != "c1" {
		t.Fatalf("dropped: %+v", dropped)
	}
}
```

- [ ] **Step 2: Implement constraint rules**

```go
// internal/pipeline/narrower/constraints.go
package narrower

import (
	"strings"

	"github.com/scriptkid23/boostmcp/pkg/candidate"
)

// ApplyHardConstraints is a v1 best-effort pre-filter. Recognized
// constraint strings are matched by simple heuristics; unknown
// constraints are deferred to SLM scoring (where the model can use
// the constraint description as guidance).
//
// TODO(phase-2): replace this with a proper constraint DSL or
// per-language analyzer (gofmt/build, ruff, etc.). The heuristics
// below intentionally only fire on a small, well-known surface to
// avoid false positives — anything we can't recognize is passed
// through, not silently rejected.
func ApplyHardConstraints(candidates []candidate.Candidate, constraints []string) ([]candidate.Candidate, []candidate.Dropped) {
	if len(constraints) == 0 {
		return candidates, nil
	}
	var survivors []candidate.Candidate
	var dropped []candidate.Dropped
	for _, c := range candidates {
		reason := checkConstraintsV1Heuristic(c.Content, constraints)
		if reason != "" {
			dropped = append(dropped, candidate.Dropped{CandidateID: c.ID, Reason: reason})
		} else {
			survivors = append(survivors, c)
		}
	}
	return survivors, dropped
}

// checkConstraintsV1Heuristic returns a drop reason if a candidate
// trips a recognized v1 rule. Unknown constraints return "" (pass through).
func checkConstraintsV1Heuristic(content string, constraints []string) string {
	for _, hc := range constraints {
		switch strings.ToLower(strings.TrimSpace(hc)) {
		case "no new dependencies":
			if looksLikeNewImport(content) {
				return "failed hard constraint: no new dependencies"
			}
		// NOTE: "must compile" is intentionally NOT handled in v1.
		// Grepping for "syntax error" or "undefined:" inside source
		// code is meaningless — those strings live in compiler output,
		// not in the generated source. Real compile-checking belongs
		// in Phase 2 (run go/gofmt/tsc against survivors). Until then,
		// "must compile" is delegated to the SLM scorer.
		default:
			// Pass through — unrecognized constraints are evaluated
			// by the SLM during scoring.
		}
	}
	return ""
}

// looksLikeNewImport is a v1 placeholder: it only catches a tiny
// hand-picked list of third-party imports across Go and Python.
// A real implementation would diff imports against the existing
// project's go.mod / requirements.txt / package.json.
func looksLikeNewImport(content string) bool {
	thirdParty := []string{
		"import numpy", "import pandas", "import requests",
		"\"github.com/", "require (",
	}
	for _, tp := range thirdParty {
		if strings.Contains(content, tp) {
			return true
		}
	}
	return false
}
```

- [ ] **Step 3: Run test — expect PASS**

Run: `go test ./internal/pipeline/narrower/... -v -run TestHardConstraintDropsCandidate`  
Expected: PASS

- [ ] **Step 4: Commit**

```bash
git add internal/pipeline/narrower/constraints.go internal/pipeline/narrower/narrower_test.go
git commit -m "feat: add hard constraint pre-filter"
```

---

### Task 3: SLM scoring

**Files:**
- Create: `internal/pipeline/narrower/scorer.go`
- Create: `internal/pipeline/narrower/scorer_test.go`

- [ ] **Step 1: Write failing test**

```go
// internal/pipeline/narrower/scorer_test.go
package narrower_test

import (
	"context"
	"encoding/json"
	"testing"

	"github.com/scriptkid23/boostmcp/internal/inference/mock"
	"github.com/scriptkid23/boostmcp/internal/pipeline/narrower"
	"github.com/scriptkid23/boostmcp/pkg/candidate"
)

func TestScoreCandidates(t *testing.T) {
	scoreJSON := `[{"candidate_id":"c1","breakdown":{"correctness":0.9,"minimal_diff":0.8}}]`
	prov := mock.New([]string{scoreJSON})
	rubric := candidate.Rubric{
		Criteria: []candidate.Criterion{
			{Name: "correctness", Weight: 0.5, Description: "correct"},
			{Name: "minimal_diff", Weight: 0.5, Description: "small"},
		},
	}
	candidates := []candidate.Candidate{{ID: "c1", Content: "code"}}

	scores, err := narrower.ScoreCandidates(context.Background(), prov, rubric, candidates)
	if err != nil {
		t.Fatal(err)
	}
	if len(scores) != 1 {
		t.Fatalf("got %d scores", len(scores))
	}
	if scores[0].TotalScore < 0.8 {
		t.Fatalf("score too low: %f", scores[0].TotalScore)
	}
}
```

- [ ] **Step 2: Implement scorer**

```go
// internal/pipeline/narrower/scorer.go
package narrower

import (
	"context"
	"encoding/json"
	"fmt"
	"strings"

	"github.com/scriptkid23/boostmcp/internal/inference"
	"github.com/scriptkid23/boostmcp/pkg/candidate"
)

type rawScore struct {
	CandidateID string             `json:"candidate_id"`
	Breakdown   map[string]float64 `json:"breakdown"`
}

func ScoreCandidates(ctx context.Context, provider inference.InferenceProvider, rubric candidate.Rubric, candidates []candidate.Candidate) ([]candidate.Score, error) {
	prompt := buildScoringPrompt(rubric, candidates)
	resp, err := provider.Generate(ctx, inference.GenerateRequest{Prompt: prompt, Temperature: 0.1})
	if err != nil {
		return nil, fmt.Errorf("scoring failed: %w", err)
	}
	raw, err := parseScoreJSON(resp.Text)
	if err != nil {
		return nil, err
	}
	return weightedScores(raw, rubric), nil
}

func buildScoringPrompt(rubric candidate.Rubric, candidates []candidate.Candidate) string {
	var b strings.Builder
	b.WriteString("Score each candidate 0.0-1.0 per criterion. Return JSON array only:\n")
	b.WriteString(`[{"candidate_id":"...","breakdown":{"criterion_name":0.0}}]` + "\n\nCriteria:\n")
	for _, c := range rubric.Criteria {
		fmt.Fprintf(&b, "- %s (weight %.2f): %s\n", c.Name, c.Weight, c.Description)
	}
	b.WriteString("\nCandidates:\n")
	for _, c := range candidates {
		fmt.Fprintf(&b, "--- %s ---\n%s\n", c.ID, c.Content)
	}
	return b.String()
}

func parseScoreJSON(text string) ([]rawScore, error) {
	text = strings.TrimSpace(text)
	start := strings.Index(text, "[")
	end := strings.LastIndex(text, "]")
	if start < 0 || end < 0 {
		return nil, fmt.Errorf("scorer response missing JSON array: %s", text)
	}
	var raw []rawScore
	if err := json.Unmarshal([]byte(text[start:end+1]), &raw); err != nil {
		return nil, fmt.Errorf("parse scorer JSON: %w", err)
	}
	return raw, nil
}

func weightedScores(raw []rawScore, rubric candidate.Rubric) []candidate.Score {
	weights := map[string]float64{}
	for _, c := range rubric.Criteria {
		weights[c.Name] = c.Weight
	}
	out := make([]candidate.Score, len(raw))
	for i, r := range raw {
		var total float64
		for name, score := range r.Breakdown {
			total += score * weights[name]
		}
		out[i] = candidate.Score{CandidateID: r.CandidateID, TotalScore: total, Breakdown: r.Breakdown}
	}
	return out
}
```

- [ ] **Step 3: Run test — expect PASS**

Run: `go test ./internal/pipeline/narrower/... -v -run TestScoreCandidates`  
Expected: PASS

- [ ] **Step 4: Commit**

```bash
git add internal/pipeline/narrower/scorer.go internal/pipeline/narrower/scorer_test.go
git commit -m "feat: add SLM-based rubric scoring"
```

---

### Task 4: Top-k selection

**Files:**
- Modify: `internal/pipeline/narrower/narrower.go`
- Modify: `internal/pipeline/narrower/narrower_test.go`

- [ ] **Step 1: Write failing integration test**

```go
func TestNarrowReturnsTopK(t *testing.T) {
	scoreJSON := `[
		{"candidate_id":"c1","breakdown":{"correctness":0.9,"minimal_diff":0.9}},
		{"candidate_id":"c2","breakdown":{"correctness":0.5,"minimal_diff":0.5}}
	]`
	prov := mock.New([]string{scoreJSON})
	cfg := &config.Config{DefaultModel: "m", DefaultTopK: 2, TimeoutMs: 5000}
	n := narrower.New(prov, cfg)

	rubric := candidate.Rubric{
		Criteria: []candidate.Criterion{
			{Name: "correctness", Weight: 0.5, Description: "correct"},
			{Name: "minimal_diff", Weight: 0.5, Description: "small"},
		},
	}
	candidates := []candidate.Candidate{
		{ID: "c1", Content: "best code"},
		{ID: "c2", Content: "worse code"},
	}

	result, err := n.Narrow(context.Background(), narrower.Input{Rubric: rubric, Candidates: candidates, TopK: 1})
	if err != nil {
		t.Fatal(err)
	}
	if len(result.Narrowed) != 1 || result.Narrowed[0].ID != "c1" {
		t.Fatalf("narrowed: %+v", result.Narrowed)
	}
}
```

- [ ] **Step 2: Wire Narrow() fully**

```go
func (n *Narrower) Narrow(ctx context.Context, in Input) (*candidate.NarrowResult, error) {
	if err := in.Rubric.Validate(); err != nil {
		return nil, fmt.Errorf("invalid rubric: %w", err)
	}
	topK := in.TopK
	if topK <= 0 {
		topK = n.cfg.DefaultTopK
	}

	survivors, dropped := ApplyHardConstraints(in.Candidates, in.Rubric.HardConstraints)
	if len(survivors) == 0 {
		return &candidate.NarrowResult{Narrowed: []candidate.Candidate{}, Scores: []candidate.Score{}, Dropped: dropped}, nil
	}

	scores, err := ScoreCandidates(ctx, n.provider, in.Rubric, survivors)
	if err != nil {
		return nil, err
	}

	// Surface any survivor the scorer omitted (chatty models do this).
	// Record them as Dropped rather than silently losing them.
	scored := make(map[string]bool, len(scores))
	for _, s := range scores {
		scored[s.CandidateID] = true
	}
	for _, c := range survivors {
		if !scored[c.ID] {
			dropped = append(dropped, candidate.Dropped{
				CandidateID: c.ID,
				Reason:      "scorer omitted candidate from response",
			})
		}
	}

	// Sort scores descending, pick topK.
	sort.Slice(scores, func(i, j int) bool { return scores[i].TotalScore > scores[j].TotalScore })
	if topK > len(scores) {
		topK = len(scores)
	}
	topScores := scores[:topK]

	idToCandidate := map[string]candidate.Candidate{}
	for _, c := range survivors {
		idToCandidate[c.ID] = c
	}
	narrowed := make([]candidate.Candidate, 0, topK)
	for _, s := range topScores {
		if c, ok := idToCandidate[s.CandidateID]; ok {
			narrowed = append(narrowed, c)
		}
	}
	return &candidate.NarrowResult{Narrowed: narrowed, Scores: topScores, Dropped: dropped}, nil
}
```

Add `import "sort"`.

- [ ] **Step 3: Run all narrower tests — expect PASS**

Run: `go test ./internal/pipeline/narrower/... -v`  
Expected: PASS

- [ ] **Step 4: Commit**

```bash
git add internal/pipeline/narrower/
git commit -m "feat: complete narrower with top-k selection"
```

---

### Task 5a: Scorer omits a candidate

**Files:**
- Modify: `internal/pipeline/narrower/narrower_test.go`

- [ ] **Step 1: Write test — scorer returns only 1 of 2 candidates**

```go
func TestNarrowSurfacesOmittedCandidate(t *testing.T) {
	// Scorer returns c1 only; c2 should land in Dropped, not vanish.
	scoreJSON := `[{"candidate_id":"c1","breakdown":{"correctness":0.9}}]`
	prov := mock.New([]string{scoreJSON})
	cfg := &config.Config{DefaultModel: "m", DefaultTopK: 2, TimeoutMs: 5000}
	n := narrower.New(prov, cfg)

	rubric := candidate.Rubric{
		Criteria: []candidate.Criterion{{Name: "correctness", Weight: 1.0, Description: "ok"}},
	}
	candidates := []candidate.Candidate{
		{ID: "c1", Content: "a"},
		{ID: "c2", Content: "b"},
	}
	result, err := n.Narrow(context.Background(), narrower.Input{Rubric: rubric, Candidates: candidates, TopK: 2})
	if err != nil {
		t.Fatal(err)
	}
	if len(result.Narrowed) != 1 || result.Narrowed[0].ID != "c1" {
		t.Fatalf("narrowed: %+v", result.Narrowed)
	}
	var foundOmitted bool
	for _, d := range result.Dropped {
		if d.CandidateID == "c2" && strings.Contains(d.Reason, "scorer omitted") {
			foundOmitted = true
		}
	}
	if !foundOmitted {
		t.Fatalf("expected c2 in dropped with omit reason; got: %+v", result.Dropped)
	}
}
```

Add `import "strings"` to the test file.

- [ ] **Step 2: Run — expect PASS**

Run: `go test ./internal/pipeline/narrower/... -v -run TestNarrowSurfacesOmittedCandidate`  
Expected: PASS

- [ ] **Step 3: Commit**

```bash
git add internal/pipeline/narrower/narrower_test.go
git commit -m "test: scorer-omitted candidates surface in Dropped"
```

---

### Task 5: Zero survivors after hard constraints

**Files:**
- Modify: `internal/pipeline/narrower/narrower_test.go`

- [ ] **Step 1: Write test**

```go
func TestNarrowEmptyWhenAllDropped(t *testing.T) {
	prov := mock.New([]string{})
	n := narrower.New(prov, &config.Config{DefaultTopK: 2, TimeoutMs: 5000})
	rubric := candidate.Rubric{
		Criteria:        []candidate.Criterion{{Name: "correctness", Weight: 1.0, Description: "x"}},
		HardConstraints: []string{"no new dependencies"},
	}
	candidates := []candidate.Candidate{{ID: "c1", Content: "import numpy"}}
	result, err := n.Narrow(context.Background(), narrower.Input{Rubric: rubric, Candidates: candidates, TopK: 2})
	if err != nil {
		t.Fatal(err)
	}
	if len(result.Narrowed) != 0 {
		t.Fatalf("expected empty narrowed, got %d", len(result.Narrowed))
	}
	if len(result.Dropped) != 1 {
		t.Fatalf("dropped: %+v", result.Dropped)
	}
}
```

- [ ] **Step 2: Run test — expect PASS**

Run: `go test ./internal/pipeline/narrower/... -v -run TestNarrowEmptyWhenAllDropped`  
Expected: PASS

- [ ] **Step 3: Commit**

```bash
git add internal/pipeline/narrower/narrower_test.go
git commit -m "test: verify empty narrowed set when all candidates dropped"
```

---

## v1 Limitations (documented, not blockers)

- **Single-prompt scoring.** All survivors are scored in one SLM call. For long candidates and N=16 this can approach context-window limits. Phase 2 should chunk into batches of ~4 candidates per scoring call.
- **Hard-constraint heuristics are intentionally narrow.** Only `"no new dependencies"` triggers a rule-based check; everything else is passed through to SLM scoring. See comments in `constraints.go`.
- **No score retry.** If the scorer returns unparseable JSON, the whole call fails. Phase 2 should retry once with a stricter prompt.

---

## Plan 04 Done When

- [ ] Invalid rubric rejected without SLM call
- [ ] Hard constraints filter before scoring
- [ ] SLM scores survivors, top-k returned with breakdown
- [ ] Empty narrowed set returned when all dropped (not an error)
- [ ] Candidates omitted by the scorer appear in `Dropped` (no silent loss)

**Next:** [Plan 05 — MCP Server](./2026-05-26-plan-05-mcp-server.md)
