package narrower_test

import (
	"context"
	"strings"
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

func TestNarrowSurfacesOmittedCandidate(t *testing.T) {
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
