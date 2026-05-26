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
	if !strings.Contains(string(raw), `"latency_ms"`) || !strings.Contains(string(raw), `"token_count"`) {
		t.Fatalf("snake_case JSON tags missing: %s", raw)
	}
}

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
