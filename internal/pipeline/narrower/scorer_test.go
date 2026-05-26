package narrower_test

import (
	"context"
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
