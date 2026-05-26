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
