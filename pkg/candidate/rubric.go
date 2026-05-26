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
