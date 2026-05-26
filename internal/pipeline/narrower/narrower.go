package narrower

import (
	"context"
	"fmt"
	"sort"

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

	survivors, dropped := ApplyHardConstraints(in.Candidates, in.Rubric.HardConstraints)
	if len(survivors) == 0 {
		return &candidate.NarrowResult{Narrowed: []candidate.Candidate{}, Scores: []candidate.Score{}, Dropped: dropped}, nil
	}

	scores, err := ScoreCandidates(ctx, n.provider, in.Rubric, survivors)
	if err != nil {
		return nil, err
	}

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
