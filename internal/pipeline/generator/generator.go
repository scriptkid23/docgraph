package generator

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

	type indexedResult struct {
		index int
		cand  candidate.Candidate
	}

	results := make([]indexedResult, 0, n)
	var mu sync.Mutex
	eg, egCtx := errgroup.WithContext(ctx)

	for i := 0; i < n; i++ {
		i, temp := i, temps[i]
		eg.Go(func() error {
			resp, err := g.provider.Generate(egCtx, inference.GenerateRequest{
				Prompt: prompt, Model: model, Temperature: temp,
			})
			if err != nil {
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
	_ = eg.Wait()

	sort.Slice(results, func(i, j int) bool { return results[i].index < results[j].index })

	candidates := make([]candidate.Candidate, 0, len(results))
	for _, r := range results {
		candidates = append(candidates, r.cand)
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
