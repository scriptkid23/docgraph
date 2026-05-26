package mock

import (
	"context"
	"fmt"
	"sync/atomic"

	"github.com/scriptkid23/boostmcp/internal/inference"
)

type Provider struct {
	responses []string
	counter   atomic.Uint64
}

func New(responses []string) *Provider {
	return &Provider{responses: responses}
}

func (p *Provider) Generate(_ context.Context, req inference.GenerateRequest) (*inference.GenerateResponse, error) {
	if len(p.responses) == 0 {
		return nil, fmt.Errorf("mock: no responses configured")
	}
	i := p.counter.Add(1) - 1
	text := p.responses[int(i)%len(p.responses)]
	return &inference.GenerateResponse{
		Text:       text,
		TokenCount: len(text),
		LatencyMs:  1,
		Model:      req.Model,
	}, nil
}

func (p *Provider) ListModels(_ context.Context) ([]inference.ModelInfo, error) {
	return []inference.ModelInfo{{Name: "mock-model"}}, nil
}

func (p *Provider) HealthCheck(_ context.Context) error { return nil }

func (p *Provider) Name() string { return "mock" }
