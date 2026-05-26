package mock

import (
	"context"
	"fmt"
	"sync/atomic"

	"github.com/scriptkid23/boostmcp/internal/inference"
)

type Flaky struct {
	inner   *Provider
	failMod uint64
	calls   atomic.Uint64
}

func NewFlaky(responses []string, failEveryN uint64) *Flaky {
	return &Flaky{inner: New(responses), failMod: failEveryN}
}

func (f *Flaky) Generate(ctx context.Context, req inference.GenerateRequest) (*inference.GenerateResponse, error) {
	n := f.calls.Add(1)
	if f.failMod > 0 && n%f.failMod == 0 {
		return nil, fmt.Errorf("flaky failure")
	}
	return f.inner.Generate(ctx, req)
}

func (f *Flaky) ListModels(ctx context.Context) ([]inference.ModelInfo, error) {
	return f.inner.ListModels(ctx)
}

func (f *Flaky) HealthCheck(ctx context.Context) error { return f.inner.HealthCheck(ctx) }

func (f *Flaky) Name() string { return "flaky-mock" }
