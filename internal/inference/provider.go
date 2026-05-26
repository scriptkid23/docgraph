package inference

import "context"

type GenerateRequest struct {
	Prompt      string
	Model       string
	Temperature float64
}

type GenerateResponse struct {
	Text       string
	TokenCount int
	LatencyMs  int64
	Model      string
}

type ModelInfo struct {
	Name string
}

type InferenceProvider interface {
	Generate(ctx context.Context, req GenerateRequest) (*GenerateResponse, error)
	ListModels(ctx context.Context) ([]ModelInfo, error)
	HealthCheck(ctx context.Context) error
	Name() string
}
