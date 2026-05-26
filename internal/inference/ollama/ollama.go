package ollama

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"time"

	"github.com/scriptkid23/boostmcp/internal/inference"
)

type Provider struct {
	baseURL   string
	model     string
	timeoutMs int
	client    *http.Client
}

func New(baseURL, defaultModel string, timeoutMs int) *Provider {
	return &Provider{
		baseURL:   baseURL,
		model:     defaultModel,
		timeoutMs: timeoutMs,
		client:    &http.Client{Timeout: time.Duration(timeoutMs) * time.Millisecond},
	}
}

// SetDefaultModel updates the model used when GenerateRequest.Model is empty.
func (p *Provider) SetDefaultModel(model string) {
	p.model = model
}

// DefaultModel returns the configured default model name.
func (p *Provider) DefaultModel() string {
	return p.model
}

type generateBody struct {
	Model       string  `json:"model"`
	Prompt      string  `json:"prompt"`
	Stream      bool    `json:"stream"`
	Temperature float64 `json:"temperature,omitempty"`
}

type generateResp struct {
	Response      string `json:"response"`
	EvalCount     int    `json:"eval_count"`
	TotalDuration int64  `json:"total_duration"`
}

type tagsResp struct {
	Models []struct {
		Name string `json:"name"`
	} `json:"models"`
}

func (p *Provider) Generate(ctx context.Context, req inference.GenerateRequest) (*inference.GenerateResponse, error) {
	model := req.Model
	if model == "" {
		model = p.model
	}
	body, _ := json.Marshal(generateBody{
		Model: model, Prompt: req.Prompt, Stream: false, Temperature: req.Temperature,
	})
	httpReq, err := http.NewRequestWithContext(ctx, http.MethodPost, p.baseURL+"/api/generate", bytes.NewReader(body))
	if err != nil {
		return nil, err
	}
	httpReq.Header.Set("Content-Type", "application/json")

	start := time.Now()
	httpResp, err := p.client.Do(httpReq)
	if err != nil {
		return nil, fmt.Errorf("Ollama unreachable at %s. Start Ollama first: %w", p.baseURL, err)
	}
	defer httpResp.Body.Close()
	raw, _ := io.ReadAll(httpResp.Body)
	if httpResp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("ollama generate failed (%d): %s", httpResp.StatusCode, string(raw))
	}
	var gr generateResp
	if err := json.Unmarshal(raw, &gr); err != nil {
		return nil, fmt.Errorf("parse ollama response: %w", err)
	}
	latency := time.Since(start).Milliseconds()
	if gr.TotalDuration > 0 {
		latency = gr.TotalDuration / 1_000_000
	}
	return &inference.GenerateResponse{
		Text: gr.Response, TokenCount: gr.EvalCount, LatencyMs: latency, Model: model,
	}, nil
}

func (p *Provider) HealthCheck(ctx context.Context) error {
	_, err := p.ListModels(ctx)
	if err != nil {
		return fmt.Errorf("Ollama unreachable at %s. Start Ollama first: %w", p.baseURL, err)
	}
	return nil
}

func (p *Provider) ListModels(ctx context.Context) ([]inference.ModelInfo, error) {
	httpReq, err := http.NewRequestWithContext(ctx, http.MethodGet, p.baseURL+"/api/tags", nil)
	if err != nil {
		return nil, err
	}
	httpResp, err := p.client.Do(httpReq)
	if err != nil {
		return nil, err
	}
	defer httpResp.Body.Close()
	raw, _ := io.ReadAll(httpResp.Body)
	if httpResp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("ollama tags failed (%d)", httpResp.StatusCode)
	}
	var tr tagsResp
	if err := json.Unmarshal(raw, &tr); err != nil {
		return nil, err
	}
	out := make([]inference.ModelInfo, len(tr.Models))
	for i, m := range tr.Models {
		out[i] = inference.ModelInfo{Name: m.Name}
	}
	return out, nil
}

func (p *Provider) Name() string { return "ollama" }
