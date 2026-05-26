package factory

import (
	"fmt"

	"github.com/scriptkid23/boostmcp/internal/config"
	"github.com/scriptkid23/boostmcp/internal/inference"
	"github.com/scriptkid23/boostmcp/internal/inference/ollama"
)

func NewProvider(cfg *config.Config) (inference.InferenceProvider, error) {
	switch cfg.Provider {
	case "ollama":
		return ollama.New(cfg.OllamaURL, cfg.DefaultModel, cfg.TimeoutMs), nil
	default:
		return nil, fmt.Errorf("unknown inference provider %q", cfg.Provider)
	}
}
