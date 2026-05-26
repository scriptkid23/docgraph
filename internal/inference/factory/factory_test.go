package factory_test

import (
	"testing"

	"github.com/scriptkid23/boostmcp/internal/config"
	"github.com/scriptkid23/boostmcp/internal/inference/factory"
)

func TestNewProviderOllama(t *testing.T) {
	cfg := &config.Config{Provider: "ollama", OllamaURL: "http://localhost:11434", DefaultModel: "m", TimeoutMs: 1000}
	p, err := factory.NewProvider(cfg)
	if err != nil {
		t.Fatal(err)
	}
	if p.Name() != "ollama" {
		t.Fatalf("got %q", p.Name())
	}
}

func TestNewProviderUnknown(t *testing.T) {
	cfg := &config.Config{Provider: "unknown"}
	_, err := factory.NewProvider(cfg)
	if err == nil {
		t.Fatal("expected error")
	}
}
