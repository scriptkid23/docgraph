package config_test

import (
	"testing"

	"github.com/scriptkid23/boostmcp/internal/config"
)

func TestLoadFromEnv(t *testing.T) {
	t.Setenv("BOOSTMCP_MODEL", "qwen2.5-coder:7b")
	t.Setenv("BOOSTMCP_OLLAMA_URL", "http://127.0.0.1:11434")
	t.Setenv("BOOSTMCP_TIMEOUT_MS", "60000")

	cfg, err := config.Load("")
	if err != nil {
		t.Fatal(err)
	}
	if cfg.DefaultModel != "qwen2.5-coder:7b" {
		t.Fatalf("model: got %q", cfg.DefaultModel)
	}
	if cfg.OllamaURL != "http://127.0.0.1:11434" {
		t.Fatalf("url: got %q", cfg.OllamaURL)
	}
	if cfg.TimeoutMs != 60000 {
		t.Fatalf("timeout: got %d", cfg.TimeoutMs)
	}
}

func TestLoadDefaults(t *testing.T) {
	for _, k := range []string{
		"BOOSTMCP_PROVIDER",
		"BOOSTMCP_MODEL",
		"BOOSTMCP_OLLAMA_URL",
		"BOOSTMCP_TIMEOUT_MS",
		"BOOSTMCP_MAX_CANDIDATES",
		"BOOSTMCP_DEFAULT_TOP_K",
	} {
		t.Setenv(k, "")
	}
	cfg, err := config.Load("")
	if err != nil {
		t.Fatal(err)
	}
	if cfg.DefaultModel != "codellama:7b" {
		t.Fatalf("default model: got %q", cfg.DefaultModel)
	}
	if cfg.MaxCandidates != 16 {
		t.Fatalf("max candidates: got %d", cfg.MaxCandidates)
	}
}
