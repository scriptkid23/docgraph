package config

import (
	"fmt"
	"os"
	"strconv"

	"gopkg.in/yaml.v3"
)

type Config struct {
	Provider      string `yaml:"provider"`
	DefaultModel  string `yaml:"default_model"`
	OllamaURL     string `yaml:"ollama_url"`
	TimeoutMs     int    `yaml:"timeout_ms"`
	MaxCandidates int    `yaml:"max_candidates"`
	DefaultTopK   int    `yaml:"default_top_k"`
}

func Load(yamlPath string) (*Config, error) {
	cfg := &Config{
		Provider:      "ollama",
		DefaultModel:  "codellama:7b",
		OllamaURL:     "http://localhost:11434",
		TimeoutMs:     30000,
		MaxCandidates: 16,
		DefaultTopK:   2,
	}
	if yamlPath != "" {
		data, err := os.ReadFile(yamlPath)
		if err != nil {
			return nil, fmt.Errorf("read config: %w", err)
		}
		if err := yaml.Unmarshal(data, cfg); err != nil {
			return nil, fmt.Errorf("parse config: %w", err)
		}
	}
	applyEnv(cfg)
	return cfg, nil
}

func applyEnv(cfg *Config) {
	if v := os.Getenv("BOOSTMCP_PROVIDER"); v != "" {
		cfg.Provider = v
	}
	if v := os.Getenv("BOOSTMCP_MODEL"); v != "" {
		cfg.DefaultModel = v
	}
	if v := os.Getenv("BOOSTMCP_OLLAMA_URL"); v != "" {
		cfg.OllamaURL = v
	}
	if v := os.Getenv("BOOSTMCP_TIMEOUT_MS"); v != "" {
		if n, err := strconv.Atoi(v); err == nil {
			cfg.TimeoutMs = n
		}
	}
	if v := os.Getenv("BOOSTMCP_MAX_CANDIDATES"); v != "" {
		if n, err := strconv.Atoi(v); err == nil {
			cfg.MaxCandidates = n
		}
	}
	if v := os.Getenv("BOOSTMCP_DEFAULT_TOP_K"); v != "" {
		if n, err := strconv.Atoi(v); err == nil {
			cfg.DefaultTopK = n
		}
	}
}
