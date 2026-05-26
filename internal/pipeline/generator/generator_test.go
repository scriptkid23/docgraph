package generator_test

import (
	"context"
	"testing"

	"github.com/scriptkid23/boostmcp/internal/config"
	"github.com/scriptkid23/boostmcp/internal/inference/mock"
	"github.com/scriptkid23/boostmcp/internal/pipeline/generator"
)

func TestGenerateOneCandidate(t *testing.T) {
	prov := mock.New([]string{"code-v1"})
	cfg := &config.Config{DefaultModel: "mock-model", MaxCandidates: 16, TimeoutMs: 5000}
	g := generator.New(prov, cfg)

	result, err := g.Generate(context.Background(), generator.Input{
		Prompt:      "write hello world",
		NCandidates: 1,
	})
	if err != nil {
		t.Fatal(err)
	}
	if len(result.Candidates) != 1 {
		t.Fatalf("got %d candidates", len(result.Candidates))
	}
	c := result.Candidates[0]
	if c.Content != "code-v1" {
		t.Fatalf("content: %q", c.Content)
	}
	if c.ID == "" {
		t.Fatal("id required")
	}
	if c.Metadata.Index != 0 {
		t.Fatalf("index: %d", c.Metadata.Index)
	}
}

func TestGenerateMultipleCandidates(t *testing.T) {
	prov := mock.New([]string{"a", "b", "c", "d"})
	cfg := &config.Config{DefaultModel: "mock-model", MaxCandidates: 16, TimeoutMs: 5000}
	g := generator.New(prov, cfg)

	result, err := g.Generate(context.Background(), generator.Input{
		Prompt: "task", NCandidates: 4,
	})
	if err != nil {
		t.Fatal(err)
	}
	if len(result.Candidates) != 4 {
		t.Fatalf("got %d", len(result.Candidates))
	}
	if result.GenerationStats.Requested != 4 || result.GenerationStats.Received != 4 {
		t.Fatalf("stats: %+v", result.GenerationStats)
	}
	ids := map[string]bool{}
	for _, c := range result.Candidates {
		if ids[c.ID] {
			t.Fatalf("duplicate id %s", c.ID)
		}
		ids[c.ID] = true
	}
}

func TestGeneratePartialResults(t *testing.T) {
	prov := mock.NewFlaky([]string{"ok"}, 2)
	cfg := &config.Config{DefaultModel: "m", MaxCandidates: 16, TimeoutMs: 5000}
	g := generator.New(prov, cfg)

	result, err := g.Generate(context.Background(), generator.Input{Prompt: "x", NCandidates: 4})
	if err != nil {
		t.Fatal(err)
	}
	if result.GenerationStats.Received >= 4 {
		t.Fatal("expected some failures")
	}
	if result.GenerationStats.Received == 0 {
		t.Fatal("expected at least one candidate")
	}
}
