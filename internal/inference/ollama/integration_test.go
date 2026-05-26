//go:build integration

package ollama_test

import (
	"context"
	"os"
	"testing"

	"github.com/scriptkid23/boostmcp/internal/inference"
	"github.com/scriptkid23/boostmcp/internal/inference/ollama"
)

func TestIntegrationGenerate(t *testing.T) {
	if os.Getenv("OLLAMA_INTEGRATION") != "1" {
		t.Skip("set OLLAMA_INTEGRATION=1 to run")
	}
	p := ollama.New("http://localhost:11434", "codellama:7b", 60000)
	if err := p.HealthCheck(context.Background()); err != nil {
		t.Skipf("ollama not available: %v", err)
	}
	resp, err := p.Generate(context.Background(), inference.GenerateRequest{
		Prompt: "Write a one-line Go hello world", Temperature: 0.3,
	})
	if err != nil {
		t.Fatal(err)
	}
	if resp.Text == "" {
		t.Fatal("empty response")
	}
}
