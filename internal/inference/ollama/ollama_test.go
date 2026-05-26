package ollama_test

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/scriptkid23/boostmcp/internal/inference"
	"github.com/scriptkid23/boostmcp/internal/inference/ollama"
)

func TestGenerateSuccess(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/api/generate" {
			t.Fatalf("path: %s", r.URL.Path)
		}
		json.NewEncoder(w).Encode(map[string]any{
			"response":       "func main() {}",
			"eval_count":       10,
			"total_duration":   1_200_000_000,
		})
	}))
	defer srv.Close()

	p := ollama.New(srv.URL, "codellama:7b", 30000)
	resp, err := p.Generate(context.Background(), inference.GenerateRequest{
		Prompt:      "write main",
		Temperature: 0.7,
	})
	if err != nil {
		t.Fatal(err)
	}
	if resp.Text != "func main() {}" {
		t.Fatalf("got %q", resp.Text)
	}
	if resp.TokenCount != 10 {
		t.Fatalf("tokens: %d", resp.TokenCount)
	}
}

func TestHealthCheckSuccess(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		json.NewEncoder(w).Encode(map[string]any{"models": []any{}})
	}))
	defer srv.Close()
	p := ollama.New(srv.URL, "m", 5000)
	if err := p.HealthCheck(context.Background()); err != nil {
		t.Fatal(err)
	}
}

func TestHealthCheckUnreachable(t *testing.T) {
	p := ollama.New("http://127.0.0.1:1", "m", 1000)
	err := p.HealthCheck(context.Background())
	if err == nil {
		t.Fatal("expected error")
	}
	if !strings.Contains(err.Error(), "Ollama unreachable") {
		t.Fatalf("error must be actionable per spec §8; got: %v", err)
	}
}
