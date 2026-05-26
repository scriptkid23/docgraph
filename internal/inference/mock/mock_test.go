package mock_test

import (
	"context"
	"testing"

	"github.com/scriptkid23/boostmcp/internal/inference"
	"github.com/scriptkid23/boostmcp/internal/inference/mock"
)

func TestMockGenerate(t *testing.T) {
	m := mock.New([]string{"answer-a", "answer-b"})
	resp, err := m.Generate(context.Background(), inference.GenerateRequest{Prompt: "hi"})
	if err != nil {
		t.Fatal(err)
	}
	if resp.Text != "answer-a" {
		t.Fatalf("got %q, want answer-a", resp.Text)
	}
	resp2, _ := m.Generate(context.Background(), inference.GenerateRequest{Prompt: "hi"})
	if resp2.Text != "answer-b" {
		t.Fatalf("got %q, want answer-b", resp2.Text)
	}
}
