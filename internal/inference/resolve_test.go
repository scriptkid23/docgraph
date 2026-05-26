package inference_test

import (
	"context"
	"testing"

	"github.com/scriptkid23/boostmcp/internal/inference"
	"github.com/scriptkid23/boostmcp/internal/inference/mock"
)

func TestResolveDefaultModelWithMock(t *testing.T) {
	p := mock.New([]string{"answer"})
	got, err := inference.ResolveDefaultModel(context.Background(), p, "codellama:7b")
	if err != nil {
		t.Fatal(err)
	}
	if got != "mock-model" {
		t.Fatalf("got %q", got)
	}
}
