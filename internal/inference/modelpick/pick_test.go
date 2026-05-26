package modelpick_test

import (
	"testing"

	"github.com/scriptkid23/boostmcp/internal/inference/modelpick"
)

func TestPickPreferredExact(t *testing.T) {
	got, err := modelpick.Pick([]string{"gemma3:4b", "codellama:7b"}, "codellama:7b")
	if err != nil || got != "codellama:7b" {
		t.Fatalf("got %q err=%v", got, err)
	}
}

func TestPickPreferredFamily(t *testing.T) {
	got, err := modelpick.Pick([]string{"gemma3:4b"}, "codellama:7b")
	if err != nil || got != "gemma3:4b" {
		t.Fatalf("got %q err=%v", got, err)
	}
}

func TestPickCodeHint(t *testing.T) {
	got, err := modelpick.Pick([]string{"llama3:8b", "gemma3:4b"}, "")
	if err != nil || got != "llama3:8b" {
		t.Fatalf("got %q err=%v", got, err)
	}
}

func TestPickEmptyAvailable(t *testing.T) {
	_, err := modelpick.Pick(nil, "codellama:7b")
	if err == nil {
		t.Fatal("expected error")
	}
}
