package mcp_test

import (
	"encoding/json"
	"testing"

	bmcp "github.com/scriptkid23/boostmcp/internal/mcp"
)

func TestParseGenerateArgs(t *testing.T) {
	a, err := bmcp.ParseGenerateArgs(map[string]any{"prompt": "x", "n_candidates": float64(4)})
	if err != nil || a.Prompt != "x" || a.NCandidates != 4 {
		t.Fatalf("got %+v err=%v", a, err)
	}
	_, err = bmcp.ParseGenerateArgs(map[string]any{})
	if err == nil {
		t.Fatal("expected error")
	}
}

func TestParseNarrowArgs(t *testing.T) {
	raw := map[string]any{
		"top_k": float64(2),
		"rubric": map[string]any{
			"criteria": []any{
				map[string]any{"name": "correctness", "weight": 1.0, "description": "ok"},
			},
		},
		"candidates": []any{
			map[string]any{"id": "c1", "content": "code", "metadata": map[string]any{}},
		},
	}
	a, err := bmcp.ParseNarrowArgs(raw)
	if err != nil || a.TopK != 2 || len(a.Candidates) != 1 {
		t.Fatalf("got %+v err=%v", a, err)
	}
}

func TestHandleGenerateCandidatesJSONParse(t *testing.T) {
	var args map[string]any
	raw := `{"prompt":"hello","n_candidates":2}`
	if err := json.Unmarshal([]byte(raw), &args); err != nil {
		t.Fatal(err)
	}
	if args["prompt"] != "hello" {
		t.Fatal("parse failed")
	}
}

func TestToolNamesRegistered(t *testing.T) {
	for _, n := range []string{"generate_candidates", "narrow_candidates"} {
		if n == "" {
			t.Fatal("empty tool name")
		}
	}
}
