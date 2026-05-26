package mcp

import (
	"context"
	"encoding/json"
	"fmt"

	"github.com/mark3labs/mcp-go/mcp"
	mcpserver "github.com/mark3labs/mcp-go/server"
	"github.com/scriptkid23/boostmcp/internal/pipeline/generator"
	"github.com/scriptkid23/boostmcp/internal/pipeline/narrower"
	"github.com/scriptkid23/boostmcp/pkg/candidate"
)

type generateArgs struct {
	Prompt      string `json:"prompt"`
	Context     string `json:"context"`
	NCandidates int    `json:"n_candidates"`
	Model       string `json:"model"`
}

type narrowArgs struct {
	Rubric     candidate.Rubric      `json:"rubric"`
	Candidates []candidate.Candidate `json:"candidates"`
	TopK       int                   `json:"top_k"`
}

func (s *Server) registerTools(srv *mcpserver.MCPServer) {
	srv.AddTool(mcp.NewTool("generate_candidates",
		mcp.WithDescription("Generate N diverse code candidates from local SLM"),
		mcp.WithString("prompt", mcp.Required(), mcp.Description("Task description")),
		mcp.WithString("context", mcp.Description("Optional file content or diff context")),
		mcp.WithNumber("n_candidates", mcp.Description("Number of candidates (1-16, default 4)")),
		mcp.WithString("model", mcp.Description("Optional model override")),
	), s.handleGenerateCandidates)

	srv.AddTool(mcp.NewTool("narrow_candidates",
		mcp.WithDescription("Narrow candidates using rubric, return top-k scored"),
		mcp.WithObject("rubric",
			mcp.Required(),
			mcp.Description("Rubric with weighted criteria and optional hard constraints"),
			mcp.Properties(map[string]any{
				"criteria": map[string]any{
					"type":        "array",
					"description": "Weighted scoring criteria (weights must sum to 1.0)",
					"items": map[string]any{
						"type": "object",
						"properties": map[string]any{
							"name":        map[string]any{"type": "string"},
							"weight":      map[string]any{"type": "number"},
							"description": map[string]any{"type": "string"},
						},
						"required": []string{"name", "weight", "description"},
					},
				},
				"hard_constraints": map[string]any{
					"type":  "array",
					"items": map[string]any{"type": "string"},
				},
			}),
		),
		mcp.WithArray("candidates",
			mcp.Required(),
			mcp.Description("Candidates returned from generate_candidates"),
			mcp.Items(map[string]any{
				"type": "object",
				"properties": map[string]any{
					"id":       map[string]any{"type": "string"},
					"content":  map[string]any{"type": "string"},
					"metadata": map[string]any{"type": "object"},
				},
				"required": []string{"id", "content"},
			}),
		),
		mcp.WithNumber("top_k", mcp.Description("Number of top candidates to return (default from config)")),
	), s.handleNarrowCandidates)
}

func ParseGenerateArgs(raw map[string]any) (generateArgs, error) {
	b, err := json.Marshal(raw)
	if err != nil {
		return generateArgs{}, err
	}
	var a generateArgs
	if err := json.Unmarshal(b, &a); err != nil {
		return generateArgs{}, err
	}
	if a.Prompt == "" {
		return generateArgs{}, fmt.Errorf("prompt is required")
	}
	return a, nil
}

func ParseNarrowArgs(raw map[string]any) (narrowArgs, error) {
	b, err := json.Marshal(raw)
	if err != nil {
		return narrowArgs{}, err
	}
	var a narrowArgs
	if err := json.Unmarshal(b, &a); err != nil {
		return narrowArgs{}, err
	}
	if err := a.Rubric.Validate(); err != nil {
		return narrowArgs{}, fmt.Errorf("invalid rubric: %w", err)
	}
	if len(a.Candidates) == 0 {
		return narrowArgs{}, fmt.Errorf("candidates is required")
	}
	return a, nil
}

func (s *Server) handleGenerateCandidates(ctx context.Context, req mcp.CallToolRequest) (*mcp.CallToolResult, error) {
	args, err := ParseGenerateArgs(req.GetArguments())
	if err != nil {
		return mcp.NewToolResultError(err.Error()), nil
	}
	result, err := s.generator.Generate(ctx, generator.Input{
		Prompt: args.Prompt, Context: args.Context, NCandidates: args.NCandidates, Model: args.Model,
	})
	if err != nil {
		return mcp.NewToolResultError(err.Error()), nil
	}
	out, _ := json.Marshal(result)
	return mcp.NewToolResultText(string(out)), nil
}

func (s *Server) handleNarrowCandidates(ctx context.Context, req mcp.CallToolRequest) (*mcp.CallToolResult, error) {
	args, err := ParseNarrowArgs(req.GetArguments())
	if err != nil {
		return mcp.NewToolResultError(err.Error()), nil
	}
	result, err := s.narrower.Narrow(ctx, narrower.Input{
		Rubric: args.Rubric, Candidates: args.Candidates, TopK: args.TopK,
	})
	if err != nil {
		return mcp.NewToolResultError(err.Error()), nil
	}
	out, _ := json.Marshal(result)
	return mcp.NewToolResultText(string(out)), nil
}
