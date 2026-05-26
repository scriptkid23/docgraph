package mcp

import (
	"context"

	mcpserver "github.com/mark3labs/mcp-go/server"
	"github.com/scriptkid23/boostmcp/internal/config"
	"github.com/scriptkid23/boostmcp/internal/inference"
	"github.com/scriptkid23/boostmcp/internal/inference/factory"
	"github.com/scriptkid23/boostmcp/internal/pipeline/generator"
	"github.com/scriptkid23/boostmcp/internal/pipeline/narrower"
)

type Server struct {
	cfg       *config.Config
	provider  inference.InferenceProvider
	generator *generator.Generator
	narrower  *narrower.Narrower
}

func NewServer(cfg *config.Config) (*Server, error) {
	provider, err := factory.NewProvider(cfg)
	if err != nil {
		return nil, err
	}
	return &Server{
		cfg:       cfg,
		provider:  provider,
		generator: generator.New(provider, cfg),
		narrower:  narrower.New(provider, cfg),
	}, nil
}

func (s *Server) ProviderHealth(ctx context.Context) error {
	return s.provider.HealthCheck(ctx)
}

func (s *Server) Run(ctx context.Context) error {
	_ = ctx
	srv := mcpserver.NewMCPServer("boostmcp", "1.0.0", mcpserver.WithToolCapabilities(true))
	s.registerTools(srv)
	return mcpserver.ServeStdio(srv)
}
