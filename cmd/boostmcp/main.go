package main

import (
	"context"
	"log"
	"os"

	"github.com/scriptkid23/boostmcp/internal/config"
	bmcp "github.com/scriptkid23/boostmcp/internal/mcp"
)

func main() {
	log.SetOutput(os.Stderr)

	cfg, err := config.Load("")
	if err != nil {
		log.Fatalf("config: %v", err)
	}
	srv, err := bmcp.NewServer(cfg)
	if err != nil {
		log.Fatalf("server: %v", err)
	}
	ctx := context.Background()
	if err := srv.ProviderHealth(ctx); err != nil {
		log.Printf("warning: %v", err)
	}
	if err := srv.Run(context.Background()); err != nil {
		log.Fatalf("mcp: %v", err)
	}
	os.Exit(0)
}
