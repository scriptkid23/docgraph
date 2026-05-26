package inference

import (
	"context"
	"fmt"

	"github.com/scriptkid23/boostmcp/internal/inference/modelpick"
)

// ResolveDefaultModel picks a model that exists on the provider.
// Updates preferred when it is missing but other models are available.
func ResolveDefaultModel(ctx context.Context, provider InferenceProvider, preferred string) (string, error) {
	models, err := provider.ListModels(ctx)
	if err != nil {
		return "", fmt.Errorf("list models: %w", err)
	}
	names := make([]string, len(models))
	for i, m := range models {
		names[i] = m.Name
	}
	picked, err := modelpick.Pick(names, preferred)
	if err != nil {
		return "", err
	}
	return picked, nil
}
