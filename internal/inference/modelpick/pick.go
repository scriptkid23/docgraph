package modelpick

import (
	"fmt"
	"strings"
)

// codeModelHints prefers coding-capable models when the configured default is missing.
var codeModelHints = []string{
	"codellama",
	"qwen2.5-coder",
	"deepseek-coder",
	"starcoder",
	"codegemma",
	"llama3",
	"gemma",
	"mistral",
}

// Pick chooses an Ollama model name from available tags.
// preferred is tried first (exact / family match). Then codeModelHints, then the first available.
func Pick(available []string, preferred string) (string, error) {
	if len(available) == 0 {
		return "", fmt.Errorf("no Ollama models installed; run: ollama pull <model>")
	}
	if preferred != "" {
		if name, ok := matchPreferred(available, preferred); ok {
			return name, nil
		}
	}
	for _, hint := range codeModelHints {
		if name, ok := matchHint(available, hint); ok {
			return name, nil
		}
	}
	return available[0], nil
}

func matchPreferred(available []string, preferred string) (string, bool) {
	for _, a := range available {
		if a == preferred {
			return a, true
		}
	}
	preferredBase := baseName(preferred)
	for _, a := range available {
		if baseName(a) == preferredBase {
			return a, true
		}
	}
	return "", false
}

func matchHint(available []string, hint string) (string, bool) {
	hint = strings.ToLower(hint)
	for _, a := range available {
		if strings.Contains(strings.ToLower(a), hint) {
			return a, true
		}
	}
	return "", false
}

func baseName(model string) string {
	return strings.Split(model, ":")[0]
}
