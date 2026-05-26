//go:build e2e

package mcp_test

import (
	"bufio"
	"context"
	"encoding/json"
	"io"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"
	"time"
)

func repoRoot(t *testing.T) string {
	t.Helper()
	dir, err := os.Getwd()
	if err != nil {
		t.Fatal(err)
	}
	for {
		if _, err := os.Stat(filepath.Join(dir, "go.mod")); err == nil {
			return dir
		}
		parent := filepath.Dir(dir)
		if parent == dir {
			t.Fatal("go.mod not found")
		}
		dir = parent
	}
}

func buildBinary(t *testing.T) string {
	t.Helper()
	dir := t.TempDir()
	out := filepath.Join(dir, "boostmcp.exe")
	cmd := exec.Command("go", "build", "-o", out, "./cmd/boostmcp")
	cmd.Dir = repoRoot(t)
	cmd.Stderr = os.Stderr
	if err := cmd.Run(); err != nil {
		t.Fatalf("build: %v", err)
	}
	return out
}

func TestE2EInitializeAndListTools(t *testing.T) {
	bin := buildBinary(t)
	ctx, cancel := context.WithTimeout(context.Background(), 15*time.Second)
	defer cancel()

	cmd := exec.CommandContext(ctx, bin)
	stdin, err := cmd.StdinPipe()
	if err != nil {
		t.Fatal(err)
	}
	stdout, err := cmd.StdoutPipe()
	if err != nil {
		t.Fatal(err)
	}
	stderr, err := cmd.StderrPipe()
	if err != nil {
		t.Fatal(err)
	}
	go io.Copy(io.Discard, stderr)

	if err := cmd.Start(); err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() {
		_ = stdin.Close()
		_ = cmd.Process.Kill()
		_ = cmd.Wait()
	})

	send := func(payload string) {
		if _, err := stdin.Write([]byte(payload + "\n")); err != nil {
			t.Fatalf("write: %v", err)
		}
	}
	reader := bufio.NewReader(stdout)
	readLine := func() string {
		line, err := reader.ReadString('\n')
		if err != nil {
			t.Fatalf("read: %v", err)
		}
		return line
	}

	send(`{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2024-11-05","capabilities":{},"clientInfo":{"name":"e2e","version":"1.0"}}}`)
	initResp := readLine()
	if !strings.Contains(initResp, `"result"`) {
		t.Fatalf("expected initialize result, got: %s", initResp)
	}

	send(`{"jsonrpc":"2.0","method":"notifications/initialized","params":{}}`)

	send(`{"jsonrpc":"2.0","id":2,"method":"tools/list"}`)
	toolsResp := readLine()
	if !strings.Contains(toolsResp, "generate_candidates") || !strings.Contains(toolsResp, "narrow_candidates") {
		t.Fatalf("missing tools in response: %s", toolsResp)
	}

	var rpc struct {
		JSONRPC string         `json:"jsonrpc"`
		ID      any            `json:"id"`
		Result  map[string]any `json:"result"`
		Error   any            `json:"error"`
	}
	if err := json.Unmarshal([]byte(toolsResp), &rpc); err != nil {
		t.Fatalf("tools/list response not JSON: %v\n%s", err, toolsResp)
	}
	if rpc.Error != nil {
		t.Fatalf("tools/list error: %v", rpc.Error)
	}
}
