package candidate

type Candidate struct {
	ID       string   `json:"id"`
	Content  string   `json:"content"`
	Metadata Metadata `json:"metadata"`
}

type Metadata struct {
	Index      int    `json:"index"`
	Model      string `json:"model"`
	LatencyMs  int64  `json:"latency_ms"`
	TokenCount int    `json:"token_count"`
	// DiffStats is reserved for Phase 2 — generator does not populate it in v1.
	// See spec §5.1 (mentions diff_stats) and Phase 2 roadmap §11.
	DiffStats *DiffStats `json:"diff_stats,omitempty"`
}

// DiffStats is a forward-compatible placeholder so JSON shape can evolve
// without breaking the v1 contract. v1 always emits null/omitted.
type DiffStats struct {
	Added   int `json:"added"`
	Removed int `json:"removed"`
}

type GenerationStats struct {
	TotalMs   int64  `json:"total_ms"`
	Model     string `json:"model"`
	Requested int    `json:"requested"`
	Received  int    `json:"received"`
}

type GenerateResult struct {
	Candidates      []Candidate     `json:"candidates"`
	GenerationStats GenerationStats `json:"generation_stats"`
}
