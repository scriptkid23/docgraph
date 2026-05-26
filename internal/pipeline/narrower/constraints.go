package narrower

import (
	"strings"

	"github.com/scriptkid23/boostmcp/pkg/candidate"
)

func ApplyHardConstraints(candidates []candidate.Candidate, constraints []string) ([]candidate.Candidate, []candidate.Dropped) {
	if len(constraints) == 0 {
		return candidates, nil
	}
	var survivors []candidate.Candidate
	var dropped []candidate.Dropped
	for _, c := range candidates {
		reason := checkConstraintsV1Heuristic(c.Content, constraints)
		if reason != "" {
			dropped = append(dropped, candidate.Dropped{CandidateID: c.ID, Reason: reason})
		} else {
			survivors = append(survivors, c)
		}
	}
	return survivors, dropped
}

func checkConstraintsV1Heuristic(content string, constraints []string) string {
	for _, hc := range constraints {
		switch strings.ToLower(strings.TrimSpace(hc)) {
		case "no new dependencies":
			if looksLikeNewImport(content) {
				return "failed hard constraint: no new dependencies"
			}
		default:
		}
	}
	return ""
}

func looksLikeNewImport(content string) bool {
	thirdParty := []string{
		"import numpy", "import pandas", "import requests",
		"\"github.com/", "require (",
	}
	for _, tp := range thirdParty {
		if strings.Contains(content, tp) {
			return true
		}
	}
	return false
}
