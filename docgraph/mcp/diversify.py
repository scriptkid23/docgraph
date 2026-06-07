from __future__ import annotations

import math
from typing import Any


def _dot(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


def mmr_select(
    query_vec: list[float],
    candidates: list[dict[str, Any]],
    k: int,
    lambda_: float = 0.5,
) -> list[dict[str, Any]]:
    """Maximum Marginal Relevance: balance relevance vs. diversity."""
    if not candidates or k <= 0:
        return []
    selected: list[dict[str, Any]] = []
    remaining = list(candidates)
    while remaining and len(selected) < k:
        best: dict[str, Any] | None = None
        best_score = -math.inf
        for c in remaining:
            emb = c.get("embedding")
            if emb:
                rel = _dot(query_vec, emb)
            else:
                rel = c.get("score", 0.0)
            # Prefer explicit score when embeddings are nearly tied.
            if c.get("score") is not None:
                rel = 0.5 * rel + 0.5 * c["score"]
            if not selected:
                mmr = rel
            else:
                redundancy = max(
                    _dot(emb, s["embedding"])
                    for s in selected
                    if emb and s.get("embedding")
                ) if emb and any(s.get("embedding") for s in selected) else 0.0
                mmr = lambda_ * rel - (1.0 - lambda_) * redundancy
            if mmr > best_score:
                best_score = mmr
                best = c
        if best is None:
            break
        selected.append(best)
        remaining.remove(best)
    return selected
