# Plan chunk-06 — Search Quality: Dedup, MMR Diversity, Optional Reranker

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Improve top-k precision and diversity of `search_documents` without changing the chunker. Three independent stages, all behind config flags:

1. **Content-hash dedup at write time** — never store two chunks with identical normalized text (eliminates URL-crawl boilerplate inflation).
2. **MMR diversity at query time** — top-k results no longer cluster on neighbouring chunks of the same section.
3. **Optional cross-encoder reranker** — when enabled, re-rank top-N (e.g. 30) candidates with `bge-reranker-base` (CPU-friendly, ~110 MB) to lift top-1/top-3 precision by 10–20%.

**Why:** The chunker is only half the retrieval story. With heading prefixes (chunk-02) lifting BM25/embedding scores AND a corpus that grows with multi-page URL crawls, dedup+diversity become the next bottleneck. A reranker is the single highest-leverage retrieval improvement after structural chunking.

**Architecture:** All three stages bolt onto existing surfaces — `ChromaStore.upsert_chunks` (dedup), `SearchService.search` (MMR + rerank). No new storage, no new tables. Reranker is a tiny `RerankerProvider` protocol mirroring the embedder.

**Depends on:** —
**Compatible with:** chunks already in the index (no re-index required for MMR or rerank; dedup affects only future writes).

**Spec:** `docs/superpowers/specs/2026-06-03-chunker-improvements-design.md` §3.

---

## File Structure

- **Modify** `docgraph/store/chroma.py` — content-hash dedup in `upsert_chunks`.
- **Create** `docgraph/mcp/diversify.py` — `mmr_select(query_vec, candidates, k, lambda_)`.
- **Create** `docgraph/rerank/provider.py` — `RerankerProvider` protocol.
- **Create** `docgraph/rerank/local.py` — local cross-encoder reranker (using `fastembed` or `sentence-transformers` cross-encoder).
- **Modify** `docgraph/mcp/search.py` — wire dedup-aware overfetch, MMR, and rerank stages.
- **Modify** `docgraph/config.py` — `dedup_enabled`, `mmr_lambda`, `rerank_enabled`, `rerank_model`.
- **Tests:** `tests/store/test_chroma_dedup.py`, `tests/mcp/test_diversify.py`, `tests/mcp/test_search_quality.py`, `tests/rerank/test_local.py`.

---

## Task 1: Content-hash dedup on write

**Files:**
- Modify: `docgraph/store/chroma.py`
- Test: `tests/store/test_chroma_dedup.py`

- [ ] **Step 1: Failing test**

```python
def test_upsert_skips_duplicate_chunks_within_same_doc(chroma_store):
    chunks = [
        {"id": "d_0", "embedding": [0.1]*768, "text": "Welcome to our site",
         "metadata": {"doc_id": "d", "filename": "a.md", "folder": "", "tags": "[]", "chunk_index": 0}},
        {"id": "d_1", "embedding": [0.1]*768, "text": "Welcome to our site",  # duplicate
         "metadata": {"doc_id": "d", "filename": "a.md", "folder": "", "tags": "[]", "chunk_index": 1}},
        {"id": "d_2", "embedding": [0.2]*768, "text": "unique content here",
         "metadata": {"doc_id": "d", "filename": "a.md", "folder": "", "tags": "[]", "chunk_index": 2}},
    ]
    chroma_store.upsert_chunks(chunks)
    stored = chroma_store.get_by_doc_id("d")
    texts = [c["text"] for c in stored]
    assert texts.count("Welcome to our site") == 1
    assert "unique content here" in texts
```

- [ ] **Step 2: Run — expect FAIL.**

- [ ] **Step 3: Implement**
  - In `upsert_chunks`, compute `hashlib.sha1(text.strip().lower().encode("utf-8")).hexdigest()` per chunk.
  - Store the hash as `metadata["chunk_hash"]`.
  - Drop chunks whose `chunk_hash` already appears earlier in the batch (within-batch dedup).
  - Cross-doc/across-batch dedup: keep behind a stricter flag `dedup_scope: "doc"|"global" = "doc"` to avoid surprising users (global dedup hides legitimate same-string occurrences across docs).

- [ ] **Step 4: Run — expect PASS.**

- [ ] **Step 5: Commit** — `feat(chunk-06): content-hash dedup at write time`

---

## Task 2: MMR selector

**Files:**
- Create: `docgraph/mcp/diversify.py`
- Test: `tests/mcp/test_diversify.py`

- [ ] **Step 1: Failing test**

```python
# tests/mcp/test_diversify.py
import math
from docgraph.mcp.diversify import mmr_select


def _norm(v):
    n = math.sqrt(sum(x*x for x in v)) or 1.0
    return [x/n for x in v]


def test_mmr_picks_diverse_results():
    q = _norm([1.0, 0.0])
    # Three "near-duplicate" candidates and one diverse one
    cands = [
        {"id": "a", "embedding": _norm([0.95, 0.05]), "score": 0.95},
        {"id": "b", "embedding": _norm([0.94, 0.04]), "score": 0.94},
        {"id": "c", "embedding": _norm([0.93, 0.03]), "score": 0.93},
        {"id": "d", "embedding": _norm([0.5, 0.5]), "score": 0.50},
    ]
    chosen = mmr_select(q, cands, k=2, lambda_=0.3)
    ids = [c["id"] for c in chosen]
    assert ids[0] == "a"
    assert ids[1] == "d"  # diversity wins over the near-duplicates


def test_mmr_with_lambda_1_falls_back_to_pure_relevance():
    q = _norm([1.0, 0.0])
    cands = [
        {"id": "a", "embedding": _norm([0.95, 0.05]), "score": 0.95},
        {"id": "b", "embedding": _norm([0.94, 0.04]), "score": 0.94},
        {"id": "d", "embedding": _norm([0.5, 0.5]), "score": 0.50},
    ]
    chosen = mmr_select(q, cands, k=3, lambda_=1.0)
    assert [c["id"] for c in chosen] == ["a", "b", "d"]
```

- [ ] **Step 2: Run — expect FAIL.**

- [ ] **Step 3: Implement**

```python
# docgraph/mcp/diversify.py
from __future__ import annotations
import math
from typing import Any


def _dot(a, b):
    return sum(x*y for x, y in zip(a, b))


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
        best, best_score = None, -math.inf
        for c in remaining:
            rel = _dot(query_vec, c["embedding"])
            if not selected:
                mmr = rel
            else:
                redundancy = max(_dot(c["embedding"], s["embedding"]) for s in selected)
                mmr = lambda_ * rel - (1.0 - lambda_) * redundancy
            if mmr > best_score:
                best_score, best = mmr, c
        selected.append(best)
        remaining.remove(best)
    return selected
```

- [ ] **Step 4: Run — expect PASS.**

- [ ] **Step 5: Commit** — `feat(chunk-06): MMR diversity selector`

---

## Task 3: Wire MMR into search service

**Files:**
- Modify: `docgraph/store/chroma.py`, `docgraph/mcp/search.py`, `docgraph/config.py`
- Test: extend `tests/mcp/test_search.py`

- [ ] **Step 1: Failing test** — given two near-duplicate candidates and one diverse one in the index, `search(q, top_k=2)` returns the diverse pair (not the two near-duplicates).

- [ ] **Step 2: Run — expect FAIL** (current overfetch + min_score keeps both near-dupes).

- [ ] **Step 3: Implement**
  - `ChromaStore.search` learns `include_embeddings: bool = False`; when true, return per-result vectors so MMR can compute redundancy without re-embedding.
  - `SearchService.search`:
    1. Embed query.
    2. Overfetch `top_k * 5` from Chroma with `include_embeddings=True`.
    3. Apply `min_score` filter as today.
    4. If `cfg.mmr_lambda < 1.0`: `candidates = mmr_select(query_vec, candidates, k=top_k, lambda_=cfg.mmr_lambda)`. Else just take top-k.
  - `Config.mmr_lambda: float = 0.7` (mild diversity by default).

- [ ] **Step 4: Run — expect PASS.**

- [ ] **Step 5: Commit** — `feat(chunk-06): MMR diversity in SearchService`

---

## Task 4: Reranker protocol

**Files:**
- Create: `docgraph/rerank/__init__.py`, `docgraph/rerank/provider.py`
- Test: `tests/rerank/test_provider.py`

- [ ] **Step 1: Failing test** — protocol exists; a stub implementation returns scores.

- [ ] **Step 2: Run — expect FAIL.**

- [ ] **Step 3: Implement**

```python
# docgraph/rerank/provider.py
from __future__ import annotations
from typing import Protocol


class RerankerProvider(Protocol):
    async def rerank(
        self, query: str, documents: list[str]
    ) -> list[float]:
        """Return one relevance score per document (higher = more relevant)."""
        ...
```

- [ ] **Step 4: Commit** — `feat(chunk-06): reranker provider protocol`

---

## Task 5: Local cross-encoder reranker

**Files:**
- Create: `docgraph/rerank/local.py`
- Test: `tests/rerank/test_local.py` (marked `@pytest.mark.integration` because the model is ~110 MB)

- [ ] **Step 1: Failing test** — for query "python list comprehension", reranker scores doc about list comp higher than doc about CSS grid.

- [ ] **Step 2: Run — expect FAIL.**

- [ ] **Step 3: Implement**
  - Try the existing local-embedder Rust crate first: if `crates/docgraph-embed` can host a cross-encoder ONNX model (the same fastembed runtime supports rerankers), expose `docgraph_embed.rerank(query, docs)`.
  - Otherwise, use Python `fastembed` (already a dep candidate) or `sentence_transformers.CrossEncoder` as fallback. Prefer fastembed for ONNX/CPU consistency with the rest of the stack.
  - Default model: `BAAI/bge-reranker-base` (or `bge-reranker-v2-m3` for multilingual).
  - Lazy-init pattern matching `LocalEmbedder` (one async lock, init on first call).

- [ ] **Step 4: Run — expect PASS.**

- [ ] **Step 5: Commit** — `feat(chunk-06): local cross-encoder reranker`

---

## Task 6: Wire reranker into SearchService

**Files:**
- Modify: `docgraph/mcp/search.py`, `docgraph/config.py`
- Test: extend `tests/mcp/test_search.py` (mock reranker)

- [ ] **Step 1: Failing test** — when `rerank_enabled=True` with a stub reranker that scores `"alpha"` chunks highest, the top result is the `"alpha"` chunk even when its embedding distance was worse.

- [ ] **Step 2: Run — expect FAIL.**

- [ ] **Step 3: Implement**
  - Pipeline order: vector overfetch → min_score → **rerank top-N** (default N=20) → MMR top-k.
    - Rerank before MMR so MMR runs on already high-precision candidates.
  - `Config.rerank_enabled: bool = False`, `Config.rerank_top_n: int = 20`, `Config.rerank_model: str = "BAAI/bge-reranker-base"`.
  - When disabled, skip both the rerank stage and the model load.
  - Preserve the original embedding score AND add `rerank_score` to `SearchResult` for transparency.

- [ ] **Step 4: Run — expect PASS.**

- [ ] **Step 5: Commit** — `feat(chunk-06): optional reranker stage in SearchService`

---

## Task 7: Documentation + README

**Files:**
- Modify: `README.md`, `docs/superpowers/specs/2026-06-03-chunker-improvements-design.md`

- [ ] Document new env vars (`DOCGRAPH_DEDUP_ENABLED`, `DOCGRAPH_MMR_LAMBDA`, `DOCGRAPH_RERANK_ENABLED`, `DOCGRAPH_RERANK_MODEL`).
- [ ] Add a "Search Quality Tuning" subsection covering when to enable/disable each stage.
- [ ] Commit — `docs(chunk-06): document dedup/MMR/rerank knobs`

---

## Acceptance Criteria

- [ ] Re-uploading the same boilerplate chunk to the same doc does not duplicate the row in Chroma.
- [ ] With `mmr_lambda=0.5`, top-2 search results for a synthetic "near-duplicates + diverse" fixture include the diverse candidate.
- [ ] With `rerank_enabled=True`, on a labelled retrieval benchmark (small RAGAS-style set under `tests/fixtures/eval/`), top-1 precision is ≥ baseline + 10%.
- [ ] All three stages can be independently disabled via config and the system reverts to current behaviour.
- [ ] No regression in existing `tests/mcp/test_search.py`, `tests/store/test_chroma.py`.
