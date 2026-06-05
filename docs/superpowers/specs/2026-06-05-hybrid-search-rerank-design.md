# Hybrid Search + Reranker — Design

**Status:** Approved (brainstorming complete) — pending implementation plan
**Date:** 2026-06-05
**Author:** brainstorm session, DocGraph maintainer
**Scope:** Local single-user (per existing DocGraph deployment model). Multi-user noted as future work, NOT included.

---

## 1. Goal

Improve DocGraph search quality (precision + recall) by combining three relevance signals:

1. **Dense vector** (existing) — semantic similarity via Nomic / multilingual-E5 embeddings, cosine in ChromaDB.
2. **Sparse BM25** (new) — lexical/identifier match via SQLite FTS5 virtual table.
3. **Cross-encoder rerank** (new) — relevance judgment via fastembed-rs `TextRerank` (BGE-reranker-v2-m3, multilingual).

Each signal handles a class of queries the others miss:

| Signal | Wins on | Loses on |
|---|---|---|
| Dense vector | paraphrase, cross-lingual, conceptual | exact identifiers, numbers, acronyms, negation |
| Sparse BM25 | exact terms, identifiers, version strings | synonyms, paraphrase, semantic match |
| Cross-encoder rerank | question-specific relevance, negation, ambiguity | scale (only top candidates) |

**Vector + BM25 → recall**; **rerank → precision** when top candidates are ambiguous.

---

## 2. Non-goals

- Multi-user / multi-tenant. Scope explicitly local single-user; if multi-user is needed later, a separate spec replaces SQLite/Chroma with PostgreSQL + pgvector + tsvector.
- Re-embedding existing chunks. Embed model and chunker unchanged. Existing Chroma data preserved.
- Query rewriting / HyDE / contextual chunking. Separate future spec.
- Evaluation harness (golden Q&A, recall@k metrics). Noted as follow-up, separate spec.
- Snapshot tests of exact ranking output (model-version-fragile).

---

## 3. Architecture overview

```
query
  │
  ├─► embed_query ──► Chroma ANN (top 30)         ┐
  │                                                │
  └─► (raw text)  ──► FTS5 BM25 (top 30)          │── RRF fuse ─► fused_top (15)
                                                   │       │
                                                   ┘       ▼
                                              auto-rerank gate
                                              (A floor / B single-branch / C gap)
                                                   │
                                          ┌────────┴────────┐
                                       skip                rerank
                                          │            (Rust crate,
                                          │             top-15 passages)
                                          ▼                 │
                                       return            re-sort
                                                            │
                                                            ▼
                                                         return top-K
```

**Three new layers slot into existing pipeline:**

1. **Sparse index** — Virtual table `chunks_fts` (FTS5) inside existing `data.db`. Synced on chunk insert/delete in `indexer.py`. Tokenizer: `unicode61 remove_diacritics 2 tokenchars '_.-'`.
2. **Score fusion** — `SearchService` runs Chroma + FTS5 in parallel via `asyncio.gather`, fuses with **RRF (k=60)**.
3. **Reranker** — New module in `docgraph-embed` crate using `fastembed::TextRerank::BGERerankerV2M3`. Exposed as `rerank(query, passages) -> Vec<f32>` via PyO3. Called from `SearchService` after fusion, only when auto-gate fires.

**Backward compatibility:**
- FTS5 virtual table created in `_migrate_schema()` — no external migration step.
- Existing Chroma chunks lack FTS rows; server auto-rebuilds at startup when mismatch detected. CLI command `docgraph rebuild-fts` for manual control.
- Reranker model (~600MB ONNX) downloads on first use. Failures gracefully fall back to RRF order.
- Master switches: `hybrid_enabled` and `rerank_enabled` (both default `true`). Setting either false reverts to pre-existing behavior for that layer.

**File touch list:**

| Path | Status |
|---|---|
| `docgraph/store/sqlite.py` | Modify — extend `_migrate_schema()` |
| `docgraph/store/fts.py` | New — FTS5 wrapper |
| `docgraph/ingest/indexer.py` | Modify — write to FTS5 alongside Chroma |
| `docgraph/mcp/search.py` | Rewrite — hybrid pipeline, gate logic |
| `docgraph/embed/rerank.py` | New — Python wrapper for Rust reranker |
| `docgraph/config.py` | Modify — 9 new keys |
| `docgraph/cli.py` | Modify — `rebuild-fts` subcommand |
| `docgraph/web/app.py` | Modify — `/api/health` extended |
| `crates/docgraph-embed/src/lib.rs` | Modify — split into modules |
| `crates/docgraph-embed/src/embed.rs` | New (refactor of current lib.rs) |
| `crates/docgraph-embed/src/rerank.rs` | New |
| `README.md` | Modify — config/troubleshoot sections |

---

## 4. SQLite FTS5 schema + write path

### 4.1 Schema

```sql
CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
    chunk_id UNINDEXED,
    doc_id UNINDEXED,
    folder UNINDEXED,
    tags UNINDEXED,
    chunk_index UNINDEXED,
    text,         -- weight 1.0
    filename,     -- weight 2.0 in bm25()
    -- tags column indexed (weight 1.5 in bm25())
    content='',   -- contentless: text stored in Chroma, not duplicated here
    tokenize="unicode61 remove_diacritics 2 tokenchars '_.-'"
);
```

**Optimizations adopted (per Section 2 brainstorm review):**
- Opt 1: Multi-column with weighted BM25. `bm25(chunks_fts, 1.0, 2.0, 1.5)` boosts filename + tag matches.
- Opt 2: `content=''` contentless mode. Text stored only in Chroma (single source of truth, ~50% disk saving on `data.db`).
- Opt 6: `executemany` batch insert.
- Opt 7: Async non-blocking rebuild.

**Tokenizer choices:**
- `unicode61 remove_diacritics 2` — NFKD normalization. Query `"may tinh"` matches `"máy tính"`.
- `tokenchars '_.-'` — keep `embed_query`, `v1.5`, `nomic-embed-text` as single tokens (FTS5 default splits on these).

### 4.2 Write path

`indexer.py` change (around current line 86-104):

```python
# Build both atomically; same chunk_id, same text
chroma_chunks, fts_chunks = [], []
for i, (text, vec) in enumerate(zip(chunks, vectors)):
    chunk_id = f"{doc_id}_{i}"
    metadata = {...}  # unchanged
    chroma_chunks.append({"id": chunk_id, "embedding": vec, "text": text, "metadata": metadata})
    fts_chunks.append({
        "chunk_id": chunk_id, "doc_id": doc_id, "folder": doc.folder,
        "tags": json.dumps(doc.tags), "chunk_index": i,
        "text": text, "filename": doc.filename,
    })

self._chroma.upsert_chunks(chroma_chunks)  # no transaction
self._fts.upsert_chunks(fts_chunks)        # transactional
```

**Ordering: Chroma first, FTS5 second.** Failure between leaves vector-only state (search degrades, doesn't break). FTS5 transactional commit is all-or-nothing.

### 4.3 Delete path

Both delete sites (reindex + delete document) call:

```python
self._chroma.delete_by_doc_id(doc_id)
self._fts.delete_by_doc_id(doc_id)  # executes contentless-FTS5 'delete' command
```

### 4.4 Query sanitization

User query is not FTS5 syntax. Need to escape:

```python
def _sanitize_query(text: str) -> str:
    """Convert user input to safe FTS5 MATCH expression."""
    cleaned = re.sub(r'[^\w\s_.-]', ' ', text, flags=re.UNICODE)
    tokens = [t for t in cleaned.split() if t]
    if not tokens:
        return ""
    # Wrap each token as a phrase literal — neutralizes AND/OR/NOT/NEAR/* operators
    return " ".join(f'"{t}"' for t in tokens)
```

Empty result → FTS branch returns `[]`, pipeline falls back to vector-only for that query.

Future enhancement (phase 2, not v1): smart preservation of user-typed `"phrase"` syntax. Note in code comments.

### 4.5 Migration

`_migrate_schema()` runs `CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts ...` on every server start (idempotent).

**Existing data:** chunks already in Chroma have no FTS5 rows. Two-level handling:

| State | Action |
|---|---|
| `fts_count == 0` but `chroma_count > 0` | Auto-rebuild in background (`asyncio.create_task`). Search degrades gracefully during rebuild. |
| `\|chroma_count - fts_count\| ≤ max(10, 5%)` | Within race tolerance; no action. |
| `\|chroma_count - fts_count\| > 5%` | Log warning; user runs `docgraph rebuild-fts` manually. |

CLI `docgraph rebuild-fts` walks Chroma `collection.get()` in batches of 1000 (offset-paginated), looks up `filename`/`folder`/`tags` from SQLite documents table, bulk inserts to FTS5.

---

## 5. Search pipeline (RRF + auto-rerank)

### 5.1 Parallel branches

```python
async def search(self, query, top_k, folder, tags) -> list[SearchResult]:
    k = top_k or self._cfg.default_top_k
    overfetch = max(k * 6, 30)

    vector_task = asyncio.create_task(
        self._vector_branch(query, overfetch, folder, tags)
    )
    sparse_task = asyncio.create_task(
        asyncio.to_thread(self._fts.search, query, overfetch, folder, tags)
    )
    vector_results, sparse_results = await asyncio.gather(vector_task, sparse_task)

    fused = self._rrf_fuse(vector_results, sparse_results, k_rrf=self._cfg.rrf_k)
    fused = fused[: max(k * 3, 15)]

    if self._should_rerank(fused, k, query):
        fused = await self._rerank(query, fused)

    filtered = [
        r for r in fused
        if r.rerank_score is not None or r.vector_score is None or r.vector_score >= self._cfg.min_score
    ]
    return [self._to_search_result(r) for r in filtered[:k]]
```

**Design choices:**
- Overfetch 6× (vs current 3×) — hybrid needs wider net; rerank needs ~15 candidates to matter.
- `min_score` filter at end, NOT before fusion — chunk with vector 0.25 but strong BM25 hit shouldn't be dropped.
- `min_score` overridden when `rerank_score` exists — reranker re-confirmed relevance.

### 5.2 RRF fusion

```python
@dataclass
class FusedHit:
    chunk_id: str
    text: str
    metadata: dict
    vector_score: Optional[float]  # None if only in sparse
    bm25_score: Optional[float]    # None if only in vector
    rrf_score: float
    rerank_score: Optional[float] = None

def _rrf_fuse(vector_results, sparse_results, k_rrf=60) -> list[FusedHit]:
    scores: dict[str, FusedHit] = {}
    for rank, hit in enumerate(vector_results):
        scores.setdefault(hit.chunk_id, FusedHit.from_vector(hit))
        scores[hit.chunk_id].rrf_score += 1.0 / (k_rrf + rank + 1)
    for rank, hit in enumerate(sparse_results):
        scores.setdefault(hit.chunk_id, FusedHit.from_sparse(hit))
        scores[hit.chunk_id].rrf_score += 1.0 / (k_rrf + rank + 1)
        scores[hit.chunk_id].bm25_score = hit.bm25_score
    return sorted(scores.values(), key=lambda h: -h.rrf_score)
```

**`k_rrf=60`** per Cormack et al. 2009. Exposed in config but does not require tuning.

### 5.3 Auto-rerank gate

Gate has three conditions: floor skip (A), single-branch override (B), and the default gap-ratio rerank-when-ambiguous policy. Pre-warming the model (G in brainstorm notes) is a server-lifespan concern, NOT a gate condition — handled separately in `cli.serve` startup hook.

```python
def _should_rerank(self, fused, k, query) -> bool:
    if not self._cfg.rerank_enabled or len(fused) < 2:
        return False

    top1 = fused[0].rrf_score

    # A. Floor: top-1 below floor means recall failed; rerank can't fix that
    if top1 < self._cfg.rerank_min_floor:
        return False

    # B. Single-branch coverage: if only vector OR only sparse fired, force rerank
    has_vector = any(h.vector_score is not None for h in fused[:k])
    has_sparse = any(h.bm25_score is not None for h in fused[:k])
    if not (has_vector and has_sparse):
        return True

    # Default: rerank when top-k scores are clustered (ambiguous)
    top_scores = [h.rrf_score for h in fused[:k]]
    gap = top_scores[0] - top_scores[-1]
    if top1 > 0 and gap / top1 > self._cfg.rerank_score_gap_ratio:
        return False  # clear winner — skip rerank
    return True
```

**Pre-warm (separate from gate):** when `rerank_enabled=true` and `rerank_prewarm=true`, server lifespan startup spawns `asyncio.create_task(state.reranker.prewarm())`. This loads the model and runs a dummy inference in the background so the first user query is not blocked on cold start (~5-10s).

### 5.4 Rerank execution

```python
async def _rerank(self, query: str, candidates: list[FusedHit]) -> list[FusedHit]:
    if not candidates:
        return candidates
    passages = [h.text for h in candidates]
    try:
        scores = await asyncio.wait_for(
            asyncio.to_thread(self._reranker.rerank, query, passages),
            timeout=self._cfg.rerank_timeout_sec,
        )
    except (asyncio.TimeoutError, Exception) as exc:
        logger.warning("Rerank failed, falling back to RRF order: %s", exc)
        return candidates
    for hit, score in zip(candidates, scores):
        hit.rerank_score = float(score)
    return sorted(candidates, key=lambda h: -(h.rerank_score or 0.0))
```

**All failures fall back to RRF order.** Reranker is enhancement, never failure mode.

### 5.5 Return shape (backward compatible)

`SearchResult` unchanged. `score` populated by priority:

1. `rerank_score` if set
2. `vector_score` if set
3. normalized `bm25_score` if only sparse

JSON output additionally includes `signal` field (`"rerank"|"vector"|"bm25"|"hybrid"`) for debugging via Web UI; older clients ignore it.

### 5.6 Observability

Every search emits structured log line `search_metrics` with: `query_len`, `vector_branch_size`, `sparse_branch_size`, `fused_size`, `rerank_triggered`, `rerank_reason` (`skip_disabled|skip_floor|skip_gap|force_single_branch|force_ambiguous`), `total_ms`.

---

## 6. Rust crate extension

### 6.1 File layout

```
crates/docgraph-embed/src/
├── lib.rs       # pymodule + re-export
├── embed.rs     # TextEmbedding (unchanged logic, extracted from lib.rs)
└── rerank.rs    # TextRerank (new)
```

`lib.rs` becomes thin registration:

```rust
mod embed;
mod rerank;

#[pymodule]
fn docgraph_embed(m: &Bound<'_, PyModule>) -> PyResult<()> {
    // embed functions (existing)
    m.add_function(wrap_pyfunction!(embed::init, m)?)?;
    m.add_function(wrap_pyfunction!(embed::embed, m)?)?;
    m.add_function(wrap_pyfunction!(embed::health_check, m)?)?;
    m.add_function(wrap_pyfunction!(embed::embedding_dimension, m)?)?;
    m.add_function(wrap_pyfunction!(embed::active_model, m)?)?;
    // rerank functions (new)
    m.add_function(wrap_pyfunction!(rerank::rerank_init, m)?)?;
    m.add_function(wrap_pyfunction!(rerank::rerank, m)?)?;
    m.add_function(wrap_pyfunction!(rerank::rerank_health_check, m)?)?;
    m.add_function(wrap_pyfunction!(rerank::active_rerank_model, m)?)?;
    Ok(())
}
```

### 6.2 rerank.rs

State is separate `OnceCell<Mutex<Option<RerankState>>>` (independent lifecycle from embed).

```rust
struct RerankState {
    model: TextRerank,
    model_label: String,
}

fn resolve_rerank_model(name: &str) -> PyResult<RerankerModel> {
    let base = name.split(':').next().unwrap_or(name).trim();
    if let Ok(m) = RerankerModel::from_str(base) {
        return Ok(m);
    }
    match base.to_ascii_lowercase().as_str() {
        "bge-reranker-base" => Ok(RerankerModel::BGERerankerBase),
        "bge-reranker-v2-m3" | "bge-reranker-v2" | "bge-m3-reranker" => {
            Ok(RerankerModel::BGERerankerV2M3)
        }
        "jina-reranker-v2-multilingual" | "jina-reranker-v2" => {
            Ok(RerankerModel::JinaRerankerV2BaseMultilingual)
        }
        _ => Err(PyValueError::new_err(format!(
            "unknown reranker: {name}. Try bge-reranker-v2-m3 (default), ..."
        ))),
    }
}

#[pyfunction]
#[pyo3(signature = (model="bge-reranker-v2-m3", cache_dir=None))]
pub fn rerank_init(model: &str, cache_dir: Option<&str>) -> PyResult<()> { /* load */ }

#[pyfunction]
pub fn rerank(py: Python<'_>, query: String, passages: Vec<String>) -> PyResult<Vec<f32>> {
    if passages.is_empty() { return Ok(vec![]); }
    py.allow_threads(|| {
        with_state(|model, _| {
            let refs: Vec<&str> = passages.iter().map(String::as_str).collect();
            let results = model.rerank(&query, refs, false, None)
                .map_err(|e| PyRuntimeError::new_err(format!("rerank failed: {e}")))?;
            // fastembed may return sorted; re-order to input order
            let mut by_index = vec![0.0f32; passages.len()];
            for r in results {
                by_index[r.index] = r.score;
            }
            Ok(by_index)
        })
    })
}
```

**API design decisions:**
- Return `Vec<f32>` aligned to input index (NOT `Vec<(usize, f32)>` of fastembed's `RerankResult`). Simpler `zip` pattern in Python.
- No `top_n` parameter — Python layer caps before passing. Keeps Rust as pure scoring function; ranking policy stays in Python where it's testable and flexible.
- `return_documents=false` in fastembed call — don't copy text back across FFI.
- `py.allow_threads` releases GIL during ONNX forward pass.

### 6.3 Python wrapper

```python
class Reranker:
    HEALTH_MSG = (
        "Reranker unavailable. Build the Rust crate: "
        "cd crates/docgraph-embed && maturin develop --release"
    )

    def __init__(self, model: str, cache_dir: Path) -> None: ...
    async def _ensure_init(self) -> None: ...  # async lock, idempotent
    async def prewarm(self) -> None: ...        # called from server lifespan
    async def rerank(self, query: str, passages: list[str]) -> list[float]: ...
    async def health_check(self) -> None: ...
```

Mirrors `LocalEmbedder` pattern: async lock for concurrent init, lazy load on first call, `asyncio.to_thread` for non-blocking native call.

### 6.4 Cache directory

Reranker model stored in same `~/.docgraph/models/` as embedder. fastembed-rs creates subdir per model name (no collisions). ~600MB for BGE-v2-m3.

### 6.5 Graceful degradation matrix

| Failure | Behavior |
|---|---|
| `rerank_enabled=false` | `AppState.reranker = None`; gate returns false; pipeline = RRF only. |
| `docgraph_embed` import fails | Caught at first call; log warning; gate returns false. Search via vector path. |
| Model download fails | `rerank_init` raises; caught; `/api/health` reports `rerank_status=error`. Search continues. |
| Rerank exception mid-call | Caught in `_rerank`; returns un-reranked RRF order. |
| Rerank timeout (>3s) | `asyncio.wait_for` raises; same fallback. |

Reranker is enhancement only. Failure modes never propagate to user-facing search.

---

## 7. Config + backward compat

### 7.1 New config keys

```python
@dataclass
class Config:
    # ... existing ...

    # Hybrid search
    hybrid_enabled: bool = True
    rrf_k: int = 60

    # Reranker
    rerank_enabled: bool = True
    rerank_model: str = "bge-reranker-v2-m3"
    rerank_top_n: int = 15
    rerank_timeout_sec: float = 3.0
    rerank_prewarm: bool = True

    # Auto-rerank gate
    rerank_score_gap_ratio: float = 0.5
    rerank_min_floor: float = 0.015
```

### 7.2 YAML schema

```yaml
search:
  default_top_k: 5
  min_score: 0.3
  hybrid_enabled: true
  rrf_k: 60
  rerank:
    enabled: true
    model: bge-reranker-v2-m3
    top_n: 15
    timeout_sec: 3.0
    prewarm: true
    gate:
      score_gap_ratio: 0.5
      min_floor: 0.015
```

### 7.3 Environment variables

`DOCGRAPH_HYBRID_ENABLED`, `DOCGRAPH_RRF_K`, `DOCGRAPH_RERANK_ENABLED`, `DOCGRAPH_RERANK_MODEL`, `DOCGRAPH_RERANK_TOP_N`, `DOCGRAPH_RERANK_TIMEOUT_SEC`, `DOCGRAPH_RERANK_PREWARM`, `DOCGRAPH_RERANK_SCORE_GAP_RATIO`, `DOCGRAPH_RERANK_MIN_FLOOR`. Parse follows existing pattern in `_apply_env()`.

### 7.4 Config validation

```python
def validate(self) -> None:
    if not self.rerank_enabled and self.rerank_prewarm:
        self.rerank_prewarm = False  # silent coerce
    if self.rrf_k < 1:
        raise ValueError(f"rrf_k must be >= 1, got {self.rrf_k}")
    if not (0.0 <= self.rerank_score_gap_ratio <= 1.0):
        raise ValueError(f"rerank_score_gap_ratio must be in [0, 1]")
    if self.rerank_min_floor < 0:
        raise ValueError(f"rerank_min_floor must be >= 0")
    if self.rerank_top_n < 1:
        raise ValueError(f"rerank_top_n must be >= 1")
    if self.rerank_timeout_sec <= 0:
        raise ValueError(f"rerank_timeout_sec must be > 0")
```

Bounds violations raise immediately. Conflicts (prewarm+disabled) coerce silently.

### 7.5 Health endpoint

`GET /api/health` extended:

```python
{
    "embedding_provider": "local",
    "mcp_sse_url": "http://127.0.0.1:8088/mcp/sse",
    "embedder": "ok",
    "hybrid_enabled": true,
    "rerank_enabled": true,
    "rerank_status": "ready",  // "ready"|"loading"|"disabled"|"error: <msg>"
    "fts_chunks": 4231,
    "chroma_chunks": 4231,
    "fts_in_sync": true
}
```

### 7.6 Backward compat matrix

| Scenario | Behavior |
|---|---|
| Upgrade, no rebuild yet | Vector-only results during background rebuild. Search keeps working. |
| `hybrid_enabled=false` (explicit) | Pre-existing vector-only path. FTS5 table exists but not queried. |
| `rerank_enabled=false`, `hybrid_enabled=true` | RRF fusion only, no rerank. |
| Build crate missing, `rerank_enabled=true` | Init fails at first call; rerank skipped; vector+RRF still works. |
| Config YAML missing `search.rerank` section | Defaults apply. |
| Mix of pre-upgrade and post-upgrade docs | Sparse branch ranks new docs higher until rebuild equalizes. |
| FTS5 unsupported in SQLite build (rare) | Migration logs error, sets `cfg.hybrid_enabled=False` at runtime. |

### 7.7 CLI

```bash
poetry run docgraph rebuild-fts
```

Bulk repopulates `chunks_fts` from existing Chroma collection. Idempotent (clears table first). Prints progress.

---

## 8. Testing strategy

### 8.1 Test tiers

| Tier | Scope | Speed | When |
|---|---|---|---|
| Unit | Pure logic (gate, RRF, sanitize, config) | <2s | Every commit |
| Integration (light) | SQLite + Chroma + mocked reranker | <10s | Every commit |
| Integration (heavy) | Real Rust crate + ~600MB ONNX model | ~30-60s | Nightly + manual |
| Benchmark | pytest-benchmark, per-stage latency | ~30s | Pre-merge |
| Security | Injection, isolation, resource limits | <5s | Every commit |

### 8.2 New test files

| File | Tier | Approx LOC |
|---|---|---|
| `tests/mcp/test_rerank_gate.py` | Unit | 80 |
| `tests/mcp/test_rrf_fusion.py` | Unit | 60 |
| `tests/store/test_fts_sanitize.py` | Unit | 30 |
| `tests/test_config.py` (extend) | Unit | +50 |
| `tests/store/test_fts.py` | Integration light | 150 |
| `tests/mcp/test_hybrid_search.py` | Integration light | 200 |
| `tests/store/test_sqlite_migration.py` | Integration light | 50 |
| `tests/cli/test_rebuild_fts.py` | Integration light | 40 |
| `tests/embed/test_rerank.py` | Integration heavy | 80 |
| `tests/test_e2e.py` (extend) | Integration heavy | +80 |
| `tests/perf/test_search_latency.py` | Benchmark | 80 |
| `tests/security/test_fts_injection.py` | Security | 60 |
| `tests/security/test_search_isolation.py` | Security | 60 |
| `tests/security/test_resource_limits.py` | Security | 50 |
| `tests/security/test_rerank_input_safety.py` | Security | 40 |

Total: ~1100 LOC test for ~800 LOC production = 1.4× ratio.

### 8.3 Gate logic test coverage

Test every branch of `_should_rerank()`:
- Disabled → skip
- <2 candidates → skip
- Below floor → skip
- Single-branch (only vector) → force rerank
- Single-branch (only sparse) → force rerank
- Clear gap (>50% ratio) → skip
- Small gap (<50% ratio) → rerank
- Edge: top1=0 → floor check fires first

### 8.4 RRF math invariants

- Consensus (chunk in both branches) → score = sum of both reciprocals.
- Independence: chunk's RRF score depends only on its rank, not on other chunks (property-based test via Hypothesis).
- Sort order: highest RRF first.

### 8.5 Benchmark budgets (soft assertions; emit to baseline)

| Operation | Median budget |
|---|---|
| FTS5 search (1000 chunks, top-50) | <50ms |
| Vector search (1000 chunks, top-50) | <30ms |
| RRF fusion (30+30) | <2ms |
| Query sanitization | <1ms |
| E2E hybrid no rerank | p95 <150ms |
| E2E hybrid with rerank | p95 <1500ms |

Baselines saved via `pytest-benchmark --benchmark-save=baseline`. CI fails on >50% regression.

### 8.6 Security tests

**Injection (`test_fts_injection.py`):**
- SQL injection in user query — verify no crash, no data loss.
- FTS5 syntactic chars (`*`, `(`, `)`, `"`, `:`, `^`, `+`, `-`, `AND`, `OR`, `NOT`, `NEAR`, `MATCH`) — all safely handled.
- Long input (10KB query), null bytes, control chars — no crash.
- Folder filter with SQL fragment — verify parameter binding.

**Isolation (`test_search_isolation.py`):**
- Folder filter applied uniformly to both branches (CRITICAL: missing filter on one branch leaks across folders after fusion).
- Tag filter respected at fusion stage.

**Resource limits (`test_resource_limits.py`):**
- 100KB query completes in <5s.
- Slow reranker (>timeout) returns within budget via RRF fallback.
- `top_k=10000` does not blow memory (cap honored).

**Rerank input safety (`test_rerank_input_safety.py`):**
- Null bytes in text — handled.
- 100KB passage — truncated by tokenizer, no crash.
- Unicode edge cases (combining, RTL, emoji, surrogate) — no crash.

### 8.7 Observability test

Verify `search_metrics` log line emitted with expected fields. No assertion on values (they vary), only shape.

### 8.8 Markers + commands

```toml
markers = [
    "integration: real external services or services",
    "rerank_model: tests downloading ~600MB rerank ONNX model",
    "benchmark: performance baseline tests",
]
```

- `poetry run pytest -m "not integration and not rerank_model and not benchmark"` — fast suite, every commit.
- `poetry run pytest -m benchmark --benchmark-only` — performance suite.
- `poetry run pytest -m integration` — nightly.
- `poetry run pytest -m rerank_model` — manual model validation.

### 8.9 Coverage targets

| Module | Target |
|---|---|
| `mcp/search.py` | 95% |
| `store/fts.py` | 90% |
| `embed/rerank.py` | 80% |
| `config.py` (new fields) | 100% |
| `cli.py` (rebuild-fts) | 70% |

### 8.10 Non-goals (tests NOT included)

- Load testing (multi-user out of scope).
- Mutation testing.
- Snapshot tests on ranking output (model-version-fragile).
- A/B compare against eval harness (separate spec).

---

## 9. Risks + mitigations

| Risk | Mitigation |
|---|---|
| Reranker model size (~600MB) surprises user | README troubleshoot section explicit. `/api/health` reports `rerank_status=loading`. |
| FTS5 + Chroma drift over time (manual DB edits) | Auto-detect at startup; CLI command for manual rebuild. |
| Rerank quality regression when fastembed updates model URL | Pin via `rerank_model` config; integration tests on real model. |
| BM25 quality poor for short queries (single token) | Acceptable — RRF still benefits when both branches return same chunk. |
| Auto-rerank gate misfires (rerank when shouldn't / vice versa) | Tunable via config; observability log shows `rerank_reason`. |
| Concurrent indexing during rebuild races FTS5 writes | Rebuild uses SAVEPOINT per batch; ongoing inserts win (idempotent on chunk_id). |
| First query after server restart slow (5-10s cold start) | `rerank_prewarm=true` triggers warmup in background lifespan. |

---

## 10. Future work (NOT v1)

- **Query rewriting / HyDE** — embed hypothetical answer for harder queries.
- **Contextual chunking** — prepend doc/section summary to each chunk before embed.
- **Eval harness** — golden Q&A set, recall@k tracking. Prerequisite for tuning.
- **Multi-user** — separate spec; rewrites storage layer to PostgreSQL + pgvector + tsvector.
- **Smart phrase preservation** in FTS5 sanitize (user-typed `"foo bar"` preserved as phrase).
- **External-content FTS5 with triggers** — refactor if mutation paths grow beyond two.
- **Per-tenant rate limit on rerank** — relevant only at multi-user.

---

## 11. Open questions

None at design freeze. All architectural decisions confirmed during brainstorm:

- Direction: Hybrid + Reranker (vs. contextual retrieval, knowledge graph, eval harness) ✓
- Doc language: mixed VN+EN → `bge-reranker-v2-m3` + `unicode61 remove_diacritics 2` ✓
- Reranker mode: auto via score gap (A + B + G) ✓
- Score fusion: RRF (k=60) ✓
- Reranker host: extend existing Rust crate ✓
- Sparse index: SQLite FTS5 (contentless, multi-column, weighted BM25) ✓
- Python cap on rerank `top_n` (vs Rust-side) ✓
- Scope: local single-user (multi-user noted as future) ✓
