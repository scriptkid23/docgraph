# Hybrid Search + Reranker Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add SQLite FTS5 sparse search + cross-encoder reranker to DocGraph's existing vector retrieval pipeline, with RRF fusion and an auto-rerank gate that fires only on ambiguous results.

**Architecture:** Three signals merge via Reciprocal Rank Fusion: dense vectors (existing Chroma), sparse BM25 (new SQLite FTS5 virtual table), and cross-encoder rerank (new module in `docgraph-embed` Rust crate using fastembed `BGERerankerV2M3`). Gate logic skips rerank when one branch dominates (clear winner) or when top-1 score is below a recall floor; forces rerank when only one branch fires. All new layers are master-switchable; failures fall back to vector-only.

**Tech Stack:** Python 3.10+, Poetry, SQLite FTS5 (`unicode61` tokenizer), ChromaDB, Rust + PyO3 + fastembed-rs, maturin, FastAPI, pytest + pytest-asyncio + pytest-benchmark.

**Spec reference:** `docs/superpowers/specs/2026-06-05-hybrid-search-rerank-design.md`

---

## File Structure

**Created:**
- `crates/docgraph-embed/src/embed.rs` — extracted from `lib.rs`
- `crates/docgraph-embed/src/rerank.rs` — new reranker bindings
- `docgraph/store/fts.py` — SQLite FTS5 wrapper
- `docgraph/embed/rerank.py` — Python reranker wrapper
- `tests/store/test_fts.py`
- `tests/store/test_fts_sanitize.py`
- `tests/store/test_sqlite_migration.py`
- `tests/mcp/test_rerank_gate.py`
- `tests/mcp/test_rrf_fusion.py`
- `tests/mcp/test_hybrid_search.py`
- `tests/embed/test_rerank.py`
- `tests/cli/__init__.py`
- `tests/cli/test_rebuild_fts.py`
- `tests/perf/__init__.py`
- `tests/perf/test_search_latency.py`
- `tests/security/__init__.py`
- `tests/security/test_fts_injection.py`
- `tests/security/test_search_isolation.py`
- `tests/security/test_resource_limits.py`
- `tests/security/test_rerank_input_safety.py`

**Modified:**
- `crates/docgraph-embed/src/lib.rs` — split into `mod embed; mod rerank;` + re-export
- `docgraph/config.py` — add 9 fields, YAML keys, env vars, `validate()`
- `docgraph/store/sqlite.py` — extend `_migrate_schema()` with `chunks_fts`
- `docgraph/ingest/indexer.py` — write/delete in FTS5 alongside Chroma
- `docgraph/mcp/search.py` — hybrid pipeline (RRF + gate + rerank)
- `docgraph/embed/factory.py` — add reranker factory (or wire in AppState)
- `docgraph/store/chroma.py` — add `count_chunks()` helper
- `docgraph/store/__init__.py` — export `FtsStore`
- `docgraph/web/deps.py` — wire `FtsStore` + `Reranker` into `AppState`
- `docgraph/web/app.py` — extend `/api/health`
- `docgraph/cli.py` — add `rebuild-fts` subcommand + lifespan pre-warm
- `docgraph/models.py` — add `signal` field to `SearchResult` (optional)
- `tests/test_config.py` — extend with new config keys
- `tests/test_e2e.py` — extend with hybrid scenarios
- `pyproject.toml` — register `pytest-benchmark` dev dep + markers
- `README.md` — config table + troubleshoot section

---

## Execution Order Notes

Tasks 1-2 (Rust crate) can run in parallel with Tasks 3-4 (Python config + migration), but later tasks depend on these. Tasks 18-25 (tests + docs) depend on functional code from Tasks 5-17.

---

## Task 1: Split Rust crate into modules

**Files:**
- Modify: `crates/docgraph-embed/src/lib.rs` (entire file)
- Create: `crates/docgraph-embed/src/embed.rs`

- [ ] **Step 1: Create `embed.rs` with existing logic**

Copy the entire content of current `crates/docgraph-embed/src/lib.rs` into a new file `crates/docgraph-embed/src/embed.rs`, then make these changes inside the new file:

1. Remove the `#[pymodule]` function at the bottom (it stays in lib.rs).
2. Change `#[pyfunction]` items to `pub fn` (add `pub` for visibility).
3. Keep all other code identical (state, helpers, resolve_model, load_model, init, embed, health_check, embedding_dimension, active_model).

Final structure of `embed.rs`:

```rust
use std::path::PathBuf;
use std::str::FromStr;
use std::sync::Mutex;

use fastembed::{EmbeddingModel, InitOptions, TextEmbedding};
use once_cell::sync::OnceCell;
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyList;

struct EmbedState { /* ... unchanged ... */ }
static STATE: OnceCell<Mutex<Option<EmbedState>>> = OnceCell::new();

fn state_mutex() -> &'static Mutex<Option<EmbedState>> { /* unchanged */ }
fn resolve_model(name: &str) -> PyResult<EmbeddingModel> { /* unchanged */ }
fn load_model(model_name: &str, cache_dir: Option<&str>) -> PyResult<EmbedState> { /* unchanged */ }
fn with_state<F, T>(f: F) -> PyResult<T> where /* unchanged */ { /* unchanged */ }

#[pyfunction]
#[pyo3(signature = (model="nomic-embed-text", cache_dir=None))]
pub fn init(model: &str, cache_dir: Option<&str>) -> PyResult<()> { /* unchanged */ }

#[pyfunction]
pub fn health_check() -> PyResult<()> { /* unchanged */ }

#[pyfunction]
pub fn embedding_dimension() -> PyResult<usize> { /* unchanged */ }

#[pyfunction]
pub fn active_model() -> PyResult<String> { /* unchanged */ }

#[pyfunction]
pub fn embed(py: Python<'_>, texts: Vec<String>) -> PyResult<Py<PyList>> { /* unchanged */ }
```

- [ ] **Step 2: Rewrite `lib.rs` as thin module declaration**

Replace entire content of `crates/docgraph-embed/src/lib.rs` with:

```rust
mod embed;

use pyo3::prelude::*;

#[pymodule]
fn docgraph_embed(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(embed::init, m)?)?;
    m.add_function(wrap_pyfunction!(embed::embed, m)?)?;
    m.add_function(wrap_pyfunction!(embed::health_check, m)?)?;
    m.add_function(wrap_pyfunction!(embed::embedding_dimension, m)?)?;
    m.add_function(wrap_pyfunction!(embed::active_model, m)?)?;
    Ok(())
}
```

- [ ] **Step 3: Build and verify nothing broke**

```bash
env -u CONDA_PREFIX -u CONDA_DEFAULT_ENV -u CONDA_SHLVL -u CONDA_PROMPT_MODIFIER \
  poetry run maturin develop --release --manifest-path crates/docgraph-embed/Cargo.toml
```

Expected: `🛠 Installed docgraph-embed-0.1.0` and zero compiler errors.

- [ ] **Step 4: Run existing embed test to confirm unchanged behavior**

```bash
poetry run pytest tests/embed/ -v
```

Expected: All existing embed tests PASS (no behavioral change from refactor).

- [ ] **Step 5: Commit**

```bash
git add crates/docgraph-embed/src/embed.rs crates/docgraph-embed/src/lib.rs
git commit -m "refactor(rust): split docgraph-embed crate into modules"
```

---

## Task 2: Add reranker module to Rust crate

**Files:**
- Create: `crates/docgraph-embed/src/rerank.rs`
- Modify: `crates/docgraph-embed/src/lib.rs`

- [ ] **Step 1: Create `rerank.rs`**

```rust
use std::path::PathBuf;
use std::str::FromStr;
use std::sync::Mutex;

use fastembed::{RerankInitOptions, RerankerModel, TextRerank};
use once_cell::sync::OnceCell;
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;

struct RerankState {
    model: TextRerank,
    model_label: String,
}

static RERANK_STATE: OnceCell<Mutex<Option<RerankState>>> = OnceCell::new();

fn state_mutex() -> &'static Mutex<Option<RerankState>> {
    RERANK_STATE.get_or_init(|| Mutex::new(None))
}

fn resolve_rerank_model(name: &str) -> PyResult<RerankerModel> {
    let base = name.split(':').next().unwrap_or(name).trim();
    if let Ok(m) = RerankerModel::from_str(base) {
        return Ok(m);
    }
    let normalized = base.to_ascii_lowercase();
    let model = match normalized.as_str() {
        "bge-reranker-base" => RerankerModel::BGERerankerBase,
        "bge-reranker-v2-m3" | "bge-reranker-v2" | "bge-m3-reranker" => {
            RerankerModel::BGERerankerV2M3
        }
        "jina-reranker-v2-multilingual" | "jina-reranker-v2" => {
            RerankerModel::JinaRerankerV2BaseMultilingual
        }
        "jina-reranker-v1-base" | "jina-reranker-v1" => {
            RerankerModel::JinaRerankerV1BaseEn
        }
        _ => {
            return Err(PyValueError::new_err(format!(
                "unknown reranker model: {name}. \
                 Try bge-reranker-v2-m3 (default), bge-reranker-base, \
                 jina-reranker-v2-multilingual."
            )));
        }
    };
    Ok(model)
}

fn load_rerank_model(model_name: &str, cache_dir: Option<&str>) -> PyResult<RerankState> {
    let model_kind = resolve_rerank_model(model_name)?;
    let mut opts = RerankInitOptions::new(model_kind).with_show_download_progress(true);
    if let Some(dir) = cache_dir {
        if !dir.is_empty() {
            opts = opts.with_cache_dir(PathBuf::from(dir));
        }
    }
    let model = TextRerank::try_new(opts)
        .map_err(|e| PyRuntimeError::new_err(format!("failed to load reranker: {e}")))?;
    Ok(RerankState {
        model,
        model_label: model_name.to_string(),
    })
}

fn with_state<F, T>(f: F) -> PyResult<T>
where
    F: FnOnce(&TextRerank, &str) -> PyResult<T>,
{
    let guard = state_mutex()
        .lock()
        .map_err(|_| PyRuntimeError::new_err("rerank lock poisoned"))?;
    let state = guard
        .as_ref()
        .ok_or_else(|| {
            PyRuntimeError::new_err("reranker not initialized; call rerank_init() first")
        })?;
    f(&state.model, &state.model_label)
}

#[pyfunction]
#[pyo3(signature = (model="bge-reranker-v2-m3", cache_dir=None))]
pub fn rerank_init(model: &str, cache_dir: Option<&str>) -> PyResult<()> {
    let loaded =
        Python::with_gil(|py| py.allow_threads(|| load_rerank_model(model, cache_dir)))?;
    let mut guard = state_mutex()
        .lock()
        .map_err(|_| PyRuntimeError::new_err("rerank lock poisoned"))?;
    *guard = Some(loaded);
    Ok(())
}

#[pyfunction]
pub fn rerank_health_check() -> PyResult<()> {
    with_state(|_, _| Ok(()))
}

#[pyfunction]
pub fn active_rerank_model() -> PyResult<String> {
    with_state(|_, label| Ok(label.to_string()))
}

#[pyfunction]
pub fn rerank(py: Python<'_>, query: String, passages: Vec<String>) -> PyResult<Vec<f32>> {
    if passages.is_empty() {
        return Ok(vec![]);
    }
    let n = passages.len();
    let scores = py.allow_threads(|| -> PyResult<Vec<f32>> {
        with_state(|model, _| {
            let refs: Vec<&str> = passages.iter().map(String::as_str).collect();
            let results = model
                .rerank(&query, refs, false, None)
                .map_err(|e| PyRuntimeError::new_err(format!("rerank failed: {e}")))?;
            if results.len() != n {
                return Err(PyRuntimeError::new_err(format!(
                    "rerank: expected {} scores, got {}",
                    n,
                    results.len()
                )));
            }
            let mut by_index = vec![0.0f32; n];
            for r in results {
                if r.index >= n {
                    return Err(PyRuntimeError::new_err(format!(
                        "rerank: result index {} out of bounds for {} passages",
                        r.index, n
                    )));
                }
                by_index[r.index] = r.score;
            }
            Ok(by_index)
        })
    })?;
    Ok(scores)
}
```

- [ ] **Step 2: Update `lib.rs` to register reranker functions**

Replace `crates/docgraph-embed/src/lib.rs` content with:

```rust
mod embed;
mod rerank;

use pyo3::prelude::*;

#[pymodule]
fn docgraph_embed(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(embed::init, m)?)?;
    m.add_function(wrap_pyfunction!(embed::embed, m)?)?;
    m.add_function(wrap_pyfunction!(embed::health_check, m)?)?;
    m.add_function(wrap_pyfunction!(embed::embedding_dimension, m)?)?;
    m.add_function(wrap_pyfunction!(embed::active_model, m)?)?;
    m.add_function(wrap_pyfunction!(rerank::rerank_init, m)?)?;
    m.add_function(wrap_pyfunction!(rerank::rerank, m)?)?;
    m.add_function(wrap_pyfunction!(rerank::rerank_health_check, m)?)?;
    m.add_function(wrap_pyfunction!(rerank::active_rerank_model, m)?)?;
    Ok(())
}
```

- [ ] **Step 3: Build the crate**

```bash
env -u CONDA_PREFIX -u CONDA_DEFAULT_ENV -u CONDA_SHLVL -u CONDA_PROMPT_MODIFIER \
  poetry run maturin develop --release --manifest-path crates/docgraph-embed/Cargo.toml
```

Expected: build succeeds, no compile errors.

- [ ] **Step 4: Smoke-test the new functions are exposed**

```bash
poetry run python -c "import docgraph_embed; print(dir(docgraph_embed))"
```

Expected: output contains `rerank_init`, `rerank`, `rerank_health_check`, `active_rerank_model`.

- [ ] **Step 5: Commit**

```bash
git add crates/docgraph-embed/src/rerank.rs crates/docgraph-embed/src/lib.rs
git commit -m "feat(rust): add reranker module to docgraph-embed crate"
```

---

## Task 3: Add hybrid + rerank config keys

**Files:**
- Modify: `docgraph/config.py`
- Modify: `tests/test_config.py`

- [ ] **Step 1: Write failing tests**

Append to `tests/test_config.py`:

```python
import pytest
from pathlib import Path

from docgraph.config import Config, load_config


class TestRerankConfig:
    def test_defaults(self, tmp_path):
        cfg = Config(data_dir=tmp_path)
        assert cfg.hybrid_enabled is True
        assert cfg.rrf_k == 60
        assert cfg.rerank_enabled is True
        assert cfg.rerank_model == "bge-reranker-v2-m3"
        assert cfg.rerank_top_n == 15
        assert cfg.rerank_timeout_sec == 3.0
        assert cfg.rerank_prewarm is True
        assert cfg.rerank_score_gap_ratio == 0.5
        assert cfg.rerank_min_floor == 0.015

    def test_validate_negative_rrf_k_raises(self, tmp_path):
        cfg = Config(data_dir=tmp_path, rrf_k=-1)
        with pytest.raises(ValueError, match="rrf_k must be >= 1"):
            cfg.validate()

    def test_validate_gap_ratio_out_of_range(self, tmp_path):
        cfg = Config(data_dir=tmp_path, rerank_score_gap_ratio=1.5)
        with pytest.raises(ValueError, match="rerank_score_gap_ratio"):
            cfg.validate()

    def test_validate_negative_floor_raises(self, tmp_path):
        cfg = Config(data_dir=tmp_path, rerank_min_floor=-0.01)
        with pytest.raises(ValueError, match="rerank_min_floor"):
            cfg.validate()

    def test_validate_top_n_must_be_positive(self, tmp_path):
        cfg = Config(data_dir=tmp_path, rerank_top_n=0)
        with pytest.raises(ValueError, match="rerank_top_n"):
            cfg.validate()

    def test_validate_timeout_must_be_positive(self, tmp_path):
        cfg = Config(data_dir=tmp_path, rerank_timeout_sec=0.0)
        with pytest.raises(ValueError, match="rerank_timeout_sec"):
            cfg.validate()

    def test_validate_coerces_prewarm_when_disabled(self, tmp_path):
        cfg = Config(data_dir=tmp_path, rerank_enabled=False, rerank_prewarm=True)
        cfg.validate()
        assert cfg.rerank_prewarm is False

    def test_env_var_overrides(self, monkeypatch, tmp_path):
        monkeypatch.setenv("DOCGRAPH_DATA_DIR", str(tmp_path))
        monkeypatch.setenv("DOCGRAPH_HYBRID_ENABLED", "false")
        monkeypatch.setenv("DOCGRAPH_RRF_K", "30")
        monkeypatch.setenv("DOCGRAPH_RERANK_ENABLED", "false")
        monkeypatch.setenv("DOCGRAPH_RERANK_MODEL", "bge-reranker-base")
        monkeypatch.setenv("DOCGRAPH_RERANK_TOP_N", "10")
        monkeypatch.setenv("DOCGRAPH_RERANK_TIMEOUT_SEC", "5.0")
        monkeypatch.setenv("DOCGRAPH_RERANK_PREWARM", "false")
        monkeypatch.setenv("DOCGRAPH_RERANK_SCORE_GAP_RATIO", "0.7")
        monkeypatch.setenv("DOCGRAPH_RERANK_MIN_FLOOR", "0.02")
        cfg = load_config()
        assert cfg.hybrid_enabled is False
        assert cfg.rrf_k == 30
        assert cfg.rerank_enabled is False
        assert cfg.rerank_model == "bge-reranker-base"
        assert cfg.rerank_top_n == 10
        assert cfg.rerank_timeout_sec == 5.0
        assert cfg.rerank_prewarm is False
        assert cfg.rerank_score_gap_ratio == 0.7
        assert cfg.rerank_min_floor == 0.02
```

- [ ] **Step 2: Run tests to verify FAIL**

```bash
poetry run pytest tests/test_config.py::TestRerankConfig -v
```

Expected: All tests FAIL — `AttributeError` on `cfg.hybrid_enabled` etc.

- [ ] **Step 3: Add fields to `Config` dataclass**

In `docgraph/config.py`, add these fields to the `Config` dataclass (insert after `max_chunks_per_doc`):

```python
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

- [ ] **Step 4: Add `validate()` method to `Config`**

In `docgraph/config.py`, add inside `Config` class (after `ensure_dirs`):

```python
    def validate(self) -> None:
        """Resolve conflicts and enforce bounds. Raise on invalid values."""
        if not self.rerank_enabled and self.rerank_prewarm:
            self.rerank_prewarm = False  # silent coerce — incoherent but harmless
        if self.rrf_k < 1:
            raise ValueError(f"rrf_k must be >= 1, got {self.rrf_k}")
        if not (0.0 <= self.rerank_score_gap_ratio <= 1.0):
            raise ValueError(
                f"rerank_score_gap_ratio must be in [0, 1], got {self.rerank_score_gap_ratio}"
            )
        if self.rerank_min_floor < 0:
            raise ValueError(
                f"rerank_min_floor must be >= 0, got {self.rerank_min_floor}"
            )
        if self.rerank_top_n < 1:
            raise ValueError(f"rerank_top_n must be >= 1, got {self.rerank_top_n}")
        if self.rerank_timeout_sec <= 0:
            raise ValueError(
                f"rerank_timeout_sec must be > 0, got {self.rerank_timeout_sec}"
            )
```

- [ ] **Step 5: Extend `_apply_env()` with new env vars**

In `docgraph/config.py`, add to the end of `_apply_env()` function:

```python
    def _bool(v: str) -> bool:
        return v.strip().lower() in ("1", "true", "yes", "on")

    if v := os.getenv("DOCGRAPH_HYBRID_ENABLED"):
        cfg.hybrid_enabled = _bool(v)
    if v := os.getenv("DOCGRAPH_RRF_K"):
        cfg.rrf_k = int(v)
    if v := os.getenv("DOCGRAPH_RERANK_ENABLED"):
        cfg.rerank_enabled = _bool(v)
    if v := os.getenv("DOCGRAPH_RERANK_MODEL"):
        cfg.rerank_model = v
    if v := os.getenv("DOCGRAPH_RERANK_TOP_N"):
        cfg.rerank_top_n = int(v)
    if v := os.getenv("DOCGRAPH_RERANK_TIMEOUT_SEC"):
        cfg.rerank_timeout_sec = float(v)
    if v := os.getenv("DOCGRAPH_RERANK_PREWARM"):
        cfg.rerank_prewarm = _bool(v)
    if v := os.getenv("DOCGRAPH_RERANK_SCORE_GAP_RATIO"):
        cfg.rerank_score_gap_ratio = float(v)
    if v := os.getenv("DOCGRAPH_RERANK_MIN_FLOOR"):
        cfg.rerank_min_floor = float(v)
```

- [ ] **Step 6: Extend `_apply_yaml()` with new keys**

In `docgraph/config.py`, add to the existing `search` branch of `_apply_yaml()`:

```python
    if search := data.get("search"):
        cfg.default_top_k = int(search.get("default_top_k", cfg.default_top_k))
        cfg.min_score = float(search.get("min_score", cfg.min_score))
        cfg.hybrid_enabled = bool(search.get("hybrid_enabled", cfg.hybrid_enabled))
        cfg.rrf_k = int(search.get("rrf_k", cfg.rrf_k))
        if rerank := search.get("rerank"):
            cfg.rerank_enabled = bool(rerank.get("enabled", cfg.rerank_enabled))
            cfg.rerank_model = str(rerank.get("model", cfg.rerank_model))
            cfg.rerank_top_n = int(rerank.get("top_n", cfg.rerank_top_n))
            cfg.rerank_timeout_sec = float(rerank.get("timeout_sec", cfg.rerank_timeout_sec))
            cfg.rerank_prewarm = bool(rerank.get("prewarm", cfg.rerank_prewarm))
            if gate := rerank.get("gate"):
                cfg.rerank_score_gap_ratio = float(
                    gate.get("score_gap_ratio", cfg.rerank_score_gap_ratio)
                )
                cfg.rerank_min_floor = float(
                    gate.get("min_floor", cfg.rerank_min_floor)
                )
```

- [ ] **Step 7: Call `validate()` in `load_config()`**

In `docgraph/config.py`, modify `load_config()` to call validate just before return:

```python
def load_config() -> Config:
    data_dir = _expand_path(os.getenv("DOCGRAPH_DATA_DIR", "~/.docgraph"))
    cfg = Config(data_dir=data_dir)
    yaml_path = cfg.data_dir / "config.yaml"
    if yaml_path.exists():
        with yaml_path.open(encoding="utf-8") as f:
            _apply_yaml(cfg, yaml.safe_load(f) or {})
    _apply_env(cfg)
    cfg.ollama_url = normalize_ollama_url(cfg.ollama_url)
    cfg.validate()
    return cfg
```

- [ ] **Step 8: Run tests to verify PASS**

```bash
poetry run pytest tests/test_config.py::TestRerankConfig -v
```

Expected: All 8 tests PASS.

- [ ] **Step 9: Commit**

```bash
git add docgraph/config.py tests/test_config.py
git commit -m "feat(config): add hybrid search + rerank config keys"
```

---

## Task 4: Extend SQLite migration with chunks_fts virtual table

**Files:**
- Modify: `docgraph/store/sqlite.py:24-42`
- Create: `tests/store/test_sqlite_migration.py`

- [ ] **Step 1: Write failing migration test**

Create `tests/store/test_sqlite_migration.py`:

```python
from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from docgraph.config import Config
from docgraph.store.sqlite import SQLiteStore


def _fts_table_exists(db_path: Path) -> bool:
    with sqlite3.connect(db_path) as conn:
        row = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='chunks_fts'"
        ).fetchone()
    return row is not None


def test_migrate_creates_chunks_fts_table(tmp_path):
    cfg = Config(data_dir=tmp_path)
    cfg.ensure_dirs()
    store = SQLiteStore(cfg)
    store.init_schema()
    assert _fts_table_exists(cfg.sqlite_path)


def test_migrate_idempotent_when_chunks_fts_exists(tmp_path):
    cfg = Config(data_dir=tmp_path)
    cfg.ensure_dirs()
    store = SQLiteStore(cfg)
    store.init_schema()
    # Second invocation must not raise
    store.init_schema()
    assert _fts_table_exists(cfg.sqlite_path)


def test_migrate_from_old_schema_preserves_documents(tmp_path):
    cfg = Config(data_dir=tmp_path)
    cfg.ensure_dirs()
    db = cfg.sqlite_path
    # Simulate old DB without chunks_fts
    with sqlite3.connect(db) as conn:
        conn.executescript(
            """
            CREATE TABLE documents (
                id TEXT PRIMARY KEY, filename TEXT NOT NULL,
                folder TEXT NOT NULL DEFAULT '', tags TEXT NOT NULL DEFAULT '[]',
                status TEXT NOT NULL DEFAULT 'ready', chunk_count INTEGER NOT NULL DEFAULT 0,
                error_message TEXT, original_path TEXT NOT NULL DEFAULT '',
                markdown_path TEXT NOT NULL DEFAULT '',
                created_at TEXT NOT NULL DEFAULT (datetime('now'))
            );
            INSERT INTO documents (id, filename) VALUES ('doc_old_001', 'legacy.md');
            """
        )
    store = SQLiteStore(cfg)
    store.init_schema()
    docs = store.list_documents()
    assert len(docs) == 1
    assert docs[0].filename == "legacy.md"
    assert _fts_table_exists(db)
```

- [ ] **Step 2: Run tests to verify FAIL**

```bash
poetry run pytest tests/store/test_sqlite_migration.py -v
```

Expected: `test_migrate_creates_chunks_fts_table` and `test_migrate_from_old_schema_preserves_documents` FAIL — no `chunks_fts` table.

- [ ] **Step 3: Extend `_migrate_schema` to create FTS5 table**

In `docgraph/store/sqlite.py`, modify `_migrate_schema()` (existing method ~line 24-42) — append after the existing column migrations:

```python
    def _migrate_schema(self, conn: sqlite3.Connection) -> None:
        cols = {row[1] for row in conn.execute("PRAGMA table_info(documents)")}
        if "progress_pct" not in cols:
            conn.execute(
                "ALTER TABLE documents ADD COLUMN progress_pct INTEGER NOT NULL DEFAULT 0"
            )
        if "progress_phase" not in cols:
            conn.execute(
                "ALTER TABLE documents ADD COLUMN progress_phase TEXT NOT NULL DEFAULT ''"
            )
        if "source_type" not in cols:
            conn.execute(
                "ALTER TABLE documents ADD COLUMN source_type TEXT NOT NULL DEFAULT 'file'"
            )
        if "source_url" not in cols:
            conn.execute(
                "ALTER TABLE documents ADD COLUMN source_url TEXT NOT NULL DEFAULT ''"
            )
        # Hybrid search FTS5 sparse index.
        # contentless (content='') — text stored only in Chroma, not duplicated.
        # tokenchars '_.-' preserves identifiers like `embed_query`, `v1.5`.
        conn.executescript(
            """
            CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
                chunk_id UNINDEXED,
                doc_id UNINDEXED,
                folder UNINDEXED,
                tags UNINDEXED,
                chunk_index UNINDEXED,
                text,
                filename,
                content='',
                tokenize="unicode61 remove_diacritics 2 tokenchars '_.-'"
            );
            """
        )
```

- [ ] **Step 4: Run tests to verify PASS**

```bash
poetry run pytest tests/store/test_sqlite_migration.py -v
```

Expected: All 3 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add docgraph/store/sqlite.py tests/store/test_sqlite_migration.py
git commit -m "feat(store): add chunks_fts virtual table migration"
```

---

## Task 5: FtsStore — query sanitization (pure function)

**Files:**
- Create: `docgraph/store/fts.py`
- Create: `tests/store/test_fts_sanitize.py`

- [ ] **Step 1: Write failing tests for `_sanitize_query`**

Create `tests/store/test_fts_sanitize.py`:

```python
from __future__ import annotations

import pytest

from docgraph.store.fts import _sanitize_query


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("máy tính", '"máy" "tính"'),
        ("embed_query", '"embed_query"'),
        ("foo*bar", '"foo" "bar"'),
        ('say "hello"', '"say" "hello"'),
        ("AND OR NOT", '"AND" "OR" "NOT"'),
        ("   ", ""),
        ("", ""),
        ("(foo)", '"foo"'),
        ("v1.5", '"v1.5"'),
        ("a-b", '"a-b"'),
        ("nomic-embed-text", '"nomic-embed-text"'),
        ("***", ""),
        ("query: with colon", '"query" "with" "colon"'),
    ],
)
def test_sanitize(raw, expected):
    assert _sanitize_query(raw) == expected
```

- [ ] **Step 2: Run tests to verify FAIL**

```bash
poetry run pytest tests/store/test_fts_sanitize.py -v
```

Expected: All FAIL with `ImportError: cannot import name '_sanitize_query'`.

- [ ] **Step 3: Create `docgraph/store/fts.py` with `_sanitize_query`**

Create `docgraph/store/fts.py`:

```python
from __future__ import annotations

import re

# Token characters preserved: alphanumeric, underscore, dot, hyphen.
# Anything else (operators, quotes, punctuation) becomes a space.
_TOKEN_KEEP = re.compile(r"[^\w\s_.-]", flags=re.UNICODE)


def _sanitize_query(text: str) -> str:
    """Convert raw user input to a safe FTS5 MATCH expression.

    Wraps each token in double quotes so FTS5 treats it as a phrase literal.
    This neutralizes operators (AND, OR, NOT, NEAR, *, +, etc.) and special
    chars. Empty / whitespace / fully-stripped input returns "".
    """
    if not text:
        return ""
    cleaned = _TOKEN_KEEP.sub(" ", text)
    tokens = [t for t in cleaned.split() if t]
    if not tokens:
        return ""
    return " ".join(f'"{t}"' for t in tokens)
```

- [ ] **Step 4: Run tests to verify PASS**

```bash
poetry run pytest tests/store/test_fts_sanitize.py -v
```

Expected: All 13 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add docgraph/store/fts.py tests/store/test_fts_sanitize.py
git commit -m "feat(store): add FTS5 query sanitizer"
```

---

## Task 6: FtsStore — CRUD operations

**Files:**
- Modify: `docgraph/store/fts.py`
- Create: `tests/store/test_fts.py`

- [ ] **Step 1: Write failing tests for CRUD**

Create `tests/store/test_fts.py`:

```python
from __future__ import annotations

import time

import pytest

from docgraph.config import Config
from docgraph.store.fts import FtsStore
from docgraph.store.sqlite import SQLiteStore


def _chunk(chunk_id, text, *, doc_id=None, folder="", filename="test.md", tags="[]"):
    if doc_id is None:
        doc_id = chunk_id.rsplit("_", 1)[0] if "_" in chunk_id else chunk_id
    chunk_index = int(chunk_id.rsplit("_", 1)[-1]) if "_" in chunk_id else 0
    return {
        "chunk_id": chunk_id,
        "doc_id": doc_id,
        "folder": folder,
        "tags": tags,
        "chunk_index": chunk_index,
        "text": text,
        "filename": filename,
    }


@pytest.fixture
def fts(tmp_path):
    cfg = Config(data_dir=tmp_path)
    cfg.ensure_dirs()
    sqlite = SQLiteStore(cfg)
    sqlite.init_schema()  # creates chunks_fts
    return FtsStore(cfg)


class TestFtsCRUD:
    def test_upsert_then_search_exact_token(self, fts):
        fts.upsert_chunks([_chunk("doc_abc_0", "DocGraph uses ChromaDB for vectors")])
        hits = fts.search("ChromaDB", top_k=10)
        assert len(hits) == 1
        assert hits[0]["chunk_id"] == "doc_abc_0"
        assert hits[0]["bm25_score"] > 0

    def test_diacritics_normalization(self, fts):
        fts.upsert_chunks([_chunk("c_0", "máy tính cá nhân")])
        hits = fts.search("may tinh", top_k=10)
        assert len(hits) == 1

    def test_identifier_with_underscore_preserved(self, fts):
        fts.upsert_chunks([
            _chunk("a_0", "Call embed_query() to embed user input"),
            _chunk("b_0", "The embed function takes text and returns vectors"),
        ])
        hits = fts.search("embed_query", top_k=10)
        assert len(hits) >= 1
        assert hits[0]["chunk_id"] == "a_0"

    def test_filename_weighted_higher(self, fts):
        fts.upsert_chunks([
            _chunk("a_0", "various config options here", filename="install.md"),
            _chunk("b_0", "various install options here", filename="config.md"),
        ])
        hits = fts.search("config", top_k=10)
        assert hits[0]["chunk_id"] == "b_0"

    def test_delete_by_doc_id_removes_all_chunks_of_doc(self, fts):
        fts.upsert_chunks(
            [_chunk(f"doc_abc_{i}", f"chunk {i} of doc_abc", doc_id="doc_abc") for i in range(5)]
        )
        fts.upsert_chunks([_chunk("doc_xyz_0", "other document content", doc_id="doc_xyz")])
        fts.delete_by_doc_id("doc_abc")
        hits = fts.search("chunk", top_k=20)
        assert all(h["doc_id"] != "doc_abc" for h in hits)
        hits = fts.search("other", top_k=20)
        assert len(hits) == 1
        assert hits[0]["doc_id"] == "doc_xyz"

    def test_folder_filter(self, fts):
        fts.upsert_chunks([
            _chunk("a_0", "shared content here", folder="docs"),
            _chunk("b_0", "shared content here", folder="code"),
        ])
        hits = fts.search("shared", top_k=10, folder="docs")
        assert len(hits) == 1
        assert hits[0]["chunk_id"] == "a_0"

    def test_empty_query_returns_empty(self, fts):
        fts.upsert_chunks([_chunk("a_0", "any text whatsoever")])
        assert fts.search("", top_k=10) == []
        assert fts.search("   ", top_k=10) == []
        assert fts.search("***", top_k=10) == []

    def test_batch_executemany_perf_1000_chunks(self, fts):
        chunks = [_chunk(f"d_{i}", f"chunk text number {i}") for i in range(1000)]
        start = time.time()
        fts.upsert_chunks(chunks)
        elapsed = time.time() - start
        assert elapsed < 2.0, f"batch insert of 1000 took {elapsed:.2f}s"
        assert fts.count_chunks() == 1000

    def test_clear_removes_everything(self, fts):
        fts.upsert_chunks([_chunk(f"d_{i}", "text") for i in range(10)])
        assert fts.count_chunks() == 10
        fts.clear()
        assert fts.count_chunks() == 0
```

- [ ] **Step 2: Run tests to verify FAIL**

```bash
poetry run pytest tests/store/test_fts.py -v
```

Expected: All FAIL — `FtsStore` not defined / methods missing.

- [ ] **Step 3: Implement `FtsStore` CRUD**

Append to `docgraph/store/fts.py`:

```python
import json
import sqlite3
from typing import Any, Optional

from docgraph.config import Config


class FtsStore:
    """SQLite FTS5 sparse index wrapper.

    Mirrors the chunk IDs and folder/tags metadata stored in Chroma so
    that hybrid search can fuse results. Text is NOT stored here
    (contentless FTS5) — it lives in Chroma only.
    """

    def __init__(self, cfg: Config) -> None:
        self._path = cfg.sqlite_path

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self._path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        return conn

    def count_chunks(self) -> int:
        with self._connect() as conn:
            row = conn.execute("SELECT COUNT(*) AS n FROM chunks_fts").fetchone()
        return int(row["n"])

    def upsert_chunks(self, chunks: list[dict[str, Any]]) -> None:
        if not chunks:
            return
        # contentless FTS5 has no UPSERT — caller must ensure chunk_ids are
        # fresh (delete by doc_id before reinsert at re-index).
        rows = [
            (
                c["chunk_id"],
                c["doc_id"],
                c.get("folder", ""),
                c.get("tags", "[]"),
                c.get("chunk_index", 0),
                c["text"],
                c.get("filename", ""),
            )
            for c in chunks
        ]
        with self._connect() as conn:
            conn.executemany(
                """INSERT INTO chunks_fts
                   (chunk_id, doc_id, folder, tags, chunk_index, text, filename)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                rows,
            )

    def delete_by_doc_id(self, doc_id: str) -> None:
        with self._connect() as conn:
            conn.execute("DELETE FROM chunks_fts WHERE doc_id = ?", (doc_id,))

    def clear(self) -> None:
        with self._connect() as conn:
            conn.execute("DELETE FROM chunks_fts")

    def search(
        self,
        query: str,
        top_k: int = 30,
        folder: Optional[str] = None,
        tags: Optional[list[str]] = None,
    ) -> list[dict[str, Any]]:
        match_expr = _sanitize_query(query)
        if not match_expr:
            return []
        sql = (
            "SELECT chunk_id, doc_id, folder, tags, chunk_index, "
            "       bm25(chunks_fts, 1.0, 2.0, 1.5) AS bm25_score "
            "FROM chunks_fts WHERE chunks_fts MATCH ? "
        )
        params: list = [match_expr]
        if folder:
            sql += "AND folder = ? "
            params.append(folder)
        # bm25() returns negative-better scores; ORDER BY bm25_score ASC == best first.
        # Flip sign so higher = better (consistent with vector score convention).
        sql += "ORDER BY bm25_score LIMIT ?"
        params.append(top_k)
        required_tags = set(tags or ())
        with self._connect() as conn:
            rows = conn.execute(sql, params).fetchall()
        out: list[dict[str, Any]] = []
        for r in rows:
            chunk_tags = _decode_tags(r["tags"])
            if required_tags and not required_tags.issubset(chunk_tags):
                continue
            out.append(
                {
                    "chunk_id": r["chunk_id"],
                    "doc_id": r["doc_id"],
                    "folder": r["folder"],
                    "tags": chunk_tags,
                    "chunk_index": int(r["chunk_index"]),
                    # Flip sign: SQLite bm25() returns smaller=better; we invert for
                    # convention "higher score = more relevant".
                    "bm25_score": -float(r["bm25_score"]),
                }
            )
        return out


def _decode_tags(raw: Any) -> list[str]:
    """Tolerant decoder — same logic as ChromaStore._decode_tags."""
    if not raw:
        return []
    if isinstance(raw, list):
        return [str(t) for t in raw]
    s = str(raw)
    if s.startswith("["):
        try:
            v = json.loads(s)
            if isinstance(v, list):
                return [str(t) for t in v]
        except json.JSONDecodeError:
            pass
    return [t for t in s.split(",") if t]
```

- [ ] **Step 4: Export `FtsStore` from `docgraph.store`**

In `docgraph/store/__init__.py`, add `FtsStore` to the imports/exports:

```python
from docgraph.store.chroma import ChromaStore
from docgraph.store.files import FileStore
from docgraph.store.fts import FtsStore
from docgraph.store.sqlite import SQLiteStore

__all__ = ["ChromaStore", "FileStore", "FtsStore", "SQLiteStore"]
```

- [ ] **Step 5: Run tests to verify PASS**

```bash
poetry run pytest tests/store/test_fts.py -v
```

Expected: All 9 tests PASS.

- [ ] **Step 6: Commit**

```bash
git add docgraph/store/fts.py docgraph/store/__init__.py tests/store/test_fts.py
git commit -m "feat(store): add FtsStore with CRUD + BM25 search"
```

---

## Task 7: ChromaStore.count_chunks() helper

**Files:**
- Modify: `docgraph/store/chroma.py`
- Modify: `tests/store/test_chroma.py` (extend existing)

- [ ] **Step 1: Write failing test**

Append to `tests/store/test_chroma.py`:

```python
def test_count_chunks(tmp_path):
    from docgraph.config import Config
    from docgraph.store.chroma import ChromaStore

    cfg = Config(data_dir=tmp_path)
    cfg.ensure_dirs()
    store = ChromaStore(cfg)
    assert store.count_chunks() == 0
    store.upsert_chunks([
        {"id": "a_0", "embedding": [0.1] * 768, "text": "x", "metadata": {"doc_id": "a"}},
        {"id": "a_1", "embedding": [0.2] * 768, "text": "y", "metadata": {"doc_id": "a"}},
    ])
    assert store.count_chunks() == 2
```

- [ ] **Step 2: Run test to verify FAIL**

```bash
poetry run pytest tests/store/test_chroma.py::test_count_chunks -v
```

Expected: FAIL — `count_chunks` does not exist.

- [ ] **Step 3: Add `count_chunks()` to `ChromaStore`**

Append to `docgraph/store/chroma.py` inside `ChromaStore` class:

```python
    def count_chunks(self) -> int:
        return int(self._collection.count())
```

- [ ] **Step 4: Run test to verify PASS**

```bash
poetry run pytest tests/store/test_chroma.py::test_count_chunks -v
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add docgraph/store/chroma.py tests/store/test_chroma.py
git commit -m "feat(store): add ChromaStore.count_chunks helper"
```

---

## Task 8: FtsStore.rebuild_from_chroma — async batch rebuild

**Files:**
- Modify: `docgraph/store/fts.py`
- Modify: `tests/store/test_fts.py`

- [ ] **Step 1: Write failing test**

Append to `tests/store/test_fts.py`:

```python
import asyncio
import pytest

from docgraph.store.chroma import ChromaStore


@pytest.fixture
def chroma_and_sqlite(tmp_path):
    cfg = Config(data_dir=tmp_path)
    cfg.ensure_dirs()
    sqlite = SQLiteStore(cfg)
    sqlite.init_schema()
    chroma = ChromaStore(cfg)
    # Seed 3 docs (each 1 chunk) into SQLite + Chroma
    from docgraph.models import DocumentRecord
    for i, name in enumerate(["a", "b", "c"]):
        sqlite.insert_document(DocumentRecord(
            id=f"doc_{name}", filename=f"{name}.md", folder=f"f{i}", tags=[f"t{i}"]
        ))
    chroma.upsert_chunks([
        {"id": f"doc_{name}_0", "embedding": [0.1] * 768, "text": f"text {name}",
         "metadata": {"doc_id": f"doc_{name}", "filename": f"{name}.md",
                      "folder": f"f{i}", "tags": f'["t{i}"]', "chunk_index": 0}}
        for i, name in enumerate(["a", "b", "c"])
    ])
    return cfg, sqlite, chroma


@pytest.mark.asyncio
async def test_rebuild_from_chroma_populates_fts(chroma_and_sqlite):
    cfg, sqlite, chroma = chroma_and_sqlite
    fts = FtsStore(cfg)
    assert fts.count_chunks() == 0
    n = await fts.rebuild_from_chroma(chroma, sqlite)
    assert n == 3
    assert fts.count_chunks() == 3
    hits = fts.search("text", top_k=10)
    assert len(hits) == 3


@pytest.mark.asyncio
async def test_rebuild_clears_old_rows(chroma_and_sqlite):
    cfg, sqlite, chroma = chroma_and_sqlite
    fts = FtsStore(cfg)
    fts.upsert_chunks([_chunk("stale_0", "stale text", doc_id="stale")])
    assert fts.count_chunks() == 1
    await fts.rebuild_from_chroma(chroma, sqlite)
    # Stale row gone, only the 3 from chroma remain
    hits = fts.search("stale", top_k=10)
    assert hits == []
    assert fts.count_chunks() == 3
```

- [ ] **Step 2: Run tests to verify FAIL**

```bash
poetry run pytest tests/store/test_fts.py::test_rebuild_from_chroma_populates_fts tests/store/test_fts.py::test_rebuild_clears_old_rows -v
```

Expected: FAIL — `rebuild_from_chroma` does not exist.

- [ ] **Step 3: Implement `rebuild_from_chroma` on `FtsStore`**

Append to `FtsStore` class in `docgraph/store/fts.py`:

```python
    async def rebuild_from_chroma(
        self,
        chroma: "ChromaStore",
        sqlite: "SQLiteStore",
        batch_size: int = 1000,
        progress_callback=None,
    ) -> int:
        """Bulk re-populate chunks_fts from Chroma's existing collection.

        Clears existing chunks_fts rows first. Reads Chroma in batches of
        `batch_size`, looks up filename/folder/tags from SQLite documents,
        bulk inserts to FTS5. Returns total chunks indexed. Yields between
        batches via asyncio.sleep(0) so the event loop is not blocked.
        """
        import asyncio as _asyncio

        self.clear()
        total = chroma.count_chunks()
        if total == 0:
            return 0
        # Cache filename per doc_id to avoid hitting SQLite for each chunk
        doc_meta: dict[str, dict] = {}
        offset = 0
        indexed = 0
        while offset < total:
            batch = chroma._collection.get(
                limit=batch_size,
                offset=offset,
                include=["documents", "metadatas"],
            )
            ids = batch.get("ids", []) or []
            if not ids:
                break
            rows = []
            for chunk_id, text, meta in zip(ids, batch["documents"], batch["metadatas"]):
                doc_id = meta.get("doc_id", "")
                if doc_id and doc_id not in doc_meta:
                    rec = sqlite.get_document(doc_id)
                    doc_meta[doc_id] = (
                        {"filename": rec.filename, "folder": rec.folder, "tags": rec.tags}
                        if rec
                        else {"filename": meta.get("filename", ""),
                              "folder": meta.get("folder", ""), "tags": []}
                    )
                m = doc_meta.get(doc_id, {})
                rows.append({
                    "chunk_id": chunk_id,
                    "doc_id": doc_id,
                    "folder": m.get("folder", meta.get("folder", "")),
                    "tags": json.dumps(m.get("tags", [])),
                    "chunk_index": int(meta.get("chunk_index", 0)),
                    "text": text or "",
                    "filename": m.get("filename", meta.get("filename", "")),
                })
            await _asyncio.to_thread(self.upsert_chunks, rows)
            indexed += len(rows)
            offset += batch_size
            if progress_callback is not None:
                progress_callback(indexed, total)
            await _asyncio.sleep(0)  # yield to event loop
        return indexed
```

Also add forward-declared imports at the top of `docgraph/store/fts.py` (after existing imports):

```python
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from docgraph.store.chroma import ChromaStore
    from docgraph.store.sqlite import SQLiteStore
```

- [ ] **Step 4: Run tests to verify PASS**

```bash
poetry run pytest tests/store/test_fts.py::test_rebuild_from_chroma_populates_fts tests/store/test_fts.py::test_rebuild_clears_old_rows -v
```

Expected: Both PASS.

- [ ] **Step 5: Commit**

```bash
git add docgraph/store/fts.py tests/store/test_fts.py
git commit -m "feat(store): add async rebuild_from_chroma for FtsStore"
```

---

## Task 9: Indexer integration — write/delete in FTS5 alongside Chroma

**Files:**
- Modify: `docgraph/ingest/indexer.py:22-117,196-201`
- Modify: `tests/ingest/test_indexer.py` (extend existing tests)

- [ ] **Step 1: Write failing test**

Append to `tests/ingest/test_indexer.py` (create if missing):

```python
import pytest

from docgraph.config import Config
from docgraph.embed.local import LocalEmbedder  # or mock
from docgraph.ingest.indexer import Indexer
from docgraph.models import DocumentRecord
from docgraph.store import ChromaStore, FileStore, FtsStore, SQLiteStore


class FakeEmbedder:
    async def embed(self, texts, for_query=False):
        return [[0.1] * 768 for _ in texts]

    async def health_check(self):
        return None


@pytest.fixture
def pipeline(tmp_path):
    cfg = Config(data_dir=tmp_path)
    cfg.ensure_dirs()
    sqlite = SQLiteStore(cfg)
    sqlite.init_schema()
    files = FileStore(cfg)
    chroma = ChromaStore(cfg)
    fts = FtsStore(cfg)
    embedder = FakeEmbedder()
    indexer = Indexer(cfg, sqlite, files, chroma, embedder, fts=fts)
    return cfg, sqlite, chroma, fts, indexer


@pytest.mark.asyncio
async def test_index_markdown_writes_to_fts(pipeline):
    cfg, sqlite, chroma, fts, indexer = pipeline
    sqlite.insert_document(DocumentRecord(
        id="doc_test", filename="t.md", folder="x", tags=["v1"]
    ))
    await indexer.index_markdown(
        "doc_test", "## Section\n\nThis is text with embed_query identifier."
    )
    assert chroma.count_chunks() > 0
    assert fts.count_chunks() == chroma.count_chunks()
    # Verify identifier searchable in FTS
    hits = fts.search("embed_query", top_k=5)
    assert len(hits) >= 1
    assert hits[0]["doc_id"] == "doc_test"


@pytest.mark.asyncio
async def test_reindex_clears_old_fts_rows(pipeline):
    cfg, sqlite, chroma, fts, indexer = pipeline
    sqlite.insert_document(DocumentRecord(id="doc_re", filename="r.md"))
    await indexer.index_markdown("doc_re", "First version text")
    fts_count_v1 = fts.count_chunks()
    await indexer.reindex_document("doc_re", markdown="Completely different content here")
    # Old chunks gone from FTS, new ones inserted
    hits = fts.search("First", top_k=5)
    assert hits == []
    hits = fts.search("different", top_k=5)
    assert len(hits) >= 1
```

- [ ] **Step 2: Run tests to verify FAIL**

```bash
poetry run pytest tests/ingest/test_indexer.py::test_index_markdown_writes_to_fts -v
```

Expected: FAIL — `Indexer` constructor doesn't accept `fts` parameter.

- [ ] **Step 3: Modify `Indexer.__init__` to accept `fts`**

In `docgraph/ingest/indexer.py`, modify the `Indexer` class:

```python
class Indexer:
    def __init__(
        self,
        cfg: Config,
        sqlite: SQLiteStore,
        files: FileStore,
        chroma: ChromaStore,
        embedder: EmbeddingProvider,
        fts: "FtsStore | None" = None,
    ) -> None:
        self._cfg = cfg
        self._sqlite = sqlite
        self._files = files
        self._chroma = chroma
        self._embedder = embedder
        self._fts = fts
```

And add the import at top:

```python
from docgraph.store.fts import FtsStore
```

- [ ] **Step 4: Modify `index_markdown` to write to FTS5**

In `docgraph/ingest/indexer.py`, modify the chunk-building loop (around line 86-104) to build both chroma and fts payloads:

```python
            chroma_chunks = []
            fts_chunks = []
            for i, (text, vec) in enumerate(zip(chunks, vectors)):
                chunk_id = f"{doc_id}_{i}"
                metadata = {
                    "doc_id": doc_id,
                    "filename": doc.filename,
                    "folder": doc.folder,
                    "tags": json.dumps(doc.tags),
                    "chunk_index": i,
                }
                if doc.source_url:
                    metadata["source_url"] = doc.source_url
                chroma_chunks.append({
                    "id": chunk_id,
                    "embedding": vec,
                    "text": text,
                    "metadata": metadata,
                })
                fts_chunks.append({
                    "chunk_id": chunk_id,
                    "doc_id": doc_id,
                    "folder": doc.folder,
                    "tags": json.dumps(doc.tags),
                    "chunk_index": i,
                    "text": text,
                    "filename": doc.filename,
                })
            self._chroma.upsert_chunks(chroma_chunks)
            if self._fts is not None and self._cfg.hybrid_enabled:
                # Best-effort: FTS failure must not break ingest.
                try:
                    self._fts.upsert_chunks(fts_chunks)
                except Exception as exc:
                    logger.warning(
                        "FTS5 upsert failed for doc_id=%s (search will degrade to vector-only): %s",
                        doc_id, exc,
                    )
```

- [ ] **Step 5: Modify reindex/delete paths to also delete FTS5 rows**

In `docgraph/ingest/indexer.py`, find the existing `self._chroma.delete_by_doc_id(doc_id)` call (around line 200, inside `reindex_document` or wherever a doc's chunks are wiped). Replace with:

```python
        self._chroma.delete_by_doc_id(doc_id)
        if self._fts is not None:
            try:
                self._fts.delete_by_doc_id(doc_id)
            except Exception as exc:
                logger.warning("FTS5 delete failed for doc_id=%s: %s", doc_id, exc)
```

Apply the same pattern to any other place in the codebase that calls `chroma.delete_by_doc_id`. Run:

```bash
grep -n "delete_by_doc_id" docgraph/
```

For each match in `docgraph/web/app.py` (the DELETE document endpoint), add the FTS delete after the Chroma delete.

- [ ] **Step 6: Run tests to verify PASS**

```bash
poetry run pytest tests/ingest/test_indexer.py -v
```

Expected: New tests PASS, existing tests still PASS.

- [ ] **Step 7: Commit**

```bash
git add docgraph/ingest/indexer.py docgraph/web/app.py tests/ingest/test_indexer.py
git commit -m "feat(ingest): write and delete chunks in FTS5 alongside Chroma"
```

---

## Task 10: Reranker Python wrapper

**Files:**
- Create: `docgraph/embed/rerank.py`
- Create: `tests/embed/test_rerank.py`

- [ ] **Step 1: Write failing test (mocked native)**

Create `tests/embed/test_rerank.py`:

```python
from __future__ import annotations

import asyncio
import sys
from unittest.mock import MagicMock

import pytest

from docgraph.embed.rerank import Reranker


class TestRerankerWrapper:
    def _install_mock_module(self, monkeypatch, init_fn=None, rerank_fn=None, health_fn=None):
        mod = MagicMock()
        mod.rerank_init = init_fn or MagicMock(return_value=None)
        mod.rerank = rerank_fn or MagicMock(return_value=[0.9, 0.5, 0.1])
        mod.rerank_health_check = health_fn or MagicMock(return_value=None)
        monkeypatch.setitem(sys.modules, "docgraph_embed", mod)
        return mod

    @pytest.mark.asyncio
    async def test_rerank_returns_scores_in_order(self, tmp_path, monkeypatch):
        mod = self._install_mock_module(monkeypatch)
        r = Reranker(model="bge-reranker-v2-m3", cache_dir=tmp_path)
        scores = await r.rerank("query", ["passage a", "passage b", "passage c"])
        assert scores == [0.9, 0.5, 0.1]
        mod.rerank_init.assert_called_once()
        mod.rerank.assert_called_once_with("query", ["passage a", "passage b", "passage c"])

    @pytest.mark.asyncio
    async def test_rerank_empty_passages_returns_empty_without_init(self, tmp_path, monkeypatch):
        mod = self._install_mock_module(monkeypatch)
        r = Reranker(model="bge-reranker-v2-m3", cache_dir=tmp_path)
        scores = await r.rerank("query", [])
        assert scores == []
        # No native calls because we short-circuited
        mod.rerank_init.assert_not_called()
        mod.rerank.assert_not_called()

    @pytest.mark.asyncio
    async def test_concurrent_init_locks_once(self, tmp_path, monkeypatch):
        mod = self._install_mock_module(monkeypatch)
        r = Reranker(model="bge-reranker-v2-m3", cache_dir=tmp_path)
        await asyncio.gather(*[r.rerank("q", ["p"]) for _ in range(5)])
        # Init called exactly once across concurrent callers
        assert mod.rerank_init.call_count == 1

    @pytest.mark.asyncio
    async def test_prewarm_does_not_raise_on_failure(self, tmp_path, monkeypatch):
        # rerank raises but prewarm must NOT propagate
        mod = self._install_mock_module(monkeypatch,
            rerank_fn=MagicMock(side_effect=RuntimeError("model not ready")))
        r = Reranker(model="bge-reranker-v2-m3", cache_dir=tmp_path)
        await r.prewarm()  # must not raise

    @pytest.mark.asyncio
    async def test_import_error_raises_helpful_message(self, tmp_path, monkeypatch):
        # Simulate missing native module
        monkeypatch.setitem(sys.modules, "docgraph_embed", None)
        # Force ImportError on import attempt
        def fake_import(name, *args, **kwargs):
            if name == "docgraph_embed":
                raise ImportError("No module named 'docgraph_embed'")
            return __import__(name, *args, **kwargs)
        monkeypatch.setattr("builtins.__import__", fake_import)
        r = Reranker(model="bge-reranker-v2-m3", cache_dir=tmp_path)
        with pytest.raises(RuntimeError, match="maturin develop"):
            await r.rerank("q", ["p"])
```

- [ ] **Step 2: Run tests to verify FAIL**

```bash
poetry run pytest tests/embed/test_rerank.py -v
```

Expected: FAIL — `docgraph.embed.rerank` does not exist.

- [ ] **Step 3: Implement `Reranker` class**

Create `docgraph/embed/rerank.py`:

```python
from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


class Reranker:
    """Cross-encoder reranker via Rust (docgraph_embed::rerank)."""

    HEALTH_MSG = (
        "Reranker unavailable. Build the Rust crate: "
        "cd crates/docgraph-embed && maturin develop --release"
    )

    def __init__(self, model: str, cache_dir: Path) -> None:
        self._model = model
        self._cache_dir = cache_dir
        self._init_lock = asyncio.Lock()
        self._initialized = False

    def _import_native(self):
        try:
            import docgraph_embed
        except ImportError as exc:
            raise RuntimeError(self.HEALTH_MSG) from exc
        if docgraph_embed is None:
            raise RuntimeError(self.HEALTH_MSG)
        return docgraph_embed

    async def _ensure_init(self) -> None:
        if self._initialized:
            return
        async with self._init_lock:
            if self._initialized:
                return
            native = self._import_native()
            self._cache_dir.mkdir(parents=True, exist_ok=True)
            await asyncio.to_thread(
                native.rerank_init, self._model, str(self._cache_dir)
            )
            self._initialized = True
            logger.info(
                "Reranker ready: model=%s cache=%s", self._model, self._cache_dir
            )

    async def prewarm(self) -> None:
        """Force model load + dummy inference. Called from server lifespan startup.
        Swallows errors so server start cannot fail because of reranker."""
        try:
            await self._ensure_init()
            native = self._import_native()
            await asyncio.to_thread(native.rerank, "warmup", ["test passage"])
            logger.info("Reranker pre-warmed")
        except Exception as exc:
            logger.warning("Rerank pre-warm failed (will retry on first call): %s", exc)

    async def rerank(self, query: str, passages: list[str]) -> list[float]:
        if not passages:
            return []
        await self._ensure_init()
        native = self._import_native()
        scores = await asyncio.to_thread(native.rerank, query, passages)
        return [float(s) for s in scores]

    async def health_check(self) -> None:
        await self._ensure_init()
        native = self._import_native()
        await asyncio.to_thread(native.rerank_health_check)
```

- [ ] **Step 4: Run tests to verify PASS**

```bash
poetry run pytest tests/embed/test_rerank.py -v
```

Expected: All 5 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add docgraph/embed/rerank.py tests/embed/test_rerank.py
git commit -m "feat(embed): add Python Reranker wrapper for Rust native module"
```

---

## Task 11: FusedHit dataclass + RRF fusion

**Files:**
- Modify: `docgraph/mcp/search.py`
- Create: `tests/mcp/test_rrf_fusion.py`

- [ ] **Step 1: Write failing test for RRF math**

Create `tests/mcp/test_rrf_fusion.py`:

```python
from __future__ import annotations

import pytest

from docgraph.mcp.search import FusedHit, _rrf_fuse


def _vec(chunk_id, score=0.5, text="t"):
    return {
        "id": chunk_id,
        "text": text,
        "doc_id": chunk_id.rsplit("_", 1)[0],
        "filename": "f.md",
        "folder": "",
        "tags": [],
        "chunk_index": 0,
        "score": score,
        "source_page": None,
    }


def _sparse(chunk_id, bm25=2.0):
    return {
        "chunk_id": chunk_id,
        "doc_id": chunk_id.rsplit("_", 1)[0],
        "folder": "",
        "tags": [],
        "chunk_index": 0,
        "bm25_score": bm25,
    }


def test_rrf_single_branch_only_vector():
    vec = [_vec("a_0", 0.9), _vec("b_0", 0.8)]
    sparse = []
    out = _rrf_fuse(vec, sparse, k_rrf=60)
    assert out[0].chunk_id == "a_0"
    assert out[0].rrf_score == pytest.approx(1 / 61)
    assert out[1].chunk_id == "b_0"
    assert out[1].rrf_score == pytest.approx(1 / 62)
    assert out[0].vector_score == 0.9
    assert out[0].bm25_score is None


def test_rrf_single_branch_only_sparse():
    vec = []
    sparse = [_sparse("a_0", 5.0), _sparse("b_0", 4.0)]
    out = _rrf_fuse(vec, sparse, k_rrf=60)
    assert out[0].chunk_id == "a_0"
    assert out[0].rrf_score == pytest.approx(1 / 61)
    assert out[0].vector_score is None
    assert out[0].bm25_score == 5.0


def test_rrf_consensus_double_counts():
    vec = [_vec("a_0", 0.9), _vec("b_0", 0.8)]
    sparse = [_sparse("a_0", 5.0), _sparse("c_0", 3.0)]
    out = _rrf_fuse(vec, sparse, k_rrf=60)
    # 'a_0' is rank 0 in both — gets both reciprocals
    assert out[0].chunk_id == "a_0"
    assert out[0].rrf_score == pytest.approx(2 / 61)
    # Both 'b_0' (vec rank 1) and 'c_0' (sparse rank 1) get 1/62
    by_id = {h.chunk_id: h for h in out}
    assert by_id["b_0"].rrf_score == pytest.approx(1 / 62)
    assert by_id["c_0"].rrf_score == pytest.approx(1 / 62)


def test_rrf_score_propagation_preserves_branch_scores():
    vec = [_vec("a_0", 0.9, text="vector text")]
    sparse = [_sparse("a_0", 5.0)]
    out = _rrf_fuse(vec, sparse, k_rrf=60)
    assert out[0].vector_score == 0.9
    assert out[0].bm25_score == 5.0
    assert out[0].text == "vector text"


def test_rrf_sorted_desc_by_score():
    vec = [_vec(f"c_{i}", 0.9 - i * 0.1) for i in range(5)]
    out = _rrf_fuse(vec, [], k_rrf=60)
    scores = [h.rrf_score for h in out]
    assert scores == sorted(scores, reverse=True)
```

- [ ] **Step 2: Run tests to verify FAIL**

```bash
poetry run pytest tests/mcp/test_rrf_fusion.py -v
```

Expected: All FAIL — `FusedHit` and `_rrf_fuse` not defined.

- [ ] **Step 3: Add `FusedHit` and `_rrf_fuse` to `docgraph/mcp/search.py`**

Replace `docgraph/mcp/search.py` content (we'll progressively add to it; this step adds fusion data types):

```python
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Optional

from docgraph.config import Config
from docgraph.embed.provider import EmbeddingProvider
from docgraph.models import SearchResult
from docgraph.store.chroma import ChromaStore
from docgraph.store.fts import FtsStore
from docgraph.store.sqlite import SQLiteStore

logger = logging.getLogger(__name__)


@dataclass
class FusedHit:
    """A chunk that surfaced from one or both branches, with fused score."""
    chunk_id: str
    text: str
    doc_id: str
    filename: str
    folder: str
    tags: list[str]
    chunk_index: int
    source_page: Optional[int]
    vector_score: Optional[float]
    bm25_score: Optional[float]
    rrf_score: float
    rerank_score: Optional[float] = None


def _rrf_fuse(
    vector_results: list[dict[str, Any]],
    sparse_results: list[dict[str, Any]],
    k_rrf: int = 60,
) -> list[FusedHit]:
    """Reciprocal Rank Fusion. Sorts vector then sparse, adds 1/(k+rank+1) per
    branch occurrence. Returns hits sorted by descending rrf_score."""
    fused: dict[str, FusedHit] = {}
    for rank, hit in enumerate(vector_results):
        cid = hit["id"]
        fh = FusedHit(
            chunk_id=cid,
            text=hit.get("text", ""),
            doc_id=hit.get("doc_id", ""),
            filename=hit.get("filename", ""),
            folder=hit.get("folder", ""),
            tags=list(hit.get("tags") or []),
            chunk_index=int(hit.get("chunk_index", 0)),
            source_page=hit.get("source_page"),
            vector_score=float(hit["score"]) if "score" in hit else None,
            bm25_score=None,
            rrf_score=0.0,
        )
        fused[cid] = fh
        fh.rrf_score += 1.0 / (k_rrf + rank + 1)

    for rank, hit in enumerate(sparse_results):
        cid = hit["chunk_id"]
        if cid in fused:
            fh = fused[cid]
            fh.bm25_score = float(hit["bm25_score"])
        else:
            fh = FusedHit(
                chunk_id=cid,
                text="",  # sparse path doesn't carry text; will be filled later
                doc_id=hit.get("doc_id", ""),
                filename="",
                folder=hit.get("folder", ""),
                tags=list(hit.get("tags") or []),
                chunk_index=int(hit.get("chunk_index", 0)),
                source_page=None,
                vector_score=None,
                bm25_score=float(hit["bm25_score"]),
                rrf_score=0.0,
            )
            fused[cid] = fh
        fh.rrf_score += 1.0 / (k_rrf + rank + 1)

    return sorted(fused.values(), key=lambda h: -h.rrf_score)
```

- [ ] **Step 4: Run tests to verify PASS**

```bash
poetry run pytest tests/mcp/test_rrf_fusion.py -v
```

Expected: All 5 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add docgraph/mcp/search.py tests/mcp/test_rrf_fusion.py
git commit -m "feat(search): add FusedHit + RRF fusion"
```

---

## Task 12: Auto-rerank gate logic

**Files:**
- Modify: `docgraph/mcp/search.py`
- Create: `tests/mcp/test_rerank_gate.py`

- [ ] **Step 1: Write failing tests for gate**

Create `tests/mcp/test_rerank_gate.py`:

```python
from __future__ import annotations

from pathlib import Path

import pytest

from docgraph.config import Config
from docgraph.mcp.search import FusedHit, _should_rerank


def _hit(rrf=0.03, has_vector=True, has_sparse=True, chunk_id="x"):
    return FusedHit(
        chunk_id=chunk_id,
        text="",
        doc_id="d",
        filename="",
        folder="",
        tags=[],
        chunk_index=0,
        source_page=None,
        vector_score=0.5 if has_vector else None,
        bm25_score=2.0 if has_sparse else None,
        rrf_score=rrf,
    )


@pytest.fixture
def cfg(tmp_path):
    c = Config(data_dir=tmp_path)
    c.rerank_enabled = True
    c.rerank_min_floor = 0.015
    c.rerank_score_gap_ratio = 0.5
    return c


class TestShouldRerank:
    def test_skip_when_rerank_disabled(self, cfg):
        cfg.rerank_enabled = False
        assert _should_rerank(cfg, [_hit(0.03), _hit(0.02)], k=5) is False

    def test_skip_when_zero_candidates(self, cfg):
        assert _should_rerank(cfg, [], k=5) is False

    def test_skip_when_one_candidate(self, cfg):
        assert _should_rerank(cfg, [_hit(0.03)], k=5) is False

    def test_skip_when_top1_below_floor(self, cfg):
        cands = [_hit(rrf=0.010, chunk_id="a"), _hit(rrf=0.008, chunk_id="b")]
        assert _should_rerank(cfg, cands, k=5) is False

    def test_force_rerank_when_only_vector_branch(self, cfg):
        cands = [_hit(rrf=0.03, has_sparse=False, chunk_id=f"c{i}") for i in range(3)]
        assert _should_rerank(cfg, cands, k=5) is True

    def test_force_rerank_when_only_sparse_branch(self, cfg):
        cands = [_hit(rrf=0.03, has_vector=False, chunk_id=f"c{i}") for i in range(3)]
        assert _should_rerank(cfg, cands, k=5) is True

    def test_skip_when_gap_clear(self, cfg):
        # gap_ratio = (0.030 - 0.005) / 0.030 = 0.833 > 0.5 → skip
        cands = [
            _hit(rrf=0.030, chunk_id="a"),
            _hit(rrf=0.020, chunk_id="b"),
            _hit(rrf=0.015, chunk_id="c"),
            _hit(rrf=0.010, chunk_id="d"),
            _hit(rrf=0.005, chunk_id="e"),
        ]
        assert _should_rerank(cfg, cands, k=5) is False

    def test_rerank_when_gap_small(self, cfg):
        # gap_ratio = (0.030 - 0.025) / 0.030 = 0.166 < 0.5 → rerank
        cands = [_hit(rrf=0.030 - i * 0.001, chunk_id=f"c{i}") for i in range(5)]
        assert _should_rerank(cfg, cands, k=5) is True

    def test_floor_check_fires_before_zero_division(self, cfg):
        # top1 = 0 → floor check (0 < 0.015) skips before gap-ratio division
        cands = [_hit(rrf=0.0, chunk_id="a"), _hit(rrf=0.0, chunk_id="b")]
        assert _should_rerank(cfg, cands, k=5) is False
```

- [ ] **Step 2: Run tests to verify FAIL**

```bash
poetry run pytest tests/mcp/test_rerank_gate.py -v
```

Expected: FAIL — `_should_rerank` undefined.

- [ ] **Step 3: Add `_should_rerank` to `docgraph/mcp/search.py`**

Append to `docgraph/mcp/search.py`:

```python
def _should_rerank(cfg: Config, fused: list[FusedHit], k: int) -> tuple[bool, str]:
    """Return (decision, reason) for observability."""
    if not cfg.rerank_enabled:
        return False, "skip_disabled"
    if len(fused) < 2:
        return False, "skip_too_few_candidates"

    top1 = fused[0].rrf_score

    # A. Floor — top-1 below recall floor; rerank can't fix bad recall
    if top1 < cfg.rerank_min_floor:
        return False, "skip_floor"

    # B. Single-branch override — only one signal fired; force rerank
    window = fused[: max(k, 2)]
    has_vector = any(h.vector_score is not None for h in window)
    has_sparse = any(h.bm25_score is not None for h in window)
    if not (has_vector and has_sparse):
        return True, "force_single_branch"

    # Default: rerank when top-k scores are clustered (ambiguous)
    top_window = fused[:k] if len(fused) >= k else fused
    top_scores = [h.rrf_score for h in top_window]
    gap = top_scores[0] - top_scores[-1]
    if top1 > 0 and gap / top1 > cfg.rerank_score_gap_ratio:
        return False, "skip_gap"
    return True, "force_ambiguous"
```

Note: the test imports `_should_rerank` expecting `-> bool`. We changed signature to return tuple. Update test to handle this:

In `tests/mcp/test_rerank_gate.py`, modify every assert from `_should_rerank(...) is True/False` to `_should_rerank(...)[0] is True/False`. Apply this find/replace across the file:

```python
# OLD
assert _should_rerank(cfg, [...], k=5) is False
# NEW
assert _should_rerank(cfg, [...], k=5)[0] is False
```

- [ ] **Step 4: Run tests to verify PASS**

```bash
poetry run pytest tests/mcp/test_rerank_gate.py -v
```

Expected: All 9 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add docgraph/mcp/search.py tests/mcp/test_rerank_gate.py
git commit -m "feat(search): add auto-rerank gate with floor + single-branch override"
```

---

## Task 13: SearchService hybrid pipeline

**Files:**
- Modify: `docgraph/mcp/search.py`
- Create: `tests/mcp/test_hybrid_search.py`

- [ ] **Step 1: Write failing tests**

Create `tests/mcp/test_hybrid_search.py`:

```python
from __future__ import annotations

import asyncio
from dataclasses import dataclass

import pytest

from docgraph.config import Config
from docgraph.mcp.search import SearchService
from docgraph.models import DocumentRecord
from docgraph.store import ChromaStore, FtsStore, SQLiteStore


class FakeEmbedder:
    async def embed(self, texts, for_query=False):
        return [[0.1] * 768 for _ in texts]

    async def health_check(self):
        return None


class FakeReranker:
    def __init__(self, score_map=None):
        self.score_map = score_map or {}
        self.call_count = 0

    async def rerank(self, query, passages):
        self.call_count += 1
        return [self.score_map.get(p, 0.5) for p in passages]

    async def prewarm(self):
        return None


class BrokenReranker:
    async def rerank(self, query, passages):
        raise RuntimeError("rerank failed")

    async def prewarm(self):
        return None


class SlowReranker:
    def __init__(self, delay):
        self.delay = delay

    async def rerank(self, query, passages):
        await asyncio.sleep(self.delay)
        return [0.0] * len(passages)

    async def prewarm(self):
        return None


@pytest.fixture
def seeded(tmp_path):
    cfg = Config(data_dir=tmp_path)
    cfg.hybrid_enabled = True
    cfg.rerank_enabled = True
    cfg.rerank_timeout_sec = 1.0
    cfg.ensure_dirs()
    sqlite = SQLiteStore(cfg)
    sqlite.init_schema()
    chroma = ChromaStore(cfg)
    fts = FtsStore(cfg)
    # Seed
    docs = [
        ("doc_a", "DocGraph stores vectors in ChromaDB"),
        ("doc_b", "User interface built with React and Vite"),
        ("doc_c", "Indexer converts files to markdown"),
    ]
    for did, text in docs:
        sqlite.insert_document(DocumentRecord(id=did, filename=f"{did}.md", folder="x"))
        chroma.upsert_chunks([{
            "id": f"{did}_0", "embedding": [0.1] * 768, "text": text,
            "metadata": {"doc_id": did, "filename": f"{did}.md", "folder": "x",
                         "tags": "[]", "chunk_index": 0},
        }])
        fts.upsert_chunks([{
            "chunk_id": f"{did}_0", "doc_id": did, "folder": "x", "tags": "[]",
            "chunk_index": 0, "text": text, "filename": f"{did}.md",
        }])
    return cfg, sqlite, chroma, fts


class TestHybridPipeline:
    @pytest.mark.asyncio
    async def test_hybrid_returns_results(self, seeded):
        cfg, sqlite, chroma, fts = seeded
        svc = SearchService(cfg, sqlite, chroma, FakeEmbedder(), fts=fts, reranker=None)
        results = await svc.search("DocGraph")
        assert len(results) > 0
        assert any("DocGraph" in r.text or "vectors" in r.text for r in results)

    @pytest.mark.asyncio
    async def test_disable_hybrid_skips_fts(self, seeded, monkeypatch):
        cfg, sqlite, chroma, fts = seeded
        cfg.hybrid_enabled = False
        call_count = {"n": 0}
        orig_search = fts.search
        def tracked(*args, **kwargs):
            call_count["n"] += 1
            return orig_search(*args, **kwargs)
        monkeypatch.setattr(fts, "search", tracked)
        svc = SearchService(cfg, sqlite, chroma, FakeEmbedder(), fts=fts, reranker=None)
        await svc.search("test")
        assert call_count["n"] == 0

    @pytest.mark.asyncio
    async def test_rerank_invoked_when_ambiguous(self, seeded):
        cfg, sqlite, chroma, fts = seeded
        cfg.rerank_score_gap_ratio = 0.99  # almost always rerank
        reranker = FakeReranker()
        svc = SearchService(cfg, sqlite, chroma, FakeEmbedder(), fts=fts, reranker=reranker)
        await svc.search("DocGraph React")
        assert reranker.call_count == 1

    @pytest.mark.asyncio
    async def test_rerank_failure_falls_back_to_rrf(self, seeded):
        cfg, sqlite, chroma, fts = seeded
        cfg.rerank_score_gap_ratio = 0.99
        svc = SearchService(
            cfg, sqlite, chroma, FakeEmbedder(), fts=fts, reranker=BrokenReranker()
        )
        results = await svc.search("DocGraph React")
        assert len(results) > 0  # no exception propagated

    @pytest.mark.asyncio
    async def test_rerank_timeout_falls_back_to_rrf(self, seeded):
        cfg, sqlite, chroma, fts = seeded
        cfg.rerank_score_gap_ratio = 0.99
        cfg.rerank_timeout_sec = 0.1
        svc = SearchService(
            cfg, sqlite, chroma, FakeEmbedder(), fts=fts, reranker=SlowReranker(delay=2.0)
        )
        import time
        start = time.time()
        results = await svc.search("DocGraph React")
        elapsed = time.time() - start
        assert elapsed < 1.0  # bounded by timeout
        assert len(results) > 0
```

- [ ] **Step 2: Run tests to verify FAIL**

```bash
poetry run pytest tests/mcp/test_hybrid_search.py -v
```

Expected: All FAIL — `SearchService` constructor doesn't accept `fts`/`reranker`.

- [ ] **Step 3: Rewrite `SearchService.search` with hybrid pipeline**

Replace the existing `SearchService` class in `docgraph/mcp/search.py` with:

```python
import asyncio


class SearchService:
    def __init__(
        self,
        cfg: Config,
        sqlite: SQLiteStore,
        chroma: ChromaStore,
        embedder: EmbeddingProvider,
        fts: Optional[FtsStore] = None,
        reranker=None,
    ) -> None:
        self._cfg = cfg
        self._sqlite = sqlite
        self._chroma = chroma
        self._embedder = embedder
        self._fts = fts
        self._reranker = reranker

    async def search(
        self,
        query: str,
        top_k: Optional[int] = None,
        folder: Optional[str] = None,
        tags: Optional[list[str]] = None,
    ) -> list[SearchResult]:
        k = top_k or self._cfg.default_top_k
        # Cap top_k to a sane upper bound to prevent OOM on huge requests
        k = min(k, 100)
        overfetch = max(k * 6, 30)

        use_hybrid = self._cfg.hybrid_enabled and self._fts is not None

        # 1) Embed query (always needed for vector branch)
        vectors = await self._embedder.embed([query], for_query=True)
        query_vec = vectors[0]

        # 2) Parallel branches
        vector_task = asyncio.create_task(
            asyncio.to_thread(
                self._chroma.search, query_vec, overfetch, folder, tags
            )
        )
        if use_hybrid:
            sparse_task = asyncio.create_task(
                asyncio.to_thread(self._fts.search, query, overfetch, folder, tags)
            )
            vector_results, sparse_results = await asyncio.gather(
                vector_task, sparse_task
            )
        else:
            vector_results = await vector_task
            sparse_results = []

        # 3) Fuse with RRF
        fused = _rrf_fuse(vector_results, sparse_results, k_rrf=self._cfg.rrf_k)
        fused = fused[: max(k * 3, self._cfg.rerank_top_n)]

        # Backfill text for sparse-only hits (text was not carried)
        chunk_to_text: dict[str, str] = {
            h["id"]: h.get("text", "") for h in vector_results
        }
        for fh in fused:
            if not fh.text and fh.chunk_id in chunk_to_text:
                fh.text = chunk_to_text[fh.chunk_id]
        # Some sparse-only hits may have no text yet — pull from chroma metadata as last resort
        # (Best-effort; if text is empty, downstream caller still gets metadata.)

        # 4) Gate + rerank
        decision, reason = _should_rerank(self._cfg, fused, k)
        rerank_ran = False
        if decision and self._reranker is not None:
            rerank_window = fused[: self._cfg.rerank_top_n]
            try:
                scores = await asyncio.wait_for(
                    self._reranker.rerank(query, [h.text for h in rerank_window]),
                    timeout=self._cfg.rerank_timeout_sec,
                )
                for h, s in zip(rerank_window, scores):
                    h.rerank_score = float(s)
                rerank_window.sort(key=lambda h: -(h.rerank_score or 0.0))
                # Replace head with reranked window
                fused = rerank_window + fused[self._cfg.rerank_top_n :]
                rerank_ran = True
            except Exception as exc:
                logger.warning(
                    "Rerank failed or timed out; falling back to RRF order: %s", exc
                )

        # 5) Filter by min_score (only when no rerank score)
        filtered: list[FusedHit] = []
        for h in fused:
            if h.rerank_score is not None:
                filtered.append(h)
            elif h.vector_score is None or h.vector_score >= self._cfg.min_score:
                filtered.append(h)

        # 6) Emit metrics log
        logger.info(
            "search_metrics",
            extra={
                "query_len": len(query),
                "vector_branch_size": len(vector_results),
                "sparse_branch_size": len(sparse_results),
                "fused_size": len(fused),
                "rerank_triggered": rerank_ran,
                "rerank_reason": reason,
                "k": k,
            },
        )

        return [self._to_result(h) for h in filtered[:k]]

    @staticmethod
    def _to_result(h: FusedHit) -> SearchResult:
        # Score priority: rerank > vector > normalized bm25
        if h.rerank_score is not None:
            score = float(h.rerank_score)
        elif h.vector_score is not None:
            score = float(h.vector_score)
        else:
            # bm25 is unbounded; map roughly to [0, 1) via 1 - 1/(1+x)
            score = 1.0 - 1.0 / (1.0 + max(float(h.bm25_score or 0.0), 0.0))
        return SearchResult(
            text=h.text,
            doc_id=h.doc_id,
            filename=h.filename,
            folder=h.folder,
            tags=h.tags,
            chunk_index=h.chunk_index,
            score=score,
            source_page=h.source_page,
        )
```

- [ ] **Step 4: Run tests to verify PASS**

```bash
poetry run pytest tests/mcp/test_hybrid_search.py -v
```

Expected: All 5 tests PASS.

- [ ] **Step 5: Verify pre-existing search tests still pass**

```bash
poetry run pytest tests/mcp/ -v
```

Expected: All MCP tests (old + new) PASS.

- [ ] **Step 6: Commit**

```bash
git add docgraph/mcp/search.py tests/mcp/test_hybrid_search.py
git commit -m "feat(search): hybrid pipeline with RRF + auto-rerank gate"
```

---

## Task 14: Wire FtsStore + Reranker into AppState

**Files:**
- Modify: `docgraph/web/deps.py`
- Modify: `docgraph/embed/factory.py` (add reranker factory)
- Modify: `tests/web/test_deps.py` (extend or create)

- [ ] **Step 1: Write failing test**

Create or extend `tests/web/test_deps.py`:

```python
from __future__ import annotations

from docgraph.config import Config
from docgraph.web.deps import AppState


def test_appstate_includes_fts(tmp_path):
    cfg = Config(data_dir=tmp_path)
    cfg.hybrid_enabled = True
    state = AppState.create(cfg)
    assert state.fts is not None
    assert state.fts.count_chunks() == 0


def test_appstate_fts_none_when_hybrid_disabled(tmp_path):
    cfg = Config(data_dir=tmp_path)
    cfg.hybrid_enabled = False
    state = AppState.create(cfg)
    assert state.fts is None


def test_appstate_reranker_none_when_disabled(tmp_path):
    cfg = Config(data_dir=tmp_path)
    cfg.rerank_enabled = False
    state = AppState.create(cfg)
    assert state.reranker is None


def test_appstate_reranker_created_when_enabled(tmp_path):
    cfg = Config(data_dir=tmp_path)
    cfg.rerank_enabled = True
    state = AppState.create(cfg)
    assert state.reranker is not None
```

- [ ] **Step 2: Run tests to verify FAIL**

```bash
poetry run pytest tests/web/test_deps.py -v
```

Expected: All FAIL — `state.fts` / `state.reranker` don't exist.

- [ ] **Step 3: Add reranker factory**

Add to `docgraph/embed/factory.py` (append, do not modify existing):

```python
def create_reranker(cfg):
    """Create a Reranker or return None if disabled."""
    from docgraph.embed.rerank import Reranker
    if not cfg.rerank_enabled:
        return None
    return Reranker(model=cfg.rerank_model, cache_dir=cfg.local_model_dir)
```

- [ ] **Step 4: Extend `AppState`**

Replace `docgraph/web/deps.py` content with:

```python
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from docgraph.config import Config
from docgraph.embed.factory import create_embedder, create_reranker
from docgraph.embed.provider import EmbeddingProvider
from docgraph.ingest.indexer import Indexer
from docgraph.mcp.search import SearchService
from docgraph.store import ChromaStore, FileStore, FtsStore, SQLiteStore


@dataclass
class AppState:
    cfg: Config
    sqlite: SQLiteStore
    files: FileStore
    chroma: ChromaStore
    embedder: EmbeddingProvider
    fts: Optional[FtsStore]
    reranker: object  # Reranker | None — avoids import cycle

    @classmethod
    def create(cls, cfg: Config) -> "AppState":
        cfg.ensure_dirs()
        sqlite = SQLiteStore(cfg)
        sqlite.init_schema()
        fts = FtsStore(cfg) if cfg.hybrid_enabled else None
        return cls(
            cfg=cfg,
            sqlite=sqlite,
            files=FileStore(cfg),
            chroma=ChromaStore(cfg),
            embedder=create_embedder(cfg),
            fts=fts,
            reranker=create_reranker(cfg),
        )

    def indexer(self) -> Indexer:
        return Indexer(
            self.cfg, self.sqlite, self.files, self.chroma, self.embedder, fts=self.fts
        )

    def search_service(self) -> SearchService:
        return SearchService(
            self.cfg, self.sqlite, self.chroma, self.embedder,
            fts=self.fts, reranker=self.reranker,
        )
```

- [ ] **Step 5: Run tests to verify PASS**

```bash
poetry run pytest tests/web/test_deps.py -v
```

Expected: All 4 tests PASS.

- [ ] **Step 6: Verify SearchService callers use new factory**

```bash
grep -rn "SearchService(" docgraph/ | grep -v "deps.py"
```

For each caller (likely in `docgraph/mcp/server.py`), replace direct `SearchService(...)` construction with `state.search_service()`.

- [ ] **Step 7: Commit**

```bash
git add docgraph/embed/factory.py docgraph/web/deps.py tests/web/test_deps.py docgraph/mcp/server.py
git commit -m "feat(deps): wire FtsStore + Reranker into AppState"
```

---

## Task 15: rebuild-fts CLI subcommand + auto-rebuild on startup

**Files:**
- Modify: `docgraph/cli.py`
- Create: `tests/cli/__init__.py` (empty)
- Create: `tests/cli/test_rebuild_fts.py`

- [ ] **Step 1: Write failing test**

Create `tests/cli/__init__.py` as empty file.

Create `tests/cli/test_rebuild_fts.py`:

```python
from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from docgraph.config import Config
from docgraph.models import DocumentRecord
from docgraph.store import ChromaStore, FtsStore, SQLiteStore


def _seed_chroma(cfg, n):
    sqlite = SQLiteStore(cfg)
    sqlite.init_schema()
    chroma = ChromaStore(cfg)
    for i in range(n):
        sqlite.insert_document(DocumentRecord(
            id=f"doc_{i}", filename=f"f{i}.md", folder="x", tags=["t"]
        ))
        chroma.upsert_chunks([{
            "id": f"doc_{i}_0", "embedding": [0.1] * 768, "text": f"text {i}",
            "metadata": {"doc_id": f"doc_{i}", "filename": f"f{i}.md",
                         "folder": "x", "tags": '["t"]', "chunk_index": 0},
        }])


def test_rebuild_fts_populates_index(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("DOCGRAPH_DATA_DIR", str(tmp_path))
    cfg = Config(data_dir=tmp_path)
    cfg.ensure_dirs()
    _seed_chroma(cfg, n=3)
    from docgraph.cli import cmd_rebuild_fts
    import argparse
    cmd_rebuild_fts(argparse.Namespace())
    fts = FtsStore(cfg)
    assert fts.count_chunks() == 3
    out = capsys.readouterr().out
    assert "3 chunks" in out or "Done" in out


def test_rebuild_fts_empty_chroma(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("DOCGRAPH_DATA_DIR", str(tmp_path))
    cfg = Config(data_dir=tmp_path)
    cfg.ensure_dirs()
    sqlite = SQLiteStore(cfg)
    sqlite.init_schema()
    ChromaStore(cfg)
    from docgraph.cli import cmd_rebuild_fts
    import argparse
    cmd_rebuild_fts(argparse.Namespace())
    out = capsys.readouterr().out
    assert "Nothing to rebuild" in out
```

- [ ] **Step 2: Run tests to verify FAIL**

```bash
poetry run pytest tests/cli/test_rebuild_fts.py -v
```

Expected: FAIL — `cmd_rebuild_fts` not defined.

- [ ] **Step 3: Add `cmd_rebuild_fts` and CLI subcommand**

Modify `docgraph/cli.py`:

Add at top of file:

```python
def cmd_rebuild_fts(args) -> None:
    cfg = load_config()
    cfg.ensure_dirs()
    from docgraph.store import ChromaStore, FtsStore, SQLiteStore
    sqlite = SQLiteStore(cfg)
    sqlite.init_schema()
    chroma = ChromaStore(cfg)
    fts = FtsStore(cfg)
    print("Counting chunks...")
    n = chroma.count_chunks()
    print(f"Chroma has {n} chunks")
    if n == 0:
        print("Nothing to rebuild.")
        return
    print("Clearing existing FTS index...")
    fts.clear()
    print(f"Rebuilding FTS index ({n} chunks)...")

    def progress(done, total):
        print(f"  {done}/{total}")

    asyncio.run(fts.rebuild_from_chroma(chroma, sqlite, progress_callback=progress))
    print(f"Done. FTS index now has {fts.count_chunks()} chunks.")
```

Modify `main()` to register the subcommand:

```python
def main() -> None:
    parser = argparse.ArgumentParser(prog="docgraph")
    sub = parser.add_subparsers(dest="command")
    serve_parser = sub.add_parser("serve", help="Start Web UI + MCP server")
    serve_parser.add_argument(
        "--stdio", action="store_true",
        help="Use MCP stdio instead of HTTP SSE",
    )
    sub.add_parser("rebuild-fts", help="Rebuild FTS index from Chroma (one-shot)")
    args = parser.parse_args()
    if args.command == "serve":
        _run_serve(stdio=args.stdio)
        return
    if args.command == "rebuild-fts":
        cmd_rebuild_fts(args)
        return
    parser.print_help()
    sys.exit(1)
```

- [ ] **Step 4: Run tests to verify PASS**

```bash
poetry run pytest tests/cli/test_rebuild_fts.py -v
```

Expected: Both tests PASS.

- [ ] **Step 5: Commit**

```bash
git add docgraph/cli.py tests/cli/
git commit -m "feat(cli): add rebuild-fts subcommand"
```

---

## Task 16: Auto-rebuild + pre-warm in server lifespan

**Files:**
- Modify: `docgraph/web/app.py`
- Modify: `docgraph/cli.py`

- [ ] **Step 1: Add lifespan startup hook to FastAPI app**

In `docgraph/web/app.py`, find `create_app` and add a lifespan handler. If the file already has one, extend it. Add:

```python
from contextlib import asynccontextmanager


@asynccontextmanager
async def _lifespan(app):
    state = app.state.app_state  # set by create_app
    cfg = state.cfg
    # Pre-warm reranker in background
    if cfg.rerank_enabled and cfg.rerank_prewarm and state.reranker is not None:
        asyncio.create_task(state.reranker.prewarm())
    # Auto-rebuild FTS if mismatched
    if cfg.hybrid_enabled and state.fts is not None:
        chroma_n = state.chroma.count_chunks()
        fts_n = state.fts.count_chunks()
        if fts_n == 0 and chroma_n > 0:
            logger.warning(
                "FTS index empty but Chroma has %d chunks. Auto-rebuilding in background.",
                chroma_n,
            )
            asyncio.create_task(
                state.fts.rebuild_from_chroma(state.chroma, state.sqlite)
            )
        elif chroma_n > 0 and abs(chroma_n - fts_n) > max(10, chroma_n * 0.05):
            logger.warning(
                "FTS index out of sync (chroma=%d, fts=%d). "
                "Run `docgraph rebuild-fts` to fix.",
                chroma_n, fts_n,
            )
    yield


def create_app(cfg, state=None, mount_mcp=True):
    if state is None:
        state = AppState.create(cfg)
    app = FastAPI(lifespan=_lifespan)
    app.state.app_state = state
    # ... rest of existing setup ...
```

Add `import asyncio` and `logger = logging.getLogger(__name__)` if not already present.

- [ ] **Step 2: Smoke test the app starts without error**

```bash
poetry run python -c "
from docgraph.config import load_config
from docgraph.web.app import create_app
cfg = load_config()
app = create_app(cfg, mount_mcp=False)
print('App created with lifespan:', app.router.lifespan_context)
"
```

Expected: prints app + lifespan_context object. No exceptions.

- [ ] **Step 3: Commit**

```bash
git add docgraph/web/app.py
git commit -m "feat(web): pre-warm reranker + auto-rebuild FTS in lifespan"
```

---

## Task 17: Extend /api/health

**Files:**
- Modify: `docgraph/web/app.py`
- Modify: `tests/web/test_app.py` (extend existing or create)

- [ ] **Step 1: Write failing test**

Append to `tests/web/test_app.py`:

```python
import pytest
from fastapi.testclient import TestClient

from docgraph.config import Config
from docgraph.web.app import create_app
from docgraph.web.deps import AppState


def test_health_reports_hybrid_status(tmp_path):
    cfg = Config(data_dir=tmp_path)
    cfg.hybrid_enabled = True
    cfg.rerank_enabled = False
    state = AppState.create(cfg)
    app = create_app(cfg, state=state, mount_mcp=False)
    with TestClient(app) as client:
        resp = client.get("/api/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["hybrid_enabled"] is True
    assert body["rerank_enabled"] is False
    assert body["rerank_status"] == "disabled"
    assert "fts_chunks" in body
    assert "chroma_chunks" in body
    assert "fts_in_sync" in body


def test_health_reports_rerank_status_ready_after_init(tmp_path):
    cfg = Config(data_dir=tmp_path)
    cfg.rerank_enabled = True
    cfg.rerank_prewarm = False
    state = AppState.create(cfg)
    # Simulate reranker initialized
    state.reranker._initialized = True
    app = create_app(cfg, state=state, mount_mcp=False)
    with TestClient(app) as client:
        resp = client.get("/api/health")
    assert resp.json()["rerank_status"] == "ready"
```

- [ ] **Step 2: Run test to verify FAIL**

```bash
poetry run pytest tests/web/test_app.py::test_health_reports_hybrid_status -v
```

Expected: FAIL — health response missing new fields.

- [ ] **Step 3: Extend `/api/health` endpoint**

In `docgraph/web/app.py`, find the existing `/api/health` route and extend it:

```python
@app.get("/api/health")
async def health():
    state: AppState = app.state.app_state
    cfg = state.cfg
    # Reranker status
    if not cfg.rerank_enabled:
        rerank_status = "disabled"
    elif state.reranker is None:
        rerank_status = "disabled"
    elif getattr(state.reranker, "_initialized", False):
        rerank_status = "ready"
    else:
        rerank_status = "loading"
    # FTS sync
    chroma_n = state.chroma.count_chunks()
    fts_n = state.fts.count_chunks() if state.fts else None
    fts_in_sync = (
        None if fts_n is None
        else abs(chroma_n - fts_n) <= max(10, chroma_n * 0.05)
    )
    return {
        "embedding_provider": cfg.embed_provider,
        "mcp_sse_url": f"http://{cfg.web_host}:{cfg.web_port}/mcp/sse",
        "hybrid_enabled": cfg.hybrid_enabled,
        "rerank_enabled": cfg.rerank_enabled,
        "rerank_status": rerank_status,
        "chroma_chunks": chroma_n,
        "fts_chunks": fts_n,
        "fts_in_sync": fts_in_sync,
    }
```

- [ ] **Step 4: Run tests to verify PASS**

```bash
poetry run pytest tests/web/test_app.py -v
```

Expected: All PASS.

- [ ] **Step 5: Commit**

```bash
git add docgraph/web/app.py tests/web/test_app.py
git commit -m "feat(web): extend /api/health with hybrid + rerank status"
```

---

## Task 18: Security tests — FTS injection

**Files:**
- Create: `tests/security/__init__.py`
- Create: `tests/security/test_fts_injection.py`

- [ ] **Step 1: Create empty package init**

Create `tests/security/__init__.py` as empty file.

- [ ] **Step 2: Write tests**

Create `tests/security/test_fts_injection.py`:

```python
from __future__ import annotations

import pytest

from docgraph.config import Config
from docgraph.store.fts import FtsStore
from docgraph.store.sqlite import SQLiteStore


@pytest.fixture
def fts(tmp_path):
    cfg = Config(data_dir=tmp_path)
    cfg.ensure_dirs()
    sqlite = SQLiteStore(cfg)
    sqlite.init_schema()
    f = FtsStore(cfg)
    f.upsert_chunks([
        {"chunk_id": "doc_a_0", "doc_id": "doc_a", "folder": "x", "tags": "[]",
         "chunk_index": 0, "text": "normal content here", "filename": "a.md"},
        {"chunk_id": "doc_b_0", "doc_id": "doc_b", "folder": "y", "tags": "[]",
         "chunk_index": 0, "text": "more normal content", "filename": "b.md"},
    ])
    return f


@pytest.mark.parametrize("evil", [
    "'; DROP TABLE chunks_fts; --",
    "' OR '1'='1",
    "'); DELETE FROM documents; --",
    "\\'; SELECT * FROM sqlite_master; --",
    "'; ATTACH DATABASE '/tmp/x.db' AS evil; --",
    "x' UNION SELECT * FROM sqlite_master; --",
])
def test_sql_injection_in_query_does_not_corrupt_data(fts, evil):
    # Must not raise and must not delete data
    hits = fts.search(evil, top_k=10)
    assert isinstance(hits, list)
    assert fts.count_chunks() == 2


@pytest.mark.parametrize("syntactic", [
    "*", "(", ")", '"', ":", "^", "+", "-",
    "AND", "OR", "NOT", "NEAR", "MATCH",
    '"unclosed', '"foo" AND OR',
    "a" * 10000,
    "\x00\x01\x02",
])
def test_fts5_special_chars_do_not_crash(fts, syntactic):
    hits = fts.search(syntactic, top_k=10)
    assert isinstance(hits, list)


def test_folder_filter_uses_parameter_binding(fts):
    evil = "x'; DROP TABLE chunks_fts; --"
    hits = fts.search("normal", top_k=10, folder=evil)
    assert hits == []
    assert fts.count_chunks() == 2
```

- [ ] **Step 3: Run tests to verify PASS (sanitizer should already protect)**

```bash
poetry run pytest tests/security/test_fts_injection.py -v
```

Expected: All PASS (Task 5's sanitizer + parameter binding cover all cases).

- [ ] **Step 4: Commit**

```bash
git add tests/security/__init__.py tests/security/test_fts_injection.py
git commit -m "test(security): SQL injection + FTS5 special char fuzzing"
```

---

## Task 19: Security tests — search isolation across folders/tags

**Files:**
- Create: `tests/security/test_search_isolation.py`

- [ ] **Step 1: Write tests**

Create `tests/security/test_search_isolation.py`:

```python
from __future__ import annotations

import pytest

from docgraph.config import Config
from docgraph.mcp.search import SearchService
from docgraph.models import DocumentRecord
from docgraph.store import ChromaStore, FtsStore, SQLiteStore


class FakeEmbedder:
    async def embed(self, texts, for_query=False):
        return [[0.1] * 768 for _ in texts]


def _seed_two_folders(cfg):
    sqlite = SQLiteStore(cfg)
    sqlite.init_schema()
    chroma = ChromaStore(cfg)
    fts = FtsStore(cfg)
    for folder in ("public", "private"):
        did = f"doc_{folder}"
        sqlite.insert_document(DocumentRecord(id=did, filename=f"{folder}.md", folder=folder))
        chroma.upsert_chunks([{
            "id": f"{did}_0", "embedding": [0.1] * 768,
            "text": "shared content phrase here",
            "metadata": {"doc_id": did, "filename": f"{folder}.md",
                         "folder": folder, "tags": "[]", "chunk_index": 0},
        }])
        fts.upsert_chunks([{
            "chunk_id": f"{did}_0", "doc_id": did, "folder": folder, "tags": "[]",
            "chunk_index": 0, "text": "shared content phrase here", "filename": f"{folder}.md",
        }])
    return sqlite, chroma, fts


@pytest.mark.asyncio
async def test_folder_filter_isolates_results(tmp_path):
    cfg = Config(data_dir=tmp_path)
    cfg.hybrid_enabled = True
    cfg.rerank_enabled = False
    cfg.ensure_dirs()
    sqlite, chroma, fts = _seed_two_folders(cfg)
    svc = SearchService(cfg, sqlite, chroma, FakeEmbedder(), fts=fts, reranker=None)
    results = await svc.search("shared content", folder="public")
    assert len(results) >= 1
    assert all(r.folder == "public" for r in results)
    assert not any(r.folder == "private" for r in results)


@pytest.mark.asyncio
async def test_folder_filter_applies_to_both_branches(tmp_path):
    """Critical: if folder filter is only applied to one branch, fusion
    would leak the other folder's chunks."""
    cfg = Config(data_dir=tmp_path)
    cfg.hybrid_enabled = True
    cfg.rerank_enabled = False
    cfg.ensure_dirs()
    sqlite, chroma, fts = _seed_two_folders(cfg)
    svc = SearchService(cfg, sqlite, chroma, FakeEmbedder(), fts=fts, reranker=None)
    for folder in ("public", "private"):
        results = await svc.search("shared", folder=folder)
        assert all(r.folder == folder for r in results), \
            f"Filter leaked across folders for folder={folder}: {[r.folder for r in results]}"
```

- [ ] **Step 2: Run tests**

```bash
poetry run pytest tests/security/test_search_isolation.py -v
```

Expected: Both PASS (vector branch uses Chroma's `where`, sparse branch uses SQL `AND folder=?`).

- [ ] **Step 3: Commit**

```bash
git add tests/security/test_search_isolation.py
git commit -m "test(security): verify folder isolation across vector + sparse branches"
```

---

## Task 20: Security tests — resource limits

**Files:**
- Create: `tests/security/test_resource_limits.py`

- [ ] **Step 1: Write tests**

Create `tests/security/test_resource_limits.py`:

```python
from __future__ import annotations

import asyncio
import time

import pytest

from docgraph.config import Config
from docgraph.mcp.search import SearchService
from docgraph.models import DocumentRecord
from docgraph.store import ChromaStore, FtsStore, SQLiteStore


class FakeEmbedder:
    async def embed(self, texts, for_query=False):
        return [[0.1] * 768 for _ in texts]


class SlowReranker:
    def __init__(self, delay):
        self.delay = delay
    async def rerank(self, q, p):
        await asyncio.sleep(self.delay)
        return [0.5] * len(p)


@pytest.fixture
def svc(tmp_path):
    cfg = Config(data_dir=tmp_path)
    cfg.hybrid_enabled = True
    cfg.rerank_enabled = False
    cfg.ensure_dirs()
    sqlite = SQLiteStore(cfg)
    sqlite.init_schema()
    chroma = ChromaStore(cfg)
    fts = FtsStore(cfg)
    for i in range(50):
        sqlite.insert_document(DocumentRecord(id=f"doc_{i}", filename=f"{i}.md"))
        chroma.upsert_chunks([{
            "id": f"doc_{i}_0", "embedding": [0.1] * 768,
            "text": f"document {i} content",
            "metadata": {"doc_id": f"doc_{i}", "filename": f"{i}.md",
                         "folder": "", "tags": "[]", "chunk_index": 0},
        }])
        fts.upsert_chunks([{
            "chunk_id": f"doc_{i}_0", "doc_id": f"doc_{i}", "folder": "", "tags": "[]",
            "chunk_index": 0, "text": f"document {i} content", "filename": f"{i}.md",
        }])
    return cfg, sqlite, chroma, fts


@pytest.mark.asyncio
async def test_huge_query_returns_in_time(svc):
    cfg, sqlite, chroma, fts = svc
    s = SearchService(cfg, sqlite, chroma, FakeEmbedder(), fts=fts, reranker=None)
    huge = "máy tính " * 10000  # ~80KB
    start = time.time()
    results = await asyncio.wait_for(s.search(huge), timeout=10.0)
    elapsed = time.time() - start
    assert elapsed < 5.0
    assert isinstance(results, list)


@pytest.mark.asyncio
async def test_rerank_timeout_bounds_total_time(svc):
    cfg, sqlite, chroma, fts = svc
    cfg.rerank_enabled = True
    cfg.rerank_score_gap_ratio = 0.99  # always rerank
    cfg.rerank_timeout_sec = 0.2
    s = SearchService(
        cfg, sqlite, chroma, FakeEmbedder(),
        fts=fts, reranker=SlowReranker(delay=5.0),
    )
    start = time.time()
    results = await s.search("document")
    elapsed = time.time() - start
    assert elapsed < 1.0
    assert len(results) > 0


@pytest.mark.asyncio
async def test_top_k_request_capped(svc):
    cfg, sqlite, chroma, fts = svc
    s = SearchService(cfg, sqlite, chroma, FakeEmbedder(), fts=fts, reranker=None)
    results = await s.search("document", top_k=10000)
    assert len(results) <= 100  # internal cap honored
```

- [ ] **Step 2: Run tests**

```bash
poetry run pytest tests/security/test_resource_limits.py -v
```

Expected: All 3 PASS.

- [ ] **Step 3: Commit**

```bash
git add tests/security/test_resource_limits.py
git commit -m "test(security): resource limit fuzzing (large queries, timeouts, top_k cap)"
```

---

## Task 21: Benchmark tests

**Files:**
- Create: `tests/perf/__init__.py`
- Create: `tests/perf/test_search_latency.py`
- Modify: `pyproject.toml` (add pytest-benchmark dev dep + marker)

- [ ] **Step 1: Add pytest-benchmark dependency + marker**

Modify `pyproject.toml`. Under `[tool.poetry.group.dev.dependencies]` add:

```toml
pytest-benchmark = "^4.0"
```

Under `[tool.pytest.ini_options].markers` ensure these exist (append if not):

```toml
markers = [
    "integration: tests requiring Ollama or external services",
    "rerank_model: tests downloading rerank ONNX model (~600MB)",
    "benchmark: performance baseline tests (uses pytest-benchmark)",
]
```

Install:

```bash
poetry lock --no-update && poetry install
```

- [ ] **Step 2: Create empty init**

Create `tests/perf/__init__.py` as empty file.

- [ ] **Step 3: Write benchmark tests**

Create `tests/perf/test_search_latency.py`:

```python
from __future__ import annotations

import pytest

from docgraph.config import Config
from docgraph.mcp.search import _rrf_fuse
from docgraph.store.fts import FtsStore, _sanitize_query
from docgraph.store.sqlite import SQLiteStore


@pytest.fixture
def fts_1000(tmp_path):
    cfg = Config(data_dir=tmp_path)
    cfg.ensure_dirs()
    sqlite = SQLiteStore(cfg)
    sqlite.init_schema()
    f = FtsStore(cfg)
    f.upsert_chunks([
        {"chunk_id": f"d_{i}", "doc_id": f"d{i // 10}", "folder": "x", "tags": "[]",
         "chunk_index": i % 10, "text": f"chunk text number {i} content sample",
         "filename": "f.md"}
        for i in range(1000)
    ])
    return f


@pytest.mark.benchmark
def test_fts_search_1000_chunks(benchmark, fts_1000):
    result = benchmark(lambda: fts_1000.search("chunk text", top_k=50))
    assert len(result) > 0
    median = benchmark.stats["median"]
    assert median < 0.100, f"FTS5 search median {median*1000:.1f}ms > budget 100ms"


@pytest.mark.benchmark
def test_rrf_fusion_30_each(benchmark):
    vec = [{
        "id": f"v{i}", "score": 1.0 / (i + 1), "text": "t", "doc_id": f"d{i}",
        "filename": "f", "folder": "", "tags": [], "chunk_index": i, "source_page": None,
    } for i in range(30)]
    sparse = [{
        "chunk_id": f"s{i}", "bm25_score": 1.0 / (i + 1), "doc_id": f"d{i}",
        "folder": "", "tags": [], "chunk_index": i,
    } for i in range(30)]
    result = benchmark(lambda: _rrf_fuse(vec, sparse, k_rrf=60))
    assert len(result) > 0
    assert benchmark.stats["median"] < 0.005


@pytest.mark.benchmark
def test_sanitize_long_query(benchmark):
    long_q = "máy tính " * 100
    benchmark(lambda: _sanitize_query(long_q))
    assert benchmark.stats["median"] < 0.005
```

- [ ] **Step 4: Run benchmarks**

```bash
poetry run pytest tests/perf/ -v --benchmark-only
```

Expected: All 3 PASS with reported medians under budget.

- [ ] **Step 5: Commit**

```bash
git add tests/perf/ pyproject.toml poetry.lock
git commit -m "test(perf): latency benchmarks for FTS5, RRF fusion, sanitize"
```

---

## Task 22: Integration test — real reranker model

**Files:**
- Modify: `tests/embed/test_rerank.py`

- [ ] **Step 1: Append integration tests for real model**

Append to `tests/embed/test_rerank.py`:

```python
@pytest.mark.rerank_model
class TestRerankerWithRealModel:
    """Downloads ~600MB BGE-reranker-v2-m3 ONNX model on first run."""

    @pytest.fixture(scope="class")
    def event_loop(self):
        import asyncio
        loop = asyncio.new_event_loop()
        yield loop
        loop.close()

    @pytest.fixture(scope="class")
    async def reranker(self, tmp_path_factory):
        from docgraph.embed.rerank import Reranker
        cache = tmp_path_factory.mktemp("models")
        r = Reranker(model="bge-reranker-v2-m3", cache_dir=cache)
        await r.prewarm()
        return r

    @pytest.mark.asyncio
    async def test_relevance_ordering(self, reranker):
        scores = await reranker.rerank(
            "What does DocGraph use for vector storage?",
            [
                "DocGraph stores vectors in ChromaDB with cosine similarity.",
                "The web UI is built with React and Vite.",
                "MarkItDown converts files to Markdown.",
            ],
        )
        assert len(scores) == 3
        assert scores[0] > scores[1]
        assert scores[0] > scores[2]

    @pytest.mark.asyncio
    async def test_multilingual_vn_en(self, reranker):
        scores = await reranker.rerank(
            "DocGraph dùng database nào để lưu vector?",
            [
                "DocGraph stores vectors in ChromaDB.",
                "User interface built with React.",
            ],
        )
        assert scores[0] > scores[1]

    @pytest.mark.asyncio
    async def test_concurrent_rerank_no_deadlock(self, reranker):
        import asyncio
        results = await asyncio.gather(*[
            reranker.rerank(f"query {i}", ["passage A", "passage B"]) for i in range(5)
        ])
        assert all(len(r) == 2 for r in results)

    @pytest.mark.asyncio
    async def test_empty_and_single_passage(self, reranker):
        assert await reranker.rerank("q", []) == []
        scores = await reranker.rerank("q", ["only one"])
        assert len(scores) == 1
```

- [ ] **Step 2: Run integration tests (manual, expect first run to download model)**

```bash
poetry run pytest tests/embed/test_rerank.py -m rerank_model -v
```

Expected: First run downloads ~600MB and PASSES; subsequent runs use cache.

- [ ] **Step 3: Commit**

```bash
git add tests/embed/test_rerank.py
git commit -m "test(embed): integration tests against real BGE-reranker-v2-m3"
```

---

## Task 23: Update README

**Files:**
- Modify: `README.md`

- [ ] **Step 1: Add config rows to env vars table**

Find the "Biến môi trường" table in README.md and append rows:

```markdown
| `DOCGRAPH_HYBRID_ENABLED` | `true` | Bật hybrid search (vector + BM25). |
| `DOCGRAPH_RRF_K` | `60` | Hằng số Reciprocal Rank Fusion. |
| `DOCGRAPH_RERANK_ENABLED` | `true` | Bật cross-encoder reranker. |
| `DOCGRAPH_RERANK_MODEL` | `bge-reranker-v2-m3` | Model reranker (fastembed). |
| `DOCGRAPH_RERANK_TOP_N` | `15` | Số candidate đưa vào reranker. |
| `DOCGRAPH_RERANK_TIMEOUT_SEC` | `3.0` | Timeout cho mỗi lần rerank. |
| `DOCGRAPH_RERANK_PREWARM` | `true` | Warmup model ở server start. |
| `DOCGRAPH_RERANK_SCORE_GAP_RATIO` | `0.5` | Ngưỡng skip rerank khi top-1 dominate. |
| `DOCGRAPH_RERANK_MIN_FLOOR` | `0.015` | Skip rerank khi top-1 RRF dưới floor. |
```

- [ ] **Step 2: Add troubleshoot section entries**

Find "Khắc phục sự cố" and add at the end:

```markdown
**"Hybrid search trả về kết quả như cũ (vector-only)"**
Index FTS chưa được populate (sau upgrade từ version cũ).
- Cách 1 (tự động): Server start sẽ phát hiện và rebuild trong background.
- Cách 2 (chủ động): `poetry run docgraph rebuild-fts`.

**"Reranker bị disable / báo lỗi"**
- Kiểm tra `GET /api/health` field `rerank_status`.
- `error: ImportError` → rebuild Rust crate vào Poetry venv:
  ```bash
  env -u CONDA_PREFIX poetry run maturin develop --release \
    --manifest-path crates/docgraph-embed/Cargo.toml
  ```
- `loading` → đợi 5-10s, model đang download (lần đầu, ~600MB).
- `disabled` → set `DOCGRAPH_RERANK_ENABLED=true` hoặc bật trong YAML.

**"First search sau khi start chậm 10s"**
Reranker cold start. `rerank_prewarm: true` (default) đã warmup ở background; nếu vẫn chậm, kiểm tra server log xem prewarm có lỗi không.
```

- [ ] **Step 3: Commit**

```bash
git add README.md
git commit -m "docs: hybrid search + reranker config and troubleshooting"
```

---

## Task 24: E2E integration test

**Files:**
- Modify: `tests/test_e2e.py`

- [ ] **Step 1: Append hybrid e2e test**

Append to `tests/test_e2e.py`:

```python
@pytest.mark.asyncio
async def test_hybrid_e2e_identifier_query(tmp_path):
    """Upload markdown with code identifier, query exact symbol, verify hit."""
    from docgraph.config import Config
    from docgraph.models import DocumentRecord
    from docgraph.web.deps import AppState

    cfg = Config(data_dir=tmp_path)
    cfg.hybrid_enabled = True
    cfg.rerank_enabled = False  # skip rerank to keep test fast + deterministic
    state = AppState.create(cfg)

    markdown = (
        "## Embedding\n\n"
        "Call `embed_query(text)` to embed user queries.\n\n"
        "## Vector storage\n\n"
        "ChromaDB stores embeddings with cosine similarity."
    )
    state.sqlite.insert_document(DocumentRecord(
        id="doc_e2e", filename="readme.md", folder="docs", tags=["test"]
    ))
    await state.indexer().index_markdown("doc_e2e", markdown)
    assert state.fts.count_chunks() > 0

    results = await state.search_service().search("embed_query", top_k=3)
    assert len(results) >= 1
    # Identifier should rank chunk 0 (the "embed_query" line) first or near top
    top_text = results[0].text
    assert "embed_query" in top_text
```

- [ ] **Step 2: Run e2e test**

```bash
poetry run pytest tests/test_e2e.py::test_hybrid_e2e_identifier_query -v
```

Expected: PASS.

- [ ] **Step 3: Commit**

```bash
git add tests/test_e2e.py
git commit -m "test(e2e): hybrid search identifier match"
```

---

## Task 25: Final verification — run full suite

**Files:** none (verification only)

- [ ] **Step 1: Run all fast tests**

```bash
poetry run pytest -m "not integration and not rerank_model and not benchmark" -v
```

Expected: ALL tests PASS.

- [ ] **Step 2: Run benchmarks**

```bash
poetry run pytest -m benchmark --benchmark-only -v
```

Expected: All benchmarks PASS within budgets.

- [ ] **Step 3: Smoke test server start**

```bash
DOCGRAPH_DATA_DIR=/tmp/docgraph-smoke poetry run python -c "
import asyncio
from docgraph.config import load_config
from docgraph.web.app import create_app
cfg = load_config()
app = create_app(cfg, mount_mcp=False)
print('OK')
"
```

Expected: prints `OK`, no exceptions.

- [ ] **Step 4: Smoke test rebuild-fts CLI**

```bash
DOCGRAPH_DATA_DIR=/tmp/docgraph-smoke poetry run docgraph rebuild-fts
```

Expected: Either "Nothing to rebuild" (empty Chroma) or progress + "Done" message.

- [ ] **Step 5: Verify GET /api/health shape**

```bash
DOCGRAPH_DATA_DIR=/tmp/docgraph-smoke poetry run python -c "
from fastapi.testclient import TestClient
from docgraph.config import load_config
from docgraph.web.app import create_app
app = create_app(load_config(), mount_mcp=False)
with TestClient(app) as c:
    print(c.get('/api/health').json())
"
```

Expected: JSON contains `hybrid_enabled`, `rerank_enabled`, `rerank_status`, `chroma_chunks`, `fts_chunks`, `fts_in_sync`.

- [ ] **Step 6: Cleanup smoke test dir**

```bash
rm -rf /tmp/docgraph-smoke
```

---

## Self-Review Checklist

Before declaring the plan complete, the implementer should verify against the spec:

- [ ] FTS5 virtual table created with `unicode61 remove_diacritics 2 tokenchars '_.-'` (Spec 4.1)
- [ ] `content=''` contentless mode (Spec 4.1 — Opt 2)
- [ ] Multi-column with weighted BM25 1.0/2.0/1.5 (Spec 4.1 — Opt 1)
- [ ] `executemany` batch insert (Spec 4.1 — Opt 6)
- [ ] Async non-blocking rebuild (Spec 4.1 — Opt 7)
- [ ] Order: Chroma first, FTS5 second (Spec 4.2)
- [ ] FTS5 delete on doc removal + reindex (Spec 4.3)
- [ ] Query sanitization neutralizes FTS5 operators (Spec 4.4)
- [ ] Auto-rebuild on startup mismatch (Spec 4.5)
- [ ] RRF k=60 (Spec 5.2)
- [ ] Overfetch 6× (Spec 5.1)
- [ ] `min_score` filter after fusion, override by rerank (Spec 5.1)
- [ ] Gate condition A: floor (Spec 5.3)
- [ ] Gate condition B: single-branch override (Spec 5.3)
- [ ] Gate default: gap ratio (Spec 5.3)
- [ ] Pre-warm at lifespan startup (Spec 5.3 — G)
- [ ] Rerank timeout (`asyncio.wait_for`) with RRF fallback (Spec 5.4)
- [ ] `search_metrics` log line (Spec 5.6)
- [ ] Rust crate split into `embed.rs` + `rerank.rs` (Spec 6.1)
- [ ] `rerank()` returns `Vec<f32>` aligned to input index (Spec 6.2)
- [ ] No `top_n` parameter in Rust rerank (Spec 6.2)
- [ ] Shared cache dir with embedder (Spec 6.4)
- [ ] Graceful degradation: all 5 failure cases (Spec 6.5)
- [ ] 9 config keys (Spec 7.1)
- [ ] YAML + env var support (Spec 7.2, 7.3)
- [ ] `Config.validate()` enforces bounds + coerces prewarm (Spec 7.4)
- [ ] `/api/health` extended (Spec 7.5)
- [ ] `rebuild-fts` CLI (Spec 7.7)
- [ ] All 15 new test files created (Spec 8.2)
- [ ] Gate test covers all branches (Spec 8.3)
- [ ] RRF math invariants tested (Spec 8.4)
- [ ] Benchmark soft assertions (Spec 8.5)
- [ ] Security tests for injection, isolation, resource limits (Spec 8.6)

If any item is unchecked, the implementer should add the missing piece before declaring complete.

---

## Execution Notes

**Estimated effort:** 25 tasks × ~15-30 min average = ~10-12 hours focused work for one engineer familiar with DocGraph internals.

**Critical path:**
Tasks 1-2 (Rust) and Task 3 (config) can parallelize. Tasks 4-9 (FTS5 path) must be sequential. Tasks 10 (Reranker wrapper) and Tasks 11-13 (search pipeline) can parallelize after Task 10 completes. Tasks 14-17 (wiring) sequential after their dependencies. Tasks 18-24 (tests + docs) can interleave.

**Build pitfall (from memory):** `maturin develop` MUST run with `env -u CONDA_PREFIX -u CONDA_DEFAULT_ENV -u CONDA_SHLVL -u CONDA_PROMPT_MODIFIER` prefix on this machine because Anaconda is on PATH and conflicts with Poetry's `VIRTUAL_ENV`. Use the full incantation in Tasks 1, 2, and during any debugging.

**Commit cadence:** One commit per task as shown. Each commit should be green (tests pass) before moving on.
