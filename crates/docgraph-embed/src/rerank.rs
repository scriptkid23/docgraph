use std::path::PathBuf;
use std::str::FromStr;
use std::sync::RwLock;

use fastembed::{RerankInitOptions, RerankerModel, TextRerank};
use once_cell::sync::OnceCell;
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;

struct RerankState {
    model: TextRerank,
    model_label: String,
}

static RERANK_STATE: OnceCell<RwLock<Option<RerankState>>> = OnceCell::new();

fn state_rwlock() -> &'static RwLock<Option<RerankState>> {
    RERANK_STATE.get_or_init(|| RwLock::new(None))
}

fn resolve_rerank_model(name: &str) -> PyResult<RerankerModel> {
    let base = name.split(':').next().unwrap_or(name).trim();
    if let Ok(m) = RerankerModel::from_str(base) {
        return Ok(m);
    }
    let normalized = base.to_ascii_lowercase();
    // NOTE: fastembed 4.9 enum variant names are unusual — `JINA` is all-caps and
    // `Multiligual` is misspelled in the upstream library. Do not "correct" these.
    // fastembed also dropped the V1 Base variant; only V1 Turbo exists.
    let model = match normalized.as_str() {
        "bge-reranker-base" => RerankerModel::BGERerankerBase,
        "bge-reranker-v2-m3" | "bge-reranker-v2" | "bge-m3-reranker" => {
            RerankerModel::BGERerankerV2M3
        }
        "jina-reranker-v2-multilingual" | "jina-reranker-v2" => {
            RerankerModel::JINARerankerV2BaseMultiligual
        }
        "jina-reranker-v1-turbo" | "jina-reranker-v1-turbo-en" => {
            RerankerModel::JINARerankerV1TurboEn
        }
        _ => {
            return Err(PyValueError::new_err(format!(
                "unknown reranker model: {name}. \
                 Try bge-reranker-v2-m3 (default), bge-reranker-base, \
                 jina-reranker-v2-multilingual, or jina-reranker-v1-turbo."
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

fn with_state<F, T>(f: F) -> Result<T, String>
where
    F: FnOnce(&TextRerank, &str) -> Result<T, String>,
{
    let guard = state_rwlock()
        .read()
        .map_err(|_| "rerank lock poisoned".to_string())?;
    let state = guard
        .as_ref()
        .ok_or_else(|| "reranker not initialized; call rerank_init() first".to_string())?;
    f(&state.model, &state.model_label)
}

#[pyfunction]
#[pyo3(signature = (model="bge-reranker-v2-m3", cache_dir=None))]
pub fn rerank_init(model: &str, cache_dir: Option<&str>) -> PyResult<()> {
    let loaded =
        Python::with_gil(|py| py.allow_threads(|| load_rerank_model(model, cache_dir)))?;
    let mut guard = state_rwlock()
        .write()
        .map_err(|_| PyRuntimeError::new_err("rerank lock poisoned"))?;
    *guard = Some(loaded);
    Ok(())
}

#[pyfunction]
pub fn rerank_health_check() -> PyResult<()> {
    with_state(|_, _| Ok(())).map_err(PyRuntimeError::new_err)
}

#[pyfunction]
pub fn active_rerank_model() -> PyResult<String> {
    with_state(|_, label| Ok(label.to_string())).map_err(PyRuntimeError::new_err)
}

#[pyfunction]
pub fn rerank(py: Python<'_>, query: String, passages: Vec<String>) -> PyResult<Vec<f32>> {
    if passages.is_empty() {
        return Ok(vec![]);
    }
    let n = passages.len();
    // Inside allow_threads we ONLY produce Result<_, String> — no PyErr construction.
    let result: Result<Vec<f32>, String> = py.allow_threads(|| {
        with_state(|model, _| {
            let refs: Vec<&str> = passages.iter().map(String::as_str).collect();
            let results = model
                .rerank(query.as_str(), refs, false, None)
                .map_err(|e| format!("rerank failed: {e}"))?;
            if results.len() != n {
                return Err(format!(
                    "rerank: expected {} scores, got {}",
                    n,
                    results.len()
                ));
            }
            let mut by_index = vec![0.0f32; n];
            for r in results {
                if r.index >= n {
                    return Err(format!(
                        "rerank: result index {} out of bounds for {} passages",
                        r.index, n
                    ));
                }
                by_index[r.index] = r.score;
            }
            Ok(by_index)
        })
    });
    // PyErr constructed here, with GIL held (we're back outside allow_threads).
    result.map_err(PyRuntimeError::new_err)
}
