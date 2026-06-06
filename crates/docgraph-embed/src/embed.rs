use std::path::PathBuf;
use std::str::FromStr;
use std::sync::RwLock;

use fastembed::{EmbeddingModel, InitOptions, TextEmbedding};
use once_cell::sync::OnceCell;
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyList;

struct EmbedState {
    model: TextEmbedding,
    model_label: String,
    dim: usize,
}

static STATE: OnceCell<RwLock<Option<EmbedState>>> = OnceCell::new();

fn state_rwlock() -> &'static RwLock<Option<EmbedState>> {
    STATE.get_or_init(|| RwLock::new(None))
}

/// Map friendly names to fastembed ONNX models.
fn resolve_model(name: &str) -> PyResult<EmbeddingModel> {
    let base = name.split(':').next().unwrap_or(name).trim();
    if let Ok(model) = EmbeddingModel::from_str(base) {
        return Ok(model);
    }
    let normalized = base.to_ascii_lowercase();
    let model = match normalized.as_str() {
        "multilingual-e5"
        | "multilingual-e5-base"
        | "multilingual"
        | "multilingual-e5-base-onnx" => EmbeddingModel::MultilingualE5Base,
        "multilingual-e5-large" => EmbeddingModel::MultilingualE5Large,
        "multilingual-e5-small" => EmbeddingModel::MultilingualE5Small,
        "nomic-embed-text" | "nomic-embed-text:latest" => EmbeddingModel::NomicEmbedTextV15,
        "nomic-embed-text-v1" => EmbeddingModel::NomicEmbedTextV1,
        "nomic-embed-text-v1.5" => EmbeddingModel::NomicEmbedTextV15,
        "nomic-embed-text-v1.5-q" => EmbeddingModel::NomicEmbedTextV15Q,
        "all-minilm-l6-v2" => EmbeddingModel::AllMiniLML6V2,
        "bge-m3" | "bge-m3:latest" => {
            return Err(PyValueError::new_err(
                "bge-m3 is not available in the local ONNX embedder; \
                 use multilingual-e5-base/large or set DOCGRAPH_EMBED_PROVIDER=ollama"
                    .to_string(),
            ));
        }
        _ => {
            return Err(PyValueError::new_err(format!(
                "unknown embedding model: {name}. \
                 Try nomic-embed-text (default), multilingual-e5-base, \
                 multilingual-e5-large, or a fastembed EmbeddingModel name."
            )));
        }
    };
    Ok(model)
}

fn load_model(model_name: &str, cache_dir: Option<&str>) -> PyResult<EmbedState> {
    let embedding_model = resolve_model(model_name)?;
    let info = TextEmbedding::get_model_info(&embedding_model).map_err(|e| {
        PyRuntimeError::new_err(format!("model info unavailable: {e}"))
    })?;
    let dim = info.dim;

    let mut opts =
        InitOptions::new(embedding_model).with_show_download_progress(true);
    if let Some(dir) = cache_dir {
        if !dir.is_empty() {
            opts = opts.with_cache_dir(PathBuf::from(dir));
        }
    }

    let model = TextEmbedding::try_new(opts)
        .map_err(|e| PyRuntimeError::new_err(format!("failed to load embedding model: {e}")))?;

    Ok(EmbedState {
        model,
        model_label: model_name.to_string(),
        dim,
    })
}

fn with_state<F, T>(f: F) -> Result<T, String>
where
    F: FnOnce(&TextEmbedding, usize, &str) -> Result<T, String>,
{
    let guard = state_rwlock()
        .read()
        .map_err(|_| "embedder lock poisoned".to_string())?;
    let state = guard
        .as_ref()
        .ok_or_else(|| "embedder not initialized; call init() first".to_string())?;
    f(&state.model, state.dim, &state.model_label)
}

#[pyfunction]
#[pyo3(signature = (model="nomic-embed-text", cache_dir=None))]
pub fn init(model: &str, cache_dir: Option<&str>) -> PyResult<()> {
    let loaded = Python::with_gil(|py| py.allow_threads(|| load_model(model, cache_dir)))?;
    let mut guard = state_rwlock()
        .write()
        .map_err(|_| PyRuntimeError::new_err("embedder lock poisoned"))?;
    *guard = Some(loaded);
    Ok(())
}

#[pyfunction]
pub fn health_check() -> PyResult<()> {
    with_state(|_, _, _| Ok(())).map_err(PyRuntimeError::new_err)
}

#[pyfunction]
pub fn embedding_dimension() -> PyResult<usize> {
    with_state(|_, dim, _| Ok(dim)).map_err(PyRuntimeError::new_err)
}

#[pyfunction]
pub fn active_model() -> PyResult<String> {
    with_state(|_, _, label| Ok(label.to_string())).map_err(PyRuntimeError::new_err)
}

#[pyfunction]
pub fn embed(py: Python<'_>, texts: Vec<String>) -> PyResult<Py<PyList>> {
    if texts.is_empty() {
        return Ok(PyList::empty(py).unbind());
    }

    // Inside allow_threads we ONLY produce Result<_, String> — no PyErr construction.
    let result: Result<Vec<Vec<f32>>, String> = py.allow_threads(|| {
        with_state(|model, dim, _| {
            let refs: Vec<&str> = texts.iter().map(String::as_str).collect();
            let embeddings = model
                .embed(refs, None)
                .map_err(|e| format!("embed failed: {e}"))?;
            if embeddings.len() != texts.len() {
                return Err(format!(
                    "expected {} embeddings, got {}",
                    texts.len(),
                    embeddings.len()
                ));
            }
            for (i, vec) in embeddings.iter().enumerate() {
                if vec.len() != dim {
                    return Err(format!(
                        "embedding {} has dim {} (expected {})",
                        i,
                        vec.len(),
                        dim
                    ));
                }
            }
            Ok(embeddings)
        })
    });
    // PyErr constructed here, with GIL held (we're back outside allow_threads).
    let vectors = result.map_err(PyRuntimeError::new_err)?;

    let py_list = PyList::empty(py);
    for vec in vectors {
        let row = PyList::empty(py);
        for v in vec {
            row.append(v)?;
        }
        py_list.append(row)?;
    }
    Ok(py_list.unbind())
}
