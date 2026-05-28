use std::path::PathBuf;
use std::str::FromStr;
use std::sync::Mutex;

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

static STATE: OnceCell<Mutex<Option<EmbedState>>> = OnceCell::new();

fn state_mutex() -> &'static Mutex<Option<EmbedState>> {
    STATE.get_or_init(|| Mutex::new(None))
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

fn with_state<F, T>(f: F) -> PyResult<T>
where
    F: FnOnce(&mut TextEmbedding, usize, &str) -> PyResult<T>,
{
    let mut guard = state_mutex()
        .lock()
        .map_err(|_| PyRuntimeError::new_err("embedder lock poisoned"))?;
    let state = guard
        .as_mut()
        .ok_or_else(|| PyRuntimeError::new_err("embedder not initialized; call init() first"))?;
    f(&mut state.model, state.dim, &state.model_label)
}

#[pyfunction]
#[pyo3(signature = (model="nomic-embed-text", cache_dir=None))]
fn init(model: &str, cache_dir: Option<&str>) -> PyResult<()> {
    let loaded = Python::with_gil(|py| py.allow_threads(|| load_model(model, cache_dir)))?;
    let mut guard = state_mutex()
        .lock()
        .map_err(|_| PyRuntimeError::new_err("embedder lock poisoned"))?;
    *guard = Some(loaded);
    Ok(())
}

#[pyfunction]
fn health_check() -> PyResult<()> {
    with_state(|_, _, _| Ok(()))
}

#[pyfunction]
fn embedding_dimension() -> PyResult<usize> {
    with_state(|_, dim, _| Ok(dim))
}

#[pyfunction]
fn active_model() -> PyResult<String> {
    with_state(|_, _, label| Ok(label.to_string()))
}

#[pyfunction]
fn embed(py: Python<'_>, texts: Vec<String>) -> PyResult<Py<PyList>> {
    if texts.is_empty() {
        return Ok(PyList::empty(py).unbind());
    }

    let vectors = py.allow_threads(|| {
        with_state(|model, dim, _| {
            let refs: Vec<&str> = texts.iter().map(String::as_str).collect();
            let embeddings = model
                .embed(refs, None)
                .map_err(|e| PyRuntimeError::new_err(format!("embed failed: {e}")))?;
            if embeddings.len() != texts.len() {
                return Err(PyRuntimeError::new_err(format!(
                    "expected {} embeddings, got {}",
                    texts.len(),
                    embeddings.len()
                )));
            }
            for (i, vec) in embeddings.iter().enumerate() {
                if vec.len() != dim {
                    return Err(PyRuntimeError::new_err(format!(
                        "embedding {} has dim {} (expected {})",
                        i,
                        vec.len(),
                        dim
                    )));
                }
            }
            Ok(embeddings)
        })
    })?;

    let rows = vectors
        .into_iter()
        .map(|vec| PyList::new(py, vec))
        .collect::<PyResult<Vec<_>>>()?;
    Ok(PyList::new(py, rows)?.unbind())
}

#[pymodule]
fn docgraph_embed(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(init, m)?)?;
    m.add_function(wrap_pyfunction!(embed, m)?)?;
    m.add_function(wrap_pyfunction!(health_check, m)?)?;
    m.add_function(wrap_pyfunction!(embedding_dimension, m)?)?;
    m.add_function(wrap_pyfunction!(active_model, m)?)?;
    Ok(())
}
