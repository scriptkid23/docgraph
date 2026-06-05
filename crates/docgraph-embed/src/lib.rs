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
