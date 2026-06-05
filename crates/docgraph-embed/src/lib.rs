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
