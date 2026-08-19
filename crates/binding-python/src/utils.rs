use pyo3::prelude::*;
use pyo3::types::PyString;
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};

pub(crate) fn hash_value<T: Hash + ?Sized>(value: &T) -> u64 {
    let mut hasher = DefaultHasher::new();
    value.hash(&mut hasher);
    hasher.finish()
}

pub(crate) fn python_string_literal(value: &str) -> String {
    Python::attach(|py| {
        PyString::new(py, value)
            .repr()
            .expect("str.__repr__ must succeed")
            .to_string_lossy()
            .into_owned()
    })
}

/// Formats a bool as a Python `True`/`False` literal for use in reprs.
pub(crate) fn python_bool(value: bool) -> &'static str {
    if value { "True" } else { "False" }
}
