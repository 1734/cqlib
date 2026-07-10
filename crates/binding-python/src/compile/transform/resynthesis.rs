// This code is part of Cqlib.
//
// (C) Copyright China Telecom Quantum Group 2026
//
// This code is licensed under the Apache License, Version 2.0. You may
// obtain a copy of this license in the LICENSE.txt file in the root directory
// of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
//
// Any modifications or derivative works of this code must retain this
// copyright notice, and modified files need to carry a notice indicating
// that they have been altered from the originals.

//! Python bindings for numeric two-qubit block resynthesis.

use super::PyTransformResult;
use super::decompose::config::{PyTwoQubitUnitaryDecomposeBasis, target_for_basis};
use crate::circuit::PyCircuit;
use crate::compile::commutation::PyCommutationConfig;
use crate::compile::error::compiler_error_to_py_err;
use cqlib_core::compile::transform::{
    ResynthesizeTwoQubitBlocks, Transformer, TwoQubitBlockResynthesisConfig,
    resynthesize_two_qubit_blocks,
};
use pyo3::prelude::*;

/// Registers resynthesis bindings as `_native.compile.transform.resynthesis`.
pub(crate) fn register_resynthesis_module(parent: &Bound<'_, PyModule>) -> PyResult<()> {
    let m = PyModule::new(parent.py(), "resynthesis")?;
    m.add_class::<PyTwoQubitBlockResynthesisConfig>()?;
    m.add_class::<PyResynthesizeTwoQubitBlocks>()?;
    m.add_function(pyo3::wrap_pyfunction!(
        py_resynthesize_two_qubit_blocks,
        &m
    )?)?;

    parent.add_submodule(&m)?;
    parent
        .py()
        .import("sys")?
        .getattr("modules")?
        .set_item("cqlib._native.compile.transform.resynthesis", &m)?;
    Ok(())
}

/// Configuration for numeric two-qubit block resynthesis.
#[pyclass(
    name = "TwoQubitBlockResynthesisConfig",
    module = "cqlib.compile.transform.resynthesis",
    from_py_object
)]
#[derive(Clone, Debug)]
pub struct PyTwoQubitBlockResynthesisConfig {
    pub(crate) inner: TwoQubitBlockResynthesisConfig,
    two_qubit_basis: PyTwoQubitUnitaryDecomposeBasis,
}

#[pymethods]
impl PyTwoQubitBlockResynthesisConfig {
    /// Creates a bounded numeric resynthesis configuration.
    #[allow(clippy::too_many_arguments)]
    #[new]
    #[pyo3(signature = (*, two_qubit_basis=None, enhanced=false, max_block_ops=None, max_crossed_ops=None, max_scan_span=None, skip_labeled_ops=true, recurse_control_flow=true, commutation=None))]
    fn new(
        two_qubit_basis: Option<PyTwoQubitUnitaryDecomposeBasis>,
        enhanced: bool,
        max_block_ops: Option<usize>,
        max_crossed_ops: Option<usize>,
        max_scan_span: Option<usize>,
        skip_labeled_ops: bool,
        recurse_control_flow: bool,
        commutation: Option<PyCommutationConfig>,
    ) -> Self {
        let two_qubit_basis = two_qubit_basis.unwrap_or(PyTwoQubitUnitaryDecomposeBasis {
            inner: cqlib_core::compile::transform::decompose::unitary::TwoQubitUnitaryDecomposeBasis::PauliRotations,
        });
        let target = target_for_basis(two_qubit_basis.inner);
        let mut inner = if enhanced {
            TwoQubitBlockResynthesisConfig::enhanced(target)
        } else {
            TwoQubitBlockResynthesisConfig::normal(target)
        };
        if let Some(value) = max_block_ops {
            inner.max_block_ops = value;
        }
        if let Some(value) = max_crossed_ops {
            inner.max_crossed_ops = value;
        }
        if let Some(value) = max_scan_span {
            inner.max_scan_span = value;
        }
        inner.skip_labeled_ops = skip_labeled_ops;
        inner.recurse_control_flow = recurse_control_flow;
        if let Some(value) = commutation {
            inner.commutation = value.inner;
        }
        Self {
            inner,
            two_qubit_basis,
        }
    }

    #[getter]
    fn two_qubit_basis(&self) -> PyTwoQubitUnitaryDecomposeBasis {
        self.two_qubit_basis
    }

    #[getter]
    fn max_block_ops(&self) -> usize {
        self.inner.max_block_ops
    }

    #[getter]
    fn max_crossed_ops(&self) -> usize {
        self.inner.max_crossed_ops
    }

    #[getter]
    fn max_scan_span(&self) -> usize {
        self.inner.max_scan_span
    }

    #[getter]
    fn skip_labeled_ops(&self) -> bool {
        self.inner.skip_labeled_ops
    }

    #[getter]
    fn recurse_control_flow(&self) -> bool {
        self.inner.recurse_control_flow
    }

    #[getter]
    fn commutation(&self) -> PyCommutationConfig {
        self.inner.commutation.clone().into()
    }

    fn __repr__(&self) -> String {
        format!(
            "TwoQubitBlockResynthesisConfig(two_qubit_basis={}, max_block_ops={}, max_crossed_ops={}, max_scan_span={}, skip_labeled_ops={}, recurse_control_flow={}, commutation={:?})",
            self.two_qubit_basis.repr_value(),
            self.inner.max_block_ops,
            self.inner.max_crossed_ops,
            self.inner.max_scan_span,
            self.inner.skip_labeled_ops,
            self.inner.recurse_control_flow,
            self.inner.commutation,
        )
    }

    fn __eq__(&self, other: &Self) -> bool {
        self.inner == other.inner
    }

    fn __copy__(&self) -> Self {
        self.clone()
    }

    fn __deepcopy__(&self, _memo: &Bound<'_, PyAny>) -> Self {
        self.clone()
    }
}

/// Reusable transformer for numeric two-qubit block resynthesis.
#[pyclass(
    name = "ResynthesizeTwoQubitBlocks",
    module = "cqlib.compile.transform.resynthesis",
    from_py_object
)]
#[derive(Clone, Debug)]
pub struct PyResynthesizeTwoQubitBlocks {
    config: PyTwoQubitBlockResynthesisConfig,
}

#[pymethods]
impl PyResynthesizeTwoQubitBlocks {
    #[new]
    #[pyo3(signature = (config=None))]
    fn new(config: Option<PyTwoQubitBlockResynthesisConfig>) -> Self {
        Self {
            config: config.unwrap_or_else(|| {
                PyTwoQubitBlockResynthesisConfig::new(
                    None, false, None, None, None, true, true, None,
                )
            }),
        }
    }

    #[getter]
    fn config(&self) -> PyTwoQubitBlockResynthesisConfig {
        self.config.clone()
    }

    /// Runs resynthesis without mutating the input circuit.
    fn run(&self, py: Python<'_>, circuit: PyRef<'_, PyCircuit>) -> PyResult<PyTransformResult> {
        let circuit = circuit.inner.clone();
        let config = self.config.inner.clone();
        py.detach(move || ResynthesizeTwoQubitBlocks::new(config).transform(&circuit, None))
            .map(PyTransformResult::from)
            .map_err(compiler_error_to_py_err)
    }

    fn __repr__(&self) -> String {
        format!(
            "ResynthesizeTwoQubitBlocks(config={})",
            self.config.__repr__()
        )
    }

    fn __eq__(&self, other: &Self) -> bool {
        self.config.inner == other.config.inner
    }

    fn __copy__(&self) -> Self {
        self.clone()
    }

    fn __deepcopy__(&self, _memo: &Bound<'_, PyAny>) -> Self {
        self.clone()
    }
}

/// Resynthesizes fixed numeric two-qubit blocks without mutating `circuit`.
#[pyfunction(name = "resynthesize_two_qubit_blocks")]
#[pyo3(signature = (circuit, config=None))]
fn py_resynthesize_two_qubit_blocks(
    py: Python<'_>,
    circuit: PyRef<'_, PyCircuit>,
    config: Option<PyTwoQubitBlockResynthesisConfig>,
) -> PyResult<PyTransformResult> {
    let circuit = circuit.inner.clone();
    let config = config.map_or_else(
        || {
            PyTwoQubitBlockResynthesisConfig::new(None, false, None, None, None, true, true, None)
                .inner
        },
        |value| value.inner,
    );
    py.detach(move || resynthesize_two_qubit_blocks(&circuit, config))
        .map(PyTransformResult::from)
        .map_err(compiler_error_to_py_err)
}
