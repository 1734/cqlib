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

//! Python bindings for frozen circuits and circuit-defined gates.
//!
//! A [`PyFrozenCircuit`] is built from the self-contained construction IR and
//! then consumed by [`PyCircuitGate`]. This mirrors the core ownership model
//! without depending on the mutable Python circuit builder.

use crate::circuit::bit::PyQubit;
use crate::circuit::classical::PyClassicalType;
use crate::circuit::error::CircuitError as PyCircuitError;
use crate::circuit::operation::PyValueOperation;
use cqlib_core::circuit::Circuit;
use cqlib_core::circuit::gate::{CircuitGate, FrozenCircuit};
use pyo3::prelude::*;

/// Immutable circuit definition suitable for use inside a gate.
#[pyclass(name = "FrozenCircuit", module = "cqlib.circuit.gates", from_py_object)]
#[derive(Clone, Debug)]
pub struct PyFrozenCircuit {
    pub(crate) inner: FrozenCircuit,
}

#[pymethods]
impl PyFrozenCircuit {
    /// Builds and validates a frozen circuit from construction-IR operations.
    #[new]
    #[pyo3(signature = (qubits, operations, classical_vars=None, classical_values=None))]
    fn new(
        qubits: Vec<PyQubit>,
        operations: Vec<PyValueOperation>,
        classical_vars: Option<Vec<PyClassicalType>>,
        classical_values: Option<Vec<PyClassicalType>>,
    ) -> PyResult<Self> {
        let circuit = Circuit::from_operations(
            qubits.into_iter().map(|qubit| qubit.inner).collect(),
            operations.into_iter().map(|operation| operation.inner),
            classical_vars.map(|types| types.into_iter().map(|ty| ty.inner).collect()),
            classical_values.map(|types| types.into_iter().map(|ty| ty.inner).collect()),
        )
        .map_err(|error| PyCircuitError::new_err(error.to_string()))?;
        Ok(Self {
            inner: FrozenCircuit::new(circuit),
        })
    }

    /// Returns the frozen circuit's qubits in storage order.
    #[getter]
    fn qubits(&self) -> Vec<PyQubit> {
        self.inner
            .circuit()
            .qubits()
            .into_iter()
            .map(PyQubit::from)
            .collect()
    }

    /// Returns the number of stored operations.
    #[getter]
    fn num_operations(&self) -> usize {
        self.inner.circuit().operations().len()
    }

    /// Returns self-contained operations with circuit parameters resolved.
    #[getter]
    fn operations(&self) -> PyResult<Vec<PyValueOperation>> {
        (0..self.inner.circuit().operations().len())
            .map(|index| {
                self.inner
                    .circuit()
                    .index(index)
                    .map(PyValueOperation::from)
                    .map_err(|error| PyCircuitError::new_err(error.to_string()))
            })
            .collect()
    }

    /// Returns symbolic parameter names in circuit insertion order.
    #[getter]
    fn symbols(&self) -> Vec<String> {
        self.inner.circuit().symbols().iter().cloned().collect()
    }

    fn __repr__(&self) -> String {
        format!(
            "FrozenCircuit(qubits={}, operations={})",
            self.inner.circuit().num_qubits(),
            self.inner.circuit().operations().len()
        )
    }

    fn __copy__(&self) -> Self {
        self.clone()
    }

    fn __deepcopy__(&self, _memo: &Bound<'_, PyAny>) -> Self {
        self.clone()
    }
}

/// Composite gate defined by an immutable circuit.
#[pyclass(name = "CircuitGate", module = "cqlib.circuit.gates", from_py_object)]
#[derive(Clone, Debug)]
pub struct PyCircuitGate {
    pub(crate) inner: CircuitGate,
}

#[pymethods]
impl PyCircuitGate {
    /// Creates a reusable gate from a frozen circuit definition.
    #[new]
    fn new(name: String, circuit: PyFrozenCircuit) -> PyResult<Self> {
        CircuitGate::new(name, circuit.inner)
            .map(|inner| Self { inner })
            .map_err(|error| PyCircuitError::new_err(error.to_string()))
    }

    /// Returns the gate name.
    #[getter]
    fn name(&self) -> String {
        self.inner.name().to_string()
    }

    /// Returns the number of qubits used by the definition.
    #[getter]
    fn num_qubits(&self) -> usize {
        self.inner.num_qubits()
    }

    /// Returns the number of positional symbolic parameters.
    #[getter]
    fn num_params(&self) -> usize {
        self.inner.num_params()
    }

    /// Returns symbolic parameter names in positional order.
    #[getter]
    fn symbols(&self) -> Vec<String> {
        self.inner.symbols().into_iter().collect()
    }

    /// Returns the immutable circuit definition.
    #[getter]
    fn circuit(&self) -> PyFrozenCircuit {
        PyFrozenCircuit {
            inner: self.inner.circuit().as_ref().clone(),
        }
    }

    /// Returns the inverse circuit gate.
    fn inverse(&self) -> PyResult<Self> {
        self.inner
            .inverse()
            .map(|inner| Self { inner })
            .map_err(|error| PyCircuitError::new_err(error.to_string()))
    }

    fn __repr__(&self) -> String {
        format!(
            "CircuitGate({:?}, qubits={}, params={})",
            self.inner.name(),
            self.inner.num_qubits(),
            self.inner.num_params()
        )
    }

    fn __copy__(&self) -> Self {
        self.clone()
    }

    fn __deepcopy__(&self, _memo: &Bound<'_, PyAny>) -> Self {
        self.clone()
    }
}

impl From<CircuitGate> for PyCircuitGate {
    fn from(inner: CircuitGate) -> Self {
        Self { inner }
    }
}
