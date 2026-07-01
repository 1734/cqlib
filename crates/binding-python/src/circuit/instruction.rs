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

//! Python wrappers for storage and construction instruction sum types.
//!
//! [`PyInstruction`] represents a circuit-local storage instruction.
//! [`PyValueInstruction`] is the construction form and owns recursive
//! value-level classical control flow without circuit parameter-table indices.

use crate::circuit::error::CircuitError as PyCircuitError;
use crate::circuit::{
    PyCircuitGate, PyClassicalControlOp, PyDirective, PyMcGate, PyStandardGate, PyUnitaryGate,
};
use cqlib_core::circuit::{Instruction, ValueInstruction};
use pyo3::prelude::*;

/// Python wrapper around the core storage-IR instruction enum.
#[pyclass(name = "Instruction", module = "cqlib.circuit", from_py_object)]
#[derive(Debug, Clone)]
pub struct PyInstruction {
    pub(crate) inner: Instruction,
}

impl From<Instruction> for PyInstruction {
    fn from(inner: Instruction) -> Self {
        Self { inner }
    }
}

impl From<PyInstruction> for Instruction {
    fn from(py: PyInstruction) -> Self {
        py.inner
    }
}

#[pymethods]
impl PyInstruction {
    #[staticmethod]
    fn from_standard_gate(gate: PyStandardGate) -> PyResult<Self> {
        if !gate.params.is_empty() {
            return Err(PyCircuitError::new_err(
                "Instruction does not own parameters; use ValueOperation.from_standard_gate()",
            ));
        }
        Ok(Self {
            inner: Instruction::Standard(gate.inner),
        })
    }

    #[staticmethod]
    fn from_mc_gate(gate: PyMcGate) -> PyResult<Self> {
        if !gate.params.is_empty() {
            return Err(PyCircuitError::new_err(
                "Instruction does not own parameters; use ValueOperation.from_mc_gate()",
            ));
        }
        Ok(Self {
            inner: Instruction::McGate(Box::new(gate.inner)),
        })
    }

    #[staticmethod]
    fn from_unitary_gate(gate: PyUnitaryGate) -> Self {
        Self {
            inner: Instruction::UnitaryGate(Box::new(gate.into())),
        }
    }

    /// Creates a storage instruction from a circuit-defined gate.
    #[staticmethod]
    fn from_circuit_gate(gate: PyCircuitGate) -> Self {
        Self {
            inner: Instruction::CircuitGate(Box::new(gate.inner)),
        }
    }

    #[staticmethod]
    fn from_directive(directive: PyDirective) -> Self {
        Self {
            inner: Instruction::Directive(directive.inner),
        }
    }

    #[staticmethod]
    fn delay() -> Self {
        Self {
            inner: Instruction::Delay,
        }
    }

    #[getter]
    fn name(&self) -> String {
        self.inner.to_string()
    }

    #[getter]
    fn instruction_type(&self) -> &'static str {
        match &self.inner {
            Instruction::Standard(_) => "standard",
            Instruction::McGate(_) => "mcgate",
            Instruction::UnitaryGate(_) => "unitary",
            Instruction::CircuitGate(_) => "circuit",
            Instruction::Directive(_) => "directive",
            Instruction::ClassicalData(_) => "classical_data",
            Instruction::ClassicalControl(_) => "classical_control",
            Instruction::Delay => "delay",
        }
    }

    #[getter]
    fn is_standard(&self) -> bool {
        matches!(self.inner, Instruction::Standard(_))
    }

    #[getter]
    fn is_mcgate(&self) -> bool {
        matches!(self.inner, Instruction::McGate(_))
    }

    #[getter]
    fn is_unitary(&self) -> bool {
        matches!(self.inner, Instruction::UnitaryGate(_))
    }

    #[getter]
    fn is_circuit_gate(&self) -> bool {
        matches!(self.inner, Instruction::CircuitGate(_))
    }

    #[getter]
    fn is_directive(&self) -> bool {
        matches!(self.inner, Instruction::Directive(_))
    }

    #[getter]
    fn is_classical_control(&self) -> bool {
        matches!(self.inner, Instruction::ClassicalControl(_))
    }

    #[getter]
    fn is_classical_data(&self) -> bool {
        matches!(self.inner, Instruction::ClassicalData(_))
    }

    #[getter]
    fn is_delay(&self) -> bool {
        matches!(self.inner, Instruction::Delay)
    }

    #[getter]
    fn standard_gate(&self) -> Option<PyStandardGate> {
        match &self.inner {
            Instruction::Standard(gate) => Some(PyStandardGate::from(*gate, vec![])),
            _ => None,
        }
    }

    #[getter]
    fn directive(&self) -> Option<PyDirective> {
        match &self.inner {
            Instruction::Directive(directive) => Some(PyDirective::from(*directive)),
            _ => None,
        }
    }

    fn __str__(&self) -> String {
        format!("{}", self.inner)
    }

    fn __repr__(&self) -> String {
        format!("Instruction({})", self.name())
    }

    fn __copy__(&self) -> Self {
        self.clone()
    }

    fn __deepcopy__(&self, _memo: &Bound<'_, PyAny>) -> Self {
        self.clone()
    }
}

/// Python wrapper around the self-contained construction-IR instruction enum.
#[pyclass(name = "ValueInstruction", module = "cqlib.circuit", from_py_object)]
#[derive(Debug, Clone)]
pub struct PyValueInstruction {
    pub(crate) inner: ValueInstruction,
}

impl From<ValueInstruction> for PyValueInstruction {
    fn from(inner: ValueInstruction) -> Self {
        Self { inner }
    }
}

#[pymethods]
impl PyValueInstruction {
    #[staticmethod]
    fn from_instruction(instruction: PyInstruction) -> Self {
        Self {
            inner: ValueInstruction::from_instruction(instruction.inner),
        }
    }

    #[staticmethod]
    fn from_classical_control(control: PyClassicalControlOp) -> Self {
        Self {
            inner: ValueInstruction::ClassicalControl(control.inner),
        }
    }

    #[getter]
    fn is_classical_control(&self) -> bool {
        self.inner.is_classical_control()
    }

    #[getter]
    fn is_instruction(&self) -> bool {
        self.inner.is_instruction()
    }

    #[getter]
    fn instruction(&self) -> Option<PyInstruction> {
        self.inner
            .as_instruction()
            .cloned()
            .map(PyInstruction::from)
    }

    #[getter]
    fn classical_control(&self) -> Option<PyClassicalControlOp> {
        match &self.inner {
            ValueInstruction::ClassicalControl(control) => Some(control.clone().into()),
            ValueInstruction::Instruction(_) => None,
        }
    }

    fn __str__(&self) -> String {
        format!("{}", self.inner)
    }

    fn __repr__(&self) -> String {
        format!("ValueInstruction(\"{}\")", self.inner)
    }

    fn __copy__(&self) -> Self {
        self.clone()
    }

    fn __deepcopy__(&self, _memo: &Bound<'_, PyAny>) -> Self {
        self.clone()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use cqlib_core::circuit::{Parameter, StandardGate};

    #[test]
    fn storage_instruction_rejects_bound_gate_parameters() {
        let gate = PyStandardGate::from(StandardGate::RX, vec![Parameter::symbol("theta")]);
        assert!(PyInstruction::from_standard_gate(gate).is_err());
    }
}
