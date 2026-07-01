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

//! Python bindings for cqlib-core error-mitigation APIs.

pub mod error;

use crate::circuit::{PyCircuit, PyInstruction};
use crate::error_mitigation::error::error_mitigation_error_to_py_err;
use crate::qis::hamiltonian::PyHamiltonian;
use cqlib_core::circuit::Instruction;
use cqlib_core::error_mitigation::{
    ErrorMitigation, ExtrapolateMethod, MitigatedResult, MitigationMethod, ProcessArgs, RunArgs,
    VirtualDistillation, VirtualDistillationConfig, ZNEMitigation, ZneConfig,
};
use pyo3::exceptions::PyTypeError;
use pyo3::prelude::*;
use std::cell::RefCell;
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};

type EstimatorErrorCell = RefCell<Option<PyErr>>;

fn convert_gate_set(gate_set: Option<Vec<PyInstruction>>) -> Option<Vec<Instruction>> {
    gate_set.map(|gates| gates.into_iter().map(|gate| gate.inner).collect())
}

fn call_python_estimator(
    estimator: &Bound<'_, PyAny>,
    error: &EstimatorErrorCell,
    circuit: &cqlib_core::circuit::Circuit,
    hamiltonian: Option<&cqlib_core::qis::Hamiltonian>,
    shots: Option<usize>,
) -> (f64, f64) {
    if error.borrow().is_some() {
        return (f64::NAN, f64::NAN);
    }

    let py_circuit = PyCircuit::from(circuit.clone());
    let py_hamiltonian = hamiltonian.cloned().map(PyHamiltonian::from);
    match estimator
        .call1((py_circuit, py_hamiltonian, shots))
        .and_then(|value| value.extract::<(f64, f64)>())
    {
        Ok(result) => result,
        Err(py_error) => {
            *error.borrow_mut() = Some(py_error);
            (f64::NAN, f64::NAN)
        }
    }
}

fn take_estimator_error(error: EstimatorErrorCell) -> PyResult<()> {
    if let Some(error) = error.into_inner() {
        Err(error)
    } else {
        Ok(())
    }
}

#[pyclass(
    name = "ExtrapolateMethod",
    module = "cqlib.error_mitigation",
    from_py_object
)]
#[derive(Clone, Copy, Debug)]
pub struct PyExtrapolateMethod {
    inner: ExtrapolateMethod,
}

impl From<ExtrapolateMethod> for PyExtrapolateMethod {
    fn from(inner: ExtrapolateMethod) -> Self {
        Self { inner }
    }
}

#[pymethods]
impl PyExtrapolateMethod {
    #[staticmethod]
    fn polynomial() -> Self {
        ExtrapolateMethod::Polynomial.into()
    }

    #[staticmethod]
    fn exponential() -> Self {
        ExtrapolateMethod::Exponential.into()
    }

    fn __repr__(&self) -> &'static str {
        match self.inner {
            ExtrapolateMethod::Polynomial => "ExtrapolateMethod.polynomial()",
            ExtrapolateMethod::Exponential => "ExtrapolateMethod.exponential()",
        }
    }

    fn __str__(&self) -> &'static str {
        match self.inner {
            ExtrapolateMethod::Polynomial => "polynomial",
            ExtrapolateMethod::Exponential => "exponential",
        }
    }

    fn __eq__(&self, other: &Self) -> bool {
        self.inner == other.inner
    }

    fn __hash__(&self) -> u64 {
        let mut hasher = DefaultHasher::new();
        match self.inner {
            ExtrapolateMethod::Polynomial => 0_u8.hash(&mut hasher),
            ExtrapolateMethod::Exponential => 1_u8.hash(&mut hasher),
        }
        hasher.finish()
    }

    fn __copy__(&self) -> Self {
        *self
    }

    fn __deepcopy__(&self, _memo: &Bound<'_, PyAny>) -> Self {
        *self
    }
}

#[pyclass(name = "ZneConfig", module = "cqlib.error_mitigation", from_py_object)]
#[derive(Clone, Debug)]
pub struct PyZneConfig {
    inner: ZneConfig,
}

#[pymethods]
impl PyZneConfig {
    #[new]
    fn new(fold_levels: Vec<i32>) -> Self {
        Self {
            inner: ZneConfig { fold_levels },
        }
    }

    #[getter]
    fn fold_levels(&self) -> Vec<i32> {
        self.inner.fold_levels.clone()
    }

    fn __repr__(&self) -> String {
        format!("ZneConfig(fold_levels={:?})", self.inner.fold_levels)
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

#[pyclass(
    name = "VirtualDistillationConfig",
    module = "cqlib.error_mitigation",
    from_py_object
)]
#[derive(Clone, Debug)]
pub struct PyVirtualDistillationConfig {
    inner: VirtualDistillationConfig,
}

#[pymethods]
impl PyVirtualDistillationConfig {
    #[new]
    fn new(copies: usize) -> Self {
        Self {
            inner: VirtualDistillationConfig { copies },
        }
    }

    #[getter]
    fn copies(&self) -> usize {
        self.inner.copies
    }

    fn __repr__(&self) -> String {
        format!("VirtualDistillationConfig(copies={})", self.inner.copies)
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

#[pyclass(
    name = "MitigationMethod",
    module = "cqlib.error_mitigation",
    from_py_object
)]
#[derive(Clone, Debug)]
pub struct PyMitigationMethod {
    inner: MitigationMethod,
}

#[pymethods]
impl PyMitigationMethod {
    #[staticmethod]
    fn zne(config: PyZneConfig) -> Self {
        Self {
            inner: MitigationMethod::Zne(config.inner),
        }
    }

    #[staticmethod]
    fn virtual_distillation(config: PyVirtualDistillationConfig) -> Self {
        Self {
            inner: MitigationMethod::VirtualDistillation(config.inner),
        }
    }

    #[getter]
    fn method_type(&self) -> &'static str {
        match self.inner {
            MitigationMethod::Zne(_) => "zne",
            MitigationMethod::VirtualDistillation(_) => "virtual_distillation",
        }
    }

    fn __repr__(&self) -> String {
        match &self.inner {
            MitigationMethod::Zne(config) => {
                format!(
                    "MitigationMethod.zne(ZneConfig(fold_levels={:?}))",
                    config.fold_levels
                )
            }
            MitigationMethod::VirtualDistillation(config) => format!(
                "MitigationMethod.virtual_distillation(VirtualDistillationConfig(copies={}))",
                config.copies
            ),
        }
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

#[pyclass(name = "RunArgs", module = "cqlib.error_mitigation", from_py_object)]
#[derive(Clone, Debug)]
pub struct PyRunArgs {
    inner: RunArgs,
}

#[pymethods]
impl PyRunArgs {
    #[staticmethod]
    #[pyo3(signature = (gate_set=None, shots=None))]
    fn zne(gate_set: Option<Vec<PyInstruction>>, shots: Option<usize>) -> Self {
        Self {
            inner: RunArgs::Zne {
                gate_set: convert_gate_set(gate_set),
                shots,
            },
        }
    }

    #[staticmethod]
    fn virtual_distillation(shots_numerator: usize, shots_denominator: usize) -> Self {
        Self {
            inner: RunArgs::VirtualDistillation {
                shots_numerator,
                shots_denominator,
            },
        }
    }

    #[getter]
    fn method_type(&self) -> &'static str {
        match self.inner {
            RunArgs::Zne { .. } => "zne",
            RunArgs::VirtualDistillation { .. } => "virtual_distillation",
        }
    }

    fn __repr__(&self) -> String {
        match &self.inner {
            RunArgs::Zne { gate_set, shots } => {
                format!("RunArgs.zne(gate_set={:?}, shots={shots:?})", gate_set)
            }
            RunArgs::VirtualDistillation {
                shots_numerator,
                shots_denominator,
            } => format!(
                "RunArgs.virtual_distillation(shots_numerator={shots_numerator}, shots_denominator={shots_denominator})"
            ),
        }
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

#[pyclass(
    name = "ProcessArgs",
    module = "cqlib.error_mitigation",
    from_py_object
)]
#[derive(Clone, Copy, Debug)]
pub struct PyProcessArgs {
    inner: ProcessArgs,
}

#[pymethods]
impl PyProcessArgs {
    #[staticmethod]
    #[pyo3(signature = (method, degree=None))]
    fn zne(method: PyExtrapolateMethod, degree: Option<usize>) -> Self {
        Self {
            inner: ProcessArgs::Zne {
                method: method.inner,
                degree,
            },
        }
    }

    #[staticmethod]
    fn virtual_distillation() -> Self {
        Self {
            inner: ProcessArgs::VirtualDistillation,
        }
    }

    #[getter]
    fn method_type(&self) -> &'static str {
        match self.inner {
            ProcessArgs::Zne { .. } => "zne",
            ProcessArgs::VirtualDistillation => "virtual_distillation",
        }
    }

    fn __repr__(&self) -> String {
        match self.inner {
            ProcessArgs::Zne { method, degree } => format!(
                "ProcessArgs.zne(method={}, degree={degree:?})",
                PyExtrapolateMethod::from(method).__str__()
            ),
            ProcessArgs::VirtualDistillation => "ProcessArgs.virtual_distillation()".to_string(),
        }
    }

    fn __eq__(&self, other: &Self) -> bool {
        self.inner == other.inner
    }

    fn __copy__(&self) -> Self {
        *self
    }

    fn __deepcopy__(&self, _memo: &Bound<'_, PyAny>) -> Self {
        *self
    }
}

#[pyclass(
    name = "MitigatedResult",
    module = "cqlib.error_mitigation",
    skip_from_py_object
)]
#[derive(Clone, Debug)]
pub struct PyMitigatedResult {
    inner: MitigatedResult,
}

impl From<MitigatedResult> for PyMitigatedResult {
    fn from(inner: MitigatedResult) -> Self {
        Self { inner }
    }
}

#[pymethods]
impl PyMitigatedResult {
    #[getter]
    fn expectation(&self) -> f64 {
        self.inner.expectation
    }

    #[getter]
    fn variance(&self) -> Option<f64> {
        self.inner.variance
    }

    fn __repr__(&self) -> String {
        format!(
            "MitigatedResult(expectation={}, variance={:?})",
            self.inner.expectation, self.inner.variance
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

#[pyclass(
    name = "ZNEMitigation",
    module = "cqlib.error_mitigation",
    skip_from_py_object
)]
#[derive(Clone, Debug)]
pub struct PyZNEMitigation {
    inner: ZNEMitigation,
}

#[pymethods]
impl PyZNEMitigation {
    #[new]
    fn new(circuit: PyRef<'_, PyCircuit>, fold_levels: Vec<i32>) -> Self {
        Self {
            inner: ZNEMitigation::new(circuit.inner.clone(), fold_levels),
        }
    }

    #[getter]
    fn circuit(&self) -> PyCircuit {
        self.inner.circuit().clone().into()
    }

    #[getter]
    fn fold_levels(&self) -> Vec<i32> {
        self.inner.fold_levels().to_vec()
    }

    #[getter]
    fn noise_factors(&self) -> Vec<i32> {
        self.inner.noise_factors().to_vec()
    }

    #[pyo3(signature = (gate_set=None))]
    fn fold_circuits(&self, gate_set: Option<Vec<PyInstruction>>) -> PyResult<Vec<PyCircuit>> {
        let gate_set = convert_gate_set(gate_set);
        self.inner
            .fold_circuits(gate_set.as_deref())
            .map(|circuits| circuits.into_iter().map(PyCircuit::from).collect())
            .map_err(|error| crate::circuit::error::CircuitError::new_err(error.to_string()))
    }

    fn run_em_sequence(
        &self,
        gate_set: Option<Vec<PyInstruction>>,
        hamiltonian: PyRef<'_, PyHamiltonian>,
        estimator: &Bound<'_, PyAny>,
    ) -> PyResult<Vec<f64>> {
        self.run_em_sequence_with_shots(gate_set, hamiltonian, None, estimator)
    }

    #[pyo3(signature = (gate_set, hamiltonian, shots, estimator))]
    fn run_em_sequence_with_shots(
        &self,
        gate_set: Option<Vec<PyInstruction>>,
        hamiltonian: PyRef<'_, PyHamiltonian>,
        shots: Option<usize>,
        estimator: &Bound<'_, PyAny>,
    ) -> PyResult<Vec<f64>> {
        if !estimator.is_callable() {
            return Err(PyTypeError::new_err("estimator must be callable"));
        }

        let gate_set = convert_gate_set(gate_set);
        let estimator_error = RefCell::new(None);
        let result = self.inner.run_em_sequence_with_shots(
            gate_set.as_deref(),
            &hamiltonian.inner,
            shots,
            &|circuit, hamiltonian, shots| {
                call_python_estimator(estimator, &estimator_error, circuit, hamiltonian, shots)
            },
        );
        take_estimator_error(estimator_error)?;
        result.map_err(error_mitigation_error_to_py_err)
    }

    fn extrapolate(
        &self,
        noisy_results: Vec<f64>,
        method: PyExtrapolateMethod,
        degree: usize,
    ) -> PyResult<f64> {
        self.inner
            .extrapolate(&noisy_results, method.inner, degree)
            .map_err(error_mitigation_error_to_py_err)
    }

    fn poly_extrapolate(&self, noisy_results: Vec<f64>, degree: usize) -> PyResult<f64> {
        self.inner
            .poly_extrapolate(&noisy_results, degree)
            .map_err(error_mitigation_error_to_py_err)
    }

    fn exp_extrapolate(&self, noisy_results: Vec<f64>) -> PyResult<f64> {
        self.inner
            .exp_extrapolate(&noisy_results)
            .map_err(error_mitigation_error_to_py_err)
    }

    fn __repr__(&self) -> String {
        format!(
            "ZNEMitigation(fold_levels={:?}, noise_factors={:?})",
            self.inner.fold_levels(),
            self.inner.noise_factors()
        )
    }

    fn __copy__(&self) -> Self {
        self.clone()
    }

    fn __deepcopy__(&self, _memo: &Bound<'_, PyAny>) -> Self {
        self.clone()
    }
}

#[pyclass(
    name = "VirtualDistillation",
    module = "cqlib.error_mitigation",
    skip_from_py_object
)]
#[derive(Clone, Debug)]
pub struct PyVirtualDistillation {
    inner: VirtualDistillation,
}

#[pymethods]
impl PyVirtualDistillation {
    #[new]
    fn new(circuit: PyRef<'_, PyCircuit>, copies: usize) -> PyResult<Self> {
        VirtualDistillation::new(circuit.inner.clone(), copies)
            .map(|inner| Self { inner })
            .map_err(error_mitigation_error_to_py_err)
    }

    #[getter]
    fn copies(&self) -> usize {
        self.inner.copies()
    }

    fn set_copies(&mut self, copies: usize) -> PyResult<()> {
        self.inner
            .set_copies(copies)
            .map_err(error_mitigation_error_to_py_err)
    }

    fn build_copy_swap_circuit(&self) -> PyResult<PyCircuit> {
        self.inner
            .build_copy_swap_circuit()
            .map(PyCircuit::from)
            .map_err(|error| crate::circuit::error::CircuitError::new_err(error.to_string()))
    }

    fn run_denominator_circuit(
        &self,
        shots: usize,
        estimator: &Bound<'_, PyAny>,
    ) -> PyResult<(f64, f64)> {
        if !estimator.is_callable() {
            return Err(PyTypeError::new_err("estimator must be callable"));
        }

        let estimator_error = RefCell::new(None);
        let result = self
            .inner
            .run_denominator_circuit(shots, &|circuit, hamiltonian, shots| {
                call_python_estimator(estimator, &estimator_error, circuit, hamiltonian, shots)
            });
        take_estimator_error(estimator_error)?;
        result.map_err(|error| crate::circuit::error::CircuitError::new_err(error.to_string()))
    }

    fn run_numerator_circuit(
        &self,
        hamiltonian: PyRef<'_, PyHamiltonian>,
        shots: usize,
        estimator: &Bound<'_, PyAny>,
    ) -> PyResult<(f64, f64)> {
        if !estimator.is_callable() {
            return Err(PyTypeError::new_err("estimator must be callable"));
        }

        let estimator_error = RefCell::new(None);
        let result = self.inner.run_numerator_circuit(
            &hamiltonian.inner,
            shots,
            &|circuit, hamiltonian, shots| {
                call_python_estimator(estimator, &estimator_error, circuit, hamiltonian, shots)
            },
        );
        take_estimator_error(estimator_error)?;
        result.map_err(error_mitigation_error_to_py_err)
    }

    fn run_vd(
        &self,
        hamiltonian: PyRef<'_, PyHamiltonian>,
        shots_numerator: usize,
        shots_denominator: usize,
        estimator: &Bound<'_, PyAny>,
    ) -> PyResult<(f64, f64)> {
        if !estimator.is_callable() {
            return Err(PyTypeError::new_err("estimator must be callable"));
        }

        let estimator_error = RefCell::new(None);
        let result = self.inner.run_vd(
            &hamiltonian.inner,
            shots_numerator,
            shots_denominator,
            &|circuit, hamiltonian, shots| {
                call_python_estimator(estimator, &estimator_error, circuit, hamiltonian, shots)
            },
        );
        take_estimator_error(estimator_error)?;
        result.map_err(error_mitigation_error_to_py_err)
    }

    fn __repr__(&self) -> String {
        format!("VirtualDistillation(copies={})", self.inner.copies())
    }

    fn __copy__(&self) -> Self {
        self.clone()
    }

    fn __deepcopy__(&self, _memo: &Bound<'_, PyAny>) -> Self {
        self.clone()
    }
}

#[pyclass(
    name = "ErrorMitigation",
    module = "cqlib.error_mitigation",
    skip_from_py_object
)]
#[derive(Clone, Debug)]
pub struct PyErrorMitigation {
    inner: ErrorMitigation,
}

#[pymethods]
impl PyErrorMitigation {
    #[new]
    fn new(circuit: PyRef<'_, PyCircuit>, method: PyMitigationMethod) -> PyResult<Self> {
        ErrorMitigation::new(circuit.inner.clone(), method.inner)
            .map(|inner| Self { inner })
            .map_err(error_mitigation_error_to_py_err)
    }

    fn run(
        &mut self,
        hamiltonian: PyRef<'_, PyHamiltonian>,
        run_args: PyRunArgs,
        estimator: &Bound<'_, PyAny>,
    ) -> PyResult<()> {
        if !estimator.is_callable() {
            return Err(PyTypeError::new_err("estimator must be callable"));
        }

        let estimator_error = RefCell::new(None);
        let result = self.inner.run(
            &hamiltonian.inner,
            run_args.inner,
            &|circuit, hamiltonian, shots| {
                call_python_estimator(estimator, &estimator_error, circuit, hamiltonian, shots)
            },
        );
        take_estimator_error(estimator_error)?;
        result.map_err(error_mitigation_error_to_py_err)
    }

    fn get_mitigated(&mut self, process_args: PyProcessArgs) -> PyResult<PyMitigatedResult> {
        self.inner
            .get_mitigated(process_args.inner)
            .map(PyMitigatedResult::from)
            .map_err(error_mitigation_error_to_py_err)
    }

    fn __repr__(&self) -> &'static str {
        "ErrorMitigation()"
    }

    fn __copy__(&self) -> Self {
        self.clone()
    }

    fn __deepcopy__(&self, _memo: &Bound<'_, PyAny>) -> Self {
        self.clone()
    }
}

pub(crate) fn register_error_mitigation_module(parent: &Bound<'_, PyModule>) -> PyResult<()> {
    let m = PyModule::new(parent.py(), "error_mitigation")?;

    error::register_errors(&m)?;
    m.add_class::<PyExtrapolateMethod>()?;
    m.add_class::<PyZneConfig>()?;
    m.add_class::<PyVirtualDistillationConfig>()?;
    m.add_class::<PyMitigationMethod>()?;
    m.add_class::<PyRunArgs>()?;
    m.add_class::<PyProcessArgs>()?;
    m.add_class::<PyMitigatedResult>()?;
    m.add_class::<PyZNEMitigation>()?;
    m.add_class::<PyVirtualDistillation>()?;
    m.add_class::<PyErrorMitigation>()?;

    parent.add_submodule(&m)?;
    parent
        .py()
        .import("sys")?
        .getattr("modules")?
        .set_item("cqlib._native.error_mitigation", &m)?;

    Ok(())
}
