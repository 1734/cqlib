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

use crate::circuit::{PyCircuit, PyInstruction};
use crate::compile::error::{CompilerConfigError, compiler_error_to_py_err};
use crate::compile::resource::PyResourcePolicy;
use crate::device::device_impl::PyDevice;
use crate::device::layout::PyLayout;
use cqlib_core::circuit::{Instruction, StandardGate};
use cqlib_core::compile::resource::ResourcePolicy;
use cqlib_core::compile::{
    CompileConfig, CompileMode, CompileResult, CompileTarget, CompilerWorkflow,
    DeviceCompilationMetadata, DeviceCompileTarget, WorkflowStepReport, compile,
};
use pyo3::prelude::*;
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};

/// Python-facing target-basis item accepted by compile configuration builders.
///
/// The Python API accepts either case-insensitive standard-gate names for
/// concise call sites or fully constructed `Instruction` objects when callers
/// need to pass an instruction value directly. String names are normalized with
/// ASCII uppercase before matching the standard-gate table; unknown names are
/// reported as configuration errors.
#[derive(FromPyObject)]
pub enum PyTargetBasisItem {
    Name(String),
    Instruction(PyInstruction),
}

impl PyTargetBasisItem {
    fn into_instruction(self) -> PyResult<Instruction> {
        let name = match self {
            Self::Name(name) => name,
            Self::Instruction(instruction) => return Ok(instruction.inner),
        };

        let gate = match name.to_ascii_uppercase().as_str() {
            "I" => StandardGate::I,
            "H" => StandardGate::H,
            "RX" => StandardGate::RX,
            "RXX" => StandardGate::RXX,
            "RXY" => StandardGate::RXY,
            "RY" => StandardGate::RY,
            "RYY" => StandardGate::RYY,
            "RZ" => StandardGate::RZ,
            "RZX" => StandardGate::RZX,
            "RZZ" => StandardGate::RZZ,
            "S" => StandardGate::S,
            "SDG" => StandardGate::SDG,
            "SWAP" => StandardGate::SWAP,
            "T" => StandardGate::T,
            "TDG" => StandardGate::TDG,
            "U" => StandardGate::U,
            "X" => StandardGate::X,
            "XY" => StandardGate::XY,
            "X2P" => StandardGate::X2P,
            "X2M" => StandardGate::X2M,
            "XY2P" => StandardGate::XY2P,
            "XY2M" => StandardGate::XY2M,
            "Y" => StandardGate::Y,
            "Y2P" => StandardGate::Y2P,
            "Y2M" => StandardGate::Y2M,
            "Z" => StandardGate::Z,
            "PHASE" => StandardGate::Phase,
            "GPHASE" => StandardGate::GPhase,
            "CX" => StandardGate::CX,
            "CCX" => StandardGate::CCX,
            "CY" => StandardGate::CY,
            "CZ" => StandardGate::CZ,
            "CRX" => StandardGate::CRX,
            "CRY" => StandardGate::CRY,
            "CRZ" => StandardGate::CRZ,
            "FSIM" => StandardGate::FSIM,
            _ => {
                return Err(CompilerConfigError::new_err(format!(
                    "unknown standard gate in compile target basis: {name:?}"
                )));
            }
        };
        Ok(Instruction::Standard(gate))
    }
}

/// Builds the core compiler configuration from Python-facing snapshots.
fn build_compile_config(
    mode: Option<PyCompileMode>,
    target: Option<PyCompileTarget>,
    resource_policy: Option<PyResourcePolicy>,
) -> CompileConfig {
    CompileConfig {
        mode: mode.map_or(CompileMode::Normal, |mode| mode.inner),
        target: target.map_or(CompileTarget::Logical, |target| target.inner),
        resource_policy: resource_policy
            .map_or_else(ResourcePolicy::default, |policy| policy.inner),
    }
}

/// Optimization effort selected for the compiler workflow.
#[pyclass(name = "CompileMode", module = "cqlib.compile", from_py_object)]
#[derive(Clone, Copy, Debug)]
pub struct PyCompileMode {
    pub(crate) inner: CompileMode,
}

impl From<CompileMode> for PyCompileMode {
    fn from(inner: CompileMode) -> Self {
        Self { inner }
    }
}

impl From<PyCompileMode> for CompileMode {
    fn from(value: PyCompileMode) -> Self {
        value.inner
    }
}

impl PyCompileMode {
    pub(crate) fn repr_label(&self) -> &'static str {
        match self.inner {
            CompileMode::Normal => "CompileMode.Normal",
            CompileMode::Enhanced => "CompileMode.Enhanced",
        }
    }
}

/// Device-specific inputs for a physical compilation target.
#[pyclass(name = "DeviceCompileTarget", module = "cqlib.compile", from_py_object)]
#[derive(Clone, Debug)]
pub struct PyDeviceCompileTarget {
    pub(crate) inner: DeviceCompileTarget,
}

impl From<DeviceCompileTarget> for PyDeviceCompileTarget {
    fn from(inner: DeviceCompileTarget) -> Self {
        Self { inner }
    }
}

#[pymethods]
impl PyDeviceCompileTarget {
    /// Creates an immutable snapshot of device compilation inputs.
    #[new]
    #[pyo3(signature = (device, *, initial_layout=None, seed=None))]
    fn new(
        device: PyRef<'_, PyDevice>,
        initial_layout: Option<PyRef<'_, PyLayout>>,
        seed: Option<u32>,
    ) -> Self {
        Self {
            inner: DeviceCompileTarget {
                device: device.inner.clone(),
                initial_layout: initial_layout.map(|layout| layout.inner.clone()),
                seed,
            },
        }
    }

    #[getter]
    fn device(&self) -> PyDevice {
        self.inner.device.clone().into()
    }

    #[getter]
    fn initial_layout(&self) -> Option<PyLayout> {
        self.inner.initial_layout.clone().map(Into::into)
    }

    #[getter]
    fn seed(&self) -> Option<u32> {
        self.inner.seed
    }

    fn __repr__(&self) -> String {
        format!(
            "DeviceCompileTarget(device=Device(name={:?}), initial_layout={}, seed={:?})",
            self.inner.device.name(),
            if self.inner.initial_layout.is_some() {
                "Layout(...)"
            } else {
                "None"
            },
            self.inner.seed,
        )
    }

    fn __copy__(&self) -> Self {
        self.clone()
    }

    fn __deepcopy__(&self, _memo: &Bound<'_, PyAny>) -> Self {
        self.clone()
    }
}

/// Mutually exclusive logical, basis, or physical-device compile target.
#[pyclass(name = "CompileTarget", module = "cqlib.compile", from_py_object)]
#[derive(Clone, Debug)]
pub struct PyCompileTarget {
    pub(crate) inner: CompileTarget,
}

impl From<CompileTarget> for PyCompileTarget {
    fn from(inner: CompileTarget) -> Self {
        Self { inner }
    }
}

#[pymethods]
impl PyCompileTarget {
    /// Returns a target that preserves logical operations.
    #[staticmethod]
    fn logical() -> Self {
        CompileTarget::Logical.into()
    }

    /// Returns a target that lowers to an explicit standard-gate basis.
    #[staticmethod]
    fn basis(instructions: Vec<PyTargetBasisItem>) -> PyResult<Self> {
        let instructions = instructions
            .into_iter()
            .map(PyTargetBasisItem::into_instruction)
            .collect::<PyResult<Vec<_>>>()?;
        Ok(CompileTarget::Basis(instructions).into())
    }

    /// Returns a physical-device compile target.
    #[staticmethod]
    fn device(target: PyDeviceCompileTarget) -> Self {
        CompileTarget::Device(target.inner).into()
    }

    #[getter]
    fn kind(&self) -> &'static str {
        match self.inner {
            CompileTarget::Logical => "logical",
            CompileTarget::Basis(_) => "basis",
            CompileTarget::Device(_) => "device",
        }
    }

    #[getter]
    fn basis_instructions(&self) -> Option<Vec<PyInstruction>> {
        match &self.inner {
            CompileTarget::Basis(instructions) => {
                Some(instructions.iter().cloned().map(Into::into).collect())
            }
            CompileTarget::Logical | CompileTarget::Device(_) => None,
        }
    }

    #[getter]
    fn device_target(&self) -> Option<PyDeviceCompileTarget> {
        match &self.inner {
            CompileTarget::Device(target) => Some(target.clone().into()),
            CompileTarget::Logical | CompileTarget::Basis(_) => None,
        }
    }

    fn __repr__(&self) -> String {
        match &self.inner {
            CompileTarget::Logical => "CompileTarget.logical()".to_string(),
            CompileTarget::Basis(instructions) => format!(
                "CompileTarget.basis([{}])",
                instructions
                    .iter()
                    .map(|instruction| format!("{:?}", instruction.to_string()))
                    .collect::<Vec<_>>()
                    .join(", ")
            ),
            CompileTarget::Device(target) => format!(
                "CompileTarget.device(DeviceCompileTarget(device=Device(name={:?}), initial_layout={}, seed={:?}))",
                target.device.name(),
                if target.initial_layout.is_some() {
                    "Layout(...)"
                } else {
                    "None"
                },
                target.seed,
            ),
        }
    }

    fn __copy__(&self) -> Self {
        self.clone()
    }

    fn __deepcopy__(&self, _memo: &Bound<'_, PyAny>) -> Self {
        self.clone()
    }
}

#[pymethods]
impl PyCompileMode {
    /// Returns the normal production compiler mode.
    #[staticmethod]
    fn normal() -> Self {
        Self {
            inner: CompileMode::Normal,
        }
    }

    /// Returns the enhanced compiler mode.
    #[staticmethod]
    fn enhanced() -> Self {
        Self {
            inner: CompileMode::Enhanced,
        }
    }

    fn __repr__(&self) -> &'static str {
        self.repr_label()
    }

    fn __str__(&self) -> &'static str {
        match self.inner {
            CompileMode::Normal => "normal",
            CompileMode::Enhanced => "enhanced",
        }
    }

    fn __eq__(&self, other: &Self) -> bool {
        self.inner == other.inner
    }

    fn __hash__(&self) -> u64 {
        let mut hasher = DefaultHasher::new();
        match self.inner {
            CompileMode::Normal => 0_u8,
            CompileMode::Enhanced => 1_u8,
        }
        .hash(&mut hasher);
        hasher.finish()
    }

    fn __copy__(&self) -> Self {
        *self
    }

    fn __deepcopy__(&self, _memo: &Bound<'_, PyAny>) -> Self {
        *self
    }
}

/// Immutable compiler workflow configuration snapshot.
#[pyclass(name = "CompileConfig", module = "cqlib.compile", from_py_object)]
#[derive(Clone, Debug)]
pub struct PyCompileConfig {
    pub(crate) inner: CompileConfig,
}

impl From<CompileConfig> for PyCompileConfig {
    fn from(inner: CompileConfig) -> Self {
        Self { inner }
    }
}

impl From<PyCompileConfig> for CompileConfig {
    fn from(value: PyCompileConfig) -> Self {
        value.inner
    }
}

impl PyCompileConfig {
    pub(crate) fn repr(&self) -> String {
        let policy = self.inner.resource_policy;

        format!(
            "CompileConfig(mode={}, target={}, resource_policy=ResourcePolicy(max_pre_layout_clean_ancillas={}, allow_dirty_borrowing={}))",
            PyCompileMode::from(self.inner.mode).repr_label(),
            PyCompileTarget::from(self.inner.target.clone()).__repr__(),
            policy.max_pre_layout_clean_ancillas,
            if policy.allow_dirty_borrowing {
                "True"
            } else {
                "False"
            },
        )
    }
}

#[pymethods]
impl PyCompileConfig {
    /// Creates an immutable compiler workflow configuration snapshot.
    #[new]
    #[pyo3(signature = (*, mode=None, target=None, resource_policy=None))]
    fn new(
        mode: Option<PyCompileMode>,
        target: Option<PyCompileTarget>,
        resource_policy: Option<PyResourcePolicy>,
    ) -> Self {
        build_compile_config(mode, target, resource_policy).into()
    }

    #[getter]
    fn mode(&self) -> PyCompileMode {
        self.inner.mode.into()
    }

    #[getter]
    fn target(&self) -> PyCompileTarget {
        self.inner.target.clone().into()
    }

    #[getter]
    fn resource_policy(&self) -> PyResourcePolicy {
        self.inner.resource_policy.into()
    }

    fn __repr__(&self) -> String {
        self.repr()
    }

    fn __copy__(&self) -> Self {
        self.clone()
    }

    fn __deepcopy__(&self, _memo: &Bound<'_, PyAny>) -> Self {
        self.clone()
    }
}

/// Physical layout metadata returned by device compilation.
#[pyclass(
    name = "DeviceCompilationMetadata",
    module = "cqlib.compile",
    skip_from_py_object
)]
#[derive(Clone, Debug)]
pub struct PyDeviceCompilationMetadata {
    inner: DeviceCompilationMetadata,
}

impl From<DeviceCompilationMetadata> for PyDeviceCompilationMetadata {
    fn from(inner: DeviceCompilationMetadata) -> Self {
        Self { inner }
    }
}

#[pymethods]
impl PyDeviceCompilationMetadata {
    #[getter]
    fn initial_layout(&self) -> PyLayout {
        self.inner.initial_layout.clone().into()
    }

    #[getter]
    fn final_layout(&self) -> PyLayout {
        self.inner.final_layout.clone().into()
    }

    fn __repr__(&self) -> String {
        format!(
            "DeviceCompilationMetadata(initial_layout={:?}, final_layout={:?})",
            self.inner.initial_layout, self.inner.final_layout
        )
    }

    fn __copy__(&self) -> Self {
        self.clone()
    }

    fn __deepcopy__(&self, _memo: &Bound<'_, PyAny>) -> Self {
        self.clone()
    }

    fn __eq__(&self, other: &Self) -> bool {
        self.inner == other.inner
    }
}

/// Per-step execution record produced by a compiler workflow run.
#[pyclass(
    name = "WorkflowStepReport",
    module = "cqlib.compile",
    skip_from_py_object
)]
#[derive(Clone, Debug)]
pub struct PyWorkflowStepReport {
    pub(crate) inner: WorkflowStepReport,
}

impl From<WorkflowStepReport> for PyWorkflowStepReport {
    fn from(inner: WorkflowStepReport) -> Self {
        Self { inner }
    }
}

#[pymethods]
impl PyWorkflowStepReport {
    #[getter]
    fn stage(&self) -> &'static str {
        self.inner.stage
    }

    #[getter]
    fn name(&self) -> &'static str {
        self.inner.name
    }

    #[getter]
    fn changed(&self) -> bool {
        self.inner.changed
    }

    #[getter]
    fn skipped(&self) -> bool {
        self.inner.skipped
    }

    #[getter]
    fn reason(&self) -> Option<String> {
        self.inner.reason.clone()
    }

    fn __repr__(&self) -> String {
        match &self.inner.reason {
            Some(reason) => format!(
                "WorkflowStepReport(stage={:?}, name={:?}, changed={}, skipped={}, reason={:?})",
                self.inner.stage, self.inner.name, self.inner.changed, self.inner.skipped, reason
            ),
            None => format!(
                "WorkflowStepReport(stage={:?}, name={:?}, changed={}, skipped={}, reason=None)",
                self.inner.stage, self.inner.name, self.inner.changed, self.inner.skipped
            ),
        }
    }

    fn __copy__(&self) -> Self {
        self.clone()
    }

    fn __deepcopy__(&self, _memo: &Bound<'_, PyAny>) -> Self {
        self.clone()
    }
}

/// Result returned by `cqlib.compile.compile`.
#[pyclass(name = "CompileResult", module = "cqlib.compile", skip_from_py_object)]
#[derive(Clone, Debug)]
pub struct PyCompileResult {
    pub(crate) inner: CompileResult,
}

impl From<CompileResult> for PyCompileResult {
    fn from(inner: CompileResult) -> Self {
        Self { inner }
    }
}

#[pymethods]
impl PyCompileResult {
    #[getter]
    fn circuit(&self) -> PyCircuit {
        PyCircuit::from(self.inner.circuit.clone())
    }

    #[getter]
    fn changed(&self) -> bool {
        self.inner.changed
    }

    #[getter]
    fn mode(&self) -> PyCompileMode {
        self.inner.mode.into()
    }

    #[getter]
    fn steps(&self) -> Vec<PyWorkflowStepReport> {
        self.inner
            .steps
            .iter()
            .cloned()
            .map(PyWorkflowStepReport::from)
            .collect()
    }

    #[getter]
    fn device_metadata(&self) -> Option<PyDeviceCompilationMetadata> {
        self.inner.device_metadata.clone().map(Into::into)
    }

    /// Returns the first workflow report with the requested step name.
    fn step(&self, name: &str) -> Option<PyWorkflowStepReport> {
        self.inner.step(name).cloned().map(Into::into)
    }

    /// Returns whether a non-skipped report with this name changed the circuit.
    fn step_changed(&self, name: &str) -> bool {
        self.inner.step_changed(name)
    }

    fn __repr__(&self) -> String {
        format!(
            "CompileResult(changed={}, mode={}, steps={})",
            self.inner.changed,
            PyCompileMode::from(self.inner.mode).repr_label(),
            self.inner.steps.len()
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

/// Reusable compiler optimization workflow.
#[pyclass(
    name = "CompilerWorkflow",
    module = "cqlib.compile",
    skip_from_py_object
)]
pub struct PyCompilerWorkflow {
    inner: CompilerWorkflow,
}

#[pymethods]
impl PyCompilerWorkflow {
    /// Creates a workflow from a configuration snapshot.
    #[new]
    #[pyo3(signature = (config=None))]
    fn new(config: Option<PyCompileConfig>) -> PyResult<Self> {
        let config = match config {
            Some(config) => config.inner,
            None => build_compile_config(None, None, None),
        };
        Ok(Self {
            inner: CompilerWorkflow::new(config),
        })
    }

    #[getter]
    fn config(&self) -> PyCompileConfig {
        self.inner.config().clone().into()
    }

    /// Runs the workflow without modifying the input circuit.
    fn run(&self, py: Python<'_>, circuit: PyRef<'_, PyCircuit>) -> PyResult<PyCompileResult> {
        let circuit = circuit.inner.clone();
        let config = self.inner.config().clone();
        py.detach(move || CompilerWorkflow::new(config).run(&circuit))
            .map(PyCompileResult::from)
            .map_err(compiler_error_to_py_err)
    }

    fn __repr__(&self) -> String {
        format!("CompilerWorkflow(config={})", self.config().repr())
    }
}

/// Compiles a circuit with the configured compiler workflow.
#[pyfunction(name = "compile")]
#[pyo3(signature = (circuit, *, mode=None, target=None, resource_policy=None))]
pub fn py_compile(
    py: Python<'_>,
    circuit: PyRef<'_, PyCircuit>,
    mode: Option<PyCompileMode>,
    target: Option<PyCompileTarget>,
    resource_policy: Option<PyResourcePolicy>,
) -> PyResult<PyCompileResult> {
    let config = build_compile_config(mode, target, resource_policy);
    let circuit = circuit.inner.clone();

    py.detach(move || compile(&circuit, config))
        .map(PyCompileResult::from)
        .map_err(compiler_error_to_py_err)
}
