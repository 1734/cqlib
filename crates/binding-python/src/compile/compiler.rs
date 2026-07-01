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
    CompileConfig, CompileMode, CompileResult, CompilerWorkflow, WorkflowStepReport, compile,
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
                    "unknown standard gate in target_basis: {name:?}"
                )));
            }
        };
        Ok(Instruction::Standard(gate))
    }
}

/// Builds the core compiler configuration from Python-facing optional fields.
///
/// This is the shared conversion point for `CompileConfig`, `CompilerWorkflow`,
/// and `compile`. It applies Python-level defaults, converts target-basis
/// entries into core instructions, clones mutable device/layout inputs, and
/// falls back to the conservative default resource policy when none is given.
fn build_compile_config<'py>(
    mode: Option<PyCompileMode>,
    target_basis: Option<Vec<PyTargetBasisItem>>,
    device: Option<PyRef<'py, PyDevice>>,
    initial_layout: Option<PyRef<'py, PyLayout>>,
    resource_policy: Option<PyResourcePolicy>,
    seed: Option<u32>,
) -> PyResult<CompileConfig> {
    Ok(CompileConfig {
        mode: mode.map_or(CompileMode::Normal, |mode| mode.inner),
        target_basis: target_basis
            .map(|basis| {
                basis
                    .into_iter()
                    .map(PyTargetBasisItem::into_instruction)
                    .collect()
            })
            .transpose()?,
        device: device.map(|device| device.inner.clone()),
        initial_layout: initial_layout.map(|layout| layout.inner.clone()),
        resource_policy: resource_policy
            .map_or_else(ResourcePolicy::default, |policy| policy.inner),
        seed,
    })
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
            CompileMode::Normal => "CompileMode.normal()",
            CompileMode::Enhanced => "CompileMode.enhanced()",
        }
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
        let target_basis = self.inner.target_basis.as_ref().map_or_else(
            || "None".to_string(),
            |basis| {
                format!(
                    "[{}]",
                    basis
                        .iter()
                        .map(|instruction| format!("{:?}", instruction.to_string()))
                        .collect::<Vec<_>>()
                        .join(", ")
                )
            },
        );
        let device = self.inner.device.as_ref().map_or_else(
            || "None".to_string(),
            |device| format!("Device(name={:?})", device.name()),
        );
        let initial_layout = self.inner.initial_layout.as_ref().map_or_else(
            || "None".to_string(),
            |layout| {
                format!(
                    "Layout(num_logical={}, num_vacant_physical={}, num_physical={})",
                    layout.num_logical(),
                    layout.num_vacant_physical(),
                    layout.num_physical(),
                )
            },
        );
        let policy = self.inner.resource_policy;
        let seed = self
            .inner
            .seed
            .map_or_else(|| "None".to_string(), |seed| seed.to_string());

        format!(
            "CompileConfig(mode={}, target_basis={}, device={}, initial_layout={}, resource_policy=ResourcePolicy(max_pre_layout_clean_ancillas={}, allow_dirty_borrowing={}), seed={})",
            PyCompileMode::from(self.inner.mode).repr_label(),
            target_basis,
            device,
            initial_layout,
            policy.max_pre_layout_clean_ancillas,
            if policy.allow_dirty_borrowing {
                "True"
            } else {
                "False"
            },
            seed,
        )
    }
}

#[pymethods]
impl PyCompileConfig {
    /// Creates an immutable compiler workflow configuration snapshot.
    #[new]
    #[pyo3(signature = (*, mode=None, target_basis=None, device=None, initial_layout=None, resource_policy=None, seed=None))]
    fn new(
        mode: Option<PyCompileMode>,
        target_basis: Option<Vec<PyTargetBasisItem>>,
        device: Option<PyRef<'_, PyDevice>>,
        initial_layout: Option<PyRef<'_, PyLayout>>,
        resource_policy: Option<PyResourcePolicy>,
        seed: Option<u32>,
    ) -> PyResult<Self> {
        build_compile_config(
            mode,
            target_basis,
            device,
            initial_layout,
            resource_policy,
            seed,
        )
        .map(Self::from)
    }

    #[getter]
    fn mode(&self) -> PyCompileMode {
        self.inner.mode.into()
    }

    #[getter]
    fn target_basis(&self) -> Option<Vec<PyInstruction>> {
        self.inner
            .target_basis
            .as_ref()
            .map(|basis| basis.iter().cloned().map(PyInstruction::from).collect())
    }

    #[getter]
    fn device(&self) -> Option<PyDevice> {
        self.inner.device.clone().map(PyDevice::from)
    }

    #[getter]
    fn initial_layout(&self) -> Option<PyLayout> {
        self.inner.initial_layout.clone().map(PyLayout::from)
    }

    #[getter]
    fn resource_policy(&self) -> PyResourcePolicy {
        self.inner.resource_policy.into()
    }

    #[getter]
    fn seed(&self) -> Option<u32> {
        self.inner.seed
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
            None => build_compile_config(None, None, None, None, None, None)?,
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
#[pyo3(signature = (circuit, *, mode=None, target_basis=None, device=None, initial_layout=None, resource_policy=None, seed=None))]
#[allow(clippy::too_many_arguments)]
pub fn py_compile(
    py: Python<'_>,
    circuit: PyRef<'_, PyCircuit>,
    mode: Option<PyCompileMode>,
    target_basis: Option<Vec<PyTargetBasisItem>>,
    device: Option<PyRef<'_, PyDevice>>,
    initial_layout: Option<PyRef<'_, PyLayout>>,
    resource_policy: Option<PyResourcePolicy>,
    seed: Option<u32>,
) -> PyResult<PyCompileResult> {
    let config = build_compile_config(
        mode,
        target_basis,
        device,
        initial_layout,
        resource_policy,
        seed,
    )?;
    let circuit = circuit.inner.clone();

    py.detach(move || compile(&circuit, config))
        .map(PyCompileResult::from)
        .map_err(compiler_error_to_py_err)
}
