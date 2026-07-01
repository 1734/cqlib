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

use crate::circuit::PyCircuit;
use crate::compile::error::compiler_error_to_py_err;
use crate::compile::sabre::PySabreConfig;
use crate::device::device_impl::PyDevice;
use crate::device::layout::PyLayout;
use cqlib_core::compile::sabre::SabreConfig;
use cqlib_core::compile::transform::layout::build_physical_layout_graph;
use cqlib_core::compile::transform::{
    LayoutDiagnostics, LayoutObjective, LayoutResult, LayoutScore, Vf2EdgeRequirement,
    Vf2LayoutConfig, greedy_layout, sabre_layout, trivial_layout, vf2_perfect_layout,
};
use pyo3::prelude::*;
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};

/// Registers layout bindings as `_native.compile.transform.layout`.
pub(crate) fn register_layout_module(parent: &Bound<'_, PyModule>) -> PyResult<()> {
    let m = PyModule::new(parent.py(), "layout")?;

    m.add_class::<PyLayoutObjective>()?;
    m.add_class::<PyLayoutScore>()?;
    m.add_class::<PyLayoutDiagnostics>()?;
    m.add_class::<PyLayoutResult>()?;
    m.add_class::<PyVf2EdgeRequirement>()?;
    m.add_class::<PyVf2LayoutConfig>()?;
    m.add_function(pyo3::wrap_pyfunction!(py_trivial_layout, &m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(py_greedy_layout, &m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(py_vf2_perfect_layout, &m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(py_sabre_layout, &m)?)?;

    parent.add_submodule(&m)?;
    parent
        .py()
        .import("sys")?
        .getattr("modules")?
        .set_item("cqlib._native.compile.transform.layout", &m)?;

    Ok(())
}

#[pyfunction(name = "trivial_layout")]
#[pyo3(signature = (circuit, device, objective=None))]
fn py_trivial_layout(
    py: Python<'_>,
    circuit: PyRef<'_, PyCircuit>,
    device: PyRef<'_, PyDevice>,
    objective: Option<PyLayoutObjective>,
) -> PyResult<PyLayoutResult> {
    let circuit = circuit.inner.clone();
    let device = device.inner.clone();
    let objective = objective.map_or_else(LayoutObjective::topology_only, |value| value.inner);
    py.detach(move || trivial_layout(&circuit, &device, &objective))
        .map(Into::into)
        .map_err(compiler_error_to_py_err)
}

#[pyfunction(name = "greedy_layout")]
#[pyo3(signature = (circuit, device, objective=None))]
fn py_greedy_layout(
    py: Python<'_>,
    circuit: PyRef<'_, PyCircuit>,
    device: PyRef<'_, PyDevice>,
    objective: Option<PyLayoutObjective>,
) -> PyResult<PyLayoutResult> {
    let circuit = circuit.inner.clone();
    let device = device.inner.clone();
    let objective = objective.map_or_else(LayoutObjective::topology_only, |value| value.inner);
    py.detach(move || greedy_layout(&circuit, &device, &objective))
        .map(Into::into)
        .map_err(compiler_error_to_py_err)
}

#[pyfunction(name = "vf2_perfect_layout")]
#[pyo3(signature = (circuit, device, objective=None, config=None))]
fn py_vf2_perfect_layout(
    py: Python<'_>,
    circuit: PyRef<'_, PyCircuit>,
    device: PyRef<'_, PyDevice>,
    objective: Option<PyLayoutObjective>,
    config: Option<PyVf2LayoutConfig>,
) -> PyResult<PyLayoutResult> {
    let circuit = circuit.inner.clone();
    let device = device.inner.clone();
    let objective = objective.map_or_else(LayoutObjective::topology_only, |value| value.inner);
    let config = config.map_or_else(Vf2LayoutConfig::default, |value| value.inner);
    py.detach(move || vf2_perfect_layout(&circuit, &device, &objective, &config))
        .map(Into::into)
        .map_err(compiler_error_to_py_err)
}

#[pyfunction(name = "sabre_layout")]
#[pyo3(signature = (circuit, device, objective=None, config=None))]
fn py_sabre_layout(
    py: Python<'_>,
    circuit: PyRef<'_, PyCircuit>,
    device: PyRef<'_, PyDevice>,
    objective: Option<PyLayoutObjective>,
    config: Option<PySabreConfig>,
) -> PyResult<PyLayoutResult> {
    let circuit = circuit.inner.clone();
    let device = device.inner.clone();
    let objective = objective.map_or_else(LayoutObjective::topology_only, |value| value.inner);
    let config = config.map_or_else(SabreConfig::default, |value| value.inner);
    py.detach(move || sabre_layout(&circuit, &device, &objective, &config))
        .map(Into::into)
        .map_err(compiler_error_to_py_err)
}

/// Weighted objective used to rank candidate initial layouts.
#[pyclass(
    name = "LayoutObjective",
    module = "cqlib.compile.transform.layout",
    from_py_object
)]
#[derive(Clone, Debug)]
pub struct PyLayoutObjective {
    pub(crate) inner: LayoutObjective,
}

impl From<LayoutObjective> for PyLayoutObjective {
    fn from(inner: LayoutObjective) -> Self {
        Self { inner }
    }
}

#[pymethods]
impl PyLayoutObjective {
    /// Creates a layout objective from explicit non-negative weights.
    #[new]
    #[pyo3(signature = (*, distance_weight=1.0, direction_weight=1.0, two_qubit_error_weight=0.0, readout_error_weight=0.0))]
    fn new(
        distance_weight: f64,
        direction_weight: f64,
        two_qubit_error_weight: f64,
        readout_error_weight: f64,
    ) -> Self {
        LayoutObjective {
            distance_weight,
            direction_weight,
            two_qubit_error_weight,
            readout_error_weight,
        }
        .into()
    }

    /// Returns the topology-only objective.
    #[staticmethod]
    fn topology_only() -> Self {
        LayoutObjective::topology_only().into()
    }

    /// Returns the default fidelity-aware objective.
    #[staticmethod]
    fn fidelity_aware() -> Self {
        LayoutObjective::fidelity_aware().into()
    }

    /// Selects a fidelity-aware objective when the device has usable calibration data.
    #[staticmethod]
    fn auto_from_device(device: PyRef<'_, PyDevice>) -> PyResult<Self> {
        let physical =
            build_physical_layout_graph(&device.inner).map_err(compiler_error_to_py_err)?;
        Ok(LayoutObjective::auto_from_physical(&physical).into())
    }

    /// Returns a fidelity-aware objective, rejecting devices without usable calibration data.
    #[staticmethod]
    fn fidelity_required(device: PyRef<'_, PyDevice>) -> PyResult<Self> {
        let physical =
            build_physical_layout_graph(&device.inner).map_err(compiler_error_to_py_err)?;
        LayoutObjective::fidelity_required(&physical)
            .map(Into::into)
            .map_err(compiler_error_to_py_err)
    }

    #[getter]
    fn distance_weight(&self) -> f64 {
        self.inner.distance_weight
    }

    #[getter]
    fn direction_weight(&self) -> f64 {
        self.inner.direction_weight
    }

    #[getter]
    fn two_qubit_error_weight(&self) -> f64 {
        self.inner.two_qubit_error_weight
    }

    #[getter]
    fn readout_error_weight(&self) -> f64 {
        self.inner.readout_error_weight
    }

    #[getter]
    fn uses_fidelity(&self) -> bool {
        self.inner.uses_fidelity()
    }

    fn __repr__(&self) -> String {
        format!(
            "LayoutObjective(distance_weight={}, direction_weight={}, two_qubit_error_weight={}, readout_error_weight={})",
            self.inner.distance_weight,
            self.inner.direction_weight,
            self.inner.two_qubit_error_weight,
            self.inner.readout_error_weight,
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

/// Breakdown of a layout objective score.
#[pyclass(
    name = "LayoutScore",
    module = "cqlib.compile.transform.layout",
    skip_from_py_object
)]
#[derive(Clone, Debug)]
pub struct PyLayoutScore {
    inner: LayoutScore,
}

impl From<LayoutScore> for PyLayoutScore {
    fn from(inner: LayoutScore) -> Self {
        Self { inner }
    }
}

#[pymethods]
impl PyLayoutScore {
    #[getter]
    fn total(&self) -> f64 {
        self.inner.total
    }

    #[getter]
    fn distance(&self) -> f64 {
        self.inner.distance
    }

    #[getter]
    fn direction(&self) -> f64 {
        self.inner.direction
    }

    #[getter]
    fn two_qubit_error(&self) -> f64 {
        self.inner.two_qubit_error
    }

    #[getter]
    fn readout_error(&self) -> f64 {
        self.inner.readout_error
    }

    #[getter]
    fn used_fidelity(&self) -> bool {
        self.inner.used_fidelity
    }

    fn __repr__(&self) -> String {
        format!(
            "LayoutScore(total={}, distance={}, direction={}, two_qubit_error={}, readout_error={}, used_fidelity={})",
            self.inner.total,
            self.inner.distance,
            self.inner.direction,
            self.inner.two_qubit_error,
            self.inner.readout_error,
            python_bool(self.inner.used_fidelity),
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

/// Diagnostics emitted by an initial-layout algorithm.
#[pyclass(
    name = "LayoutDiagnostics",
    module = "cqlib.compile.transform.layout",
    skip_from_py_object
)]
#[derive(Clone, Debug)]
pub struct PyLayoutDiagnostics {
    inner: LayoutDiagnostics,
}

impl From<LayoutDiagnostics> for PyLayoutDiagnostics {
    fn from(inner: LayoutDiagnostics) -> Self {
        Self { inner }
    }
}

#[pymethods]
impl PyLayoutDiagnostics {
    #[getter]
    fn is_perfect(&self) -> bool {
        self.inner.is_perfect
    }

    #[getter]
    fn candidates_evaluated(&self) -> usize {
        self.inner.candidates_evaluated
    }

    #[getter]
    fn used_fidelity(&self) -> bool {
        self.inner.used_fidelity
    }

    #[getter]
    fn notes(&self) -> Vec<String> {
        self.inner.notes.clone()
    }

    fn __repr__(&self) -> String {
        format!(
            "LayoutDiagnostics(is_perfect={}, candidates_evaluated={}, used_fidelity={}, notes={:?})",
            python_bool(self.inner.is_perfect),
            self.inner.candidates_evaluated,
            python_bool(self.inner.used_fidelity),
            self.inner.notes,
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

/// Selected initial layout, score, and diagnostics.
#[pyclass(
    name = "LayoutResult",
    module = "cqlib.compile.transform.layout",
    skip_from_py_object
)]
#[derive(Clone, Debug)]
pub struct PyLayoutResult {
    inner: LayoutResult,
}

impl From<LayoutResult> for PyLayoutResult {
    fn from(inner: LayoutResult) -> Self {
        Self { inner }
    }
}

#[pymethods]
impl PyLayoutResult {
    #[getter]
    fn layout(&self) -> PyLayout {
        self.inner.layout.clone().into()
    }

    #[getter]
    fn score(&self) -> Option<PyLayoutScore> {
        self.inner.score.clone().map(Into::into)
    }

    #[getter]
    fn diagnostics(&self) -> PyLayoutDiagnostics {
        self.inner.diagnostics.clone().into()
    }

    fn __repr__(&self) -> String {
        format!(
            "LayoutResult(layout={:?}, score={:?}, diagnostics={:?})",
            self.inner.layout, self.inner.score, self.inner.diagnostics,
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

/// Selects which logical interactions are hard topology constraints for VF2.
#[pyclass(
    name = "Vf2EdgeRequirement",
    module = "cqlib.compile.transform.layout",
    from_py_object
)]
#[derive(Clone, Copy, Debug)]
pub struct PyVf2EdgeRequirement {
    pub(crate) inner: Vf2EdgeRequirement,
}

impl From<Vf2EdgeRequirement> for PyVf2EdgeRequirement {
    fn from(inner: Vf2EdgeRequirement) -> Self {
        Self { inner }
    }
}

#[pymethods]
impl PyVf2EdgeRequirement {
    #[staticmethod]
    fn positive_interactions() -> Self {
        Vf2EdgeRequirement::PositiveInteractions.into()
    }

    #[staticmethod]
    fn all_interactions() -> Self {
        Vf2EdgeRequirement::AllInteractions.into()
    }

    fn __repr__(&self) -> &'static str {
        match self.inner {
            Vf2EdgeRequirement::PositiveInteractions => {
                "Vf2EdgeRequirement.positive_interactions()"
            }
            Vf2EdgeRequirement::AllInteractions => "Vf2EdgeRequirement.all_interactions()",
        }
    }

    fn __str__(&self) -> &'static str {
        match self.inner {
            Vf2EdgeRequirement::PositiveInteractions => "positive_interactions",
            Vf2EdgeRequirement::AllInteractions => "all_interactions",
        }
    }

    fn __eq__(&self, other: &Self) -> bool {
        self.inner == other.inner
    }

    fn __hash__(&self) -> u64 {
        let discriminant = match self.inner {
            Vf2EdgeRequirement::PositiveInteractions => 0_u8,
            Vf2EdgeRequirement::AllInteractions => 1,
        };
        let mut hasher = DefaultHasher::new();
        discriminant.hash(&mut hasher);
        hasher.finish()
    }

    fn __copy__(&self) -> Self {
        *self
    }

    fn __deepcopy__(&self, _memo: &Bound<'_, PyAny>) -> Self {
        *self
    }
}

/// Configuration for VF2 perfect-layout search.
#[pyclass(
    name = "Vf2LayoutConfig",
    module = "cqlib.compile.transform.layout",
    from_py_object
)]
#[derive(Clone, Debug)]
pub struct PyVf2LayoutConfig {
    pub(crate) inner: Vf2LayoutConfig,
}

impl From<Vf2LayoutConfig> for PyVf2LayoutConfig {
    fn from(inner: Vf2LayoutConfig) -> Self {
        Self { inner }
    }
}

#[pymethods]
impl PyVf2LayoutConfig {
    #[new]
    #[pyo3(signature = (*, candidate_limit=10, call_limit=None, edge_requirement=None))]
    fn new(
        candidate_limit: usize,
        call_limit: Option<usize>,
        edge_requirement: Option<PyVf2EdgeRequirement>,
    ) -> Self {
        Self {
            inner: Vf2LayoutConfig {
                candidate_limit,
                call_limit,
                edge_requirement: edge_requirement
                    .map_or(Vf2EdgeRequirement::PositiveInteractions, |requirement| {
                        requirement.inner
                    }),
            },
        }
    }

    #[getter]
    fn candidate_limit(&self) -> usize {
        self.inner.candidate_limit
    }

    #[getter]
    fn call_limit(&self) -> Option<usize> {
        self.inner.call_limit
    }

    #[getter]
    fn edge_requirement(&self) -> PyVf2EdgeRequirement {
        self.inner.edge_requirement.into()
    }

    fn __repr__(&self) -> String {
        format!(
            "Vf2LayoutConfig(candidate_limit={}, call_limit={:?}, edge_requirement={})",
            self.inner.candidate_limit,
            self.inner.call_limit,
            PyVf2EdgeRequirement::from(self.inner.edge_requirement).__repr__(),
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

const fn python_bool(value: bool) -> &'static str {
    if value { "True" } else { "False" }
}
