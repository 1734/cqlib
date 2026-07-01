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
use crate::device::device_impl::PyDevice;
use crate::device::layout::PyLayout;
use cqlib_core::compile::sabre::{
    SabreConfig, SabreHeuristicConfig, SabreRoutingDiagnostics, SabreRoutingResult,
    SabreTrialObjective, sabre_route,
};
use pyo3::prelude::*;
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};

/// Objective used to select the best result among independent SABRE trials.
#[pyclass(
    name = "SabreTrialObjective",
    module = "cqlib.compile.sabre",
    from_py_object
)]
#[derive(Clone, Copy, Debug)]
pub struct PySabreTrialObjective {
    inner: SabreTrialObjective,
}

impl From<SabreTrialObjective> for PySabreTrialObjective {
    fn from(inner: SabreTrialObjective) -> Self {
        Self { inner }
    }
}

#[pymethods]
impl PySabreTrialObjective {
    /// Selects the trial with the fewest inserted SWAPs.
    #[staticmethod]
    fn swap_count() -> Self {
        SabreTrialObjective::SwapCount.into()
    }

    /// Selects the trial with the smallest two-qubit depth.
    #[staticmethod]
    fn depth() -> Self {
        SabreTrialObjective::Depth.into()
    }

    /// Minimizes SWAP count first and two-qubit depth second.
    #[staticmethod]
    fn swap_then_depth() -> Self {
        SabreTrialObjective::SwapThenDepth.into()
    }

    /// Minimizes two-qubit depth first and SWAP count second.
    #[staticmethod]
    fn depth_then_swap() -> Self {
        SabreTrialObjective::DepthThenSwap.into()
    }

    fn __repr__(&self) -> &'static str {
        match self.inner {
            SabreTrialObjective::SwapCount => "SabreTrialObjective.swap_count()",
            SabreTrialObjective::Depth => "SabreTrialObjective.depth()",
            SabreTrialObjective::SwapThenDepth => "SabreTrialObjective.swap_then_depth()",
            SabreTrialObjective::DepthThenSwap => "SabreTrialObjective.depth_then_swap()",
        }
    }

    fn __str__(&self) -> &'static str {
        match self.inner {
            SabreTrialObjective::SwapCount => "swap_count",
            SabreTrialObjective::Depth => "depth",
            SabreTrialObjective::SwapThenDepth => "swap_then_depth",
            SabreTrialObjective::DepthThenSwap => "depth_then_swap",
        }
    }

    fn __eq__(&self, other: &Self) -> bool {
        self.inner == other.inner
    }

    fn __hash__(&self) -> u64 {
        let discriminant = match self.inner {
            SabreTrialObjective::SwapCount => 0_u8,
            SabreTrialObjective::Depth => 1,
            SabreTrialObjective::SwapThenDepth => 2,
            SabreTrialObjective::DepthThenSwap => 3,
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

/// Swap-selection weights and fallback limits used by SABRE.
#[pyclass(
    name = "SabreHeuristicConfig",
    module = "cqlib.compile.sabre",
    from_py_object
)]
#[derive(Clone, Debug)]
pub struct PySabreHeuristicConfig {
    inner: SabreHeuristicConfig,
}

impl From<SabreHeuristicConfig> for PySabreHeuristicConfig {
    fn from(inner: SabreHeuristicConfig) -> Self {
        Self { inner }
    }
}

#[pymethods]
impl PySabreHeuristicConfig {
    /// Creates a SABRE swap-selection configuration.
    #[new]
    #[pyo3(signature = (*, basic_weight=1.0, lookahead_weights=None, decay_increment=Some(0.001), decay_reset=5, attempt_limit=1000, best_epsilon=1e-10))]
    fn new(
        basic_weight: f64,
        lookahead_weights: Option<Vec<f64>>,
        decay_increment: Option<f64>,
        decay_reset: usize,
        attempt_limit: usize,
        best_epsilon: f64,
    ) -> Self {
        Self {
            inner: SabreHeuristicConfig {
                basic_weight,
                lookahead_weights: lookahead_weights.unwrap_or_else(|| vec![0.5]),
                decay_increment,
                decay_reset,
                attempt_limit,
                best_epsilon,
            },
        }
    }

    #[getter]
    fn basic_weight(&self) -> f64 {
        self.inner.basic_weight
    }

    #[getter]
    fn lookahead_weights(&self) -> Vec<f64> {
        self.inner.lookahead_weights.clone()
    }

    #[getter]
    fn decay_increment(&self) -> Option<f64> {
        self.inner.decay_increment
    }

    #[getter]
    fn decay_reset(&self) -> usize {
        self.inner.decay_reset
    }

    #[getter]
    fn attempt_limit(&self) -> usize {
        self.inner.attempt_limit
    }

    #[getter]
    fn best_epsilon(&self) -> f64 {
        self.inner.best_epsilon
    }

    fn __repr__(&self) -> String {
        format!(
            "SabreHeuristicConfig(basic_weight={}, lookahead_weights={:?}, decay_increment={:?}, decay_reset={}, attempt_limit={}, best_epsilon={})",
            self.inner.basic_weight,
            self.inner.lookahead_weights,
            self.inner.decay_increment,
            self.inner.decay_reset,
            self.inner.attempt_limit,
            self.inner.best_epsilon,
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

/// Configuration shared by SABRE layout refinement and routing.
#[pyclass(name = "SabreConfig", module = "cqlib.compile.sabre", from_py_object)]
#[derive(Clone, Debug)]
pub struct PySabreConfig {
    pub(crate) inner: SabreConfig,
}

impl From<SabreConfig> for PySabreConfig {
    fn from(inner: SabreConfig) -> Self {
        Self { inner }
    }
}

#[pymethods]
impl PySabreConfig {
    /// Creates a SABRE configuration using core defaults for omitted objects.
    #[new]
    #[pyo3(signature = (*, layout_trials=10, refinement_iterations=1, layout_scoring_trials=1, routing_trials=5, trial_objective=None, seed=None, heuristic=None))]
    fn new(
        layout_trials: usize,
        refinement_iterations: usize,
        layout_scoring_trials: usize,
        routing_trials: usize,
        trial_objective: Option<PySabreTrialObjective>,
        seed: Option<u64>,
        heuristic: Option<PySabreHeuristicConfig>,
    ) -> Self {
        Self {
            inner: SabreConfig {
                layout_trials,
                refinement_iterations,
                layout_scoring_trials,
                routing_trials,
                trial_objective: trial_objective
                    .map_or(SabreTrialObjective::SwapThenDepth, |objective| {
                        objective.inner
                    }),
                seed,
                heuristic: heuristic
                    .map_or_else(SabreHeuristicConfig::default, |value| value.inner),
            },
        }
    }

    /// Returns a compact deterministic configuration for tests and examples.
    #[staticmethod]
    fn deterministic_seeded(seed: u64) -> Self {
        SabreConfig::deterministic_seeded(seed).into()
    }

    #[getter]
    fn layout_trials(&self) -> usize {
        self.inner.layout_trials
    }

    #[getter]
    fn refinement_iterations(&self) -> usize {
        self.inner.refinement_iterations
    }

    #[getter]
    fn layout_scoring_trials(&self) -> usize {
        self.inner.layout_scoring_trials
    }

    #[getter]
    fn routing_trials(&self) -> usize {
        self.inner.routing_trials
    }

    #[getter]
    fn trial_objective(&self) -> PySabreTrialObjective {
        self.inner.trial_objective.into()
    }

    #[getter]
    fn seed(&self) -> Option<u64> {
        self.inner.seed
    }

    #[getter]
    fn heuristic(&self) -> PySabreHeuristicConfig {
        self.inner.heuristic.clone().into()
    }

    fn __repr__(&self) -> String {
        format!(
            "SabreConfig(layout_trials={}, refinement_iterations={}, layout_scoring_trials={}, routing_trials={}, trial_objective={}, seed={:?}, heuristic={:?})",
            self.inner.layout_trials,
            self.inner.refinement_iterations,
            self.inner.layout_scoring_trials,
            self.inner.routing_trials,
            PySabreTrialObjective::from(self.inner.trial_objective).__repr__(),
            self.inner.seed,
            self.inner.heuristic,
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

/// Diagnostics emitted by a completed SABRE routing run.
#[pyclass(
    name = "SabreRoutingDiagnostics",
    module = "cqlib.compile.sabre",
    skip_from_py_object
)]
#[derive(Clone, Debug)]
pub struct PySabreRoutingDiagnostics {
    inner: SabreRoutingDiagnostics,
}

impl From<SabreRoutingDiagnostics> for PySabreRoutingDiagnostics {
    fn from(inner: SabreRoutingDiagnostics) -> Self {
        Self { inner }
    }
}

#[pymethods]
impl PySabreRoutingDiagnostics {
    #[getter]
    fn trials_evaluated(&self) -> usize {
        self.inner.trials_evaluated
    }

    #[getter]
    fn selected_trial_index(&self) -> usize {
        self.inner.selected_trial_index
    }

    #[getter]
    fn fallback_count(&self) -> usize {
        self.inner.fallback_count
    }

    #[getter]
    fn control_flow_blocks_routed(&self) -> usize {
        self.inner.control_flow_blocks_routed
    }

    #[getter]
    fn two_qubit_depth(&self) -> usize {
        self.inner.two_qubit_depth
    }

    #[getter]
    fn operation_count(&self) -> usize {
        self.inner.operation_count
    }

    fn __repr__(&self) -> String {
        format!(
            "SabreRoutingDiagnostics(trials_evaluated={}, selected_trial_index={}, fallback_count={}, control_flow_blocks_routed={}, two_qubit_depth={}, operation_count={})",
            self.inner.trials_evaluated,
            self.inner.selected_trial_index,
            self.inner.fallback_count,
            self.inner.control_flow_blocks_routed,
            self.inner.two_qubit_depth,
            self.inner.operation_count,
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

/// Routed circuit, selected layouts, and routing diagnostics.
#[pyclass(
    name = "SabreRoutingResult",
    module = "cqlib.compile.sabre",
    skip_from_py_object
)]
#[derive(Clone, Debug)]
pub struct PySabreRoutingResult {
    inner: SabreRoutingResult,
}

impl From<SabreRoutingResult> for PySabreRoutingResult {
    fn from(inner: SabreRoutingResult) -> Self {
        Self { inner }
    }
}

#[pymethods]
impl PySabreRoutingResult {
    #[getter]
    fn circuit(&self) -> PyCircuit {
        self.inner.circuit.clone().into()
    }

    #[getter]
    fn initial_layout(&self) -> PyLayout {
        self.inner.initial_layout.clone().into()
    }

    #[getter]
    fn final_layout(&self) -> PyLayout {
        self.inner.final_layout.clone().into()
    }

    #[getter]
    fn swap_count(&self) -> usize {
        self.inner.swap_count
    }

    #[getter]
    fn diagnostics(&self) -> PySabreRoutingDiagnostics {
        self.inner.diagnostics.clone().into()
    }

    fn __repr__(&self) -> String {
        format!(
            "SabreRoutingResult(swap_count={}, diagnostics={:?})",
            self.inner.swap_count, self.inner.diagnostics
        )
    }

    fn __copy__(&self) -> Self {
        self.clone()
    }

    fn __deepcopy__(&self, _memo: &Bound<'_, PyAny>) -> Self {
        self.clone()
    }
}

/// Routes a circuit onto a device topology with the SABRE heuristic.
#[pyfunction(name = "sabre_route")]
#[pyo3(signature = (circuit, device, initial_layout, config=None))]
pub fn py_sabre_route(
    py: Python<'_>,
    circuit: PyRef<'_, PyCircuit>,
    device: PyRef<'_, PyDevice>,
    initial_layout: PyRef<'_, PyLayout>,
    config: Option<PySabreConfig>,
) -> PyResult<PySabreRoutingResult> {
    let circuit = circuit.inner.clone();
    let device = device.inner.clone();
    let initial_layout = initial_layout.inner.clone();
    let config = config.map_or_else(SabreConfig::default, |value| value.inner);

    py.detach(move || sabre_route(&circuit, &device, &initial_layout, &config))
        .map(PySabreRoutingResult::from)
        .map_err(compiler_error_to_py_err)
}
