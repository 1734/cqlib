// This code is part of Cqlib.
//
// (C) Copyright China Telecom Quantum Group 2025-2026
//
// This code is licensed under the Apache License, Version 2.0. You may
// obtain a copy of this license in the LICENSE.txt file in the root directory
// of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
//
// Any modifications or derivative works of this code must retain this
// copyright notice, and modified files need to carry a notice indicating
// that they have been altered from the originals.

use super::cost::{CalibrationEstimator, NativePlanCost, RobustDurationKey, RobustErrorKey};
use super::dag::{SabreControlFlow, SabreDag, SabreNode, SabreNodeKind};
use super::heuristic::{SabreConfig, SabreHeuristicConfig, SabreTrialObjective};
use super::layer::{Layer, RequirementPlacement};
use crate::circuit::value_instruction::storage_operation_to_value;
use crate::circuit::{
    Circuit, CircuitParam, ClassicalControlOp, ControlBody, ForOp, IfOp, Instruction, Operation,
    Parameter, Qubit, StandardGate, SwitchCase, SwitchOp, WhileOp,
};
use crate::compile::device_planning::{DeviceGateState, NativePlanAvailability, NativePlanCatalog};
use crate::compile::error::DeviceLoweringFailure;
use crate::compile::knowledge::KnowledgeInstructionKey;
use crate::compile::physical_target::PhysicalLayoutGraph;
use crate::compile::{CompilerError, SabreRoutingFailure};
use crate::device::{Device, Layout, LogicalQubit, PhysicalQubit};
use indexmap::IndexSet;
use rand::rngs::StdRng;
use rand::seq::IndexedRandom;
use rand::{Rng, SeedableRng};
use rayon::prelude::*;
use rustworkx_core::petgraph::Direction;
use rustworkx_core::petgraph::graph::{NodeIndex, UnGraph};
use rustworkx_core::petgraph::visit::EdgeRef;
use rustworkx_core::token_swapper::token_swapper;
use smallvec::{SmallVec, smallvec};
use std::cmp::Ordering;
use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet, VecDeque};
use std::sync::{Arc, Mutex};

const CONTROL_FLOW_EPILOGUE_TRIALS: usize = 4;
const EAGER_PAIR_STATE_BUDGET: usize = 1_000_000;
const LAZY_PAIR_CACHE_BUDGET: usize = 100_000;

/// Routed circuit and layout metadata produced by [`sabre_route`].
#[derive(Debug, Clone)]
pub struct SabreRoutingResult {
    /// Physical circuit with inserted SWAP operations.
    pub circuit: Circuit,
    /// Initial logical-to-physical layout used by the selected trial.
    pub initial_layout: Layout,
    /// Final logical-to-physical layout after all routed operations.
    pub final_layout: Layout,
    /// Number of inserted SWAP operations, including control-flow epilogues.
    pub swap_count: usize,
    /// Diagnostics describing routing search behavior.
    pub diagnostics: SabreRoutingDiagnostics,
}

/// Diagnostics emitted by SABRE routing.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct SabreRoutingDiagnostics {
    /// Number of routing trials evaluated.
    pub trials_evaluated: usize,
    /// Zero-based index of the selected routing trial.
    pub selected_trial_index: usize,
    /// Number of times the shortest-path fallback was used.
    pub fallback_count: usize,
    /// Number of recursively routed control-flow blocks.
    pub control_flow_blocks_routed: usize,
    /// ASAP two-qubit depth of the selected routed operation stream.
    pub two_qubit_depth: usize,
    /// Total number of operations in the selected routed operation stream.
    pub operation_count: usize,
    /// Predicted native two-qubit operation count after device lowering.
    pub native_two_qubit_count: usize,
    /// ASAP two-qubit depth using the selected exact-qargs native plans.
    pub native_two_qubit_depth: usize,
    /// Predicted total native operation count after device lowering.
    pub native_operation_count: usize,
    /// Independent-gate negative log-success proxy, when calibration exists.
    pub predicted_log_error: Option<f64>,
    /// Native leaves whose error calibration could not be estimated.
    pub unavailable_error_count: u32,
    /// Native leaves using conservative error imputation.
    pub imputed_error_count: u32,
    /// Sum of native gate durations, when duration calibration exists.
    ///
    /// This is workload, not an ASAP-scheduled circuit makespan.
    pub duration_work: Option<f64>,
    /// Native-leaf ASAP makespan using exact-qargs selected lowering plans.
    /// `None` means duration calibration is disabled/incomplete or a dynamic
    /// non-zero loop has no statically known execution count.
    pub predicted_makespan: Option<f64>,
    /// Dynamic `for`/`while` loops whose total execution cost is unknown.
    pub unknown_loop_count: usize,
    /// Number of distinct unary/pair requirement signatures prepared.
    pub requirement_signature_count: usize,
    /// Pair-placement lower-bound states retained eagerly by the target.
    pub eager_pair_state_count: usize,
    /// Exact pair-state lower-bound lookups served by the lazy path.
    pub lazy_pair_lookup_count: usize,
    /// Lazy pair-state results retained in the bounded target cache.
    pub lazy_pair_cached_count: usize,
}

#[derive(Debug, Clone)]
pub(crate) struct TrialResult {
    pub(crate) operations: Vec<Operation>,
    pub(crate) final_layout: Layout,
    pub(crate) swap_count: usize,
    pub(crate) fallback_count: usize,
    pub(crate) control_flow_blocks_routed: usize,
    pub(crate) quality: TrialQuality,
}

#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub(crate) struct TrialQuality {
    pub(crate) swap_count: usize,
    pub(crate) two_qubit_depth: usize,
    pub(crate) operation_count: usize,
    pub(crate) native_two_qubit_ops: usize,
    pub(crate) native_two_qubit_depth: usize,
    pub(crate) native_total_ops: usize,
    pub(crate) error: Option<RobustErrorKey>,
    pub(crate) duration: Option<RobustDurationKey>,
    pub(crate) makespan: Option<f64>,
    pub(crate) unknown_loop_count: usize,
}

/// Routes `circuit` from `initial_layout` by inserting SWAP operations.
///
/// The returned circuit uses physical qubit identifiers as its circuit qubits.
/// Two-qubit operations in the routed circuit are adjacent in the usable
/// physical topology.  Control-flow bodies are routed recursively and restored
/// to their entry layout before leaving the block.
pub fn sabre_route(
    circuit: &Circuit,
    device: &Device,
    initial_layout: &Layout,
    config: &SabreConfig,
) -> Result<SabreRoutingResult, CompilerError> {
    validate_config(config)?;
    // Build a dense, reusable view of the physical topology once. The routing
    // loop indexes into this structure heavily for adjacency, distance, and
    // deterministic candidate ordering.
    let physical = PhysicalLayoutGraph::from_device(device)?;
    let sabre = SabreDag::from_operations(circuit.operations())?;
    let target = RoutingTarget::from_device(device, &physical, &sabre)?;
    let logical_qubits = circuit
        .qubits()
        .into_iter()
        .map(LogicalQubit::from_qubit)
        .collect::<Vec<_>>();
    let initial_layout =
        normalize_initial_layout_for_target(&logical_qubits, &target, initial_layout)?;
    validate_reachable_interactions_for_target(&sabre, &target, &initial_layout)?;

    // Trials share the normalized layout and DAG but use independent seeds for
    // tie-breaking. Selection stays deterministic for a configured seed because
    // result comparison falls back to the trial index.
    let trial_results = trial_seeds(config.seed, config.routing_trials)
        .into_par_iter()
        .enumerate()
        .map(|(index, seed)| {
            let heuristic = trial_heuristic_profile(&config.heuristic, index);
            route_trial_unchecked(&sabre, &target, &initial_layout, &heuristic, seed)
                .map(|result| (index, result))
        })
        .collect::<Result<Vec<_>, CompilerError>>()?;
    let swap_limit = trial_swap_limit(
        config.trial_objective,
        config.swap_regret_ratio,
        trial_results.iter().map(|(_, result)| result.quality),
    );
    let (best_index, best) = trial_results
        .into_iter()
        .filter(|(_, result)| result.quality.swap_count <= swap_limit)
        .min_by(|(left_index, left), (right_index, right)| {
            compare_trial_quality(
                config.trial_objective,
                left.quality,
                *left_index,
                right.quality,
                *right_index,
            )
        })
        .expect("routing_trials is validated to be non-zero");

    // Routing rewrites operation qubits but keeps symbolic parameters by index.
    // Rebuild the routed circuit's parameter table in first-use order, then
    // remap nested control-flow bodies to the new table.
    let mut parameter_order = IndexSet::<Parameter>::new();
    for operation in &best.operations {
        for param in &operation.params {
            if let CircuitParam::Index(index) = param {
                let parameter = circuit
                    .parameters()
                    .get_index(*index as usize)
                    .cloned()
                    .ok_or(crate::circuit::CircuitError::InvalidParameterIndex(*index))?;
                parameter_order.insert(parameter);
            }
        }
    }
    for parameter in circuit.parameters() {
        parameter_order.insert(parameter.clone());
    }
    let parameter_indices = circuit
        .parameters()
        .iter()
        .map(|parameter| {
            parameter_order
                .get_index_of(parameter)
                .expect("source parameters are included in routed parameter order")
                as u32
        })
        .collect::<Vec<_>>();
    let mapped_operations = best
        .operations
        .iter()
        .map(|operation| remap_parameter_indices(operation, &parameter_indices))
        .collect::<Result<Vec<_>, _>>()?;
    let routed_operations = mapped_operations
        .iter()
        .map(|operation| {
            storage_operation_to_value(operation.clone(), &|param| match param {
                CircuitParam::Fixed(value) => Ok((*value).into()),
                CircuitParam::Index(index) => parameter_order
                    .get_index(*index as usize)
                    .cloned()
                    .map(Into::into)
                    .ok_or(crate::circuit::CircuitError::InvalidParameterIndex(*index)),
            })
        })
        .collect::<Result<Vec<_>, crate::circuit::CircuitError>>()?;
    let mut routed = Circuit::from_operations(
        target
            .physical_qubits
            .iter()
            .copied()
            .map(PhysicalQubit::qubit)
            .collect(),
        routed_operations,
        Some(circuit.classical_vars().to_vec()),
        Some(circuit.classical_values().to_vec()),
    )?;
    for parameter in parameter_order {
        routed.add_parameter(parameter);
    }
    routed.set_global_phase(circuit.global_phase());
    let (lazy_pair_lookup_count, lazy_pair_cached_count) = target.lazy_pair_cache_stats();

    Ok(SabreRoutingResult {
        circuit: routed,
        initial_layout,
        final_layout: best.final_layout,
        swap_count: best.swap_count,
        diagnostics: SabreRoutingDiagnostics {
            trials_evaluated: config.routing_trials,
            selected_trial_index: best_index,
            fallback_count: best.fallback_count,
            control_flow_blocks_routed: best.control_flow_blocks_routed,
            two_qubit_depth: best.quality.two_qubit_depth,
            operation_count: best.quality.operation_count,
            native_two_qubit_count: best.quality.native_two_qubit_ops,
            native_two_qubit_depth: best.quality.native_two_qubit_depth,
            native_operation_count: best.quality.native_total_ops,
            predicted_log_error: best.quality.error.map(|key| key.log_error),
            unavailable_error_count: best.quality.error.map_or(0, |key| key.unavailable_count),
            imputed_error_count: best.quality.error.map_or(0, |key| key.imputed_count),
            duration_work: best.quality.duration.map(|key| key.duration_work),
            predicted_makespan: best.quality.makespan,
            unknown_loop_count: best.quality.unknown_loop_count,
            requirement_signature_count: target.requirements.len(),
            eager_pair_state_count: target.eager_pair_state_count,
            lazy_pair_lookup_count,
            lazy_pair_cached_count,
        },
    })
}

/// Validates a SABRE routing configuration.
///
/// This check intentionally ignores layout-refinement fields such as
/// [`SabreConfig::layout_trials`] and [`SabreConfig::layout_scoring_trials`].
/// Routing starts from a concrete initial layout and does not depend on those
/// layout-only knobs.
pub fn validate_config(config: &SabreConfig) -> Result<(), CompilerError> {
    if config.routing_trials == 0 {
        return Err(CompilerError::InvalidInput(
            "sabre routing_trials must be greater than zero".to_string(),
        ));
    }
    if !(config.swap_regret_ratio.is_finite() && config.swap_regret_ratio >= 0.0) {
        return Err(CompilerError::InvalidInput(
            "sabre swap_regret_ratio must be finite and non-negative".to_string(),
        ));
    }
    config.heuristic.validate()
}

pub(crate) fn route_trial(
    sabre: &SabreDag,
    target: &RoutingTarget,
    initial_layout: &Layout,
    heuristic: &SabreHeuristicConfig,
    seed: u64,
) -> Result<TrialResult, CompilerError> {
    validate_reachable_interactions_for_target(sabre, target, initial_layout)?;
    route_trial_unchecked(sabre, target, initial_layout, heuristic, seed)
}

pub(crate) fn route_trial_unchecked(
    sabre: &SabreDag,
    target: &RoutingTarget,
    initial_layout: &Layout,
    heuristic: &SabreHeuristicConfig,
    seed: u64,
) -> Result<TrialResult, CompilerError> {
    let mut output = TrialOutput::new(seed);
    let mut state = RoutingState::new(sabre, target, initial_layout.clone(), heuristic, seed);

    // Initial operations are dependency-free one-qubit or non-quantum work.
    // They can be emitted immediately under the starting layout.
    for operation in &sabre.initial {
        output
            .operations
            .push(map_operation(operation, &state.layout)?);
    }

    state.update_route(
        sabre,
        target,
        heuristic,
        &mut output,
        &sabre.first_layer,
        None,
    )?;
    state.populate_extended_set(sabre, target)?;

    let mut routable_nodes = Vec::with_capacity(2);
    let mut search_steps_since_decay_reset = 0usize;
    while !state.front_layer.is_empty() {
        let mut current_swaps = Vec::new();
        let mut seen_mappings = HashSet::from([mapping_signature(&state.layout, target)]);
        let mut repeated_mapping = false;
        // Search accumulates speculative SWAPs until at least one front-layer
        // node becomes adjacent. Those SWAPs are emitted only when the routed
        // node is actually emitted, preserving a compact operation stream.
        while routable_nodes.is_empty()
            && !repeated_mapping
            && current_swaps.len() < heuristic.attempt_limit
        {
            let best_swap =
                state.choose_best_swap(target, heuristic, current_swaps.last().copied())?;
            state.apply_swap(best_swap.physical, target)?;
            current_swaps.push(best_swap.physical);
            repeated_mapping = !seen_mappings.insert(mapping_signature(&state.layout, target));
            let executable =
                |requirement, placement| target.terminal_cost_for(requirement, placement).is_some();
            for candidate in best_swap
                .indices
                .into_iter()
                .filter_map(|index| state.front_layer.routable_node_on_index(index, &executable))
            {
                if !routable_nodes.contains(&candidate) {
                    routable_nodes.push(candidate);
                }
            }

            if let Some(increment) = heuristic.decay_increment {
                search_steps_since_decay_reset += 1;
                if search_steps_since_decay_reset >= heuristic.decay_reset {
                    for value in &mut state.decay {
                        *value = 1.0;
                    }
                    search_steps_since_decay_reset = 0;
                } else {
                    state.decay[best_swap.indices[0]] += increment;
                    state.decay[best_swap.indices[1]] += increment;
                }
            }
        }

        if routable_nodes.is_empty() {
            // The heuristic failed to make progress within its attempt budget.
            // Roll back speculative swaps, then force progress along a shortest
            // path so the router cannot livelock on a poor local score.
            for swap in current_swaps.drain(..).rev() {
                state.apply_swap(swap, target)?;
            }
            output.fallback_count += 1;
            let forced = state.force_enable_closest_node(target, &mut current_swaps)?;
            routable_nodes.extend(forced);
        }

        let distance = |requirement, placement| target.distance_for(requirement, placement);
        for node in &routable_nodes {
            state.front_layer.remove(*node, &distance)?;
        }
        state.update_route(
            sabre,
            target,
            heuristic,
            &mut output,
            &routable_nodes,
            Some(current_swaps),
        )?;
        state.lookahead_layers.iter_mut().for_each(Layer::clear);
        state.populate_extended_set(sabre, target)?;
        if heuristic.decay_increment.is_some() {
            for value in &mut state.decay {
                *value = 1.0;
            }
        }
        routable_nodes.clear();
    }

    let quality = trial_quality(&output.operations, output.swap_count, target)?;
    Ok(TrialResult {
        operations: output.operations,
        final_layout: state.layout,
        swap_count: output.swap_count,
        fallback_count: output.fallback_count,
        control_flow_blocks_routed: output.control_flow_blocks_routed,
        quality,
    })
}

/// Derives per-trial seeds from an optional workflow seed.
///
/// A configured seed seeds the seed generator, not every trial directly. This
/// gives reproducible but distinct tie-breaking streams per trial.
pub(crate) fn trial_seeds(seed: Option<u64>, count: usize) -> Vec<u64> {
    let mut rng = StdRng::seed_from_u64(seed.unwrap_or_else(rand::random));
    (0..count).map(|_| rng.random()).collect()
}

/// Produces deterministic, bounded heuristic diversity across routing trials.
///
/// Seed-only trials explore near-ties but share the same blind spots. These
/// profiles vary lookahead and decay without changing the common routing
/// engine or final constrained quality objective.
pub(crate) fn trial_heuristic_profile(
    base: &SabreHeuristicConfig,
    trial_index: usize,
) -> SabreHeuristicConfig {
    let mut profile = base.clone();
    match trial_index % 4 {
        0 => {}
        1 => {
            profile.decay_increment = None;
        }
        2 => {
            for weight in &mut profile.lookahead_weights {
                *weight *= 0.5;
            }
        }
        3 => {
            for weight in &mut profile.lookahead_weights {
                *weight *= 1.5;
            }
            if let Some(increment) = &mut profile.decay_increment {
                *increment *= 2.0;
            }
        }
        _ => unreachable!(),
    }
    profile
}

/// Normalizes an initial layout against a device's usable physical topology.
///
/// The returned layout contains the supplied logical qubits and every usable
/// physical qubit from `device`. Logical qubits must already be mapped by
/// `initial_layout`; extra usable physical qubits remain vacant.
pub fn normalize_initial_layout(
    logical_qubits: &[LogicalQubit],
    device: &Device,
    initial_layout: &Layout,
) -> Result<Layout, CompilerError> {
    let physical = PhysicalLayoutGraph::from_device(device)?;
    let target = RoutingTarget::from_physical(&physical)?;
    normalize_initial_layout_for_target(logical_qubits, &target, initial_layout)
}

pub(crate) fn normalize_initial_layout_for_target(
    logical_qubits: &[LogicalQubit],
    target: &RoutingTarget,
    initial_layout: &Layout,
) -> Result<Layout, CompilerError> {
    let mut mapping = BTreeMap::new();
    for logical in logical_qubits {
        let physical = initial_layout.get_physical(*logical).ok_or_else(|| {
            CompilerError::InvalidInput(format!(
                "sabre initial layout does not map logical qubit {logical}"
            ))
        })?;
        if !target.physical_set.contains(&physical) {
            return Err(CompilerError::InvalidInput(format!(
                "sabre initial layout maps logical qubit {logical} to unusable physical qubit {physical}"
            )));
        }
        mapping.insert(*logical, physical);
    }
    Layout::new(
        logical_qubits.to_vec(),
        target.physical_qubits.clone(),
        Some(mapping),
    )
    .map_err(|error| {
        CompilerError::InvalidInput(format!("sabre initial layout is invalid: {error}"))
    })
}

/// Validates that every two-qubit interaction in `circuit` is reachable.
///
/// The check uses the usable physical topology of `device` and the logical to
/// physical mapping in `initial_layout`. It validates reachability, not current
/// adjacency; non-adjacent but connected interactions can still be routed by
/// SABRE.
pub fn validate_reachable_interactions(
    circuit: &Circuit,
    device: &Device,
    initial_layout: &Layout,
) -> Result<(), CompilerError> {
    let physical = PhysicalLayoutGraph::from_device(device)?;
    let sabre = SabreDag::from_operations(circuit.operations())?;
    let target = RoutingTarget::from_device(device, &physical, &sabre)?;
    let logical_qubits = circuit
        .qubits()
        .into_iter()
        .map(LogicalQubit::from_qubit)
        .collect::<Vec<_>>();
    let initial_layout =
        normalize_initial_layout_for_target(&logical_qubits, &target, initial_layout)?;
    validate_reachable_interactions_for_target(&sabre, &target, &initial_layout)
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct InteractionOperation {
    instruction: KnowledgeInstructionKey,
    qarg_roles: SmallVec<[u8; 2]>,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
enum InteractionSignature {
    /// Layout refinement DAGs deliberately omit concrete operations.
    GenericPair,
    Unary(Vec<InteractionOperation>),
    Pair(Vec<InteractionOperation>),
}

#[derive(Debug, Clone)]
enum RequirementTable {
    Unary {
        terminals: Vec<Option<NativePlanCost>>,
        lower_bounds: Vec<Option<RouteLowerBound>>,
    },
    Pair {
        terminals: BTreeMap<[usize; 2], NativePlanCost>,
        lower_bounds: Option<PairStateTable<RouteLowerBound>>,
    },
}

#[derive(Debug, Clone)]
struct PairStateTable<T> {
    width: usize,
    values: Vec<Option<T>>,
}

impl<T: Copy> PairStateTable<T> {
    fn new(width: usize) -> Self {
        Self {
            width,
            values: vec![None; width.saturating_mul(width.saturating_sub(1))],
        }
    }

    fn get(&self, left: usize, right: usize) -> Option<T> {
        pair_state_index(self.width, left, right)
            .and_then(|index| self.values.get(index))
            .copied()
            .flatten()
    }

    fn set(&mut self, left: usize, right: usize, value: T) {
        if let Some(index) = pair_state_index(self.width, left, right) {
            self.values[index] = Some(value);
        }
    }

    fn state_count(&self) -> usize {
        self.values.len()
    }
}

fn pair_state_index(width: usize, left: usize, right: usize) -> Option<usize> {
    if width < 2 || left >= width || right >= width || left == right {
        return None;
    }
    let right_without_diagonal = if right < left { right } else { right - 1 };
    left.checked_mul(width - 1)?
        .checked_add(right_without_diagonal)
}

#[derive(Debug, Default)]
struct LazyPairCache {
    values: HashMap<(usize, usize, usize), Option<RouteLowerBound>>,
    lookup_count: usize,
}

#[derive(Debug, Clone, Copy, Default, PartialEq)]
struct RouteLowerBound {
    remaining_swaps: u32,
    native: NativePlanCost,
}

#[derive(Debug, Clone)]
struct TimedNativeLeaf {
    ordered_qargs: SmallVec<[PhysicalQubit; 2]>,
    duration: f64,
}

impl RouteLowerBound {
    fn with_swap(self, swap: NativePlanCost) -> Self {
        Self {
            remaining_swaps: self.remaining_swaps.saturating_add(1),
            native: swap.combine(self.native),
        }
    }

    fn combine(self, other: Self) -> Self {
        Self {
            remaining_swaps: self.remaining_swaps.saturating_add(other.remaining_swaps),
            native: self.native.combine(other.native),
        }
    }

    fn compare(self, other: Self) -> Ordering {
        self.remaining_swaps
            .cmp(&other.remaining_swaps)
            .then_with(|| compare_optional_native_cost(Some(self.native), Some(other.native)))
    }
}

#[derive(Debug, Clone)]
pub(crate) struct RoutingTarget {
    pub(crate) physical_qubits: Vec<PhysicalQubit>,
    physical_set: BTreeSet<PhysicalQubit>,
    physical_index: BTreeMap<PhysicalQubit, usize>,
    physical_order_indices: Vec<usize>,
    neighbors_by_index: Vec<Vec<usize>>,
    interaction_ids: HashMap<InteractionSignature, usize>,
    requirements: Vec<RequirementTable>,
    swap_costs: Vec<Vec<Option<NativePlanCost>>>,
    native_costs: HashMap<DeviceGateState, NativePlanCost>,
    native_timings: HashMap<DeviceGateState, Option<Vec<TimedNativeLeaf>>>,
    native_unsupported: HashMap<DeviceGateState, DeviceLoweringFailure>,
    native_cost_enabled: bool,
    native_duration_enabled: bool,
    eager_pair_state_count: usize,
    lazy_pair_cache: Arc<Mutex<LazyPairCache>>,
    graph: UnGraph<(), ()>,
    graph_index: BTreeMap<PhysicalQubit, NodeIndex>,
    physical_by_index: Vec<PhysicalQubit>,
}

struct PreparedRoutingParts {
    movement_edges: Vec<(usize, usize)>,
    interaction_ids: HashMap<InteractionSignature, usize>,
    requirements: Vec<RequirementTable>,
    swap_costs: Vec<Vec<Option<NativePlanCost>>>,
    native_costs: HashMap<DeviceGateState, NativePlanCost>,
    native_timings: HashMap<DeviceGateState, Option<Vec<TimedNativeLeaf>>>,
    native_unsupported: HashMap<DeviceGateState, DeviceLoweringFailure>,
    native_cost_enabled: bool,
    native_duration_enabled: bool,
}

impl RoutingTarget {
    /// Builds the dense routing view used by SABRE scoring.
    ///
    /// The target keeps both semantic physical-qubit ids and dense indices.
    /// Dense indices make layer scoring cheap; semantic ids keep diagnostics
    /// and emitted SWAP operations stable.
    pub(crate) fn from_physical(physical: &PhysicalLayoutGraph) -> Result<Self, CompilerError> {
        let edges = undirected_topology_edges(physical);
        let count = physical.physical_qubits().len();
        let neighbors = adjacency_from_edges(count, &edges);
        let topology_swap_costs = default_swap_costs(count, &edges);
        let mut generic_terminals = BTreeMap::new();
        for &(left, right) in &edges {
            generic_terminals.insert([left, right], NativePlanCost::default());
            generic_terminals.insert([right, left], NativePlanCost::default());
        }
        let mut pair_state_budget = EAGER_PAIR_STATE_BUDGET;
        Self::from_prepared_parts(
            physical,
            PreparedRoutingParts {
                movement_edges: edges,
                interaction_ids: HashMap::from([(InteractionSignature::GenericPair, 0)]),
                requirements: vec![RequirementTable::Pair {
                    lower_bounds: eager_pair_route_lower_bounds(
                        &neighbors,
                        &generic_terminals,
                        &topology_swap_costs,
                        &mut pair_state_budget,
                    ),
                    terminals: generic_terminals,
                }],
                swap_costs: topology_swap_costs,
                native_costs: HashMap::new(),
                native_timings: HashMap::new(),
                native_unsupported: HashMap::new(),
                native_cost_enabled: false,
                native_duration_enabled: false,
            },
        )
    }

    /// Builds a device-aware target whose movement edges are exactly those on
    /// which the final device lowerer can realize an emitted SWAP.
    pub(crate) fn from_device(
        device: &Device,
        physical: &PhysicalLayoutGraph,
        sabre: &SabreDag,
    ) -> Result<Self, CompilerError> {
        Self::from_device_with_pair_state_budget(device, physical, sabre, EAGER_PAIR_STATE_BUDGET)
    }

    fn from_device_with_pair_state_budget(
        device: &Device,
        physical: &PhysicalLayoutGraph,
        sabre: &SabreDag,
        pair_state_budget: usize,
    ) -> Result<Self, CompilerError> {
        let topology_edges = undirected_topology_edges(physical);
        let physical_qubits = physical.physical_qubits();
        let native_mode = device_declares_routing_capability(device, physical_qubits);
        let mut pair_state_budget = pair_state_budget;
        let signatures = ordered_interaction_signatures(sabre)?;
        let interaction_ids = signatures
            .iter()
            .cloned()
            .enumerate()
            .map(|(index, signature)| (signature, index))
            .collect::<HashMap<_, _>>();

        if !native_mode {
            return Self::from_prepared_parts(
                physical,
                prepare_topology_only_parts(
                    physical_qubits.len(),
                    topology_edges,
                    interaction_ids,
                    &signatures,
                    &mut pair_state_budget,
                ),
            );
        }

        let roots = collect_device_plan_roots(physical_qubits, &topology_edges, &signatures);

        let catalog = NativePlanCatalog::build(device, roots)?;
        let estimator = CalibrationEstimator::from_device(device, physical_qubits);
        let native_identity = estimator.identity_cost();
        let native_costs = catalog
            .iter()
            .map(|(state, summary)| (state.clone(), estimator.cost(summary)))
            .collect();
        let native_timings = catalog
            .iter()
            .map(|(state, summary)| {
                let leaves = summary
                    .leaves
                    .iter()
                    .map(|leaf| {
                        Some(TimedNativeLeaf {
                            ordered_qargs: leaf.ordered_qargs.clone(),
                            duration: estimator.leaf_duration(leaf)?,
                        })
                    })
                    .collect::<Option<Vec<_>>>();
                (state.clone(), leaves)
            })
            .collect();
        let native_unsupported = catalog
            .iter_availability()
            .filter_map(|(state, availability)| {
                let NativePlanAvailability::Unsupported(failure) = availability else {
                    return None;
                };
                Some((state.clone(), failure.clone()))
            })
            .collect();
        let count = physical_qubits.len();
        let mut movement_edges = Vec::new();
        let mut swap_costs = vec![vec![None; count]; count];
        for &(left_index, right_index) in &topology_edges {
            let state = DeviceGateState::standard(
                StandardGate::SWAP,
                smallvec![physical_qubits[left_index], physical_qubits[right_index]],
            );
            if let Some(summary) = catalog.summary(&state) {
                let cost = estimator.cost(summary);
                movement_edges.push((left_index, right_index));
                swap_costs[left_index][right_index] = Some(cost);
                swap_costs[right_index][left_index] = Some(cost);
            }
        }

        let neighbors = adjacency_from_edges(count, &movement_edges);
        let requirements = build_device_requirement_tables(
            &signatures,
            DeviceRequirementInputs {
                physical_qubits,
                topology_edges: &topology_edges,
                movement_edges: &movement_edges,
                neighbors: &neighbors,
                swap_costs: &swap_costs,
                catalog: &catalog,
                estimator: &estimator,
                native_identity,
            },
            &mut pair_state_budget,
        );

        Self::from_prepared_parts(
            physical,
            PreparedRoutingParts {
                movement_edges,
                interaction_ids,
                requirements,
                swap_costs,
                native_costs,
                native_timings,
                native_unsupported,
                native_cost_enabled: true,
                native_duration_enabled: estimator.duration_enabled(),
            },
        )
    }

    fn from_prepared_parts(
        physical: &PhysicalLayoutGraph,
        parts: PreparedRoutingParts,
    ) -> Result<Self, CompilerError> {
        let PreparedRoutingParts {
            movement_edges,
            interaction_ids,
            requirements,
            swap_costs,
            native_costs,
            native_timings,
            native_unsupported,
            native_cost_enabled,
            native_duration_enabled,
        } = parts;
        let physical_qubits = physical.physical_qubits().to_vec();
        let physical_set = physical_qubits.iter().copied().collect::<BTreeSet<_>>();
        let mut graph = UnGraph::with_capacity(physical_qubits.len(), 0);
        let mut graph_index = BTreeMap::new();
        let mut physical_index = BTreeMap::new();
        let mut physical_by_index = Vec::with_capacity(physical_qubits.len());

        for (dense_index, physical) in physical_qubits.iter().copied().enumerate() {
            let graph_node = graph.add_node(());
            graph_index.insert(physical, graph_node);
            physical_index.insert(physical, dense_index);
            physical_by_index.push(physical);
        }
        let mut neighbors_by_index = vec![Vec::new(); physical_qubits.len()];
        let mut physical_order_indices = (0..physical_qubits.len()).collect::<Vec<_>>();
        physical_order_indices.sort_unstable_by_key(|index| physical_qubits[*index]);

        for (left_index, right_index) in movement_edges {
            let left = physical_qubits[left_index];
            let right = physical_qubits[right_index];
            neighbors_by_index[left_index].push(right_index);
            neighbors_by_index[right_index].push(left_index);
            graph.add_edge(graph_index[&left], graph_index[&right], ());
        }
        for items in &mut neighbors_by_index {
            items.sort_unstable_by_key(|index| physical_qubits[*index]);
        }

        let eager_pair_state_count = requirements
            .iter()
            .filter_map(|requirement| match requirement {
                RequirementTable::Pair {
                    lower_bounds: Some(lower_bounds),
                    ..
                } => Some(lower_bounds.state_count()),
                RequirementTable::Unary { .. }
                | RequirementTable::Pair {
                    lower_bounds: None, ..
                } => None,
            })
            .sum();

        Ok(Self {
            physical_qubits,
            physical_set,
            physical_index,
            physical_order_indices,
            neighbors_by_index,
            interaction_ids,
            requirements,
            swap_costs,
            native_costs,
            native_timings,
            native_unsupported,
            native_cost_enabled,
            native_duration_enabled,
            eager_pair_state_count,
            lazy_pair_cache: Arc::new(Mutex::new(LazyPairCache::default())),
            graph,
            graph_index,
            physical_by_index,
        })
    }

    fn physical_index(&self, physical: PhysicalQubit) -> Result<usize, CompilerError> {
        self.physical_index.get(&physical).copied().ok_or_else(|| {
            CompilerError::InvalidInput(format!(
                "physical qubit {physical} is not usable in the target topology"
            ))
        })
    }

    fn physical_at(&self, index: usize) -> Result<PhysicalQubit, CompilerError> {
        self.physical_qubits.get(index).copied().ok_or_else(|| {
            CompilerError::InvariantViolation(format!(
                "physical index {index} is outside target topology of length {}",
                self.physical_qubits.len()
            ))
        })
    }

    fn distance_for(
        &self,
        requirement: usize,
        placement: RequirementPlacement,
    ) -> Result<f64, CompilerError> {
        self.distance_steps_for(requirement, placement)
            .map(|distance| f64::from(distance) + 1.0)
            .ok_or_else(|| {
                CompilerError::InvalidInput(format!(
                    "routing requirement {requirement} at {placement:?} cannot reach an executable terminal using lowerable SWAPs"
                ))
            })
    }

    fn terminal_cost_for(
        &self,
        requirement: usize,
        placement: RequirementPlacement,
    ) -> Option<NativePlanCost> {
        match (self.requirements.get(requirement)?, placement) {
            (RequirementTable::Unary { terminals, .. }, RequirementPlacement::Unary(physical)) => {
                terminals.get(physical).copied().flatten()
            }
            (
                RequirementTable::Pair { terminals, .. },
                RequirementPlacement::Pair([left, right]),
            ) => terminals.get(&[left, right]).copied(),
            _ => None,
        }
    }

    fn has_terminal(&self, requirement: usize) -> bool {
        match self.requirements.get(requirement) {
            Some(RequirementTable::Unary { terminals, .. }) => {
                terminals.iter().any(Option::is_some)
            }
            Some(RequirementTable::Pair { terminals, .. }) => !terminals.is_empty(),
            None => false,
        }
    }

    fn distance_steps_for(
        &self,
        requirement: usize,
        placement: RequirementPlacement,
    ) -> Option<u32> {
        self.route_lower_bound_for(requirement, placement)
            .map(|bound| bound.remaining_swaps)
    }

    fn route_lower_bound_for(
        &self,
        requirement: usize,
        placement: RequirementPlacement,
    ) -> Option<RouteLowerBound> {
        match (self.requirements.get(requirement)?, placement) {
            (
                RequirementTable::Unary { lower_bounds, .. },
                RequirementPlacement::Unary(physical),
            ) => lower_bounds.get(physical).copied().flatten(),
            (
                RequirementTable::Pair {
                    terminals,
                    lower_bounds,
                },
                RequirementPlacement::Pair([left, right]),
            ) => lower_bounds.as_ref().map_or_else(
                || self.lazy_pair_route_lower_bound(requirement, terminals, left, right),
                |bounds| bounds.get(left, right),
            ),
            _ => None,
        }
    }

    fn lazy_pair_route_lower_bound(
        &self,
        requirement: usize,
        terminals: &BTreeMap<[usize; 2], NativePlanCost>,
        left: usize,
        right: usize,
    ) -> Option<RouteLowerBound> {
        let key = (requirement, left, right);
        {
            let mut cache = self
                .lazy_pair_cache
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner());
            cache.lookup_count = cache.lookup_count.saturating_add(1);
            if let Some(value) = cache.values.get(&key) {
                return *value;
            }
        }

        let value = pair_route_lower_bound_from_state(
            &self.neighbors_by_index,
            terminals,
            &self.swap_costs,
            [left, right],
        );
        let mut cache = self
            .lazy_pair_cache
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        if cache.values.len() < LAZY_PAIR_CACHE_BUDGET {
            cache.values.entry(key).or_insert(value);
        }
        value
    }

    fn lazy_pair_cache_stats(&self) -> (usize, usize) {
        let cache = self
            .lazy_pair_cache
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        (cache.lookup_count, cache.values.len())
    }

    fn interaction_id_for_node(
        &self,
        sabre: &SabreDag,
        node: NodeIndex,
    ) -> Result<usize, CompilerError> {
        let signature = interaction_signature(&sabre.graph[node])?;
        self.interaction_ids
            .get(&signature)
            .copied()
            .or_else(|| {
                self.interaction_ids
                    .get(&InteractionSignature::GenericPair)
                    .copied()
            })
            .ok_or_else(|| {
                CompilerError::InvariantViolation(
                    "routing target has no generic interaction model".to_string(),
                )
            })
    }

    fn swap_cost(&self, left_index: usize, right_index: usize) -> Option<NativePlanCost> {
        self.swap_costs
            .get(left_index)
            .and_then(|row| row.get(right_index))
            .copied()
            .flatten()
    }

    fn native_cost(&self, state: &DeviceGateState) -> Option<NativePlanCost> {
        self.native_costs.get(state).copied()
    }

    fn native_timing(&self, state: &DeviceGateState) -> Option<&Option<Vec<TimedNativeLeaf>>> {
        self.native_timings.get(state)
    }

    fn unsupported_native_plan(&self, state: &DeviceGateState) -> Option<&DeviceLoweringFailure> {
        self.native_unsupported.get(state)
    }
}

fn undirected_topology_edges(physical: &PhysicalLayoutGraph) -> Vec<(usize, usize)> {
    physical.undirected_edges_by_index().collect()
}

fn device_declares_routing_capability(device: &Device, physical_qubits: &[PhysicalQubit]) -> bool {
    if !device.native_gates().is_empty() {
        return true;
    }
    if physical_qubits.iter().copied().any(|physical| {
        device
            .qubit_properties(physical)
            .is_some_and(|properties| !properties.native_instructions().is_empty())
    }) {
        return true;
    }
    let usable = physical_qubits.iter().copied().collect::<BTreeSet<_>>();
    device.topology().undirected_edges().any(|(left, right)| {
        usable.contains(&left)
            && usable.contains(&right)
            && [
                device.edge_properties(left, right),
                device.edge_properties(right, left),
            ]
            .into_iter()
            .flatten()
            .any(|properties| !properties.native_instructions().is_empty())
    })
}

fn interaction_signature(node: &SabreNode) -> Result<InteractionSignature, CompilerError> {
    let logicals: SmallVec<[LogicalQubit; 2]> = match &node.kind {
        SabreNodeKind::Unary(logical) => smallvec![*logical],
        SabreNodeKind::TwoQ(pair) => SmallVec::from_slice(pair),
        SabreNodeKind::Synchronize | SabreNodeKind::ControlFlow(_) => {
            return Ok(InteractionSignature::GenericPair);
        }
    };
    if node.operations.is_empty() {
        return Ok(InteractionSignature::GenericPair);
    }

    let mut operations = Vec::new();
    for operation in &node.operations {
        let Some(instruction) = KnowledgeInstructionKey::from_instruction(&operation.instruction)
        else {
            match operation.instruction {
                Instruction::UnitaryGate(_) | Instruction::CircuitGate(_) => {
                    return Err(CompilerError::InvalidInput(format!(
                        "sabre device preparation requires {} to be decomposed before routing",
                        operation.instruction
                    )));
                }
                _ => continue,
            }
        };
        if matches!(
            instruction,
            KnowledgeInstructionKey::Standard(StandardGate::GPhase)
        ) {
            continue;
        }
        let mut qarg_roles = SmallVec::new();
        for qubit in &operation.qubits {
            let logical = LogicalQubit::from_qubit(*qubit);
            let Some(role) = logicals.iter().position(|candidate| *candidate == logical) else {
                return Err(CompilerError::InvariantViolation(format!(
                    "folded SABRE node on {logicals:?} contains operation {} on unrelated logical qubit {logical}",
                    operation.instruction
                )));
            };
            qarg_roles.push(role as u8);
        }
        operations.push(InteractionOperation {
            instruction,
            qarg_roles,
        });
    }
    if operations.is_empty() {
        Ok(InteractionSignature::GenericPair)
    } else {
        Ok(match &node.kind {
            SabreNodeKind::Unary(_) => InteractionSignature::Unary(operations),
            SabreNodeKind::TwoQ(_) => InteractionSignature::Pair(operations),
            SabreNodeKind::Synchronize | SabreNodeKind::ControlFlow(_) => unreachable!(),
        })
    }
}

fn collect_interaction_signatures(
    sabre: &SabreDag,
    output: &mut HashSet<InteractionSignature>,
) -> Result<(), CompilerError> {
    for node in sabre.graph.node_weights() {
        match &node.kind {
            SabreNodeKind::Unary(_) | SabreNodeKind::TwoQ(_) => {
                output.insert(interaction_signature(node)?);
            }
            SabreNodeKind::ControlFlow(SabreControlFlow::If {
                then_body,
                else_body,
                ..
            }) => {
                collect_interaction_signatures(then_body, output)?;
                if let Some(else_body) = else_body {
                    collect_interaction_signatures(else_body, output)?;
                }
            }
            SabreNodeKind::ControlFlow(
                SabreControlFlow::While { body, .. } | SabreControlFlow::For { body, .. },
            ) => collect_interaction_signatures(body, output)?,
            SabreNodeKind::ControlFlow(SabreControlFlow::Switch { cases, default, .. }) => {
                for case in cases {
                    collect_interaction_signatures(&case.body, output)?;
                }
                if let Some(default) = default {
                    collect_interaction_signatures(default, output)?;
                }
            }
            SabreNodeKind::Synchronize => {}
        }
    }
    Ok(())
}

fn states_for_requirement(
    operations: &[InteractionOperation],
    ordered_qargs: &[PhysicalQubit],
) -> Vec<DeviceGateState> {
    operations
        .iter()
        .map(|operation| DeviceGateState {
            instruction: operation.instruction.clone(),
            ordered_qargs: operation
                .qarg_roles
                .iter()
                .map(|role| ordered_qargs[usize::from(*role)])
                .collect(),
        })
        .collect()
}

fn combined_catalog_cost(
    catalog: &NativePlanCatalog,
    estimator: &CalibrationEstimator,
    states: Vec<DeviceGateState>,
) -> Option<NativePlanCost> {
    let mut cost = None;
    for state in states {
        let summary = catalog.summary(&state)?;
        let next = estimator.cost(summary);
        cost = Some(cost.map_or(next, |current: NativePlanCost| current.combine(next)));
    }
    Some(cost.unwrap_or_default())
}

fn ordered_interaction_signatures(
    sabre: &SabreDag,
) -> Result<Vec<InteractionSignature>, CompilerError> {
    let mut signatures = HashSet::from([InteractionSignature::GenericPair]);
    collect_interaction_signatures(sabre, &mut signatures)?;
    let mut signatures = signatures.into_iter().collect::<Vec<_>>();
    signatures.sort_by_key(|signature| format!("{signature:?}"));
    if let Some(position) = signatures
        .iter()
        .position(|signature| matches!(signature, InteractionSignature::GenericPair))
    {
        signatures.swap(0, position);
    }
    Ok(signatures)
}

fn prepare_topology_only_parts(
    count: usize,
    topology_edges: Vec<(usize, usize)>,
    interaction_ids: HashMap<InteractionSignature, usize>,
    signatures: &[InteractionSignature],
    pair_state_budget: &mut usize,
) -> PreparedRoutingParts {
    let neighbors = adjacency_from_edges(count, &topology_edges);
    let topology_swap_costs = default_swap_costs(count, &topology_edges);
    let requirements = signatures
        .iter()
        .map(|signature| match signature {
            InteractionSignature::Unary(_) => {
                let terminals = vec![Some(NativePlanCost::default()); count];
                RequirementTable::Unary {
                    lower_bounds: unary_route_lower_bounds(
                        &neighbors,
                        &terminals,
                        &topology_swap_costs,
                    ),
                    terminals,
                }
            }
            InteractionSignature::GenericPair | InteractionSignature::Pair(_) => {
                let mut terminals = BTreeMap::new();
                for &(left, right) in &topology_edges {
                    terminals.insert([left, right], NativePlanCost::default());
                    terminals.insert([right, left], NativePlanCost::default());
                }
                RequirementTable::Pair {
                    lower_bounds: eager_pair_route_lower_bounds(
                        &neighbors,
                        &terminals,
                        &topology_swap_costs,
                        pair_state_budget,
                    ),
                    terminals,
                }
            }
        })
        .collect();
    PreparedRoutingParts {
        movement_edges: topology_edges,
        interaction_ids,
        requirements,
        swap_costs: topology_swap_costs,
        native_costs: HashMap::new(),
        native_timings: HashMap::new(),
        native_unsupported: HashMap::new(),
        native_cost_enabled: false,
        native_duration_enabled: false,
    }
}

fn collect_device_plan_roots(
    physical_qubits: &[PhysicalQubit],
    topology_edges: &[(usize, usize)],
    signatures: &[InteractionSignature],
) -> Vec<DeviceGateState> {
    let mut roots = Vec::new();
    for signature in signatures {
        if let InteractionSignature::Unary(operations) = signature {
            for &physical in physical_qubits {
                roots.extend(states_for_requirement(operations, &[physical]));
            }
        }
    }
    for &(left_index, right_index) in topology_edges {
        let left = physical_qubits[left_index];
        let right = physical_qubits[right_index];
        for ordered in [[left, right], [right, left]] {
            roots.push(DeviceGateState::standard(
                StandardGate::SWAP,
                SmallVec::from_slice(&ordered),
            ));
        }
        for signature in signatures {
            let InteractionSignature::Pair(operations) = signature else {
                continue;
            };
            for ordered in [[left, right], [right, left]] {
                roots.extend(states_for_requirement(operations, &ordered));
            }
        }
    }
    roots
}

struct DeviceRequirementInputs<'a> {
    physical_qubits: &'a [PhysicalQubit],
    topology_edges: &'a [(usize, usize)],
    movement_edges: &'a [(usize, usize)],
    neighbors: &'a [Vec<usize>],
    swap_costs: &'a [Vec<Option<NativePlanCost>>],
    catalog: &'a NativePlanCatalog,
    estimator: &'a CalibrationEstimator,
    native_identity: NativePlanCost,
}

fn build_device_requirement_tables(
    signatures: &[InteractionSignature],
    inputs: DeviceRequirementInputs<'_>,
    pair_state_budget: &mut usize,
) -> Vec<RequirementTable> {
    let DeviceRequirementInputs {
        physical_qubits,
        topology_edges,
        movement_edges,
        neighbors,
        swap_costs,
        catalog,
        estimator,
        native_identity,
    } = inputs;
    let count = physical_qubits.len();
    let mut requirements = Vec::with_capacity(signatures.len());
    for signature in signatures {
        match signature {
            InteractionSignature::GenericPair => {
                let mut terminals = BTreeMap::new();
                for &(left, right) in movement_edges {
                    terminals.insert([left, right], native_identity);
                    terminals.insert([right, left], native_identity);
                }
                requirements.push(RequirementTable::Pair {
                    lower_bounds: eager_pair_route_lower_bounds(
                        neighbors,
                        &terminals,
                        swap_costs,
                        pair_state_budget,
                    ),
                    terminals,
                });
            }
            InteractionSignature::Unary(operations) => {
                let mut terminals = vec![None; count];
                for (physical_index, physical) in physical_qubits.iter().copied().enumerate() {
                    let states = states_for_requirement(operations, &[physical]);
                    if let Some(cost) = combined_catalog_cost(catalog, estimator, states) {
                        terminals[physical_index] = Some(cost);
                    }
                }
                requirements.push(RequirementTable::Unary {
                    lower_bounds: unary_route_lower_bounds(neighbors, &terminals, swap_costs),
                    terminals,
                });
            }
            InteractionSignature::Pair(operations) => {
                let mut terminals = BTreeMap::new();
                for &(left_index, right_index) in topology_edges {
                    for (source, target) in [(left_index, right_index), (right_index, left_index)] {
                        let ordered = [physical_qubits[source], physical_qubits[target]];
                        let states = states_for_requirement(operations, &ordered);
                        if let Some(cost) = combined_catalog_cost(catalog, estimator, states) {
                            terminals.insert([source, target], cost);
                        }
                    }
                }
                requirements.push(RequirementTable::Pair {
                    lower_bounds: eager_pair_route_lower_bounds(
                        neighbors,
                        &terminals,
                        swap_costs,
                        pair_state_budget,
                    ),
                    terminals,
                });
            }
        }
    }

    // Refinement DAGs deliberately use a generic pair signature. It is
    // terminal wherever at least one exact source interaction is terminal;
    // final route scoring still checks the exact folded signature.
    if requirements.len() > 1 {
        let generic_terminals = requirements[1..]
            .iter()
            .filter_map(|requirement| match requirement {
                RequirementTable::Pair { terminals, .. } => Some(terminals.keys().copied()),
                RequirementTable::Unary { .. } => None,
            })
            .flatten()
            .collect::<BTreeSet<_>>();
        if let RequirementTable::Pair {
            terminals,
            lower_bounds,
        } = &mut requirements[0]
        {
            terminals.extend(
                generic_terminals
                    .into_iter()
                    .map(|placement| (placement, native_identity)),
            );
            if lower_bounds.is_some() {
                *lower_bounds = Some(pair_route_lower_bounds(neighbors, terminals, swap_costs));
            }
        }
    }
    requirements
}

fn adjacency_from_edges(count: usize, edges: &[(usize, usize)]) -> Vec<Vec<usize>> {
    let mut neighbors = vec![Vec::new(); count];
    for &(left, right) in edges {
        neighbors[left].push(right);
        neighbors[right].push(left);
    }
    for adjacent in &mut neighbors {
        adjacent.sort_unstable();
        adjacent.dedup();
    }
    neighbors
}

fn default_swap_costs(count: usize, edges: &[(usize, usize)]) -> Vec<Vec<Option<NativePlanCost>>> {
    let mut costs = vec![vec![None; count]; count];
    for &(left, right) in edges {
        costs[left][right] = Some(NativePlanCost::default());
        costs[right][left] = Some(NativePlanCost::default());
    }
    costs
}

fn unary_route_lower_bounds(
    swap_neighbors: &[Vec<usize>],
    terminals: &[Option<NativePlanCost>],
    swap_costs: &[Vec<Option<NativePlanCost>>],
) -> Vec<Option<RouteLowerBound>> {
    let mut bounds = terminals
        .iter()
        .map(|terminal| {
            terminal.map(|native| RouteLowerBound {
                remaining_swaps: 0,
                native,
            })
        })
        .collect::<Vec<_>>();
    let mut queue = bounds
        .iter()
        .enumerate()
        .filter_map(|(physical, bound)| bound.map(|_| physical))
        .collect::<VecDeque<_>>();
    while let Some(physical) = queue.pop_front() {
        let current = bounds[physical].expect("queued unary state has a lower bound");
        for &predecessor in &swap_neighbors[physical] {
            let Some(swap) = swap_costs[physical][predecessor] else {
                continue;
            };
            let candidate = current.with_swap(swap);
            if bounds[predecessor].is_none_or(|previous| candidate.compare(previous).is_lt()) {
                bounds[predecessor] = Some(candidate);
                queue.push_back(predecessor);
            }
        }
    }
    bounds
}

fn pair_route_lower_bounds(
    swap_neighbors: &[Vec<usize>],
    terminals: &BTreeMap<[usize; 2], NativePlanCost>,
    swap_costs: &[Vec<Option<NativePlanCost>>],
) -> PairStateTable<RouteLowerBound> {
    let count = swap_neighbors.len();
    let mut bounds = PairStateTable::new(count);
    let mut queue = VecDeque::new();
    for (&[left, right], &native) in terminals {
        if left == right || left >= count || right >= count {
            continue;
        }
        bounds.set(
            left,
            right,
            RouteLowerBound {
                remaining_swaps: 0,
                native,
            },
        );
        queue.push_back((left, right));
    }

    while let Some((left, right)) = queue.pop_front() {
        let current = bounds
            .get(left, right)
            .expect("queued pair state has a lower bound");
        for endpoint in [left, right] {
            for &neighbor in &swap_neighbors[endpoint] {
                let Some(swap) = swap_costs[endpoint][neighbor] else {
                    continue;
                };
                let RequirementPlacement::Pair([previous_left, previous_right]) =
                    RequirementPlacement::Pair([left, right]).after_swap([endpoint, neighbor])
                else {
                    unreachable!();
                };
                let candidate = current.with_swap(swap);
                if bounds
                    .get(previous_left, previous_right)
                    .is_none_or(|bound| candidate.compare(bound).is_lt())
                {
                    bounds.set(previous_left, previous_right, candidate);
                    queue.push_back((previous_left, previous_right));
                }
            }
        }
    }
    bounds
}

fn eager_pair_route_lower_bounds(
    swap_neighbors: &[Vec<usize>],
    terminals: &BTreeMap<[usize; 2], NativePlanCost>,
    swap_costs: &[Vec<Option<NativePlanCost>>],
    remaining_budget: &mut usize,
) -> Option<PairStateTable<RouteLowerBound>> {
    let count = swap_neighbors.len();
    let states = count.saturating_mul(count.saturating_sub(1));
    if states > *remaining_budget {
        return None;
    }
    *remaining_budget -= states;
    Some(pair_route_lower_bounds(
        swap_neighbors,
        terminals,
        swap_costs,
    ))
}

fn pair_route_lower_bound_from_state(
    swap_neighbors: &[Vec<usize>],
    terminals: &BTreeMap<[usize; 2], NativePlanCost>,
    swap_costs: &[Vec<Option<NativePlanCost>>],
    start: [usize; 2],
) -> Option<RouteLowerBound> {
    let count = swap_neighbors.len();
    pair_state_index(count, start[0], start[1])?;
    let mut frontier = BTreeMap::from([(start, None::<NativePlanCost>)]);
    let mut visited_depth = BTreeMap::from([(start, 0_u32)]);
    let mut depth = 0_u32;

    while !frontier.is_empty() {
        let mut best_terminal = None::<NativePlanCost>;
        for (placement, path_cost) in &frontier {
            let Some(&terminal) = terminals.get(placement) else {
                continue;
            };
            let candidate = path_cost.map_or(terminal, |path| path.combine(terminal));
            if best_terminal.is_none_or(|best| {
                compare_optional_native_cost(Some(candidate), Some(best)).is_lt()
            }) {
                best_terminal = Some(candidate);
            }
        }
        if let Some(native) = best_terminal {
            return Some(RouteLowerBound {
                remaining_swaps: depth,
                native,
            });
        }

        let next_depth = depth.saturating_add(1);
        let mut next = BTreeMap::<[usize; 2], Option<NativePlanCost>>::new();
        for (&placement, &path_cost) in &frontier {
            for endpoint in placement {
                for &neighbor in &swap_neighbors[endpoint] {
                    let Some(swap) = swap_costs[endpoint][neighbor] else {
                        continue;
                    };
                    let RequirementPlacement::Pair(next_placement) =
                        RequirementPlacement::Pair(placement).after_swap([endpoint, neighbor])
                    else {
                        unreachable!();
                    };
                    if visited_depth
                        .get(&next_placement)
                        .is_some_and(|visited| *visited < next_depth)
                    {
                        continue;
                    }
                    visited_depth.entry(next_placement).or_insert(next_depth);
                    let candidate = Some(path_cost.map_or(swap, |path| path.combine(swap)));
                    let previous = next.entry(next_placement).or_insert(None);
                    if previous.is_none()
                        || compare_optional_native_cost(candidate, *previous).is_lt()
                    {
                        *previous = candidate;
                    }
                }
            }
        }
        frontier = next;
        depth = next_depth;
    }
    None
}

#[derive(Debug)]
struct RoutingState {
    layout: Layout,
    front_layer: Layer,
    lookahead_layers: Vec<Layer>,
    required_predecessors: Vec<u32>,
    decay: Vec<f64>,
    rng: StdRng,
}

#[derive(Debug, Clone, Copy)]
struct SwapChoice {
    physical: [PhysicalQubit; 2],
    indices: [usize; 2],
}

impl RoutingState {
    /// Creates mutable state for one SABRE routing trial.
    ///
    /// `required_predecessors` is the mutable readiness counter for DAG
    /// scheduling. Lookahead temporarily edits the same counters and restores
    /// them before returning to the real routing loop.
    fn new(
        sabre: &SabreDag,
        target: &RoutingTarget,
        layout: Layout,
        heuristic: &SabreHeuristicConfig,
        seed: u64,
    ) -> Self {
        let mut required_predecessors = vec![0; sabre.graph.node_count()];
        for edge in sabre.graph.edge_references() {
            required_predecessors[edge.target().index()] += 1;
        }

        Self {
            layout,
            front_layer: Layer::new(sabre.graph.node_count(), target.physical_qubits.len()),
            lookahead_layers: vec![
                Layer::new(
                    sabre.graph.node_count(),
                    target.physical_qubits.len()
                );
                heuristic.lookahead_weights.len()
            ],
            required_predecessors,
            decay: vec![1.0; target.physical_qubits.len()],
            rng: StdRng::seed_from_u64(seed),
        }
    }

    /// Applies a physical SWAP to the layout and all cached layer scores.
    ///
    /// The layout and every cached layer must move together; otherwise future
    /// SWAP deltas would be scored against stale physical positions.
    fn apply_swap(
        &mut self,
        swap: [PhysicalQubit; 2],
        target: &RoutingTarget,
    ) -> Result<(), CompilerError> {
        let swap_indices = [
            target.physical_index(swap[0])?,
            target.physical_index(swap[1])?,
        ];
        let distance = |requirement, placement| target.distance_for(requirement, placement);
        self.front_layer.apply_swap(swap_indices, &distance)?;
        for layer in &mut self.lookahead_layers {
            layer.apply_swap(swap_indices, &distance)?;
        }
        self.layout
            .swap_physical(swap[0], swap[1])
            .map_err(|error| {
                CompilerError::InvariantViolation(format!(
                    "sabre attempted an invalid physical swap {swap:?}: {error}"
                ))
            })
    }

    fn update_route(
        &mut self,
        sabre: &SabreDag,
        target: &RoutingTarget,
        heuristic: &SabreHeuristicConfig,
        output: &mut TrialOutput,
        nodes: &[NodeIndex],
        initial_swaps: Option<Vec<[PhysicalQubit; 2]>>,
    ) -> Result<(), CompilerError> {
        let mut to_visit = nodes.iter().copied().collect::<VecDeque<_>>();
        let mut pending_swaps = initial_swaps;

        while let Some(node_id) = to_visit.pop_front() {
            let node = &sabre.graph[node_id];
            match &node.kind {
                SabreNodeKind::Unary(logical) => {
                    let physical = physical_for(&self.layout, *logical)?;
                    let requirement = target.interaction_id_for_node(sabre, node_id)?;
                    let placement = RequirementPlacement::Unary(target.physical_index(physical)?);
                    if target.terminal_cost_for(requirement, placement).is_none() {
                        let distance =
                            |requirement, placement| target.distance_for(requirement, placement);
                        self.front_layer
                            .insert(node_id, requirement, placement, &distance)?;
                        continue;
                    }
                    output.apply_pending_swaps(pending_swaps.take());
                    for operation in &node.operations {
                        output
                            .operations
                            .push(map_operation(operation, &self.layout)?);
                    }
                }
                SabreNodeKind::TwoQ(pair) => {
                    // A two-qubit node that is still non-adjacent becomes part
                    // of the front layer. Adjacent nodes flush any pending
                    // SWAPs and are emitted under the current layout.
                    let physical = [
                        physical_for(&self.layout, pair[0])?,
                        physical_for(&self.layout, pair[1])?,
                    ];
                    let interaction = target.interaction_id_for_node(sabre, node_id)?;
                    let physical_indices = [
                        target.physical_index(physical[0])?,
                        target.physical_index(physical[1])?,
                    ];
                    if target
                        .terminal_cost_for(
                            interaction,
                            RequirementPlacement::Pair(physical_indices),
                        )
                        .is_none()
                    {
                        let distance =
                            |requirement, placement| target.distance_for(requirement, placement);
                        self.front_layer.insert(
                            node_id,
                            interaction,
                            RequirementPlacement::Pair(physical_indices),
                            &distance,
                        )?;
                        continue;
                    }
                    output.apply_pending_swaps(pending_swaps.take());
                    for operation in &node.operations {
                        output
                            .operations
                            .push(map_operation(operation, &self.layout)?);
                    }
                }
                SabreNodeKind::Synchronize => {
                    // Synchronize nodes preserve parent-level ordering around
                    // classical or barrier-like effects but do not add an
                    // adjacency constraint of their own.
                    output.apply_pending_swaps(pending_swaps.take());
                    for operation in &node.operations {
                        output
                            .operations
                            .push(map_operation(operation, &self.layout)?);
                    }
                }
                SabreNodeKind::ControlFlow(flow) => {
                    // Control-flow bodies are routed recursively from the
                    // current entry layout and restore that layout on exit, so
                    // parent routing can continue with a single layout state.
                    output.apply_pending_swaps(pending_swaps.take());
                    self.route_control_flow_node(
                        flow,
                        &node.operations,
                        target,
                        heuristic,
                        output,
                    )?;
                }
            }

            for edge in sabre.graph.edges_directed(node_id, Direction::Outgoing) {
                let successor = edge.target();
                self.required_predecessors[successor.index()] -= 1;
                if self.required_predecessors[successor.index()] == 0 {
                    to_visit.push_back(successor);
                }
            }
        }

        if pending_swaps.is_some() {
            return Err(CompilerError::InvariantViolation(
                "sabre selected swaps that did not route any front-layer node".to_string(),
            ));
        }
        Ok(())
    }

    fn route_control_flow_node(
        &mut self,
        flow: &SabreControlFlow,
        operations: &[Operation],
        target: &RoutingTarget,
        heuristic: &SabreHeuristicConfig,
        output: &mut TrialOutput,
    ) -> Result<(), CompilerError> {
        let Some((first, rest)) = operations.split_first() else {
            return Ok(());
        };
        // The SABRE DAG keeps the representative control-flow operation first
        // and may attach additional bookkeeping operations after it. Rebuild the
        // first operation with routed bodies, then map the remaining operations
        // through the unchanged parent layout.
        let routed = match flow {
            SabreControlFlow::If {
                condition,
                then_body,
                else_body,
            } => {
                let then_result = route_control_flow_body(
                    then_body,
                    target,
                    &self.layout,
                    heuristic,
                    output.next_nested_seed(),
                )?;
                let else_result = else_body
                    .as_ref()
                    .map(|body| {
                        route_control_flow_body(
                            body,
                            target,
                            &self.layout,
                            heuristic,
                            output.next_nested_seed(),
                        )
                    })
                    .transpose()?;
                output.merge_nested(&then_result);
                if let Some(result) = &else_result {
                    output.merge_nested(result);
                }
                let flow = ClassicalControlOp::If(IfOp::new(
                    condition.clone(),
                    ControlBody::new(then_result.operations),
                    else_result.map(|result| ControlBody::new(result.operations)),
                )?);
                let qubits = flow.used_qubits().into_iter().collect();
                Operation {
                    instruction: Instruction::ClassicalControl(flow),
                    qubits,
                    params: SmallVec::new(),
                    label: first.label.clone(),
                }
            }
            SabreControlFlow::While { condition, body } => {
                let body_result = route_control_flow_body(
                    body,
                    target,
                    &self.layout,
                    heuristic,
                    output.next_nested_seed(),
                )?;
                output.merge_nested(&body_result);
                let flow = ClassicalControlOp::While(WhileOp::new(
                    condition.clone(),
                    ControlBody::new(body_result.operations),
                )?);
                let qubits = flow.used_qubits().into_iter().collect();
                Operation {
                    instruction: Instruction::ClassicalControl(flow),
                    qubits,
                    params: SmallVec::new(),
                    label: first.label.clone(),
                }
            }
            SabreControlFlow::For {
                var,
                start,
                stop,
                step,
                body,
            } => {
                let body_result = route_control_flow_body(
                    body,
                    target,
                    &self.layout,
                    heuristic,
                    output.next_nested_seed(),
                )?;
                output.merge_nested(&body_result);
                let flow = ClassicalControlOp::For(ForOp::new(
                    *var,
                    start.clone(),
                    stop.clone(),
                    step.clone(),
                    ControlBody::new(body_result.operations),
                )?);
                let qubits = flow.used_qubits().into_iter().collect();
                Operation {
                    instruction: Instruction::ClassicalControl(flow),
                    qubits,
                    params: SmallVec::new(),
                    label: first.label.clone(),
                }
            }
            SabreControlFlow::Switch {
                target: switch_target,
                cases,
                default,
            } => {
                let mut routed_cases = Vec::with_capacity(cases.len());
                for case in cases {
                    let result = route_control_flow_body(
                        &case.body,
                        target,
                        &self.layout,
                        heuristic,
                        output.next_nested_seed(),
                    )?;
                    output.merge_nested(&result);
                    routed_cases.push(SwitchCase::new(
                        case.value,
                        ControlBody::new(result.operations),
                    ));
                }
                let routed_default = default
                    .as_ref()
                    .map(|body| {
                        route_control_flow_body(
                            body,
                            target,
                            &self.layout,
                            heuristic,
                            output.next_nested_seed(),
                        )
                    })
                    .transpose()?;
                if let Some(result) = &routed_default {
                    output.merge_nested(result);
                }
                let flow = ClassicalControlOp::Switch(SwitchOp::new(
                    switch_target.clone(),
                    routed_cases,
                    routed_default.map(|result| ControlBody::new(result.operations)),
                )?);
                let qubits = flow.used_qubits().into_iter().collect();
                Operation {
                    instruction: Instruction::ClassicalControl(flow),
                    qubits,
                    params: SmallVec::new(),
                    label: first.label.clone(),
                }
            }
        };
        output.operations.push(routed);
        for operation in rest {
            output
                .operations
                .push(map_operation(operation, &self.layout)?);
        }
        Ok(())
    }

    fn populate_extended_set(
        &mut self,
        sabre: &SabreDag,
        target: &RoutingTarget,
    ) -> Result<(), CompilerError> {
        // Build fixed-depth lookahead layers from the current front layer. Synchronize
        // and control-flow nodes are transparent for depth counting because they do
        // not add a parent-level two-qubit adjacency constraint.
        let mut next_visit = self.front_layer.iter_nodes().collect::<Vec<_>>();
        let mut to_visit = Vec::new();
        let mut decremented = BTreeMap::<NodeIndex, u32>::new();

        for layer in &mut self.lookahead_layers {
            for node in next_visit.drain(..) {
                for edge in sabre.graph.edges_directed(node, Direction::Outgoing) {
                    let successor = edge.target();
                    *decremented.entry(successor).or_insert(0) += 1;
                    self.required_predecessors[successor.index()] -= 1;
                    if self.required_predecessors[successor.index()] == 0 {
                        to_visit.push(successor);
                    }
                }
            }

            let mut index = 0;
            while index < to_visit.len() {
                let node = to_visit[index];
                match &sabre.graph[node].kind {
                    SabreNodeKind::Unary(logical) => {
                        if let Ok(physical) = physical_for(&self.layout, *logical) {
                            let requirement = target.interaction_id_for_node(sabre, node)?;
                            let distance = |requirement, placement| {
                                target.distance_for(requirement, placement)
                            };
                            layer.insert(
                                node,
                                requirement,
                                RequirementPlacement::Unary(target.physical_index(physical)?),
                                &distance,
                            )?;
                            next_visit.push(node);
                        }
                    }
                    SabreNodeKind::TwoQ(pair) => {
                        if let (Ok(left), Ok(right)) = (
                            physical_for(&self.layout, pair[0]),
                            physical_for(&self.layout, pair[1]),
                        ) {
                            let interaction = target.interaction_id_for_node(sabre, node)?;
                            let distance = |requirement, placement| {
                                target.distance_for(requirement, placement)
                            };
                            layer.insert(
                                node,
                                interaction,
                                RequirementPlacement::Pair([
                                    target.physical_index(left)?,
                                    target.physical_index(right)?,
                                ]),
                                &distance,
                            )?;
                            next_visit.push(node);
                        }
                        // Missing physical mappings are ignored defensively here.
                        // Normal routing entrypoints normalize complete layouts before
                        // creating state, so this only affects future partial-layout use.
                    }
                    SabreNodeKind::Synchronize | SabreNodeKind::ControlFlow(_) => {
                        for edge in sabre.graph.edges_directed(node, Direction::Outgoing) {
                            let successor = edge.target();
                            *decremented.entry(successor).or_insert(0) += 1;
                            self.required_predecessors[successor.index()] -= 1;
                            if self.required_predecessors[successor.index()] == 0 {
                                to_visit.push(successor);
                            }
                        }
                    }
                }
                index += 1;
            }
            to_visit.clear();
        }

        // Lookahead exploration temporarily relaxes predecessor counts; restore
        // them before the real routing state advances.
        for (node, amount) in decremented {
            self.required_predecessors[node.index()] += amount;
        }
        Ok(())
    }

    fn choose_best_swap(
        &mut self,
        target: &RoutingTarget,
        heuristic: &SabreHeuristicConfig,
        previous_swap: Option<[PhysicalQubit; 2]>,
    ) -> Result<SwapChoice, CompilerError> {
        if self.front_layer.is_empty() {
            return Err(CompilerError::InvariantViolation(
                "sabre cannot select a SWAP for an empty front layer".to_string(),
            ));
        }
        let mut candidates = Vec::new();
        for active_index in self
            .front_layer
            .active_indices_in_order(&target.physical_order_indices)
        {
            let active = target.physical_at(active_index)?;
            for &neighbor_index in &target.neighbors_by_index[active_index] {
                let neighbor = target.physical_at(neighbor_index)?;
                candidates.push(if active <= neighbor {
                    SwapChoice {
                        physical: [active, neighbor],
                        indices: [active_index, neighbor_index],
                    }
                } else {
                    SwapChoice {
                        physical: [neighbor, active],
                        indices: [neighbor_index, active_index],
                    }
                });
            }
        }
        candidates.sort_unstable_by_key(|candidate| candidate.physical);
        candidates.dedup_by(|left, right| left.physical == right.physical);
        if candidates.len() > 1
            && let Some(previous_swap) = previous_swap
        {
            candidates.retain(|candidate| candidate.physical != previous_swap);
        }
        if candidates.is_empty() {
            return Err(CompilerError::TransformFailed {
                name: "sabre_route",
                reason: "no candidate SWAP can affect the front layer".to_string(),
            });
        }

        // Select topology-tied candidates, then prefer the cheapest exact
        // native route cost.
        let distance = |requirement, placement| target.distance_for(requirement, placement);
        let mut best_score = f64::INFINITY;
        let mut scored = Vec::with_capacity(candidates.len());
        for candidate in candidates {
            let decay = if heuristic.decay_increment.is_some() {
                self.decay[candidate.indices[0]].max(self.decay[candidate.indices[1]])
            } else {
                1.0
            };
            let score = heuristic_score_after_swap(
                &self.front_layer,
                &self.lookahead_layers,
                heuristic,
                candidate.indices,
                &distance,
                decay,
                target.physical_qubits.len(),
            )?;

            best_score = best_score.min(score);
            scored.push((candidate, score));
        }

        let has_native_cost = target.native_cost_enabled
            && scored
                .iter()
                .any(|(candidate, _)| self.route_cost_after_swap(target, *candidate).is_some());
        let mut eligible = scored
            .into_iter()
            .filter(|(_, score)| *score <= best_score + heuristic.best_epsilon)
            .map(|(candidate, _)| candidate)
            .collect::<Vec<_>>();
        if has_native_cost {
            eligible.sort_by(|left, right| {
                compare_optional_route_bound(
                    self.route_cost_after_swap(target, *left),
                    self.route_cost_after_swap(target, *right),
                )
                .then_with(|| left.physical.cmp(&right.physical))
            });
            if let Some(best) = eligible.first().copied() {
                let best_cost = self.route_cost_after_swap(target, best);
                eligible.retain(|candidate| {
                    compare_optional_route_bound(
                        self.route_cost_after_swap(target, *candidate),
                        best_cost,
                    ) == Ordering::Equal
                });
            }
        }

        eligible.choose(&mut self.rng).copied().ok_or_else(|| {
            CompilerError::InvariantViolation("sabre found no best SWAP".to_string())
        })
    }

    fn route_cost_after_swap(
        &self,
        target: &RoutingTarget,
        candidate: SwapChoice,
    ) -> Option<RouteLowerBound> {
        let mut cost = RouteLowerBound {
            remaining_swaps: 1,
            native: target.swap_cost(candidate.indices[0], candidate.indices[1])?,
        };
        for (requirement, placement) in self.front_layer.placements_after_swap(candidate.indices) {
            cost = cost.combine(target.route_lower_bound_for(requirement, placement)?);
        }
        Some(cost)
    }

    fn force_enable_closest_node(
        &mut self,
        target: &RoutingTarget,
        current_swaps: &mut Vec<[PhysicalQubit; 2]>,
    ) -> Result<Vec<NodeIndex>, CompilerError> {
        // Fallback follows the exact unary/pair state lower bound. Each step is
        // a lowerable SWAP and strictly reduces the remaining distance.
        let (closest_node, requirement, mut placement) = self
            .front_layer
            .iter()
            .min_by(
                |(_, left_interaction, left), (_, right_interaction, right)| {
                    target
                        .distance_steps_for(*left_interaction, *left)
                        .unwrap_or(u32::MAX)
                        .cmp(
                            &target
                                .distance_steps_for(*right_interaction, *right)
                                .unwrap_or(u32::MAX),
                        )
                },
            )
            .ok_or_else(|| {
                CompilerError::InvariantViolation(
                    "sabre fallback called with an empty front layer".to_string(),
                )
            })?;
        while let Some(distance) = target.distance_steps_for(requirement, placement) {
            if distance == 0 {
                break;
            }
            let mut improving = Vec::new();
            let endpoints: SmallVec<[usize; 2]> = match placement {
                RequirementPlacement::Unary(physical) => smallvec![physical],
                RequirementPlacement::Pair(pair) => SmallVec::from_slice(&pair),
            };
            for endpoint in endpoints {
                for &neighbor in &target.neighbors_by_index[endpoint] {
                    let swap = [endpoint, neighbor];
                    let next = placement.after_swap(swap);
                    if target.distance_steps_for(requirement, next) == Some(distance - 1) {
                        improving.push((swap, next));
                    }
                }
            }
            improving.sort_by(|(left_swap, _), (right_swap, _)| {
                compare_optional_native_cost(
                    target.swap_cost(left_swap[0], left_swap[1]),
                    target.swap_cost(right_swap[0], right_swap[1]),
                )
                .then_with(|| {
                    let left = [
                        target.physical_qubits[left_swap[0]],
                        target.physical_qubits[left_swap[1]],
                    ];
                    let right = [
                        target.physical_qubits[right_swap[0]],
                        target.physical_qubits[right_swap[1]],
                    ];
                    left.cmp(&right)
                })
            });
            let Some((swap_indices, next)) = improving.into_iter().next() else {
                return Err(CompilerError::InvariantViolation(format!(
                    "routing-state distance {distance} has no improving lowerable SWAP"
                )));
            };
            let swap = [
                target.physical_at(swap_indices[0])?,
                target.physical_at(swap_indices[1])?,
            ];
            self.apply_swap(swap, target)?;
            current_swaps.push(swap);
            placement = next;
        }

        let routed = self
            .front_layer
            .iter()
            .filter_map(|(node, requirement, placement)| {
                target
                    .terminal_cost_for(requirement, placement)
                    .is_some()
                    .then_some(node)
            })
            .collect::<Vec<_>>();
        if !routed.contains(&closest_node) {
            return Err(CompilerError::InvariantViolation(
                "routing-state fallback did not enable its selected node".to_string(),
            ));
        }
        Ok(routed)
    }
}

fn mapping_signature(layout: &Layout, target: &RoutingTarget) -> Vec<Option<LogicalQubit>> {
    target
        .physical_qubits
        .iter()
        .map(|physical| layout.get_logical(*physical))
        .collect()
}

fn compare_optional_native_cost(
    left: Option<NativePlanCost>,
    right: Option<NativePlanCost>,
) -> Ordering {
    match (left, right) {
        (Some(left), Some(right)) => left
            .native_two_qubit_ops
            .cmp(&right.native_two_qubit_ops)
            .then_with(|| match (left.error, right.error) {
                (Some(left), Some(right)) => left.compare(right),
                _ => Ordering::Equal,
            })
            .then_with(|| match (left.duration, right.duration) {
                (Some(left), Some(right)) => left.compare(right),
                _ => Ordering::Equal,
            })
            .then_with(|| left.native_total_ops.cmp(&right.native_total_ops)),
        (Some(_), None) => Ordering::Less,
        (None, Some(_)) => Ordering::Greater,
        (None, None) => Ordering::Equal,
    }
}

fn compare_optional_route_bound(
    left: Option<RouteLowerBound>,
    right: Option<RouteLowerBound>,
) -> Ordering {
    match (left, right) {
        (Some(left), Some(right)) => left.compare(right),
        (Some(_), None) => Ordering::Less,
        (None, Some(_)) => Ordering::Greater,
        (None, None) => Ordering::Equal,
    }
}

fn heuristic_score_after_swap(
    front_layer: &Layer,
    lookahead_layers: &[Layer],
    heuristic: &SabreHeuristicConfig,
    swap: [usize; 2],
    distances: &impl Fn(usize, RequirementPlacement) -> Result<f64, CompilerError>,
    decay: f64,
    device_width: usize,
) -> Result<f64, CompilerError> {
    let mut score = heuristic.basic_weight * front_layer.total_score_after_swap(swap, distances)?;
    let lookahead_scale = device_width.max(1) as f64;
    for (layer, weight) in lookahead_layers
        .iter()
        .zip(heuristic.lookahead_weights.iter().copied())
    {
        score += weight * layer.total_score_after_swap(swap, distances)? / lookahead_scale;
    }
    Ok(score + (decay - 1.0).max(0.0))
}

#[derive(Debug, Default)]
struct TrialOutput {
    operations: Vec<Operation>,
    swap_count: usize,
    fallback_count: usize,
    control_flow_blocks_routed: usize,
    nested_seed_counter: u64,
}

impl TrialOutput {
    fn new(seed: u64) -> Self {
        Self {
            nested_seed_counter: seed,
            ..Self::default()
        }
    }

    fn apply_pending_swaps(&mut self, swaps: Option<Vec<[PhysicalQubit; 2]>>) {
        if let Some(swaps) = swaps {
            self.swap_count += swaps.len();
            self.operations
                .extend(swaps.into_iter().map(swap_operation));
        }
    }

    fn next_nested_seed(&mut self) -> u64 {
        let seed = self.nested_seed_counter;
        self.nested_seed_counter = self.nested_seed_counter.wrapping_add(1);
        seed
    }

    fn merge_nested(&mut self, nested: &TrialResult) {
        self.swap_count += nested.swap_count;
        self.fallback_count += nested.fallback_count;
        self.control_flow_blocks_routed += nested.control_flow_blocks_routed + 1;
    }
}

/// Routes a control-flow body and restores its entry layout before exit.
///
/// A control-flow body must be layout-neutral from the parent router's point of
/// view: whichever branch or iteration executes, the body exits with the same
/// logical-to-physical mapping it entered with.
fn route_control_flow_body(
    sabre: &SabreDag,
    target: &RoutingTarget,
    entry_layout: &Layout,
    heuristic: &SabreHeuristicConfig,
    seed: u64,
) -> Result<TrialResult, CompilerError> {
    let mut result = route_trial(sabre, target, entry_layout, heuristic, seed)?;
    let epilogue_swaps = restore_layout_swaps(target, &result.final_layout, entry_layout, seed)?;
    let mut layout = result.final_layout.clone();
    for swap in &epilogue_swaps {
        layout.swap_physical(swap[0], swap[1]).map_err(|error| {
            CompilerError::InvariantViolation(format!(
                "sabre control-flow epilogue generated an invalid SWAP: {error}"
            ))
        })?;
    }
    let control_transfer = matches!(
        result
            .operations
            .last()
            .map(|operation| &operation.instruction),
        Some(Instruction::ClassicalControl(
            ClassicalControlOp::Break | ClassicalControlOp::Continue
        ))
    )
    .then(|| result.operations.pop().expect("last operation exists"));
    result
        .operations
        .extend(epilogue_swaps.iter().copied().map(swap_operation));
    result.operations.extend(control_transfer);
    result.swap_count += epilogue_swaps.len();
    result.final_layout = layout;
    result.quality = trial_quality(&result.operations, result.swap_count, target)?;
    if result.final_layout.l2p_map() != entry_layout.l2p_map() {
        return Err(CompilerError::InvariantViolation(
            "sabre control-flow epilogue did not restore the entry layout".to_string(),
        ));
    }
    Ok(result)
}

/// Computes SWAPs that restore one layout to another on the target graph.
///
/// Token swapping computes an epilogue that returns every live logical qubit to
/// its entry physical location. Vacant physical qubits are irrelevant and are
/// omitted from the token mapping.
fn restore_layout_swaps(
    target: &RoutingTarget,
    current: &Layout,
    desired: &Layout,
    seed: u64,
) -> Result<Vec<[PhysicalQubit; 2]>, CompilerError> {
    let mapping = desired
        .physical_qubits()
        .filter_map(|physical| {
            let logical = current.get_logical(physical)?;
            let desired_physical = desired
                .get_physical(logical)
                .expect("desired layout maps logical qubits it reports");
            Some((
                target.graph_index[&physical],
                target.graph_index[&desired_physical],
            ))
        })
        .collect();

    let swaps = token_swapper(
        &target.graph,
        mapping,
        Some(CONTROL_FLOW_EPILOGUE_TRIALS),
        Some(seed),
        None,
    )
    .map_err(|error| CompilerError::TransformFailed {
        name: "sabre_route",
        reason: format!("failed to restore control-flow layout: {error}"),
    })?;
    swaps
        .into_iter()
        .map(|(left, right)| {
            Ok([
                target.physical_by_index[left.index()],
                target.physical_by_index[right.index()],
            ])
        })
        .collect()
}

fn map_operation(operation: &Operation, layout: &Layout) -> Result<Operation, CompilerError> {
    Ok(Operation {
        instruction: operation.instruction.clone(),
        qubits: operation
            .qubits
            .iter()
            .copied()
            .map(|qubit| {
                physical_for(layout, LogicalQubit::from_qubit(qubit)).map(PhysicalQubit::qubit)
            })
            .collect::<Result<SmallVec<[Qubit; 3]>, _>>()?,
        params: operation.params.clone(),
        label: operation.label.clone(),
    })
}

/// Remaps operation parameter indices into the routed circuit's parameter table.
///
/// Parameter indices are scoped to a circuit table, while routed nested
/// operations are rebuilt before the final table exists. This function walks
/// recursively so every body points at the reordered routed table.
fn remap_parameter_indices(
    operation: &Operation,
    parameter_indices: &[u32],
) -> Result<Operation, CompilerError> {
    let mut mapped = operation.clone();
    for param in &mut mapped.params {
        if let CircuitParam::Index(index) = param {
            *index = *parameter_indices
                .get(*index as usize)
                .ok_or(crate::circuit::CircuitError::InvalidParameterIndex(*index))?;
        }
    }
    mapped.instruction = match &operation.instruction {
        Instruction::ClassicalControl(ClassicalControlOp::If(op)) => {
            let then_body = op
                .then_body()
                .operations()
                .iter()
                .map(|operation| remap_parameter_indices(operation, parameter_indices))
                .collect::<Result<Vec<_>, _>>()?;
            let else_body = op
                .else_body()
                .map(|body| {
                    body.operations()
                        .iter()
                        .map(|operation| remap_parameter_indices(operation, parameter_indices))
                        .collect::<Result<Vec<_>, _>>()
                        .map(ControlBody::new)
                })
                .transpose()?;
            Instruction::ClassicalControl(ClassicalControlOp::If(IfOp::new(
                op.condition().clone(),
                ControlBody::new(then_body),
                else_body,
            )?))
        }
        Instruction::ClassicalControl(ClassicalControlOp::While(op)) => {
            let body = op
                .body()
                .operations()
                .iter()
                .map(|operation| remap_parameter_indices(operation, parameter_indices))
                .collect::<Result<Vec<_>, _>>()?;
            Instruction::ClassicalControl(ClassicalControlOp::While(WhileOp::new(
                op.condition().clone(),
                ControlBody::new(body),
            )?))
        }
        Instruction::ClassicalControl(ClassicalControlOp::For(op)) => {
            let body = op
                .body()
                .operations()
                .iter()
                .map(|operation| remap_parameter_indices(operation, parameter_indices))
                .collect::<Result<Vec<_>, _>>()?;
            Instruction::ClassicalControl(ClassicalControlOp::For(ForOp::new(
                op.var(),
                op.start().clone(),
                op.stop().clone(),
                op.step().clone(),
                ControlBody::new(body),
            )?))
        }
        Instruction::ClassicalControl(ClassicalControlOp::Switch(op)) => {
            let cases = op
                .cases()
                .iter()
                .map(|case| {
                    let body = case
                        .body()
                        .operations()
                        .iter()
                        .map(|operation| remap_parameter_indices(operation, parameter_indices))
                        .collect::<Result<Vec<_>, _>>()?;
                    Ok(SwitchCase::new(case.value(), ControlBody::new(body)))
                })
                .collect::<Result<Vec<_>, CompilerError>>()?;
            let default = op
                .default()
                .map(|body| {
                    body.operations()
                        .iter()
                        .map(|operation| remap_parameter_indices(operation, parameter_indices))
                        .collect::<Result<Vec<_>, _>>()
                        .map(ControlBody::new)
                })
                .transpose()?;
            Instruction::ClassicalControl(ClassicalControlOp::Switch(SwitchOp::new(
                op.target().clone(),
                cases,
                default,
            )?))
        }
        _ => operation.instruction.clone(),
    };
    Ok(mapped)
}

fn physical_for(layout: &Layout, logical: LogicalQubit) -> Result<PhysicalQubit, CompilerError> {
    layout.get_physical(logical).ok_or_else(|| {
        CompilerError::InvariantViolation(format!(
            "sabre layout does not map logical qubit {logical}"
        ))
    })
}

/// One exact movement-component assignment for all logical qubits.
#[derive(Debug, Clone)]
pub(crate) struct MovementComponentAssignment {
    pub(crate) components: Vec<Vec<PhysicalQubit>>,
    /// Component index in the same order as the requested logical-qubit list.
    pub(crate) logical_components: Vec<usize>,
}

#[derive(Debug, Clone)]
pub(crate) enum ComponentAssignmentSearch {
    Found(MovementComponentAssignment),
    ProvenInfeasible,
    BudgetExhausted { expansions: usize },
}

#[derive(Debug)]
struct PairComponentConstraint {
    left: usize,
    right: usize,
    allowed: BTreeSet<(usize, usize)>,
}

/// Solves the exact component-level placement constraints induced by unary and
/// ordered-pair requirements. The search is exhaustive: `None` means the
/// component model proved infeasible, never that a search budget expired.
pub(crate) fn movement_component_assignment(
    sabre: &SabreDag,
    target: &RoutingTarget,
    logical_qubits: &[LogicalQubit],
    expansion_budget: usize,
) -> Result<ComponentAssignmentSearch, CompilerError> {
    let component_indices = target.movement_component_indices();
    let components = component_indices
        .iter()
        .map(|component| {
            component
                .iter()
                .map(|index| target.physical_qubits[*index])
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    let logical_index = logical_qubits
        .iter()
        .copied()
        .enumerate()
        .map(|(index, logical)| (logical, index))
        .collect::<BTreeMap<_, _>>();
    let all_components = (0..components.len()).collect::<BTreeSet<_>>();
    let mut domains = vec![all_components; logical_qubits.len()];
    let mut pair_constraints = Vec::new();
    collect_component_constraints(
        sabre,
        target,
        &component_indices,
        &logical_index,
        &mut domains,
        &mut pair_constraints,
    )?;
    if domains.iter().any(BTreeSet::is_empty) {
        return Ok(ComponentAssignmentSearch::ProvenInfeasible);
    }

    let capacities = components.iter().map(Vec::len).collect::<Vec<_>>();
    let mut remaining = capacities;
    let mut assignment = vec![None; logical_qubits.len()];
    let mut budget = expansion_budget;
    let result = assign_logical_components(
        &domains,
        &pair_constraints,
        &mut remaining,
        &mut assignment,
        &mut budget,
    );
    match result {
        ComponentSearchProgress::Found => Ok(ComponentAssignmentSearch::Found(
            MovementComponentAssignment {
                components,
                logical_components: assignment
                    .into_iter()
                    .map(|component| component.expect("successful assignment is complete"))
                    .collect(),
            },
        )),
        ComponentSearchProgress::ProvenInfeasible => {
            Ok(ComponentAssignmentSearch::ProvenInfeasible)
        }
        ComponentSearchProgress::BudgetExhausted => {
            Ok(ComponentAssignmentSearch::BudgetExhausted {
                expansions: expansion_budget - budget,
            })
        }
    }
}

impl RoutingTarget {
    fn movement_component_indices(&self) -> Vec<Vec<usize>> {
        let mut unseen = (0..self.physical_qubits.len()).collect::<BTreeSet<_>>();
        let mut components = Vec::new();
        while let Some(start) = unseen.pop_first() {
            let mut queue = VecDeque::from([start]);
            let mut component = Vec::new();
            while let Some(physical) = queue.pop_front() {
                component.push(physical);
                for &neighbor in &self.neighbors_by_index[physical] {
                    if unseen.remove(&neighbor) {
                        queue.push_back(neighbor);
                    }
                }
            }
            component.sort_unstable_by_key(|index| self.physical_qubits[*index]);
            components.push(component);
        }
        components.sort_unstable_by_key(|component| self.physical_qubits[component[0]]);
        components
    }
}

fn collect_component_constraints(
    sabre: &SabreDag,
    target: &RoutingTarget,
    components: &[Vec<usize>],
    logical_index: &BTreeMap<LogicalQubit, usize>,
    domains: &mut [BTreeSet<usize>],
    pairs: &mut Vec<PairComponentConstraint>,
) -> Result<(), CompilerError> {
    for node_index in sabre.graph.node_indices() {
        let node = &sabre.graph[node_index];
        match &node.kind {
            SabreNodeKind::Unary(logical) => {
                let index = *logical_index.get(logical).ok_or_else(|| {
                    CompilerError::InvariantViolation(format!(
                        "unary component constraint references unknown logical qubit {logical}"
                    ))
                })?;
                let requirement = target.interaction_id_for_node(sabre, node_index)?;
                domains[index].retain(|component| {
                    components[*component].iter().any(|physical| {
                        target
                            .route_lower_bound_for(
                                requirement,
                                RequirementPlacement::Unary(*physical),
                            )
                            .is_some()
                    })
                });
            }
            SabreNodeKind::TwoQ(logicals) => {
                let left = *logical_index.get(&logicals[0]).ok_or_else(|| {
                    CompilerError::InvariantViolation(format!(
                        "pair component constraint references unknown logical qubit {}",
                        logicals[0]
                    ))
                })?;
                let right = *logical_index.get(&logicals[1]).ok_or_else(|| {
                    CompilerError::InvariantViolation(format!(
                        "pair component constraint references unknown logical qubit {}",
                        logicals[1]
                    ))
                })?;
                let requirement = target.interaction_id_for_node(sabre, node_index)?;
                let mut allowed = BTreeSet::new();
                for (left_component, left_physical) in components.iter().enumerate() {
                    for (right_component, right_physical) in components.iter().enumerate() {
                        if left_physical.iter().any(|left| {
                            right_physical.iter().any(|right| {
                                left != right
                                    && target
                                        .route_lower_bound_for(
                                            requirement,
                                            RequirementPlacement::Pair([*left, *right]),
                                        )
                                        .is_some()
                            })
                        }) {
                            allowed.insert((left_component, right_component));
                        }
                    }
                }
                if allowed.is_empty() {
                    domains[left].clear();
                    domains[right].clear();
                } else {
                    domains[left].retain(|component| {
                        allowed.iter().any(|(allowed, _)| allowed == component)
                    });
                    domains[right].retain(|component| {
                        allowed.iter().any(|(_, allowed)| allowed == component)
                    });
                    pairs.push(PairComponentConstraint {
                        left,
                        right,
                        allowed,
                    });
                }
            }
            SabreNodeKind::ControlFlow(SabreControlFlow::If {
                then_body,
                else_body,
                ..
            }) => {
                collect_component_constraints(
                    then_body,
                    target,
                    components,
                    logical_index,
                    domains,
                    pairs,
                )?;
                if let Some(else_body) = else_body {
                    collect_component_constraints(
                        else_body,
                        target,
                        components,
                        logical_index,
                        domains,
                        pairs,
                    )?;
                }
            }
            SabreNodeKind::ControlFlow(
                SabreControlFlow::While { body, .. } | SabreControlFlow::For { body, .. },
            ) => collect_component_constraints(
                body,
                target,
                components,
                logical_index,
                domains,
                pairs,
            )?,
            SabreNodeKind::ControlFlow(SabreControlFlow::Switch { cases, default, .. }) => {
                for case in cases {
                    collect_component_constraints(
                        &case.body,
                        target,
                        components,
                        logical_index,
                        domains,
                        pairs,
                    )?;
                }
                if let Some(default) = default {
                    collect_component_constraints(
                        default,
                        target,
                        components,
                        logical_index,
                        domains,
                        pairs,
                    )?;
                }
            }
            SabreNodeKind::Synchronize => {}
        }
    }
    Ok(())
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ComponentSearchProgress {
    Found,
    ProvenInfeasible,
    BudgetExhausted,
}

fn assign_logical_components(
    domains: &[BTreeSet<usize>],
    constraints: &[PairComponentConstraint],
    remaining: &mut [usize],
    assignment: &mut [Option<usize>],
    budget: &mut usize,
) -> ComponentSearchProgress {
    if assignment.iter().all(Option::is_some) {
        return ComponentSearchProgress::Found;
    }
    let Some((logical, candidates)) = assignment
        .iter()
        .enumerate()
        .filter(|(_, assigned)| assigned.is_none())
        .map(|(logical, _)| {
            let candidates = domains[logical]
                .iter()
                .copied()
                .filter(|component| remaining[*component] > 0)
                .filter(|component| {
                    component_assignment_is_consistent(logical, *component, constraints, assignment)
                })
                .collect::<Vec<_>>();
            (logical, candidates)
        })
        .min_by_key(|(logical, candidates)| (candidates.len(), *logical))
    else {
        return ComponentSearchProgress::ProvenInfeasible;
    };
    if candidates.is_empty() {
        return ComponentSearchProgress::ProvenInfeasible;
    }
    for component in candidates {
        if *budget == 0 {
            return ComponentSearchProgress::BudgetExhausted;
        }
        *budget -= 1;
        assignment[logical] = Some(component);
        remaining[component] -= 1;
        let forward_consistent = assignment
            .iter()
            .enumerate()
            .filter(|(_, assigned)| assigned.is_none())
            .all(|(unassigned, _)| {
                domains[unassigned].iter().copied().any(|candidate| {
                    remaining[candidate] > 0
                        && component_assignment_is_consistent(
                            unassigned,
                            candidate,
                            constraints,
                            assignment,
                        )
                })
            });
        if forward_consistent {
            match assign_logical_components(domains, constraints, remaining, assignment, budget) {
                ComponentSearchProgress::Found => return ComponentSearchProgress::Found,
                ComponentSearchProgress::BudgetExhausted => {
                    remaining[component] += 1;
                    assignment[logical] = None;
                    return ComponentSearchProgress::BudgetExhausted;
                }
                ComponentSearchProgress::ProvenInfeasible => {}
            }
        }
        remaining[component] += 1;
        assignment[logical] = None;
    }
    ComponentSearchProgress::ProvenInfeasible
}

fn component_assignment_is_consistent(
    logical: usize,
    component: usize,
    constraints: &[PairComponentConstraint],
    assignment: &[Option<usize>],
) -> bool {
    constraints.iter().all(|constraint| {
        if constraint.left == logical {
            assignment[constraint.right]
                .is_none_or(|right| constraint.allowed.contains(&(component, right)))
        } else if constraint.right == logical {
            assignment[constraint.left]
                .is_none_or(|left| constraint.allowed.contains(&(left, component)))
        } else {
            true
        }
    })
}

/// Validates that every interaction in a SABRE DAG is physically reachable.
///
/// Reachability is recursive because control-flow bodies are routed with the
/// same physical topology and entry-layout contract as parent operations.
pub(crate) fn validate_reachable_interactions_for_target(
    sabre: &SabreDag,
    target: &RoutingTarget,
    layout: &Layout,
) -> Result<(), CompilerError> {
    match interaction_reachability_for_target(sabre, target, layout)? {
        InteractionReachability::Reachable => Ok(()),
        InteractionReachability::UnreachableUnary {
            logical,
            physical: _,
            cause: RequirementReachabilityFailure::NoExecutableTerminal,
        } => Err(CompilerError::SabreRoutingFailed(
            SabreRoutingFailure::NoExecutableUnaryTerminal { logical },
        )),
        InteractionReachability::UnreachableUnary {
            logical,
            physical,
            cause: RequirementReachabilityFailure::MovementDisconnected,
        } => Err(CompilerError::SabreRoutingFailed(
            SabreRoutingFailure::UnreachableUnaryPlacement { logical, physical },
        )),
        InteractionReachability::UnreachablePair {
            logical,
            physical: _,
            cause: RequirementReachabilityFailure::NoExecutableTerminal,
        } => Err(CompilerError::SabreRoutingFailed(
            SabreRoutingFailure::NoExecutablePairTerminal { logical },
        )),
        InteractionReachability::UnreachablePair {
            logical,
            physical,
            cause: RequirementReachabilityFailure::MovementDisconnected,
        } => Err(CompilerError::SabreRoutingFailed(
            SabreRoutingFailure::UnreachablePairPlacement { logical, physical },
        )),
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum InteractionReachability {
    Reachable,
    UnreachableUnary {
        logical: LogicalQubit,
        physical: PhysicalQubit,
        cause: RequirementReachabilityFailure,
    },
    UnreachablePair {
        logical: [LogicalQubit; 2],
        physical: [PhysicalQubit; 2],
        cause: RequirementReachabilityFailure,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum RequirementReachabilityFailure {
    NoExecutableTerminal,
    MovementDisconnected,
}

pub(crate) fn interaction_reachability_for_target(
    sabre: &SabreDag,
    target: &RoutingTarget,
    layout: &Layout,
) -> Result<InteractionReachability, CompilerError> {
    for node_index in sabre.graph.node_indices() {
        let node = &sabre.graph[node_index];
        match &node.kind {
            SabreNodeKind::Unary(logical) => {
                let physical = physical_for(layout, *logical)?;
                let physical_index = target.physical_index(physical)?;
                let requirement = target.interaction_id_for_node(sabre, node_index)?;
                if target
                    .distance_steps_for(requirement, RequirementPlacement::Unary(physical_index))
                    .is_none()
                {
                    return Ok(InteractionReachability::UnreachableUnary {
                        logical: *logical,
                        physical,
                        cause: if target.has_terminal(requirement) {
                            RequirementReachabilityFailure::MovementDisconnected
                        } else {
                            RequirementReachabilityFailure::NoExecutableTerminal
                        },
                    });
                }
            }
            SabreNodeKind::TwoQ(pair) => {
                let left = physical_for(layout, pair[0])?;
                let right = physical_for(layout, pair[1])?;
                let left_index = target.physical_index(left)?;
                let right_index = target.physical_index(right)?;
                let interaction = target.interaction_id_for_node(sabre, node_index)?;
                if target
                    .distance_steps_for(
                        interaction,
                        RequirementPlacement::Pair([left_index, right_index]),
                    )
                    .is_none()
                {
                    return Ok(InteractionReachability::UnreachablePair {
                        logical: *pair,
                        physical: [left, right],
                        cause: if target.has_terminal(interaction) {
                            RequirementReachabilityFailure::MovementDisconnected
                        } else {
                            RequirementReachabilityFailure::NoExecutableTerminal
                        },
                    });
                }
            }
            SabreNodeKind::ControlFlow(SabreControlFlow::If {
                then_body,
                else_body,
                ..
            }) => {
                let reachable = interaction_reachability_for_target(then_body, target, layout)?;
                if reachable != InteractionReachability::Reachable {
                    return Ok(reachable);
                }
                if let Some(else_body) = else_body {
                    let reachable = interaction_reachability_for_target(else_body, target, layout)?;
                    if reachable != InteractionReachability::Reachable {
                        return Ok(reachable);
                    }
                }
            }
            SabreNodeKind::ControlFlow(
                SabreControlFlow::While { body, .. } | SabreControlFlow::For { body, .. },
            ) => {
                let reachable = interaction_reachability_for_target(body, target, layout)?;
                if reachable != InteractionReachability::Reachable {
                    return Ok(reachable);
                }
            }
            SabreNodeKind::ControlFlow(SabreControlFlow::Switch { cases, default, .. }) => {
                for case in cases {
                    let reachable =
                        interaction_reachability_for_target(&case.body, target, layout)?;
                    if reachable != InteractionReachability::Reachable {
                        return Ok(reachable);
                    }
                }
                if let Some(default) = default {
                    let reachable = interaction_reachability_for_target(default, target, layout)?;
                    if reachable != InteractionReachability::Reachable {
                        return Ok(reachable);
                    }
                }
            }
            SabreNodeKind::Synchronize => {}
        }
    }
    Ok(InteractionReachability::Reachable)
}

fn swap_operation(swap: [PhysicalQubit; 2]) -> Operation {
    Operation {
        instruction: Instruction::Standard(StandardGate::SWAP),
        qubits: smallvec![swap[0].qubit(), swap[1].qubit()],
        params: SmallVec::new(),
        label: None,
    }
}

pub(crate) fn compare_trial_quality(
    objective: SabreTrialObjective,
    left: TrialQuality,
    left_index: usize,
    right: TrialQuality,
    right_index: usize,
) -> Ordering {
    match objective {
        SabreTrialObjective::SwapCount => left
            .swap_count
            .cmp(&right.swap_count)
            .then_with(|| left_index.cmp(&right_index)),
        SabreTrialObjective::Depth => left
            .two_qubit_depth
            .cmp(&right.two_qubit_depth)
            .then_with(|| left_index.cmp(&right_index)),
        SabreTrialObjective::NativeQualityWithinSwapBudget => left
            .native_two_qubit_ops
            .cmp(&right.native_two_qubit_ops)
            .then_with(|| {
                left.native_two_qubit_depth
                    .cmp(&right.native_two_qubit_depth)
            })
            .then_with(|| match (left.error, right.error) {
                (Some(left), Some(right)) => left.compare(right),
                (Some(_), None) => Ordering::Less,
                (None, Some(_)) => Ordering::Greater,
                (None, None) => Ordering::Equal,
            })
            .then_with(|| compare_trial_duration(left, right))
            .then_with(|| left.native_total_ops.cmp(&right.native_total_ops))
            .then_with(|| left.swap_count.cmp(&right.swap_count))
            .then_with(|| left.two_qubit_depth.cmp(&right.two_qubit_depth))
            .then_with(|| left.operation_count.cmp(&right.operation_count))
            .then_with(|| left_index.cmp(&right_index)),
        SabreTrialObjective::DepthThenSwap => left
            .two_qubit_depth
            .cmp(&right.two_qubit_depth)
            .then_with(|| left.swap_count.cmp(&right.swap_count))
            .then_with(|| left.operation_count.cmp(&right.operation_count))
            .then_with(|| left_index.cmp(&right_index)),
    }
}

fn compare_trial_duration(left: TrialQuality, right: TrialQuality) -> Ordering {
    match (left.duration, right.duration) {
        (Some(left_duration), Some(right_duration)) => left_duration
            .unavailable_count
            .cmp(&right_duration.unavailable_count)
            .then_with(|| {
                left_duration
                    .imputed_count
                    .cmp(&right_duration.imputed_count)
            })
            .then_with(|| match (left.makespan, right.makespan) {
                (Some(left), Some(right)) => left.total_cmp(&right),
                (Some(_), None) => Ordering::Less,
                (None, Some(_)) => Ordering::Greater,
                (None, None) => Ordering::Equal,
            })
            .then_with(|| {
                left_duration
                    .duration_work
                    .total_cmp(&right_duration.duration_work)
            }),
        (Some(_), None) => Ordering::Less,
        (None, Some(_)) => Ordering::Greater,
        (None, None) => Ordering::Equal,
    }
}

pub(crate) fn trial_swap_limit(
    objective: SabreTrialObjective,
    swap_regret_ratio: f64,
    qualities: impl Iterator<Item = TrialQuality>,
) -> usize {
    if objective != SabreTrialObjective::NativeQualityWithinSwapBudget {
        return usize::MAX;
    }
    let best = qualities
        .map(|quality| quality.swap_count)
        .min()
        .unwrap_or(0);
    let regret = ((best as f64) * swap_regret_ratio).ceil();
    let regret = if regret >= usize::MAX as f64 {
        usize::MAX
    } else {
        regret as usize
    };
    best.saturating_add(regret)
}

fn trial_quality(
    operations: &[Operation],
    swap_count: usize,
    target: &RoutingTarget,
) -> Result<TrialQuality, CompilerError> {
    let abstract_two_qubit_depth = two_qubit_depth(operations);
    let abstract_operation_count = operation_count(operations);
    if !target.native_cost_enabled {
        return Ok(TrialQuality {
            swap_count,
            two_qubit_depth: abstract_two_qubit_depth,
            operation_count: abstract_operation_count,
            native_two_qubit_ops: two_qubit_operation_count(operations),
            native_two_qubit_depth: abstract_two_qubit_depth,
            native_total_ops: abstract_operation_count,
            error: None,
            duration: None,
            makespan: None,
            unknown_loop_count: 0,
        });
    }

    let native = native_plan_cost_for_operations(operations, target)?;
    Ok(TrialQuality {
        swap_count,
        two_qubit_depth: abstract_two_qubit_depth,
        operation_count: abstract_operation_count,
        native_two_qubit_ops: native.static_native.native_two_qubit_ops as usize,
        native_two_qubit_depth: native_two_qubit_depth(operations, target)?,
        native_total_ops: native.static_native.native_total_ops as usize,
        error: native.path_error,
        duration: native.path_duration,
        makespan: native_makespan(operations, target)?,
        unknown_loop_count: native.unknown_loop_count,
    })
}

#[derive(Debug, Clone, Copy, Default)]
struct NativeCircuitCost {
    static_native: NativePlanCost,
    path_error: Option<RobustErrorKey>,
    path_duration: Option<RobustDurationKey>,
    unknown_loop_count: usize,
}

impl NativeCircuitCost {
    fn append_gate(&mut self, cost: NativePlanCost) {
        self.static_native = combine_native_identity(self.static_native, cost);
        self.path_error = combine_error_identity(self.path_error, cost.error);
        self.path_duration = combine_duration_identity(self.path_duration, cost.duration);
    }

    fn append_sequence(&mut self, other: Self) {
        self.static_native = combine_native_identity(self.static_native, other.static_native);
        self.path_error = combine_error_identity(self.path_error, other.path_error);
        self.path_duration = combine_duration_identity(self.path_duration, other.path_duration);
        self.unknown_loop_count = self
            .unknown_loop_count
            .saturating_add(other.unknown_loop_count);
    }

    fn add_static_branch(&mut self, branch: Self) {
        self.static_native = combine_native_identity(self.static_native, branch.static_native);
    }
}

fn combine_native_identity(left: NativePlanCost, right: NativePlanCost) -> NativePlanCost {
    NativePlanCost {
        native_two_qubit_ops: left
            .native_two_qubit_ops
            .saturating_add(right.native_two_qubit_ops),
        native_total_ops: left.native_total_ops.saturating_add(right.native_total_ops),
        error: combine_error_identity(left.error, right.error),
        duration: combine_duration_identity(left.duration, right.duration),
    }
}

fn combine_error_identity(
    left: Option<RobustErrorKey>,
    right: Option<RobustErrorKey>,
) -> Option<RobustErrorKey> {
    match (left, right) {
        (Some(left), Some(right)) => Some(left.combine(right)),
        (Some(value), None) | (None, Some(value)) => Some(value),
        (None, None) => None,
    }
}

fn combine_duration_identity(
    left: Option<RobustDurationKey>,
    right: Option<RobustDurationKey>,
) -> Option<RobustDurationKey> {
    match (left, right) {
        (Some(left), Some(right)) => Some(left.combine(right)),
        (Some(value), None) | (None, Some(value)) => Some(value),
        (None, None) => None,
    }
}

fn worst_error(
    left: Option<RobustErrorKey>,
    right: Option<RobustErrorKey>,
) -> Option<RobustErrorKey> {
    match (left, right) {
        (Some(left), Some(right)) => Some(if left.compare(right).is_ge() {
            left
        } else {
            right
        }),
        (Some(value), None) | (None, Some(value)) => Some(value),
        (None, None) => None,
    }
}

fn worst_duration(
    left: Option<RobustDurationKey>,
    right: Option<RobustDurationKey>,
) -> Option<RobustDurationKey> {
    match (left, right) {
        (Some(left), Some(right)) => Some(if left.compare(right).is_ge() {
            left
        } else {
            right
        }),
        (Some(value), None) | (None, Some(value)) => Some(value),
        (None, None) => None,
    }
}

fn repeat_error(value: Option<RobustErrorKey>, count: u128) -> Option<RobustErrorKey> {
    value.map(|value| RobustErrorKey {
        unavailable_count: (u128::from(value.unavailable_count) * count).min(u128::from(u32::MAX))
            as u32,
        imputed_count: (u128::from(value.imputed_count) * count).min(u128::from(u32::MAX)) as u32,
        log_error: value.log_error * count as f64,
    })
}

fn repeat_duration(value: Option<RobustDurationKey>, count: u128) -> Option<RobustDurationKey> {
    value.map(|value| RobustDurationKey {
        unavailable_count: (u128::from(value.unavailable_count) * count).min(u128::from(u32::MAX))
            as u32,
        imputed_count: (u128::from(value.imputed_count) * count).min(u128::from(u32::MAX)) as u32,
        duration_work: value.duration_work * count as f64,
    })
}

fn native_plan_cost_for_operations(
    operations: &[Operation],
    target: &RoutingTarget,
) -> Result<NativeCircuitCost, CompilerError> {
    let mut total = NativeCircuitCost::default();
    for operation in operations {
        match &operation.instruction {
            Instruction::Standard(StandardGate::GPhase) => {}
            Instruction::Standard(_) | Instruction::McGate(_) => {
                let state = DeviceGateState::from_instruction(
                    &operation.instruction,
                    operation
                        .qubits
                        .iter()
                        .copied()
                        .map(PhysicalQubit::from_qubit)
                        .collect(),
                )
                .ok_or_else(|| {
                    CompilerError::InvariantViolation(format!(
                        "missing native-cost key for routed operation {}",
                        operation.instruction
                    ))
                })?;
                let cost = match target.native_cost(&state) {
                    Some(cost) => cost,
                    None if target.unsupported_native_plan(&state).is_some() => {
                        return Err(CompilerError::DeviceLoweringFailed(
                            target
                                .unsupported_native_plan(&state)
                                .expect("guarded unsupported plan exists")
                                .clone(),
                        ));
                    }
                    None => {
                        return Err(CompilerError::InvariantViolation(format!(
                            "routed operation {} on {:?} was not prepared in the native plan catalog",
                            operation.instruction, state.ordered_qargs
                        )));
                    }
                };
                total.append_gate(cost);
            }
            Instruction::ClassicalControl(control) => match control {
                ClassicalControlOp::If(op) => {
                    let then_cost =
                        native_plan_cost_for_operations(op.then_body().operations(), target)?;
                    let else_cost = op
                        .else_body()
                        .map(|body| native_plan_cost_for_operations(body.operations(), target))
                        .transpose()?
                        .unwrap_or_default();
                    total.add_static_branch(then_cost);
                    total.add_static_branch(else_cost);
                    total.path_error = combine_error_identity(
                        total.path_error,
                        worst_error(then_cost.path_error, else_cost.path_error),
                    );
                    total.path_duration = combine_duration_identity(
                        total.path_duration,
                        worst_duration(then_cost.path_duration, else_cost.path_duration),
                    );
                    total.unknown_loop_count = total.unknown_loop_count.saturating_add(
                        then_cost
                            .unknown_loop_count
                            .max(else_cost.unknown_loop_count),
                    );
                }
                ClassicalControlOp::While(op) => {
                    let mut body = native_plan_cost_for_operations(op.body().operations(), target)?;
                    body.unknown_loop_count = body.unknown_loop_count.saturating_add(1);
                    total.append_sequence(body);
                }
                ClassicalControlOp::For(op) => {
                    let mut body = native_plan_cost_for_operations(op.body().operations(), target)?;
                    total.add_static_branch(body);
                    if let Some(iterations) = op.static_iteration_count() {
                        body.static_native = NativePlanCost::default();
                        body.path_error = repeat_error(body.path_error, iterations);
                        body.path_duration = repeat_duration(body.path_duration, iterations);
                    } else {
                        body.static_native = NativePlanCost::default();
                        body.unknown_loop_count = body.unknown_loop_count.saturating_add(1);
                    }
                    total.append_sequence(body);
                }
                ClassicalControlOp::Switch(op) => {
                    let mut worst_path = NativeCircuitCost::default();
                    for case in op.cases() {
                        let branch =
                            native_plan_cost_for_operations(case.body().operations(), target)?;
                        total.add_static_branch(branch);
                        worst_path.path_error =
                            worst_error(worst_path.path_error, branch.path_error);
                        worst_path.path_duration =
                            worst_duration(worst_path.path_duration, branch.path_duration);
                        worst_path.unknown_loop_count =
                            worst_path.unknown_loop_count.max(branch.unknown_loop_count);
                    }
                    if let Some(body) = op.default() {
                        let branch = native_plan_cost_for_operations(body.operations(), target)?;
                        total.add_static_branch(branch);
                        worst_path.path_error =
                            worst_error(worst_path.path_error, branch.path_error);
                        worst_path.path_duration =
                            worst_duration(worst_path.path_duration, branch.path_duration);
                        worst_path.unknown_loop_count =
                            worst_path.unknown_loop_count.max(branch.unknown_loop_count);
                    }
                    total.path_error =
                        combine_error_identity(total.path_error, worst_path.path_error);
                    total.path_duration =
                        combine_duration_identity(total.path_duration, worst_path.path_duration);
                    total.unknown_loop_count = total
                        .unknown_loop_count
                        .saturating_add(worst_path.unknown_loop_count);
                }
                ClassicalControlOp::Break | ClassicalControlOp::Continue => {}
            },
            Instruction::ClassicalData(_) | Instruction::Directive(_) | Instruction::Delay => {}
            Instruction::UnitaryGate(_) | Instruction::CircuitGate(_) => {
                return Err(CompilerError::InvariantViolation(format!(
                    "unlowerable routed operation {} reached native trial scoring",
                    operation.instruction
                )));
            }
        }
    }
    Ok(total)
}

fn native_makespan(
    operations: &[Operation],
    target: &RoutingTarget,
) -> Result<Option<f64>, CompilerError> {
    if !target.native_duration_enabled {
        return Ok(None);
    }
    let mut ready = BTreeMap::<Qubit, f64>::new();
    if !schedule_native_operations(operations, target, &mut ready)? {
        return Ok(None);
    }
    Ok(Some(ready.values().copied().fold(0.0_f64, f64::max)))
}

fn schedule_native_operations(
    operations: &[Operation],
    target: &RoutingTarget,
    ready: &mut BTreeMap<Qubit, f64>,
) -> Result<bool, CompilerError> {
    for operation in operations {
        match &operation.instruction {
            Instruction::Standard(StandardGate::GPhase) => {}
            Instruction::Standard(_) | Instruction::McGate(_) => {
                let state = DeviceGateState::from_instruction(
                    &operation.instruction,
                    operation
                        .qubits
                        .iter()
                        .copied()
                        .map(PhysicalQubit::from_qubit)
                        .collect(),
                )
                .ok_or_else(|| {
                    CompilerError::InvariantViolation(format!(
                        "missing native-duration key for routed operation {}",
                        operation.instruction
                    ))
                })?;
                let Some(timing) = target.native_timing(&state) else {
                    if let Some(failure) = target.unsupported_native_plan(&state) {
                        return Err(CompilerError::DeviceLoweringFailed(failure.clone()));
                    }
                    return Err(CompilerError::InvariantViolation(format!(
                        "routed operation {} on {:?} was not prepared for native duration",
                        operation.instruction, state.ordered_qargs
                    )));
                };
                let Some(leaves) = timing else {
                    return Ok(false);
                };
                for leaf in leaves {
                    let qargs = leaf
                        .ordered_qargs
                        .iter()
                        .copied()
                        .map(PhysicalQubit::qubit)
                        .collect::<SmallVec<[Qubit; 2]>>();
                    schedule_atomic_duration(&qargs, leaf.duration, ready);
                }
            }
            Instruction::ClassicalControl(control) => {
                let Some(duration) = native_control_flow_duration(control, target)? else {
                    return Ok(false);
                };
                schedule_atomic_duration(&operation.qubits, duration, ready);
            }
            Instruction::ClassicalData(_) | Instruction::Directive(_) | Instruction::Delay => {
                schedule_atomic_duration(&operation.qubits, 0.0, ready);
            }
            Instruction::UnitaryGate(_) | Instruction::CircuitGate(_) => {
                return Err(CompilerError::InvariantViolation(format!(
                    "unlowerable routed operation {} reached native makespan scoring",
                    operation.instruction
                )));
            }
        }
    }
    Ok(true)
}

fn native_control_flow_duration(
    control: &ClassicalControlOp,
    target: &RoutingTarget,
) -> Result<Option<f64>, CompilerError> {
    match control {
        ClassicalControlOp::If(op) => {
            let Some(then_duration) =
                native_sequence_duration(op.then_body().operations(), target)?
            else {
                return Ok(None);
            };
            let else_duration = if let Some(body) = op.else_body() {
                let Some(duration) = native_sequence_duration(body.operations(), target)? else {
                    return Ok(None);
                };
                duration
            } else {
                0.0
            };
            Ok(Some(then_duration.max(else_duration)))
        }
        ClassicalControlOp::While(op) => {
            let duration = native_sequence_duration(op.body().operations(), target)?;
            Ok(duration.filter(|duration| *duration == 0.0))
        }
        ClassicalControlOp::For(op) => {
            let Some(duration) = native_sequence_duration(op.body().operations(), target)? else {
                return Ok(None);
            };
            let Some(iterations) = op.static_iteration_count() else {
                return Ok((duration == 0.0).then_some(0.0));
            };
            Ok(Some(duration * iterations as f64))
        }
        ClassicalControlOp::Switch(op) => {
            let mut duration = 0.0_f64;
            for case in op.cases() {
                let Some(branch) = native_sequence_duration(case.body().operations(), target)?
                else {
                    return Ok(None);
                };
                duration = duration.max(branch);
            }
            if let Some(body) = op.default() {
                let Some(branch) = native_sequence_duration(body.operations(), target)? else {
                    return Ok(None);
                };
                duration = duration.max(branch);
            }
            Ok(Some(duration))
        }
        ClassicalControlOp::Break | ClassicalControlOp::Continue => Ok(Some(0.0)),
    }
}

fn native_sequence_duration(
    operations: &[Operation],
    target: &RoutingTarget,
) -> Result<Option<f64>, CompilerError> {
    let mut ready = BTreeMap::new();
    if !schedule_native_operations(operations, target, &mut ready)? {
        return Ok(None);
    }
    Ok(Some(ready.values().copied().fold(0.0_f64, f64::max)))
}

fn schedule_atomic_duration(qargs: &[Qubit], duration: f64, ready: &mut BTreeMap<Qubit, f64>) {
    let start = qargs
        .iter()
        .filter_map(|qubit| ready.get(qubit).copied())
        .fold(0.0_f64, f64::max);
    let finish = start + duration;
    for qubit in qargs {
        ready.insert(*qubit, finish);
    }
}

fn native_two_qubit_depth(
    operations: &[Operation],
    target: &RoutingTarget,
) -> Result<usize, CompilerError> {
    let mut qubit_depths = BTreeMap::<Qubit, usize>::new();
    let mut max_depth = 0;
    for operation in operations {
        let local_depth = operation_local_native_two_qubit_depth(operation, target)?;
        if local_depth == 0 {
            continue;
        }
        let base = operation
            .qubits
            .iter()
            .map(|qubit| qubit_depths.get(qubit).copied().unwrap_or(0))
            .max()
            .unwrap_or(0);
        let depth = base + local_depth;
        for qubit in &operation.qubits {
            qubit_depths.insert(*qubit, depth);
        }
        max_depth = max_depth.max(depth);
    }
    Ok(max_depth)
}

fn operation_local_native_two_qubit_depth(
    operation: &Operation,
    target: &RoutingTarget,
) -> Result<usize, CompilerError> {
    match &operation.instruction {
        Instruction::Standard(StandardGate::GPhase) => Ok(0),
        Instruction::Standard(_) | Instruction::McGate(_) => {
            let state = DeviceGateState::from_instruction(
                &operation.instruction,
                operation
                    .qubits
                    .iter()
                    .copied()
                    .map(PhysicalQubit::from_qubit)
                    .collect(),
            )
            .ok_or_else(|| {
                CompilerError::InvariantViolation(format!(
                    "missing native-depth key for routed operation {}",
                    operation.instruction
                ))
            })?;
            match target.native_cost(&state) {
                Some(cost) => Ok(cost.native_two_qubit_ops as usize),
                None if target.unsupported_native_plan(&state).is_some() => {
                    Err(CompilerError::DeviceLoweringFailed(
                        target
                            .unsupported_native_plan(&state)
                            .expect("guarded unsupported plan exists")
                            .clone(),
                    ))
                }
                None => Err(CompilerError::InvariantViolation(format!(
                    "routed operation {} on {:?} was not prepared in the native plan catalog",
                    operation.instruction, state.ordered_qargs
                ))),
            }
        }
        Instruction::ClassicalControl(ClassicalControlOp::If(op)) => {
            let then_depth = native_two_qubit_depth(op.then_body().operations(), target)?;
            let else_depth = op
                .else_body()
                .map(|body| native_two_qubit_depth(body.operations(), target))
                .transpose()?
                .unwrap_or(0);
            Ok(then_depth.max(else_depth))
        }
        Instruction::ClassicalControl(ClassicalControlOp::While(op)) => {
            native_two_qubit_depth(op.body().operations(), target)
        }
        Instruction::ClassicalControl(ClassicalControlOp::For(op)) => {
            let body_depth = native_two_qubit_depth(op.body().operations(), target)?;
            let iterations = op
                .static_iteration_count()
                .unwrap_or(1)
                .min(usize::MAX as u128) as usize;
            Ok(body_depth.saturating_mul(iterations))
        }
        Instruction::ClassicalControl(ClassicalControlOp::Switch(op)) => {
            let mut depth = 0;
            for case in op.cases() {
                depth = depth.max(native_two_qubit_depth(case.body().operations(), target)?);
            }
            if let Some(body) = op.default() {
                depth = depth.max(native_two_qubit_depth(body.operations(), target)?);
            }
            Ok(depth)
        }
        _ => Ok(0),
    }
}

fn two_qubit_operation_count(operations: &[Operation]) -> usize {
    operations
        .iter()
        .map(|operation| match &operation.instruction {
            Instruction::ClassicalControl(ClassicalControlOp::If(op)) => {
                two_qubit_operation_count(op.then_body().operations())
                    + op.else_body()
                        .map(|body| two_qubit_operation_count(body.operations()))
                        .unwrap_or(0)
            }
            Instruction::ClassicalControl(ClassicalControlOp::While(op)) => {
                two_qubit_operation_count(op.body().operations())
            }
            Instruction::ClassicalControl(ClassicalControlOp::For(op)) => {
                two_qubit_operation_count(op.body().operations())
            }
            Instruction::ClassicalControl(ClassicalControlOp::Switch(op)) => {
                op.cases()
                    .iter()
                    .map(|case| two_qubit_operation_count(case.body().operations()))
                    .sum::<usize>()
                    + op.default()
                        .map(|body| two_qubit_operation_count(body.operations()))
                        .unwrap_or(0)
            }
            _ => usize::from(operation.qubits.len() == 2),
        })
        .sum()
}

/// Estimates ASAP two-qubit depth for trial ranking.
///
/// This is not a full scheduler. Control-flow contributes the maximum local
/// branch or body depth, and parent operations are chained by used qubits.
fn two_qubit_depth(operations: &[Operation]) -> usize {
    let mut qubit_depths = BTreeMap::<Qubit, usize>::new();
    let mut max_depth = 0usize;

    for operation in operations {
        let local_depth = operation_local_two_qubit_depth(operation);
        if local_depth == 0 {
            continue;
        }

        let base = operation
            .qubits
            .iter()
            .map(|qubit| qubit_depths.get(qubit).copied().unwrap_or(0))
            .max()
            .unwrap_or(0);
        let depth = base + local_depth;
        for qubit in &operation.qubits {
            qubit_depths.insert(*qubit, depth);
        }
        max_depth = max_depth.max(depth);
    }

    max_depth
}

fn operation_local_two_qubit_depth(operation: &Operation) -> usize {
    match &operation.instruction {
        Instruction::ClassicalControl(ClassicalControlOp::If(op)) => {
            let then_depth = two_qubit_depth(op.then_body().operations());
            let else_depth = op
                .else_body()
                .map(|body| two_qubit_depth(body.operations()))
                .unwrap_or(0);
            then_depth.max(else_depth)
        }
        Instruction::ClassicalControl(ClassicalControlOp::While(op)) => {
            two_qubit_depth(op.body().operations())
        }
        Instruction::ClassicalControl(ClassicalControlOp::For(op)) => {
            let iterations = op
                .static_iteration_count()
                .unwrap_or(1)
                .min(usize::MAX as u128) as usize;
            two_qubit_depth(op.body().operations()).saturating_mul(iterations)
        }
        Instruction::ClassicalControl(ClassicalControlOp::Switch(op)) => op
            .cases()
            .iter()
            .map(|case| two_qubit_depth(case.body().operations()))
            .chain(
                op.default()
                    .into_iter()
                    .map(|body| two_qubit_depth(body.operations())),
            )
            .max()
            .unwrap_or(0),
        _ if operation.qubits.len() == 2 => 1,
        _ => 0,
    }
}

fn operation_count(operations: &[Operation]) -> usize {
    operations
        .iter()
        .map(|operation| {
            1 + match &operation.instruction {
                Instruction::ClassicalControl(ClassicalControlOp::If(op)) => {
                    operation_count(op.then_body().operations())
                        + op.else_body()
                            .map(|body| operation_count(body.operations()))
                            .unwrap_or(0)
                }
                Instruction::ClassicalControl(ClassicalControlOp::While(op)) => {
                    operation_count(op.body().operations())
                }
                Instruction::ClassicalControl(ClassicalControlOp::For(op)) => {
                    operation_count(op.body().operations())
                }
                Instruction::ClassicalControl(ClassicalControlOp::Switch(op)) => {
                    op.cases()
                        .iter()
                        .map(|case| operation_count(case.body().operations()))
                        .sum::<usize>()
                        + op.default()
                            .map(|body| operation_count(body.operations()))
                            .unwrap_or(0)
                }
                _ => 0,
            }
        })
        .sum()
}

#[cfg(test)]
#[path = "score_test.rs"]
mod score_test;
