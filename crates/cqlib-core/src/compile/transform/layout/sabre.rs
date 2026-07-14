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

//! SABRE initial-layout adapter.
//!
//! This module owns layout-candidate generation, objective scoring, and
//! [`LayoutResult`] construction. The SABRE core remains in
//! [`crate::compile::sabre`] and is used here only through crate-internal
//! routing primitives.
//!
//! The dependency direction is intentional: layout may call SABRE to refine and
//! score candidate initial layouts, but the standalone SABRE routing module must
//! not depend on layout algorithms or layout result types.

use super::{
    CircuitLayoutAnalysis, LayoutDiagnostics, LayoutObjective, LayoutResult, LayoutScore,
    PhysicalLayoutGraph, Vf2LayoutConfig, analyze_circuit_for_layout, build_physical_layout_graph,
    greedy_layout_prepared, is_perfect_layout, vf2_perfect_layout_prepared,
};
use crate::circuit::Circuit;
use crate::compile::CompilerError;
use crate::compile::sabre::{
    InteractionReachability, RoutingTarget, SabreConfig, SabreDag, TrialQuality,
    compare_trial_quality, interaction_reachability_for_target,
    normalize_initial_layout_for_target, route_trial_unchecked, trial_seeds,
};
use crate::device::{Device, Layout, LogicalQubit, PhysicalQubit};
use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use rand::{Rng, SeedableRng};
use rayon::prelude::*;
use std::collections::{BTreeMap, BTreeSet, HashSet, VecDeque};

/// Circuit-side data prepared once for repeated SABRE layout selection.
///
/// The fields are intentionally private so the interaction analysis and the
/// dependency DAGs cannot be supplied from different circuits. A prepared
/// value can be reused with different physical targets, objectives, and SABRE
/// configurations.
#[derive(Debug, Clone)]
pub struct PreparedSabreCircuit {
    analysis: CircuitLayoutAnalysis,
    routing_dag: SabreDag,
    refinement_dag: SabreDag,
    logical_components: Vec<Vec<LogicalQubit>>,
}

impl PreparedSabreCircuit {
    /// Returns the reusable circuit layout analysis.
    pub fn analysis(&self) -> &CircuitLayoutAnalysis {
        &self.analysis
    }

    /// Returns logical qubits in source-circuit order.
    pub fn logical_qubits(&self) -> &[LogicalQubit] {
        &self.analysis.logical_qubits
    }
}

/// Prepares the circuit-side analysis and dependency models used by SABRE.
///
/// Preparing once and calling [`sabre_layout_prepared`] repeatedly avoids
/// rebuilding circuit analysis and dependency DAGs for every physical target
/// or SABRE configuration.
///
/// # Errors
///
/// Returns [`CompilerError::InvalidInput`] when the circuit contains an
/// operation that must be decomposed before SABRE, and propagates circuit
/// analysis or DAG-construction failures.
pub fn prepare_sabre_circuit(circuit: &Circuit) -> Result<PreparedSabreCircuit, CompilerError> {
    let analysis = analyze_circuit_for_layout(circuit)?;
    let routing_dag = SabreDag::from_operations(circuit.operations())?;
    let refinement_dag = SabreDag::refinement_workload(circuit.operations())?;
    let logical_components = logical_interaction_components(&analysis);
    Ok(PreparedSabreCircuit {
        analysis,
        routing_dag,
        refinement_dag,
        logical_components,
    })
}

/// Selects an initial layout with SABRE forward/backward refinement.
///
/// This function only returns the refined initial layout. It does not insert
/// SWAP operations or rebuild a physical circuit; callers that need routing
/// should run the SABRE routing core after selecting a layout.
///
/// Candidate layouts include deterministic component-feasible anchors,
/// reachable greedy/VF2 layouts when available, and randomized feasible trials
/// controlled by [`SabreConfig::layout_trials`]. Each candidate is refined
/// through SABRE forward/backward passes and ranked by final-route quality,
/// then by [`LayoutObjective`] as a tie-breaker.
///
/// # Errors
///
/// Returns [`CompilerError::InvalidInput`] for invalid SABRE layout
/// configuration, insufficient usable physical qubits, unreachable
/// interactions in the usable topology, or unsupported circuit operations.
///
/// # Examples
///
/// ```rust
/// use cqlib_core::circuit::{Circuit, Qubit};
/// use cqlib_core::compile::sabre::SabreConfig;
/// use cqlib_core::compile::transform::{LayoutObjective, sabre_layout};
/// use cqlib_core::device::Device;
///
/// let mut circuit = Circuit::new(3);
/// circuit.cx(Qubit::new(0), Qubit::new(2)).unwrap();
/// let device = Device::line("line-3", 3).unwrap();
///
/// let result = sabre_layout(
///     &circuit,
///     &device,
///     &LayoutObjective::topology_only(),
///     &SabreConfig::default(),
/// )
/// .unwrap();
/// assert!(result.score.is_some());
/// ```
pub fn sabre_layout(
    circuit: &Circuit,
    device: &Device,
    objective: &LayoutObjective,
    config: &SabreConfig,
) -> Result<LayoutResult, CompilerError> {
    let prepared = prepare_sabre_circuit(circuit)?;
    let physical = build_physical_layout_graph(device)?;
    sabre_layout_prepared(&prepared, &physical, objective, config)
}

/// Selects a SABRE initial layout from prepared circuit-side data.
///
/// Reuse `prepared` across targets or configurations to avoid repeating
/// circuit analysis and dependency-DAG construction.
///
/// This API intentionally replaces the former
/// `sabre_layout_prepared(circuit, analysis, physical, objective, config)`
/// signature. Keeping circuit-derived data in [`PreparedSabreCircuit`] prevents
/// callers from combining a circuit with analysis produced from another one.
///
/// # Errors
///
/// Returns [`CompilerError::InvalidInput`] for invalid SABRE configuration,
/// insufficient or component-incompatible physical capacity, and unreachable
/// interactions. Objective-scoring and routing failures are propagated.
///
/// # Examples
///
/// ```rust
/// use cqlib_core::circuit::{Circuit, Qubit};
/// use cqlib_core::compile::sabre::SabreConfig;
/// use cqlib_core::compile::transform::layout::build_physical_layout_graph;
/// use cqlib_core::compile::transform::{
///     LayoutObjective, prepare_sabre_circuit, sabre_layout_prepared,
/// };
/// use cqlib_core::device::Device;
///
/// let mut circuit = Circuit::new(3);
/// circuit.cx(Qubit::new(0), Qubit::new(2))?;
/// let prepared = prepare_sabre_circuit(&circuit)?;
/// let physical = build_physical_layout_graph(&Device::line("line-3", 3)?)?;
///
/// let result = sabre_layout_prepared(
///     &prepared,
///     &physical,
///     &LayoutObjective::topology_only(),
///     &SabreConfig::deterministic_seeded(42),
/// )?;
/// assert_eq!(result.layout.logical_qubits().count(), 3);
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
pub fn sabre_layout_prepared(
    prepared: &PreparedSabreCircuit,
    physical: &PhysicalLayoutGraph,
    objective: &LayoutObjective,
    config: &SabreConfig,
) -> Result<LayoutResult, CompilerError> {
    validate_layout_config(config)?;
    let target = RoutingTarget::from_physical(physical)?;
    let analysis = &prepared.analysis;
    let sabre = &prepared.routing_dag;
    let forwards = &prepared.refinement_dag;
    let backwards = forwards.reverse_interactions();
    let logical_qubits = analysis.logical_qubits.clone();

    if logical_qubits.len() > target.physical_qubits.len() {
        return Err(CompilerError::InvalidInput(format!(
            "sabre layout requires at least as many usable physical qubits as logical qubits; got {} logical qubits and {} usable physical qubits",
            logical_qubits.len(),
            target.physical_qubits.len()
        )));
    }

    let mut rng = StdRng::seed_from_u64(config.seed.unwrap_or_else(rand::random));
    let candidates = initial_layout_candidates(
        analysis,
        physical,
        &target,
        objective,
        &prepared.logical_components,
        config.layout_trials,
        &mut rng,
    )?;
    let candidates_evaluated = candidates.len();
    let trials = candidates
        .into_iter()
        .enumerate()
        .map(|(index, layout)| {
            let refinement_seeds = (0..config.refinement_iterations)
                .map(|_| (rng.random(), rng.random()))
                .collect();
            CandidateTrial {
                index,
                layout,
                refinement_seeds,
                scoring_seed: rng.random(),
            }
        })
        .collect::<Vec<_>>();

    let evaluations = trials
        .into_par_iter()
        .map(|trial| {
            if matches!(
                interaction_reachability_for_target(sabre, &target, &trial.layout)?,
                InteractionReachability::Unreachable { .. }
            ) {
                return Ok(None);
            }
            let mut refined = trial.layout;
            for (forward_seed, backward_seed) in trial.refinement_seeds {
                // One refinement iteration routes forward, keeps the final
                // layout, then routes the reversed interaction DAG. This is
                // the SABRE layout-refinement loop, not final circuit routing.
                refined = match route_trial_unchecked(
                    forwards,
                    &target,
                    &refined,
                    &config.heuristic,
                    forward_seed,
                ) {
                    Ok(result) => result.final_layout,
                    Err(error) => return Err(error),
                };

                refined = match route_trial_unchecked(
                    &backwards,
                    &target,
                    &refined,
                    &config.heuristic,
                    backward_seed,
                ) {
                    Ok(result) => result.final_layout,
                    Err(error) => return Err(error),
                };
            }

            // Rank refined layouts by how well they route the original DAG.
            // Multiple scoring trials reduce seed sensitivity without exposing
            // final SWAP insertion through this layout API.
            let route_quality =
                match best_route_quality(sabre, &target, &refined, config, trial.scoring_seed) {
                    Ok(quality) => quality,
                    Err(error) => return Err(error),
                };
            let score = objective.score_layout(analysis, physical, &refined)?;
            Ok(Some(CandidateEvaluation {
                index: trial.index,
                route_quality,
                layout: refined,
                score,
            }))
        })
        .collect::<Result<Vec<_>, CompilerError>>()?
        .into_iter()
        .flatten()
        .collect::<Vec<_>>();

    let best = evaluations
        .into_iter()
        .min_by(|left, right| {
            compare_trial_quality(
                config.trial_objective,
                left.route_quality,
                0,
                right.route_quality,
                0,
            )
            .then_with(|| left.score.total.total_cmp(&right.score.total))
            .then_with(|| left.index.cmp(&right.index))
        })
        .ok_or_else(|| {
            CompilerError::InvalidInput(
                "sabre layout found no candidate whose interactions are connected in the usable topology"
                    .to_string(),
            )
        })?;
    let swap_count = best.route_quality.swap_count;
    let layout = best.layout;
    let score = best.score;
    let is_perfect = is_perfect_layout(analysis, physical, &layout);

    Ok(LayoutResult {
        layout,
        score: Some(score.clone()),
        diagnostics: LayoutDiagnostics {
            is_perfect,
            candidates_evaluated,
            used_fidelity: score.used_fidelity,
            notes: vec![format!(
                "selected SABRE refined layout with {swap_count} final-route swaps"
            )],
        },
    })
}

struct CandidateTrial {
    /// Stable candidate order used as the final deterministic tie-breaker.
    index: usize,
    /// Candidate layout before forward/backward refinement.
    layout: Layout,
    /// Seed pairs for forward and backward refinement route trials.
    refinement_seeds: Vec<(u64, u64)>,
    /// Seed used to derive final-route scoring trials.
    scoring_seed: u64,
}

struct CandidateEvaluation {
    /// Original candidate index, retained after parallel evaluation.
    index: usize,
    /// Best final-route quality observed for the refined layout.
    route_quality: TrialQuality,
    /// Refined initial layout.
    layout: Layout,
    /// Objective score for the refined initial layout.
    score: LayoutScore,
}

/// Validates SABRE settings that are specific to layout selection.
///
/// The routing core owns generic SABRE validation. This adapter adds checks for
/// layout-only knobs that must be non-zero before candidate generation or
/// final-route scoring starts.
fn validate_layout_config(config: &SabreConfig) -> Result<(), CompilerError> {
    if config.layout_trials == 0 {
        return Err(CompilerError::InvalidInput(
            "sabre layout_trials must be greater than zero".to_string(),
        ));
    }
    if config.layout_scoring_trials == 0 {
        return Err(CompilerError::InvalidInput(
            "sabre layout_scoring_trials must be greater than zero".to_string(),
        ));
    }
    crate::compile::sabre::validate_config(config)
}

/// Generates the candidate set refined by SABRE layout.
///
/// Candidates include deterministic anchors, opportunistic greedy/VF2 results,
/// and random physical orders. The result is deduplicated in logical-qubit
/// order so duplicate layouts from different sources are evaluated once.
fn initial_layout_candidates(
    analysis: &CircuitLayoutAnalysis,
    physical: &PhysicalLayoutGraph,
    target: &RoutingTarget,
    objective: &LayoutObjective,
    logical_components: &[Vec<LogicalQubit>],
    layout_trials: usize,
    rng: &mut StdRng,
) -> Result<Vec<Layout>, CompilerError> {
    let mut candidates = Vec::new();
    let logical_qubits = &analysis.logical_qubits;
    let physical_components = physical.connected_components();
    let deterministic_assignment =
        component_assignment(logical_components, physical_components, None)
            .ok_or_else(|| component_capacity_error(logical_components, physical_components))?;

    // Feasible deterministic anchors guarantee that topology reachability does
    // not depend on random seed, even on a disconnected physical target.
    candidates.push(component_safe_layout(
        logical_qubits,
        logical_components,
        physical_components,
        target,
        &deterministic_assignment,
        false,
        None,
    )?);
    candidates.push(component_safe_layout(
        logical_qubits,
        logical_components,
        physical_components,
        target,
        &deterministic_assignment,
        true,
        None,
    )?);

    if let Ok(greedy) = greedy_layout_prepared(analysis, physical, objective) {
        let greedy = normalize_initial_layout_for_target(logical_qubits, target, &greedy.layout)?;
        if layout_components_are_reachable(logical_components, physical, &greedy)? {
            candidates.push(greedy);
        }
    }
    if let Ok(vf2) =
        vf2_perfect_layout_prepared(analysis, physical, objective, &Vf2LayoutConfig::default())
    {
        let vf2 = normalize_initial_layout_for_target(logical_qubits, target, &vf2.layout)?;
        if layout_components_are_reachable(logical_components, physical, &vf2)? {
            candidates.push(vf2);
        }
    }

    for _ in 0..layout_trials {
        let assignment = component_assignment(logical_components, physical_components, Some(rng))
            .expect("a deterministic feasible component assignment was already found");
        candidates.push(component_safe_layout(
            logical_qubits,
            logical_components,
            physical_components,
            target,
            &assignment,
            false,
            Some(rng),
        )?);
    }

    deduplicate_layouts(candidates, logical_qubits)
}

fn component_safe_layout(
    logical_qubits: &[LogicalQubit],
    logical_components: &[Vec<LogicalQubit>],
    physical_components: &[Vec<PhysicalQubit>],
    target: &RoutingTarget,
    assignment: &[usize],
    reverse: bool,
    mut rng: Option<&mut StdRng>,
) -> Result<Layout, CompilerError> {
    let mut mapping = BTreeMap::new();
    let mut used = BTreeSet::new();
    for (physical_index, physical_component) in physical_components.iter().enumerate() {
        let mut logical = logical_components
            .iter()
            .enumerate()
            .filter(|(index, _)| assignment[*index] == physical_index)
            .flat_map(|(_, component)| component.iter().copied())
            .collect::<Vec<_>>();
        let mut physical = physical_component.clone();
        if let Some(rng) = rng.as_deref_mut() {
            logical.shuffle(rng);
            physical.shuffle(rng);
        } else if reverse {
            physical.reverse();
        }
        for (logical, physical) in logical.into_iter().zip(physical) {
            mapping.insert(logical, physical);
            used.insert(physical);
        }
    }

    let mut idle = logical_qubits
        .iter()
        .copied()
        .filter(|logical| !mapping.contains_key(logical))
        .collect::<Vec<_>>();
    let mut remaining = target
        .physical_qubits
        .iter()
        .copied()
        .filter(|physical| !used.contains(physical))
        .collect::<Vec<_>>();
    if let Some(rng) = rng {
        idle.shuffle(rng);
        remaining.shuffle(rng);
    } else if reverse {
        remaining.reverse();
    }
    mapping.extend(idle.into_iter().zip(remaining));

    Layout::new(
        logical_qubits.to_vec(),
        target.physical_qubits.clone(),
        Some(mapping),
    )
    .map_err(|error| {
        CompilerError::InvariantViolation(format!(
            "sabre layout failed to construct an initial candidate: {error}"
        ))
    })
}

fn logical_interaction_components(analysis: &CircuitLayoutAnalysis) -> Vec<Vec<LogicalQubit>> {
    let mut adjacency: BTreeMap<LogicalQubit, BTreeSet<LogicalQubit>> = BTreeMap::new();
    for interaction in analysis
        .interactions
        .interactions()
        .iter()
        .filter(|interaction| interaction.weight > 0.0)
    {
        adjacency
            .entry(interaction.left)
            .or_default()
            .insert(interaction.right);
        adjacency
            .entry(interaction.right)
            .or_default()
            .insert(interaction.left);
    }

    let mut unseen = adjacency.keys().copied().collect::<BTreeSet<_>>();
    let mut components = Vec::new();
    while let Some(start) = unseen.pop_first() {
        let mut queue = VecDeque::from([start]);
        let mut component = Vec::new();
        while let Some(logical) = queue.pop_front() {
            component.push(logical);
            if let Some(neighbors) = adjacency.get(&logical) {
                for &neighbor in neighbors {
                    if unseen.remove(&neighbor) {
                        queue.push_back(neighbor);
                    }
                }
            }
        }
        component.sort_unstable();
        components.push(component);
    }
    components.sort_unstable_by_key(|component| component[0]);
    components
}

fn component_assignment(
    logical_components: &[Vec<LogicalQubit>],
    physical_components: &[Vec<PhysicalQubit>],
    mut rng: Option<&mut StdRng>,
) -> Option<Vec<usize>> {
    let mut items = logical_components
        .iter()
        .enumerate()
        .map(|(index, component)| (index, component.len(), component[0]))
        .collect::<Vec<_>>();
    items.sort_unstable_by_key(|(_, size, first)| (std::cmp::Reverse(*size), *first));
    if let Some(rng) = rng.as_deref_mut() {
        let mut start = 0;
        while start < items.len() {
            let size = items[start].1;
            let end = items[start..]
                .iter()
                .position(|item| item.1 != size)
                .map_or(items.len(), |offset| start + offset);
            items[start..end].shuffle(rng);
            start = end;
        }
    }

    let mut remaining = physical_components.iter().map(Vec::len).collect::<Vec<_>>();
    let mut assignment = vec![usize::MAX; logical_components.len()];
    let mut failed = HashSet::new();
    if assign_components_recursive(
        0,
        &items,
        &mut remaining,
        &mut assignment,
        &mut failed,
        &mut rng,
    ) {
        Some(assignment)
    } else {
        None
    }
}

fn assign_components_recursive(
    next: usize,
    items: &[(usize, usize, LogicalQubit)],
    remaining: &mut [usize],
    assignment: &mut [usize],
    failed: &mut HashSet<(usize, Vec<usize>)>,
    rng: &mut Option<&mut StdRng>,
) -> bool {
    if next == items.len() {
        return true;
    }
    let mut canonical_remaining = remaining.to_vec();
    canonical_remaining.sort_unstable();
    let key = (next, canonical_remaining);
    if failed.contains(&key) {
        return false;
    }

    let (logical_index, size, _) = items[next];
    let mut bins = (0..remaining.len())
        .filter(|&index| remaining[index] >= size)
        .collect::<Vec<_>>();
    if let Some(rng) = rng.as_deref_mut() {
        bins.shuffle(rng);
    } else {
        bins.sort_unstable_by_key(|&index| (remaining[index] - size, index));
    }
    let mut tried_capacities = BTreeSet::new();
    for physical_index in bins {
        if !tried_capacities.insert(remaining[physical_index]) {
            continue;
        }
        remaining[physical_index] -= size;
        assignment[logical_index] = physical_index;
        if assign_components_recursive(next + 1, items, remaining, assignment, failed, rng) {
            return true;
        }
        remaining[physical_index] += size;
        assignment[logical_index] = usize::MAX;
    }
    failed.insert(key);
    false
}

fn component_capacity_error(
    logical_components: &[Vec<LogicalQubit>],
    physical_components: &[Vec<PhysicalQubit>],
) -> CompilerError {
    let mut logical_sizes = logical_components.iter().map(Vec::len).collect::<Vec<_>>();
    logical_sizes.sort_unstable_by(|left, right| right.cmp(left));
    let mut physical_capacities = physical_components.iter().map(Vec::len).collect::<Vec<_>>();
    physical_capacities.sort_unstable_by(|left, right| right.cmp(left));
    CompilerError::InvalidInput(format!(
        "sabre layout cannot place logical interaction components {logical_sizes:?} into usable physical component capacities {physical_capacities:?}"
    ))
}

fn layout_components_are_reachable(
    logical_components: &[Vec<LogicalQubit>],
    physical: &PhysicalLayoutGraph,
    layout: &Layout,
) -> Result<bool, CompilerError> {
    for component in logical_components {
        let root = layout.get_physical(component[0]).ok_or_else(|| {
            CompilerError::InvariantViolation(format!(
                "sabre layout candidate does not map logical qubit {}",
                component[0]
            ))
        })?;
        for &logical in &component[1..] {
            let mapped = layout.get_physical(logical).ok_or_else(|| {
                CompilerError::InvariantViolation(format!(
                    "sabre layout candidate does not map logical qubit {logical}"
                ))
            })?;
            if physical.distance(root, mapped).is_none() {
                return Ok(false);
            }
        }
    }
    Ok(true)
}

/// Removes duplicate candidate layouts while preserving first occurrence.
///
/// Equality is based on physical-qubit IDs in logical-qubit order, not on
/// object identity or candidate source.
fn deduplicate_layouts(
    candidates: Vec<Layout>,
    logical_qubits: &[LogicalQubit],
) -> Result<Vec<Layout>, CompilerError> {
    let mut seen = BTreeSet::new();
    let mut unique = Vec::new();
    for layout in candidates {
        // Signatures use physical IDs in logical-qubit order so layouts built
        // from different candidate sources deduplicate consistently.
        let signature = logical_qubits
            .iter()
            .map(|logical| {
                layout
                    .get_physical(*logical)
                    .map(PhysicalQubit::id)
                    .ok_or_else(|| {
                        CompilerError::InvariantViolation(format!(
                            "sabre layout candidate does not map logical qubit {logical}"
                        ))
                    })
            })
            .collect::<Result<Vec<_>, CompilerError>>()?;
        if seen.insert(signature) {
            unique.push(layout);
        }
    }
    Ok(unique)
}

/// Returns the best observed route quality for one refined initial layout.
///
/// The candidate is first checked for reachable interactions. Then several
/// seeded unchecked route trials are run and ranked with the same SABRE trial
/// objective used by the routing core.
fn best_route_quality(
    sabre: &SabreDag,
    target: &RoutingTarget,
    initial_layout: &Layout,
    config: &SabreConfig,
    seed: u64,
) -> Result<TrialQuality, CompilerError> {
    let mut best: Option<(usize, TrialQuality)> = None;
    for (index, seed) in trial_seeds(Some(seed), config.layout_scoring_trials)
        .into_iter()
        .enumerate()
    {
        let quality =
            route_trial_unchecked(sabre, target, initial_layout, &config.heuristic, seed)?.quality;
        if best.as_ref().is_none_or(|(best_index, best_quality)| {
            compare_trial_quality(
                config.trial_objective,
                quality,
                index,
                *best_quality,
                *best_index,
            )
            .is_lt()
        }) {
            best = Some((index, quality));
        }
    }
    Ok(best
        .expect("layout_scoring_trials is validated to be non-zero")
        .1)
}
