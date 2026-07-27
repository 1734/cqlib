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
    CircuitLayoutAnalysis, GreedyCandidateOutcome, LayoutDiagnostics, LayoutObjective,
    LayoutResult, LayoutScore, PhysicalLayoutGraph, Vf2EdgeRequirement, Vf2LayoutConfig,
    Vf2PreparedOutcome, analyze_circuit_for_layout, greedy_layout_candidate_prepared,
    is_perfect_layout, try_vf2_perfect_layout_prepared,
};
use crate::circuit::Circuit;
use crate::compile::sabre::{
    ComponentAssignmentSearch, InteractionReachability, PreparedRouteMetadata,
    RequirementReachabilityFailure, RoutingTarget, SabreConfig, SabreDag, TrialQuality,
    interaction_reachability_for_target, movement_component_assignment,
    normalize_initial_layout_for_target, route_unscored_trial_with_metadata,
    trial_heuristic_profile, trial_seeds, validate_native_trial_operations,
};
use crate::compile::{CompilerError, SabreRoutingFailure};
use crate::device::{Device, Layout, LogicalQubit, PhysicalQubit};
use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use rand::{Rng, SeedableRng};
use rayon::prelude::*;
use std::collections::{BTreeMap, BTreeSet};

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
    backward_refinement_dag: SabreDag,
}

/// Device-side SABRE data prepared for one circuit's interaction signatures.
///
/// The physical graph remains the layout/scoring view. The routing target owns
/// exact native-plan summaries, SWAP-feasible connectivity, and terminal costs.
/// Route metadata for the original and bidirectional refinement DAGs is stored
/// alongside it so layout scoring and final routing reuse the same preparation.
#[derive(Debug, Clone)]
pub struct PreparedSabreDeviceTarget {
    physical: PhysicalLayoutGraph,
    routing: RoutingTarget,
    routing_metadata: PreparedRouteMetadata,
    refinement_metadata: PreparedRouteMetadata,
    backward_refinement_metadata: PreparedRouteMetadata,
}

impl PreparedSabreDeviceTarget {
    /// Returns the physical graph used by layout objective scoring.
    pub fn physical(&self) -> &PhysicalLayoutGraph {
        &self.physical
    }

    pub(crate) fn routing_target(&self) -> &RoutingTarget {
        &self.routing
    }

    pub(crate) fn routing_metadata(&self) -> &PreparedRouteMetadata {
        &self.routing_metadata
    }
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

    pub(crate) fn routing_dag(&self) -> &SabreDag {
        &self.routing_dag
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
    let backward_refinement_dag = refinement_dag.reverse_interactions();
    Ok(PreparedSabreCircuit {
        analysis,
        routing_dag,
        refinement_dag,
        backward_refinement_dag,
    })
}

/// Prepares exact device-native SABRE data for a prepared circuit.
pub fn prepare_sabre_device_target(
    prepared: &PreparedSabreCircuit,
    device: &Device,
) -> Result<PreparedSabreDeviceTarget, CompilerError> {
    let physical = PhysicalLayoutGraph::from_device(device)?;
    let routing = RoutingTarget::from_device(device, &physical, &prepared.routing_dag)?;
    let routing_metadata = PreparedRouteMetadata::new(&prepared.routing_dag, &routing)?;
    let refinement_metadata = PreparedRouteMetadata::new(&prepared.refinement_dag, &routing)?;
    let backward_refinement_metadata =
        PreparedRouteMetadata::new(&prepared.backward_refinement_dag, &routing)?;
    Ok(PreparedSabreDeviceTarget {
        physical,
        routing,
        routing_metadata,
        refinement_metadata,
        backward_refinement_metadata,
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
    let target = prepare_sabre_device_target(&prepared, device)?;
    sabre_layout_prepared(&prepared, &target, objective, config)
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
/// use cqlib_core::compile::transform::{
///     LayoutObjective, prepare_sabre_circuit, prepare_sabre_device_target,
///     sabre_layout_prepared,
/// };
/// use cqlib_core::device::Device;
///
/// let mut circuit = Circuit::new(3);
/// circuit.cx(Qubit::new(0), Qubit::new(2))?;
/// let prepared = prepare_sabre_circuit(&circuit)?;
/// let target = prepare_sabre_device_target(&prepared, &Device::line("line-3", 3)?)?;
///
/// let result = sabre_layout_prepared(
///     &prepared,
///     &target,
///     &LayoutObjective::topology_only(),
///     &SabreConfig::deterministic_seeded(42),
/// )?;
/// assert_eq!(result.layout.logical_qubits().count(), 3);
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
pub fn sabre_layout_prepared(
    prepared: &PreparedSabreCircuit,
    prepared_target: &PreparedSabreDeviceTarget,
    objective: &LayoutObjective,
    config: &SabreConfig,
) -> Result<LayoutResult, CompilerError> {
    validate_layout_config(config)?;
    let physical = &prepared_target.physical;
    let target = &prepared_target.routing;
    let analysis = &prepared.analysis;
    let sabre = &prepared.routing_dag;
    let forwards = &prepared.refinement_dag;
    let backwards = &prepared.backward_refinement_dag;
    let logical_qubits = analysis.logical_qubits.clone();

    if logical_qubits.len() > target.physical_qubits.len() {
        return Err(CompilerError::InvalidInput(format!(
            "sabre layout requires at least as many usable physical qubits as logical qubits; got {} logical qubits and {} usable physical qubits",
            logical_qubits.len(),
            target.physical_qubits.len()
        )));
    }

    let mut rng = StdRng::seed_from_u64(config.seed.unwrap_or_else(rand::random));
    let initial_candidates =
        initial_layout_candidates(prepared, prepared_target, objective, config, &mut rng)?;
    let candidates = initial_candidates.layouts;
    let candidate_notes = initial_candidates.notes;
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

    let outcomes = trials
        .into_par_iter()
        .map(|trial| {
            match interaction_reachability_for_target(sabre, target, &trial.layout)? {
                InteractionReachability::Reachable => {}
                InteractionReachability::UnreachableUnary {
                    cause: RequirementReachabilityFailure::NoExecutableTerminal,
                    ..
                }
                | InteractionReachability::UnreachablePair {
                    cause: RequirementReachabilityFailure::NoExecutableTerminal,
                    ..
                } => {
                    return Ok(CandidateOutcome::Infeasible(
                        CandidateInfeasibleReason::MissingTerminal,
                    ));
                }
                InteractionReachability::UnreachableUnary {
                    cause: RequirementReachabilityFailure::MovementDisconnected,
                    ..
                }
                | InteractionReachability::UnreachablePair {
                    cause: RequirementReachabilityFailure::MovementDisconnected,
                    ..
                } => {
                    return Ok(CandidateOutcome::Infeasible(
                        CandidateInfeasibleReason::MovementUnreachable,
                    ));
                }
            }
            let mut refined = trial.layout;
            for (iteration, (forward_seed, backward_seed)) in
                trial.refinement_seeds.into_iter().enumerate()
            {
                // One refinement iteration routes forward, keeps the final
                // layout, then routes the reversed interaction DAG. This is
                // the SABRE layout-refinement loop, not final circuit routing.
                refined = match route_unscored_trial_with_metadata(
                    forwards,
                    target,
                    &prepared_target.refinement_metadata,
                    &refined,
                    &trial_heuristic_profile(&config.heuristic, iteration * 2),
                    forward_seed,
                ) {
                    Ok(result) => result.final_layout,
                    Err(error) => return classify_candidate_error(error),
                };

                refined = match route_unscored_trial_with_metadata(
                    backwards,
                    target,
                    &prepared_target.backward_refinement_metadata,
                    &refined,
                    &trial_heuristic_profile(&config.heuristic, iteration * 2 + 1),
                    backward_seed,
                ) {
                    Ok(result) => result.final_layout,
                    Err(error) => return classify_candidate_error(error),
                };
            }

            // Rank refined layouts by how well they route the original DAG.
            // Multiple scoring trials reduce seed sensitivity without exposing
            // final SWAP insertion through this layout API.
            let route_quality = match best_route_quality(
                sabre,
                target,
                &prepared_target.routing_metadata,
                &refined,
                config,
                trial.scoring_seed,
            ) {
                Ok(quality) => quality,
                Err(error) => return classify_candidate_error(error),
            };
            let score = objective.score_layout(analysis, physical, &refined)?;
            Ok(CandidateOutcome::Success(CandidateEvaluation {
                index: trial.index,
                route_quality,
                layout: refined,
                score,
            }))
        })
        .collect::<Result<Vec<_>, CompilerError>>()?;
    let mut evaluations = Vec::new();
    let mut missing_terminal = 0usize;
    let mut movement_unreachable = 0usize;
    let mut unsupported_native = 0usize;
    for outcome in outcomes {
        match outcome {
            CandidateOutcome::Success(evaluation) => evaluations.push(evaluation),
            CandidateOutcome::Infeasible(CandidateInfeasibleReason::MissingTerminal) => {
                missing_terminal = missing_terminal.saturating_add(1);
            }
            CandidateOutcome::Infeasible(CandidateInfeasibleReason::MovementUnreachable) => {
                movement_unreachable = movement_unreachable.saturating_add(1);
            }
            CandidateOutcome::Infeasible(CandidateInfeasibleReason::UnsupportedNative) => {
                unsupported_native = unsupported_native.saturating_add(1);
            }
        }
    }

    let swap_limit = config.trial_objective.swap_limit(
        config.swap_regret_ratio,
        evaluations
            .iter()
            .map(|evaluation| evaluation.route_quality.abstract_quality.swap_count),
    );
    let best = evaluations
        .into_iter()
        .filter(|evaluation| evaluation.route_quality.abstract_quality.swap_count <= swap_limit)
        .min_by(|left, right| {
            config
                .trial_objective
                .compare(left.route_quality, 0, right.route_quality, 0)
                .then_with(|| left.score.total.total_cmp(&right.score.total))
                .then_with(|| left.index.cmp(&right.index))
        })
        .ok_or_else(|| {
            CompilerError::SabreRoutingFailed(SabreRoutingFailure::NoFeasibleLayoutCandidate {
                evaluated: candidates_evaluated,
                missing_terminal,
                movement_unreachable,
                unsupported_native,
            })
        })?;
    let swap_count = best.route_quality.abstract_quality.swap_count;
    let layout = best.layout;
    let score = best.score;
    let is_perfect = is_perfect_layout(analysis, physical, &layout);

    let mut notes = candidate_notes;
    notes.push(format!(
        "selected SABRE refined layout with {swap_count} final-route swaps"
    ));
    Ok(LayoutResult {
        layout,
        score: Some(score.clone()),
        diagnostics: LayoutDiagnostics {
            is_perfect,
            candidates_evaluated,
            used_fidelity: score.used_fidelity,
            notes,
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

enum CandidateOutcome<T> {
    Success(T),
    Infeasible(CandidateInfeasibleReason),
}

enum CandidateInfeasibleReason {
    MissingTerminal,
    MovementUnreachable,
    UnsupportedNative,
}

fn classify_candidate_error<T>(error: CompilerError) -> Result<CandidateOutcome<T>, CompilerError> {
    match error {
        CompilerError::SabreRoutingFailed(
            SabreRoutingFailure::NoExecutableUnaryTerminal { .. }
            | SabreRoutingFailure::NoExecutablePairTerminal { .. },
        ) => Ok(CandidateOutcome::Infeasible(
            CandidateInfeasibleReason::MissingTerminal,
        )),
        CompilerError::SabreRoutingFailed(
            SabreRoutingFailure::UnreachableUnaryPlacement { .. }
            | SabreRoutingFailure::UnreachablePairPlacement { .. },
        ) => Ok(CandidateOutcome::Infeasible(
            CandidateInfeasibleReason::MovementUnreachable,
        )),
        CompilerError::DeviceLoweringFailed(_) => Ok(CandidateOutcome::Infeasible(
            CandidateInfeasibleReason::UnsupportedNative,
        )),
        fatal => Err(fatal),
    }
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
    if config.layout_assignment_budget == 0 {
        return Err(CompilerError::InvalidInput(
            "sabre layout_assignment_budget must be greater than zero".to_string(),
        ));
    }
    if config.layout_scoring_trials == 0 {
        return Err(CompilerError::InvalidInput(
            "sabre layout_scoring_trials must be greater than zero".to_string(),
        ));
    }
    if let Some(vf2) = config.vf2_prepass {
        if vf2.candidate_limit == 0 {
            return Err(CompilerError::InvalidInput(
                "sabre vf2_prepass candidate_limit must be greater than zero".to_string(),
            ));
        }
        if vf2.call_limit == 0 {
            return Err(CompilerError::InvalidInput(
                "sabre vf2_prepass call_limit must be greater than zero".to_string(),
            ));
        }
    }
    config.validate()
}

struct InitialLayoutCandidates {
    layouts: Vec<Layout>,
    notes: Vec<String>,
}

/// Generates the candidate set refined by SABRE layout.
///
/// Candidates include deterministic anchors, opportunistic greedy/VF2 results,
/// and random physical orders. The result is deduplicated in logical-qubit
/// order so duplicate layouts from different sources are evaluated once.
fn initial_layout_candidates(
    prepared: &PreparedSabreCircuit,
    prepared_target: &PreparedSabreDeviceTarget,
    objective: &LayoutObjective,
    config: &SabreConfig,
    rng: &mut StdRng,
) -> Result<InitialLayoutCandidates, CompilerError> {
    let analysis = &prepared.analysis;
    let sabre = &prepared.routing_dag;
    let physical = &prepared_target.physical;
    let target = &prepared_target.routing;
    let mut candidates = Vec::new();
    let mut notes = Vec::new();
    let logical_qubits = &analysis.logical_qubits;
    let assignment = match movement_component_assignment(
        sabre,
        target,
        logical_qubits,
        config.layout_assignment_budget,
    )? {
        ComponentAssignmentSearch::Found(assignment) => assignment,
        ComponentAssignmentSearch::ProvenInfeasible => {
            return Err(CompilerError::SabreRoutingFailed(
                SabreRoutingFailure::MovementAssignmentInfeasible,
            ));
        }
        ComponentAssignmentSearch::BudgetExhausted { expansions } => {
            return Err(CompilerError::SabreRoutingFailed(
                SabreRoutingFailure::MovementAssignmentBudgetExhausted {
                    budget: config.layout_assignment_budget,
                    expansions,
                },
            ));
        }
    };

    // Deterministic anchors guarantee that exact movement reachability does not
    // depend on a random seed, including cross-component terminal pairs.
    candidates.push(movement_component_layout(
        logical_qubits,
        &assignment.components,
        &assignment.logical_components,
        target,
        false,
        None,
    )?);
    candidates.push(movement_component_layout(
        logical_qubits,
        &assignment.components,
        &assignment.logical_components,
        target,
        true,
        None,
    )?);

    match greedy_layout_candidate_prepared(analysis, physical, objective)? {
        GreedyCandidateOutcome::Found(greedy) => {
            let greedy =
                normalize_initial_layout_for_target(logical_qubits, target, &greedy.layout)?;
            if interaction_reachability_for_target(sabre, target, &greedy)?
                == InteractionReachability::Reachable
            {
                candidates.push(greedy);
            }
        }
        GreedyCandidateOutcome::Disconnected { left, right } => notes.push(format!(
            "SABRE greedy prepass candidate mapped an interaction across disconnected physical qubits {left} and {right}"
        )),
    }

    if let Some(vf2_prepass) = config.vf2_prepass {
        let vf2_config = Vf2LayoutConfig {
            candidate_limit: vf2_prepass.candidate_limit,
            call_limit: Some(vf2_prepass.call_limit),
            edge_requirement: Vf2EdgeRequirement::PositiveInteractions,
        };
        match try_vf2_perfect_layout_prepared(analysis, physical, objective, &vf2_config)? {
            Vf2PreparedOutcome::Found(vf2) => {
                let vf2 = normalize_initial_layout_for_target(logical_qubits, target, &vf2.layout)?;
                if interaction_reachability_for_target(sabre, target, &vf2)?
                    == InteractionReachability::Reachable
                {
                    candidates.push(vf2);
                }
            }
            Vf2PreparedOutcome::NoCandidate => {
                notes.push("SABRE VF2 prepass found no perfect layout candidate".to_string());
            }
            Vf2PreparedOutcome::BudgetExhausted => notes.push(format!(
                "SABRE VF2 prepass exhausted its call budget of {}",
                vf2_prepass.call_limit
            )),
        }
    }

    for _ in 0..config.layout_trials {
        candidates.push(movement_component_layout(
            logical_qubits,
            &assignment.components,
            &assignment.logical_components,
            target,
            false,
            Some(rng),
        )?);
    }

    Ok(InitialLayoutCandidates {
        layouts: deduplicate_layouts(candidates, logical_qubits)?,
        notes,
    })
}

fn movement_component_layout(
    logical_qubits: &[LogicalQubit],
    physical_components: &[Vec<PhysicalQubit>],
    logical_components: &[usize],
    target: &RoutingTarget,
    reverse: bool,
    mut rng: Option<&mut StdRng>,
) -> Result<Layout, CompilerError> {
    let mut mapping = BTreeMap::new();
    let mut used = BTreeSet::new();
    for (physical_index, physical_component) in physical_components.iter().enumerate() {
        let mut logical = logical_qubits
            .iter()
            .enumerate()
            .filter(|(index, _)| logical_components[*index] == physical_index)
            .map(|(_, logical)| *logical)
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
    metadata: &PreparedRouteMetadata,
    initial_layout: &Layout,
    config: &SabreConfig,
    seed: u64,
) -> Result<TrialQuality, CompilerError> {
    let unscored = trial_seeds(Some(seed), config.layout_scoring_trials)
        .into_iter()
        .enumerate()
        .map(|(index, seed)| {
            let heuristic = trial_heuristic_profile(&config.heuristic, index);
            route_unscored_trial_with_metadata(
                sabre,
                target,
                metadata,
                initial_layout,
                &heuristic,
                seed,
            )
            .map(|result| (index, result))
        })
        .collect::<Result<Vec<_>, CompilerError>>()?;
    unscored
        .iter()
        .try_for_each(|(_, trial)| validate_native_trial_operations(&trial.operations, target))?;
    let swap_limit = config.trial_objective.swap_limit(
        config.swap_regret_ratio,
        unscored.iter().map(|(_, trial)| trial.swap_count),
    );
    if config.trial_objective
        == crate::compile::sabre::SabreTrialObjective::NativeQualityWithinSwapBudget
    {
        Ok(unscored
            .into_iter()
            .filter(|(_, trial)| trial.swap_count <= swap_limit)
            .map(|(index, trial)| {
                let abstract_quality = trial.abstract_quality();
                trial
                    .finalize(abstract_quality, target)
                    .map(|trial| (index, trial.quality))
            })
            .collect::<Result<Vec<_>, CompilerError>>()?
            .into_iter()
            .min_by(|(left_index, left), (right_index, right)| {
                config
                    .trial_objective
                    .compare(*left, *left_index, *right, *right_index)
            })
            .expect("layout_scoring_trials is validated to be non-zero")
            .1)
    } else {
        let (_, trial, abstract_quality) = unscored
            .into_iter()
            .map(|(index, trial)| {
                let abstract_quality = trial.abstract_quality();
                (index, trial, abstract_quality)
            })
            .min_by(|(left_index, _, left), (right_index, _, right)| {
                config.trial_objective.compare(
                    TrialQuality::from_abstract(*left),
                    *left_index,
                    TrialQuality::from_abstract(*right),
                    *right_index,
                )
            })
            .expect("layout_scoring_trials is validated to be non-zero");
        Ok(trial.finalize(abstract_quality, target)?.quality)
    }
}
