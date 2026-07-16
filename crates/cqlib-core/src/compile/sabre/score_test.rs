// This code is part of Cqlib.
//
// (C) Copyright China Telecom Quantum Group 2026
//
// This code is licensed under the Apache License, Version 2.0.
// You may obtain a copy of this license in the LICENSE.txt file in
// the root directory of this source tree or at
// http://www.apache.org/licenses/LICENSE-2.0.
//
// Any modifications or derivative works of this code must retain this
// copyright notice, and modified files need to carry a notice indicating
// that they have been altered from the originals.

use super::*;
use crate::circuit::{Circuit, ClassicalExpr, ClassicalType, Qubit};
use rayon::ThreadPoolBuilder;

fn line_distance(
    _interaction: usize,
    placement: RequirementPlacement,
) -> Result<f64, CompilerError> {
    Ok(match placement {
        RequirementPlacement::Unary(physical) => physical as f64,
        RequirementPlacement::Pair([left, right]) => left.abs_diff(right) as f64,
    })
}

#[test]
fn heuristic_keeps_front_sum_and_normalizes_lookahead_by_device_width() {
    let mut front = Layer::new(2, 5);
    front
        .insert(
            NodeIndex::new(0),
            0,
            RequirementPlacement::Pair([0, 4]),
            &line_distance,
        )
        .unwrap();
    front
        .insert(
            NodeIndex::new(1),
            0,
            RequirementPlacement::Pair([1, 3]),
            &line_distance,
        )
        .unwrap();
    let mut lookahead = Layer::new(1, 5);
    lookahead
        .insert(
            NodeIndex::new(0),
            0,
            RequirementPlacement::Pair([0, 2]),
            &line_distance,
        )
        .unwrap();
    let empty = Layer::new(0, 5);
    let heuristic = SabreHeuristicConfig {
        basic_weight: 1.0,
        lookahead_weights: vec![0.5, 20.0],
        ..SabreHeuristicConfig::default()
    };

    let score = heuristic_score_after_swap(
        &front,
        &[lookahead, empty],
        &heuristic,
        [0, 1],
        &line_distance,
        1.2,
        5,
    )
    .unwrap();

    // front sum = 6; lookahead = 0.5 * 1 / 5; additive decay = 0.2.
    assert!((score - 6.3).abs() < 1e-12);
}

#[test]
fn decay_is_additive_instead_of_scaling_the_layer_cost() {
    let mut front = Layer::new(1, 4);
    front
        .insert(
            NodeIndex::new(0),
            0,
            RequirementPlacement::Pair([0, 3]),
            &line_distance,
        )
        .unwrap();
    let heuristic = SabreHeuristicConfig {
        basic_weight: 2.0,
        lookahead_weights: Vec::new(),
        ..SabreHeuristicConfig::default()
    };

    let undecayed =
        heuristic_score_after_swap(&front, &[], &heuristic, [0, 1], &line_distance, 1.0, 4)
            .unwrap();
    let decayed =
        heuristic_score_after_swap(&front, &[], &heuristic, [0, 1], &line_distance, 1.5, 4)
            .unwrap();

    assert_eq!(undecayed, 4.0);
    assert_eq!(decayed - undecayed, 0.5);
}

#[test]
fn zero_width_and_empty_layers_have_a_finite_zero_score() {
    let front = Layer::new(0, 0);
    let lookahead = Layer::new(0, 0);
    let heuristic = SabreHeuristicConfig {
        lookahead_weights: vec![1.0],
        ..SabreHeuristicConfig::default()
    };

    let score = heuristic_score_after_swap(
        &front,
        &[lookahead],
        &heuristic,
        [0, 0],
        &line_distance,
        1.0,
        0,
    )
    .unwrap();

    assert_eq!(score, 0.0);
    assert!(score.is_finite());
}

#[test]
fn pair_state_distance_preserves_terminal_direction_and_disconnection() {
    let neighbors = vec![vec![1], vec![0, 2], vec![1], Vec::new()];
    let terminals = BTreeMap::from([([0, 1], NativePlanCost::default())]);

    let swap_costs = default_swap_costs(4, &[(0, 1), (1, 2)]);
    let distances = pair_route_lower_bounds(&neighbors, &terminals, &swap_costs);

    assert_eq!(
        distances.get(0, 1).map(|bound| bound.remaining_swaps),
        Some(0)
    );
    assert_eq!(
        distances.get(1, 0).map(|bound| bound.remaining_swaps),
        Some(1)
    );
    assert_eq!(
        distances.get(2, 0).map(|bound| bound.remaining_swaps),
        Some(2)
    );
    assert_eq!(
        distances.get(0, 2).map(|bound| bound.remaining_swaps),
        Some(1)
    );
    assert_eq!(distances.get(3, 0), None);
    assert_eq!(distances.get(0, 3), None);
}

#[test]
fn lazy_pair_state_search_matches_the_eager_table() {
    let neighbors = vec![vec![1], vec![0, 2], vec![1, 3], vec![2]];
    let terminals = BTreeMap::from([([0, 1], NativePlanCost::default())]);
    let swap_costs = default_swap_costs(4, &[(0, 1), (1, 2), (2, 3)]);
    let eager = pair_route_lower_bounds(&neighbors, &terminals, &swap_costs);

    for left in 0..4 {
        for right in 0..4 {
            if left == right {
                continue;
            }
            assert_eq!(
                pair_route_lower_bound_from_state(
                    &neighbors,
                    &terminals,
                    &swap_costs,
                    [left, right],
                ),
                eager.get(left, right),
                "lazy/eager mismatch for ({left}, {right})"
            );
        }
    }
}

#[test]
fn pair_state_table_omits_impossible_diagonal_states() {
    let table = PairStateTable::<RouteLowerBound>::new(100);

    assert_eq!(table.state_count(), 100 * 99);
    assert_eq!(pair_state_index(100, 4, 4), None);
}

#[test]
fn zero_eager_budget_routes_with_exact_lazy_pair_states() {
    let device = Device::line("lazy-pair-line", 4).unwrap();
    let physical = PhysicalLayoutGraph::from_device(&device).unwrap();
    let mut circuit = Circuit::new(2);
    circuit.cx(Qubit::new(0), Qubit::new(1)).unwrap();
    let sabre = SabreDag::from_operations(circuit.operations()).unwrap();
    let target =
        RoutingTarget::from_device_with_pair_state_budget(&device, &physical, &sabre, 0).unwrap();
    let layout = Layout::from_pairs(&[(0, 0), (1, 3)], 4).unwrap();
    let requirement = target
        .interaction_id_for_node(&sabre, sabre.first_layer[0])
        .unwrap();
    let RequirementTable::Pair { terminals, .. } = &target.requirements[requirement] else {
        panic!("CX must use a pair requirement");
    };

    assert!(!terminals.is_empty());
    assert!(!target.neighbors_by_index[0].is_empty());
    assert!(target.swap_costs[0][1].is_some());
    assert!(
        pair_route_lower_bound_from_state(
            &target.neighbors_by_index,
            terminals,
            &target.swap_costs,
            [0, 3]
        )
        .is_some()
    );
    assert!(
        target
            .route_lower_bound_for(requirement, RequirementPlacement::Pair([0, 3]))
            .is_some()
    );

    let routed = route_trial(
        &sabre,
        &target,
        &layout,
        &SabreConfig::deterministic_seeded(7).heuristic,
        11,
    )
    .unwrap();
    let (lookups, cached) = target.lazy_pair_cache_stats();

    assert_eq!(target.eager_pair_state_count, 0);
    assert!(routed.swap_count > 0);
    assert!(lookups > 0);
    assert!(cached > 0);
}

#[test]
fn lazy_pair_cache_does_not_change_seeded_results_across_thread_counts() {
    let device = Device::line("lazy-pair-parallel", 6).unwrap();
    let physical = PhysicalLayoutGraph::from_device(&device).unwrap();
    let layout = Layout::from_pairs(&[(0, 0), (1, 5), (2, 2), (3, 4)], 6).unwrap();
    let mut circuit = Circuit::new(4);
    for (left, right) in [(0, 1), (2, 3), (0, 3), (1, 2), (0, 2), (1, 3)] {
        circuit.cx(Qubit::new(left), Qubit::new(right)).unwrap();
    }
    let sabre = SabreDag::from_operations(circuit.operations()).unwrap();
    let seeds = [3_u64, 5, 7, 11, 13, 17, 19, 23];
    let heuristic = SabreHeuristicConfig {
        lookahead_weights: vec![0.5, 0.25],
        decay_increment: Some(0.01),
        ..SabreHeuristicConfig::default()
    };
    let route_in_pool = |threads| {
        let target =
            RoutingTarget::from_device_with_pair_state_budget(&device, &physical, &sabre, 0)
                .unwrap();
        let results = ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()
            .unwrap()
            .install(|| {
                seeds
                    .into_par_iter()
                    .map(|seed| {
                        let result =
                            route_trial(&sabre, &target, &layout, &heuristic, seed).unwrap();
                        (
                            result.swap_count,
                            result.final_layout,
                            result.quality,
                            format!("{:?}", result.operations),
                        )
                    })
                    .collect::<Vec<_>>()
            });
        (results, target.lazy_pair_cache_stats())
    };

    let single_threaded = route_in_pool(1);
    let four_threaded = route_in_pool(4);

    assert_eq!(single_threaded, four_threaded);
}

#[test]
fn cost_to_go_charges_terminal_cost_for_every_reachable_state() {
    let neighbors = vec![vec![1], vec![0, 2], vec![1]];
    let terminal = NativePlanCost {
        native_two_qubit_ops: 5,
        native_total_ops: 7,
        error: None,
        duration: None,
    };
    let terminals = vec![None, None, Some(terminal)];
    let swap_costs = default_swap_costs(3, &[(0, 1), (1, 2)]);

    let bounds = unary_route_lower_bounds(&neighbors, &terminals, &swap_costs);

    assert_eq!(bounds[2].unwrap().remaining_swaps, 0);
    assert_eq!(bounds[1].unwrap().remaining_swaps, 1);
    assert_eq!(bounds[0].unwrap().remaining_swaps, 2);
    assert!(bounds.iter().flatten().all(|bound| {
        bound.native.native_two_qubit_ops == terminal.native_two_qubit_ops
            && bound.native.native_total_ops == terminal.native_total_ops
    }));
}

#[test]
fn cost_to_go_preserves_enabled_calibration_through_zero_cost_terminal() {
    let neighbors = vec![vec![1], vec![0]];
    let terminal = NativePlanCost {
        error: Some(RobustErrorKey::default()),
        duration: Some(RobustDurationKey::default()),
        ..NativePlanCost::default()
    };
    let swap = NativePlanCost {
        native_two_qubit_ops: 3,
        native_total_ops: 7,
        error: Some(RobustErrorKey {
            unavailable_count: 0,
            imputed_count: 0,
            log_error: 0.125,
        }),
        duration: Some(RobustDurationKey {
            unavailable_count: 0,
            imputed_count: 0,
            duration_work: 40.0,
        }),
    };
    let terminals = vec![None, Some(terminal)];
    let swap_costs = vec![vec![None, Some(swap)], vec![Some(swap), None]];

    let bounds = unary_route_lower_bounds(&neighbors, &terminals, &swap_costs);
    let routed = bounds[0].expect("the calibrated SWAP reaches the terminal");

    assert_eq!(routed.remaining_swaps, 1);
    assert_eq!(routed.native, swap);
}

#[test]
fn native_cost_comparison_prioritizes_gate_count_then_robust_error() {
    let fewer_gates_with_unknown_error = NativePlanCost {
        native_two_qubit_ops: 2,
        native_total_ops: 8,
        error: Some(RobustErrorKey {
            unavailable_count: 1,
            imputed_count: 0,
            log_error: 0.0,
        }),
        duration: None,
    };
    let more_gates_with_known_error = NativePlanCost {
        native_two_qubit_ops: 3,
        native_total_ops: 3,
        error: Some(RobustErrorKey {
            unavailable_count: 0,
            imputed_count: 0,
            log_error: 0.01,
        }),
        duration: None,
    };
    let equal_gates_with_known_error = NativePlanCost {
        native_two_qubit_ops: 2,
        ..more_gates_with_known_error
    };

    assert_eq!(
        compare_optional_native_cost(
            Some(fewer_gates_with_unknown_error),
            Some(more_gates_with_known_error)
        ),
        Ordering::Less
    );
    assert_eq!(
        compare_optional_native_cost(
            Some(equal_gates_with_known_error),
            Some(fewer_gates_with_unknown_error)
        ),
        Ordering::Less
    );
    assert_eq!(
        compare_optional_native_cost(Some(equal_gates_with_known_error), None),
        Ordering::Less
    );
}

#[test]
fn exclusive_control_flow_uses_worst_error_instead_of_branch_sum() {
    let low = RobustErrorKey {
        unavailable_count: 0,
        imputed_count: 0,
        log_error: 0.1,
    };
    let high = RobustErrorKey {
        unavailable_count: 0,
        imputed_count: 0,
        log_error: 0.3,
    };

    assert_eq!(worst_error(Some(low), Some(high)), Some(high));
    assert_ne!(worst_error(Some(low), Some(high)), Some(low.combine(high)));
}

#[test]
fn fixed_for_loop_multiplies_execution_path_two_qubit_depth() {
    let mut circuit = Circuit::new(2);
    let loop_var = circuit.var(ClassicalType::uint(3).unwrap());
    circuit
        .for_uint(
            loop_var,
            ClassicalExpr::uint_literal(3, 1).unwrap(),
            ClassicalExpr::uint_literal(3, 7).unwrap(),
            ClassicalExpr::uint_literal(3, 2).unwrap(),
            |body, _| {
                body.cx(Qubit::new(0), Qubit::new(1))?;
                Ok(())
            },
        )
        .unwrap();

    assert_eq!(two_qubit_depth(circuit.operations()), 3);
}

#[test]
fn constrained_trial_objective_uses_native_quality_in_declared_order() {
    let baseline = TrialQuality {
        swap_count: 4,
        two_qubit_depth: 2,
        operation_count: 8,
        native_two_qubit_ops: 10,
        native_two_qubit_depth: 5,
        native_total_ops: 30,
        error: Some(RobustErrorKey {
            unavailable_count: 0,
            imputed_count: 0,
            log_error: 0.2,
        }),
        duration: Some(RobustDurationKey {
            unavailable_count: 0,
            imputed_count: 0,
            duration_work: 20.0,
        }),
        makespan: Some(20.0),
        unknown_loop_count: 0,
    };
    let fewer_native_gates = TrialQuality {
        native_two_qubit_ops: 9,
        native_two_qubit_depth: 100,
        ..baseline
    };
    let shallower_native_depth = TrialQuality {
        native_two_qubit_depth: 4,
        error: Some(RobustErrorKey {
            log_error: 100.0,
            ..baseline.error.unwrap()
        }),
        ..baseline
    };
    let better_coverage = TrialQuality {
        error: Some(RobustErrorKey {
            unavailable_count: 0,
            imputed_count: 0,
            log_error: 10.0,
        }),
        ..baseline
    };
    let worse_coverage = TrialQuality {
        error: Some(RobustErrorKey {
            unavailable_count: 1,
            imputed_count: 0,
            log_error: 0.0,
        }),
        ..baseline
    };

    assert!(
        compare_trial_quality(
            SabreTrialObjective::NativeQualityWithinSwapBudget,
            fewer_native_gates,
            1,
            baseline,
            0
        )
        .is_lt()
    );
    assert!(
        compare_trial_quality(
            SabreTrialObjective::NativeQualityWithinSwapBudget,
            shallower_native_depth,
            1,
            baseline,
            0
        )
        .is_lt()
    );
    assert!(
        compare_trial_quality(
            SabreTrialObjective::NativeQualityWithinSwapBudget,
            better_coverage,
            1,
            worse_coverage,
            0
        )
        .is_lt()
    );
}

#[test]
fn trial_swap_budget_is_five_percent_with_integer_ceiling() {
    let quality = |swap_count| TrialQuality {
        swap_count,
        ..TrialQuality::default()
    };

    assert_eq!(
        trial_swap_limit(
            SabreTrialObjective::NativeQualityWithinSwapBudget,
            0.05,
            [quality(20), quality(25)].into_iter()
        ),
        21
    );
    assert_eq!(
        trial_swap_limit(
            SabreTrialObjective::NativeQualityWithinSwapBudget,
            0.05,
            [quality(19), quality(25)].into_iter()
        ),
        20
    );
    assert_eq!(
        trial_swap_limit(SabreTrialObjective::Depth, 0.05, [quality(2)].into_iter()),
        usize::MAX
    );
}

#[test]
fn trial_profiles_are_deterministic_and_do_not_mutate_the_base() {
    let base = SabreHeuristicConfig {
        lookahead_weights: vec![0.5, 0.25],
        decay_increment: Some(0.01),
        ..SabreHeuristicConfig::default()
    };

    let profiles = (0..4)
        .map(|index| trial_heuristic_profile(&base, index))
        .collect::<Vec<_>>();

    assert_eq!(profiles[0], base);
    assert_eq!(profiles[1].decay_increment, None);
    assert_eq!(profiles[2].lookahead_weights, vec![0.25, 0.125]);
    assert_eq!(profiles[3].lookahead_weights, vec![0.75, 0.375]);
    assert_eq!(profiles[3].decay_increment, Some(0.02));
    assert_eq!(trial_heuristic_profile(&base, 7), profiles[3]);
    assert_eq!(base.lookahead_weights, vec![0.5, 0.25]);
    assert_eq!(base.decay_increment, Some(0.01));
}

#[test]
fn device_target_prepares_both_orderings_of_emitted_swap() {
    let device = Device::line("ordered-swap", 2)
        .unwrap()
        .with_native_gates(vec![
            Instruction::Standard(StandardGate::H),
            Instruction::Standard(StandardGate::CX),
        ])
        .unwrap();
    let physical = PhysicalLayoutGraph::from_device(&device).unwrap();
    let mut circuit = Circuit::new(2);
    circuit.cx(Qubit::new(0), Qubit::new(1)).unwrap();
    let sabre = SabreDag::from_operations(circuit.operations()).unwrap();
    let target = RoutingTarget::from_device(&device, &physical, &sabre).unwrap();
    let reversed = DeviceGateState::standard(
        StandardGate::SWAP,
        smallvec![PhysicalQubit::new(1), PhysicalQubit::new(0)],
    );

    assert!(target.native_cost(&reversed).is_some());
}
