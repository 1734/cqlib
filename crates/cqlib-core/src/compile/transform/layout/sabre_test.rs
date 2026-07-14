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

use super::*;
use crate::circuit::{Circuit, ClassicalExpr, Instruction, Qubit, StandardGate};
use crate::compile::CompilerError;
use crate::compile::sabre::SabreConfig;
use crate::compile::transform::route_sabre;
use crate::device::{Device, EdgeProp, InstructionProp, PhysicalQubit, Topology};
use std::collections::HashSet;

fn disconnected_device(name: &str, component_ids: &[&[u32]]) -> Device {
    let qubits = component_ids
        .iter()
        .flat_map(|component| component.iter().copied())
        .map(PhysicalQubit::new)
        .collect::<Vec<_>>();
    let edges = component_ids
        .iter()
        .flat_map(|component| component.windows(2))
        .map(|pair| {
            (
                PhysicalQubit::new(pair[0]),
                PhysicalQubit::new(pair[1]),
                "cx".to_string(),
            )
        })
        .collect::<Vec<_>>();
    let topology = Topology::new(qubits.clone(), edges).unwrap();
    Device::new(name, qubits.into_iter().collect(), topology).unwrap()
}

#[test]
fn sabre_layout_is_reproducible_for_same_seed() {
    let device = Device::line("line", 4).unwrap();
    let objective = LayoutObjective::topology_only();
    let config = SabreConfig::deterministic_seeded(7);
    let mut circuit = Circuit::new(3);
    circuit.cx(Qubit::new(0), Qubit::new(2)).unwrap();
    circuit.cx(Qubit::new(1), Qubit::new(2)).unwrap();

    let first = sabre_layout(&circuit, &device, &objective, &config).unwrap();
    let second = sabre_layout(&circuit, &device, &objective, &config).unwrap();

    assert_eq!(first.layout.l2p_map(), second.layout.l2p_map());
    assert_eq!(
        first.score.as_ref().map(|score| score.total),
        second.score.as_ref().map(|score| score.total)
    );
}

#[test]
fn sabre_layout_prepared_matches_top_level_entry() {
    let device = Device::line("line", 4).unwrap();
    let objective = LayoutObjective::topology_only();
    let config = SabreConfig::deterministic_seeded(7);
    let mut circuit = Circuit::new(3);
    circuit.cx(Qubit::new(0), Qubit::new(2)).unwrap();
    circuit.cx(Qubit::new(1), Qubit::new(2)).unwrap();

    let prepared_circuit = prepare_sabre_circuit(&circuit).unwrap();
    let physical = build_physical_layout_graph(&device).unwrap();
    let top_level = sabre_layout(&circuit, &device, &objective, &config).unwrap();
    let prepared =
        sabre_layout_prepared(&prepared_circuit, &physical, &objective, &config).unwrap();

    assert_eq!(top_level.layout.l2p_map(), prepared.layout.l2p_map());
    assert_eq!(
        top_level.score.as_ref().map(|score| score.total),
        prepared.score.as_ref().map(|score| score.total)
    );
}

#[test]
fn prepared_sabre_circuit_can_be_reused_across_targets_and_configs() {
    let objective = LayoutObjective::topology_only();
    let mut circuit = Circuit::new(3);
    circuit.cx(Qubit::new(0), Qubit::new(2)).unwrap();
    let prepared = prepare_sabre_circuit(&circuit).unwrap();

    let line3 = build_physical_layout_graph(&Device::line("line-3", 3).unwrap()).unwrap();
    let line4 = build_physical_layout_graph(&Device::line("line-4", 4).unwrap()).unwrap();
    let first = sabre_layout_prepared(
        &prepared,
        &line3,
        &objective,
        &SabreConfig::deterministic_seeded(3),
    )
    .unwrap();
    let second = sabre_layout_prepared(
        &prepared,
        &line4,
        &objective,
        &SabreConfig::deterministic_seeded(11),
    )
    .unwrap();

    assert_eq!(prepared.logical_qubits().len(), 3);
    assert_eq!(first.layout.logical_qubits().count(), 3);
    assert_eq!(second.layout.logical_qubits().count(), 3);
}

#[test]
fn disconnected_interleaved_components_succeed_for_every_seed() {
    // Two physical stars whose ids are deliberately interleaved. A global
    // trivial/reverse assignment splits every logical component across them.
    let physical = (0..8).map(PhysicalQubit::new).collect::<Vec<_>>();
    let edges = [(0, 2), (0, 4), (0, 6), (1, 3), (1, 5), (1, 7)]
        .into_iter()
        .map(|(left, right)| {
            (
                PhysicalQubit::new(left),
                PhysicalQubit::new(right),
                "cx".to_string(),
            )
        })
        .collect();
    let topology = Topology::new(physical.clone(), edges).unwrap();
    let device = Device::new("dual-star", physical.into_iter().collect(), topology).unwrap();
    let objective = LayoutObjective::topology_only();
    let mut circuit = Circuit::new(8);
    for [left, right] in [
        [0, 1],
        [1, 2],
        [2, 3],
        [3, 0],
        [4, 5],
        [5, 6],
        [6, 7],
        [7, 4],
    ] {
        circuit.cx(Qubit::new(left), Qubit::new(right)).unwrap();
    }

    for seed in [0, 1, 2, 7, 99] {
        sabre_layout(
            &circuit,
            &device,
            &objective,
            &SabreConfig::deterministic_seeded(seed),
        )
        .unwrap();
    }
}

#[test]
fn exact_component_packing_handles_four_three_three_into_six_four() {
    let device = disconnected_device("six-four", &[&[0, 1, 2, 3, 4, 5], &[6, 7, 8, 9]]);
    let mut circuit = Circuit::new(10);
    for [left, right] in [[0, 1], [1, 2], [2, 3], [4, 5], [5, 6], [7, 8], [8, 9]] {
        circuit.cx(Qubit::new(left), Qubit::new(right)).unwrap();
    }

    let result = sabre_layout(
        &circuit,
        &device,
        &LayoutObjective::topology_only(),
        &SabreConfig::deterministic_seeded(7),
    );

    assert!(result.is_ok());
}

#[test]
fn exact_component_packing_backtracks_past_best_fit_dead_end() {
    // Best-fit decreasing first places size 3 into capacity 4, after which the
    // four size-2 components cannot fit. The exact solution is 3+2+2 in the
    // capacity-7 component and 2+2 in the capacity-4 component.
    let device = disconnected_device("seven-four", &[&[0, 1, 2, 3, 4, 5, 6], &[7, 8, 9, 10]]);
    let mut circuit = Circuit::new(11);
    for [left, right] in [[0, 1], [1, 2], [3, 4], [5, 6], [7, 8], [9, 10]] {
        circuit.cx(Qubit::new(left), Qubit::new(right)).unwrap();
    }

    let result = sabre_layout(
        &circuit,
        &device,
        &LayoutObjective::topology_only(),
        &SabreConfig::deterministic_seeded(7),
    );

    assert!(result.is_ok());
}

#[test]
fn infeasible_component_packing_reports_stable_sizes_and_capacities() {
    let device = disconnected_device("six-four", &[&[0, 1, 2, 3, 4, 5], &[6, 7, 8, 9]]);
    let mut circuit = Circuit::new(10);
    for [left, right] in [
        [0, 1],
        [1, 2],
        [2, 3],
        [3, 4],
        [5, 6],
        [6, 7],
        [7, 8],
        [8, 9],
    ] {
        circuit.cx(Qubit::new(left), Qubit::new(right)).unwrap();
    }

    let error = sabre_layout(
        &circuit,
        &device,
        &LayoutObjective::topology_only(),
        &SabreConfig::deterministic_seeded(7),
    )
    .unwrap_err();

    assert!(matches!(
        error,
        CompilerError::InvalidInput(message)
            if message.contains("logical interaction components [5, 5]")
                && message.contains("physical component capacities [6, 4]")
    ));
}

#[test]
fn control_flow_body_only_interactions_participate_in_component_packing() {
    let device = disconnected_device("three-two", &[&[0, 1, 2], &[3, 4]]);
    let mut circuit = Circuit::new(5);
    circuit
        .if_(ClassicalExpr::bool_literal(true), |body| {
            body.cx(Qubit::new(0), Qubit::new(1))?;
            body.cx(Qubit::new(1), Qubit::new(2))?;
            Ok(())
        })
        .unwrap();
    circuit.cx(Qubit::new(3), Qubit::new(4)).unwrap();

    sabre_layout(
        &circuit,
        &device,
        &LayoutObjective::topology_only(),
        &SabreConfig::deterministic_seeded(7),
    )
    .unwrap();
}

#[test]
fn disconnected_target_control_flow_routing_restores_body_layout() {
    let device = disconnected_device("three-two", &[&[0, 1, 2], &[3, 4]]);
    let mut circuit = Circuit::new(5);
    circuit
        .if_(ClassicalExpr::bool_literal(true), |body| {
            body.cx(Qubit::new(0), Qubit::new(1))?;
            body.cx(Qubit::new(1), Qubit::new(2))?;
            body.cx(Qubit::new(0), Qubit::new(2))?;
            Ok(())
        })
        .unwrap();
    circuit.cx(Qubit::new(3), Qubit::new(4)).unwrap();

    let result = route_sabre(
        &circuit,
        &device,
        &LayoutObjective::topology_only(),
        &SabreConfig::deterministic_seeded(7),
    )
    .unwrap();

    assert_eq!(result.diagnostics().control_flow_blocks_routed, 1);
}

#[test]
fn sabre_layout_returns_perfect_layout_when_candidate_can_match_interactions() {
    let device = Device::line("line", 3).unwrap();
    let objective = LayoutObjective::topology_only();
    let config = SabreConfig::deterministic_seeded(7);
    let mut circuit = Circuit::new(3);
    circuit.cx(Qubit::new(0), Qubit::new(2)).unwrap();

    let result = sabre_layout(&circuit, &device, &objective, &config).unwrap();

    assert!(result.diagnostics.is_perfect);
    assert_eq!(result.score.unwrap().distance, 1.0);
}

#[test]
fn sabre_layout_rejects_zero_layout_trials() {
    let device = Device::line("line", 2).unwrap();
    let objective = LayoutObjective::topology_only();
    let mut config = SabreConfig::deterministic_seeded(7);
    config.layout_trials = 0;
    let mut circuit = Circuit::new(2);
    circuit.cx(Qubit::new(0), Qubit::new(1)).unwrap();

    let error = sabre_layout(&circuit, &device, &objective, &config).unwrap_err();

    assert!(
        matches!(error, CompilerError::InvalidInput(message) if message.contains("layout_trials"))
    );
}

#[test]
fn sabre_layout_rejects_zero_layout_scoring_trials() {
    let device = Device::line("line", 2).unwrap();
    let objective = LayoutObjective::topology_only();
    let mut config = SabreConfig::deterministic_seeded(7);
    config.layout_scoring_trials = 0;
    let mut circuit = Circuit::new(2);
    circuit.cx(Qubit::new(0), Qubit::new(1)).unwrap();

    let error = sabre_layout(&circuit, &device, &objective, &config).unwrap_err();

    assert!(
        matches!(error, CompilerError::InvalidInput(message) if message.contains("layout_scoring_trials"))
    );
}

#[test]
fn sabre_layout_rejects_insufficient_physical_qubits() {
    let p0 = PhysicalQubit::new(0);
    let topology = Topology::new(vec![p0], vec![]).unwrap();
    let device = Device::new("one", HashSet::from_iter([p0]), topology).unwrap();
    let objective = LayoutObjective::topology_only();
    let config = SabreConfig::deterministic_seeded(7);
    let circuit = Circuit::new(2);

    let error = sabre_layout(&circuit, &device, &objective, &config).unwrap_err();

    assert!(
        matches!(error, CompilerError::InvalidInput(message) if message.contains("2 logical qubits") && message.contains("1 usable physical qubits"))
    );
}

#[test]
fn sabre_layout_scores_control_flow_body_interactions() {
    let device = Device::line("line", 3).unwrap();
    let objective = LayoutObjective::topology_only();
    let config = SabreConfig::deterministic_seeded(7);
    let mut circuit = Circuit::new(3);
    circuit
        .if_(ClassicalExpr::bool_literal(true), |body| {
            body.cx(Qubit::new(0), Qubit::new(2))?;
            Ok(())
        })
        .unwrap();

    let result = sabre_layout(&circuit, &device, &objective, &config).unwrap();

    assert!(result.diagnostics.is_perfect);
    assert_eq!(result.score.unwrap().distance, 1.0);
}

#[test]
fn sabre_layout_reports_topology_only_scoring() {
    let device = Device::line("line", 2).unwrap();
    let objective = LayoutObjective::topology_only();
    let config = SabreConfig::deterministic_seeded(7);
    let mut circuit = Circuit::new(2);
    circuit.cx(Qubit::new(0), Qubit::new(1)).unwrap();

    let result = sabre_layout(&circuit, &device, &objective, &config).unwrap();

    assert!(!result.diagnostics.used_fidelity);
    assert!(!result.score.unwrap().used_fidelity);
}

#[test]
fn sabre_layout_reports_fidelity_scoring() {
    let p0 = PhysicalQubit::new(0);
    let p1 = PhysicalQubit::new(1);
    let p2 = PhysicalQubit::new(2);
    let topology = Topology::new(
        vec![p0, p1, p2],
        vec![(p0, p1, "cx".to_string()), (p1, p2, "cx".to_string())],
    )
    .unwrap();
    let mut device = Device::new("line", HashSet::from_iter([p0, p1, p2]), topology).unwrap();
    device
        .add_edge_properties(
            p0,
            p1,
            EdgeProp::new()
                .with_native_instruction(InstructionProp::new(
                    Instruction::Standard(StandardGate::CX),
                    0.08,
                ))
                .unwrap(),
        )
        .unwrap();
    device
        .add_edge_properties(
            p1,
            p2,
            EdgeProp::new()
                .with_native_instruction(InstructionProp::new(
                    Instruction::Standard(StandardGate::CX),
                    0.02,
                ))
                .unwrap(),
        )
        .unwrap();
    let physical = build_physical_layout_graph(&device).unwrap();
    let objective = LayoutObjective::auto_from_physical(&physical);
    let config = SabreConfig::deterministic_seeded(7);
    let mut circuit = Circuit::new(2);
    circuit.cx(Qubit::new(0), Qubit::new(1)).unwrap();

    let result = sabre_layout(&circuit, &device, &objective, &config).unwrap();

    assert!(result.diagnostics.used_fidelity);
    assert!(result.score.unwrap().used_fidelity);
}
