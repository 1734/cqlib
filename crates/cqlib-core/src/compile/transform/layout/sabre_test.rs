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
use crate::device::{Device, EdgeProp, InstructionProp, PhysicalQubit, Topology};
use std::collections::HashSet;

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

    let analysis = analyze_circuit_for_layout(&circuit).unwrap();
    let physical = build_physical_layout_graph(&device).unwrap();
    let top_level = sabre_layout(&circuit, &device, &objective, &config).unwrap();
    let prepared =
        sabre_layout_prepared(&circuit, &analysis, &physical, &objective, &config).unwrap();

    assert_eq!(top_level.layout.l2p_map(), prepared.layout.l2p_map());
    assert_eq!(
        top_level.score.as_ref().map(|score| score.total),
        prepared.score.as_ref().map(|score| score.total)
    );
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
