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
use super::*;
use crate::circuit::{ClassicalExpr, Qubit};
use crate::compile::test_utils::assert_compiled_circuit_equivalent;
use crate::compile::transform::decompose::unitary::TwoQubitSynthesisTarget;

fn cx_config() -> TwoQubitBlockResynthesisConfig {
    config_for_native_2q(StandardGate::CX)
}

fn config_for_native_2q(gate: StandardGate) -> TwoQubitBlockResynthesisConfig {
    TwoQubitBlockResynthesisConfig::normal(
        TwoQubitSynthesisTarget::from_standard_gates(
            vec![
                StandardGate::U,
                StandardGate::H,
                StandardGate::RX,
                StandardGate::RY,
                StandardGate::RZ,
                StandardGate::S,
                StandardGate::SDG,
            ],
            vec![gate],
            true,
        )
        .unwrap(),
    )
}

fn standard_ops(circuit: &Circuit) -> Vec<StandardGate> {
    circuit
        .operations()
        .iter()
        .filter_map(|operation| match operation.instruction {
            Instruction::Standard(gate) => Some(gate),
            _ => None,
        })
        .collect()
}

fn two_qubit_op_count(circuit: &Circuit) -> usize {
    circuit
        .operations()
        .iter()
        .filter(|operation| operation.qubits.len() == 2)
        .count()
}

#[test]
fn cancels_adjacent_cx_pair() {
    let q0 = Qubit::new(0);
    let q1 = Qubit::new(1);
    let mut circuit = Circuit::new(2);
    circuit.cx(q0, q1).unwrap();
    circuit.cx(q0, q1).unwrap();

    let result = resynthesize_two_qubit_blocks(&circuit, cx_config()).unwrap();

    assert!(result.changed);
    assert!(result.circuit.operations().is_empty());
    assert_compiled_circuit_equivalent(&result.circuit, &circuit);
}

#[test]
fn single_two_qubit_gate_is_not_resynthesized_without_improvement() {
    let q0 = Qubit::new(0);
    let q1 = Qubit::new(1);
    let mut circuit = Circuit::new(2);
    circuit.cx(q0, q1).unwrap();

    let result = resynthesize_two_qubit_blocks(&circuit, cx_config()).unwrap();

    assert!(!result.changed);
    assert_eq!(standard_ops(&result.circuit), vec![StandardGate::CX]);
}

#[test]
fn symbolic_operation_in_block_window_is_preserved() {
    let q0 = Qubit::new(0);
    let q1 = Qubit::new(1);
    let mut circuit = Circuit::new(2);
    circuit.rz(q0, Parameter::symbol("theta")).unwrap();
    circuit.cx(q0, q1).unwrap();

    let result = resynthesize_two_qubit_blocks(&circuit, cx_config()).unwrap();

    assert!(!result.changed);
    assert_eq!(
        standard_ops(&result.circuit),
        vec![StandardGate::RZ, StandardGate::CX]
    );
}

#[test]
fn labeled_two_qubit_gates_are_boundaries() {
    let q0 = Qubit::new(0);
    let q1 = Qubit::new(1);
    let mut circuit = Circuit::new(2);
    circuit
        .append(
            Instruction::Standard(StandardGate::CX),
            [q0, q1],
            std::iter::empty::<ParameterValue>(),
            Some("keep-a"),
        )
        .unwrap();
    circuit
        .append(
            Instruction::Standard(StandardGate::CX),
            [q0, q1],
            std::iter::empty::<ParameterValue>(),
            Some("keep-b"),
        )
        .unwrap();

    let result = resynthesize_two_qubit_blocks(&circuit, cx_config()).unwrap();

    assert!(!result.changed);
    assert_eq!(
        standard_ops(&result.circuit),
        vec![StandardGate::CX, StandardGate::CX]
    );
}

#[test]
fn control_flow_without_profitable_body_block_remains_noop() {
    let q0 = Qubit::new(0);
    let q1 = Qubit::new(1);
    let mut circuit = Circuit::new(2);
    circuit
        .if_else(
            ClassicalExpr::bool_literal(true),
            |body| body.cx(q0, q1),
            |_| Ok(()),
        )
        .unwrap();

    let result = resynthesize_two_qubit_blocks(&circuit, cx_config()).unwrap();

    assert!(!result.changed);
    assert_eq!(result.circuit.operations().len(), 1);
    assert!(matches!(
        result.circuit.operations()[0].instruction,
        Instruction::ClassicalControl(_)
    ));
}

#[test]
fn control_flow_body_is_resynthesized_recursively() {
    let q0 = Qubit::new(0);
    let q1 = Qubit::new(1);
    let mut circuit = Circuit::new(2);
    circuit
        .if_else(
            ClassicalExpr::bool_literal(true),
            |body| {
                body.cx(q0, q1)?;
                body.cx(q0, q1)
            },
            |_| Ok(()),
        )
        .unwrap();

    let result = resynthesize_two_qubit_blocks(&circuit, cx_config()).unwrap();

    assert!(result.changed);
    let Instruction::ClassicalControl(ClassicalControlOp::If(if_op)) =
        &result.circuit.operations()[0].instruction
    else {
        panic!("expected if operation");
    };
    assert!(if_op.then_body().operations().is_empty());
}

#[test]
fn recurse_control_flow_false_preserves_profitable_body_block() {
    let q0 = Qubit::new(0);
    let q1 = Qubit::new(1);
    let mut circuit = Circuit::new(2);
    circuit
        .if_else(
            ClassicalExpr::bool_literal(true),
            |body| {
                body.cx(q0, q1)?;
                body.cx(q0, q1)
            },
            |_| Ok(()),
        )
        .unwrap();
    let mut config = cx_config();
    config.recurse_control_flow = false;

    let result = resynthesize_two_qubit_blocks(&circuit, config).unwrap();

    assert!(!result.changed);
}

#[test]
fn barrier_prevents_across_boundary_resynthesis() {
    let q0 = Qubit::new(0);
    let q1 = Qubit::new(1);
    let mut circuit = Circuit::new(2);
    circuit.cx(q0, q1).unwrap();
    circuit.barrier(vec![q0, q1]).unwrap();
    circuit.cx(q0, q1).unwrap();

    let result = resynthesize_two_qubit_blocks(&circuit, cx_config()).unwrap();

    assert!(!result.changed);
    assert_eq!(
        standard_ops(&result.circuit),
        vec![StandardGate::CX, StandardGate::CX]
    );
}

#[test]
fn reset_prevents_across_boundary_resynthesis() {
    let q0 = Qubit::new(0);
    let q1 = Qubit::new(1);
    let mut circuit = Circuit::new(2);
    circuit.cx(q0, q1).unwrap();
    circuit.reset(q0).unwrap();
    circuit.cx(q0, q1).unwrap();

    let result = resynthesize_two_qubit_blocks(&circuit, cx_config()).unwrap();

    assert!(!result.changed);
    assert_eq!(
        standard_ops(&result.circuit),
        vec![StandardGate::CX, StandardGate::CX]
    );
}

#[test]
fn symbolic_two_qubit_gate_is_skipped() {
    let q0 = Qubit::new(0);
    let q1 = Qubit::new(1);
    let mut circuit = Circuit::new(2);
    circuit.rzz(q0, q1, Parameter::symbol("theta")).unwrap();
    circuit.rzz(q0, q1, Parameter::symbol("theta")).unwrap();

    let result = resynthesize_two_qubit_blocks(&circuit, cx_config()).unwrap();

    assert!(!result.changed);
    assert_eq!(
        standard_ops(&result.circuit),
        vec![StandardGate::RZZ, StandardGate::RZZ]
    );
}

#[test]
fn disjoint_crossed_operation_is_preserved_while_block_is_removed() {
    let q0 = Qubit::new(0);
    let q1 = Qubit::new(1);
    let q2 = Qubit::new(2);
    let mut circuit = Circuit::new(3);
    circuit.cx(q0, q1).unwrap();
    circuit.h(q2).unwrap();
    circuit.cx(q0, q1).unwrap();

    let result = resynthesize_two_qubit_blocks(&circuit, cx_config()).unwrap();

    assert!(result.changed);
    assert_eq!(standard_ops(&result.circuit), vec![StandardGate::H]);
    assert_compiled_circuit_equivalent(&result.circuit, &circuit);
}

#[test]
fn swap_pair_is_resynthesized_when_cost_improves() {
    let q0 = Qubit::new(0);
    let q1 = Qubit::new(1);
    let mut circuit = Circuit::new(2);
    circuit.swap(q0, q1).unwrap();
    circuit.swap(q0, q1).unwrap();

    let result = resynthesize_two_qubit_blocks(&circuit, cx_config()).unwrap();

    assert!(result.changed);
    assert!(result.circuit.operations().is_empty());
    assert_compiled_circuit_equivalent(&result.circuit, &circuit);
}

#[test]
fn interleaved_swap_dependency_is_not_moved_by_left_absorption() {
    let q0 = Qubit::new(0);
    let q1 = Qubit::new(1);
    let q2 = Qubit::new(2);
    let mut circuit = Circuit::new(3);
    circuit.h(q0).unwrap();
    circuit.rx(q1, 0.17).unwrap();
    circuit.ry(q2, -0.19).unwrap();
    circuit.swap(q0, q2).unwrap();
    circuit.swap(q1, q2).unwrap();
    let config = TwoQubitBlockResynthesisConfig::normal(
        TwoQubitSynthesisTarget::from_standard_gates(
            vec![StandardGate::H, StandardGate::RX, StandardGate::RY],
            vec![StandardGate::RXX, StandardGate::RYY, StandardGate::RZZ],
            true,
        )
        .unwrap(),
    );

    let result = resynthesize_two_qubit_blocks(&circuit, config).unwrap();

    assert_compiled_circuit_equivalent(&result.circuit, &circuit);
}

#[test]
fn adjacent_inverse_pair_resynthesizes_across_supported_backends() {
    let q0 = Qubit::new(0);
    let q1 = Qubit::new(1);
    let configs = [
        ("cx", config_for_native_2q(StandardGate::CX)),
        ("cz", config_for_native_2q(StandardGate::CZ)),
        ("rzz", config_for_native_2q(StandardGate::RZZ)),
        ("pauli-fallback", TwoQubitBlockResynthesisConfig::default()),
    ];

    for (name, config) in configs {
        let mut circuit = Circuit::new(2);
        circuit.cx(q0, q1).unwrap();
        circuit.cx(q0, q1).unwrap();

        let result = resynthesize_two_qubit_blocks(&circuit, config).unwrap();

        assert!(result.changed, "target {name} should accept identity block");
        assert_eq!(two_qubit_op_count(&result.circuit), 0);
        assert_compiled_circuit_equivalent(&result.circuit, &circuit);
    }
}

#[test]
fn three_cx_block_is_compressed_and_equivalent() {
    let q0 = Qubit::new(0);
    let q1 = Qubit::new(1);
    let mut circuit = Circuit::new(2);
    circuit.cx(q0, q1).unwrap();
    circuit.cx(q0, q1).unwrap();
    circuit.cx(q0, q1).unwrap();

    let result = resynthesize_two_qubit_blocks(&circuit, cx_config()).unwrap();

    assert!(result.changed);
    assert!(two_qubit_op_count(&result.circuit) < 3);
    assert_compiled_circuit_equivalent(&result.circuit, &circuit);
}

#[test]
fn mixed_one_and_two_qubit_block_remains_semantically_equivalent() {
    let q0 = Qubit::new(0);
    let q1 = Qubit::new(1);
    let mut circuit = Circuit::new(2);
    circuit.h(q0).unwrap();
    circuit.cx(q0, q1).unwrap();
    circuit.x(q1).unwrap();
    circuit.cx(q0, q1).unwrap();

    let result = resynthesize_two_qubit_blocks(&circuit, cx_config()).unwrap();

    assert!(result.changed);
    assert_compiled_circuit_equivalent(&result.circuit, &circuit);
}

#[test]
fn numeric_rotation_mixed_block_preserves_semantics() {
    let q0 = Qubit::new(0);
    let q1 = Qubit::new(1);
    let mut circuit = Circuit::new(2);
    circuit.rx(q0, 0.3).unwrap();
    circuit.cx(q0, q1).unwrap();
    circuit.rz(q1, 0.7).unwrap();
    circuit.cz(q0, q1).unwrap();
    circuit.rx(q0, 1.2).unwrap();

    let result = resynthesize_two_qubit_blocks(&circuit, cx_config()).unwrap();

    assert_compiled_circuit_equivalent(&result.circuit, &circuit);
}
