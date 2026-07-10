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

use super::{TargetBasisCostModel, TargetBasisLowerer};
use crate::circuit::{Circuit, Instruction, ParameterValue, Qubit, StandardGate, ValueOperation};
use crate::compile::CompilerError;
use crate::compile::transform::Transformer;
use crate::util::test_utils::{assert_compiled_circuit_equivalent, standard_ops};

fn target_basis(gates: &[StandardGate]) -> Vec<Instruction> {
    gates.iter().copied().map(Instruction::Standard).collect()
}

fn run_target_lowering(circuit: &Circuit, basis: &[StandardGate]) -> Circuit {
    TargetBasisLowerer::new(target_basis(basis))
        .unwrap()
        .transform(circuit, None)
        .unwrap()
        .circuit
}

fn assert_only_target_standard_gates(circuit: &Circuit, basis: &[StandardGate]) {
    for operation in circuit.operations() {
        match operation.instruction {
            Instruction::Standard(gate) => assert!(
                basis.contains(&gate),
                "gate {gate} is outside target basis {basis:?}"
            ),
            ref instruction => panic!("unexpected non-standard instruction {instruction}"),
        }
    }
}

fn assert_target_lowering_fails_with(
    circuit: &Circuit,
    basis: &[StandardGate],
    expected_snippets: &[&str],
) {
    let err = TargetBasisLowerer::new(target_basis(basis))
        .unwrap()
        .transform(circuit, None)
        .unwrap_err();
    let CompilerError::InvalidInput(message) = err else {
        panic!("expected invalid input error, got {err:?}");
    };
    for snippet in expected_snippets {
        assert!(
            message.contains(snippet),
            "expected error message {message:?} to contain {snippet:?}"
        );
    }
}

#[test]
fn swap_lowers_to_rz_x2p_cz_basis() {
    let q0 = Qubit::new(0);
    let q1 = Qubit::new(1);
    let mut circuit = Circuit::new(2);
    circuit.swap(q0, q1).unwrap();
    let basis = [StandardGate::RZ, StandardGate::X2P, StandardGate::CZ];

    let result = run_target_lowering(&circuit, &basis);

    assert_only_target_standard_gates(&result, &basis);
    assert!(!standard_ops(&result).contains(&StandardGate::SWAP));
}

#[test]
fn arbitrary_u_lowers_to_qcis_half_rotation_basis() {
    let q0 = Qubit::new(0);
    let mut circuit = Circuit::new(1);
    circuit
        .append(
            Instruction::Standard(StandardGate::U),
            vec![q0],
            vec![
                ParameterValue::Fixed(0.7),
                ParameterValue::Fixed(-0.4),
                ParameterValue::Fixed(0.9),
            ],
            None,
        )
        .unwrap();
    let basis = [
        StandardGate::RZ,
        StandardGate::X2P,
        StandardGate::X2M,
        StandardGate::Y2P,
        StandardGate::Y2M,
        StandardGate::CZ,
        StandardGate::GPhase,
    ];

    let result = run_target_lowering(&circuit, &basis);

    assert_only_target_standard_gates(&result, &basis);
    assert_compiled_circuit_equivalent(&result, &circuit);
}

#[test]
fn exact_cost_model_matches_the_target_lowerer_output() {
    let q0 = Qubit::new(0);
    let q1 = Qubit::new(1);
    let basis = [StandardGate::RZ, StandardGate::X2P, StandardGate::CZ];
    let operations = vec![
        ValueOperation::from_standard(StandardGate::CX, [q0, q1], []),
        ValueOperation::from_standard(
            StandardGate::U,
            [q1],
            [0.7.into(), (-0.4).into(), 0.9.into()],
        ),
    ];
    let source = Circuit::from_operations(vec![q0, q1], operations.clone(), None, None).unwrap();
    let model = TargetBasisCostModel::new(target_basis(&basis)).unwrap();

    let estimated = model
        .cost_of_fixed_operations(vec![q0, q1], operations)
        .unwrap();
    let lowered = run_target_lowering(&source, &basis);
    let physical_ops = lowered
        .operations()
        .iter()
        .filter(|operation| {
            !matches!(
                operation.instruction,
                Instruction::Standard(StandardGate::GPhase)
            )
        })
        .collect::<Vec<_>>();

    assert_eq!(estimated.total_ops, physical_ops.len());
    assert_eq!(
        estimated.two_qubit_ops,
        physical_ops
            .iter()
            .filter(|operation| operation.qubits.len() == 2)
            .count()
    );
    assert_eq!(
        estimated.parameterized_ops,
        physical_ops
            .iter()
            .filter(|operation| !operation.params.is_empty())
            .count()
    );
    assert_eq!(estimated.depth, lowered.depth(false).unwrap());
}

#[test]
fn swap_lowers_to_original_qcis_bug_basis() {
    let q0 = Qubit::new(0);
    let q1 = Qubit::new(1);
    let mut circuit = Circuit::new(2);
    circuit.swap(q0, q1).unwrap();
    let basis = [
        StandardGate::I,
        StandardGate::X2P,
        StandardGate::X,
        StandardGate::RZ,
        StandardGate::CZ,
    ];

    let result = run_target_lowering(&circuit, &basis);

    assert_only_target_standard_gates(&result, &basis);
    assert!(!standard_ops(&result).contains(&StandardGate::SWAP));
}

#[test]
fn h_lowers_to_rz_x2p_cz_basis() {
    let q0 = Qubit::new(0);
    let mut circuit = Circuit::new(1);
    circuit.h(q0).unwrap();
    let basis = [StandardGate::RZ, StandardGate::X2P, StandardGate::CZ];

    let result = run_target_lowering(&circuit, &basis);

    assert_only_target_standard_gates(&result, &basis);
    assert!(!standard_ops(&result).contains(&StandardGate::H));
}

#[test]
fn cx_lowers_to_rz_x2p_cz_basis() {
    let q0 = Qubit::new(0);
    let q1 = Qubit::new(1);
    let mut circuit = Circuit::new(2);
    circuit.cx(q0, q1).unwrap();
    let basis = [StandardGate::RZ, StandardGate::X2P, StandardGate::CZ];

    let result = run_target_lowering(&circuit, &basis);

    assert_only_target_standard_gates(&result, &basis);
    assert!(!standard_ops(&result).contains(&StandardGate::CX));
}

#[test]
fn ccx_lowers_to_rz_x2p_cz_basis() {
    let q0 = Qubit::new(0);
    let q1 = Qubit::new(1);
    let q2 = Qubit::new(2);
    let mut circuit = Circuit::new(3);
    circuit.ccx(q0, q1, q2).unwrap();
    let basis = [StandardGate::RZ, StandardGate::X2P, StandardGate::CZ];

    let result = run_target_lowering(&circuit, &basis);

    assert_only_target_standard_gates(&result, &basis);
    assert!(!standard_ops(&result).contains(&StandardGate::CCX));
}

#[test]
fn swap_fails_for_rz_x2p_basis_without_entangling_gate() {
    let q0 = Qubit::new(0);
    let q1 = Qubit::new(1);
    let mut circuit = Circuit::new(2);
    circuit.swap(q0, q1).unwrap();

    assert_target_lowering_fails_with(
        &circuit,
        &[StandardGate::RZ, StandardGate::X2P],
        &["cannot lower", "SWAP", "RZ", "X2P"],
    );
}

#[test]
fn swap_fails_for_rz_cz_basis_without_half_rotation() {
    let q0 = Qubit::new(0);
    let q1 = Qubit::new(1);
    let mut circuit = Circuit::new(2);
    circuit.swap(q0, q1).unwrap();

    assert_target_lowering_fails_with(
        &circuit,
        &[StandardGate::RZ, StandardGate::CZ],
        &["cannot lower", "SWAP", "RZ", "CZ"],
    );
}
