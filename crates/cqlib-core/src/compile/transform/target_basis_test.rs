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

use super::TargetBasisLowerer;
use crate::circuit::{Circuit, Instruction, Qubit, StandardGate};
use crate::compile::CompilerError;
use crate::compile::transform::Transformer;
use crate::util::test_utils::standard_ops;

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
