use super::*;
use crate::circuit::{Instruction, Qubit, StandardGate, ValueOperation};

#[test]
fn cost_order_prefers_fewer_two_qubit_ops_before_total_gate_count() {
    let lower_two_qubit = ResynthesisCost {
        two_qubit_ops: 1,
        total_ops: 8,
        ..ResynthesisCost::default()
    };
    let higher_two_qubit = ResynthesisCost {
        two_qubit_ops: 2,
        total_ops: 2,
        ..ResynthesisCost::default()
    };

    assert!(lower_two_qubit < higher_two_qubit);
}

#[test]
fn gphase_replacements_are_cost_free() {
    let op = ValueOperation::from_standard(StandardGate::GPhase, [], [0.3.into()]);

    assert_eq!(cost_of_replacements(&[op]), ResynthesisCost::default());
}

#[test]
fn replacement_cost_counts_two_qubit_gate_and_depth() {
    let q0 = Qubit::new(0);
    let q1 = Qubit::new(1);
    let op = ValueOperation::from_standard(StandardGate::CX, [q0, q1], []);

    let cost = cost_of_replacements(&[op]);

    assert_eq!(cost.two_qubit_ops, 1);
    assert_eq!(cost.total_ops, 1);
    assert_eq!(cost.depth_estimate, 1);
}

#[test]
fn unsupported_replacement_is_worse_than_standard_operation() {
    let q0 = Qubit::new(0);
    let unsupported = ValueOperation {
        instruction: crate::circuit::ValueInstruction::from_instruction(Instruction::Delay),
        qubits: smallvec::smallvec![q0],
        params: smallvec::smallvec![],
        label: None,
    };

    assert_eq!(cost_of_replacements(&[unsupported]).unsupported_ops, 1);
}

#[test]
fn depth_estimate_keeps_disjoint_single_qubit_ops_parallel() {
    let q0 = Qubit::new(0);
    let q1 = Qubit::new(1);
    let ops = [
        ValueOperation::from_standard(StandardGate::H, [q0], []),
        ValueOperation::from_standard(StandardGate::X, [q1], []),
    ];

    let cost = cost_of_replacements(&ops);

    assert_eq!(cost.total_ops, 2);
    assert_eq!(cost.depth_estimate, 1);
}

#[test]
fn depth_estimate_serializes_ops_on_shared_qubit() {
    let q0 = Qubit::new(0);
    let ops = [
        ValueOperation::from_standard(StandardGate::H, [q0], []),
        ValueOperation::from_standard(StandardGate::X, [q0], []),
    ];

    let cost = cost_of_replacements(&ops);

    assert_eq!(cost.total_ops, 2);
    assert_eq!(cost.depth_estimate, 2);
}
