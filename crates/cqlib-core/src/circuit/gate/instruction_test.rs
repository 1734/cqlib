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

use crate::circuit::{
    Circuit, CircuitId, ClassicalExpr, ClassicalType, ClassicalValue, ClassicalVar,
};

#[test]
fn reports_measurements_and_classical_value_reads() {
    let circuit_id = CircuitId::new();
    let result = ClassicalValue::new(circuit_id, 0, ClassicalType::Bit);
    let target = ClassicalVar::new(circuit_id, 0, ClassicalType::Bit);

    let measurement = Instruction::ClassicalData(ClassicalDataOp::MeasureBit { result });
    assert!(measurement.has_measurement());
    assert!(!measurement.reads_value(result));

    let store = Instruction::ClassicalData(ClassicalDataOp::Store {
        target,
        value: ClassicalExpr::value(result),
    });
    assert!(!store.has_measurement());
    assert!(store.reads_value(result));

    let gate = Instruction::Standard(StandardGate::X);
    assert!(!gate.has_measurement());
    assert!(!gate.reads_value(result));
}

#[test]
fn canonicalize_form_collapses_supported_mc_gate_forms() {
    let cases = [
        (MCGate::new(0, StandardGate::X), StandardGate::X),
        (MCGate::new(1, StandardGate::X), StandardGate::CX),
        (MCGate::new(2, StandardGate::X), StandardGate::CCX),
        (MCGate::new(1, StandardGate::CX), StandardGate::CCX),
        (MCGate::new(1, StandardGate::Y), StandardGate::CY),
        (MCGate::new(1, StandardGate::Z), StandardGate::CZ),
        (MCGate::new(1, StandardGate::RX), StandardGate::CRX),
        (MCGate::new(1, StandardGate::RY), StandardGate::CRY),
        (MCGate::new(1, StandardGate::RZ), StandardGate::CRZ),
    ];

    for (mc_gate, expected) in cases {
        let expected_controls = expected.num_ctrl_qubits();
        let expected_qubits = expected.num_qubits();
        let expected_params = expected.num_params();
        let canonical = Instruction::McGate(Box::new(mc_gate)).canonicalize_form();

        let Instruction::Standard(actual) = canonical else {
            panic!("expected standard gate {expected:?}");
        };
        assert_eq!(actual, expected);
        assert_eq!(actual.num_ctrl_qubits(), expected_controls);
        assert_eq!(actual.num_qubits(), expected_qubits);
        assert_eq!(actual.num_params(), expected_params);
    }
}

#[test]
fn canonicalize_form_keeps_non_promotable_mc_gate_forms() {
    let cases = [
        MCGate::new(3, StandardGate::X),
        MCGate::new(1, StandardGate::H),
        MCGate::new(1, StandardGate::RXX),
    ];

    for mc_gate in cases {
        let expected_controls = mc_gate.num_ctrl_qubits();
        let expected_qubits = mc_gate.num_qubits();
        let expected_params = mc_gate.num_params();
        let canonical = Instruction::McGate(Box::new(mc_gate)).canonicalize_form();

        let Instruction::McGate(actual) = canonical else {
            panic!("expected MCGate");
        };
        assert_eq!(actual.num_ctrl_qubits(), expected_controls);
        assert_eq!(actual.num_qubits(), expected_qubits);
        assert_eq!(actual.num_params(), expected_params);
    }
}

#[test]
fn control_on_mc_gate_counts_inherent_base_controls_once() {
    let instruction = Instruction::McGate(Box::new(MCGate::new(1, StandardGate::CX)));

    let controlled = instruction.control(1).unwrap();

    let Instruction::McGate(actual) = controlled else {
        panic!("expected MCGate");
    };
    assert_eq!(actual.base_gate(), &StandardGate::X);
    assert_eq!(actual.num_ctrl_qubits(), 3);
    assert_eq!(actual.num_qubits(), 4);
}

#[test]
fn gate_arity_reports_fixed_gate_like_instructions() {
    let unitary = UnitaryGate::new("custom", 2, 3);
    let circuit_gate = CircuitGate::new("composite", FrozenCircuit::new(Circuit::new(4))).unwrap();

    let cases = [
        (Instruction::Standard(StandardGate::H), Some((1, 0))),
        (Instruction::Standard(StandardGate::RX), Some((1, 1))),
        (Instruction::Standard(StandardGate::GPhase), Some((0, 1))),
        (
            Instruction::McGate(Box::new(MCGate::new(2, StandardGate::RZ))),
            Some((3, 1)),
        ),
        (Instruction::UnitaryGate(Box::new(unitary)), Some((2, 3))),
        (
            Instruction::CircuitGate(Box::new(circuit_gate)),
            Some((4, 0)),
        ),
        (Instruction::Directive(Directive::Measure), Some((1, 0))),
        (Instruction::Directive(Directive::Reset), Some((1, 0))),
        (Instruction::Delay, Some((1, 1))),
    ];

    for (instruction, expected) in cases {
        assert_eq!(instruction.gate_arity(), expected);
    }
}

#[test]
fn gate_arity_excludes_variable_or_contextual_instructions() {
    assert_eq!(
        Instruction::Directive(Directive::Barrier).gate_arity(),
        None
    );
    assert_eq!(
        Instruction::ClassicalControl(ClassicalControlOp::Break).gate_arity(),
        None
    );
}

#[test]
fn classical_control_instruction_is_non_unitary_and_not_controllable() {
    let instruction = Instruction::ClassicalControl(ClassicalControlOp::Continue);

    assert!(instruction.matrix(&[]).is_none());
    assert!(instruction.inverse(&[]).is_none());
    assert!(instruction.control(1).is_none());
    assert_eq!(instruction.to_string(), "continue");
}

#[test]
fn classical_control_op_converts_to_instruction() {
    let instruction: Instruction = ClassicalControlOp::Break.into();

    assert!(matches!(
        instruction,
        Instruction::ClassicalControl(ClassicalControlOp::Break)
    ));
}
