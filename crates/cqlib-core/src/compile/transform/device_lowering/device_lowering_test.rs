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

use super::DeviceLowerer;
use crate::circuit::{
    Circuit, ClassicalControlOp, ClassicalExpr, ClassicalType, Instruction, Parameter,
    ParameterValue, Qubit, StandardGate,
};
use crate::compile::CompilerError;
use crate::compile::transform::Transformer;
use crate::device::{Device, PhysicalQubit};
use crate::util::test_utils::{assert_compiled_circuit_equivalent, standard_ops};
use std::collections::HashMap;
use std::f64::consts::PI;

#[test]
fn exact_native_circuit_is_unchanged() {
    let mut circuit = Circuit::new(2);
    circuit.h(Qubit::new(0)).unwrap();
    circuit.cx(Qubit::new(0), Qubit::new(1)).unwrap();
    let device =
        Device::line_from_qubits("native", vec![PhysicalQubit::new(0), PhysicalQubit::new(1)])
            .unwrap()
            .with_native_gates(vec![
                Instruction::Standard(StandardGate::H),
                Instruction::Standard(StandardGate::CX),
            ])
            .unwrap();

    let result = DeviceLowerer::new(&device)
        .transform(&circuit, None)
        .unwrap();

    assert!(!result.changed);
    device.validate_circuit(&result.circuit).unwrap();
}

#[test]
fn reverse_native_swap_uses_symmetric_direction_template() {
    let mut circuit = Circuit::new(2);
    circuit.swap(Qubit::new(0), Qubit::new(1)).unwrap();
    let device = Device::line_from_qubits(
        "reverse-swap",
        vec![PhysicalQubit::new(1), PhysicalQubit::new(0)],
    )
    .unwrap()
    .with_native_gates(vec![Instruction::Standard(StandardGate::SWAP)])
    .unwrap();

    let result = DeviceLowerer::new(&device)
        .transform(&circuit, None)
        .unwrap();

    assert!(result.changed);
    assert_eq!(
        result.circuit.operations()[0].qubits.as_slice(),
        &[Qubit::new(1), Qubit::new(0)]
    );
    device.validate_circuit(&result.circuit).unwrap();
}

#[test]
fn missing_reverse_cx_support_returns_structured_failure() {
    let mut circuit = Circuit::new(2);
    circuit.cx(Qubit::new(0), Qubit::new(1)).unwrap();
    let device = Device::line_from_qubits(
        "reverse-cx-without-h",
        vec![PhysicalQubit::new(1), PhysicalQubit::new(0)],
    )
    .unwrap()
    .with_native_gates(vec![Instruction::Standard(StandardGate::CX)])
    .unwrap();

    let error = DeviceLowerer::new(&device)
        .transform(&circuit, None)
        .unwrap_err();
    let CompilerError::DeviceLoweringFailed(failure) = error else {
        panic!("expected device lowering failure");
    };
    assert_eq!(
        failure.qargs,
        vec![PhysicalQubit::new(0), PhysicalQubit::new(1)]
    );
    assert!(
        failure
            .attempted_candidates
            .iter()
            .any(|candidate| candidate.template == "direction_reverse_CX")
    );
}

#[test]
fn reverse_cx_direction_template_preserves_unitary() {
    let q0 = Qubit::new(0);
    let q1 = Qubit::new(1);
    let mut source = Circuit::new(2);
    source.cx(q0, q1).unwrap();
    let device = Device::line_from_qubits(
        "reverse-cx-equivalence",
        vec![PhysicalQubit::new(1), PhysicalQubit::new(0)],
    )
    .unwrap()
    .with_native_gates(vec![
        Instruction::Standard(StandardGate::H),
        Instruction::Standard(StandardGate::CX),
    ])
    .unwrap();

    let result = DeviceLowerer::new(&device)
        .transform(&source, None)
        .unwrap();

    assert_eq!(
        standard_ops(&result.circuit),
        vec![
            StandardGate::H,
            StandardGate::H,
            StandardGate::CX,
            StandardGate::H,
            StandardGate::H,
        ]
    );
    assert_eq!(result.circuit.operations()[2].qubits.as_slice(), &[q1, q0]);
    assert_compiled_circuit_equivalent(&result.circuit, &source);
    device.validate_circuit(&result.circuit).unwrap();
}

#[test]
fn reverse_rzx_direction_template_preserves_symbolic_unitary() {
    let q0 = Qubit::new(0);
    let q1 = Qubit::new(1);
    let theta = Parameter::symbol("theta");
    let mut source = Circuit::new(2);
    source.rzx(q0, q1, theta).unwrap();
    let device = Device::line_from_qubits(
        "reverse-rzx-equivalence",
        vec![PhysicalQubit::new(1), PhysicalQubit::new(0)],
    )
    .unwrap()
    .with_native_gates(vec![
        Instruction::Standard(StandardGate::H),
        Instruction::Standard(StandardGate::RZX),
    ])
    .unwrap();

    let result = DeviceLowerer::new(&device)
        .transform(&source, None)
        .unwrap();

    assert_eq!(
        standard_ops(&result.circuit),
        vec![
            StandardGate::H,
            StandardGate::H,
            StandardGate::RZX,
            StandardGate::H,
            StandardGate::H,
        ]
    );
    assert_eq!(result.circuit.operations()[2].qubits.as_slice(), &[q1, q0]);
    assert!(result.circuit.symbols().contains("theta"));
    for value in [0.0, 0.37, -PI / 2.0, PI] {
        let bindings = Some(HashMap::from([("theta", value)]));
        let bound_source = source.assign_parameters(&bindings).unwrap();
        let bound_result = result.circuit.assign_parameters(&bindings).unwrap();
        assert_compiled_circuit_equivalent(&bound_result, &bound_source);
    }
    device.validate_circuit(&result.circuit).unwrap();
}

#[test]
fn symmetric_direction_templates_preserve_unitary() {
    let q0 = Qubit::new(0);
    let q1 = Qubit::new(1);
    let cases = [
        (StandardGate::CZ, vec![]),
        (StandardGate::SWAP, vec![]),
        (StandardGate::RXX, vec![ParameterValue::Fixed(0.37)]),
        (StandardGate::RYY, vec![ParameterValue::Fixed(-0.41)]),
        (StandardGate::RZZ, vec![ParameterValue::Fixed(0.23)]),
        (
            StandardGate::FSIM,
            vec![ParameterValue::Fixed(0.31), ParameterValue::Fixed(-0.27)],
        ),
    ];

    for (gate, params) in cases {
        let mut source = Circuit::new(2);
        source
            .append(Instruction::Standard(gate), [q0, q1], params, None)
            .unwrap();
        let device = Device::line_from_qubits(
            format!("reverse-{gate:?}-equivalence"),
            vec![PhysicalQubit::new(1), PhysicalQubit::new(0)],
        )
        .unwrap()
        .with_native_gates(vec![Instruction::Standard(gate)])
        .unwrap();

        let result = DeviceLowerer::new(&device)
            .transform(&source, None)
            .unwrap();

        assert_eq!(standard_ops(&result.circuit), vec![gate], "gate={gate:?}");
        assert_eq!(
            result.circuit.operations()[0].qubits.as_slice(),
            &[q1, q0],
            "gate={gate:?}"
        );
        assert_compiled_circuit_equivalent(&result.circuit, &source);
        device.validate_circuit(&result.circuit).unwrap();
    }
}

#[test]
fn top_level_gphase_is_folded_without_a_device_capability() {
    let mut circuit = Circuit::new(1);
    circuit
        .append(
            Instruction::Standard(StandardGate::GPhase),
            std::iter::empty::<Qubit>(),
            [ParameterValue::Fixed(0.25)],
            None,
        )
        .unwrap();
    circuit.h(Qubit::new(0)).unwrap();
    let device = Device::line("phase", 1)
        .unwrap()
        .with_native_gates(vec![Instruction::Standard(StandardGate::H)])
        .unwrap();

    let result = DeviceLowerer::new(&device)
        .transform(&circuit, None)
        .unwrap();

    assert_eq!(result.circuit.operations().len(), 1);
    assert!((result.circuit.global_phase().evaluate(&None).unwrap() - 0.25).abs() < 1e-12);
    device.validate_circuit(&result.circuit).unwrap();
}

#[test]
fn body_local_gphase_remains_a_semantic_marker_accepted_by_verifier() {
    let mut circuit = Circuit::new(1);
    circuit
        .if_(ClassicalExpr::bool_literal(true), |body| {
            body.append(
                Instruction::Standard(StandardGate::GPhase),
                std::iter::empty::<Qubit>(),
                [ParameterValue::Fixed(0.375)],
                None,
            )?;
            body.h(Qubit::new(0))?;
            Ok(())
        })
        .unwrap();
    let device = Device::line("body-phase", 1)
        .unwrap()
        .with_native_gates(vec![Instruction::Standard(StandardGate::H)])
        .unwrap();

    let result = DeviceLowerer::new(&device)
        .transform(&circuit, None)
        .unwrap();

    let Instruction::ClassicalControl(ClassicalControlOp::If(control)) =
        &result.circuit.operations()[0].instruction
    else {
        panic!("expected if control");
    };
    assert!(matches!(
        control.then_body().operations()[0].instruction,
        Instruction::Standard(StandardGate::GPhase)
    ));
    device.validate_circuit(&result.circuit).unwrap();
}

#[test]
fn selected_rule_binds_symbolic_parameters_when_emitted() {
    let theta = Parameter::symbol("theta");
    let mut circuit = Circuit::new(2);
    circuit
        .crz(Qubit::new(0), Qubit::new(1), theta.clone())
        .unwrap();
    let device = Device::line("parameter-rule", 2)
        .unwrap()
        .with_native_gates(vec![
            Instruction::Standard(StandardGate::H),
            Instruction::Standard(StandardGate::RZ),
            Instruction::Standard(StandardGate::CX),
        ])
        .unwrap();

    let result = DeviceLowerer::new(&device)
        .transform(&circuit, None)
        .unwrap();

    assert!(result.changed);
    assert!(result.circuit.symbols().contains("theta"));
    device.validate_circuit(&result.circuit).unwrap();
}

#[test]
fn while_for_and_switch_bodies_are_lowered_recursively() {
    let q0 = Qubit::new(0);
    let q1 = Qubit::new(1);
    let mut circuit = Circuit::new(2);
    circuit
        .while_(ClassicalExpr::bool_literal(true), |body| body.cx(q0, q1))
        .unwrap();
    let loop_var = circuit.var(ClassicalType::uint(2).unwrap());
    circuit
        .for_uint(
            loop_var,
            ClassicalExpr::uint_literal(2, 0).unwrap(),
            ClassicalExpr::uint_literal(2, 1).unwrap(),
            ClassicalExpr::uint_literal(2, 1).unwrap(),
            |body, _| body.cx(q0, q1),
        )
        .unwrap();
    circuit
        .switch(ClassicalExpr::uint_literal(2, 0).unwrap(), |switch| {
            switch.value(0, |body| body.cx(q0, q1))?;
            switch.default(|body| body.cx(q0, q1))?;
            Ok(())
        })
        .unwrap();
    let device = Device::line_from_qubits(
        "control-flow",
        vec![PhysicalQubit::new(1), PhysicalQubit::new(0)],
    )
    .unwrap()
    .with_native_gates(vec![
        Instruction::Standard(StandardGate::H),
        Instruction::Standard(StandardGate::CX),
    ])
    .unwrap();

    let result = DeviceLowerer::new(&device)
        .transform(&circuit, None)
        .unwrap();

    assert!(result.changed);
    device.validate_circuit(&result.circuit).unwrap();
}
