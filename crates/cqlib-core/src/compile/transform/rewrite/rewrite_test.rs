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

use super::{KnowledgeRewriter, RewriteConfig};
use crate::circuit::{
    Circuit, CircuitParam, ClassicalControlOp, ClassicalExpr, Directive, Instruction, MCGate,
    Parameter, Qubit, StandardGate,
};
use crate::compile::CompilerError;
use crate::compile::knowledge::library::RuleKind;
use crate::compile::test_utils::standard_ops;

#[test]
fn cancels_adjacent_self_inverse_gates() {
    let q0 = Qubit::new(0);
    let mut circuit = Circuit::new(1);
    circuit.h(q0).unwrap();
    circuit.h(q0).unwrap();

    let result = KnowledgeRewriter::production().run(&circuit).unwrap();

    assert!(result.changed);
    assert!(result.circuit.operations().is_empty());
    assert!(result.stats.reached_fixpoint);
}

#[test]
fn cancels_across_commuting_disjoint_operation() {
    let q0 = Qubit::new(0);
    let q1 = Qubit::new(1);
    let mut circuit = Circuit::new(2);
    circuit.h(q0).unwrap();
    circuit.x(q1).unwrap();
    circuit.h(q0).unwrap();

    let config = RewriteConfig::production().with_enabled_kinds(vec![RuleKind::Cancel]);
    let result = KnowledgeRewriter::new(config).run(&circuit).unwrap();

    assert!(result.changed);
    assert_eq!(standard_ops(&result.circuit), vec![StandardGate::X]);
    assert_eq!(result.circuit.operations()[0].qubits.as_slice(), &[q1]);
}

#[test]
fn does_not_cancel_across_non_commuting_operation() {
    let q0 = Qubit::new(0);
    let mut circuit = Circuit::new(1);
    circuit.h(q0).unwrap();
    circuit.x(q0).unwrap();
    circuit.h(q0).unwrap();

    let config = RewriteConfig::production().with_enabled_kinds(vec![RuleKind::Cancel]);
    let result = KnowledgeRewriter::new(config).run(&circuit).unwrap();

    assert!(!result.changed);
    assert_eq!(
        standard_ops(&result.circuit),
        vec![StandardGate::H, StandardGate::X, StandardGate::H]
    );
}

#[test]
fn protects_labeled_operations_by_default() {
    let q0 = Qubit::new(0);
    let mut circuit = Circuit::new(1);
    circuit
        .append(
            Instruction::Standard(StandardGate::H),
            [q0],
            std::iter::empty(),
            Some("keep"),
        )
        .unwrap();
    circuit.h(q0).unwrap();

    let result = KnowledgeRewriter::production().run(&circuit).unwrap();

    assert!(!result.changed);
    assert_eq!(
        standard_ops(&result.circuit),
        vec![StandardGate::H, StandardGate::H]
    );
}

#[test]
fn does_not_cross_labeled_skipped_operation() {
    let q0 = Qubit::new(0);
    let q1 = Qubit::new(1);
    let mut circuit = Circuit::new(2);
    circuit.h(q0).unwrap();
    circuit
        .append(
            Instruction::Standard(StandardGate::X),
            [q1],
            std::iter::empty(),
            Some("skip"),
        )
        .unwrap();
    circuit.h(q0).unwrap();

    let config = RewriteConfig::production().with_enabled_kinds(vec![RuleKind::Cancel]);
    let result = KnowledgeRewriter::new(config).run(&circuit).unwrap();

    assert!(!result.changed);
    assert_eq!(
        standard_ops(&result.circuit),
        vec![StandardGate::H, StandardGate::X, StandardGate::H]
    );
}

#[test]
fn barrier_splits_rewrite_blocks() {
    let q0 = Qubit::new(0);
    let mut circuit = Circuit::new(1);
    circuit.h(q0).unwrap();
    circuit.barrier(vec![q0]).unwrap();
    circuit.h(q0).unwrap();

    let config = RewriteConfig::production().with_enabled_kinds(vec![RuleKind::Cancel]);
    let result = KnowledgeRewriter::new(config).run(&circuit).unwrap();

    assert!(!result.changed);
    assert_eq!(result.circuit.operations().len(), 3);
    assert!(matches!(
        result.circuit.operations()[1].instruction,
        Instruction::Directive(Directive::Barrier)
    ));
}

#[test]
fn merges_numeric_rotations() {
    let q0 = Qubit::new(0);
    let mut circuit = Circuit::new(1);
    circuit.rz(q0, 0.25).unwrap();
    circuit.rz(q0, 0.5).unwrap();

    let config = RewriteConfig::production().with_enabled_kinds(vec![RuleKind::Merge]);
    let result = KnowledgeRewriter::new(config).run(&circuit).unwrap();

    assert!(result.changed);
    assert_eq!(standard_ops(&result.circuit), vec![StandardGate::RZ]);
    assert!(matches!(
        result.circuit.operations()[0].params[0],
        CircuitParam::Fixed(value) if (value - 0.75).abs() < 1e-12
    ));
}

#[test]
fn merges_symbolic_rotations() {
    let q0 = Qubit::new(0);
    let theta = Parameter::symbol("theta");
    let mut circuit = Circuit::new(1);
    circuit.rz(q0, theta.clone()).unwrap();
    circuit.rz(q0, 0.5).unwrap();

    let config = RewriteConfig::production().with_enabled_kinds(vec![RuleKind::Merge]);
    let result = KnowledgeRewriter::new(config).run(&circuit).unwrap();

    assert!(result.changed);
    assert_eq!(standard_ops(&result.circuit), vec![StandardGate::RZ]);
    let merged = operation_param(&result.circuit, &result.circuit.operations()[0].params[0]);
    let expected = theta + Parameter::from(0.5);
    assert!(merged.provably_equal(&expected, 1e-12));
}

#[test]
fn merges_rz_across_same_qubit_commuting_s_gate() {
    let q0 = Qubit::new(0);
    let mut circuit = Circuit::new(1);
    circuit.rz(q0, 0.25).unwrap();
    circuit.s(q0).unwrap();
    circuit.rz(q0, 0.5).unwrap();

    let config = RewriteConfig::production().with_enabled_kinds(vec![RuleKind::Merge]);
    let result = KnowledgeRewriter::new(config).run(&circuit).unwrap();

    assert!(result.changed);
    assert_eq!(
        standard_ops(&result.circuit),
        vec![StandardGate::RZ, StandardGate::S]
    );
    assert!(matches!(
        result.circuit.operations()[0].params[0],
        CircuitParam::Fixed(value) if (value - 0.75).abs() < 1e-12
    ));
}

#[test]
fn merges_rz_across_same_qubit_commuting_phase_gate() {
    let q0 = Qubit::new(0);
    let mut circuit = Circuit::new(1);
    circuit.rz(q0, 0.25).unwrap();
    circuit.phase(q0, 0.125).unwrap();
    circuit.rz(q0, 0.5).unwrap();

    let config = RewriteConfig::production().with_enabled_kinds(vec![RuleKind::Merge]);
    let result = KnowledgeRewriter::new(config).run(&circuit).unwrap();

    assert!(result.changed);
    assert_eq!(
        standard_ops(&result.circuit),
        vec![StandardGate::RZ, StandardGate::Phase]
    );
    assert!(matches!(
        result.circuit.operations()[0].params[0],
        CircuitParam::Fixed(value) if (value - 0.75).abs() < 1e-12
    ));
    assert!(matches!(
        result.circuit.operations()[1].params[0],
        CircuitParam::Fixed(value) if (value - 0.125).abs() < 1e-12
    ));
}

#[test]
fn cancels_z_across_same_qubit_commuting_s_gate() {
    let q0 = Qubit::new(0);
    let mut circuit = Circuit::new(1);
    circuit.z(q0).unwrap();
    circuit.s(q0).unwrap();
    circuit.z(q0).unwrap();

    let config = RewriteConfig::production().with_enabled_kinds(vec![RuleKind::Cancel]);
    let result = KnowledgeRewriter::new(config).run(&circuit).unwrap();

    assert!(result.changed);
    assert_eq!(standard_ops(&result.circuit), vec![StandardGate::S]);
}

#[test]
fn does_not_merge_rz_across_non_commuting_h_gate() {
    let q0 = Qubit::new(0);
    let mut circuit = Circuit::new(1);
    circuit.rz(q0, 0.25).unwrap();
    circuit.h(q0).unwrap();
    circuit.rz(q0, 0.5).unwrap();

    let config = RewriteConfig::production().with_enabled_kinds(vec![RuleKind::Merge]);
    let result = KnowledgeRewriter::new(config).run(&circuit).unwrap();

    assert!(!result.changed);
    assert_eq!(
        standard_ops(&result.circuit),
        vec![StandardGate::RZ, StandardGate::H, StandardGate::RZ]
    );
}

#[test]
fn does_not_cancel_x_across_non_commuting_z_gate() {
    let q0 = Qubit::new(0);
    let mut circuit = Circuit::new(1);
    circuit.x(q0).unwrap();
    circuit.z(q0).unwrap();
    circuit.x(q0).unwrap();

    let config = RewriteConfig::production().with_enabled_kinds(vec![RuleKind::Cancel]);
    let result = KnowledgeRewriter::new(config).run(&circuit).unwrap();

    assert!(!result.changed);
    assert_eq!(
        standard_ops(&result.circuit),
        vec![StandardGate::X, StandardGate::Z, StandardGate::X]
    );
}

#[test]
fn commuting_match_respects_max_window_ops() {
    let q0 = Qubit::new(0);
    let mut circuit = Circuit::new(1);
    circuit.rz(q0, 0.25).unwrap();
    circuit.s(q0).unwrap();
    circuit.t(q0).unwrap();
    circuit.rz(q0, 0.5).unwrap();

    let tight_config = RewriteConfig::production()
        .with_enabled_kinds(vec![RuleKind::Merge])
        .with_max_window_ops(1);
    let tight_result = KnowledgeRewriter::new(tight_config).run(&circuit).unwrap();

    assert!(!tight_result.changed);
    assert_eq!(
        standard_ops(&tight_result.circuit),
        vec![
            StandardGate::RZ,
            StandardGate::S,
            StandardGate::T,
            StandardGate::RZ
        ]
    );

    let wide_config = RewriteConfig::production()
        .with_enabled_kinds(vec![RuleKind::Merge])
        .with_max_window_ops(4);
    let wide_result = KnowledgeRewriter::new(wide_config).run(&circuit).unwrap();

    assert!(wide_result.changed);
    assert_eq!(
        standard_ops(&wide_result.circuit),
        vec![StandardGate::RZ, StandardGate::S, StandardGate::T]
    );
    assert!(matches!(
        wide_result.circuit.operations()[0].params[0],
        CircuitParam::Fixed(value) if (value - 0.75).abs() < 1e-12
    ));
}

#[test]
fn folds_top_level_gphase_into_circuit_global_phase() {
    let mut circuit = Circuit::new(1);
    circuit
        .append(
            Instruction::Standard(StandardGate::GPhase),
            std::iter::empty::<Qubit>(),
            [Parameter::from(0.25).into()],
            None,
        )
        .unwrap();

    let result = KnowledgeRewriter::production().run(&circuit).unwrap();

    assert!(result.changed);
    assert!(result.circuit.operations().is_empty());
    assert!(
        result
            .circuit
            .global_phase()
            .provably_equal(&Parameter::from(0.25), 1e-12)
    );
}

#[test]
fn lowers_to_explicit_target_basis() {
    let q0 = Qubit::new(0);
    let q1 = Qubit::new(1);
    let mut circuit = Circuit::new(2);
    circuit.cx(q0, q1).unwrap();

    let config = RewriteConfig::lowering()
        .with_target_instructions(vec![
            Instruction::Standard(StandardGate::H),
            Instruction::Standard(StandardGate::CZ),
        ])
        .unwrap();
    let result = KnowledgeRewriter::new(config).run(&circuit).unwrap();

    assert!(result.changed);
    assert_eq!(
        standard_ops(&result.circuit),
        vec![StandardGate::H, StandardGate::CZ, StandardGate::H]
    );
    assert_eq!(result.circuit.operations()[0].qubits.as_slice(), &[q1]);
    assert_eq!(result.circuit.operations()[1].qubits.as_slice(), &[q0, q1]);
    assert_eq!(result.circuit.operations()[2].qubits.as_slice(), &[q1]);
}

#[test]
fn target_basis_lowering_preserves_physical_source_gate() {
    let q0 = Qubit::new(0);
    let mut circuit = Circuit::new(1);
    circuit.x2p(q0).unwrap();

    let config = RewriteConfig::lowering()
        .with_target_instructions(vec![
            Instruction::Standard(StandardGate::RZ),
            Instruction::Standard(StandardGate::X2P),
        ])
        .unwrap();
    let result = KnowledgeRewriter::new(config).run(&circuit).unwrap();

    assert_eq!(standard_ops(&result.circuit), vec![StandardGate::X2P]);
    assert!(!standard_ops(&result.circuit).contains(&StandardGate::RY));
}

#[test]
fn one_round_limit_stops_before_second_step_lowering() {
    let q0 = Qubit::new(0);
    let q1 = Qubit::new(1);
    let q2 = Qubit::new(2);
    let mut circuit = Circuit::new(3);
    circuit
        .append(
            Instruction::McGate(Box::new(MCGate::new(2, StandardGate::X))),
            [q0, q1, q2],
            std::iter::empty::<crate::circuit::ParameterValue>(),
            None,
        )
        .unwrap();

    let config = RewriteConfig::lowering()
        .with_enabled_kinds(vec![RuleKind::Decompose])
        .with_max_rounds(1);
    let result = KnowledgeRewriter::new(config).run(&circuit).unwrap();

    assert!(result.changed);
    assert_eq!(standard_ops(&result.circuit), vec![StandardGate::CCX]);
    assert!(matches!(
        result.circuit.operations()[0].instruction,
        Instruction::Standard(StandardGate::CCX)
    ));
}

#[test]
fn two_rounds_continue_chain_beyond_first_replacement() {
    let q0 = Qubit::new(0);
    let q1 = Qubit::new(1);
    let q2 = Qubit::new(2);
    let mut circuit = Circuit::new(3);
    circuit
        .append(
            Instruction::McGate(Box::new(MCGate::new(2, StandardGate::X))),
            [q0, q1, q2],
            std::iter::empty::<crate::circuit::ParameterValue>(),
            None,
        )
        .unwrap();

    let config = RewriteConfig::lowering()
        .with_enabled_kinds(vec![RuleKind::Decompose])
        .with_max_rounds(2);
    let result = KnowledgeRewriter::new(config).run(&circuit).unwrap();

    assert!(result.changed);
    assert!(result.circuit.operations().iter().all(|operation| {
        !matches!(
            operation.instruction,
            Instruction::McGate(_) | Instruction::Standard(StandardGate::CCX)
        )
    }));
    assert_eq!(
        standard_ops(&result.circuit),
        vec![
            StandardGate::H,
            StandardGate::CX,
            StandardGate::TDG,
            StandardGate::CX,
            StandardGate::T,
            StandardGate::CX,
            StandardGate::TDG,
            StandardGate::CX,
            StandardGate::T,
            StandardGate::T,
            StandardGate::H,
            StandardGate::CX,
            StandardGate::T,
            StandardGate::TDG,
            StandardGate::CX
        ]
    );
}

#[test]
fn lowering_reaches_target_basis_through_multiple_steps() {
    let q0 = Qubit::new(0);
    let q1 = Qubit::new(1);
    let q2 = Qubit::new(2);
    let mut circuit = Circuit::new(3);
    circuit
        .append(
            Instruction::McGate(Box::new(MCGate::new(2, StandardGate::X))),
            [q0, q1, q2],
            std::iter::empty::<crate::circuit::ParameterValue>(),
            None,
        )
        .unwrap();

    let config = RewriteConfig::lowering()
        .with_enabled_kinds(vec![RuleKind::Decompose])
        .with_target_instructions(vec![
            Instruction::Standard(StandardGate::H),
            Instruction::Standard(StandardGate::CX),
            Instruction::Standard(StandardGate::T),
            Instruction::Standard(StandardGate::TDG),
        ])
        .unwrap()
        .with_max_rounds(4);
    let result = KnowledgeRewriter::new(config).run(&circuit).unwrap();

    assert!(result.changed);
    assert!(result.circuit.operations().iter().all(|operation| matches!(
        operation.instruction,
        Instruction::Standard(
            StandardGate::H | StandardGate::CX | StandardGate::T | StandardGate::TDG
        )
    )));
    assert!(result.stats.rules_applied >= 2);
    assert!(result.stats.rounds_executed >= 3);
}

#[test]
fn lowering_fails_when_target_basis_cannot_be_satisfied() {
    let q0 = Qubit::new(0);
    let mut circuit = Circuit::new(1);
    circuit.h(q0).unwrap();

    let config = RewriteConfig::lowering()
        .with_target_instructions(vec![Instruction::Standard(StandardGate::CZ)])
        .unwrap();
    let err = KnowledgeRewriter::new(config).run(&circuit).unwrap_err();

    assert!(matches!(err, CompilerError::InvalidInput(_)));
}

#[test]
fn optimize_mode_does_not_apply_decomposition_rules() {
    let q0 = Qubit::new(0);
    let q1 = Qubit::new(1);
    let mut circuit = Circuit::new(2);
    circuit.cx(q0, q1).unwrap();

    let config = RewriteConfig::production().with_enabled_kinds(vec![RuleKind::Decompose]);
    let result = KnowledgeRewriter::new(config).run(&circuit).unwrap();

    assert!(!result.changed);
    assert_eq!(standard_ops(&result.circuit), vec![StandardGate::CX]);
}

#[test]
fn rejects_invalid_target_basis_configuration() {
    let err = RewriteConfig::lowering()
        .with_target_instructions(vec![Instruction::Delay])
        .unwrap_err();

    assert!(matches!(err, CompilerError::InvalidInput(_)));
}

#[test]
fn rejects_zero_round_limit() {
    let circuit = Circuit::new(1);
    let err = KnowledgeRewriter::new(RewriteConfig::production().with_max_rounds(0))
        .run(&circuit)
        .unwrap_err();

    assert!(matches!(err, CompilerError::InvalidInput(_)));
}

#[test]
fn preserves_control_flow_body_local_global_phase() {
    let q1 = Qubit::new(1);
    let mut circuit = Circuit::new(2);
    circuit
        .if_else(
            ClassicalExpr::bool_literal(true),
            |body| {
                body.h(q1)?;
                body.y(q1)?;
                body.h(q1)
            },
            |_| Ok(()),
        )
        .unwrap();

    let result = KnowledgeRewriter::production().run(&circuit).unwrap();

    assert!(result.changed);
    let Instruction::ClassicalControl(ClassicalControlOp::If(op)) =
        &result.circuit.operations()[0].instruction
    else {
        panic!("expected if operation");
    };
    assert_eq!(op.then_body().operations().len(), 2);
    assert!(matches!(
        op.then_body().operations()[0].instruction,
        Instruction::Standard(StandardGate::GPhase)
    ));
    assert!(matches!(
        op.then_body().operations()[1].instruction,
        Instruction::Standard(StandardGate::Y)
    ));
}

#[test]
fn rewrites_false_branch_and_while_body() {
    let q1 = Qubit::new(1);

    let mut if_circuit = Circuit::new(2);
    if_circuit
        .if_else(
            ClassicalExpr::bool_literal(true),
            |body| body.x(q1),
            |body| {
                body.h(q1)?;
                body.h(q1)
            },
        )
        .unwrap();
    let if_result = KnowledgeRewriter::production().run(&if_circuit).unwrap();
    let Instruction::ClassicalControl(ClassicalControlOp::If(if_op)) =
        &if_result.circuit.operations()[0].instruction
    else {
        panic!("expected if operation");
    };
    assert_eq!(if_op.then_body().operations().len(), 1);
    assert!(if_op.else_body().unwrap().operations().is_empty());

    let mut while_circuit = Circuit::new(2);
    while_circuit
        .while_(ClassicalExpr::bool_literal(true), |body| {
            body.h(q1)?;
            body.h(q1)
        })
        .unwrap();
    let while_result = KnowledgeRewriter::production().run(&while_circuit).unwrap();
    let Instruction::ClassicalControl(ClassicalControlOp::While(while_op)) =
        &while_result.circuit.operations()[0].instruction
    else {
        panic!("expected while operation");
    };
    assert!(while_op.body().operations().is_empty());
}

#[test]
fn rewrites_runtime_classical_control_body_preserving_handles() {
    let mut circuit = Circuit::new(1);
    let measured = circuit.measure(Qubit::new(0)).unwrap();
    circuit
        .if_(
            ClassicalExpr::bit_to_bool(measured.expr()).unwrap(),
            |body| {
                body.x(Qubit::new(0))?;
                body.x(Qubit::new(0))
            },
        )
        .unwrap();

    let result = KnowledgeRewriter::production().run(&circuit).unwrap();

    assert!(result.changed);
    assert_eq!(result.circuit.classical_values().len(), 1);
    assert!(result.circuit.validate().is_ok());
    let Instruction::ClassicalControl(ClassicalControlOp::If(op)) =
        &result.circuit.operations()[1].instruction
    else {
        panic!("expected runtime classical if operation");
    };
    assert!(op.then_body().operations().is_empty());
}

#[test]
fn rewrites_control_flow_body_with_valid_rebuilt_parameter_table() {
    let q1 = Qubit::new(1);
    let theta = Parameter::symbol("theta");
    let mut circuit = Circuit::new(2);
    circuit
        .if_else(
            ClassicalExpr::bool_literal(true),
            |body| {
                body.rz(q1, theta.clone())?;
                body.rz(q1, 0.5)
            },
            |_| Ok(()),
        )
        .unwrap();

    let config = RewriteConfig::production().with_enabled_kinds(vec![RuleKind::Merge]);
    let result = KnowledgeRewriter::new(config).run(&circuit).unwrap();
    let Instruction::ClassicalControl(ClassicalControlOp::If(op)) =
        &result.circuit.operations()[0].instruction
    else {
        panic!("expected if operation");
    };
    assert_eq!(op.then_body().operations().len(), 1);
    let body_param = &op.then_body().operations()[0].params[0];
    let merged = operation_param(&result.circuit, body_param);
    assert!(merged.provably_equal(&(theta + Parameter::from(0.5)), 1e-12));
}

fn operation_param(circuit: &Circuit, param: &CircuitParam) -> Parameter {
    match param {
        CircuitParam::Fixed(value) => Parameter::from(*value),
        CircuitParam::Index(index) => circuit
            .parameters()
            .get_index(*index as usize)
            .cloned()
            .expect("parameter index should exist in rebuilt circuit"),
    }
}
