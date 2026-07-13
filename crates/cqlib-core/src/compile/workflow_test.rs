// This code is part of Cqlib.
//
// (C) Copyright China Telecom Quantum Group 2025-2026
//
// This code is licensed under the Apache License, Version 2.0.
// You may obtain a copy of this license in the LICENSE.txt file in
// the root directory of this source tree or at
// http://www.apache.org/licenses/LICENSE-2.0.
//
// Any modifications or derivative works of this code must retain this
// copyright notice, and modified files need to carry a notice indicating
// that they have been altered from the originals.

use super::{CompilerWorkflow, RewritePhase, WorkflowState};
use crate::circuit::gate::FrozenCircuit;
use crate::circuit::{
    Circuit, CircuitGate, CircuitParam, Instruction, MCGate, Parameter, ParameterValue, Qubit,
    StandardGate, UnitaryGate,
};
use crate::compile::resource::ResourcePolicy;
use crate::compile::transform::CircuitAnalysis;
use crate::compile::transform::decompose::unitary::TwoQubitSynthesisTarget;
use crate::compile::{CompileConfig, CompileMode, CompilerError, compile};
use crate::device::{Device, Layout};
use crate::util::test_utils::{
    assert_compiled_circuit_equivalent, contains_high_level_gate, standard_ops, step_changed,
    two_qubit_device,
};
use ndarray::array;
use num_complex::Complex64;
use std::collections::HashMap;
use std::f64::consts::PI;

fn compile_config(mode: CompileMode) -> CompileConfig {
    CompileConfig {
        mode,
        target_basis: None,
        device: None,
        initial_layout: None,
        resource_policy: ResourcePolicy::default(),
        seed: None,
    }
}

fn run_workflow(circuit: &Circuit, mode: CompileMode) -> super::CompileResult {
    CompilerWorkflow::new(compile_config(mode))
        .run(circuit)
        .unwrap()
}

fn workflow_state_with_target_basis(target_basis: Vec<Instruction>) -> WorkflowState {
    let current = Circuit::new(1);
    let two_qubit_target = TwoQubitSynthesisTarget::from_instructions(Some(&target_basis)).unwrap();
    WorkflowState {
        analysis: CircuitAnalysis::analyze(&current),
        current,
        changed: false,
        steps: Vec::new(),
        target_basis: Some(target_basis),
        two_qubit_target,
    }
}

fn binding_case(bindings: &[(&'static str, f64)]) -> Option<HashMap<&'static str, f64>> {
    Some(bindings.iter().copied().collect())
}

fn assert_bindings_preserve_semantics(
    source: &Circuit,
    compiled: &Circuit,
    binding_cases: &[Option<HashMap<&'static str, f64>>],
) {
    for bindings in binding_cases {
        let bound_source = source.assign_parameters(bindings).unwrap();
        let bound_compiled = compiled.assign_parameters(bindings).unwrap();
        assert_compiled_circuit_equivalent(&bound_compiled, &bound_source);
    }
}

fn operation_parameter(circuit: &Circuit, param: &CircuitParam) -> Parameter {
    match param {
        CircuitParam::Fixed(value) => Parameter::from(*value),
        CircuitParam::Index(index) => circuit
            .parameters()
            .get_index(*index as usize)
            .cloned()
            .expect("parameter index should exist in rebuilt workflow circuit"),
    }
}

fn circuit_contains_symbol(circuit: &Circuit, symbol: &str) -> bool {
    if circuit.global_phase().get_symbols().contains(symbol) {
        return true;
    }

    circuit
        .operations()
        .iter()
        .flat_map(|operation| operation.params.iter())
        .map(|param| operation_parameter(circuit, param))
        .any(|parameter| parameter.get_symbols().contains(symbol))
}

#[test]
fn normal_workflow_cancels_adjacent_self_inverse_gates() {
    let q0 = Qubit::new(0);
    let mut circuit = Circuit::new(1);
    circuit.h(q0).unwrap();
    circuit.h(q0).unwrap();

    let result = run_workflow(&circuit, CompileMode::Normal);

    assert!(result.changed);
    assert_eq!(result.mode, CompileMode::Normal);
    assert!(result.circuit.operations().is_empty());
}

#[test]
fn normal_workflow_reports_no_change_for_stable_circuit() {
    let q0 = Qubit::new(0);
    let mut circuit = Circuit::new(1);
    circuit.h(q0).unwrap();

    let result = run_workflow(&circuit, CompileMode::Normal);

    assert!(!result.changed);
    assert_eq!(standard_ops(&result.circuit), vec![StandardGate::H]);
}

#[test]
fn normal_workflow_keeps_measurement_circuit_through_definition_stage() {
    let mut circuit = Circuit::new(2);
    circuit
        .measure_bits([Qubit::new(0), Qubit::new(1)])
        .unwrap();

    let result = run_workflow(&circuit, CompileMode::Normal);

    assert_eq!(result.circuit.operations().len(), 1);
    assert_eq!(
        result.circuit.classical_values(),
        circuit.classical_values()
    );
    assert!(
        result
            .steps
            .iter()
            .any(|step| step.name == "decompose.definitions" && !step.changed)
    );
}

#[test]
fn normal_workflow_reports_staged_order() {
    let mut circuit = Circuit::new(1);
    circuit.h(Qubit::new(0)).unwrap();

    let result = CompilerWorkflow::new(compile_config(CompileMode::Normal))
        .run(&circuit)
        .unwrap();

    assert_eq!(
        result
            .steps
            .iter()
            .map(|step| step.name)
            .collect::<Vec<_>>(),
        vec![
            "resolve.target",
            "validate.resources",
            "canonicalize.input",
            "decompose.definitions",
            "optimize.pre_decomposition",
            "decompose.unitary",
            "decompose.mc_gates",
            "canonicalize.after_decomposition",
            "resynthesize.two_qubit_blocks",
            "optimize.post_decomposition",
            "decompose.routing_basis",
            "route.sabre",
            "translate.target_basis",
            "canonicalize.output",
        ]
    );
    assert!(result.steps[10].skipped);
    assert!(result.steps[11].skipped);
    assert!(result.steps[12].skipped);
}

#[test]
fn enhanced_workflow_uses_richer_stage_sequence() {
    let mut circuit = Circuit::new(1);
    circuit.rz(Qubit::new(0), 0.25).unwrap();
    circuit.rz(Qubit::new(0), 0.5).unwrap();
    circuit.rz(Qubit::new(0), -0.75).unwrap();

    let normal = run_workflow(&circuit, CompileMode::Normal);
    let enhanced = run_workflow(&circuit, CompileMode::Enhanced);

    assert!(enhanced.changed);
    assert!(enhanced.steps.len() > normal.steps.len());
    assert_eq!(
        enhanced
            .steps
            .iter()
            .map(|step| step.name)
            .collect::<Vec<_>>(),
        vec![
            "resolve.target",
            "validate.resources",
            "canonicalize.input",
            "decompose.definitions",
            "optimize.pre_decomposition",
            "decompose.unitary",
            "decompose.mc_gates",
            "canonicalize.after_decomposition",
            "resynthesize.two_qubit_blocks",
            "optimize.post_decomposition",
            "decompose.routing_basis",
            "route.sabre",
            "resynthesize.two_qubit_blocks.post_routing",
            "optimize.post_routing",
            "translate.target_basis",
            "optimize.target_cleanup",
            "canonicalize.output",
        ]
    );
    assert!(enhanced.steps[10].skipped);
    assert!(enhanced.steps[11].skipped);
    assert!(enhanced.steps[12].skipped);
    assert!(enhanced.steps[13].skipped);
    assert!(enhanced.steps[14].skipped);
    assert!(enhanced.circuit.operations().is_empty());
}

#[test]
fn workflow_expands_circuit_gate_definitions_before_optimization() {
    let q0 = Qubit::new(0);
    let mut definition = Circuit::new(1);
    definition.h(q0).unwrap();
    let gate = CircuitGate::new("H_DEF", FrozenCircuit::new(definition)).unwrap();

    let mut circuit = Circuit::new(1);
    circuit
        .circuit_gate(gate, vec![q0], Vec::<ParameterValue>::new())
        .unwrap();

    let result = run_workflow(&circuit, CompileMode::Normal);

    assert!(step_changed(&result, "decompose.definitions"));
    assert_eq!(standard_ops(&result.circuit), vec![StandardGate::H]);
    assert!(!contains_high_level_gate(&result.circuit));
}

#[test]
fn workflow_synthesizes_matrix_backed_unitary_gates() {
    let q0 = Qubit::new(0);
    let matrix = array![
        [Complex64::new(0.0, 0.0), Complex64::new(1.0, 0.0)],
        [Complex64::new(1.0, 0.0), Complex64::new(0.0, 0.0)],
    ];
    let gate = UnitaryGate::new("X_MATRIX", 1, 0)
        .with_matrix(matrix)
        .unwrap();
    let mut circuit = Circuit::new(1);
    circuit.unitary(gate, vec![q0]).unwrap();

    let result = run_workflow(&circuit, CompileMode::Normal);

    assert!(step_changed(&result, "decompose.unitary"));
    assert!(!contains_high_level_gate(&result.circuit));
    assert!(!result.circuit.operations().is_empty());
}

#[test]
fn workflow_uses_cx_kak_for_cx_target() {
    let gate = UnitaryGate::new("SWAP_MATRIX", 2, 0)
        .with_matrix(StandardGate::SWAP.matrix(&[]).unwrap().into_owned())
        .unwrap();
    let mut circuit = Circuit::new(2);
    circuit
        .unitary(gate, vec![Qubit::new(0), Qubit::new(1)])
        .unwrap();
    let config = CompileConfig {
        target_basis: Some(vec![
            Instruction::Standard(StandardGate::U),
            Instruction::Standard(StandardGate::CX),
        ]),
        ..compile_config(CompileMode::Normal)
    };

    let result = CompilerWorkflow::new(config).run(&circuit).unwrap();
    let ops = standard_ops(&result.circuit);

    assert!(step_changed(&result, "decompose.unitary"));
    assert_eq!(
        ops.iter().filter(|gate| **gate == StandardGate::CX).count(),
        3
    );
    assert!(!ops.contains(&StandardGate::CZ));
    assert!(!ops.contains(&StandardGate::RXX));
    assert!(!ops.contains(&StandardGate::RYY));
    assert!(!ops.contains(&StandardGate::RZZ));
    assert_compiled_circuit_equivalent(&result.circuit, &circuit);
}

#[test]
fn workflow_runs_two_qubit_resynthesis_after_decomposition() {
    let gate = UnitaryGate::new("SWAP_MATRIX", 2, 0)
        .with_matrix(StandardGate::SWAP.matrix(&[]).unwrap().into_owned())
        .unwrap();
    let mut circuit = Circuit::new(2);
    circuit
        .unitary(gate.clone(), vec![Qubit::new(0), Qubit::new(1)])
        .unwrap();
    circuit
        .unitary(gate, vec![Qubit::new(0), Qubit::new(1)])
        .unwrap();
    let config = CompileConfig {
        target_basis: Some(vec![
            Instruction::Standard(StandardGate::U),
            Instruction::Standard(StandardGate::CX),
        ]),
        ..compile_config(CompileMode::Normal)
    };

    let result = CompilerWorkflow::new(config).run(&circuit).unwrap();
    let ops = standard_ops(&result.circuit);

    assert!(step_changed(&result, "resynthesize.two_qubit_blocks"));
    assert!(
        ops.iter().filter(|gate| **gate == StandardGate::CX).count() < 6,
        "resynthesis should reduce the six-CX decomposition of two SWAPs"
    );
    assert_compiled_circuit_equivalent(&result.circuit, &circuit);
}

#[test]
fn workflow_uses_cz_kak_for_cz_target() {
    let gate = UnitaryGate::new("SWAP_MATRIX", 2, 0)
        .with_matrix(StandardGate::SWAP.matrix(&[]).unwrap().into_owned())
        .unwrap();
    let mut circuit = Circuit::new(2);
    circuit
        .unitary(gate, vec![Qubit::new(0), Qubit::new(1)])
        .unwrap();
    let config = CompileConfig {
        target_basis: Some(vec![
            Instruction::Standard(StandardGate::U),
            Instruction::Standard(StandardGate::CZ),
        ]),
        ..compile_config(CompileMode::Normal)
    };

    let result = CompilerWorkflow::new(config).run(&circuit).unwrap();
    let ops = standard_ops(&result.circuit);

    assert!(step_changed(&result, "decompose.unitary"));
    assert_eq!(
        ops.iter().filter(|gate| **gate == StandardGate::CZ).count(),
        3
    );
    assert!(!ops.contains(&StandardGate::CX));
    assert!(!ops.contains(&StandardGate::RXX));
    assert!(!ops.contains(&StandardGate::RYY));
    assert!(!ops.contains(&StandardGate::RZZ));
    assert_compiled_circuit_equivalent(&result.circuit, &circuit);
}

#[test]
fn workflow_uses_cy_kak_for_cy_target() {
    let gate = UnitaryGate::new("SWAP_MATRIX", 2, 0)
        .with_matrix(StandardGate::SWAP.matrix(&[]).unwrap().into_owned())
        .unwrap();
    let mut circuit = Circuit::new(2);
    circuit
        .unitary(gate, vec![Qubit::new(0), Qubit::new(1)])
        .unwrap();
    let config = CompileConfig {
        target_basis: Some(vec![
            Instruction::Standard(StandardGate::U),
            Instruction::Standard(StandardGate::CY),
        ]),
        ..compile_config(CompileMode::Normal)
    };

    let result = CompilerWorkflow::new(config).run(&circuit).unwrap();
    let ops = standard_ops(&result.circuit);

    assert!(step_changed(&result, "decompose.unitary"));
    assert_eq!(
        ops.iter().filter(|gate| **gate == StandardGate::CY).count(),
        3
    );
    assert!(!ops.contains(&StandardGate::CX));
    assert!(!ops.contains(&StandardGate::CZ));
    assert!(!ops.contains(&StandardGate::RXX));
    assert!(!ops.contains(&StandardGate::RYY));
    assert!(!ops.contains(&StandardGate::RZZ));
    assert_compiled_circuit_equivalent(&result.circuit, &circuit);
}

#[test]
fn workflow_uses_rzz_kak_for_rzz_only_entangler_target() {
    let gate = UnitaryGate::new("SWAP_MATRIX", 2, 0)
        .with_matrix(StandardGate::SWAP.matrix(&[]).unwrap().into_owned())
        .unwrap();
    let mut circuit = Circuit::new(2);
    circuit
        .unitary(gate, vec![Qubit::new(0), Qubit::new(1)])
        .unwrap();
    let config = CompileConfig {
        target_basis: Some(vec![
            Instruction::Standard(StandardGate::U),
            Instruction::Standard(StandardGate::H),
            Instruction::Standard(StandardGate::RX),
            Instruction::Standard(StandardGate::RZZ),
        ]),
        ..compile_config(CompileMode::Normal)
    };

    let result = CompilerWorkflow::new(config).run(&circuit).unwrap();
    let ops = standard_ops(&result.circuit);

    assert!(step_changed(&result, "decompose.unitary"));
    assert_eq!(
        ops.iter()
            .filter(|gate| **gate == StandardGate::RZZ)
            .count(),
        3
    );
    assert!(!ops.contains(&StandardGate::CX));
    assert!(!ops.contains(&StandardGate::CY));
    assert!(!ops.contains(&StandardGate::CZ));
    assert!(!ops.contains(&StandardGate::RXX));
    assert!(!ops.contains(&StandardGate::RYY));
    assert_compiled_circuit_equivalent(&result.circuit, &circuit);
}

#[test]
fn workflow_keeps_pauli_kak_for_full_ising_target() {
    let gate = UnitaryGate::new("SWAP_MATRIX", 2, 0)
        .with_matrix(StandardGate::SWAP.matrix(&[]).unwrap().into_owned())
        .unwrap();
    let mut circuit = Circuit::new(2);
    circuit
        .unitary(gate, vec![Qubit::new(0), Qubit::new(1)])
        .unwrap();
    let config = CompileConfig {
        target_basis: Some(vec![
            Instruction::Standard(StandardGate::U),
            Instruction::Standard(StandardGate::RXX),
            Instruction::Standard(StandardGate::RYY),
            Instruction::Standard(StandardGate::RZZ),
        ]),
        ..compile_config(CompileMode::Normal)
    };

    let result = CompilerWorkflow::new(config).run(&circuit).unwrap();
    let ops = standard_ops(&result.circuit);

    assert!(step_changed(&result, "decompose.unitary"));
    assert!(!ops.contains(&StandardGate::CX));
    assert!(!ops.contains(&StandardGate::CY));
    assert!(!ops.contains(&StandardGate::CZ));
    assert!(ops.contains(&StandardGate::RXX));
    assert!(ops.contains(&StandardGate::RYY));
    assert!(ops.contains(&StandardGate::RZZ));
    assert_compiled_circuit_equivalent(&result.circuit, &circuit);
}

#[test]
fn workflow_keeps_pauli_kak_for_partial_ising_target() {
    let rxx = StandardGate::RXX.matrix(&[0.7]).unwrap().into_owned();
    let ryy = StandardGate::RYY.matrix(&[-0.4]).unwrap().into_owned();
    let matrix = rxx.dot(&ryy);
    let gate = UnitaryGate::new("PARTIAL_ISING_MATRIX", 2, 0)
        .with_matrix(matrix)
        .unwrap();
    let mut circuit = Circuit::new(2);
    circuit
        .unitary(gate, vec![Qubit::new(0), Qubit::new(1)])
        .unwrap();
    let config = CompileConfig {
        target_basis: Some(vec![
            Instruction::Standard(StandardGate::U),
            Instruction::Standard(StandardGate::H),
            Instruction::Standard(StandardGate::RX),
            Instruction::Standard(StandardGate::RXX),
            Instruction::Standard(StandardGate::RZZ),
        ]),
        ..compile_config(CompileMode::Normal)
    };

    let result = CompilerWorkflow::new(config).run(&circuit).unwrap();
    let ops = standard_ops(&result.circuit);

    assert!(step_changed(&result, "decompose.unitary"));
    assert!(ops.contains(&StandardGate::RZZ));
    assert!(!ops.contains(&StandardGate::RXX));
    assert!(!ops.contains(&StandardGate::RYY));
    assert!(!ops.contains(&StandardGate::CX));
    assert!(!ops.contains(&StandardGate::CY));
    assert!(!ops.contains(&StandardGate::CZ));
    assert_compiled_circuit_equivalent(&result.circuit, &circuit);
}

#[test]
fn workflow_keeps_pauli_kak_without_target() {
    let gate = UnitaryGate::new("SWAP_MATRIX", 2, 0)
        .with_matrix(StandardGate::SWAP.matrix(&[]).unwrap().into_owned())
        .unwrap();
    let mut circuit = Circuit::new(2);
    circuit
        .unitary(gate, vec![Qubit::new(0), Qubit::new(1)])
        .unwrap();

    let result = run_workflow(&circuit, CompileMode::Normal);
    let ops = standard_ops(&result.circuit);

    assert!(step_changed(&result, "decompose.unitary"));
    assert!(!ops.contains(&StandardGate::CX));
    assert!(!ops.contains(&StandardGate::CY));
    assert!(!ops.contains(&StandardGate::CZ));
    assert!(
        ops.contains(&StandardGate::RXX)
            || ops.contains(&StandardGate::RYY)
            || ops.contains(&StandardGate::RZZ)
    );
    assert_compiled_circuit_equivalent(&result.circuit, &circuit);
}

#[test]
fn workflow_decomposes_multi_controlled_gates() {
    let qubits = (0..4).map(Qubit::new).collect::<Vec<_>>();
    let mut circuit = Circuit::new(4);
    circuit
        .append(
            Instruction::McGate(Box::new(MCGate::new(3, StandardGate::X))),
            qubits.clone(),
            Vec::<ParameterValue>::new(),
            None,
        )
        .unwrap();

    let result = run_workflow(&circuit, CompileMode::Normal);

    assert!(step_changed(&result, "decompose.mc_gates"));
    assert!(!contains_high_level_gate(&result.circuit));
}

#[test]
fn workflow_reports_rewrite_change_for_symbolic_merge() {
    let q0 = Qubit::new(0);
    let theta = Parameter::symbol("theta");
    let mut circuit = Circuit::new(1);
    circuit.rz(q0, theta.clone()).unwrap();
    circuit.rz(q0, 0.5).unwrap();

    let result = run_workflow(&circuit, CompileMode::Normal);

    assert!(
        step_changed(&result, "optimize.pre_decomposition")
            || step_changed(&result, "optimize.post_decomposition")
    );
    assert_eq!(result.circuit.operations().len(), 1);
    let merged = operation_parameter(&result.circuit, &result.circuit.operations()[0].params[0]);
    assert!(merged.provably_equal(&(theta.clone() + Parameter::from(0.5)), 1e-12));
    assert_bindings_preserve_semantics(
        &circuit,
        &result.circuit,
        &[
            binding_case(&[("theta", 0.0)]),
            binding_case(&[("theta", 0.25)]),
            binding_case(&[("theta", -PI / 4.0)]),
        ],
    );
}

#[test]
fn workflow_decomposes_parameterized_mc_gate() {
    let qubits = (0..3).map(Qubit::new).collect::<Vec<_>>();
    let theta = Parameter::symbol("theta");
    let mut circuit = Circuit::new(3);
    circuit
        .append(
            Instruction::McGate(Box::new(MCGate::new(2, StandardGate::RZ))),
            qubits,
            vec![ParameterValue::Param(theta.clone())],
            None,
        )
        .unwrap();

    let result = run_workflow(&circuit, CompileMode::Normal);

    assert!(step_changed(&result, "decompose.mc_gates"));
    assert!(!contains_high_level_gate(&result.circuit));
    assert!(circuit_contains_symbol(&result.circuit, "theta"));
    assert_bindings_preserve_semantics(
        &circuit,
        &result.circuit,
        &[
            binding_case(&[("theta", 0.0)]),
            binding_case(&[("theta", 0.31)]),
            binding_case(&[("theta", PI / 3.0)]),
        ],
    );
}

#[test]
fn workflow_routes_parameterized_circuit_when_device_present() {
    let q0 = Qubit::new(0);
    let q2 = Qubit::new(2);
    let theta = Parameter::symbol("theta");
    let phi = Parameter::symbol("phi");
    let mut circuit = Circuit::new(3);
    circuit.rx(q0, theta.clone()).unwrap();
    circuit.rz(q2, phi.clone()).unwrap();
    circuit.cx(q0, q2).unwrap();

    let result = CompilerWorkflow::new(CompileConfig {
        mode: CompileMode::Normal,
        target_basis: None,
        device: Some(Device::line("workflow-param-line", 3).unwrap()),
        initial_layout: None,
        resource_policy: ResourcePolicy::default(),
        seed: Some(11),
    })
    .run(&circuit)
    .unwrap();

    assert!(step_changed(&result, "route.sabre"));
    assert!(circuit_contains_symbol(&result.circuit, "theta"));
    assert!(circuit_contains_symbol(&result.circuit, "phi"));
    assert!(result.circuit.operations().iter().any(|operation| {
        matches!(
            operation.instruction,
            Instruction::Standard(StandardGate::RX)
        ) && matches!(operation.params.as_slice(), [CircuitParam::Index(_)])
    }));
    assert!(result.circuit.operations().iter().any(|operation| {
        matches!(
            operation.instruction,
            Instruction::Standard(StandardGate::RZ)
        ) && matches!(operation.params.as_slice(), [CircuitParam::Index(_)])
    }));
}

#[test]
fn workflow_routes_from_supplied_initial_layout() {
    let mut circuit = Circuit::new(1);
    circuit.h(Qubit::new(0)).unwrap();
    let layout = Layout::from_pairs(&[(0, 2)], 3).unwrap();

    let result = CompilerWorkflow::new(CompileConfig {
        mode: CompileMode::Normal,
        target_basis: None,
        device: Some(Device::line("layout-line", 3).unwrap()),
        initial_layout: Some(layout),
        resource_policy: ResourcePolicy::default(),
        seed: Some(17),
    })
    .run(&circuit)
    .unwrap();

    let route = result
        .steps
        .iter()
        .find(|step| step.name == "route.sabre")
        .unwrap();
    assert!(!route.skipped);
    assert!(route.changed);
    assert!(
        route
            .reason
            .as_deref()
            .is_some_and(|reason| reason.contains("supplied initial layout"))
    );
    assert_eq!(
        result.circuit.operations()[0].qubits.as_slice(),
        &[Qubit::new(2)]
    );
}

#[test]
fn workflow_rejects_initial_layout_without_device() {
    let mut circuit = Circuit::new(1);
    circuit.h(Qubit::new(0)).unwrap();
    let layout = Layout::from_pairs(&[(0, 0)], 1).unwrap();

    let err = CompilerWorkflow::new(CompileConfig {
        mode: CompileMode::Normal,
        target_basis: None,
        device: None,
        initial_layout: Some(layout),
        resource_policy: ResourcePolicy::default(),
        seed: None,
    })
    .run(&circuit)
    .unwrap_err();

    assert!(matches!(
        err,
        CompilerError::InvalidInput(reason) if reason.contains("initial layout requires a target device")
    ));
}

#[test]
fn workflow_target_translation_keeps_parameterized_semantics() {
    let q0 = Qubit::new(0);
    let q1 = Qubit::new(1);
    let theta = Parameter::symbol("theta");
    let mut circuit = Circuit::new(2);
    circuit.h(q0).unwrap();
    circuit.crz(q0, q1, theta.clone()).unwrap();

    let result = CompilerWorkflow::new(CompileConfig {
        mode: CompileMode::Normal,
        target_basis: Some(vec![
            Instruction::Standard(StandardGate::RZ),
            Instruction::Standard(StandardGate::X2P),
            Instruction::Standard(StandardGate::X2M),
            Instruction::Standard(StandardGate::Y2P),
            Instruction::Standard(StandardGate::Y2M),
            Instruction::Standard(StandardGate::CZ),
            Instruction::Standard(StandardGate::GPhase),
        ]),
        device: None,
        initial_layout: None,
        resource_policy: ResourcePolicy::default(),
        seed: None,
    })
    .run(&circuit)
    .unwrap();

    assert!(step_changed(&result, "translate.target_basis"));
    assert!(circuit_contains_symbol(&result.circuit, "theta"));
    assert!(standard_ops(&result.circuit).iter().all(|gate| matches!(
        gate,
        StandardGate::RZ
            | StandardGate::X2P
            | StandardGate::X2M
            | StandardGate::Y2P
            | StandardGate::Y2M
            | StandardGate::CZ
            | StandardGate::GPhase
    )));
    assert_bindings_preserve_semantics(
        &circuit,
        &result.circuit,
        &[
            binding_case(&[("theta", 0.0)]),
            binding_case(&[("theta", 0.21)]),
            binding_case(&[("theta", -PI / 5.0)]),
        ],
    );
}

#[test]
fn target_basis_translation_runs_after_definition_decomposition() {
    let q0 = Qubit::new(0);
    let q1 = Qubit::new(1);
    let mut definition = Circuit::new(2);
    definition.cx(q0, q1).unwrap();
    let gate = CircuitGate::new("CX_DEF", FrozenCircuit::new(definition)).unwrap();
    let mut circuit = Circuit::new(2);
    circuit
        .circuit_gate(gate, vec![q0, q1], Vec::<ParameterValue>::new())
        .unwrap();

    let result = CompilerWorkflow::new(CompileConfig {
        mode: CompileMode::Normal,
        target_basis: Some(vec![
            Instruction::Standard(StandardGate::H),
            Instruction::Standard(StandardGate::CZ),
        ]),
        device: None,
        initial_layout: None,
        resource_policy: ResourcePolicy::default(),
        seed: None,
    })
    .run(&circuit)
    .unwrap();

    assert!(step_changed(&result, "decompose.definitions"));
    assert!(step_changed(&result, "translate.target_basis"));
    assert_eq!(
        standard_ops(&result.circuit),
        vec![StandardGate::H, StandardGate::CZ, StandardGate::H]
    );
}

#[test]
fn explicit_target_basis_runs_lowering() {
    let q0 = Qubit::new(0);
    let q1 = Qubit::new(1);
    let mut circuit = Circuit::new(2);
    circuit.cx(q0, q1).unwrap();

    let result = CompilerWorkflow::new(CompileConfig {
        mode: CompileMode::Normal,
        target_basis: Some(vec![
            Instruction::Standard(StandardGate::H),
            Instruction::Standard(StandardGate::CZ),
        ]),
        device: None,
        initial_layout: None,
        resource_policy: ResourcePolicy::default(),
        seed: None,
    })
    .run(&circuit)
    .unwrap();

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
fn target_basis_failure_is_reported() {
    let q0 = Qubit::new(0);
    let mut circuit = Circuit::new(1);
    circuit.h(q0).unwrap();

    let err = CompilerWorkflow::new(CompileConfig {
        mode: CompileMode::Normal,
        target_basis: Some(vec![Instruction::Standard(StandardGate::CZ)]),
        device: None,
        initial_layout: None,
        resource_policy: ResourcePolicy::default(),
        seed: None,
    })
    .run(&circuit)
    .unwrap_err();

    assert!(matches!(err, CompilerError::InvalidInput(_)));
}

#[test]
fn mc_gate_target_basis_is_rejected_by_workflow_contract() {
    let mut circuit = Circuit::new(2);
    circuit.cx(Qubit::new(0), Qubit::new(1)).unwrap();

    let err = CompilerWorkflow::new(CompileConfig {
        mode: CompileMode::Normal,
        target_basis: Some(vec![Instruction::McGate(Box::new(MCGate::new(
            1,
            StandardGate::X,
        )))]),
        device: None,
        initial_layout: None,
        resource_policy: ResourcePolicy::default(),
        seed: None,
    })
    .run(&circuit)
    .unwrap_err();

    assert!(matches!(err, CompilerError::InvalidInput(_)));
}

#[test]
fn device_native_gates_are_used_as_target_basis() {
    let q0 = Qubit::new(0);
    let q1 = Qubit::new(1);
    let mut circuit = Circuit::new(2);
    circuit.cx(q0, q1).unwrap();
    let device = two_qubit_device(vec![
        Instruction::Standard(StandardGate::H),
        Instruction::Standard(StandardGate::CZ),
    ]);

    let result = CompilerWorkflow::new(CompileConfig {
        mode: CompileMode::Enhanced,
        target_basis: None,
        device: Some(device),
        initial_layout: None,
        resource_policy: ResourcePolicy::default(),
        seed: None,
    })
    .run(&circuit)
    .unwrap();

    assert_eq!(
        standard_ops(&result.circuit),
        vec![StandardGate::H, StandardGate::CZ, StandardGate::H]
    );
}

#[test]
fn device_workflow_routes_circuit_before_target_translation() {
    let q0 = Qubit::new(0);
    let q1 = Qubit::new(1);
    let q2 = Qubit::new(2);
    let mut circuit = Circuit::new(3);
    circuit.cx(q0, q1).unwrap();
    circuit.cx(q1, q2).unwrap();
    circuit.cx(q0, q2).unwrap();
    let device = Device::line("test-device", 3).unwrap();

    let result = CompilerWorkflow::new(CompileConfig {
        mode: CompileMode::Normal,
        target_basis: None,
        device: Some(device),
        initial_layout: None,
        resource_policy: ResourcePolicy::default(),
        seed: Some(7),
    })
    .run(&circuit)
    .unwrap();

    assert!(step_changed(&result, "route.sabre"));
    assert!(standard_ops(&result.circuit).contains(&StandardGate::SWAP));
    assert!(
        result
            .steps
            .iter()
            .find(|step| step.name == "route.sabre")
            .is_some_and(|step| !step.skipped)
    );
}

#[test]
fn routed_swaps_are_lowered_to_device_native_basis() {
    let q0 = Qubit::new(0);
    let q1 = Qubit::new(1);
    let q2 = Qubit::new(2);
    let mut circuit = Circuit::new(3);
    circuit.cx(q0, q1).unwrap();
    circuit.cx(q1, q2).unwrap();
    circuit.cx(q0, q2).unwrap();
    let device = Device::line("test-device", 3)
        .unwrap()
        .with_native_gates(vec![
            Instruction::Standard(StandardGate::H),
            Instruction::Standard(StandardGate::CZ),
        ])
        .unwrap();

    let result = CompilerWorkflow::new(CompileConfig {
        mode: CompileMode::Normal,
        target_basis: None,
        device: Some(device),
        initial_layout: None,
        resource_policy: ResourcePolicy::default(),
        seed: Some(7),
    })
    .run(&circuit)
    .unwrap();

    assert!(step_changed(&result, "route.sabre"));
    assert!(step_changed(&result, "translate.target_basis"));
    assert!(
        standard_ops(&result.circuit)
            .iter()
            .all(|gate| matches!(gate, StandardGate::H | StandardGate::CZ))
    );
}

#[test]
fn routed_swaps_are_lowered_to_qcis_native_subset() {
    let q0 = Qubit::new(0);
    let q1 = Qubit::new(1);
    let q2 = Qubit::new(2);
    let mut circuit = Circuit::new(3);
    circuit.cx(q0, q1).unwrap();
    circuit.cx(q1, q2).unwrap();
    circuit.cx(q0, q2).unwrap();
    let device = Device::line("test-device", 3)
        .unwrap()
        .with_native_gates(vec![
            Instruction::Standard(StandardGate::RZ),
            Instruction::Standard(StandardGate::X2P),
            Instruction::Standard(StandardGate::CZ),
        ])
        .unwrap();

    let result = CompilerWorkflow::new(CompileConfig {
        mode: CompileMode::Normal,
        target_basis: None,
        device: Some(device),
        initial_layout: None,
        resource_policy: ResourcePolicy::default(),
        seed: Some(7),
    })
    .run(&circuit)
    .unwrap();

    let routing_basis_step = result
        .steps
        .iter()
        .find(|step| step.name == "decompose.routing_basis")
        .expect("workflow should report routing-basis decomposition");
    assert!(!routing_basis_step.skipped);
    assert!(!routing_basis_step.changed);
    assert!(step_changed(&result, "route.sabre"));
    assert!(step_changed(&result, "translate.target_basis"));
    assert!(standard_ops(&result.circuit).iter().all(|gate| matches!(
        gate,
        StandardGate::RZ | StandardGate::X2P | StandardGate::CZ
    )));
    assert!(!standard_ops(&result.circuit).contains(&StandardGate::SWAP));
}

#[test]
fn enhanced_device_workflow_runs_post_routing_cleanup() {
    let q0 = Qubit::new(0);
    let q1 = Qubit::new(1);
    let q2 = Qubit::new(2);
    let mut circuit = Circuit::new(3);
    circuit.cx(q0, q1).unwrap();
    circuit.cx(q1, q2).unwrap();
    circuit.cx(q0, q2).unwrap();
    let device = Device::line("test-device", 3).unwrap();

    let result = CompilerWorkflow::new(CompileConfig {
        mode: CompileMode::Enhanced,
        target_basis: None,
        device: Some(device),
        initial_layout: None,
        resource_policy: ResourcePolicy::default(),
        seed: Some(7),
    })
    .run(&circuit)
    .unwrap();

    let post_routing = result
        .steps
        .iter()
        .find(|step| step.name == "optimize.post_routing")
        .unwrap();
    assert!(!post_routing.skipped);
}

#[test]
fn device_capacity_blocks_clean_ancilla_allocation_but_allows_no_aux_fallback() {
    let qubits = (0..4).map(Qubit::new).collect::<Vec<_>>();
    let mut circuit = Circuit::new(4);
    circuit
        .append(
            Instruction::McGate(Box::new(MCGate::new(3, StandardGate::X))),
            qubits,
            Vec::<ParameterValue>::new(),
            None,
        )
        .unwrap();
    let device = Device::line("test-device", 4).unwrap();

    let result = CompilerWorkflow::new(CompileConfig {
        mode: CompileMode::Normal,
        target_basis: None,
        device: Some(device),
        initial_layout: None,
        resource_policy: ResourcePolicy {
            max_pre_layout_clean_ancillas: 2,
            allow_dirty_borrowing: false,
        },
        seed: None,
    })
    .run(&circuit)
    .unwrap();

    assert!(step_changed(&result, "decompose.mc_gates"));
    assert_eq!(result.circuit.qubits().len(), 4);
    assert!(!contains_high_level_gate(&result.circuit));
}

#[test]
fn device_capacity_rejects_source_circuit_that_is_too_wide() {
    let mut circuit = Circuit::new(3);
    circuit.h(Qubit::new(0)).unwrap();
    let device = Device::line("test-device", 2).unwrap();

    let err = CompilerWorkflow::new(CompileConfig {
        mode: CompileMode::Normal,
        target_basis: None,
        device: Some(device),
        initial_layout: None,
        resource_policy: ResourcePolicy::default(),
        seed: None,
    })
    .run(&circuit)
    .unwrap_err();

    assert!(matches!(
        err,
        CompilerError::InvalidInput(reason) if reason.contains("source circuit uses 3 logical qubits")
    ));
}

#[test]
fn device_capacity_rejects_too_wide_source_before_mc_decomposition() {
    let qubits = (0..3).map(Qubit::new).collect::<Vec<_>>();
    let mut circuit = Circuit::new(3);
    circuit
        .append(
            Instruction::McGate(Box::new(MCGate::new(2, StandardGate::X))),
            qubits,
            Vec::<ParameterValue>::new(),
            None,
        )
        .unwrap();
    let device = Device::line("test-device", 2).unwrap();

    let err = CompilerWorkflow::new(CompileConfig {
        mode: CompileMode::Normal,
        target_basis: None,
        device: Some(device),
        initial_layout: None,
        resource_policy: ResourcePolicy::default(),
        seed: None,
    })
    .run(&circuit)
    .unwrap_err();

    assert!(matches!(
        err,
        CompilerError::InvalidInput(reason) if reason.contains("source circuit uses 3 logical qubits")
    ));
}

#[test]
fn compile_matches_built_workflow() {
    let q0 = Qubit::new(0);
    let q1 = Qubit::new(1);
    let mut circuit = Circuit::new(2);
    circuit.h(q0).unwrap();
    circuit.x(q1).unwrap();
    circuit.h(q0).unwrap();

    let direct = compile(&circuit, compile_config(CompileMode::Normal)).unwrap();
    let built = CompilerWorkflow::new(compile_config(CompileMode::Normal))
        .run(&circuit)
        .unwrap();

    assert_eq!(direct.changed, built.changed);
    assert_eq!(standard_ops(&direct.circuit), standard_ops(&built.circuit));
    assert_eq!(direct.steps, built.steps);
}

#[test]
fn workflow_config_can_build_enhanced_workflow() {
    let workflow = CompilerWorkflow::new(compile_config(CompileMode::Enhanced));

    assert_eq!(workflow.config().mode, CompileMode::Enhanced);
}

#[test]
fn target_basis_constraints_only_apply_to_target_cleanup_rewrite_phase() {
    let workflow = CompilerWorkflow::new(compile_config(CompileMode::Normal));
    let target_basis = vec![
        Instruction::Standard(StandardGate::RZ),
        Instruction::Standard(StandardGate::X2P),
        Instruction::Standard(StandardGate::CZ),
    ];
    let state = workflow_state_with_target_basis(target_basis.clone());

    for phase in [
        RewritePhase::PreDecomposition,
        RewritePhase::PostDecomposition,
        RewritePhase::PostRouting,
    ] {
        let config = workflow.rewrite_config_for_state(phase, &state).unwrap();
        assert!(config.target_instruction_basis().is_none());
    }

    let cleanup_config = workflow
        .rewrite_config_for_state(RewritePhase::TargetCleanup, &state)
        .unwrap();
    let cleanup_basis = cleanup_config.target_instruction_basis().unwrap();
    assert_eq!(cleanup_basis.len(), target_basis.len());
    assert!(matches!(
        cleanup_basis.as_slice(),
        [
            Instruction::Standard(StandardGate::RZ),
            Instruction::Standard(StandardGate::X2P),
            Instruction::Standard(StandardGate::CZ)
        ]
    ));
}
