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

//! Shared helpers for crate-local tests.

use crate::circuit::{
    Circuit, Instruction, MCGate, ParameterValue, Qubit, StandardGate, ValueInstruction,
    circuit_to_matrix, operation::ValueOperation,
};
use crate::compile::CompileResult;
use crate::compile::transform::decompose::mc_gate::Su2RotationAxis;
use crate::device::{Device, PhysicalQubit, Topology};
use crate::qis::Statevector;
use ndarray::Array2;
use num_complex::Complex64;
use proptest::prelude::*;
use std::collections::HashSet;

/// Tolerance for floating-point comparisons in mc_gate decomposition tests.
pub const EPSILON: f64 = 1e-9;

/// Asserts that an operation is a parameter-free, unlabeled standard gate.
pub fn assert_standard_operation(
    operation: &ValueOperation,
    expected_gate: StandardGate,
    expected_qubits: &[Qubit],
) {
    assert!(matches!(
        operation.instruction,
        ValueInstruction::Instruction(Instruction::Standard(gate)) if gate == expected_gate
    ));
    assert_eq!(operation.qubits.as_slice(), expected_qubits);
    assert!(operation.params.is_empty());
    assert!(operation.label.is_none());
}

/// Asserts equality for value-operation sequences.
pub fn assert_value_operations_equal(actual: &[ValueOperation], expected: &[ValueOperation]) {
    assert_eq!(actual.len(), expected.len());
    for (actual_operation, expected_operation) in actual.iter().zip(expected) {
        match (
            &actual_operation.instruction,
            &expected_operation.instruction,
        ) {
            (
                ValueInstruction::Instruction(Instruction::Standard(actual_gate)),
                ValueInstruction::Instruction(Instruction::Standard(expected_gate)),
            ) => {
                assert_eq!(actual_gate, expected_gate);
            }
            (actual_instruction, expected_instruction) => {
                panic!(
                    "instruction mismatch: actual={actual_instruction:?}, expected={expected_instruction:?}"
                );
            }
        }
        assert_eq!(actual_operation.qubits, expected_operation.qubits);
        assert_eq!(
            actual_operation.params.len(),
            expected_operation.params.len()
        );
        for (actual_parameter, expected_parameter) in actual_operation
            .params
            .iter()
            .zip(&expected_operation.params)
        {
            match (actual_parameter, expected_parameter) {
                (ParameterValue::Fixed(actual), ParameterValue::Fixed(expected)) => {
                    assert_eq!(actual.to_bits(), expected.to_bits());
                }
                (ParameterValue::Param(actual), ParameterValue::Param(expected)) => {
                    assert_eq!(actual, expected);
                }
                _ => panic!(
                    "parameter mismatch: actual={actual_parameter:?}, expected={expected_parameter:?}"
                ),
            }
        }
        assert_eq!(actual_operation.label, expected_operation.label);
    }
}

/// Asserts that every operation references only qubits from the allowed set.
pub fn assert_value_operations_only_use_qubits(
    operations: &[ValueOperation],
    allowed_qubits: &[Qubit],
) {
    let allowed_qubits: HashSet<_> = allowed_qubits.iter().copied().collect();

    for (operation_index, operation) in operations.iter().enumerate() {
        for qubit in &operation.qubits {
            assert!(
                allowed_qubits.contains(qubit),
                "operation {operation_index} references disallowed qubit {qubit}: {operation:?}"
            );
        }
    }
}

/// Builds a circuit from self-contained value operations.
pub fn circuit_from_value_operations(
    num_qubits: usize,
    operations: Vec<ValueOperation>,
) -> Circuit {
    Circuit::from_operations(
        (0..num_qubits)
            .map(|index| Qubit::new(index as u32))
            .collect(),
        operations,
        None,
        None,
    )
    .unwrap()
}

/// Applies value operations to a clone of an initial statevector.
pub fn statevector_after_value_operations(
    initial_state: &Statevector,
    operations: &[ValueOperation],
) -> Statevector {
    let circuit = circuit_from_value_operations(initial_state.num_qubits, operations.to_vec());
    let mut statevector = initial_state.clone();
    statevector.apply_circuit(&circuit).unwrap();
    statevector
}

/// Asserts equality of complete statevectors up to one global phase.
pub fn assert_statevectors_equal_up_to_global_phase(
    actual: &Statevector,
    expected: &Statevector,
    epsilon: f64,
) {
    assert_eq!(actual.num_qubits, expected.num_qubits);
    assert_eq!(actual.data().len(), expected.data().len());

    let reference_index = expected
        .data()
        .iter()
        .position(|amplitude| amplitude.norm() > epsilon)
        .expect("expected statevector must contain a nonzero amplitude");
    let actual_reference = actual.data()[reference_index];
    let expected_reference = expected.data()[reference_index];
    assert!(
        actual_reference.norm() > epsilon,
        "actual statevector has zero amplitude at reference index {reference_index}"
    );

    let global_phase = actual_reference / expected_reference;
    assert!(
        (global_phase.norm() - 1.0).abs() < epsilon,
        "statevectors differ in amplitude magnitude at reference index {reference_index}: actual={actual_reference}, expected={expected_reference}"
    );

    for (index, (actual_amplitude, expected_amplitude)) in
        actual.data().iter().zip(expected.data()).enumerate()
    {
        let phase_adjusted_expected = global_phase * expected_amplitude;
        assert!(
            (*actual_amplitude - phase_adjusted_expected).norm() < epsilon,
            "statevectors differ at index {index}: actual={actual_amplitude}, expected={phase_adjusted_expected}"
        );
    }
}

/// Asserts approximate equality of complete matrices.
pub fn assert_matrix_approx_eq(
    actual: &Array2<Complex64>,
    expected: &Array2<Complex64>,
    epsilon: f64,
) {
    assert_eq!(actual.shape(), expected.shape());
    for ((row, column), expected_amplitude) in expected.indexed_iter() {
        assert!(
            (actual[[row, column]] - expected_amplitude).norm() < epsilon,
            "matrix mismatch at row {row}, column {column}: actual={}, expected={expected_amplitude}",
            actual[[row, column]]
        );
    }
}

/// Asserts that a matrix is unitary: U† * U = I.
pub fn assert_is_unitary(matrix: &Array2<Complex64>, epsilon: f64) {
    assert_eq!(
        matrix.nrows(),
        matrix.ncols(),
        "unitary matrix must be square, got {}x{}",
        matrix.nrows(),
        matrix.ncols()
    );

    let product = matrix.t().mapv(|value| value.conj()).dot(matrix);
    for row in 0..matrix.nrows() {
        for column in 0..matrix.ncols() {
            let expected = if row == column {
                Complex64::new(1.0, 0.0)
            } else {
                Complex64::new(0.0, 0.0)
            };
            let diff = (product[[row, column]] - expected).norm();
            assert!(
                diff < epsilon,
                "matrix is not unitary at row {row}, column {column}: actual={}, expected={expected}, diff={diff}",
                product[[row, column]]
            );
        }
    }
}

/// Returns whether a compiler workflow step changed the circuit.
pub fn step_changed(result: &CompileResult, name: &str) -> bool {
    result
        .steps
        .iter()
        .any(|step| step.name == name && step.changed && !step.skipped)
}

/// Asserts approximate equality for selected matrix columns.
pub fn assert_selected_matrix_columns_approx_eq(
    actual: &Array2<Complex64>,
    expected: &Array2<Complex64>,
    columns: impl IntoIterator<Item = usize>,
    epsilon: f64,
) {
    assert_eq!(actual.shape(), expected.shape());
    for column in columns {
        for row in 0..expected.nrows() {
            assert!(
                (actual[[row, column]] - expected[[row, column]]).norm() < epsilon,
                "matrix mismatch at row {row}, column {column}: actual={}, expected={}",
                actual[[row, column]],
                expected[[row, column]]
            );
        }
    }
}

/// Asserts selected matrix columns are equal up to one global phase.
pub fn assert_selected_matrix_columns_equal_up_to_global_phase(
    actual: &Array2<Complex64>,
    expected: &Array2<Complex64>,
    columns: impl IntoIterator<Item = usize>,
    epsilon: f64,
) {
    assert_eq!(actual.shape(), expected.shape());
    let columns: Vec<_> = columns.into_iter().collect();
    let (reference_actual, reference_expected) = columns
        .iter()
        .flat_map(|column| {
            (0..expected.nrows()).map(move |row| (actual[[row, *column]], expected[[row, *column]]))
        })
        .find(|(_, expected)| expected.norm() > epsilon)
        .expect("selected expected columns must contain a nonzero amplitude");
    let global_phase = reference_actual / reference_expected;

    assert!((global_phase.norm() - 1.0).abs() < epsilon);
    for column in columns {
        for row in 0..expected.nrows() {
            let expected_amplitude = global_phase * expected[[row, column]];
            assert!(
                (actual[[row, column]] - expected_amplitude).norm() < epsilon,
                "matrix mismatch at row {row}, column {column}: actual={}, expected={expected_amplitude}",
                actual[[row, column]]
            );
        }
    }
}

/// Asserts that two matrices are equal up to one global phase.
pub fn assert_matrices_equal_up_to_global_phase(
    actual: &Array2<Complex64>,
    expected: &Array2<Complex64>,
    epsilon: f64,
) {
    assert_eq!(actual.shape(), expected.shape());
    let (reference_actual, reference_expected) = actual
        .iter()
        .zip(expected.iter())
        .find(|(_, expected)| expected.norm() > epsilon)
        .expect("expected matrix must contain a nonzero amplitude");
    assert!(
        reference_actual.norm() > epsilon,
        "actual matrix has zero amplitude where expected matrix is nonzero"
    );

    let global_phase = reference_actual / reference_expected;
    assert!(
        (global_phase.norm() - 1.0).abs() < epsilon,
        "matrices differ in reference amplitude magnitude: actual={reference_actual}, expected={reference_expected}"
    );

    for ((row, column), expected_amplitude) in expected.indexed_iter() {
        let phase_adjusted_expected = global_phase * expected_amplitude;
        assert!(
            (actual[[row, column]] - phase_adjusted_expected).norm() < epsilon,
            "matrix mismatch at row {row}, column {column}: actual={}, expected={phase_adjusted_expected}",
            actual[[row, column]]
        );
    }
}

/// Asserts that two circuits have the same unitary matrix up to global phase.
pub fn assert_circuits_equivalent_up_to_global_phase(
    actual: &Circuit,
    expected: &Circuit,
    epsilon: f64,
) {
    let actual_matrix = circuit_to_matrix(actual, None).unwrap();
    let expected_matrix = circuit_to_matrix(expected, None).unwrap();
    assert_matrices_equal_up_to_global_phase(&actual_matrix, &expected_matrix, epsilon);
}

/// Asserts that a compiled circuit preserves a source circuit's unitary.
pub fn assert_compiled_circuit_equivalent(actual: &Circuit, expected: &Circuit) {
    assert_circuits_equivalent_up_to_global_phase(actual, expected, EPSILON);
}

/// Extracts standard-gate operations from a circuit.
pub fn standard_ops(circuit: &Circuit) -> Vec<StandardGate> {
    circuit
        .operations()
        .iter()
        .filter_map(|operation| match operation.instruction {
            Instruction::Standard(gate) => Some(gate),
            _ => None,
        })
        .collect()
}

/// Builds a Bell-state preparation circuit.
pub fn bell_circuit() -> Circuit {
    let q0 = Qubit::new(0);
    let q1 = Qubit::new(1);
    let mut circuit = Circuit::new(2);
    circuit.h(q0).unwrap();
    circuit.cx(q0, q1).unwrap();
    circuit
}

/// Builds an n-qubit GHZ-state preparation circuit.
pub fn ghz_circuit(num_qubits: usize) -> Circuit {
    assert!(num_qubits >= 2);
    let mut circuit = Circuit::new(num_qubits);
    circuit.h(Qubit::new(0)).unwrap();
    for index in 0..num_qubits - 1 {
        circuit
            .cx(Qubit::new(index as u32), Qubit::new(index as u32 + 1))
            .unwrap();
    }
    circuit
}

/// Builds a three-qubit QFT circuit using controlled rotations and a final SWAP.
pub fn qft3_circuit() -> Circuit {
    let q0 = Qubit::new(0);
    let q1 = Qubit::new(1);
    let q2 = Qubit::new(2);
    let mut circuit = Circuit::new(3);
    circuit.h(q2).unwrap();
    circuit.crz(q1, q2, std::f64::consts::FRAC_PI_2).unwrap();
    circuit.h(q1).unwrap();
    circuit.crz(q0, q2, std::f64::consts::FRAC_PI_4).unwrap();
    circuit.crz(q0, q1, std::f64::consts::FRAC_PI_2).unwrap();
    circuit.h(q0).unwrap();
    circuit.swap(q0, q2).unwrap();
    circuit
}

/// Builds a two-qubit line device with the given native gates.
pub fn two_qubit_device(native_gates: Vec<Instruction>) -> Device {
    Device::line("test-device", 2)
        .unwrap()
        .with_native_gates(native_gates)
        .unwrap()
}

/// Asserts every two-qubit operation is supported by a topology edge.
pub fn assert_two_qubit_operations_supported_by_topology(circuit: &Circuit, topology: &Topology) {
    for operation in circuit.operations() {
        if operation.qubits.len() == 2 {
            let first = PhysicalQubit::new(operation.qubits[0].id());
            let second = PhysicalQubit::new(operation.qubits[1].id());
            assert!(
                topology.supports_coupling_either_direction(first, second),
                "operation {operation:?} is not supported by topology"
            );
        }
    }
}

/// Asserts all operations use only the requested standard-gate basis.
pub fn assert_only_standard_gates(circuit: &Circuit, allowed: &[StandardGate]) {
    for operation in circuit.operations() {
        assert!(
            matches!(operation.instruction, Instruction::Standard(gate) if allowed.contains(&gate)),
            "unexpected operation in circuit: {operation:?}"
        );
    }
}

#[derive(Debug, Clone)]
pub enum GeneratedMatrixGate {
    Single(StandardGate, usize),
    Parametric(StandardGate, usize, f64),
    Two(StandardGate, usize, usize),
    TwoParametric(StandardGate, usize, usize, f64),
}

/// Generates small circuits that are cheap to convert to matrices.
pub fn generated_small_matrix_circuit() -> impl Strategy<Value = Circuit> {
    (0usize..=4).prop_flat_map(|num_qubits| {
        prop::collection::vec(generated_matrix_gate(num_qubits), 0usize..=24).prop_map(
            move |operations| {
                let mut circuit = Circuit::new(num_qubits);
                for operation in operations {
                    append_generated_matrix_gate(&mut circuit, operation);
                }
                circuit
            },
        )
    })
}

pub fn generated_small_routable_circuit() -> impl Strategy<Value = Circuit> {
    prop::collection::vec(generated_line5_long_range_gate(), 1usize..=12).prop_map(|operations| {
        let mut circuit = Circuit::new(5);
        for operation in operations {
            append_generated_matrix_gate(&mut circuit, operation);
        }
        circuit
    })
}

fn generated_matrix_gate(num_qubits: usize) -> BoxedStrategy<GeneratedMatrixGate> {
    if num_qubits == 0 {
        return Just(GeneratedMatrixGate::Single(StandardGate::I, 0)).boxed();
    }

    let single = (
        prop_oneof![
            Just(StandardGate::I),
            Just(StandardGate::H),
            Just(StandardGate::X),
            Just(StandardGate::Y),
            Just(StandardGate::Z),
            Just(StandardGate::S),
            Just(StandardGate::SDG),
            Just(StandardGate::T),
            Just(StandardGate::TDG),
        ],
        0usize..num_qubits,
    )
        .prop_map(|(gate, qubit)| GeneratedMatrixGate::Single(gate, qubit));
    let angle = -std::f64::consts::TAU..std::f64::consts::TAU;
    let parametric = (
        prop_oneof![
            Just(StandardGate::RX),
            Just(StandardGate::RY),
            Just(StandardGate::RZ),
            Just(StandardGate::Phase),
        ],
        0usize..num_qubits,
        angle.clone(),
    )
        .prop_map(|(gate, qubit, theta)| GeneratedMatrixGate::Parametric(gate, qubit, theta));

    if num_qubits == 1 {
        return prop_oneof![single, parametric].boxed();
    }

    let two_qubits = (0usize..num_qubits, 0usize..num_qubits).prop_filter(
        "two-qubit operation endpoints must be distinct",
        |(first, second)| first != second,
    );
    let two = (
        prop_oneof![
            Just(StandardGate::CX),
            Just(StandardGate::CZ),
            Just(StandardGate::SWAP),
        ],
        two_qubits.clone(),
    )
        .prop_map(|(gate, (first, second))| GeneratedMatrixGate::Two(gate, first, second));
    let two_parametric = (
        prop_oneof![
            Just(StandardGate::CRX),
            Just(StandardGate::CRY),
            Just(StandardGate::CRZ),
            Just(StandardGate::RXX),
            Just(StandardGate::RYY),
            Just(StandardGate::RZZ),
        ],
        two_qubits,
        angle,
    )
        .prop_map(|(gate, (first, second), theta)| {
            GeneratedMatrixGate::TwoParametric(gate, first, second, theta)
        });

    prop_oneof![single, parametric, two, two_parametric].boxed()
}

fn generated_line5_long_range_gate() -> BoxedStrategy<GeneratedMatrixGate> {
    let single = (
        prop_oneof![
            Just(StandardGate::H),
            Just(StandardGate::X),
            Just(StandardGate::Z),
            Just(StandardGate::RZ),
        ],
        0usize..5,
        -std::f64::consts::TAU..std::f64::consts::TAU,
    )
        .prop_map(|(gate, qubit, theta)| match gate {
            StandardGate::RZ => GeneratedMatrixGate::Parametric(gate, qubit, theta),
            _ => GeneratedMatrixGate::Single(gate, qubit),
        });
    let pair = prop_oneof![
        Just((0usize, 2usize)),
        Just((0usize, 3usize)),
        Just((0usize, 4usize)),
        Just((1usize, 3usize)),
        Just((1usize, 4usize)),
        Just((2usize, 4usize)),
    ];
    let two = (
        prop_oneof![Just(StandardGate::CX), Just(StandardGate::CZ)],
        pair.clone(),
    )
        .prop_map(|(gate, (first, second))| GeneratedMatrixGate::Two(gate, first, second));
    let two_parametric = (
        prop_oneof![
            Just(StandardGate::CRX),
            Just(StandardGate::CRY),
            Just(StandardGate::CRZ)
        ],
        pair,
        -std::f64::consts::TAU..std::f64::consts::TAU,
    )
        .prop_map(|(gate, (first, second), theta)| {
            GeneratedMatrixGate::TwoParametric(gate, first, second, theta)
        });

    prop_oneof![single, two, two_parametric].boxed()
}

fn append_generated_matrix_gate(circuit: &mut Circuit, operation: GeneratedMatrixGate) {
    match operation {
        GeneratedMatrixGate::Single(_, qubit) if circuit.qubits().is_empty() => {
            debug_assert_eq!(qubit, 0);
        }
        GeneratedMatrixGate::Single(gate, qubit) => {
            circuit
                .append(
                    Instruction::Standard(gate),
                    [Qubit::new(qubit as u32)],
                    std::iter::empty(),
                    None,
                )
                .unwrap();
        }
        GeneratedMatrixGate::Parametric(gate, qubit, theta) => {
            circuit
                .append(
                    Instruction::Standard(gate),
                    [Qubit::new(qubit as u32)],
                    [ParameterValue::Fixed(theta)],
                    None,
                )
                .unwrap();
        }
        GeneratedMatrixGate::Two(gate, first, second) => {
            circuit
                .append(
                    Instruction::Standard(gate),
                    [Qubit::new(first as u32), Qubit::new(second as u32)],
                    std::iter::empty(),
                    None,
                )
                .unwrap();
        }
        GeneratedMatrixGate::TwoParametric(gate, first, second, theta) => {
            circuit
                .append(
                    Instruction::Standard(gate),
                    [Qubit::new(first as u32), Qubit::new(second as u32)],
                    [ParameterValue::Fixed(theta)],
                    None,
                )
                .unwrap();
        }
    }
}

/// Returns whether a circuit still contains non-lowered gate-like instructions.
pub fn contains_high_level_gate(circuit: &Circuit) -> bool {
    circuit.operations().iter().any(|operation| {
        matches!(
            operation.instruction,
            Instruction::CircuitGate(_) | Instruction::UnitaryGate(_) | Instruction::McGate(_)
        )
    })
}

// Returns whether a named compiler workflow step reported a change.
// pub fn step_changed(result: &CompileResult, name: &str) -> bool {
//     result
//         .steps
//         .iter()
//         .find(|step| step.name == name)
//         .is_some_and(|step| step.changed)
// }

/// Asserts that an operation is a standard gate with a single fixed parameter.
pub fn assert_fixed_parameter_operation(
    operation: &ValueOperation,
    expected_gate: StandardGate,
    expected_qubits: &[Qubit],
    expected_theta: f64,
) {
    assert!(matches!(
        operation.instruction,
        ValueInstruction::Instruction(Instruction::Standard(gate)) if gate == expected_gate
    ));
    assert_eq!(operation.qubits.as_slice(), expected_qubits);
    assert!(matches!(
        operation.params.as_slice(),
        [ParameterValue::Fixed(theta)] if theta.to_bits() == expected_theta.to_bits()
    ));
    assert!(operation.label.is_none());
}

/// Maps an SU(2) rotation axis to the corresponding single-qubit standard gate.
pub fn rotation(axis: Su2RotationAxis) -> StandardGate {
    match axis {
        Su2RotationAxis::X => StandardGate::RX,
        Su2RotationAxis::Y => StandardGate::RY,
        Su2RotationAxis::Z => StandardGate::RZ,
    }
}

/// Maps an SU(2) rotation axis to the corresponding controlled standard gate.
pub fn controlled_rotation(axis: Su2RotationAxis) -> StandardGate {
    match axis {
        Su2RotationAxis::X => StandardGate::CRX,
        Su2RotationAxis::Y => StandardGate::CRY,
        Su2RotationAxis::Z => StandardGate::CRZ,
    }
}

/// Builds the matrix of a multi-controlled standard gate.
pub fn mc_gate_matrix(
    num_qubits: usize,
    num_controls: u8,
    gate: StandardGate,
    qubits: Vec<Qubit>,
    params: impl IntoIterator<Item = ParameterValue>,
) -> Array2<Complex64> {
    let mut circuit = Circuit::new(num_qubits);
    circuit
        .append(
            Instruction::McGate(Box::new(MCGate::new(num_controls, gate))),
            qubits,
            params,
            None,
        )
        .unwrap();
    circuit_to_matrix(&circuit, None).unwrap()
}

/// Returns the unique nonzero matrix output for one computational-basis input.
pub fn single_nonzero_matrix_output(
    matrix: &Array2<Complex64>,
    input_basis_state: usize,
    epsilon: f64,
) -> (usize, Complex64) {
    let outputs: Vec<_> = (0..matrix.nrows())
        .filter_map(|output_basis_state| {
            let amplitude = matrix[[output_basis_state, input_basis_state]];
            (amplitude.norm() > epsilon).then_some((output_basis_state, amplitude))
        })
        .collect();

    assert_eq!(
        outputs.len(),
        1,
        "input basis state {input_basis_state} has outputs {outputs:?}"
    );
    outputs[0]
}
