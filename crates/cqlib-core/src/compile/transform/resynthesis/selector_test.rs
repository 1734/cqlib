use super::*;
use crate::circuit::{CircuitParam, Operation, Parameter, Qubit};
use crate::compile::transform::decompose::unitary::TwoQubitUnitaryDecomposeBasis;
use ndarray::Array2;
use ndarray::linalg::kron;
use smallvec::smallvec;

fn view<'a>(order: usize, operation: &'a Operation) -> OperationView<'a> {
    OperationView::new(
        order,
        operation,
        operation
            .params
            .iter()
            .map(|param| match param {
                CircuitParam::Fixed(value) => Parameter::from(*value),
                CircuitParam::Index(_) => unreachable!(),
            })
            .collect(),
    )
}

fn standard_op(gate: StandardGate, qubits: &[Qubit]) -> Operation {
    Operation {
        instruction: Instruction::Standard(gate),
        qubits: qubits.iter().copied().collect(),
        params: smallvec![],
        label: None,
    }
}

fn block(qubits: [Qubit; 2], matched_orders: Vec<usize>) -> TwoQubitNumericBlock {
    TwoQubitNumericBlock {
        qubits,
        matched_orders,
        crossed_orders: vec![],
        matched_1q_count: 0,
        matched_2q_count: 2,
        contains_swap: false,
    }
}

fn block_with_crossed(
    qubits: [Qubit; 2],
    matched_orders: Vec<usize>,
    crossed_orders: Vec<usize>,
) -> TwoQubitNumericBlock {
    TwoQubitNumericBlock {
        qubits,
        matched_orders,
        crossed_orders,
        matched_1q_count: 0,
        matched_2q_count: 3,
        contains_swap: false,
    }
}

#[test]
fn reversed_two_qubit_gate_uses_swap_conjugation() {
    let q0 = Qubit::new(0);
    let q1 = Qubit::new(1);
    let cx = StandardGate::CX.matrix(&[]).unwrap().into_owned();
    let swap = StandardGate::SWAP.matrix(&[]).unwrap().into_owned();
    let expected = swap.dot(&cx).dot(&swap);
    let op = standard_op(StandardGate::CX, &[q1, q0]);
    let ops = vec![view(0, &op)];

    assert_eq!(
        block_matrix(&block([q0, q1], vec![0]), &ops).unwrap(),
        expected
    );
}

#[test]
fn single_qubit_expansion_respects_canonical_order() {
    let q0 = Qubit::new(0);
    let q1 = Qubit::new(1);
    let h = standard_op(StandardGate::H, &[q0]);
    let x = standard_op(StandardGate::X, &[q1]);
    let ops = vec![view(0, &h), view(1, &x)];
    let h_matrix = StandardGate::H.matrix(&[]).unwrap().into_owned();
    let x_matrix = StandardGate::X.matrix(&[]).unwrap().into_owned();
    let expected = kron(&h_matrix.view(), &x_matrix.view());

    assert_eq!(
        block_matrix(&block([q0, q1], vec![0, 1]), &ops).unwrap(),
        expected
    );
}

#[test]
fn block_matrix_multiplies_operations_in_source_order() {
    let q0 = Qubit::new(0);
    let q1 = Qubit::new(1);
    let h = standard_op(StandardGate::H, &[q0]);
    let cx = standard_op(StandardGate::CX, &[q0, q1]);
    let ops = vec![view(0, &h), view(1, &cx)];
    let h_matrix = StandardGate::H.matrix(&[]).unwrap().into_owned();
    let identity = Array2::eye(2);
    let expanded_h = kron(&h_matrix.view(), &identity.view());
    let expected = StandardGate::CX.matrix(&[]).unwrap().dot(&expanded_h);

    assert_eq!(
        block_matrix(&block([q0, q1], vec![0, 1]), &ops).unwrap(),
        expected
    );
}

#[test]
fn select_patches_keeps_strictly_improving_non_overlapping_candidate() {
    let q0 = Qubit::new(0);
    let q1 = Qubit::new(1);
    let first = standard_op(StandardGate::CX, &[q0, q1]);
    let second = standard_op(StandardGate::CX, &[q0, q1]);
    let ops = vec![view(0, &first), view(1, &second)];
    let blocks = vec![block([q0, q1], vec![0, 1])];
    let commutation = CachedCommutation::new(
        TwoQubitBlockResynthesisConfig::normal(TwoQubitUnitaryDecomposeBasis::Cx).commutation,
    );

    let patches = select_patches(
        blocks,
        &ops,
        &commutation,
        &TwoQubitBlockResynthesisConfig::normal(TwoQubitUnitaryDecomposeBasis::Cx),
    )
    .unwrap();

    assert_eq!(patches.len(), 1);
    assert_eq!(patches[0].matched_orders, vec![0, 1]);
    assert!(patches[0].replacement.is_empty());
}

#[test]
fn select_patches_deduplicates_identical_blocks_before_selection() {
    let q0 = Qubit::new(0);
    let q1 = Qubit::new(1);
    let first = standard_op(StandardGate::CX, &[q0, q1]);
    let second = standard_op(StandardGate::CX, &[q0, q1]);
    let ops = vec![view(0, &first), view(1, &second)];
    let blocks = vec![block([q0, q1], vec![0, 1]), block([q0, q1], vec![0, 1])];
    let config = TwoQubitBlockResynthesisConfig::normal(TwoQubitUnitaryDecomposeBasis::Cx);
    let commutation = CachedCommutation::new(config.commutation.clone());

    let patches = select_patches(blocks, &ops, &commutation, &config).unwrap();

    assert_eq!(patches.len(), 1);
}

#[test]
fn select_patches_uses_non_overlapping_greedy_selection() {
    let q0 = Qubit::new(0);
    let q1 = Qubit::new(1);
    let first = standard_op(StandardGate::CX, &[q0, q1]);
    let second = standard_op(StandardGate::CX, &[q0, q1]);
    let third = standard_op(StandardGate::CX, &[q0, q1]);
    let ops = vec![view(0, &first), view(1, &second), view(2, &third)];
    let blocks = vec![block([q0, q1], vec![0, 1]), block([q0, q1], vec![1, 2])];
    let config = TwoQubitBlockResynthesisConfig::normal(TwoQubitUnitaryDecomposeBasis::Cx);
    let commutation = CachedCommutation::new(config.commutation.clone());

    let patches = select_patches(blocks, &ops, &commutation, &config).unwrap();

    assert_eq!(patches.len(), 1);
    assert_eq!(patches[0].matched_orders, vec![0, 1]);
}

#[test]
fn select_patches_rejects_when_replacement_does_not_commute_with_crossed_op() {
    let q0 = Qubit::new(0);
    let q1 = Qubit::new(1);
    let first = standard_op(StandardGate::CX, &[q0, q1]);
    let crossed = standard_op(StandardGate::H, &[q0]);
    let second = standard_op(StandardGate::CX, &[q0, q1]);
    let third = standard_op(StandardGate::CX, &[q0, q1]);
    let ops = vec![
        view(0, &first),
        view(1, &crossed),
        view(2, &second),
        view(3, &third),
    ];
    let blocks = vec![block_with_crossed([q0, q1], vec![0, 2, 3], vec![1])];
    let config = TwoQubitBlockResynthesisConfig::normal(TwoQubitUnitaryDecomposeBasis::Cx);
    let commutation = CachedCommutation::new(config.commutation.clone());

    let patches = select_patches(blocks, &ops, &commutation, &config).unwrap();

    assert!(patches.is_empty());
}

#[test]
fn select_patches_accepts_when_replacement_commutes_with_disjoint_crossed_op() {
    let q0 = Qubit::new(0);
    let q1 = Qubit::new(1);
    let q2 = Qubit::new(2);
    let first = standard_op(StandardGate::CX, &[q0, q1]);
    let crossed = standard_op(StandardGate::H, &[q2]);
    let second = standard_op(StandardGate::CX, &[q0, q1]);
    let third = standard_op(StandardGate::CX, &[q0, q1]);
    let ops = vec![
        view(0, &first),
        view(1, &crossed),
        view(2, &second),
        view(3, &third),
    ];
    let blocks = vec![block_with_crossed([q0, q1], vec![0, 2, 3], vec![1])];
    let config = TwoQubitBlockResynthesisConfig::normal(TwoQubitUnitaryDecomposeBasis::Cx);
    let commutation = CachedCommutation::new(config.commutation.clone());

    let patches = select_patches(blocks, &ops, &commutation, &config).unwrap();

    assert_eq!(patches.len(), 1);
    assert_eq!(patches[0].crossed_orders, vec![1]);
}

#[test]
fn select_patches_rejects_single_cx_without_cost_improvement() {
    let q0 = Qubit::new(0);
    let q1 = Qubit::new(1);
    let cx = standard_op(StandardGate::CX, &[q0, q1]);
    let ops = vec![view(0, &cx)];
    let blocks = vec![block([q0, q1], vec![0])];
    let config = TwoQubitBlockResynthesisConfig::normal(TwoQubitUnitaryDecomposeBasis::Cx);
    let commutation = CachedCommutation::new(config.commutation.clone());

    let patches = select_patches(blocks, &ops, &commutation, &config).unwrap();

    assert!(patches.is_empty());
}
