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

//! Candidate collection for two-qubit block resynthesis.
//!
//! Collection starts from each fixed numeric two-qubit standard gate and
//! expands through a temporary dependency DAG. Operations on the same qubit
//! pair, plus fixed numeric one-qubit gates on either qubit, become matched
//! block operations. Other dependency-path operations may be crossed only when
//! they commute exactly with every matched operation they move across.

use super::commutation::{CachedCommutation, OperationView};
use super::config::TwoQubitBlockResynthesisConfig;
use super::dag_collector::collect_two_qubit_blocks_dag;
use crate::circuit::{Circuit, Instruction, Qubit, StandardGate};
use crate::compile::CompilerError;

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct TwoQubitNumericBlock {
    /// Canonical qubit order used for matrix construction and synthesis.
    pub qubits: [Qubit; 2],
    /// Source orders included in the synthesized unitary.
    pub matched_orders: Vec<usize>,
    /// Source orders skipped while collecting the block and preserved in place.
    pub crossed_orders: Vec<usize>,
    pub matched_1q_count: usize,
    pub matched_2q_count: usize,
    pub contains_swap: bool,
}

impl TwoQubitNumericBlock {
    pub(crate) fn first_order(&self) -> usize {
        self.matched_orders[0]
    }

    pub(crate) fn last_order(&self) -> usize {
        *self
            .matched_orders
            .last()
            .unwrap_or(&self.matched_orders[0])
    }

    pub(crate) fn span(&self) -> usize {
        self.last_order() - self.first_order() + 1
    }

    pub(crate) fn is_promising(&self) -> bool {
        self.matched_2q_count >= 2
            || self.contains_swap
            || (self.matched_2q_count >= 1 && self.matched_1q_count >= 1)
    }
}

pub(crate) fn collect_two_qubit_blocks(
    source: &Circuit,
    ops: &[OperationView<'_>],
    commutation: &mut CachedCommutation,
    config: &TwoQubitBlockResynthesisConfig,
) -> Result<Vec<TwoQubitNumericBlock>, CompilerError> {
    collect_two_qubit_blocks_dag(source, ops, commutation, config)
}

pub(super) struct BlockBuilder<'a> {
    pub(super) qubits: [Qubit; 2],
    matched_positions: Vec<usize>,
    crossed_positions: Vec<usize>,
    pub(super) ops: &'a [OperationView<'a>],
}

impl<'a> BlockBuilder<'a> {
    pub(super) fn new(qubits: [Qubit; 2], anchor: usize, ops: &'a [OperationView<'a>]) -> Self {
        Self {
            qubits,
            matched_positions: vec![anchor],
            crossed_positions: Vec::new(),
            ops,
        }
    }

    pub(super) fn matched_len(&self) -> usize {
        self.matched_positions.len()
    }

    pub(super) fn crossed_len(&self) -> usize {
        self.crossed_positions.len()
    }

    pub(super) fn add_matched(&mut self, position: usize) {
        self.matched_positions.push(position);
    }

    pub(super) fn add_crossed(&mut self, position: usize) {
        self.crossed_positions.push(position);
    }

    pub(super) fn can_add_candidate(
        &self,
        candidate: usize,
        commutation: &CachedCommutation,
    ) -> bool {
        let candidate_view = &self.ops[candidate];
        self.crossed_positions.iter().all(|&crossed| {
            let crossed_view = &self.ops[crossed];
            !shares_any_qubit(crossed_view, candidate_view)
                || commutation.commute_ops_skip_cache(crossed_view, candidate_view)
        })
    }

    pub(super) fn can_cross(
        &mut self,
        skipped: usize,
        commutation: &mut CachedCommutation,
    ) -> bool {
        let skipped_view = &self.ops[skipped];
        self.matched_positions.iter().all(|&matched| {
            let matched_view = &self.ops[matched];
            !shares_any_qubit(skipped_view, matched_view)
                || commutation.commute_ops(skipped_view, matched_view)
        })
    }

    pub(super) fn finish(mut self) -> TwoQubitNumericBlock {
        self.matched_positions.sort_unstable();
        self.matched_positions.dedup();
        self.crossed_positions.sort_unstable();
        self.crossed_positions.dedup();

        let mut matched_1q_count = 0;
        let mut matched_2q_count = 0;
        let mut contains_swap = false;
        for &position in &self.matched_positions {
            let op = self.ops[position].operation;
            match op.qubits.len() {
                1 => matched_1q_count += 1,
                2 => matched_2q_count += 1,
                _ => {}
            }
            contains_swap |= matches!(op.instruction, Instruction::Standard(StandardGate::SWAP));
        }

        TwoQubitNumericBlock {
            qubits: self.qubits,
            matched_orders: self.matched_positions,
            crossed_orders: self.crossed_positions,
            matched_1q_count,
            matched_2q_count,
            contains_swap,
        }
    }
}

pub(super) fn is_block_candidate(view: &OperationView<'_>, pair: [Qubit; 2]) -> bool {
    if !is_fixed_numeric_standard(view) {
        return false;
    }
    match view.operation.qubits.as_slice() {
        [q] => *q == pair[0] || *q == pair[1],
        [a, b] => (*a == pair[0] && *b == pair[1]) || (*a == pair[1] && *b == pair[0]),
        _ => false,
    }
}

pub(super) fn is_hard_boundary(
    view: &OperationView<'_>,
    config: &TwoQubitBlockResynthesisConfig,
) -> bool {
    if config.skip_labeled_ops && view.operation.label.is_some() {
        return true;
    }
    !matches!(view.operation.instruction, Instruction::Standard(_))
        || view.operation.qubits.len() > 2
}

pub(super) fn is_fixed_numeric_standard(view: &OperationView<'_>) -> bool {
    let Instruction::Standard(gate) = view.operation.instruction else {
        return false;
    };
    if gate.num_qubits() != view.operation.qubits.len() || gate.num_params() != view.params.len() {
        return false;
    }
    view.params
        .iter()
        .all(|param| param.evaluate(&None).is_ok_and(f64::is_finite))
}

pub(super) fn shares_any_qubit(lhs: &OperationView<'_>, rhs: &OperationView<'_>) -> bool {
    lhs.operation
        .qubits
        .iter()
        .any(|qubit| rhs.operation.qubits.contains(qubit))
}
