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

//! Candidate synthesis and selection for two-qubit resynthesis.
//!
//! Selection has two stages. First, syntactically duplicate blocks are removed
//! and promising blocks are synthesized in a deterministic priority order.
//! Second, accepted patches are greedily filtered so no two selected patches
//! rewrite the same source operation.

use super::collector::TwoQubitNumericBlock;
use super::commutation::{CachedCommutation, OperationView};
use super::config::TwoQubitBlockResynthesisConfig;
use super::cost::{ResynthesisCost, cost_of_replacements, cost_of_source_ops};
use crate::circuit::{Instruction, StandardGate, ValueOperation};
use crate::compile::CompilerError;
use crate::compile::transform::decompose::unitary::{
    TwoQubitUnitarySynthesisResult, synthesize_numeric_2q_unitary,
};
use ndarray::Array2;
use ndarray::linalg::kron;
use num_complex::Complex64;
use std::cmp::{Ordering, Reverse};
use std::collections::HashSet;

#[derive(Debug, Clone)]
pub(crate) struct BlockPatch {
    /// Source order where the synthesized replacement is emitted.
    pub first_order: usize,
    /// Source operations consumed by this patch.
    pub matched_orders: Vec<usize>,
    /// Source operations preserved in place after post-synthesis commutation
    /// verification.
    pub crossed_orders: Vec<usize>,
    pub replacement: Vec<ValueOperation>,
    pub before_cost: ResynthesisCost,
    pub after_cost: ResynthesisCost,
    pub synthesis_phase: f64,
}

pub(crate) fn select_patches(
    blocks: Vec<TwoQubitNumericBlock>,
    ops: &[OperationView<'_>],
    commutation: &CachedCommutation,
    config: &TwoQubitBlockResynthesisConfig,
) -> Result<Vec<BlockPatch>, CompilerError> {
    let mut seen = HashSet::new();
    let mut blocks = blocks
        .into_iter()
        .filter(|block| seen.insert(block.matched_orders.clone()))
        .collect::<Vec<_>>();
    blocks.sort_by(compare_blocks);

    let mut patches = Vec::new();
    for block in blocks {
        if let Some(patch) = try_synthesize_block(&block, ops, commutation, config)? {
            patches.push(patch);
        }
    }
    patches.sort_by(compare_patches);

    let mut covered = HashSet::new();
    let mut selected = Vec::new();
    for patch in patches {
        if patch
            .matched_orders
            .iter()
            .any(|order| covered.contains(order))
        {
            continue;
        }
        for order in &patch.matched_orders {
            covered.insert(*order);
        }
        selected.push(patch);
    }
    selected.sort_by_key(|patch| patch.first_order);
    Ok(selected)
}

fn compare_blocks(lhs: &TwoQubitNumericBlock, rhs: &TwoQubitNumericBlock) -> Ordering {
    // Synthesis is the expensive step. Prefer candidates most likely to improve:
    // more 2q gates, larger matched unitary, denser span, SWAP involvement, then
    // smaller source span and earlier deterministic position.
    let lhs_density = lhs.matched_orders.len() * rhs.span();
    let rhs_density = rhs.matched_orders.len() * lhs.span();
    Reverse(lhs.matched_2q_count)
        .cmp(&Reverse(rhs.matched_2q_count))
        .then_with(|| Reverse(lhs.matched_orders.len()).cmp(&Reverse(rhs.matched_orders.len())))
        .then_with(|| Reverse(lhs_density).cmp(&Reverse(rhs_density)))
        .then_with(|| Reverse(lhs.contains_swap).cmp(&Reverse(rhs.contains_swap)))
        .then_with(|| lhs.span().cmp(&rhs.span()))
        .then_with(|| lhs.first_order().cmp(&rhs.first_order()))
}

fn compare_patches(lhs: &BlockPatch, rhs: &BlockPatch) -> Ordering {
    let lhs_reduction = lhs
        .before_cost
        .two_qubit_ops
        .saturating_sub(lhs.after_cost.two_qubit_ops);
    let rhs_reduction = rhs
        .before_cost
        .two_qubit_ops
        .saturating_sub(rhs.after_cost.two_qubit_ops);

    lhs.after_cost
        .cmp(&rhs.after_cost)
        .then_with(|| rhs_reduction.cmp(&lhs_reduction))
        .then_with(|| Reverse(lhs.matched_orders.len()).cmp(&Reverse(rhs.matched_orders.len())))
        .then_with(|| lhs.first_order.cmp(&rhs.first_order))
}

fn try_synthesize_block(
    block: &TwoQubitNumericBlock,
    ops: &[OperationView<'_>],
    commutation: &CachedCommutation,
    config: &TwoQubitBlockResynthesisConfig,
) -> Result<Option<BlockPatch>, CompilerError> {
    let matrix = match block_matrix(block, ops) {
        Ok(matrix) => matrix,
        Err(_) => return Ok(None),
    };
    let TwoQubitUnitarySynthesisResult {
        operations,
        global_phase,
    } = match synthesize_numeric_2q_unitary(&matrix, block.qubits, config.two_qubit_basis) {
        Ok(result) => result,
        Err(_) => return Ok(None),
    };

    let matched = block
        .matched_orders
        .iter()
        .map(|&order| &ops[order])
        .collect::<Vec<_>>();
    let before_cost = cost_of_source_ops(&matched);
    let after_cost = cost_of_replacements(&operations);
    if after_cost >= before_cost {
        return Ok(None);
    }

    let crossed = block
        .crossed_orders
        .iter()
        .map(|&order| &ops[order])
        .collect::<Vec<_>>();
    if !commutation.replacements_commute_with_crossed(&crossed, &operations) {
        return Ok(None);
    }

    Ok(Some(BlockPatch {
        first_order: block.first_order(),
        matched_orders: block.matched_orders.clone(),
        crossed_orders: block.crossed_orders.clone(),
        replacement: operations,
        before_cost,
        after_cost,
        synthesis_phase: global_phase,
    }))
}

// Matrix construction uses the same convention as `circuit_to_matrix`: source
// operations are multiplied as `gate_n * ... * gate_0`, where `gate_0` is the
// earliest source operation. `block.qubits[0]` is the first tensor factor.
fn block_matrix(
    block: &TwoQubitNumericBlock,
    ops: &[OperationView<'_>],
) -> Result<Array2<Complex64>, CompilerError> {
    let mut result = Array2::<Complex64>::eye(4);
    let mut orders = block.matched_orders.clone();
    orders.sort_unstable();
    for order in orders {
        let view = &ops[order];
        let Instruction::Standard(gate) = view.operation.instruction else {
            return Err(CompilerError::InvariantViolation(
                "resynthesis matrix requested for non-standard operation".to_string(),
            ));
        };
        let params = view
            .params
            .iter()
            .map(|param| param.evaluate(&None))
            .collect::<Result<Vec<_>, _>>()
            .map_err(|_| {
                CompilerError::InvariantViolation(
                    "resynthesis matrix requested for symbolic operation".to_string(),
                )
            })?;
        let matrix = gate
            .matrix(&params)
            .map_err(CompilerError::Circuit)?
            .into_owned();
        let identity = Array2::<Complex64>::eye(2);
        let expanded = match view.operation.qubits.as_slice() {
            [q] if *q == block.qubits[0] => kron(&matrix.view(), &identity.view()),
            [q] if *q == block.qubits[1] => kron(&identity.view(), &matrix.view()),
            [a, b] if *a == block.qubits[0] && *b == block.qubits[1] => matrix,
            [a, b] if *a == block.qubits[1] && *b == block.qubits[0] => {
                let swap = StandardGate::SWAP.matrix(&[]).unwrap().into_owned();
                swap.dot(&matrix).dot(&swap)
            }
            _ => {
                return Err(CompilerError::InvariantViolation(
                    "resynthesis block contains operation outside canonical qubits".to_string(),
                ));
            }
        };
        result = expanded.dot(&result);
    }
    Ok(result)
}

#[cfg(test)]
#[path = "selector_test.rs"]
mod selector_test;
