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

//! Local cost model for numeric block resynthesis.

use super::commutation::OperationView;
use crate::circuit::{Instruction, Qubit, StandardGate, ValueInstruction, ValueOperation};
use std::cmp::Ordering;
use std::collections::HashMap;

/// Local cost used to accept only strictly improving resynthesis patches.
///
/// Ordering is lexicographic and intentionally favors hardware-relevant
/// improvements before cosmetic gate-count reductions:
///
/// 1. unsupported operations,
/// 2. two-qubit operations,
/// 3. local depth estimate,
/// 4. total operations,
/// 5. parameterized operations,
/// 6. multi-qubit operations.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub(crate) struct ResynthesisCost {
    pub unsupported_ops: usize,
    pub two_qubit_ops: usize,
    pub depth_estimate: usize,
    pub total_ops: usize,
    pub parameterized_ops: usize,
    pub multi_qubit_ops: usize,
}

impl Ord for ResynthesisCost {
    fn cmp(&self, other: &Self) -> Ordering {
        self.unsupported_ops
            .cmp(&other.unsupported_ops)
            .then_with(|| self.two_qubit_ops.cmp(&other.two_qubit_ops))
            .then_with(|| self.depth_estimate.cmp(&other.depth_estimate))
            .then_with(|| self.total_ops.cmp(&other.total_ops))
            .then_with(|| self.parameterized_ops.cmp(&other.parameterized_ops))
            .then_with(|| self.multi_qubit_ops.cmp(&other.multi_qubit_ops))
    }
}

impl PartialOrd for ResynthesisCost {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

pub(crate) fn cost_of_source_ops(ops: &[&OperationView<'_>]) -> ResynthesisCost {
    let mut cost = ResynthesisCost::default();
    let mut depths = HashMap::new();
    for view in ops {
        add_operation_cost(
            &mut cost,
            &mut depths,
            match view.operation.instruction {
                Instruction::Standard(gate) => Some(gate),
                _ => None,
            },
            view.operation.qubits.as_slice(),
            view.operation.params.len(),
        );
    }
    cost
}

pub(crate) fn cost_of_replacements(ops: &[ValueOperation]) -> ResynthesisCost {
    let mut cost = ResynthesisCost::default();
    let mut depths = HashMap::new();
    for op in ops {
        let gate = match &op.instruction {
            ValueInstruction::Instruction(Instruction::Standard(gate)) => Some(*gate),
            ValueInstruction::Instruction(_) => None,
            ValueInstruction::ClassicalControl(_) => None,
        };
        add_operation_cost(
            &mut cost,
            &mut depths,
            gate,
            op.qubits.as_slice(),
            op.params.len(),
        );
    }
    cost
}

fn add_operation_cost(
    cost: &mut ResynthesisCost,
    depths: &mut HashMap<Qubit, usize>,
    gate: Option<StandardGate>,
    qubits: &[Qubit],
    param_count: usize,
) {
    if gate == Some(StandardGate::GPhase) {
        return;
    }

    cost.total_ops += 1;
    if gate.is_none() {
        cost.unsupported_ops += 1;
    }
    match qubits.len() {
        0 | 1 => {}
        2 => cost.two_qubit_ops += 1,
        _ => cost.multi_qubit_ops += 1,
    }
    if param_count > 0 {
        cost.parameterized_ops += 1;
    }
    if qubits.is_empty() {
        return;
    }

    let next = qubits
        .iter()
        .filter_map(|qubit| depths.get(qubit))
        .max()
        .copied()
        .unwrap_or(0)
        + 1;
    for &qubit in qubits {
        depths.insert(qubit, next);
    }
    cost.depth_estimate = cost.depth_estimate.max(next);
}

#[cfg(test)]
#[path = "cost_test.rs"]
mod cost_test;
