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

//! DAG-backed candidate collection for two-qubit block resynthesis.
//!
//! The DAG is a pass-local dependency view. It only improves candidate
//! discovery; final replacement legality is still checked by the selector via
//! exact matrix synthesis, strict cost improvement, and replacement/crossed
//! commutation validation.

use super::collector::{
    BlockBuilder, TwoQubitNumericBlock, is_block_candidate, is_fixed_numeric_standard,
    is_hard_boundary,
};
use super::commutation::{CachedCommutation, OperationView};
use super::config::TwoQubitBlockResynthesisConfig;
use crate::circuit::{Circuit, CircuitDag, Instruction};
use crate::compile::CompilerError;
use rustworkx_core::petgraph::prelude::NodeIndex;
use std::collections::{HashSet, VecDeque};

pub(super) fn collect_two_qubit_blocks_dag(
    source: &Circuit,
    ops: &[OperationView<'_>],
    commutation: &mut CachedCommutation,
    config: &TwoQubitBlockResynthesisConfig,
) -> Result<Vec<TwoQubitNumericBlock>, CompilerError> {
    // TODO: optimize by building the DAG directly from operation views if this
    // O(n) analysis construction becomes visible in benchmarks.
    let operations = ops
        .iter()
        .map(|view| view.operation.clone())
        .collect::<Vec<_>>();
    let dag =
        CircuitDag::from_circuit_operations(source, &operations).map_err(CompilerError::Circuit)?;

    let mut blocks = Vec::new();
    for anchor in 0..ops.len() {
        let anchor_view = &ops[anchor];
        if is_hard_boundary(anchor_view, config) || !is_fixed_numeric_standard(anchor_view) {
            continue;
        }
        let Instruction::Standard(_) = anchor_view.operation.instruction else {
            continue;
        };
        if anchor_view.operation.qubits.len() != 2 {
            continue;
        }

        let qubits = [
            anchor_view.operation.qubits[0],
            anchor_view.operation.qubits[1],
        ];
        let Some(node) = dag.node_for_order(anchor) else {
            return Err(CompilerError::InvariantViolation(format!(
                "resynthesis DAG is missing operation order {anchor}"
            )));
        };
        let mut builder = BlockBuilder::new(qubits, anchor, ops);
        collect_dag_direction(
            &dag,
            &mut builder,
            node,
            Direction::Left,
            commutation,
            config,
        );
        collect_dag_direction(
            &dag,
            &mut builder,
            node,
            Direction::Right,
            commutation,
            config,
        );
        let block = builder.finish();
        if block.is_promising() {
            blocks.push(block);
        }
    }
    Ok(blocks)
}

#[derive(Debug, Clone, Copy)]
enum Direction {
    Left,
    Right,
}

fn collect_dag_direction(
    dag: &CircuitDag,
    builder: &mut BlockBuilder<'_>,
    anchor: NodeIndex,
    direction: Direction,
    commutation: &mut CachedCommutation,
    config: &TwoQubitBlockResynthesisConfig,
) {
    let Some(anchor_order) = dag.operation_order(anchor) else {
        return;
    };
    let mut frontier = VecDeque::from(sorted_neighbors(dag, anchor, direction));
    let mut seen = HashSet::new();
    let mut accepted = HashSet::from([anchor_order]);
    let mut visited = 0usize;

    while visited < config.max_scan_span {
        let Some(node) = frontier.pop_front() else {
            break;
        };
        if !seen.insert(node) {
            continue;
        }
        let Some(order) = dag.operation_order(node) else {
            continue;
        };
        visited += 1;

        if !dependencies_are_accepted(dag, node, direction, anchor_order, &accepted) {
            continue;
        }

        let view = &builder.ops[order];
        if is_hard_boundary(view, config) {
            continue;
        }

        if is_block_candidate(view, builder.qubits) {
            if builder.matched_len() >= config.max_block_ops
                || !builder.can_add_candidate(order, commutation)
            {
                continue;
            }
            builder.add_matched(order);
            accepted.insert(order);
            push_neighbors(&mut frontier, dag, node, direction);
            continue;
        }

        if builder.crossed_len() >= config.max_crossed_ops || !builder.can_cross(order, commutation)
        {
            continue;
        }
        builder.add_crossed(order);
        accepted.insert(order);
        push_neighbors(&mut frontier, dag, node, direction);
    }
}

fn dependencies_are_accepted(
    dag: &CircuitDag,
    node: NodeIndex,
    direction: Direction,
    anchor_order: usize,
    accepted: &HashSet<usize>,
) -> bool {
    let dependencies = match direction {
        Direction::Left => dag.successors(node).collect::<Vec<_>>(),
        Direction::Right => dag.predecessors(node).collect::<Vec<_>>(),
    };
    dependencies.into_iter().all(|dependency| {
        let Some(order) = dag.operation_order(dependency) else {
            return true;
        };
        let between_anchor_and_node = match direction {
            Direction::Left => order <= anchor_order,
            Direction::Right => order >= anchor_order,
        };
        !between_anchor_and_node || accepted.contains(&order)
    })
}

fn push_neighbors(
    frontier: &mut VecDeque<NodeIndex>,
    dag: &CircuitDag,
    node: NodeIndex,
    direction: Direction,
) {
    for neighbor in sorted_neighbors(dag, node, direction) {
        frontier.push_back(neighbor);
    }
}

fn sorted_neighbors(dag: &CircuitDag, node: NodeIndex, direction: Direction) -> Vec<NodeIndex> {
    let mut neighbors = match direction {
        Direction::Left => dag.predecessors(node).collect::<Vec<_>>(),
        Direction::Right => dag.successors(node).collect::<Vec<_>>(),
    };
    neighbors.sort_by_key(|neighbor| {
        let order = dag.operation_order(*neighbor).unwrap_or(usize::MAX);
        match direction {
            Direction::Left => (usize::MAX - order, neighbor.index()),
            Direction::Right => (order, neighbor.index()),
        }
    });
    neighbors
}

#[cfg(test)]
#[path = "dag_collector_test.rs"]
mod dag_collector_test;
