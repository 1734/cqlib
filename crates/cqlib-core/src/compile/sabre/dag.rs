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

//! Dependency DAG used by the compiler SABRE implementation.
//!
//! The DAG is built from circuit operation order and logical-qubit overlap.
//! Nodes are intentionally coarser than single operations: consecutive
//! operations that share the same dependency boundary can be folded together so
//! routing sees the smallest set of scheduling barriers needed for progress.
//!
//! [`SabreNodeKind::TwoQ`] represents a two-logical-qubit interaction that
//! must be adjacent before it can be emitted. Dependencies are derived from a
//! per-wire frontier: each new operation depends on the latest node touching
//! any of its logical qubits, and then becomes the frontier for those qubits.
//! This is enough for SABRE because layout routing only reasons about
//! two-qubit interaction readiness.
//!
//! [`SabreNodeKind::Synchronize`] is used for zero- and one-qubit operations,
//! delays, and directives. These operations do not create a routed two-qubit
//! interaction, but they still preserve sequencing at the current dependency
//! boundary. An empty-qubit barrier is a global synchronization boundary: it
//! waits for every active wire and becomes a dependency of every subsequent
//! operation. Other initial synchronize operations that touch no mapped
//! frontier stay in [`SabreDag::initial`].
//!
//! Control-flow operations become recursive DAG nodes. The outer node preserves
//! the control-flow operation as a scheduling boundary, while each body is
//! decomposed into its own [`SabreDag`] so routing can restore layouts at block
//! boundaries.

use crate::circuit::{
    ClassicalControlOp, ClassicalExpr, ClassicalVar, Directive, Instruction, Operation,
};
use crate::compile::CompilerError;
use crate::device::LogicalQubit;
use rustworkx_core::petgraph::Direction;
use rustworkx_core::petgraph::graph::DiGraph;
use rustworkx_core::petgraph::prelude::NodeIndex;
use std::collections::{BTreeMap, BTreeSet};

#[derive(Debug, Clone)]
pub(crate) enum SabreNodeKind {
    Synchronize,
    TwoQ([LogicalQubit; 2]),
    ControlFlow(SabreControlFlow),
}

#[derive(Debug, Clone)]
pub(crate) enum SabreControlFlow {
    If {
        condition: ClassicalExpr,
        then_body: SabreDag,
        else_body: Option<SabreDag>,
    },
    While {
        condition: ClassicalExpr,
        body: SabreDag,
    },
    For {
        var: ClassicalVar,
        start: ClassicalExpr,
        stop: ClassicalExpr,
        step: ClassicalExpr,
        body: SabreDag,
    },
    Switch {
        target: ClassicalExpr,
        cases: Vec<SabreSwitchCase>,
        default: Option<SabreDag>,
    },
}

#[derive(Debug, Clone)]
pub(crate) struct SabreSwitchCase {
    pub(crate) value: u128,
    pub(crate) body: SabreDag,
}

#[derive(Debug, Clone)]
pub(crate) struct SabreNode {
    pub(crate) operations: Vec<Operation>,
    pub(crate) kind: SabreNodeKind,
}

#[derive(Debug, Clone)]
pub(crate) struct SabreDag {
    pub(crate) initial: Vec<Operation>,
    pub(crate) graph: DiGraph<SabreNode, ()>,
    pub(crate) first_layer: Vec<NodeIndex>,
}

impl SabreDag {
    pub(crate) fn from_operations(operations: &[Operation]) -> Result<Self, CompilerError> {
        let mut initial = Vec::new();
        let mut graph = DiGraph::new();
        let mut wire_pos: BTreeMap<LogicalQubit, NodeIndex> = BTreeMap::new();
        let mut first_layer = Vec::new();
        let mut global_barrier = None;

        for operation in operations {
            let kind = kind_from_operation(operation)?;
            let is_global_barrier = matches!(
                operation.instruction,
                Instruction::Directive(Directive::Barrier)
            ) && operation.qubits.is_empty();
            let ordering_barrier = matches!(
                operation.instruction,
                Instruction::ClassicalData(_) | Instruction::ClassicalControl(_)
            ) || is_global_barrier;
            let qubits = operation
                .qubits
                .iter()
                .copied()
                .map(LogicalQubit::from_qubit)
                .collect::<Vec<_>>();

            let mut parents = global_barrier.into_iter().collect::<Vec<_>>();
            if ordering_barrier {
                for parent in wire_pos.values().copied() {
                    if !parents.contains(&parent) {
                        parents.push(parent);
                    }
                }
            } else {
                for logical in &qubits {
                    if let Some(parent) = wire_pos.get(logical).copied()
                        && !parents.contains(&parent)
                    {
                        parents.push(parent);
                    }
                }
            }
            let predecessors = match parents.as_slice() {
                [] => Predecessors::AllUnmapped,
                [parent] => Predecessors::Single(*parent),
                _ => Predecessors::Multiple(parents),
            };
            let mut created_node = None;
            match predecessors {
                Predecessors::AllUnmapped => match kind {
                    SabreNodeKind::Synchronize if !ordering_barrier => {
                        initial.push(operation.clone())
                    }
                    kind => {
                        let node = graph.add_node(SabreNode {
                            operations: vec![operation.clone()],
                            kind,
                        });
                        first_layer.push(node);
                        created_node = Some(node);
                        for logical in qubits {
                            wire_pos.insert(logical, node);
                        }
                    }
                },
                Predecessors::Single(previous) => {
                    // Synchronize operations share the same dependency boundary,
                    // and consecutive two-qubit operations on the same active
                    // wires remain routable once the first one is routed.
                    let fold_into_previous = !ordering_barrier
                        && match (&graph[previous].kind, &kind) {
                            (_, SabreNodeKind::Synchronize) => true,
                            (SabreNodeKind::TwoQ(previous), SabreNodeKind::TwoQ(current)) => {
                                previous == current || *previous == [current[1], current[0]]
                            }
                            _ => false,
                        };
                    if fold_into_previous {
                        graph[previous].operations.push(operation.clone());
                        for logical in qubits {
                            wire_pos.insert(logical, previous);
                        }
                    } else {
                        let node = graph.add_node(SabreNode {
                            operations: vec![operation.clone()],
                            kind,
                        });
                        graph.add_edge(previous, node, ());
                        created_node = Some(node);
                        for logical in qubits {
                            wire_pos.insert(logical, node);
                        }
                    }
                }
                Predecessors::Multiple(parents) => {
                    let node = graph.add_node(SabreNode {
                        operations: vec![operation.clone()],
                        kind,
                    });
                    created_node = Some(node);
                    for parent in parents {
                        if graph.find_edge(parent, node).is_none() {
                            graph.add_edge(parent, node, ());
                        }
                    }
                    for logical in qubits {
                        wire_pos.insert(logical, node);
                    }
                }
            }
            if ordering_barrier {
                global_barrier = created_node;
            }
        }

        Ok(Self {
            initial,
            graph,
            first_layer,
        })
    }

    /// Builds the static interaction workload used by bidirectional layout
    /// refinement.
    ///
    /// Control-flow bodies are included once. Alternative branches remain
    /// parallel when they touch disjoint logical wires, while a stable branch
    /// order adds per-wire dependencies between branches that share a wire.
    /// This preserves the front-layer invariant that two ready interactions
    /// cannot occupy the same logical qubit.
    pub(crate) fn refinement_workload(operations: &[Operation]) -> Result<Self, CompilerError> {
        let mut builder = WorkloadBuilder::default();
        let mut state = WorkloadState::default();
        builder.add_operations(operations, &mut state)?;
        let first_layer = builder.graph.externals(Direction::Incoming).collect();
        Ok(Self {
            initial: Vec::new(),
            graph: builder.graph,
            first_layer,
        })
    }

    pub(crate) fn reverse_interactions(&self) -> Self {
        let mut graph = self.graph.clone();
        graph.reverse();
        let first_layer = graph.externals(Direction::Incoming).collect();
        Self {
            initial: Vec::new(),
            graph,
            first_layer,
        }
    }
}

#[derive(Default)]
struct WorkloadState {
    wire_frontier: BTreeMap<LogicalQubit, NodeIndex>,
    global_barrier: Option<NodeIndex>,
}

#[derive(Default)]
struct WorkloadBuilder {
    graph: DiGraph<SabreNode, ()>,
}

impl WorkloadBuilder {
    fn add_operations(
        &mut self,
        operations: &[Operation],
        state: &mut WorkloadState,
    ) -> Result<BTreeSet<LogicalQubit>, CompilerError> {
        let mut touched = BTreeSet::new();
        for operation in operations {
            match &operation.instruction {
                Instruction::ClassicalControl(flow) => {
                    touched.extend(self.add_control_flow(flow, state)?);
                }
                Instruction::ClassicalData(_) => {
                    self.add_global_synchronize(state);
                }
                Instruction::Directive(Directive::Barrier) => {
                    if operation.qubits.is_empty() {
                        self.add_global_synchronize(state);
                    } else {
                        let qubits = operation
                            .qubits
                            .iter()
                            .copied()
                            .map(LogicalQubit::from_qubit)
                            .collect::<Vec<_>>();
                        self.add_wire_synchronize(&qubits, state);
                    }
                }
                Instruction::Directive(_) | Instruction::Delay => {}
                _ => match operation.qubits.len() {
                    0 | 1 => {}
                    2 => {
                        let pair = [
                            LogicalQubit::from_qubit(operation.qubits[0]),
                            LogicalQubit::from_qubit(operation.qubits[1]),
                        ];
                        self.add_two_qubit(pair, state);
                        touched.extend(pair);
                    }
                    arity => {
                        return Err(CompilerError::InvalidInput(format!(
                            "sabre requires operations with more than two qubits to be decomposed before routing; found {arity}-qubit operation {}",
                            operation.instruction
                        )));
                    }
                },
            }
        }
        Ok(touched)
    }

    fn add_two_qubit(&mut self, pair: [LogicalQubit; 2], state: &mut WorkloadState) {
        let mut parents = BTreeSet::new();
        parents.extend(state.global_barrier);
        parents.extend(
            pair.iter()
                .filter_map(|logical| state.wire_frontier.get(logical).copied()),
        );
        let node = self.add_node(SabreNodeKind::TwoQ(pair), parents);
        state.wire_frontier.insert(pair[0], node);
        state.wire_frontier.insert(pair[1], node);
    }

    fn add_global_synchronize(&mut self, state: &mut WorkloadState) -> NodeIndex {
        let mut parents = state
            .wire_frontier
            .values()
            .copied()
            .collect::<BTreeSet<_>>();
        parents.extend(state.global_barrier);
        let node = self.add_node(SabreNodeKind::Synchronize, parents);
        for frontier in state.wire_frontier.values_mut() {
            *frontier = node;
        }
        state.global_barrier = Some(node);
        node
    }

    fn add_wire_synchronize(
        &mut self,
        qubits: &[LogicalQubit],
        state: &mut WorkloadState,
    ) -> NodeIndex {
        let mut parents = BTreeSet::new();
        parents.extend(state.global_barrier);
        parents.extend(
            qubits
                .iter()
                .filter_map(|logical| state.wire_frontier.get(logical).copied()),
        );
        let node = self.add_node(SabreNodeKind::Synchronize, parents);
        for &logical in qubits {
            state.wire_frontier.insert(logical, node);
        }
        node
    }

    fn add_control_flow(
        &mut self,
        flow: &ClassicalControlOp,
        state: &mut WorkloadState,
    ) -> Result<BTreeSet<LogicalQubit>, CompilerError> {
        let fork = self.add_global_synchronize(state);
        let branches = match flow {
            ClassicalControlOp::If(op) => {
                let mut branches = vec![op.then_body().operations()];
                if let Some(else_body) = op.else_body() {
                    branches.push(else_body.operations());
                }
                branches
            }
            ClassicalControlOp::While(op) => vec![op.body().operations()],
            ClassicalControlOp::For(op) => vec![op.body().operations()],
            ClassicalControlOp::Switch(op) => {
                let mut branches = op
                    .cases()
                    .iter()
                    .map(|case| case.body().operations())
                    .collect::<Vec<_>>();
                if let Some(default) = op.default() {
                    branches.push(default.operations());
                }
                branches
            }
            ClassicalControlOp::Break | ClassicalControlOp::Continue => Vec::new(),
        };

        let base_frontier = state.wire_frontier.clone();
        let mut cross_branch_frontier = base_frontier.clone();
        let mut all_touched = BTreeSet::new();
        let mut join_parents = BTreeSet::from([fork]);
        for branch in branches {
            let mut branch_state = WorkloadState {
                wire_frontier: cross_branch_frontier.clone(),
                global_barrier: Some(fork),
            };
            let branch_touched = self.add_operations(branch, &mut branch_state)?;
            for &logical in &branch_touched {
                if let Some(frontier) = branch_state.wire_frontier.get(&logical).copied() {
                    cross_branch_frontier.insert(logical, frontier);
                    join_parents.insert(frontier);
                }
            }
            if let Some(barrier) = branch_state.global_barrier
                && barrier != fork
            {
                join_parents.insert(barrier);
            }
            all_touched.extend(branch_touched);
        }

        let join = self.add_node(SabreNodeKind::Synchronize, join_parents);
        for frontier in state.wire_frontier.values_mut() {
            *frontier = join;
        }
        for &logical in &all_touched {
            state.wire_frontier.insert(logical, join);
        }
        state.global_barrier = Some(join);
        Ok(all_touched)
    }

    fn add_node(&mut self, kind: SabreNodeKind, parents: BTreeSet<NodeIndex>) -> NodeIndex {
        let node = self.graph.add_node(SabreNode {
            operations: Vec::new(),
            kind,
        });
        for parent in parents {
            if parent != node && self.graph.find_edge(parent, node).is_none() {
                self.graph.add_edge(parent, node, ());
            }
        }
        node
    }
}

enum Predecessors {
    AllUnmapped,
    Single(NodeIndex),
    Multiple(Vec<NodeIndex>),
}

fn kind_from_operation(operation: &Operation) -> Result<SabreNodeKind, CompilerError> {
    match &operation.instruction {
        Instruction::ClassicalControl(flow) => match flow {
            ClassicalControlOp::If(op) => Ok(SabreNodeKind::ControlFlow(SabreControlFlow::If {
                condition: op.condition().clone(),
                then_body: SabreDag::from_operations(op.then_body().operations())?,
                else_body: op
                    .else_body()
                    .map(|body| SabreDag::from_operations(body.operations()))
                    .transpose()?,
            })),
            ClassicalControlOp::While(op) => {
                Ok(SabreNodeKind::ControlFlow(SabreControlFlow::While {
                    condition: op.condition().clone(),
                    body: SabreDag::from_operations(op.body().operations())?,
                }))
            }
            ClassicalControlOp::For(op) => Ok(SabreNodeKind::ControlFlow(SabreControlFlow::For {
                var: op.var(),
                start: op.start().clone(),
                stop: op.stop().clone(),
                step: op.step().clone(),
                body: SabreDag::from_operations(op.body().operations())?,
            })),
            ClassicalControlOp::Switch(op) => {
                let cases = op
                    .cases()
                    .iter()
                    .map(|case| {
                        Ok(SabreSwitchCase {
                            value: case.value(),
                            body: SabreDag::from_operations(case.body().operations())?,
                        })
                    })
                    .collect::<Result<Vec<_>, CompilerError>>()?;
                let default = op
                    .default()
                    .map(|body| SabreDag::from_operations(body.operations()))
                    .transpose()?;
                Ok(SabreNodeKind::ControlFlow(SabreControlFlow::Switch {
                    target: op.target().clone(),
                    cases,
                    default,
                }))
            }
            ClassicalControlOp::Break | ClassicalControlOp::Continue => {
                Ok(SabreNodeKind::Synchronize)
            }
        },
        Instruction::Directive(_) | Instruction::Delay => Ok(SabreNodeKind::Synchronize),
        _ => match operation.qubits.len() {
            0 | 1 => Ok(SabreNodeKind::Synchronize),
            2 => Ok(SabreNodeKind::TwoQ([
                LogicalQubit::from_qubit(operation.qubits[0]),
                LogicalQubit::from_qubit(operation.qubits[1]),
            ])),
            arity => Err(CompilerError::InvalidInput(format!(
                "sabre requires operations with more than two qubits to be decomposed before routing; found {arity}-qubit operation {}",
                operation.instruction
            ))),
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::circuit::{Circuit, ClassicalExpr, Qubit};

    fn interaction_nodes(dag: &SabreDag) -> Vec<NodeIndex> {
        dag.graph
            .node_indices()
            .filter(|&node| matches!(dag.graph[node].kind, SabreNodeKind::TwoQ(_)))
            .collect()
    }

    #[test]
    fn refinement_orders_alternative_branches_that_share_a_wire() {
        let mut circuit = Circuit::new(3);
        circuit
            .if_else(
                ClassicalExpr::bool_literal(true),
                |then_body| {
                    then_body.cx(Qubit::new(0), Qubit::new(1))?;
                    Ok(())
                },
                |else_body| {
                    else_body.cx(Qubit::new(0), Qubit::new(2))?;
                    Ok(())
                },
            )
            .unwrap();

        let dag = SabreDag::refinement_workload(circuit.operations()).unwrap();
        let interactions = interaction_nodes(&dag);

        assert_eq!(interactions.len(), 2);
        assert!(
            dag.graph
                .find_edge(interactions[0], interactions[1])
                .is_some(),
            "stable branch order must serialize interactions sharing q0"
        );
    }

    #[test]
    fn refinement_keeps_disjoint_alternative_branches_parallel() {
        let mut circuit = Circuit::new(4);
        circuit
            .if_else(
                ClassicalExpr::bool_literal(true),
                |then_body| {
                    then_body.cx(Qubit::new(0), Qubit::new(1))?;
                    Ok(())
                },
                |else_body| {
                    else_body.cx(Qubit::new(2), Qubit::new(3))?;
                    Ok(())
                },
            )
            .unwrap();

        let dag = SabreDag::refinement_workload(circuit.operations()).unwrap();
        let interactions = interaction_nodes(&dag);

        assert_eq!(interactions.len(), 2);
        assert!(
            dag.graph
                .find_edge(interactions[0], interactions[1])
                .is_none()
        );
        assert!(
            dag.graph
                .find_edge(interactions[1], interactions[0])
                .is_none()
        );
    }

    #[test]
    fn refinement_recursively_includes_nested_body_interactions() {
        let mut circuit = Circuit::new(3);
        circuit
            .if_(ClassicalExpr::bool_literal(true), |body| {
                body.if_(ClassicalExpr::bool_literal(true), |nested| {
                    nested.cx(Qubit::new(0), Qubit::new(2))?;
                    Ok(())
                })?;
                Ok(())
            })
            .unwrap();

        let dag = SabreDag::refinement_workload(circuit.operations()).unwrap();
        assert_eq!(interaction_nodes(&dag).len(), 1);
    }

    #[test]
    fn refinement_preserves_explicit_wire_barrier_dependencies() {
        let mut circuit = Circuit::new(4);
        circuit.cx(Qubit::new(0), Qubit::new(1)).unwrap();
        circuit.barrier(vec![Qubit::new(0), Qubit::new(2)]).unwrap();
        circuit.cx(Qubit::new(2), Qubit::new(3)).unwrap();

        let dag = SabreDag::refinement_workload(circuit.operations()).unwrap();
        let interactions = interaction_nodes(&dag);
        let barrier = dag
            .graph
            .node_indices()
            .find(|&node| {
                matches!(dag.graph[node].kind, SabreNodeKind::Synchronize)
                    && dag.graph.find_edge(interactions[0], node).is_some()
                    && dag.graph.find_edge(node, interactions[1]).is_some()
            })
            .expect("wire barrier must synchronize the two interaction frontiers");

        assert!(matches!(
            dag.graph[barrier].kind,
            SabreNodeKind::Synchronize
        ));
    }
}
