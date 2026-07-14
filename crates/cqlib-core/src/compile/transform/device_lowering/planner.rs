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

//! Finite exact-qargs hypergraph planning for device instruction lowering.

use super::DeviceGateState;
use super::templates::{self, DirectionTemplate};
use crate::circuit::{Instruction, ParameterValue, StandardGate};
use crate::compile::error::{
    DeviceLoweringCandidateFailure, DeviceLoweringDependency, DeviceLoweringFailure,
};
use crate::compile::knowledge::{KnowledgeInstructionKey, RuleId, RuleKind, RuleLibrary};
use crate::device::Device;
use smallvec::SmallVec;
use std::cmp::Ordering;
use std::collections::{BinaryHeap, HashMap, HashSet};

const MAX_DIAGNOSTIC_CANDIDATES: usize = 16;

type StateId = usize;
type EdgeId = usize;

/// The physical cost minimized by device lowering.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, PartialOrd, Ord)]
pub(super) struct DevicePlanCost {
    pub(super) two_qubit_ops: u32,
    pub(super) total_ops: u32,
    // A well-founded tertiary metric makes equal-physical-cost one-child
    // equivalences settle after their dependencies and prevents cyclic plans.
    derivation_steps: u32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum PlanTemplate {
    Rule(RuleId),
    Direction(DirectionTemplate),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum PlanChoice {
    Native,
    Template(PlanTemplate),
}

#[derive(Debug, Clone)]
struct HyperEdge {
    parent: StateId,
    children: Vec<StateId>,
    template: PlanTemplate,
    stable_name: String,
}

#[derive(Debug, Clone, Copy, Default)]
struct PendingEdge {
    remaining_children: usize,
    accumulated_cost: DevicePlanCost,
}

#[derive(Debug, Clone)]
struct QueueEntry {
    state: StateId,
    cost: DevicePlanCost,
    choice: PlanChoice,
    priority: u8,
    stable_name: String,
}

impl PartialEq for QueueEntry {
    fn eq(&self, other: &Self) -> bool {
        self.state == other.state
            && self.cost == other.cost
            && self.priority == other.priority
            && self.stable_name == other.stable_name
    }
}

impl Eq for QueueEntry {}

impl PartialOrd for QueueEntry {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for QueueEntry {
    fn cmp(&self, other: &Self) -> Ordering {
        // BinaryHeap is a max-heap; reverse every comparison for a stable
        // minimum queue. Native leaves win an exact tie.
        other
            .cost
            .cmp(&self.cost)
            .then_with(|| other.priority.cmp(&self.priority))
            .then_with(|| other.stable_name.cmp(&self.stable_name))
            .then_with(|| other.state.cmp(&self.state))
    }
}

/// One solved plan table for the finite closure reachable from circuit roots.
pub(super) struct DevicePlanner<'a> {
    device: &'a Device,
    states: Vec<DeviceGateState>,
    state_ids: HashMap<DeviceGateState, StateId>,
    edges: Vec<HyperEdge>,
    plans: Vec<Option<PlanChoice>>,
    costs: Vec<Option<DevicePlanCost>>,
}

impl<'a> DevicePlanner<'a> {
    pub(super) fn build(
        device: &'a Device,
        library: &RuleLibrary,
        roots: impl IntoIterator<Item = DeviceGateState>,
    ) -> Result<Self, String> {
        let mut builder = GraphBuilder::new(library);
        for root in roots {
            builder.intern(root);
        }
        builder.expand()?;

        let mut planner = Self {
            device,
            plans: vec![None; builder.states.len()],
            costs: vec![None; builder.states.len()],
            states: builder.states,
            state_ids: builder.state_ids,
            edges: builder.edges,
        };
        planner.solve();
        Ok(planner)
    }

    pub(super) fn plan_for(&self, state: &DeviceGateState) -> Option<PlanChoice> {
        self.state_ids.get(state).and_then(|id| self.plans[*id])
    }

    pub(super) fn failure_for(&self, state: &DeviceGateState) -> DeviceLoweringFailure {
        let attempted_candidates = self
            .state_ids
            .get(state)
            .into_iter()
            .flat_map(|state_id| {
                self.edges
                    .iter()
                    .filter(move |edge| edge.parent == *state_id)
            })
            .take(MAX_DIAGNOSTIC_CANDIDATES)
            .map(|edge| {
                let mut seen = HashSet::new();
                let unsatisfied_dependencies = edge
                    .children
                    .iter()
                    .copied()
                    .filter(|child| self.plans[*child].is_none())
                    .filter(|child| seen.insert(*child))
                    .map(|child| DeviceLoweringDependency {
                        instruction: instruction_from_key(&self.states[child].instruction),
                        qargs: self.states[child].ordered_qargs.to_vec(),
                    })
                    .collect();
                DeviceLoweringCandidateFailure {
                    template: edge.stable_name.clone(),
                    unsatisfied_dependencies,
                }
            })
            .collect();

        DeviceLoweringFailure {
            instruction: instruction_from_key(&state.instruction),
            qargs: state.ordered_qargs.to_vec(),
            attempted_candidates,
        }
    }

    fn solve(&mut self) {
        let mut reverse_incidence = vec![Vec::<(EdgeId, u32)>::new(); self.states.len()];
        let mut pending = Vec::with_capacity(self.edges.len());
        let mut queue = BinaryHeap::new();

        for (state_id, state) in self.states.iter().enumerate() {
            let instruction = instruction_from_key(&state.instruction);
            if self
                .device
                .supports_native_instruction(&instruction, &state.ordered_qargs)
            {
                queue.push(QueueEntry {
                    state: state_id,
                    cost: DevicePlanCost {
                        two_qubit_ops: u32::from(state.ordered_qargs.len() == 2),
                        total_ops: 1,
                        derivation_steps: 0,
                    },
                    choice: PlanChoice::Native,
                    priority: 0,
                    stable_name: String::new(),
                });
            }
        }

        for (edge_id, edge) in self.edges.iter().enumerate() {
            let mut multiplicities = HashMap::<StateId, u32>::new();
            for child in &edge.children {
                *multiplicities.entry(*child).or_default() += 1;
            }
            pending.push(PendingEdge {
                remaining_children: multiplicities.len(),
                accumulated_cost: DevicePlanCost::default(),
            });
            for (child, multiplicity) in multiplicities {
                reverse_incidence[child].push((edge_id, multiplicity));
            }
            if edge.children.is_empty() {
                queue.push(QueueEntry {
                    state: edge.parent,
                    cost: DevicePlanCost {
                        derivation_steps: 1,
                        ..DevicePlanCost::default()
                    },
                    choice: PlanChoice::Template(edge.template),
                    priority: 1,
                    stable_name: edge.stable_name.clone(),
                });
            }
        }

        while let Some(entry) = queue.pop() {
            if self.plans[entry.state].is_some() {
                continue;
            }
            self.plans[entry.state] = Some(entry.choice);
            self.costs[entry.state] = Some(entry.cost);

            for &(edge_id, multiplicity) in &reverse_incidence[entry.state] {
                let edge_state = &mut pending[edge_id];
                edge_state.accumulated_cost += entry.cost.scaled(multiplicity);
                edge_state.remaining_children -= 1;
                if edge_state.remaining_children == 0 {
                    let edge = &self.edges[edge_id];
                    let mut cost = edge_state.accumulated_cost;
                    cost.derivation_steps = cost.derivation_steps.saturating_add(1);
                    queue.push(QueueEntry {
                        state: edge.parent,
                        cost,
                        choice: PlanChoice::Template(edge.template),
                        priority: 1,
                        stable_name: edge.stable_name.clone(),
                    });
                }
            }
        }
    }
}

impl DevicePlanCost {
    fn scaled(self, count: u32) -> Self {
        Self {
            two_qubit_ops: self.two_qubit_ops.saturating_mul(count),
            total_ops: self.total_ops.saturating_mul(count),
            derivation_steps: self.derivation_steps.saturating_mul(count),
        }
    }
}

impl std::ops::AddAssign for DevicePlanCost {
    fn add_assign(&mut self, rhs: Self) {
        self.two_qubit_ops = self.two_qubit_ops.saturating_add(rhs.two_qubit_ops);
        self.total_ops = self.total_ops.saturating_add(rhs.total_ops);
        self.derivation_steps = self.derivation_steps.saturating_add(rhs.derivation_steps);
    }
}

struct GraphBuilder<'a> {
    library: &'a RuleLibrary,
    states: Vec<DeviceGateState>,
    state_ids: HashMap<DeviceGateState, StateId>,
    edges: Vec<HyperEdge>,
}

impl<'a> GraphBuilder<'a> {
    fn new(library: &'a RuleLibrary) -> Self {
        Self {
            library,
            states: Vec::new(),
            state_ids: HashMap::new(),
            edges: Vec::new(),
        }
    }

    fn intern(&mut self, state: DeviceGateState) -> StateId {
        if let Some(id) = self.state_ids.get(&state) {
            return *id;
        }
        let id = self.states.len();
        self.states.push(state.clone());
        self.state_ids.insert(state, id);
        id
    }

    fn expand(&mut self) -> Result<(), String> {
        let mut cursor = 0;
        while cursor < self.states.len() {
            let parent_state = self.states[cursor].clone();
            let mut candidates = self.rule_candidates(&parent_state)?;
            candidates.extend(
                templates::candidates(&parent_state)
                    .into_iter()
                    .map(|template| {
                        (
                            template.stable_name(),
                            PlanTemplate::Direction(template),
                            template.child_states(&parent_state),
                        )
                    }),
            );
            candidates.sort_by(|lhs, rhs| lhs.0.cmp(&rhs.0));

            for (stable_name, template, child_states) in candidates {
                let children = child_states
                    .into_iter()
                    .map(|child| self.intern(child))
                    .collect();
                self.edges.push(HyperEdge {
                    parent: cursor,
                    children,
                    template,
                    stable_name,
                });
            }
            cursor += 1;
        }
        Ok(())
    }

    fn rule_candidates(
        &self,
        parent: &DeviceGateState,
    ) -> Result<Vec<(String, PlanTemplate, Vec<DeviceGateState>)>, String> {
        let instruction = instruction_from_key(&parent.instruction);
        let rule_ids = self
            .library
            .candidates_for_first_instruction(&instruction)
            .map_err(|error| error.to_string())?;
        let mut candidates = Vec::new();

        for &rule_id in rule_ids {
            let Some(metadata) = self.library.metadata(rule_id) else {
                return Err(format!("missing device-lowering rule metadata {rule_id:?}"));
            };
            if !matches!(
                metadata.kind,
                RuleKind::Decompose | RuleKind::HardwareNative
            ) {
                continue;
            }
            let Some(rule) = self.library.get(rule_id) else {
                return Err(format!("missing device-lowering rule {rule_id:?}"));
            };
            if rule.operations.len() != 1
                || rule
                    .conditions
                    .as_ref()
                    .is_some_and(|conditions| !conditions.is_empty())
                || !source_params_are_generic_and_distinct(&rule.operations[0].params)
            {
                continue;
            }
            let source = &rule.operations[0];
            if source.qubits.len() != parent.ordered_qargs.len() {
                continue;
            }
            let qarg_bindings = source
                .qubits
                .iter()
                .copied()
                .zip(parent.ordered_qargs.iter().copied())
                .collect::<HashMap<_, _>>();
            let mut children = Vec::new();
            let mut valid = true;
            for item in &rule.target {
                let Some(key) = KnowledgeInstructionKey::from_instruction(&item.instruction) else {
                    valid = false;
                    break;
                };
                if matches!(key, KnowledgeInstructionKey::Standard(StandardGate::GPhase)) {
                    continue;
                }
                let ordered_qargs = item
                    .qubits
                    .iter()
                    .map(|qubit| qarg_bindings.get(qubit).copied())
                    .collect::<Option<SmallVec<[_; 2]>>>();
                let Some(ordered_qargs) = ordered_qargs else {
                    valid = false;
                    break;
                };
                children.push(DeviceGateState {
                    instruction: key,
                    ordered_qargs,
                });
            }
            if valid {
                candidates.push((
                    format!("rule:{}", rule.name),
                    PlanTemplate::Rule(rule_id),
                    children,
                ));
            }
        }
        Ok(candidates)
    }
}

fn source_params_are_generic_and_distinct(params: &Option<SmallVec<[ParameterValue; 1]>>) -> bool {
    let mut symbols = HashSet::new();
    params.as_deref().unwrap_or(&[]).iter().all(|param| {
        let ParameterValue::Param(parameter) = param else {
            return false;
        };
        parameter
            .as_symbol()
            .is_some_and(|symbol| symbols.insert(symbol))
    })
}

pub(super) fn instruction_from_key(key: &KnowledgeInstructionKey) -> Instruction {
    match key {
        KnowledgeInstructionKey::Standard(gate) => Instruction::Standard(*gate),
        KnowledgeInstructionKey::McGate(gate) => Instruction::McGate(Box::new(gate.clone())),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::compile::knowledge::rule::{Rule, RuleItem};
    use crate::device::PhysicalQubit;

    fn item(gate: StandardGate, qubits: &[u32]) -> RuleItem {
        RuleItem::standard(gate, qubits, Vec::new())
    }

    fn rule(
        name: &str,
        source: StandardGate,
        source_qubits: &[u32],
        target: Vec<RuleItem>,
    ) -> Rule {
        Rule::new(name, vec![item(source, source_qubits)], target)
    }

    fn state(gate: StandardGate, qargs: &[u32]) -> DeviceGateState {
        DeviceGateState::standard(
            gate,
            qargs.iter().copied().map(PhysicalQubit::new).collect(),
        )
    }

    fn library(rules: Vec<Rule>) -> RuleLibrary {
        RuleLibrary::from_rules(rules, RuleKind::Decompose).unwrap()
    }

    #[test]
    fn dead_two_state_cycle_terminates_without_a_plan() {
        let library = library(vec![
            rule(
                "x_to_h",
                StandardGate::X,
                &[0],
                vec![item(StandardGate::H, &[0])],
            ),
            rule(
                "h_to_x",
                StandardGate::H,
                &[0],
                vec![item(StandardGate::X, &[0])],
            ),
        ]);
        let device = Device::line("dead-cycle", 1).unwrap();
        let root = state(StandardGate::X, &[0]);

        let planner = DevicePlanner::build(&device, &library, [root.clone()]).unwrap();

        assert!(planner.plan_for(&root).is_none());
    }

    #[test]
    fn cycle_with_native_exit_propagates_a_finite_plan() {
        let library = library(vec![
            rule(
                "x_to_h",
                StandardGate::X,
                &[0],
                vec![item(StandardGate::H, &[0])],
            ),
            rule(
                "h_to_x",
                StandardGate::H,
                &[0],
                vec![item(StandardGate::X, &[0])],
            ),
        ]);
        let device = Device::line("cycle-exit", 1)
            .unwrap()
            .with_native_gates(vec![Instruction::Standard(StandardGate::H)])
            .unwrap();
        let root = state(StandardGate::X, &[0]);

        let planner = DevicePlanner::build(&device, &library, [root.clone()]).unwrap();

        assert!(matches!(
            planner.plan_for(&root),
            Some(PlanChoice::Template(PlanTemplate::Rule(_)))
        ));
    }

    #[test]
    fn repeated_child_occurrences_contribute_multiplicity_to_cost() {
        let library = library(vec![rule(
            "swap_to_three_cx",
            StandardGate::SWAP,
            &[0, 1],
            vec![
                item(StandardGate::CX, &[0, 1]),
                item(StandardGate::CX, &[0, 1]),
                item(StandardGate::CX, &[0, 1]),
            ],
        )]);
        let device = Device::line("multiplicity", 2)
            .unwrap()
            .with_native_gates(vec![Instruction::Standard(StandardGate::CX)])
            .unwrap();
        let root = state(StandardGate::SWAP, &[0, 1]);

        let planner = DevicePlanner::build(&device, &library, [root.clone()]).unwrap();
        let root_id = planner.state_ids[&root];

        assert_eq!(
            planner.costs[root_id],
            Some(DevicePlanCost {
                two_qubit_ops: 3,
                total_ops: 3,
                derivation_steps: 1,
            })
        );
    }

    #[test]
    fn ordered_qargs_have_independent_capability_states() {
        let library = RuleLibrary::new();
        let device = Device::line("one-qubit", 1)
            .unwrap()
            .with_native_gates(vec![Instruction::Standard(StandardGate::X)])
            .unwrap();
        let supported = state(StandardGate::X, &[0]);
        let unsupported = state(StandardGate::X, &[1]);

        let planner =
            DevicePlanner::build(&device, &library, [supported.clone(), unsupported.clone()])
                .unwrap();

        assert_eq!(planner.plan_for(&supported), Some(PlanChoice::Native));
        assert!(planner.plan_for(&unsupported).is_none());
    }

    #[test]
    fn cost_prefers_fewer_two_qubit_leaves_before_total_gate_count() {
        let library = library(vec![
            rule(
                "swap_to_cx",
                StandardGate::SWAP,
                &[0, 1],
                vec![item(StandardGate::CX, &[0, 1])],
            ),
            rule(
                "swap_to_h_pair",
                StandardGate::SWAP,
                &[0, 1],
                vec![item(StandardGate::H, &[0]), item(StandardGate::H, &[1])],
            ),
        ]);
        let device = Device::line("cost-order", 2)
            .unwrap()
            .with_native_gates(vec![
                Instruction::Standard(StandardGate::H),
                Instruction::Standard(StandardGate::CX),
            ])
            .unwrap();
        let root = state(StandardGate::SWAP, &[0, 1]);

        let planner = DevicePlanner::build(&device, &library, [root.clone()]).unwrap();
        let Some(PlanChoice::Template(PlanTemplate::Rule(rule_id))) = planner.plan_for(&root)
        else {
            panic!("expected a rule plan");
        };

        assert_eq!(library.get(rule_id).unwrap().name, "swap_to_h_pair");
    }

    #[test]
    fn equal_cost_rule_choice_is_stable_across_library_order() {
        let make_rules = || {
            vec![
                rule(
                    "a_x_to_h",
                    StandardGate::X,
                    &[0],
                    vec![item(StandardGate::H, &[0])],
                ),
                rule(
                    "b_x_to_z",
                    StandardGate::X,
                    &[0],
                    vec![item(StandardGate::Z, &[0])],
                ),
            ]
        };
        let mut reversed = make_rules();
        reversed.reverse();
        let libraries = [library(make_rules()), library(reversed)];
        let device = Device::line("stable-tie", 1)
            .unwrap()
            .with_native_gates(vec![
                Instruction::Standard(StandardGate::H),
                Instruction::Standard(StandardGate::Z),
            ])
            .unwrap();
        let root = state(StandardGate::X, &[0]);

        for library in &libraries {
            let planner = DevicePlanner::build(&device, library, [root.clone()]).unwrap();
            let Some(PlanChoice::Template(PlanTemplate::Rule(rule_id))) = planner.plan_for(&root)
            else {
                panic!("expected a rule plan");
            };
            assert_eq!(library.get(rule_id).unwrap().name, "a_x_to_h");
        }
    }

    #[test]
    fn direction_template_preserves_exact_ordered_qargs() {
        let parent = state(StandardGate::CX, &[0, 1]);
        let children = DirectionTemplate::Cx.child_states(&parent);

        assert_eq!(children[2], state(StandardGate::CX, &[1, 0]));
        assert_eq!(
            children[0].ordered_qargs.as_slice(),
            &[PhysicalQubit::new(0)]
        );
        assert_eq!(
            children[1].ordered_qargs.as_slice(),
            &[PhysicalQubit::new(1)]
        );
    }
}
