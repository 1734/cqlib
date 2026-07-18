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
//
// This file is an original modification of Cqlib.

//! Device-aware cost and placement domains for exact two-qubit synthesis.
//!
//! The public two-qubit synthesis target describes a logical gate basis. A
//! device target is different: before layout it changes the set of physical
//! terminals available to routing, while after layout it must be evaluated on
//! one exact ordered pair. This module keeps that distinction internal and
//! evaluates source and synthesized sequences through the same exact-qargs
//! plans later used by device lowering.

use crate::circuit::{Circuit, Instruction, Qubit, StandardGate, ValueInstruction, ValueOperation};
use crate::compile::CompilerError;
use crate::compile::device_planning::{CalibrationEstimator, DeviceGateState, NativePlanCatalog};
use crate::device::{Device, PhysicalQubit};
use smallvec::smallvec;
use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet, VecDeque};
use std::sync::Arc;

/// Explicit interpretation of circuit qubit identifiers during device synthesis.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum DeviceSynthesisPlacement {
    /// Circuit qubits are logical; compare candidates over physical placement domains.
    PreLayoutEnvelope,
    /// Circuit qubits are routed physical identifiers and may be converted explicitly.
    ExactPhysical,
}

pub(crate) use crate::compile::device_planning::DevicePhysicalCost;

/// Placement coverage compared before pre-layout physical costs.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, PartialOrd, Ord)]
pub(crate) struct DeviceCoverageKey {
    /// Eligible movement components with no executable ordered pair.
    pub(crate) uncovered_components: usize,
    /// Eligible ordered topology pairs outside the candidate's terminal domain.
    pub(crate) infeasible_ordered_pairs: usize,
}

pub(crate) type OrderedPairDomain = BTreeSet<[PhysicalQubit; 2]>;

/// Pre-layout feasibility and worst physical cost for one operation sequence.
#[derive(Debug, Clone)]
pub(crate) struct DevicePreLayoutEvaluation {
    pub(crate) domain: OrderedPairDomain,
    pub(crate) coverage: DeviceCoverageKey,
    pub(crate) worst_cost: DevicePhysicalCost,
}

#[derive(Debug)]
struct DeviceTwoQubitSynthesisData {
    placement: DeviceSynthesisPlacement,
    eligible_pairs: BTreeSet<[PhysicalQubit; 2]>,
    eligible_components: BTreeSet<usize>,
    component_by_qubit: HashMap<PhysicalQubit, usize>,
    native_backends: BTreeMap<[PhysicalQubit; 2], HashSet<StandardGate>>,
    catalog: NativePlanCatalog,
    estimator: CalibrationEstimator,
}

/// Pass-local exact device planning data shared by all matrices and blocks.
#[derive(Debug, Clone)]
pub(crate) struct DeviceTwoQubitSynthesisContext {
    data: Arc<DeviceTwoQubitSynthesisData>,
}

impl DeviceTwoQubitSynthesisContext {
    /// Builds one batched catalog for a synthesis pass.
    pub(crate) fn build(
        device: &Device,
        circuit: &Circuit,
        placement: DeviceSynthesisPlacement,
    ) -> Result<Self, CompilerError> {
        let physical_qubits = device.usable_qubits().collect::<Vec<_>>();
        let topology_pairs = ordered_topology_pairs(device, &physical_qubits);
        let ordered_pairs = match placement {
            DeviceSynthesisPlacement::PreLayoutEnvelope => topology_pairs.clone(),
            DeviceSynthesisPlacement::ExactPhysical => {
                collect_exact_physical_pairs(circuit.operations())
            }
        };
        let root_qubits = match placement {
            DeviceSynthesisPlacement::PreLayoutEnvelope => physical_qubits.clone(),
            DeviceSynthesisPlacement::ExactPhysical => circuit
                .qubits()
                .iter()
                .copied()
                .map(PhysicalQubit::from_qubit)
                .collect(),
        };
        let source_gates = collect_source_standard_gates(circuit.operations());
        let roots = catalog_roots(&root_qubits, &ordered_pairs, &source_gates);
        let catalog = NativePlanCatalog::build(device, roots)?;
        let estimator = CalibrationEstimator::from_device(device, &physical_qubits);
        let (component_by_qubit, eligible_components, eligible_pairs) = match placement {
            DeviceSynthesisPlacement::PreLayoutEnvelope => {
                let (component_by_qubit, _component_sizes) =
                    movement_components(&physical_qubits, &topology_pairs, &catalog);
                // Terminal pairs remain useful even when no lowerable SWAP
                // joins their movement components. Excluding such pairs would
                // reject valid fixed placements on devices that can execute a
                // 2Q gate but cannot synthesize SWAP.
                let eligible_components = topology_pairs
                    .iter()
                    .flat_map(|pair| pair.iter())
                    .filter_map(|qubit| component_by_qubit.get(qubit).copied())
                    .collect::<BTreeSet<_>>();
                let eligible_pairs = topology_pairs.iter().copied().collect();
                (component_by_qubit, eligible_components, eligible_pairs)
            }
            DeviceSynthesisPlacement::ExactPhysical => {
                (HashMap::new(), BTreeSet::new(), BTreeSet::new())
            }
        };
        let native_backends = native_backend_map(device, &ordered_pairs);

        Ok(Self {
            data: Arc::new(DeviceTwoQubitSynthesisData {
                placement,
                eligible_pairs,
                eligible_components,
                component_by_qubit,
                native_backends,
                catalog,
                estimator,
            }),
        })
    }

    pub(crate) fn placement(&self) -> DeviceSynthesisPlacement {
        self.data.placement
    }

    /// KAK backends directly native somewhere relevant to this request.
    pub(crate) fn native_two_qubit_backends(&self, qubits: [Qubit; 2]) -> HashSet<StandardGate> {
        match self.data.placement {
            DeviceSynthesisPlacement::PreLayoutEnvelope => self
                .data
                .eligible_pairs
                .iter()
                .filter_map(|pair| self.data.native_backends.get(pair))
                .flat_map(|gates| gates.iter().copied())
                .collect(),
            DeviceSynthesisPlacement::ExactPhysical => {
                let pair = qubits.map(PhysicalQubit::from_qubit);
                [pair, [pair[1], pair[0]]]
                    .into_iter()
                    .filter_map(|ordered| self.data.native_backends.get(&ordered))
                    .flat_map(|gates| gates.iter().copied())
                    .collect()
            }
        }
    }

    pub(crate) fn evaluate_pre_layout(
        &self,
        operations: &[ValueOperation],
        logical_qubits: [Qubit; 2],
    ) -> Option<DevicePreLayoutEvaluation> {
        if self.data.placement != DeviceSynthesisPlacement::PreLayoutEnvelope {
            return None;
        }
        let domain = self.feasible_domain(operations, logical_qubits);
        let worst_cost = self.worst_cost_on_domain(operations, logical_qubits, &domain)?;
        Some(DevicePreLayoutEvaluation {
            coverage: self.coverage_key(&domain),
            domain,
            worst_cost,
        })
    }

    pub(crate) fn feasible_domain(
        &self,
        operations: &[ValueOperation],
        logical_qubits: [Qubit; 2],
    ) -> OrderedPairDomain {
        self.data
            .eligible_pairs
            .iter()
            .copied()
            .filter(|pair| {
                self.cost_on_pair(operations, logical_qubits, *pair)
                    .is_some()
            })
            .collect()
    }

    pub(crate) fn worst_cost_on_domain(
        &self,
        operations: &[ValueOperation],
        logical_qubits: [Qubit; 2],
        domain: &OrderedPairDomain,
    ) -> Option<DevicePhysicalCost> {
        domain
            .iter()
            .filter_map(|pair| self.cost_on_pair(operations, logical_qubits, *pair))
            .max_by(|left, right| left.compare(*right))
    }

    pub(crate) fn exact_cost(
        &self,
        operations: &[ValueOperation],
        physical_qubits: [Qubit; 2],
    ) -> Option<DevicePhysicalCost> {
        if self.data.placement != DeviceSynthesisPlacement::ExactPhysical {
            return None;
        }
        let pair = physical_qubits.map(PhysicalQubit::from_qubit);
        self.cost_on_pair(operations, physical_qubits, pair)
    }

    /// Costs one flat operation sequence on the physical qargs carried by the
    /// operations themselves.
    ///
    /// Every gate-like operation is expanded through the same selected native
    /// plan used by [`DeviceLowerer`](crate::compile::transform::DeviceLowerer).
    /// Control flow and non-gate instructions are deliberately outside this
    /// sequence-level API and make the sequence unavailable for costing.
    pub(crate) fn exact_sequence_cost(
        &self,
        operations: &[ValueOperation],
    ) -> Option<DevicePhysicalCost> {
        if self.data.placement != DeviceSynthesisPlacement::ExactPhysical {
            return None;
        }

        let mut leaves = Vec::new();
        let mut aggregate = self.data.estimator.identity_cost();
        for operation in operations {
            let ValueInstruction::Instruction(instruction) = &operation.instruction else {
                return None;
            };
            if matches!(instruction, Instruction::Standard(StandardGate::GPhase)) {
                continue;
            }
            let ordered_qargs = operation
                .qubits
                .iter()
                .copied()
                .map(PhysicalQubit::from_qubit)
                .collect();
            let state = DeviceGateState::from_instruction(instruction, ordered_qargs)?;
            let summary = self.data.catalog.summary(&state)?;
            aggregate = aggregate.combine(self.data.estimator.cost(summary));
            leaves.extend(summary.leaves.iter().cloned());
        }
        Some(
            self.data
                .estimator
                .schedule_physical_cost(&leaves, aggregate),
        )
    }

    fn coverage_key(&self, domain: &OrderedPairDomain) -> DeviceCoverageKey {
        let covered_components = domain
            .iter()
            .flat_map(|pair| pair.iter())
            .filter_map(|qubit| self.data.component_by_qubit.get(qubit).copied())
            .collect::<HashSet<_>>();
        DeviceCoverageKey {
            uncovered_components: self
                .data
                .eligible_components
                .iter()
                .filter(|component| !covered_components.contains(component))
                .count(),
            infeasible_ordered_pairs: self.data.eligible_pairs.len().saturating_sub(domain.len()),
        }
    }

    fn cost_on_pair(
        &self,
        operations: &[ValueOperation],
        circuit_qubits: [Qubit; 2],
        physical_pair: [PhysicalQubit; 2],
    ) -> Option<DevicePhysicalCost> {
        let mut leaves = Vec::new();
        let mut aggregate = self.data.estimator.identity_cost();
        for operation in operations {
            let ValueInstruction::Instruction(instruction) = &operation.instruction else {
                return None;
            };
            if matches!(instruction, Instruction::Standard(StandardGate::GPhase)) {
                continue;
            }
            let ordered_qargs = operation
                .qubits
                .iter()
                .map(|qubit| {
                    if *qubit == circuit_qubits[0] {
                        Some(physical_pair[0])
                    } else if *qubit == circuit_qubits[1] {
                        Some(physical_pair[1])
                    } else {
                        None
                    }
                })
                .collect::<Option<_>>()?;
            let state = DeviceGateState::from_instruction(instruction, ordered_qargs)?;
            let summary = self.data.catalog.summary(&state)?;
            aggregate = aggregate.combine(self.data.estimator.cost(summary));
            leaves.extend(summary.leaves.iter().cloned());
        }
        Some(
            self.data
                .estimator
                .schedule_physical_cost(&leaves, aggregate),
        )
    }
}

fn ordered_topology_pairs(
    device: &Device,
    physical_qubits: &[PhysicalQubit],
) -> Vec<[PhysicalQubit; 2]> {
    let usable = physical_qubits.iter().copied().collect::<HashSet<_>>();
    let mut pairs = BTreeSet::new();
    for (left, right) in device.topology().undirected_edges() {
        if usable.contains(&left) && usable.contains(&right) {
            pairs.insert([left, right]);
            pairs.insert([right, left]);
        }
    }
    pairs.into_iter().collect()
}

fn catalog_roots(
    physical_qubits: &[PhysicalQubit],
    ordered_pairs: &[[PhysicalQubit; 2]],
    source_gates: &HashSet<StandardGate>,
) -> Vec<DeviceGateState> {
    let mut unary_gates = StandardGate::all()
        .iter()
        .copied()
        .filter(|gate| gate.num_qubits() == 1)
        .collect::<HashSet<_>>();
    let mut pair_gates = source_gates
        .iter()
        .copied()
        .filter(|gate| gate.num_qubits() == 2)
        .collect::<HashSet<_>>();
    pair_gates.extend([
        StandardGate::CX,
        StandardGate::CY,
        StandardGate::CZ,
        StandardGate::RXX,
        StandardGate::RYY,
        StandardGate::RZZ,
        StandardGate::SWAP,
    ]);
    unary_gates.insert(StandardGate::U);

    let mut roots = Vec::new();
    for &qubit in physical_qubits {
        for &gate in &unary_gates {
            roots.push(DeviceGateState::standard(gate, smallvec![qubit]));
        }
    }
    for &pair in ordered_pairs {
        for &gate in &pair_gates {
            roots.push(DeviceGateState::standard(gate, smallvec![pair[0], pair[1]]));
        }
    }
    roots
}

fn collect_source_standard_gates(
    operations: &[crate::circuit::Operation],
) -> HashSet<StandardGate> {
    use crate::circuit::ClassicalControlOp;

    let mut gates = HashSet::new();
    for operation in operations {
        match &operation.instruction {
            Instruction::Standard(gate) => {
                gates.insert(*gate);
            }
            Instruction::ClassicalControl(control) => match control {
                ClassicalControlOp::If(op) => {
                    gates.extend(collect_source_standard_gates(op.then_body().operations()));
                    if let Some(body) = op.else_body() {
                        gates.extend(collect_source_standard_gates(body.operations()));
                    }
                }
                ClassicalControlOp::While(op) => {
                    gates.extend(collect_source_standard_gates(op.body().operations()));
                }
                ClassicalControlOp::For(op) => {
                    gates.extend(collect_source_standard_gates(op.body().operations()));
                }
                ClassicalControlOp::Switch(op) => {
                    for case in op.cases() {
                        gates.extend(collect_source_standard_gates(case.body().operations()));
                    }
                    if let Some(body) = op.default() {
                        gates.extend(collect_source_standard_gates(body.operations()));
                    }
                }
                ClassicalControlOp::Break | ClassicalControlOp::Continue => {}
            },
            _ => {}
        }
    }
    gates
}

fn collect_exact_physical_pairs(
    operations: &[crate::circuit::Operation],
) -> Vec<[PhysicalQubit; 2]> {
    use crate::circuit::ClassicalControlOp;

    let mut pairs = BTreeSet::new();
    for operation in operations {
        match &operation.instruction {
            Instruction::Standard(gate)
                if gate.num_qubits() == 2 && operation.qubits.len() == 2 =>
            {
                let pair = operation
                    .qubits
                    .iter()
                    .copied()
                    .map(PhysicalQubit::from_qubit)
                    .collect::<Vec<_>>();
                pairs.insert([pair[0], pair[1]]);
                pairs.insert([pair[1], pair[0]]);
            }
            Instruction::ClassicalControl(control) => match control {
                ClassicalControlOp::If(op) => {
                    pairs.extend(collect_exact_physical_pairs(op.then_body().operations()));
                    if let Some(body) = op.else_body() {
                        pairs.extend(collect_exact_physical_pairs(body.operations()));
                    }
                }
                ClassicalControlOp::While(op) => {
                    pairs.extend(collect_exact_physical_pairs(op.body().operations()));
                }
                ClassicalControlOp::For(op) => {
                    pairs.extend(collect_exact_physical_pairs(op.body().operations()));
                }
                ClassicalControlOp::Switch(op) => {
                    for case in op.cases() {
                        pairs.extend(collect_exact_physical_pairs(case.body().operations()));
                    }
                    if let Some(body) = op.default() {
                        pairs.extend(collect_exact_physical_pairs(body.operations()));
                    }
                }
                ClassicalControlOp::Break | ClassicalControlOp::Continue => {}
            },
            _ => {}
        }
    }
    pairs.into_iter().collect()
}

fn native_backend_map(
    device: &Device,
    ordered_pairs: &[[PhysicalQubit; 2]],
) -> BTreeMap<[PhysicalQubit; 2], HashSet<StandardGate>> {
    const BACKENDS: [StandardGate; 6] = [
        StandardGate::CX,
        StandardGate::CY,
        StandardGate::CZ,
        StandardGate::RXX,
        StandardGate::RYY,
        StandardGate::RZZ,
    ];
    ordered_pairs
        .iter()
        .copied()
        .map(|pair| {
            let gates = BACKENDS
                .into_iter()
                .filter(|gate| {
                    device
                        .supports_native_instruction(&Instruction::Standard(*gate), pair.as_slice())
                })
                .collect();
            (pair, gates)
        })
        .collect()
}

fn movement_components(
    physical_qubits: &[PhysicalQubit],
    ordered_pairs: &[[PhysicalQubit; 2]],
    catalog: &NativePlanCatalog,
) -> (HashMap<PhysicalQubit, usize>, Vec<usize>) {
    let mut neighbors = HashMap::<PhysicalQubit, Vec<PhysicalQubit>>::new();
    for &pair in ordered_pairs {
        let state = DeviceGateState::standard(StandardGate::SWAP, smallvec![pair[0], pair[1]]);
        if catalog.summary(&state).is_some() {
            neighbors.entry(pair[0]).or_default().push(pair[1]);
            neighbors.entry(pair[1]).or_default().push(pair[0]);
        }
    }

    let mut component_by_qubit = HashMap::new();
    let mut component_sizes = Vec::new();
    for &start in physical_qubits {
        if component_by_qubit.contains_key(&start) {
            continue;
        }
        let component = component_sizes.len();
        let mut size = 0;
        let mut queue = VecDeque::from([start]);
        component_by_qubit.insert(start, component);
        while let Some(qubit) = queue.pop_front() {
            size += 1;
            for &neighbor in neighbors.get(&qubit).into_iter().flatten() {
                if component_by_qubit.insert(neighbor, component).is_none() {
                    queue.push_back(neighbor);
                }
            }
        }
        component_sizes.push(size);
    }
    (component_by_qubit, component_sizes)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::circuit::{ParameterValue, UnitaryGate};
    use crate::compile::device_planning::cost::MetricAvailability;
    use crate::compile::transform::decompose::unitary::TwoQubitUnitaryDecomposeBasis;
    use crate::compile::transform::decompose::unitary::unitary_2q::{
        plan_numeric_2q_unitary_for_device, select_device_unitary_candidate,
    };
    use crate::device::{EdgeProp, InstructionProp};

    #[test]
    fn pre_layout_prefers_broad_family_over_single_calibrated_edge() {
        let p1 = PhysicalQubit::new(1);
        let p2 = PhysicalQubit::new(2);
        let mut device = Device::bidirectional_line("coverage", 4)
            .unwrap()
            .with_native_gates(vec![
                Instruction::Standard(StandardGate::U),
                Instruction::Standard(StandardGate::CX),
            ])
            .unwrap();
        for (control, target) in [(p1, p2), (p2, p1)] {
            device
                .add_edge_properties(
                    control,
                    target,
                    EdgeProp::new()
                        .with_native_instruction(InstructionProp::new(
                            Instruction::Standard(StandardGate::CZ),
                            0.0001,
                        ))
                        .unwrap(),
                )
                .unwrap();
        }
        let matrix = StandardGate::SWAP.matrix(&[]).unwrap().into_owned();
        let gate = UnitaryGate::new("SWAP", 2, 0)
            .with_matrix(matrix.clone())
            .unwrap();
        let mut circuit = Circuit::new(4);
        circuit
            .unitary(gate, vec![Qubit::new(0), Qubit::new(3)])
            .unwrap();
        let context = DeviceTwoQubitSynthesisContext::build(
            &device,
            &circuit,
            DeviceSynthesisPlacement::PreLayoutEnvelope,
        )
        .unwrap();
        let qubits = [Qubit::new(0), Qubit::new(3)];
        let candidates = plan_numeric_2q_unitary_for_device(&matrix, qubits, &context).unwrap();
        let selected = select_device_unitary_candidate(candidates, qubits, &context).unwrap();

        assert_eq!(selected.backend, TwoQubitUnitaryDecomposeBasis::Cx);
        assert!(selected.operations.iter().all(|operation| {
            !matches!(
                operation.instruction,
                ValueInstruction::Instruction(Instruction::Standard(StandardGate::CZ))
            )
        }));
        assert!(selected.operations.iter().all(|operation| {
            operation
                .params
                .iter()
                .all(|param| matches!(param, ParameterValue::Fixed(_)))
        }));
    }

    #[test]
    fn equal_physical_cost_is_not_a_strict_improvement() {
        let cost = DevicePhysicalCost {
            native_two_qubit_ops: 3,
            native_two_qubit_depth: 3,
            error: MetricAvailability::Disabled,
            total_native_depth: 7,
            native_total_ops: 11,
            duration: MetricAvailability::Disabled,
            makespan: MetricAvailability::Disabled,
        };

        assert!(!cost.strictly_better_than(cost));
    }

    #[test]
    fn exact_sequence_cost_supports_one_qubit_only_circuits() {
        let device = Device::line("one-qubit-sequence", 1)
            .unwrap()
            .with_native_gates(vec![Instruction::Standard(StandardGate::U)])
            .unwrap()
            .with_default_single_qubit_error(0.001);
        let q0 = Qubit::new(0);
        let mut circuit = Circuit::new(1);
        circuit.u(q0, 0.3, -0.2, 0.7).unwrap();
        let context = DeviceTwoQubitSynthesisContext::build(
            &device,
            &circuit,
            DeviceSynthesisPlacement::ExactPhysical,
        )
        .unwrap();
        let operations = vec![ValueOperation {
            instruction: ValueInstruction::from_instruction(Instruction::Standard(StandardGate::U)),
            qubits: smallvec![q0],
            params: smallvec![
                ParameterValue::Fixed(0.3),
                ParameterValue::Fixed(-0.2),
                ParameterValue::Fixed(0.7),
            ],
            label: None,
        }];

        let cost = context.exact_sequence_cost(&operations).unwrap();

        assert_eq!(cost.native_two_qubit_ops, 0);
        assert_eq!(cost.native_total_ops, 1);
        assert_eq!(cost.total_native_depth, 1);
    }
}
