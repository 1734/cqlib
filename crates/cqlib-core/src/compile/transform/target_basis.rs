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

//! Deterministic target-basis lowering.
//!
//! This pass owns correctness-oriented translation to a physical instruction
//! basis. It uses single-operation decomposition rules as a planning graph and
//! then lowers each gate-like operation linearly. Optimizing rewrites such as
//! cancellation, merge, and commutation remain in the knowledge rewriter.

use crate::circuit::{
    Circuit, ClassicalControlOp, Instruction, Operation, Parameter, ParameterValue, Qubit,
    StandardGate, ValueClassicalControlOp, ValueControlBody, ValueInstruction, ValueOperation,
    ValueSwitchCase,
};
use crate::compile::CompilerError;
use crate::compile::knowledge::rule::{Rule, RuleItem};
use crate::compile::knowledge::{
    KnowledgeInstructionKey, MatchedReplacement, RuleId, RuleKind, RuleLibrary,
};
use crate::compile::transform::rebuild::{CircuitRebuildContext, ClassicalRemap};
use crate::compile::transform::{CircuitAnalysis, TransformResult, Transformer};
use smallvec::{SmallVec, smallvec};
use std::collections::{HashMap, HashSet};

const MAX_PLANNING_ROUNDS: usize = 128;

/// Lowers a circuit to an explicit gate-like target instruction basis.
#[derive(Debug, Clone)]
pub struct TargetBasisLowerer {
    target_basis: Vec<Instruction>,
    plans: LoweringPlans,
}

/// Canonical identity of a standard-gate target basis.
///
/// The signature is order- and duplicate-insensitive because target-basis
/// lowering treats the configured instructions as a capability set. `GPhase`
/// is omitted because it is implicit and has no effect on generated templates.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct TargetBasisSignature {
    gates: Vec<u8>,
}

/// Cost of lowering a fixed standard-gate operation sequence to a target basis.
///
/// Global-phase operations are intentionally excluded: they have no physical
/// gate cost and do not occupy a qubit wire.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct TargetBasisCost {
    pub two_qubit_ops: usize,
    pub depth: usize,
    pub total_ops: usize,
    pub parameterized_ops: usize,
}

/// Reusable, exact target-basis cost evaluator.
///
/// The evaluator owns the same lowering plans used by [`TargetBasisLowerer`].
/// It therefore measures the concrete output of the active rule library rather
/// than maintaining a separate heuristic approximation for synthesis choices.
#[derive(Debug, Clone)]
pub struct TargetBasisCostModel {
    signature: TargetBasisSignature,
    lowerer: TargetBasisLowerer,
}

impl PartialEq for TargetBasisCostModel {
    fn eq(&self, other: &Self) -> bool {
        self.signature == other.signature
    }
}

impl Eq for TargetBasisCostModel {}

#[derive(Debug, Clone)]
struct LoweringPlans {
    physical_keys: HashSet<KnowledgeInstructionKey>,
    plan_by_key: HashMap<KnowledgeInstructionKey, GatePlan>,
    target_display: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct GatePlan {
    rule_id: RuleId,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
struct PlanCost {
    rank: usize,
    total_ops: usize,
    two_qubit_ops: usize,
    parameterized_ops: usize,
    rule_id: usize,
}

#[derive(Debug, Clone)]
struct LowerableOperation {
    instruction: Instruction,
    qubits: SmallVec<[Qubit; 3]>,
    params: SmallVec<[ParameterValue; 1]>,
    label: Option<Box<str>>,
}

enum LoweringTarget<'a> {
    TopLevel {
        output: &'a mut Vec<ValueOperation>,
        phase_delta: &'a mut Parameter,
    },
    ControlFlowBody {
        output: &'a mut Vec<ValueOperation>,
        phase_delta: &'a mut Parameter,
    },
}

struct CircuitLowerer<'a> {
    source: &'a Circuit,
    plans: &'a LoweringPlans,
    library: &'a RuleLibrary,
    rebuild: CircuitRebuildContext,
    changed: bool,
}

impl TargetBasisLowerer {
    /// Creates a target-basis lowerer from a non-empty gate-like basis.
    pub fn new(target_basis: Vec<Instruction>) -> Result<Self, CompilerError> {
        if target_basis.is_empty() {
            return Err(CompilerError::InvalidInput(
                "target-basis lowering requires a non-empty target basis".to_string(),
            ));
        }

        let library = RuleLibrary::builtin_rules()
            .map_err(|err| CompilerError::InvariantViolation(err.to_string()))?;
        let plans = LoweringPlans::build(&target_basis, library)?;

        Ok(Self {
            target_basis,
            plans,
        })
    }

    /// Returns the configured target instruction basis in insertion order.
    pub fn target_basis(&self) -> &[Instruction] {
        &self.target_basis
    }
}

impl TargetBasisSignature {
    /// Builds a canonical signature from standard target gates.
    pub fn from_standard_gates(gates: &[StandardGate]) -> Self {
        let mut gates = gates
            .iter()
            .filter(|gate| **gate != StandardGate::GPhase)
            .map(|gate| *gate as u8)
            .collect::<Vec<_>>();
        gates.sort_unstable();
        gates.dedup();
        Self { gates }
    }
}

impl TargetBasisCostModel {
    /// Builds an exact cost model for a non-empty standard-gate target basis.
    pub fn new(target_basis: Vec<Instruction>) -> Result<Self, CompilerError> {
        let gates = target_basis
            .iter()
            .map(|instruction| match instruction {
                Instruction::Standard(gate) => Ok(*gate),
                _ => Err(CompilerError::InvalidInput(format!(
                    "target-basis cost model requires standard instructions, got {instruction:?}"
                ))),
            })
            .collect::<Result<Vec<_>, _>>()?;
        let lowerer = TargetBasisLowerer::new(target_basis)?;
        Ok(Self {
            signature: TargetBasisSignature::from_standard_gates(&gates),
            lowerer,
        })
    }

    /// Returns the canonical target identity used by synthesis caches.
    pub const fn signature(&self) -> &TargetBasisSignature {
        &self.signature
    }

    /// Lowers fixed standard-gate operations and returns the exact resulting
    /// target-basis cost.
    ///
    /// The supplied operations must reference only `qubits`, carry fixed
    /// finite parameters, and contain no classical control. These constraints
    /// match the numeric two-qubit synthesis and resynthesis callers.
    pub fn cost_of_fixed_operations(
        &self,
        qubits: Vec<Qubit>,
        operations: Vec<ValueOperation>,
    ) -> Result<TargetBasisCost, CompilerError> {
        let source = Circuit::from_operations(qubits, operations, None, None)
            .map_err(CompilerError::Circuit)?;
        let lowered = self.lowerer.transform(&source, None)?.circuit;
        let mut cost = TargetBasisCost::default();
        let mut depths = HashMap::new();
        for operation in lowered.operations() {
            let Instruction::Standard(gate) = operation.instruction else {
                return Err(CompilerError::InvariantViolation(
                    "target-basis lowering emitted a non-standard operation while estimating cost"
                        .to_string(),
                ));
            };
            if gate == StandardGate::GPhase {
                continue;
            }
            cost.total_ops += 1;
            if !operation.params.is_empty() {
                cost.parameterized_ops += 1;
            }
            if operation.qubits.len() == 2 {
                cost.two_qubit_ops += 1;
            }
            if operation.qubits.is_empty() {
                continue;
            }
            let next_depth = operation
                .qubits
                .iter()
                .filter_map(|qubit| depths.get(qubit))
                .max()
                .copied()
                .unwrap_or(0)
                + 1;
            for qubit in &operation.qubits {
                depths.insert(*qubit, next_depth);
            }
            cost.depth = cost.depth.max(next_depth);
        }
        Ok(cost)
    }
}

impl Transformer for TargetBasisLowerer {
    fn name(&self) -> &'static str {
        "target_basis_lowering"
    }

    fn transform(
        &self,
        circuit: &Circuit,
        _analysis: Option<&CircuitAnalysis>,
    ) -> Result<TransformResult, CompilerError> {
        let library = RuleLibrary::builtin_rules()
            .map_err(|err| CompilerError::InvariantViolation(err.to_string()))?;
        CircuitLowerer::run(circuit, &self.plans, library)
    }
}

impl LoweringPlans {
    fn build(target_basis: &[Instruction], library: &RuleLibrary) -> Result<Self, CompilerError> {
        let mut physical_keys = HashSet::with_capacity(target_basis.len());
        for instruction in target_basis {
            let Some(key) = KnowledgeInstructionKey::from_instruction(instruction) else {
                return Err(CompilerError::InvalidInput(format!(
                    "unsupported target-basis instruction {instruction:?}"
                )));
            };
            physical_keys.insert(key);
        }

        let target_display = target_basis
            .iter()
            .map(Instruction::to_string)
            .collect::<Vec<_>>()
            .join(", ");
        let mut cost_by_key = physical_keys
            .iter()
            .cloned()
            .map(|key| {
                let cost = if is_implicit_key(&key) {
                    PlanCost::zero(key_rule_sort_value(&key))
                } else {
                    PlanCost {
                        rank: 0,
                        total_ops: 1,
                        two_qubit_ops: key_num_qubits(&key).is_some_and(|num| num == 2) as usize,
                        parameterized_ops: key_num_params(&key)
                            .is_some_and(|num_params| num_params > 0)
                            as usize,
                        rule_id: key_rule_sort_value(&key),
                    }
                };
                (key, cost)
            })
            .collect::<HashMap<_, _>>();
        cost_by_key.insert(
            KnowledgeInstructionKey::Standard(StandardGate::GPhase),
            PlanCost::zero(usize::MAX),
        );

        let candidates = collect_candidate_rules(library);
        let mut plan_by_key = HashMap::new();

        for _ in 0..MAX_PLANNING_ROUNDS {
            let mut changed = false;
            for &rule_id in &candidates {
                let rule = library.get(rule_id).ok_or_else(|| {
                    CompilerError::InvariantViolation(format!(
                        "missing target-basis lowering rule id {}",
                        rule_id.as_usize()
                    ))
                })?;
                let source_item = &rule.operations[0];
                let source_key =
                    KnowledgeInstructionKey::from_instruction(&source_item.instruction)
                        .ok_or_else(|| {
                            CompilerError::InvariantViolation(format!(
                                "unsupported source instruction in target-basis rule {:?}",
                                source_item.instruction
                            ))
                        })?;
                if physical_keys.contains(&source_key) || is_implicit_key(&source_key) {
                    continue;
                }

                let Some(candidate_cost) =
                    rewrite_cost(&rule.target, &cost_by_key, rule_id.as_usize())
                else {
                    continue;
                };
                let current = cost_by_key.get(&source_key).copied();
                if current.is_none_or(|current| candidate_cost < current) {
                    cost_by_key.insert(source_key.clone(), candidate_cost);
                    plan_by_key.insert(source_key, GatePlan { rule_id });
                    changed = true;
                }
            }

            if !changed {
                return Ok(Self {
                    physical_keys,
                    plan_by_key,
                    target_display,
                });
            }
        }

        Err(CompilerError::InvariantViolation(
            "target-basis lowering planning did not converge".to_string(),
        ))
    }

    fn is_physical(&self, key: &KnowledgeInstructionKey) -> bool {
        self.physical_keys.contains(key)
    }

    fn plan_for(&self, key: &KnowledgeInstructionKey) -> Option<GatePlan> {
        self.plan_by_key.get(key).copied()
    }
}

impl PlanCost {
    const fn zero(rule_id: usize) -> Self {
        Self {
            rank: 0,
            total_ops: 0,
            two_qubit_ops: 0,
            parameterized_ops: 0,
            rule_id,
        }
    }
}

impl<'a> CircuitLowerer<'a> {
    fn run(
        source: &'a Circuit,
        plans: &'a LoweringPlans,
        library: &'a RuleLibrary,
    ) -> Result<TransformResult, CompilerError> {
        let rebuild = CircuitRebuildContext::new(source);
        let root_classical = rebuild.root_classical().clone();
        let mut lowerer = Self {
            source,
            plans,
            library,
            rebuild,
            changed: false,
        };
        let mut phase_delta = Parameter::from(0.0);
        let mut operations = Vec::with_capacity(source.operations().len());
        lowerer.lower_sequence(
            source.operations(),
            &root_classical,
            LoweringTarget::TopLevel {
                output: &mut operations,
                phase_delta: &mut phase_delta,
            },
        )?;

        let global_phase = &source.global_phase() + &phase_delta;
        let circuit = lowerer
            .rebuild
            .finish(source.qubits(), operations, global_phase)?;

        Ok(TransformResult {
            circuit,
            changed: lowerer.changed,
        })
    }

    fn lower_sequence(
        &mut self,
        operations: &[Operation],
        classical_remap: &ClassicalRemap,
        mut target: LoweringTarget<'_>,
    ) -> Result<(), CompilerError> {
        for operation in operations {
            self.lower_operation(operation, classical_remap, &mut target)?;
        }
        Ok(())
    }

    fn lower_operation(
        &mut self,
        operation: &Operation,
        classical_remap: &ClassicalRemap,
        target: &mut LoweringTarget<'_>,
    ) -> Result<(), CompilerError> {
        match &operation.instruction {
            Instruction::Standard(_) | Instruction::McGate(_) => {
                let params =
                    CircuitRebuildContext::resolve_source_params(self.source, &operation.params)?;
                let op = LowerableOperation {
                    instruction: operation.instruction.clone(),
                    qubits: operation.qubits.clone(),
                    params,
                    label: operation.label.clone(),
                };
                self.lower_gate_like(op, target)
            }
            Instruction::ClassicalControl(control) => {
                let instruction = self.lower_control_flow(control, classical_remap)?;
                let qubits = instruction.used_qubits().into_iter().collect();
                self.push_value_operation(
                    ValueOperation {
                        instruction: ValueInstruction::ClassicalControl(instruction),
                        qubits,
                        params: SmallVec::new(),
                        label: operation.label.clone(),
                    },
                    target,
                );
                Ok(())
            }
            Instruction::UnitaryGate(_) | Instruction::CircuitGate(_) => {
                Err(CompilerError::InvalidInput(format!(
                    "cannot lower {} to target basis {{{}}}: definitions and unitaries must be decomposed before target-basis translation",
                    operation.instruction, self.plans.target_display
                )))
            }
            Instruction::ClassicalData(_) | Instruction::Directive(_) | Instruction::Delay => {
                let operation = self.rebuild.remap_preserved_operation(
                    self.source,
                    operation,
                    classical_remap,
                )?;
                self.push_value_operation(operation, target);
                Ok(())
            }
        }
    }

    fn lower_gate_like(
        &mut self,
        operation: LowerableOperation,
        target: &mut LoweringTarget<'_>,
    ) -> Result<(), CompilerError> {
        let key =
            KnowledgeInstructionKey::from_instruction(&operation.instruction).ok_or_else(|| {
                CompilerError::InvariantViolation(format!(
                    "missing target-basis key for gate-like instruction {}",
                    operation.instruction
                ))
            })?;
        if is_implicit_key(&key) {
            self.changed = true;
            self.accumulate_phase(gphase_param(&operation)?, target);
            return Ok(());
        }
        if self.plans.is_physical(&key) {
            self.emit_physical_gate(operation, target);
            return Ok(());
        }

        let Some(plan) = self.plans.plan_for(&key) else {
            return Err(CompilerError::InvalidInput(format!(
                "cannot lower {} to target basis {{{}}}: no decomposition plan",
                operation.instruction, self.plans.target_display
            )));
        };
        let rule = self.library.get(plan.rule_id).ok_or_else(|| {
            CompilerError::InvariantViolation(format!(
                "missing target-basis lowering rule id {}",
                plan.rule_id.as_usize()
            ))
        })?;
        let replacements = instantiate_single_source_rule(rule, &operation)?;
        self.changed = true;
        for replacement in replacements {
            self.lower_replacement(replacement, target)?;
        }
        Ok(())
    }

    fn lower_replacement(
        &mut self,
        replacement: MatchedReplacement,
        target: &mut LoweringTarget<'_>,
    ) -> Result<(), CompilerError> {
        self.lower_gate_like(
            LowerableOperation {
                instruction: replacement.instruction,
                qubits: replacement.qubits,
                params: SmallVec::from_vec(replacement.params.into_vec()),
                label: None,
            },
            target,
        )
    }

    fn emit_physical_gate(
        &mut self,
        operation: LowerableOperation,
        target: &mut LoweringTarget<'_>,
    ) {
        self.push_value_operation(
            ValueOperation {
                instruction: ValueInstruction::from_instruction(operation.instruction),
                qubits: operation.qubits,
                params: operation.params,
                label: operation.label,
            },
            target,
        );
    }

    fn lower_control_flow(
        &mut self,
        control: &ClassicalControlOp,
        classical_remap: &ClassicalRemap,
    ) -> Result<ValueClassicalControlOp, CompilerError> {
        Ok(match control {
            ClassicalControlOp::If(op) => {
                let mut then_body = Vec::with_capacity(op.then_body().operations().len());
                let mut then_phase = Parameter::from(0.0);
                self.lower_sequence(
                    op.then_body().operations(),
                    classical_remap,
                    LoweringTarget::ControlFlowBody {
                        output: &mut then_body,
                        phase_delta: &mut then_phase,
                    },
                )?;
                self.prepend_body_phase(&mut then_body, then_phase);

                let else_body = op
                    .else_body()
                    .map(|body| {
                        let mut lowered = Vec::with_capacity(body.operations().len());
                        let mut phase = Parameter::from(0.0);
                        self.lower_sequence(
                            body.operations(),
                            classical_remap,
                            LoweringTarget::ControlFlowBody {
                                output: &mut lowered,
                                phase_delta: &mut phase,
                            },
                        )?;
                        self.prepend_body_phase(&mut lowered, phase);
                        Ok::<_, CompilerError>(ValueControlBody::new(lowered))
                    })
                    .transpose()?;

                ValueClassicalControlOp::If {
                    condition: classical_remap.remap_expr(op.condition())?,
                    then_body: ValueControlBody::new(then_body),
                    else_body,
                }
            }
            ClassicalControlOp::While(op) => {
                let mut body = Vec::with_capacity(op.body().operations().len());
                let mut phase = Parameter::from(0.0);
                self.lower_sequence(
                    op.body().operations(),
                    classical_remap,
                    LoweringTarget::ControlFlowBody {
                        output: &mut body,
                        phase_delta: &mut phase,
                    },
                )?;
                self.prepend_body_phase(&mut body, phase);
                ValueClassicalControlOp::While {
                    condition: classical_remap.remap_expr(op.condition())?,
                    body: ValueControlBody::new(body),
                }
            }
            ClassicalControlOp::For(op) => {
                let mut body = Vec::with_capacity(op.body().operations().len());
                let mut phase = Parameter::from(0.0);
                self.lower_sequence(
                    op.body().operations(),
                    classical_remap,
                    LoweringTarget::ControlFlowBody {
                        output: &mut body,
                        phase_delta: &mut phase,
                    },
                )?;
                self.prepend_body_phase(&mut body, phase);
                ValueClassicalControlOp::For {
                    var: classical_remap.remap_var(op.var())?,
                    start: classical_remap.remap_expr(op.start())?,
                    stop: classical_remap.remap_expr(op.stop())?,
                    step: classical_remap.remap_expr(op.step())?,
                    body: ValueControlBody::new(body),
                }
            }
            ClassicalControlOp::Switch(op) => {
                let cases = op
                    .cases()
                    .iter()
                    .map(|case| {
                        let mut lowered = Vec::with_capacity(case.body().operations().len());
                        let mut phase = Parameter::from(0.0);
                        self.lower_sequence(
                            case.body().operations(),
                            classical_remap,
                            LoweringTarget::ControlFlowBody {
                                output: &mut lowered,
                                phase_delta: &mut phase,
                            },
                        )?;
                        self.prepend_body_phase(&mut lowered, phase);
                        Ok::<_, CompilerError>(ValueSwitchCase::new(
                            case.value(),
                            ValueControlBody::new(lowered),
                        ))
                    })
                    .collect::<Result<Vec<_>, _>>()?;
                let default = op
                    .default()
                    .map(|body| {
                        let mut lowered = Vec::with_capacity(body.operations().len());
                        let mut phase = Parameter::from(0.0);
                        self.lower_sequence(
                            body.operations(),
                            classical_remap,
                            LoweringTarget::ControlFlowBody {
                                output: &mut lowered,
                                phase_delta: &mut phase,
                            },
                        )?;
                        self.prepend_body_phase(&mut lowered, phase);
                        Ok::<_, CompilerError>(ValueControlBody::new(lowered))
                    })
                    .transpose()?;
                ValueClassicalControlOp::Switch {
                    target: classical_remap.remap_expr(op.target())?,
                    cases,
                    default,
                }
            }
            ClassicalControlOp::Break => ValueClassicalControlOp::Break,
            ClassicalControlOp::Continue => ValueClassicalControlOp::Continue,
        })
    }

    fn prepend_body_phase(&mut self, body: &mut Vec<ValueOperation>, phase: Parameter) {
        if phase.is_zero() {
            return;
        }
        self.changed = true;
        body.insert(
            0,
            ValueOperation {
                instruction: ValueInstruction::from_instruction(Instruction::Standard(
                    StandardGate::GPhase,
                )),
                qubits: SmallVec::new(),
                params: smallvec![ParameterValue::from(phase)],
                label: None,
            },
        );
    }

    fn push_value_operation(&mut self, operation: ValueOperation, target: &mut LoweringTarget<'_>) {
        match target {
            LoweringTarget::TopLevel { output, .. }
            | LoweringTarget::ControlFlowBody { output, .. } => output.push(operation),
        }
    }

    fn accumulate_phase(&mut self, phase: Parameter, target: &mut LoweringTarget<'_>) {
        match target {
            LoweringTarget::TopLevel { phase_delta, .. }
            | LoweringTarget::ControlFlowBody { phase_delta, .. } => {
                **phase_delta = &**phase_delta + &phase;
            }
        }
    }
}

fn collect_candidate_rules(library: &RuleLibrary) -> Vec<RuleId> {
    [RuleKind::Decompose, RuleKind::HardwareNative]
        .into_iter()
        .flat_map(|kind| library.rules_by_kind(kind).iter().copied())
        .filter(|rule_id| {
            let Some(rule) = library.get(*rule_id) else {
                return false;
            };
            rule.operations.len() == 1
                && rule
                    .conditions
                    .as_ref()
                    .is_none_or(|conditions| conditions.is_empty())
                && source_params_are_directly_instantiable(&rule.operations[0])
        })
        .collect()
}

fn rewrite_cost(
    target: &[crate::compile::knowledge::rule::RuleItem],
    cost_by_key: &HashMap<KnowledgeInstructionKey, PlanCost>,
    rule_id: usize,
) -> Option<PlanCost> {
    let mut rank = 0usize;
    let mut total_ops = 0usize;
    let mut two_qubit_ops = 0usize;
    let mut parameterized_ops = 0usize;

    for item in target {
        let key = KnowledgeInstructionKey::from_instruction(&item.instruction)?;
        if is_implicit_key(&key) {
            continue;
        }
        let cost = cost_by_key.get(&key).copied()?;
        rank = rank.max(cost.rank);
        total_ops = total_ops.saturating_add(cost.total_ops);
        two_qubit_ops = two_qubit_ops.saturating_add(cost.two_qubit_ops);
        parameterized_ops = parameterized_ops.saturating_add(cost.parameterized_ops);
    }

    Some(PlanCost {
        rank: rank.saturating_add(1),
        total_ops,
        two_qubit_ops,
        parameterized_ops,
        rule_id,
    })
}

fn source_params_are_directly_instantiable(source: &RuleItem) -> bool {
    source
        .params
        .as_deref()
        .unwrap_or(&[])
        .iter()
        .all(|param| match param {
            ParameterValue::Fixed(_) => false,
            ParameterValue::Param(parameter) => parameter.as_symbol().is_some(),
        })
}

fn instantiate_single_source_rule(
    rule: &Rule,
    operation: &LowerableOperation,
) -> Result<Vec<MatchedReplacement>, CompilerError> {
    debug_assert_eq!(rule.operations.len(), 1);
    let source = rule.operations.first().ok_or_else(|| {
        CompilerError::InvariantViolation(format!(
            "target-basis lowering rule {} has no source operation",
            rule.name
        ))
    })?;

    let mut qubit_bindings = HashMap::with_capacity(source.qubits.len());
    for (&rule_qubit, &actual_qubit) in source.qubits.iter().zip(&operation.qubits) {
        qubit_bindings.insert(rule_qubit, actual_qubit);
    }

    let source_params = source.params.as_deref().unwrap_or(&[]);
    if source_params.len() != operation.params.len() {
        return Err(CompilerError::InvariantViolation(format!(
            "planned target-basis rule {} parameter arity does not match {}",
            rule.name, operation.instruction
        )));
    }

    let mut parameter_bindings: HashMap<String, Parameter> = HashMap::new();
    for (pattern, actual) in source_params.iter().zip(&operation.params) {
        match pattern {
            ParameterValue::Fixed(expected) => {
                let actual = parameter_value_to_parameter(actual);
                if !Parameter::from(*expected)
                    .provably_equal(&actual, crate::compile::PARAMETER_EQ_TOLERANCE)
                {
                    return Err(CompilerError::InvariantViolation(format!(
                        "planned target-basis rule {} fixed parameter does not match {}",
                        rule.name, operation.instruction
                    )));
                }
            }
            ParameterValue::Param(pattern) => {
                let Some(symbol) = pattern.as_symbol() else {
                    return Err(CompilerError::InvariantViolation(format!(
                        "target-basis rule {} uses a non-symbol source parameter",
                        rule.name
                    )));
                };
                let actual = parameter_value_to_parameter(actual);
                if let Some(bound) = parameter_bindings.get(&symbol) {
                    if !bound.provably_equal(&actual, crate::compile::PARAMETER_EQ_TOLERANCE) {
                        return Err(CompilerError::InvariantViolation(format!(
                            "planned target-basis rule {} repeated parameter does not match {}",
                            rule.name, operation.instruction
                        )));
                    }
                } else {
                    parameter_bindings.insert(symbol, actual);
                }
            }
        }
    }

    let mut replacements = Vec::with_capacity(rule.target.len());
    for item in &rule.target {
        let key =
            KnowledgeInstructionKey::from_instruction(&item.instruction).ok_or_else(|| {
                CompilerError::InvariantViolation(format!(
                    "unsupported target instruction in target-basis rule {:?}",
                    item.instruction
                ))
            })?;
        let qubits = item
            .qubits
            .iter()
            .map(|rule_qubit| {
                qubit_bindings.get(rule_qubit).copied().ok_or_else(|| {
                    CompilerError::InvariantViolation(format!(
                        "target-basis rule {} references unbound qubit {}",
                        rule.name, rule_qubit
                    ))
                })
            })
            .collect::<Result<SmallVec<[Qubit; 3]>, _>>()?;
        let params = item
            .params
            .as_deref()
            .unwrap_or(&[])
            .iter()
            .map(|param| instantiate_rule_param(param, &parameter_bindings, &rule.name))
            .collect::<Result<SmallVec<[ParameterValue; 3]>, _>>()?;

        replacements.push(MatchedReplacement {
            instruction: item.instruction.clone(),
            qubits,
            params,
            key,
        });
    }

    Ok(replacements)
}

fn instantiate_rule_param(
    param: &ParameterValue,
    bindings: &HashMap<String, Parameter>,
    rule_name: &str,
) -> Result<ParameterValue, CompilerError> {
    let parameter = match param {
        ParameterValue::Fixed(value) => Parameter::from(*value),
        ParameterValue::Param(parameter) => {
            for symbol in parameter.get_symbols() {
                if !bindings.contains_key(&symbol) {
                    return Err(CompilerError::InvariantViolation(format!(
                        "target-basis rule {rule_name} references unbound parameter {symbol}"
                    )));
                }
            }
            parameter.substitute_many(bindings)
        }
    };
    Ok(ParameterValue::from(parameter))
}

fn parameter_value_to_parameter(param: &ParameterValue) -> Parameter {
    match param {
        ParameterValue::Fixed(value) => Parameter::from(*value),
        ParameterValue::Param(parameter) => parameter.clone(),
    }
}

fn gphase_param(operation: &LowerableOperation) -> Result<Parameter, CompilerError> {
    let phase = operation.params.first().ok_or_else(|| {
        CompilerError::InvariantViolation("GPhase operation must contain one parameter".to_string())
    })?;
    Ok(match phase {
        ParameterValue::Fixed(value) => Parameter::from(*value),
        ParameterValue::Param(parameter) => parameter.clone(),
    })
}

fn is_implicit_key(key: &KnowledgeInstructionKey) -> bool {
    matches!(key, KnowledgeInstructionKey::Standard(StandardGate::GPhase))
}

fn key_num_qubits(key: &KnowledgeInstructionKey) -> Option<usize> {
    Some(match key {
        KnowledgeInstructionKey::Standard(gate) => gate.num_qubits(),
        KnowledgeInstructionKey::McGate(gate) => gate.num_qubits(),
    })
}

fn key_num_params(key: &KnowledgeInstructionKey) -> Option<usize> {
    Some(match key {
        KnowledgeInstructionKey::Standard(gate) => gate.num_params(),
        KnowledgeInstructionKey::McGate(gate) => gate.num_params(),
    })
}

fn key_rule_sort_value(key: &KnowledgeInstructionKey) -> usize {
    match key {
        KnowledgeInstructionKey::Standard(gate) => *gate as usize,
        KnowledgeInstructionKey::McGate(gate) => 10_000 + gate.num_qubits(),
    }
}

#[cfg(test)]
#[path = "./target_basis_test.rs"]
mod target_basis_test;
