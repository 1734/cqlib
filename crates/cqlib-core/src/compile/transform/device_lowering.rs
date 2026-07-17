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

//! Exact ordered-qargs lowering to a concrete device instruction set.
//!
//! Routing deliberately uses an undirected connectivity projection. This
//! transform runs after routing and closes the remaining ISA gap: every
//! gate-like operation is either retained as an exact native capability or
//! lowered through a finite, recursively verified plan whose leaves are exact
//! device capabilities. A separate terminal device verifier remains the final
//! workflow safety boundary.

use crate::circuit::{
    Circuit, ClassicalControlOp, Instruction, Operation, Parameter, ParameterValue, Qubit,
    StandardGate, ValueClassicalControlOp, ValueControlBody, ValueInstruction, ValueOperation,
    ValueSwitchCase,
};
use crate::compile::CompilerError;
use crate::compile::device_planning::{
    DeviceGateState, DevicePlanner, DevicePlannerError, DirectionTemplate, PlanChoice, PlanId,
    PlanTemplate,
};
use crate::compile::knowledge::{
    ConcreteOperationView, KnowledgeInstructionKey, RuleLibrary, instantiate_target,
    rule_matches_operations,
};
use crate::compile::transform::rebuild::{CircuitRebuildContext, ClassicalRemap};
use crate::compile::transform::{CircuitAnalysis, TransformResult, Transformer};
use crate::device::{Device, PhysicalQubit};
use smallvec::{SmallVec, smallvec};
use std::collections::HashSet;

#[derive(Debug, Clone)]
pub(super) struct LowerableOperation {
    pub(super) instruction: Instruction,
    pub(super) qubits: SmallVec<[Qubit; 3]>,
    pub(super) params: SmallVec<[ParameterValue; 1]>,
    pub(super) label: Option<Box<str>>,
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

/// Lowers a routed physical circuit to one device's exact native instruction
/// capabilities, including ordered qargs and local capability overrides.
#[derive(Debug, Clone, Copy)]
pub struct DeviceLowerer<'a> {
    device: &'a Device,
}

impl<'a> DeviceLowerer<'a> {
    /// Creates a lowerer borrowing one immutable device capability model.
    pub const fn new(device: &'a Device) -> Self {
        Self { device }
    }

    /// Returns the device whose exact capabilities define lowering leaves.
    pub const fn device(&self) -> &'a Device {
        self.device
    }
}

impl Transformer for DeviceLowerer<'_> {
    fn name(&self) -> &'static str {
        "device_lowering"
    }

    fn transform(
        &self,
        circuit: &Circuit,
        _analysis: Option<&CircuitAnalysis>,
    ) -> Result<TransformResult, CompilerError> {
        let library = RuleLibrary::builtin_rules()
            .map_err(|error| CompilerError::InvariantViolation(error.to_string()))?;
        let mut roots = collect_root_states(circuit)?;
        roots.sort_by_key(DeviceGateState::stable_sort_key);
        roots.dedup();
        let planner = DevicePlanner::build(self.device, library, roots)
            .map_err(DevicePlannerError::into_compiler_error)?;
        DeviceCircuitLowerer::run(circuit, self.device, library, &planner)
    }
}

struct DeviceCircuitLowerer<'a> {
    source: &'a Circuit,
    device: &'a Device,
    library: &'a RuleLibrary,
    planner: &'a DevicePlanner<'a>,
    rebuild: CircuitRebuildContext,
    changed: bool,
}

impl<'a> DeviceCircuitLowerer<'a> {
    fn run(
        source: &'a Circuit,
        device: &'a Device,
        library: &'a RuleLibrary,
        planner: &'a DevicePlanner<'a>,
    ) -> Result<TransformResult, CompilerError> {
        let rebuild = CircuitRebuildContext::new(source);
        let root_classical = rebuild.root_classical().clone();
        let mut lowerer = Self {
            source,
            device,
            library,
            planner,
            rebuild,
            changed: false,
        };
        let mut operations = Vec::with_capacity(source.operations().len());
        let mut phase_delta = Parameter::from(0.0);
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
                self.lower_gate_like(
                    LowerableOperation {
                        instruction: operation.instruction.clone(),
                        qubits: operation.qubits.clone(),
                        params,
                        label: operation.label.clone(),
                    },
                    target,
                )
            }
            Instruction::ClassicalControl(control) => {
                let instruction = self.lower_control_flow(control, classical_remap)?;
                self.push_operation(
                    ValueOperation {
                        qubits: instruction.used_qubits().into_iter().collect(),
                        instruction: ValueInstruction::ClassicalControl(instruction),
                        params: SmallVec::new(),
                        label: operation.label.clone(),
                    },
                    target,
                );
                Ok(())
            }
            Instruction::UnitaryGate(_) | Instruction::CircuitGate(_) => {
                Err(CompilerError::InvalidInput(format!(
                    "cannot lower {} to device instructions: definitions and unitaries must be decomposed first",
                    operation.instruction
                )))
            }
            Instruction::ClassicalData(_) | Instruction::Directive(_) | Instruction::Delay => {
                let operation = self.rebuild.remap_preserved_operation(
                    self.source,
                    operation,
                    classical_remap,
                )?;
                self.push_operation(operation, target);
                Ok(())
            }
        }
    }

    fn lower_gate_like(
        &mut self,
        operation: LowerableOperation,
        target: &mut LoweringTarget<'_>,
    ) -> Result<(), CompilerError> {
        if matches!(
            operation.instruction,
            Instruction::Standard(StandardGate::GPhase)
        ) {
            self.changed = true;
            self.accumulate_phase(gphase_param(&operation)?, target);
            return Ok(());
        }

        let ordered_qargs = operation
            .qubits
            .iter()
            .copied()
            .map(PhysicalQubit::from_qubit)
            .collect();
        let state = DeviceGateState::from_instruction(&operation.instruction, ordered_qargs)
            .ok_or_else(|| {
                CompilerError::InvariantViolation(format!(
                    "missing device-lowering key for {}",
                    operation.instruction
                ))
            })?;
        let Some(plan) = self.planner.selected_plan_for(&state) else {
            return Err(CompilerError::DeviceLoweringFailed(
                self.planner.failure_for(&state),
            ));
        };
        self.lower_gate_like_with_plan(operation, state, plan, target)
    }

    fn lower_gate_like_with_plan(
        &mut self,
        operation: LowerableOperation,
        state: DeviceGateState,
        plan: PlanId,
        target: &mut LoweringTarget<'_>,
    ) -> Result<(), CompilerError> {
        let planned_state = self.planner.state_for_plan(plan).ok_or_else(|| {
            CompilerError::InvariantViolation(format!("unknown selected device plan {plan:?}"))
        })?;
        if planned_state != &state {
            return Err(CompilerError::InvariantViolation(format!(
                "device plan {plan:?} targets {planned_state:?}, but lowering requested {state:?}"
            )));
        }
        let choice = self.planner.choice_for_plan(plan).ok_or_else(|| {
            CompilerError::InvariantViolation(format!("device plan {plan:?} has no choice"))
        })?;
        match choice {
            PlanChoice::Native => {
                if !self
                    .device
                    .supports_native_instruction(&operation.instruction, &state.ordered_qargs)
                {
                    return Err(CompilerError::InvariantViolation(format!(
                        "device planner selected unsupported native leaf {} on {:?}",
                        operation.instruction, state.ordered_qargs
                    )));
                }
                self.push_operation(
                    ValueOperation {
                        instruction: ValueInstruction::from_instruction(operation.instruction),
                        qubits: operation.qubits,
                        params: operation.params,
                        label: operation.label,
                    },
                    target,
                );
            }
            PlanChoice::Template(template) => {
                self.changed = true;
                let child_plans = self
                    .planner
                    .children_for_plan(plan)
                    .ok_or_else(|| {
                        CompilerError::InvariantViolation(format!(
                            "device plan {plan:?} has no child plan list"
                        ))
                    })?
                    .to_vec();
                let replacements = match template {
                    PlanTemplate::Rule(rule_id) => {
                        let rule = self.library.get(rule_id).ok_or_else(|| {
                            CompilerError::InvariantViolation(format!(
                                "missing selected device-lowering rule {rule_id:?}"
                            ))
                        })?;
                        instantiate_rule(rule, &operation)?
                    }
                    PlanTemplate::Direction(template) => {
                        instantiate_direction_template(template, &operation)
                    }
                };
                let mut children = child_plans.into_iter();
                for replacement in replacements {
                    if matches!(
                        replacement.instruction,
                        Instruction::Standard(StandardGate::GPhase)
                    ) {
                        self.lower_gate_like(replacement, target)?;
                        continue;
                    }
                    let child = children.next().ok_or_else(|| {
                        CompilerError::InvariantViolation(format!(
                            "device plan {plan:?} emitted more non-phase children than planned"
                        ))
                    })?;
                    let ordered_qargs = replacement
                        .qubits
                        .iter()
                        .copied()
                        .map(PhysicalQubit::from_qubit)
                        .collect();
                    let child_state =
                        DeviceGateState::from_instruction(&replacement.instruction, ordered_qargs)
                            .ok_or_else(|| {
                                CompilerError::InvariantViolation(format!(
                                    "missing device-lowering key for planned child {}",
                                    replacement.instruction
                                ))
                            })?;
                    self.lower_gate_like_with_plan(replacement, child_state, child, target)?;
                }
                if children.next().is_some() {
                    return Err(CompilerError::InvariantViolation(format!(
                        "device plan {plan:?} emitted fewer non-phase children than planned"
                    )));
                }
            }
        }
        Ok(())
    }

    fn lower_control_flow(
        &mut self,
        control: &ClassicalControlOp,
        classical_remap: &ClassicalRemap,
    ) -> Result<ValueClassicalControlOp, CompilerError> {
        Ok(match control {
            ClassicalControlOp::If(op) => ValueClassicalControlOp::If {
                condition: classical_remap.remap_expr(op.condition())?,
                then_body: self.lower_body(op.then_body(), classical_remap)?,
                else_body: op
                    .else_body()
                    .map(|body| self.lower_body(body, classical_remap))
                    .transpose()?,
            },
            ClassicalControlOp::While(op) => ValueClassicalControlOp::While {
                condition: classical_remap.remap_expr(op.condition())?,
                body: self.lower_body(op.body(), classical_remap)?,
            },
            ClassicalControlOp::For(op) => ValueClassicalControlOp::For {
                var: classical_remap.remap_var(op.var())?,
                start: classical_remap.remap_expr(op.start())?,
                stop: classical_remap.remap_expr(op.stop())?,
                step: classical_remap.remap_expr(op.step())?,
                body: self.lower_body(op.body(), classical_remap)?,
            },
            ClassicalControlOp::Switch(op) => ValueClassicalControlOp::Switch {
                target: classical_remap.remap_expr(op.target())?,
                cases: op
                    .cases()
                    .iter()
                    .map(|case| {
                        Ok(ValueSwitchCase::new(
                            case.value(),
                            self.lower_body(case.body(), classical_remap)?,
                        ))
                    })
                    .collect::<Result<_, CompilerError>>()?,
                default: op
                    .default()
                    .map(|body| self.lower_body(body, classical_remap))
                    .transpose()?,
            },
            ClassicalControlOp::Break => ValueClassicalControlOp::Break,
            ClassicalControlOp::Continue => ValueClassicalControlOp::Continue,
        })
    }

    fn lower_body(
        &mut self,
        body: &crate::circuit::ControlBody,
        classical_remap: &ClassicalRemap,
    ) -> Result<ValueControlBody, CompilerError> {
        let mut output = Vec::with_capacity(body.operations().len());
        let mut phase_delta = Parameter::from(0.0);
        self.lower_sequence(
            body.operations(),
            classical_remap,
            LoweringTarget::ControlFlowBody {
                output: &mut output,
                phase_delta: &mut phase_delta,
            },
        )?;
        self.prepend_body_phase(&mut output, phase_delta);
        Ok(ValueControlBody::new(output))
    }

    fn prepend_body_phase(&mut self, body: &mut Vec<ValueOperation>, phase: Parameter) {
        if phase.is_zero() {
            return;
        }
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

    fn push_operation(&mut self, operation: ValueOperation, target: &mut LoweringTarget<'_>) {
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

fn collect_root_states(circuit: &Circuit) -> Result<Vec<DeviceGateState>, CompilerError> {
    fn collect(operations: &[Operation], roots: &mut HashSet<DeviceGateState>) {
        for operation in operations {
            match &operation.instruction {
                Instruction::Standard(StandardGate::GPhase) => {}
                Instruction::Standard(_) | Instruction::McGate(_) => {
                    if let Some(instruction) =
                        KnowledgeInstructionKey::from_instruction(&operation.instruction)
                    {
                        roots.insert(DeviceGateState {
                            instruction,
                            ordered_qargs: operation
                                .qubits
                                .iter()
                                .copied()
                                .map(PhysicalQubit::from_qubit)
                                .collect(),
                        });
                    }
                }
                Instruction::ClassicalControl(control) => match control {
                    ClassicalControlOp::If(op) => {
                        collect(op.then_body().operations(), roots);
                        if let Some(body) = op.else_body() {
                            collect(body.operations(), roots);
                        }
                    }
                    ClassicalControlOp::While(op) => collect(op.body().operations(), roots),
                    ClassicalControlOp::For(op) => collect(op.body().operations(), roots),
                    ClassicalControlOp::Switch(op) => {
                        for case in op.cases() {
                            collect(case.body().operations(), roots);
                        }
                        if let Some(body) = op.default() {
                            collect(body.operations(), roots);
                        }
                    }
                    ClassicalControlOp::Break | ClassicalControlOp::Continue => {}
                },
                _ => {}
            }
        }
    }

    let mut roots = HashSet::new();
    collect(circuit.operations(), &mut roots);
    Ok(roots.into_iter().collect())
}

fn instantiate_rule(
    rule: &crate::compile::knowledge::rule::Rule,
    operation: &LowerableOperation,
) -> Result<Vec<LowerableOperation>, CompilerError> {
    let params = operation
        .params
        .iter()
        .map(parameter_value_to_parameter)
        .collect::<Vec<_>>();
    let view = ConcreteOperationView::new(&operation.instruction, &operation.qubits, &params);
    let bindings = rule_matches_operations(rule, &[view])
        .map_err(|error| CompilerError::InvariantViolation(error.to_string()))?
        .ok_or_else(|| {
            CompilerError::InvariantViolation(format!(
                "selected device-lowering rule {} does not match {}",
                rule.name, operation.instruction
            ))
        })?;
    let replacements = instantiate_target(&rule.target, &bindings)
        .map_err(|error| CompilerError::InvariantViolation(error.to_string()))?;
    Ok(replacements
        .into_iter()
        .map(|replacement| LowerableOperation {
            instruction: replacement.instruction,
            qubits: replacement.qubits,
            params: SmallVec::from_vec(replacement.params.into_vec()),
            label: None,
        })
        .collect())
}

fn instantiate_direction_template(
    template: DirectionTemplate,
    source: &LowerableOperation,
) -> Vec<LowerableOperation> {
    debug_assert_eq!(source.qubits.len(), 2);
    let q0 = source.qubits[0];
    let q1 = source.qubits[1];
    let reversed = || LowerableOperation {
        instruction: source.instruction.clone(),
        qubits: smallvec![q1, q0],
        params: source.params.clone(),
        label: source.label.clone(),
    };
    match template {
        DirectionTemplate::Cx | DirectionTemplate::Rzx => {
            let h = |qubit| LowerableOperation {
                instruction: Instruction::Standard(StandardGate::H),
                qubits: smallvec![qubit],
                params: SmallVec::new(),
                label: None,
            };
            vec![h(q0), h(q1), reversed(), h(q0), h(q1)]
        }
        DirectionTemplate::Symmetric(_) => vec![reversed()],
    }
}

fn parameter_value_to_parameter(param: &ParameterValue) -> Parameter {
    match param {
        ParameterValue::Fixed(value) => Parameter::from(*value),
        ParameterValue::Param(parameter) => parameter.clone(),
    }
}

fn gphase_param(operation: &LowerableOperation) -> Result<Parameter, CompilerError> {
    operation
        .params
        .first()
        .map(parameter_value_to_parameter)
        .ok_or_else(|| {
            CompilerError::InvariantViolation(
                "GPhase operation must contain one parameter".to_string(),
            )
        })
}

#[cfg(test)]
#[path = "device_lowering_test.rs"]
mod device_lowering_test;
