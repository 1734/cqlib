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

//! The `CommutativeCancellation` transform and its circuit traversal.

use crate::circuit::{
    Circuit, ClassicalControlOp, Instruction, Operation, Parameter, ValueClassicalControlOp,
    ValueControlBody, ValueInstruction, ValueOperation, ValueSwitchCase,
};
use crate::compile::CompilerError;
use crate::compile::commutation::{CommutationChecker, CommutationConfig};
use crate::compile::transform::analysis::CircuitAnalysis;
use crate::compile::transform::commutative_cancellation::sets::{
    OperationView, find_cancellable_ops, is_unitary_gate_like,
};
use crate::compile::transform::rebuild::{CircuitRebuildContext, ClassicalRemap};
use crate::compile::transform::transformer::{TransformOutcome, Transformer};
use smallvec::SmallVec;

/// Global self-inverse cancellation over exact commutation sets.
///
/// See the [module documentation](super) for the analysis contract.
#[derive(Debug, Clone)]
pub struct CommutativeCancellation {
    checker: CommutationChecker,
}

impl CommutativeCancellation {
    /// Builds the pass with a conservative checker: knowledge-base
    /// commutation rules enabled, matrix fallback disabled, and only exact
    /// (phase-free) commutation proofs accepted.
    pub fn new() -> Self {
        Self {
            checker: CommutationChecker::with_config(CommutationConfig {
                enable_rule_oracle: true,
                enable_matrix_fallback: false,
                max_matrix_qubits: 4,
            }),
        }
    }
}

impl Default for CommutativeCancellation {
    fn default() -> Self {
        Self::new()
    }
}

impl Transformer for CommutativeCancellation {
    fn name(&self) -> &'static str {
        "optimize.commutative_cancellation"
    }

    fn transform(
        &self,
        circuit: &Circuit,
        _analysis: Option<&CircuitAnalysis>,
    ) -> Result<TransformOutcome, CompilerError> {
        CancellationPass::run(circuit, &self.checker)
    }
}

struct CancellationPass<'source, 'checker> {
    source: &'source Circuit,
    checker: &'checker CommutationChecker,
    rebuild: CircuitRebuildContext,
}

impl<'source, 'checker> CancellationPass<'source, 'checker> {
    fn run(
        source: &'source Circuit,
        checker: &'checker CommutationChecker,
    ) -> Result<TransformOutcome, CompilerError> {
        let rebuild = CircuitRebuildContext::new(source);
        let root_classical = rebuild.root_classical().clone();
        let mut pass = Self {
            source,
            checker,
            rebuild,
        };
        let (operations, changed) = pass.process_sequence(source.operations(), &root_classical)?;
        if !changed {
            return Ok(TransformOutcome::Unchanged);
        }
        let circuit = pass
            .rebuild
            .finish(source.qubits(), operations, source.global_phase())?;
        Ok(TransformOutcome::Changed(circuit))
    }

    /// Processes one flat operation sequence, recursing into control flow.
    ///
    /// Non-unitary instructions and labeled operations are hard barriers:
    /// they split the sequence into pure unitary blocks and are preserved
    /// verbatim.
    fn process_sequence(
        &mut self,
        operations: &[Operation],
        classical_remap: &ClassicalRemap,
    ) -> Result<(Vec<ValueOperation>, bool), CompilerError> {
        let mut output = Vec::with_capacity(operations.len());
        let mut block: Vec<&Operation> = Vec::new();
        let mut changed = false;

        for operation in operations {
            if let Instruction::ClassicalControl(control) = &operation.instruction {
                changed |= self.flush_block(&mut block, &mut output, classical_remap)?;
                let (instruction, body_changed) =
                    self.rebuild_control_flow(control, classical_remap)?;
                changed |= body_changed;
                output.push(ValueOperation {
                    qubits: instruction.used_qubits().into_iter().collect(),
                    instruction: ValueInstruction::ClassicalControl(instruction),
                    params: CircuitRebuildContext::resolve_source_params(
                        self.source,
                        &operation.params,
                    )?,
                    label: operation.label.clone(),
                });
            } else if is_unitary_gate_like(&operation.instruction) && operation.label.is_none() {
                block.push(operation);
            } else {
                changed |= self.flush_block(&mut block, &mut output, classical_remap)?;
                output.push(self.rebuild.remap_preserved_operation(
                    self.source,
                    operation,
                    classical_remap,
                )?);
            }
        }
        changed |= self.flush_block(&mut block, &mut output, classical_remap)?;
        Ok((output, changed))
    }

    /// Runs cancellation over one accumulated unitary block and appends the
    /// surviving operations to `output`.
    ///
    /// Parameters are resolved once per operation up front; a resolution
    /// failure indicates corrupted circuit IR (an invalid parameter index or
    /// non-finite value) and is propagated instead of degrading the analysis.
    fn flush_block(
        &mut self,
        block: &mut Vec<&Operation>,
        output: &mut Vec<ValueOperation>,
        classical_remap: &ClassicalRemap,
    ) -> Result<bool, CompilerError> {
        if block.is_empty() {
            return Ok(false);
        }

        let mut views = Vec::with_capacity(block.len());
        for (order, op) in block.iter().enumerate() {
            let params: SmallVec<[Parameter; 3]> = op
                .params
                .iter()
                .map(|param| {
                    self.source
                        .resolve_parameter(param)
                        .map_err(CompilerError::Circuit)
                })
                .collect::<Result<_, _>>()?;
            views.push(OperationView { order, op, params });
        }

        let deleted = find_cancellable_ops(self.checker, &views);
        let mut changed = false;
        for view in &views {
            if deleted[view.order] {
                changed = true;
                continue;
            }
            output.push(self.rebuild.remap_preserved_operation(
                self.source,
                view.op,
                classical_remap,
            )?);
        }
        block.clear();
        Ok(changed)
    }

    fn rebuild_body(
        &mut self,
        operations: &[Operation],
        classical_remap: &ClassicalRemap,
    ) -> Result<(ValueControlBody, bool), CompilerError> {
        let (operations, changed) = self.process_sequence(operations, classical_remap)?;
        Ok((ValueControlBody::new(operations), changed))
    }

    /// Recursively rebuilds control-flow bodies. Control flow itself is a
    /// barrier across scopes: bodies are analyzed independently and no phase
    /// is produced, matching the exact-identity guarantee of the pass.
    fn rebuild_control_flow(
        &mut self,
        control: &ClassicalControlOp,
        classical_remap: &ClassicalRemap,
    ) -> Result<(ValueClassicalControlOp, bool), CompilerError> {
        Ok(match control {
            ClassicalControlOp::If(op) => {
                let (then_body, then_changed) =
                    self.rebuild_body(op.then_body().operations(), classical_remap)?;
                let else_rewrite = op
                    .else_body()
                    .map(|body| self.rebuild_body(body.operations(), classical_remap))
                    .transpose()?;
                let else_changed = else_rewrite.as_ref().is_some_and(|(_, changed)| *changed);
                (
                    ValueClassicalControlOp::If {
                        condition: classical_remap.remap_expr(op.condition())?,
                        then_body,
                        else_body: else_rewrite.map(|(body, _)| body),
                    },
                    then_changed || else_changed,
                )
            }
            ClassicalControlOp::While(op) => {
                let (body, changed) = self.rebuild_body(op.body().operations(), classical_remap)?;
                (
                    ValueClassicalControlOp::While {
                        condition: classical_remap.remap_expr(op.condition())?,
                        body,
                    },
                    changed,
                )
            }
            ClassicalControlOp::For(op) => {
                let (body, changed) = self.rebuild_body(op.body().operations(), classical_remap)?;
                (
                    ValueClassicalControlOp::For {
                        var: classical_remap.remap_var(op.var())?,
                        start: classical_remap.remap_expr(op.start())?,
                        stop: classical_remap.remap_expr(op.stop())?,
                        step: classical_remap.remap_expr(op.step())?,
                        body,
                    },
                    changed,
                )
            }
            ClassicalControlOp::Switch(op) => {
                let mut changed = false;
                let cases = op
                    .cases()
                    .iter()
                    .map(|case| {
                        let (body, body_changed) =
                            self.rebuild_body(case.body().operations(), classical_remap)?;
                        changed |= body_changed;
                        Ok(ValueSwitchCase::new(case.value(), body))
                    })
                    .collect::<Result<Vec<_>, CompilerError>>()?;
                let default_rewrite = op
                    .default()
                    .map(|body| self.rebuild_body(body.operations(), classical_remap))
                    .transpose()?;
                changed |= default_rewrite
                    .as_ref()
                    .is_some_and(|(_, body_changed)| *body_changed);
                (
                    ValueClassicalControlOp::Switch {
                        target: classical_remap.remap_expr(op.target())?,
                        cases,
                        default: default_rewrite.map(|(body, _)| body),
                    },
                    changed,
                )
            }
            ClassicalControlOp::Break => (ValueClassicalControlOp::Break, false),
            ClassicalControlOp::Continue => (ValueClassicalControlOp::Continue, false),
        })
    }
}

#[cfg(test)]
#[path = "commutative_cancellation_test.rs"]
mod commutative_cancellation_test;
