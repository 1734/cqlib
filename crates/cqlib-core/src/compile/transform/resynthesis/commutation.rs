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

//! Local commutation adapter for numeric block resynthesis.
//!
//! The compiler-wide [`CommutationChecker`] remains the semantic proof engine.
//! This adapter adds the operation-level shape and per-pass cache needed by
//! resynthesis collectors, without changing the public commutation API.

#![allow(dead_code)]

use crate::circuit::{Instruction, Operation, Parameter, ValueInstruction, ValueOperation};
use crate::compile::commutation::{CommutationChecker, CommutationConfig};
use smallvec::SmallVec;
use std::collections::HashMap;

/// Source operation plus parameters resolved against its owning circuit.
pub(crate) struct OperationView<'a> {
    /// Stable source-operation order within the pass-local operation stream.
    pub(crate) order: usize,
    /// Source operation.
    pub(crate) operation: &'a Operation,
    /// Resolved parameters.
    ///
    /// Every [`crate::circuit::CircuitParam::Index`] has already been expanded
    /// from the source circuit parameter table. This slice is ready to pass
    /// directly to [`CommutationChecker::check`].
    pub(crate) params: SmallVec<[Parameter; 3]>,
}

/// Pass-local cached exact commutation queries for source operations.
pub(crate) struct CachedCommutation {
    checker: CommutationChecker,
    cache: HashMap<(usize, usize), bool>,
    exact_only: bool,
}

impl CachedCommutation {
    /// Builds a cached adapter around a configured commutation checker.
    pub(crate) fn new(config: CommutationConfig) -> Self {
        Self {
            checker: CommutationChecker::with_config(config),
            cache: HashMap::new(),
            exact_only: true,
        }
    }

    /// Returns whether two source operations commute under the adapter policy.
    ///
    /// Results are cached by normalized source order. Replacement operations are
    /// intentionally excluded from this cache because they do not have stable
    /// source-order identities.
    pub(crate) fn commute_ops(&mut self, lhs: &OperationView<'_>, rhs: &OperationView<'_>) -> bool {
        if lhs.order == rhs.order {
            return true;
        }

        let key = normalized_pair(lhs.order, rhs.order);
        if let Some(result) = self.cache.get(&key) {
            return *result;
        }

        let result = self
            .checker
            .check(
                &lhs.operation.instruction,
                &lhs.operation.qubits,
                &lhs.params,
                &rhs.operation.instruction,
                &rhs.operation.qubits,
                &rhs.params,
            )
            .is_some_and(|commutation| !self.exact_only || commutation.is_exact());

        self.cache.insert(key, result);
        result
    }

    /// Verifies that every replacement can remain at the patch site while
    /// crossed source operations stay around it.
    ///
    /// This is a post-synthesis safety check: proving a matched source
    /// operation can cross a skipped operation is not enough once the matched
    /// operations have been replaced by a newly synthesized sequence.
    pub(crate) fn replacements_commute_with_crossed(
        &self,
        crossed: &[&OperationView<'_>],
        replacements: &[ValueOperation],
    ) -> bool {
        for crossed_op in crossed {
            for replacement in replacements {
                if !shares_any_qubit(&crossed_op.operation.qubits, &replacement.qubits) {
                    continue;
                }
                if !self.replacement_commutes_with_op(crossed_op, replacement) {
                    return false;
                }
            }
        }
        true
    }

    fn replacement_commutes_with_op(
        &self,
        operation: &OperationView<'_>,
        replacement: &ValueOperation,
    ) -> bool {
        let Some(replacement_instruction) = replacement_instruction(replacement) else {
            return false;
        };
        let replacement_params = replacement
            .params
            .iter()
            .map(Parameter::from)
            .collect::<SmallVec<[Parameter; 3]>>();

        self.checker
            .check(
                &operation.operation.instruction,
                &operation.operation.qubits,
                &operation.params,
                replacement_instruction,
                &replacement.qubits,
                &replacement_params,
            )
            .is_some_and(|commutation| !self.exact_only || commutation.is_exact())
    }

    #[cfg(test)]
    fn cache_len(&self) -> usize {
        self.cache.len()
    }
}

fn normalized_pair(lhs: usize, rhs: usize) -> (usize, usize) {
    if lhs <= rhs { (lhs, rhs) } else { (rhs, lhs) }
}

fn replacement_instruction(replacement: &ValueOperation) -> Option<&Instruction> {
    match &replacement.instruction {
        ValueInstruction::Instruction(instruction) => Some(instruction),
        ValueInstruction::ClassicalControl(_) => None,
    }
}

fn shares_any_qubit(lhs: &[crate::circuit::Qubit], rhs: &[crate::circuit::Qubit]) -> bool {
    lhs.iter().any(|qubit| rhs.contains(qubit))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::circuit::{
        CircuitParam, Qubit, StandardGate, ValueClassicalControlOp, ValueInstruction,
        circuit_param::ParameterValue,
    };
    use crate::compile::commutation::CommutationConfig;
    use smallvec::smallvec;

    fn checker() -> CachedCommutation {
        CachedCommutation::new(CommutationConfig {
            enable_rule_oracle: true,
            enable_matrix_fallback: false,
            max_matrix_qubits: 4,
        })
    }

    fn op(gate: StandardGate, qubits: &[Qubit], params: &[f64]) -> Operation {
        Operation {
            instruction: Instruction::Standard(gate),
            qubits: qubits.iter().copied().collect(),
            params: params.iter().copied().map(CircuitParam::Fixed).collect(),
            label: None,
        }
    }

    fn view<'a>(
        order: usize,
        operation: &'a Operation,
        params: SmallVec<[Parameter; 3]>,
    ) -> OperationView<'a> {
        OperationView {
            order,
            operation,
            params,
        }
    }

    fn fixed_params(values: &[f64]) -> SmallVec<[Parameter; 3]> {
        values.iter().copied().map(Parameter::from).collect()
    }

    #[test]
    fn disjoint_source_operations_commute() {
        let h = op(StandardGate::H, &[Qubit::new(0)], &[]);
        let x = op(StandardGate::X, &[Qubit::new(1)], &[]);
        let h_view = view(0, &h, smallvec![]);
        let x_view = view(1, &x, smallvec![]);

        assert!(checker().commute_ops(&h_view, &x_view));
    }

    #[test]
    fn same_operation_commutes_without_cache_entry() {
        let h = op(StandardGate::H, &[Qubit::new(0)], &[]);
        let h_view = view(7, &h, smallvec![]);
        let mut checker = checker();

        assert!(checker.commute_ops(&h_view, &h_view));
        assert_eq!(checker.cache_len(), 0);
    }

    #[test]
    fn symbolic_same_axis_rotations_commute() {
        let first = op(StandardGate::RZ, &[Qubit::new(0)], &[]);
        let second = op(StandardGate::RZ, &[Qubit::new(0)], &[]);
        let first_view = view(0, &first, smallvec![Parameter::symbol("a")]);
        let second_view = view(1, &second, smallvec![Parameter::symbol("b")]);

        assert!(checker().commute_ops(&first_view, &second_view));
    }

    #[test]
    fn same_qubit_h_and_x_do_not_commute() {
        let h = op(StandardGate::H, &[Qubit::new(0)], &[]);
        let x = op(StandardGate::X, &[Qubit::new(0)], &[]);
        let h_view = view(0, &h, smallvec![]);
        let x_view = view(1, &x, smallvec![]);

        assert!(!checker().commute_ops(&h_view, &x_view));
    }

    #[test]
    fn reversed_source_query_reuses_normalized_cache_key() {
        let h = op(StandardGate::H, &[Qubit::new(0)], &[]);
        let x = op(StandardGate::X, &[Qubit::new(1)], &[]);
        let h_view = view(3, &h, smallvec![]);
        let x_view = view(9, &x, smallvec![]);
        let mut checker = checker();

        assert!(checker.commute_ops(&h_view, &x_view));
        assert_eq!(checker.cache_len(), 1);
        assert!(checker.commute_ops(&x_view, &h_view));
        assert_eq!(checker.cache_len(), 1);
    }

    #[test]
    fn empty_crossed_or_replacements_are_safe() {
        let op = op(StandardGate::H, &[Qubit::new(0)], &[]);
        let op_view = view(0, &op, smallvec![]);
        let replacement = ValueOperation::from_standard(StandardGate::X, [Qubit::new(0)], []);
        let checker = checker();

        assert!(checker.replacements_commute_with_crossed(&[], &[replacement.clone()]));
        assert!(checker.replacements_commute_with_crossed(&[&op_view], &[]));
    }

    #[test]
    fn disjoint_replacement_commutes_with_crossed_operation() {
        let crossed = op(StandardGate::H, &[Qubit::new(0)], &[]);
        let crossed_view = view(0, &crossed, smallvec![]);
        let replacement = ValueOperation::from_standard(StandardGate::X, [Qubit::new(1)], []);

        assert!(checker().replacements_commute_with_crossed(&[&crossed_view], &[replacement]));
    }

    #[test]
    fn shared_non_commuting_replacement_is_rejected() {
        let crossed = op(StandardGate::H, &[Qubit::new(0)], &[]);
        let crossed_view = view(0, &crossed, smallvec![]);
        let replacement = ValueOperation::from_standard(StandardGate::X, [Qubit::new(0)], []);

        assert!(!checker().replacements_commute_with_crossed(&[&crossed_view], &[replacement]));
    }

    #[test]
    fn classical_control_replacement_is_rejected() {
        let crossed = op(StandardGate::H, &[Qubit::new(0)], &[]);
        let crossed_view = view(0, &crossed, smallvec![]);
        let replacement = ValueOperation {
            instruction: ValueInstruction::ClassicalControl(ValueClassicalControlOp::Break),
            qubits: smallvec![Qubit::new(0)],
            params: smallvec![],
            label: None,
        };

        assert!(!checker().replacements_commute_with_crossed(&[&crossed_view], &[replacement]));
    }

    #[test]
    fn symbolic_parameters_do_not_panic() {
        let crossed = op(StandardGate::RZ, &[Qubit::new(0)], &[]);
        let crossed_view = view(0, &crossed, smallvec![Parameter::symbol("theta")]);
        let replacement = ValueOperation::from_standard(
            StandardGate::RZ,
            [Qubit::new(0)],
            [ParameterValue::Param(Parameter::symbol("phi"))],
        );

        assert!(checker().replacements_commute_with_crossed(&[&crossed_view], &[replacement]));
    }

    #[test]
    fn fixed_replacement_params_bridge_to_checker() {
        let crossed = op(StandardGate::RZ, &[Qubit::new(0)], &[]);
        let crossed_view = view(0, &crossed, fixed_params(&[0.25]));
        let replacement = ValueOperation::from_standard(
            StandardGate::RZ,
            [Qubit::new(0)],
            [ParameterValue::Fixed(0.5)],
        );

        assert!(checker().replacements_commute_with_crossed(&[&crossed_view], &[replacement]));
    }
}
