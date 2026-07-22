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

//! Rule compilation and dependency-aware sequence matching.
//!
//! The matcher prepares the knowledge-rule library for repeated local scans.
//! Rules are indexed by their first instruction key, filtered by rewrite mode,
//! qubit width, target-basis constraints, and static cost before any expensive
//! matching is attempted.
//!
//! Matching is dependency-aware rather than strictly contiguous. A rule item may
//! be matched after commuting past unrelated operations when the compiler
//! commutation checker proves that reordering is safe for the candidate block.
//! This lets local identities apply across independent gates without changing
//! the relative order of non-commuting operations.
//!
//! Every successful match becomes a candidate patch with a before/after local
//! cost. Candidate patches are sorted by final cost, original cost reduction,
//! covered span, static size delta, source position, and rule id. The selector
//! then greedily keeps the first non-overlapping patches, producing a stable
//! patch set that never rewrites the same operation span twice in one round.

use crate::circuit::{
    Circuit, CircuitParam, Instruction, Operation, Parameter, ParameterValue, Qubit, StandardGate,
};
use crate::compile::commutation::{CommutationChecker, CommutationConfig};
use crate::compile::error::CompilerError;
use crate::compile::knowledge::library::{RuleKind, RuleLibrary};
use crate::compile::knowledge::matcher::KnowledgeInstructionKey as RewriteInstructionKey;
use crate::compile::knowledge::matcher::{
    ConcreteOperationView, MatchBindings, conditions_hold as knowledge_conditions_hold,
    instantiate_target as knowledge_instantiate_target,
    match_rule_item as knowledge_match_rule_item,
};
use crate::compile::knowledge::rule::{Rule, RuleItem};
use crate::compile::transform::rewrite::basis::TargetContext;
use crate::compile::transform::rewrite::config::{GPhaseCost, LocalRewriteCost, RewriteConfig};
use smallvec::SmallVec;
use std::collections::{HashMap, HashSet};
use std::ops::Range;

/// A rewrite rule prepared for repeated matching.
struct CompiledRule {
    id: usize,
    kind: RuleKind,
    match_len: usize,
    qubit_count: usize,
    static_cost_delta: isize,
    source_keys: SmallVec<[RewriteInstructionKey; 8]>,
    match_keys: SmallVec<[RewriteInstructionKey; 4]>,
    rewrite_keys: SmallVec<[RewriteInstructionKey; 4]>,
    rule: Rule,
}

/// Compiled rule collection with a first-instruction candidate index.
pub(super) struct CompiledRuleSet {
    rules: Vec<CompiledRule>,
    first_key_map: HashMap<RewriteInstructionKey, SmallVec<[usize; 8]>>,
    commutation: CommutationChecker,
}

/// One operation emitted by a rewrite target.
#[derive(Debug, Clone)]
pub(super) struct ReplacementItem {
    pub(super) instruction: Instruction,
    pub(super) qubits: SmallVec<[Qubit; 3]>,
    pub(super) params: SmallVec<[ParameterValue; 3]>,
    key: RewriteInstructionKey,
}

/// A selected replacement for matched operation positions in one block.
#[derive(Debug, Clone)]
pub(super) struct RewritePatch {
    pub(super) rule_id: usize,
    static_cost_delta: isize,
    pub(super) first_position: usize,
    pub(super) last_position: usize,
    pub(super) matched_positions: Vec<usize>,
    pub(super) replacements: Vec<ReplacementItem>,
}

#[derive(Clone)]
struct CandidatePatch {
    patch: RewritePatch,
    before: LocalRewriteCost,
    after: LocalRewriteCost,
}

struct BlockContext<'a> {
    operations: &'a [Operation],
    instruction_keys: Vec<RewriteInstructionKey>,
    resolved_params: Vec<SmallVec<[Parameter; 3]>>,
    instruction_set: HashSet<RewriteInstructionKey>,
    qubit_count: usize,
}

impl<'a> BlockContext<'a> {
    /// Builds cached matching context for one linear operation block.
    ///
    /// Per-position instruction keys and resolved symbolic parameters are
    /// cached once because rule matching probes the same operation many times
    /// across anchors, rules, and commutation checks.
    fn new(circuit: &'a Circuit, operations: &'a [Operation]) -> Result<Self, CompilerError> {
        let mut resolved_params = Vec::with_capacity(operations.len());
        let mut instruction_keys = Vec::with_capacity(operations.len());
        let mut touched_qubits = HashSet::new();
        let mut instruction_set = HashSet::new();

        for operation in operations {
            let key = RewriteInstructionKey::from_instruction(&operation.instruction).ok_or_else(
                || {
                    CompilerError::InvariantViolation(format!(
                        "rewrite block contains unsupported instruction {:?}",
                        operation.instruction
                    ))
                },
            )?;
            let params = operation
                .params
                .iter()
                .map(|param| resolve_operation_param(circuit, param))
                .collect::<Result<SmallVec<[_; 3]>, _>>()?;

            for &qubit in &operation.qubits {
                touched_qubits.insert(qubit);
            }
            instruction_set.insert(key.clone());
            instruction_keys.push(key);
            resolved_params.push(params);
        }

        Ok(Self {
            operations,
            instruction_keys,
            resolved_params,
            instruction_set,
            qubit_count: touched_qubits.len(),
        })
    }

    fn len(&self) -> usize {
        self.operations.len()
    }

    fn operation(&self, position: usize) -> &Operation {
        &self.operations[position]
    }

    fn key(&self, position: usize) -> &RewriteInstructionKey {
        &self.instruction_keys[position]
    }

    fn params(&self, position: usize) -> &[Parameter] {
        &self.resolved_params[position]
    }
}

impl CompiledRuleSet {
    pub(super) fn from_library(library: &RuleLibrary) -> Result<Self, CompilerError> {
        let mut rules = Vec::with_capacity(library.len());
        let mut first_key_map: HashMap<RewriteInstructionKey, SmallVec<[usize; 8]>> =
            HashMap::new();
        let kind_by_id = build_kind_index(library);
        let commutation = CommutationChecker::from_library(library, rewrite_commutation_config());

        for (index, rule) in library.rules().iter().cloned().enumerate() {
            let kind = kind_by_id.get(&index).copied().unwrap_or(RuleKind::Other);
            push_compiled_rule(&mut rules, &mut first_key_map, index, kind, rule)?;
        }

        Ok(Self {
            rules,
            first_key_map,
            commutation,
        })
    }

    fn candidates_for_first_instruction(&self, key: &RewriteInstructionKey) -> &[usize] {
        self.first_key_map
            .get(key)
            .map(SmallVec::as_slice)
            .unwrap_or(&[])
    }

    fn get(&self, index: usize) -> &CompiledRule {
        &self.rules[index]
    }

    pub(super) fn lowerable_rules(
        &self,
    ) -> impl Iterator<
        Item = (
            RuleKind,
            &[RewriteInstructionKey],
            &[RewriteInstructionKey],
            bool,
        ),
    > {
        self.rules.iter().map(|rule| {
            let has_conditions = rule
                .rule
                .conditions
                .as_ref()
                .is_some_and(|conditions| !conditions.is_empty());
            (
                rule.kind,
                rule.source_keys.as_slice(),
                rule.rewrite_keys.as_slice(),
                has_conditions,
            )
        })
    }
}

fn rewrite_commutation_config() -> CommutationConfig {
    CommutationConfig {
        enable_rule_oracle: true,
        enable_matrix_fallback: false,
        max_matrix_qubits: 0,
    }
}

/// Adds one rule to the compiled rule set and first-key index.
///
/// The compiled representation keeps both full source order and unique
/// match/rewrite key sets. Source order drives the dependency-aware scan;
/// unique sets are used for cheap static filters before binding work starts.
fn push_compiled_rule(
    rules: &mut Vec<CompiledRule>,
    first_key_map: &mut HashMap<RewriteInstructionKey, SmallVec<[usize; 8]>>,
    id: usize,
    kind: RuleKind,
    rule: Rule,
) -> Result<(), CompilerError> {
    if rule.operations.is_empty() {
        return Err(CompilerError::InvariantViolation(
            "rewrite rule contains an empty match block".to_string(),
        ));
    }

    let match_len = rule.operations.len();
    let rewrite_len = rule.target.len();
    let mut rule_qubits = HashSet::new();
    for item in rule.operations.iter().chain(&rule.target) {
        rule_qubits.extend(item.qubits.iter().copied());
    }

    let mut source_keys = SmallVec::<[RewriteInstructionKey; 8]>::new();
    let mut match_keys = SmallVec::<[RewriteInstructionKey; 4]>::new();
    let mut rewrite_keys = SmallVec::<[RewriteInstructionKey; 4]>::new();

    for item in &rule.operations {
        let key = RewriteInstructionKey::from_instruction(&item.instruction).ok_or_else(|| {
            CompilerError::InvariantViolation(format!(
                "rewrite rule contains unsupported instruction {:?}",
                item.instruction
            ))
        })?;
        if !match_keys.contains(&key) {
            match_keys.push(key.clone());
        }
        source_keys.push(key);
    }
    for item in &rule.target {
        let key = RewriteInstructionKey::from_instruction(&item.instruction).ok_or_else(|| {
            CompilerError::InvariantViolation(format!(
                "rewrite rule contains unsupported instruction {:?}",
                item.instruction
            ))
        })?;
        if !rewrite_keys.contains(&key) {
            rewrite_keys.push(key.clone());
        }
    }

    first_key_map
        .entry(source_keys[0].clone())
        .or_default()
        .push(rules.len());
    rules.push(CompiledRule {
        id,
        kind,
        match_len,
        qubit_count: rule_qubits.len(),
        static_cost_delta: rewrite_len as isize - match_len as isize,
        source_keys,
        match_keys,
        rewrite_keys,
        rule,
    });
    Ok(())
}

pub(super) fn select_rewrites_in_context(
    circuit: &Circuit,
    operations: &[Operation],
    rules: &CompiledRuleSet,
    config: &RewriteConfig,
    target_context: Option<&TargetContext>,
) -> Result<Vec<RewritePatch>, CompilerError> {
    let block = BlockContext::new(circuit, operations)?;
    let mut candidates = Vec::new();

    for anchor in 0..block.len() {
        let operation = block.operation(anchor);
        if config.skips_labeled_ops() && operation.label.is_some() {
            continue;
        }
        let first_key = block.key(anchor);

        for &rule_index in rules.candidates_for_first_instruction(first_key) {
            let compiled = rules.get(rule_index);
            if !rule_passes_static_filters(compiled, config, &block, target_context) {
                continue;
            }
            if let Some(candidate) = try_match_rule(
                &block,
                anchor,
                compiled,
                &rules.commutation,
                config,
                target_context,
            )? {
                candidates.push(candidate);
            }
        }
    }

    // Rank candidate patches by the local objective before taking any patch.
    // With an explicit target, prefer the patch that legalizes more unsupported
    // operations so a cheap single-gate rewrite cannot occupy part of a better
    // multi-operation lowering. The remaining keys make the choice deterministic.
    candidates.sort_by(|lhs, rhs| {
        let lhs_unsupported_reduction = lhs
            .before
            .unsupported_ops
            .saturating_sub(lhs.after.unsupported_ops);
        let rhs_unsupported_reduction = rhs
            .before
            .unsupported_ops
            .saturating_sub(rhs.after.unsupported_ops);
        let target_reduction_order = if target_context.is_some() {
            rhs_unsupported_reduction.cmp(&lhs_unsupported_reduction)
        } else {
            std::cmp::Ordering::Equal
        };

        target_reduction_order
            .then_with(|| lhs.after.cmp(&rhs.after))
            .then_with(|| lhs.before.cmp(&rhs.before).reverse())
            .then_with(|| {
                rhs.patch
                    .matched_positions
                    .len()
                    .cmp(&lhs.patch.matched_positions.len())
            })
            .then_with(|| {
                lhs.patch
                    .static_cost_delta
                    .cmp(&rhs.patch.static_cost_delta)
            })
            .then_with(|| lhs.patch.first_position.cmp(&rhs.patch.first_position))
            .then_with(|| lhs.patch.rule_id.cmp(&rhs.patch.rule_id))
    });

    let mut occupied_spans = vec![false; block.len()];
    let mut patches = Vec::new();
    for candidate in candidates {
        let first_position = candidate.patch.first_position;
        let last_position = candidate.patch.last_position;
        if first_position > last_position || last_position >= occupied_spans.len() {
            return Err(CompilerError::InvariantViolation(format!(
                "rewrite candidate span {first_position}..={last_position} is outside block of length {}",
                block.len()
            )));
        }
        if occupied_spans[first_position..=last_position]
            .iter()
            .any(|occupied| *occupied)
        {
            continue;
        }

        occupied_spans[first_position..=last_position].fill(true);
        patches.push(candidate.patch);
    }

    patches.sort_by_key(|patch| patch.first_position);
    Ok(patches)
}

fn build_kind_index(library: &RuleLibrary) -> HashMap<usize, RuleKind> {
    let mut index = HashMap::new();
    for kind in [
        RuleKind::Simplify,
        RuleKind::Cancel,
        RuleKind::Merge,
        RuleKind::Commute,
        RuleKind::Decompose,
        RuleKind::Canonicalize,
        RuleKind::HardwareNative,
        RuleKind::Other,
    ] {
        for id in library.rules_by_kind(kind) {
            index.insert(id.as_usize(), kind);
        }
    }
    index
}

/// Applies conservative static filters before expensive rule matching.
///
/// These checks reject impossible or disabled rules before the matcher clones
/// bindings or asks the commutation oracle.
fn rule_passes_static_filters(
    rule: &CompiledRule,
    config: &RewriteConfig,
    block: &BlockContext<'_>,
    target_context: Option<&TargetContext>,
) -> bool {
    rule.kind != RuleKind::Commute
        && config.allows_kind(rule.kind)
        && rule.match_len <= config.max_pattern_len()
        && rule.qubit_count <= block.qubit_count
        && rule_passes_target_filter(rule, target_context, block)
}

/// Checks whether a rule is legal for an optional target-basis context.
///
/// Target lowering rules must only replace operations present in this block,
/// and their replacement keys must be legal for the target context.
fn rule_passes_target_filter(
    rule: &CompiledRule,
    target_context: Option<&TargetContext>,
    block: &BlockContext<'_>,
) -> bool {
    let Some(target_context) = target_context else {
        return true;
    };

    if rule_rewrites_physical_source_through_non_physical_target(rule, target_context) {
        return false;
    }

    rule.match_keys
        .iter()
        .all(|key| block.instruction_set.contains(key))
        && rule
            .rewrite_keys
            .iter()
            .all(|key| target_context.allows_rewrite_key(key))
}

fn rule_rewrites_physical_source_through_non_physical_target(
    rule: &CompiledRule,
    target_context: &TargetContext,
) -> bool {
    !rule.source_keys.is_empty()
        && rule
            .source_keys
            .iter()
            .all(|key| is_implicit_target_key(key) || target_context.physically_supports(key))
        && rule
            .rewrite_keys
            .iter()
            .any(|key| !is_implicit_target_key(key) && !target_context.physically_supports(key))
}

fn is_implicit_target_key(key: &RewriteInstructionKey) -> bool {
    matches!(key, RewriteInstructionKey::Standard(StandardGate::GPhase))
}

/// Tries to match one compiled rule at one anchor position.
///
/// The anchor must match the first rule item exactly. Later items may be found
/// after commuting past unrelated operations, but the first item owns the
/// candidate's source position and first-key index lookup.
fn try_match_rule(
    block: &BlockContext<'_>,
    anchor: usize,
    compiled: &CompiledRule,
    commutation: &CommutationChecker,
    config: &RewriteConfig,
    target_context: Option<&TargetContext>,
) -> Result<Option<CandidatePatch>, CompilerError> {
    let rule = &compiled.rule;
    let mut bindings = MatchBindings::new();

    if !match_item(
        block,
        anchor,
        &rule.operations[0],
        &compiled.source_keys[0],
        &mut bindings,
        config,
    )? {
        return Ok(None);
    }

    let mut matched_positions = vec![anchor];
    let mut skipped_positions = Vec::new();
    let mut cursor = anchor + 1;
    for (item, item_key) in rule.operations.iter().zip(&compiled.source_keys).skip(1) {
        let mut found = None;
        let limit = block.len().min(cursor + config.max_window_ops());

        for position in cursor..limit {
            if block.key(position) != item_key {
                continue;
            }
            // Matching is non-contiguous only when every skipped operation can
            // commute with the already matched prefix and the candidate item.
            // This preserves the observable order of non-commuting operations.
            if !can_skip_between(
                block,
                cursor..position,
                &matched_positions,
                position,
                commutation,
                config,
            )? {
                continue;
            }

            let mut next_bindings = bindings.clone();
            if match_item(block, position, item, item_key, &mut next_bindings, config)? {
                found = Some((position, next_bindings));
                break;
            }
        }

        let Some((position, next_bindings)) = found else {
            return Ok(None);
        };
        bindings = next_bindings;
        skipped_positions.extend(cursor..position);
        matched_positions.push(position);
        cursor = position + 1;
    }

    if !knowledge_conditions_hold(rule.conditions.as_deref(), &bindings) {
        return Ok(None);
    }
    // A skipped operation may sit before a later matched operation. Verify the
    // complete match after all positions are known, including future matches
    // that were not available to `can_skip_between` earlier in the scan.
    if !skipped_sources_commute_with_future_matches(
        block,
        &skipped_positions,
        &matched_positions,
        commutation,
        config,
    ) {
        return Ok(None);
    }

    let replacements = knowledge_instantiate_target(&rule.target, &bindings)
        .map_err(|error| CompilerError::InvariantViolation(error.to_string()))?
        .into_iter()
        .map(|item| ReplacementItem {
            instruction: item.instruction,
            qubits: item.qubits,
            params: item.params,
            key: item.key,
        })
        .collect::<Vec<_>>();

    if !replacements_commute_with_skipped(block, &skipped_positions, &replacements, commutation)? {
        return Ok(None);
    }

    // Cost is computed on matched source positions only. Skipped operations
    // remain in place around the replacement and therefore are not part of the
    // before/after objective.
    let before = cost_for_operation_positions(block, &matched_positions, target_context);
    let after = cost_for_replacements(&replacements, target_context);
    if !config.allows_rewrite(compiled.kind, before, after) {
        return Ok(None);
    }

    let first_position = matched_positions[0];
    let last_position = matched_positions.last().copied().unwrap_or(first_position);
    Ok(Some(CandidatePatch {
        before,
        after,
        patch: RewritePatch {
            rule_id: compiled.id,
            static_cost_delta: compiled.static_cost_delta,
            first_position,
            last_position,
            matched_positions,
            replacements,
        },
    }))
}

/// Checks whether operations between a matched prefix and candidate can be skipped.
///
/// Only skipped operations touching the match's active qubits can constrain the
/// rewrite. Operations on disjoint qubits are independent in this local block
/// model and do not need oracle calls.
fn can_skip_between(
    block: &BlockContext<'_>,
    skipped: Range<usize>,
    matched_positions: &[usize],
    candidate_position: usize,
    commutation: &CommutationChecker,
    config: &RewriteConfig,
) -> Result<bool, CompilerError> {
    if skipped.is_empty() {
        return Ok(true);
    }

    let mut relevant = SmallVec::<[Qubit; 4]>::new();
    for &position in matched_positions {
        for &qubit in &block.operation(position).qubits {
            if !relevant.contains(&qubit) {
                relevant.push(qubit);
            }
        }
    }
    for &qubit in &block.operation(candidate_position).qubits {
        if !relevant.contains(&qubit) {
            relevant.push(qubit);
        }
    }

    for skipped_position in skipped {
        let skipped_operation = block.operation(skipped_position);
        if config.skips_labeled_ops() && skipped_operation.label.is_some() {
            return Ok(false);
        }
        if !skipped_operation
            .qubits
            .iter()
            .any(|qubit| relevant.contains(qubit))
        {
            continue;
        }

        for &matched_position in matched_positions {
            if !operations_commute(block, skipped_position, matched_position, commutation) {
                return Ok(false);
            }
        }
        if !operations_commute(block, skipped_position, candidate_position, commutation) {
            return Ok(false);
        }
    }

    Ok(true)
}

/// Verifies skipped operations commute with all future matched positions.
///
/// This second pass catches skipped operations that were seen before all later
/// match positions were known.
fn skipped_sources_commute_with_future_matches(
    block: &BlockContext<'_>,
    skipped_positions: &[usize],
    matched_positions: &[usize],
    commutation: &CommutationChecker,
    config: &RewriteConfig,
) -> bool {
    for &skipped_position in skipped_positions {
        let skipped_operation = block.operation(skipped_position);
        if config.skips_labeled_ops() && skipped_operation.label.is_some() {
            return false;
        }
        for &matched_position in matched_positions {
            if matched_position <= skipped_position
                || !skipped_operation
                    .qubits
                    .iter()
                    .any(|qubit| block.operation(matched_position).qubits.contains(qubit))
            {
                continue;
            }
            if !operations_commute(block, skipped_position, matched_position, commutation) {
                return false;
            }
        }
    }

    true
}

/// Verifies replacements can be emitted without crossing skipped operations.
///
/// The replacement will be emitted at the match site while skipped operations
/// remain around it. Every replacement touching a skipped operation's qubits
/// must commute with that skipped operation.
fn replacements_commute_with_skipped(
    block: &BlockContext<'_>,
    skipped_positions: &[usize],
    replacements: &[ReplacementItem],
    commutation: &CommutationChecker,
) -> Result<bool, CompilerError> {
    if skipped_positions.is_empty() || replacements.is_empty() {
        return Ok(true);
    }

    for &skipped_position in skipped_positions {
        for replacement in replacements {
            let skipped_qubits = &block.operation(skipped_position).qubits;
            if !skipped_qubits
                .iter()
                .any(|qubit| replacement.qubits.contains(qubit))
            {
                continue;
            }
            if !operation_commutes_with_replacement(
                block,
                skipped_position,
                replacement,
                commutation,
            ) {
                return Ok(false);
            }
        }
    }

    Ok(true)
}

fn operations_commute(
    block: &BlockContext<'_>,
    lhs_position: usize,
    rhs_position: usize,
    commutation: &CommutationChecker,
) -> bool {
    let lhs = block.operation(lhs_position);
    let rhs = block.operation(rhs_position);
    commutation
        .check(
            &lhs.instruction,
            &lhs.qubits,
            block.params(lhs_position),
            &rhs.instruction,
            &rhs.qubits,
            block.params(rhs_position),
        )
        .is_some_and(|result| result.is_exact())
}

fn operation_commutes_with_replacement(
    block: &BlockContext<'_>,
    operation_position: usize,
    replacement: &ReplacementItem,
    commutation: &CommutationChecker,
) -> bool {
    let operation = block.operation(operation_position);
    let replacement_params = replacement
        .params
        .iter()
        .map(|value| match value {
            ParameterValue::Fixed(value) => Parameter::from(*value),
            ParameterValue::Param(parameter) => parameter.clone(),
        })
        .collect::<SmallVec<[_; 3]>>();

    commutation
        .check(
            &operation.instruction,
            &operation.qubits,
            block.params(operation_position),
            &replacement.instruction,
            &replacement.qubits,
            &replacement_params,
        )
        .is_some_and(|result| result.is_exact())
}

fn match_item(
    block: &BlockContext<'_>,
    position: usize,
    item: &RuleItem,
    item_key: &RewriteInstructionKey,
    bindings: &mut MatchBindings,
    config: &RewriteConfig,
) -> Result<bool, CompilerError> {
    let operation = block.operation(position);
    if config.skips_labeled_ops() && operation.label.is_some() {
        return Ok(false);
    }
    if block.key(position) != item_key {
        return Ok(false);
    }

    knowledge_match_rule_item(
        item,
        ConcreteOperationView {
            instruction: &operation.instruction,
            qubits: &operation.qubits,
            params: block.params(position),
        },
        bindings,
    )
    .map_err(|error| CompilerError::InvariantViolation(error.to_string()))
}

pub(super) fn resolve_operation_param(
    circuit: &Circuit,
    param: &CircuitParam,
) -> Result<Parameter, CompilerError> {
    match param {
        CircuitParam::Fixed(value) => Ok(Parameter::from(*value)),
        CircuitParam::Index(index) => circuit
            .parameters()
            .get_index(*index as usize)
            .cloned()
            .ok_or_else(|| {
                CompilerError::InvalidInput(format!("invalid rewrite parameter index {}", index))
            }),
    }
}

/// Computes the local rewrite cost for matched source operations.
///
/// The selector compares local alternatives; full-circuit scheduling is outside
/// the rewrite pass.
fn cost_for_operation_positions(
    block: &BlockContext<'_>,
    positions: &[usize],
    target_context: Option<&TargetContext>,
) -> LocalRewriteCost {
    let mut cost = LocalRewriteCost::default();
    let mut depths = HashMap::new();

    for &position in positions {
        let operation = block.operation(position);
        add_instruction_cost(
            &mut cost,
            &mut depths,
            block.key(position),
            &operation.qubits,
            operation.params.len(),
            GPhaseCost::ExplicitOperation,
            target_context,
        );
    }
    cost
}

fn cost_for_replacements(
    replacements: &[ReplacementItem],
    target_context: Option<&TargetContext>,
) -> LocalRewriteCost {
    let mut cost = LocalRewriteCost::default();
    let mut depths = HashMap::new();

    for replacement in replacements {
        add_instruction_cost(
            &mut cost,
            &mut depths,
            &replacement.key,
            &replacement.qubits,
            replacement.params.len(),
            GPhaseCost::ImplicitReplacement,
            target_context,
        );
    }
    cost
}

fn add_instruction_cost(
    cost: &mut LocalRewriteCost,
    depths: &mut HashMap<Qubit, usize>,
    key: &RewriteInstructionKey,
    qubits: &[Qubit],
    param_count: usize,
    gphase_cost: GPhaseCost,
    target_context: Option<&TargetContext>,
) {
    let target_supported = match target_context {
        Some(target_context) => target_context.physically_supports(key),
        None => true,
    };
    let standard_gate = match key {
        RewriteInstructionKey::Standard(gate) => Some(*gate),
        RewriteInstructionKey::McGate(_) => None,
    };
    let counted = cost.add_gate_like(
        standard_gate,
        target_supported,
        qubits.len(),
        param_count,
        gphase_cost,
    );
    if counted {
        if let Some(target_context) = target_context {
            cost.lowering_distance = cost
                .lowering_distance
                .saturating_add(target_context.lowering_distance(key));
        }
        update_depth_estimate(cost, depths, qubits);
    }
}

/// Updates the local ASAP-style depth estimate for one operation.
///
/// Repeated qubits within one operation are treated as a single dependency
/// edge. The estimate is deterministic and suitable for local ranking, but it
/// is not a substitute for a backend scheduler.
fn update_depth_estimate(
    cost: &mut LocalRewriteCost,
    depths: &mut HashMap<Qubit, usize>,
    qubits: &[Qubit],
) {
    if qubits.is_empty() {
        return;
    }

    let mut unique = SmallVec::<[Qubit; 3]>::new();
    for &qubit in qubits {
        if !unique.contains(&qubit) {
            unique.push(qubit);
        }
    }

    let next_depth = unique
        .iter()
        .filter_map(|qubit| depths.get(qubit))
        .max()
        .copied()
        .unwrap_or(0)
        + 1;
    for qubit in unique {
        depths.insert(qubit, next_depth);
    }
    cost.depth_estimate = cost.depth_estimate.max(next_depth);
}
