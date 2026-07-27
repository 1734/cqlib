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

//! Pass-local, bit-exact caching for two-qubit synthesis planners.
//!
//! The cache stores planner candidates rather than selected patches. A caller
//! must therefore repeat source-cost, commutation, span, and overlap checks for
//! every block. Keys preserve the exact complex floating-point representation
//! and ordered qubit arguments; numerically close or phase-equivalent matrices
//! are deliberately unrelated.
//!
//! Capacity is bounded with an admission-only policy. Existing entries remain
//! available after the budget is reached, while new keys are planned normally
//! and consumed without insertion. Planner failures and empty candidate lists
//! are both cached so deterministic negative results do not repeat expensive
//! work.

use crate::circuit::Qubit;
use crate::compile::CompilerError;
use crate::compile::transform::decompose::unitary::TwoQubitSynthesisCandidate;
use crate::compile::transform::decompose::unitary::unitary_2q::DeviceTwoQubitSynthesisCandidate;
use ndarray::Array2;
use num_complex::Complex64;
use std::collections::HashMap;

pub(super) const RESYNTHESIS_SYNTHESIS_CACHE_BUDGET: usize = 4096;

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
struct ExactTwoQubitSynthesisKey {
    matrix_bits: [(u64, u64); 16],
    ordered_qargs: [Qubit; 2],
}

impl ExactTwoQubitSynthesisKey {
    fn new(matrix: &Array2<Complex64>, ordered_qargs: [Qubit; 2]) -> Result<Self, CompilerError> {
        if matrix.dim() != (4, 4) {
            return Err(CompilerError::InvariantViolation(format!(
                "2q synthesis cache requires a 4x4 matrix, got {}x{}",
                matrix.nrows(),
                matrix.ncols()
            )));
        }
        // Index by logical row and column so the key is independent of ndarray
        // storage layout while preserving every floating-point bit exactly.
        let matrix_bits = std::array::from_fn(|index| {
            let value = matrix[(index / 4, index % 4)];
            (value.re.to_bits(), value.im.to_bits())
        });
        Ok(Self {
            matrix_bits,
            ordered_qargs,
        })
    }
}

#[derive(Debug, Clone)]
enum CachedPlan<T> {
    Candidates(Vec<T>),
    Failed,
}

#[derive(Debug, Clone, Copy)]
pub(super) enum CachedPlanView<'a, T> {
    Candidates(&'a [T]),
    Failed,
}

impl<T> CachedPlan<T> {
    fn view(&self) -> CachedPlanView<'_, T> {
        match self {
            Self::Candidates(candidates) => CachedPlanView::Candidates(candidates),
            Self::Failed => CachedPlanView::Failed,
        }
    }
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub(crate) struct TwoQubitSynthesisCacheStats {
    pub(crate) generic_lookups: usize,
    pub(crate) generic_hits: usize,
    pub(crate) generic_misses: usize,
    pub(crate) generic_entries: usize,
    pub(crate) device_lookups: usize,
    pub(crate) device_hits: usize,
    pub(crate) device_misses: usize,
    pub(crate) device_entries: usize,
    pub(crate) failed_plan_hits: usize,
    pub(crate) capacity_rejections: usize,
}

/// Both maps are valid only for the fixed synthesis target and optional device
/// context owned by the surrounding `ResynthesisPass`. If this cache ever
/// outlives one pass, its keys must also include the target, placement mode,
/// and a device-context signature.
#[derive(Debug)]
pub(super) struct TwoQubitSynthesisCache {
    generic: HashMap<ExactTwoQubitSynthesisKey, CachedPlan<TwoQubitSynthesisCandidate>>,
    device: HashMap<ExactTwoQubitSynthesisKey, CachedPlan<DeviceTwoQubitSynthesisCandidate>>,
    stats: TwoQubitSynthesisCacheStats,
    budget: usize,
}

impl Default for TwoQubitSynthesisCache {
    fn default() -> Self {
        Self::new(RESYNTHESIS_SYNTHESIS_CACHE_BUDGET)
    }
}

impl TwoQubitSynthesisCache {
    pub(super) fn new(budget: usize) -> Self {
        Self {
            generic: HashMap::new(),
            device: HashMap::new(),
            stats: TwoQubitSynthesisCacheStats::default(),
            budget,
        }
    }

    pub(super) const fn stats(&self) -> TwoQubitSynthesisCacheStats {
        self.stats
    }

    pub(super) fn with_generic_plan<R>(
        &mut self,
        matrix: &Array2<Complex64>,
        ordered_qargs: [Qubit; 2],
        planner: impl FnOnce() -> Result<Vec<TwoQubitSynthesisCandidate>, CompilerError>,
        consume: impl FnOnce(CachedPlanView<'_, TwoQubitSynthesisCandidate>) -> R,
    ) -> Result<R, CompilerError> {
        let key = ExactTwoQubitSynthesisKey::new(matrix, ordered_qargs)?;
        self.stats.generic_lookups = self.stats.generic_lookups.saturating_add(1);
        if let Some(plan) = self.generic.get(&key) {
            self.stats.generic_hits = self.stats.generic_hits.saturating_add(1);
            if matches!(plan, CachedPlan::Failed) {
                self.stats.failed_plan_hits = self.stats.failed_plan_hits.saturating_add(1);
            }
            return Ok(consume(plan.view()));
        }

        self.stats.generic_misses = self.stats.generic_misses.saturating_add(1);
        let plan = match planner() {
            Ok(candidates) => CachedPlan::Candidates(candidates),
            Err(_) => CachedPlan::Failed,
        };
        if self.generic.len() < self.budget {
            self.generic.insert(key.clone(), plan);
            self.stats.generic_entries = self.generic.len();
            return Ok(consume(
                self.generic
                    .get(&key)
                    .expect("newly inserted generic synthesis plan")
                    .view(),
            ));
        }

        self.stats.capacity_rejections = self.stats.capacity_rejections.saturating_add(1);
        Ok(consume(plan.view()))
    }

    pub(super) fn with_device_plan<R>(
        &mut self,
        matrix: &Array2<Complex64>,
        ordered_qargs: [Qubit; 2],
        planner: impl FnOnce() -> Result<Vec<DeviceTwoQubitSynthesisCandidate>, CompilerError>,
        consume: impl FnOnce(CachedPlanView<'_, DeviceTwoQubitSynthesisCandidate>) -> R,
    ) -> Result<R, CompilerError> {
        let key = ExactTwoQubitSynthesisKey::new(matrix, ordered_qargs)?;
        self.stats.device_lookups = self.stats.device_lookups.saturating_add(1);
        if let Some(plan) = self.device.get(&key) {
            self.stats.device_hits = self.stats.device_hits.saturating_add(1);
            if matches!(plan, CachedPlan::Failed) {
                self.stats.failed_plan_hits = self.stats.failed_plan_hits.saturating_add(1);
            }
            return Ok(consume(plan.view()));
        }

        self.stats.device_misses = self.stats.device_misses.saturating_add(1);
        let plan = match planner() {
            Ok(candidates) => CachedPlan::Candidates(candidates),
            Err(_) => CachedPlan::Failed,
        };
        if self.device.len() < self.budget {
            self.device.insert(key.clone(), plan);
            self.stats.device_entries = self.device.len();
            return Ok(consume(
                self.device
                    .get(&key)
                    .expect("newly inserted device synthesis plan")
                    .view(),
            ));
        }

        self.stats.capacity_rejections = self.stats.capacity_rejections.saturating_add(1);
        Ok(consume(plan.view()))
    }
}

#[cfg(test)]
#[path = "synthesis_cache_test.rs"]
mod synthesis_cache_test;
