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

//! Robust calibration and native-plan costs used by SABRE routing.

use crate::circuit::{Instruction, StandardGate};
#[cfg(test)]
use crate::compile::device_planning::NativePlanCatalog;
use crate::compile::device_planning::{NativePlanLeaf, NativePlanSummary};
use crate::compile::knowledge::KnowledgeInstructionKey;
use crate::device::{Device, PhysicalQubit};
use std::cmp::Ordering;
use std::collections::{HashMap, HashSet};

/// Conservative error comparison key. Lower is better.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub(crate) struct RobustErrorKey {
    pub(crate) unavailable_count: u32,
    pub(crate) imputed_count: u32,
    pub(crate) log_error: f64,
}

impl RobustErrorKey {
    pub(crate) fn compare(self, other: Self) -> Ordering {
        self.unavailable_count
            .cmp(&other.unavailable_count)
            .then_with(|| self.imputed_count.cmp(&other.imputed_count))
            .then_with(|| self.log_error.total_cmp(&other.log_error))
    }

    pub(crate) fn combine(self, other: Self) -> Self {
        Self {
            unavailable_count: self
                .unavailable_count
                .saturating_add(other.unavailable_count),
            imputed_count: self.imputed_count.saturating_add(other.imputed_count),
            log_error: self.log_error + other.log_error,
        }
    }
}

/// Conservative duration-work comparison key. Lower is better.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub(crate) struct RobustDurationKey {
    pub(crate) unavailable_count: u32,
    pub(crate) imputed_count: u32,
    pub(crate) duration_work: f64,
}

impl RobustDurationKey {
    pub(crate) fn compare(self, other: Self) -> Ordering {
        self.unavailable_count
            .cmp(&other.unavailable_count)
            .then_with(|| self.imputed_count.cmp(&other.imputed_count))
            .then_with(|| self.duration_work.total_cmp(&other.duration_work))
    }

    pub(crate) fn combine(self, other: Self) -> Self {
        Self {
            unavailable_count: self
                .unavailable_count
                .saturating_add(other.unavailable_count),
            imputed_count: self.imputed_count.saturating_add(other.imputed_count),
            duration_work: self.duration_work + other.duration_work,
        }
    }
}

/// Native resource cost for one selected lowering plan.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub(crate) struct NativePlanCost {
    pub(crate) native_two_qubit_ops: u32,
    pub(crate) native_total_ops: u32,
    pub(crate) error: Option<RobustErrorKey>,
    pub(crate) duration: Option<RobustDurationKey>,
}

impl NativePlanCost {
    pub(crate) fn combine(self, other: Self) -> Self {
        Self {
            native_two_qubit_ops: self
                .native_two_qubit_ops
                .saturating_add(other.native_two_qubit_ops),
            native_total_ops: self.native_total_ops.saturating_add(other.native_total_ops),
            error: match (self.error, other.error) {
                (Some(left), Some(right)) => Some(left.combine(right)),
                (None, None) => None,
                _ => {
                    debug_assert!(
                        false,
                        "mixed enabled and disabled error costs in one native aggregation"
                    );
                    None
                }
            },
            duration: match (self.duration, other.duration) {
                (Some(left), Some(right)) => Some(left.combine(right)),
                (None, None) => None,
                _ => {
                    debug_assert!(
                        false,
                        "mixed enabled and disabled duration costs in one native aggregation"
                    );
                    None
                }
            },
        }
    }
}

/// Device-wide conservative estimates used only for missing calibration.
#[derive(Debug, Clone, Default)]
pub(crate) struct CalibrationEstimator {
    error_by_gate: HashMap<KnowledgeInstructionKey, f64>,
    error_by_arity: HashMap<usize, f64>,
    duration_by_gate: HashMap<KnowledgeInstructionKey, f64>,
    duration_by_arity: HashMap<usize, f64>,
    error_enabled: bool,
    duration_enabled: bool,
}

#[derive(Default)]
struct CalibrationSamples {
    errors_by_gate: HashMap<KnowledgeInstructionKey, Vec<f64>>,
    errors_by_arity: HashMap<usize, Vec<f64>>,
    durations_by_gate: HashMap<KnowledgeInstructionKey, Vec<f64>>,
    durations_by_arity: HashMap<usize, Vec<f64>>,
}

impl CalibrationSamples {
    fn record(
        &mut self,
        key: KnowledgeInstructionKey,
        arity: usize,
        error: Option<f64>,
        duration: Option<f64>,
    ) {
        if let Some(error) = error {
            self.errors_by_gate
                .entry(key.clone())
                .or_default()
                .push(error);
            self.errors_by_arity.entry(arity).or_default().push(error);
        }
        if let Some(duration) = duration {
            self.durations_by_gate
                .entry(key)
                .or_default()
                .push(duration);
            self.durations_by_arity
                .entry(arity)
                .or_default()
                .push(duration);
        }
    }
}

impl CalibrationEstimator {
    /// Returns the additive identity with the same enabled-metric shape as
    /// every plan scored by this estimator.
    pub(crate) fn identity_cost(&self) -> NativePlanCost {
        NativePlanCost {
            error: self.error_enabled.then_some(RobustErrorKey::default()),
            duration: self
                .duration_enabled
                .then_some(RobustDurationKey::default()),
            ..NativePlanCost::default()
        }
    }

    pub(crate) fn duration_enabled(&self) -> bool {
        self.duration_enabled
    }

    /// Returns the explicit or conservatively imputed duration of one selected
    /// native leaf. `None` means duration scoring is disabled or no device-wide
    /// estimate exists for the missing calibration.
    pub(crate) fn leaf_duration(&self, leaf: &NativePlanLeaf) -> Option<f64> {
        if !self.duration_enabled {
            return None;
        }
        leaf.duration.or_else(|| {
            KnowledgeInstructionKey::from_instruction(&leaf.instruction)
                .as_ref()
                .and_then(|key| self.duration_by_gate.get(key))
                .or_else(|| self.duration_by_arity.get(&leaf.ordered_qargs.len()))
                .copied()
        })
    }

    pub(crate) fn from_device(device: &Device, physical_qubits: &[PhysicalQubit]) -> Self {
        let mut samples = CalibrationSamples::default();
        let usable = physical_qubits.iter().copied().collect::<HashSet<_>>();
        let mut directed_edges = Vec::new();
        for &left in physical_qubits {
            directed_edges.extend(
                device
                    .topology()
                    .successors(left)
                    .filter(|right| usable.contains(right))
                    .map(|right| [left, right]),
            );
        }

        for gate in StandardGate::all().iter().copied() {
            let arity = gate.num_qubits();
            if !(1..=2).contains(&arity) {
                continue;
            }
            let instruction = Instruction::Standard(gate);
            let key = KnowledgeInstructionKey::Standard(gate);
            if arity == 1 {
                for &physical in physical_qubits {
                    collect_device_calibration(
                        device,
                        &instruction,
                        &key,
                        &[physical],
                        &mut samples,
                    );
                }
            } else {
                for ordered in &directed_edges {
                    collect_device_calibration(device, &instruction, &key, ordered, &mut samples);
                }
            }
        }

        Self::from_samples(samples)
    }

    #[cfg(test)]
    pub(crate) fn from_catalog(catalog: &NativePlanCatalog) -> Self {
        let mut samples = CalibrationSamples::default();
        let mut observed_leaves = HashSet::new();

        for (_, summary) in catalog.iter() {
            for leaf in &summary.leaves {
                let Some(key) = KnowledgeInstructionKey::from_instruction(&leaf.instruction) else {
                    continue;
                };
                if !observed_leaves.insert((key.clone(), leaf.ordered_qargs.clone())) {
                    continue;
                }
                samples.record(
                    key,
                    leaf.ordered_qargs.len(),
                    leaf.error_rate,
                    leaf.duration,
                );
            }
        }

        Self::from_samples(samples)
    }

    fn from_samples(samples: CalibrationSamples) -> Self {
        let error_enabled = !samples.errors_by_arity.is_empty();
        let duration_enabled = !samples.durations_by_arity.is_empty();
        Self {
            error_by_gate: quantiles(samples.errors_by_gate),
            error_by_arity: quantiles(samples.errors_by_arity),
            duration_by_gate: quantiles(samples.durations_by_gate),
            duration_by_arity: quantiles(samples.durations_by_arity),
            error_enabled,
            duration_enabled,
        }
    }

    pub(crate) fn cost(&self, summary: &NativePlanSummary) -> NativePlanCost {
        let mut error = RobustErrorKey::default();
        let mut duration = RobustDurationKey::default();

        for leaf in &summary.leaves {
            let key = KnowledgeInstructionKey::from_instruction(&leaf.instruction);
            let arity = leaf.ordered_qargs.len();
            if self.error_enabled {
                match leaf.error_rate {
                    Some(value) => error.log_error += negative_log_success(value),
                    None => match key
                        .as_ref()
                        .and_then(|key| self.error_by_gate.get(key))
                        .or_else(|| self.error_by_arity.get(&arity))
                        .copied()
                    {
                        Some(value) => {
                            error.imputed_count += 1;
                            error.log_error += negative_log_success(value);
                        }
                        None => error.unavailable_count += 1,
                    },
                }
            }

            if self.duration_enabled {
                match leaf.duration {
                    Some(value) => duration.duration_work += value,
                    None => match key
                        .as_ref()
                        .and_then(|key| self.duration_by_gate.get(key))
                        .or_else(|| self.duration_by_arity.get(&arity))
                        .copied()
                    {
                        Some(value) => {
                            duration.imputed_count += 1;
                            duration.duration_work += value;
                        }
                        None => duration.unavailable_count += 1,
                    },
                }
            }
        }

        NativePlanCost {
            native_two_qubit_ops: summary.native_two_qubit_ops,
            native_total_ops: summary.native_total_ops,
            error: self.error_enabled.then_some(error),
            duration: self.duration_enabled.then_some(duration),
        }
    }
}

fn collect_device_calibration(
    device: &Device,
    instruction: &Instruction,
    key: &KnowledgeInstructionKey,
    qargs: &[PhysicalQubit],
    samples: &mut CalibrationSamples,
) {
    if !device.supports_native_instruction(instruction, qargs) {
        return;
    }
    let Some(calibration) = device.native_instruction_calibration(instruction, qargs) else {
        return;
    };
    samples.record(
        key.clone(),
        qargs.len(),
        calibration.error_rate,
        calibration.duration,
    );
}

fn negative_log_success(error: f64) -> f64 {
    if error == 1.0 {
        f64::INFINITY
    } else {
        -(-error).ln_1p()
    }
}

fn quantiles<K: Eq + std::hash::Hash>(values: HashMap<K, Vec<f64>>) -> HashMap<K, f64> {
    values
        .into_iter()
        .map(|(key, mut values)| {
            values.sort_by(f64::total_cmp);
            let index = ((values.len() * 9).div_ceil(10)).saturating_sub(1);
            (key, values[index])
        })
        .collect()
}

#[cfg(test)]
#[path = "cost_test.rs"]
mod cost_test;
