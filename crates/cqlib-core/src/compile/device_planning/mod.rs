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

//! Shared exact-qargs device planning used by routing and final lowering.

pub(crate) mod cost;
mod planner;
mod templates;

use crate::circuit::{Instruction, StandardGate};
use crate::compile::CompilerError;
use crate::compile::error::DeviceLoweringFailure;
use crate::compile::knowledge::{KnowledgeInstructionKey, RuleLibrary};
use crate::device::{Device, PhysicalQubit};
use smallvec::SmallVec;
use std::collections::HashMap;

pub(crate) use cost::{
    CalibrationEstimator, DevicePhysicalCost, NativePlanCost, NativePlanLeaf, NativePlanSummary,
};
pub(crate) use planner::{DevicePlanner, DevicePlannerError, PlanChoice, PlanId, PlanTemplate};
pub(crate) use templates::DirectionTemplate;

/// A parameter-independent gate state on exact ordered physical qargs.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub(crate) struct DeviceGateState {
    pub(crate) instruction: KnowledgeInstructionKey,
    pub(crate) ordered_qargs: SmallVec<[PhysicalQubit; 2]>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::circuit::{Instruction, StandardGate};
    use smallvec::smallvec;

    #[test]
    fn catalog_summarizes_the_same_native_swap_plan_used_by_lowering() {
        let device = Device::line("native-plan-summary", 2)
            .unwrap()
            .with_native_gates(vec![
                Instruction::Standard(StandardGate::H),
                Instruction::Standard(StandardGate::CX),
            ])
            .unwrap()
            .with_default_single_qubit_error(0.001)
            .with_default_two_qubit_error(0.01);
        let root = DeviceGateState::standard(
            StandardGate::SWAP,
            smallvec![PhysicalQubit::new(0), PhysicalQubit::new(1)],
        );

        let catalog = NativePlanCatalog::build(&device, [root.clone()]).unwrap();
        let summary = catalog.summary(&root).expect("SWAP should be lowerable");

        assert_eq!(summary.native_two_qubit_ops, 3);
        assert_eq!(summary.native_total_ops, 7);
        assert_eq!(
            summary
                .leaves
                .iter()
                .filter(|leaf| matches!(leaf.instruction, Instruction::Standard(StandardGate::CX)))
                .count(),
            3
        );
        assert!(summary.leaves.iter().all(|leaf| match leaf.instruction {
            Instruction::Standard(StandardGate::CX) => leaf.error_rate == Some(0.01),
            Instruction::Standard(StandardGate::H) => leaf.error_rate == Some(0.001),
            _ => false,
        }));
    }

    #[test]
    fn catalog_distinguishes_unsupported_from_unprepared_roots() {
        let device = Device::line("native-plan-availability", 2).unwrap();
        let unsupported = DeviceGateState::standard(
            StandardGate::CX,
            smallvec![PhysicalQubit::new(0), PhysicalQubit::new(1)],
        );
        let unprepared = DeviceGateState::standard(
            StandardGate::CZ,
            smallvec![PhysicalQubit::new(0), PhysicalQubit::new(1)],
        );

        let catalog = NativePlanCatalog::build(&device, [unsupported.clone()]).unwrap();

        assert!(matches!(
            catalog.availability(&unsupported),
            Some(NativePlanAvailability::Unsupported(_))
        ));
        assert!(catalog.availability(&unprepared).is_none());
    }
}

impl DeviceGateState {
    pub(crate) fn standard(
        gate: StandardGate,
        ordered_qargs: SmallVec<[PhysicalQubit; 2]>,
    ) -> Self {
        Self {
            instruction: KnowledgeInstructionKey::Standard(gate),
            ordered_qargs,
        }
    }

    pub(crate) fn from_instruction(
        instruction: &Instruction,
        ordered_qargs: SmallVec<[PhysicalQubit; 2]>,
    ) -> Option<Self> {
        Some(Self {
            instruction: KnowledgeInstructionKey::from_instruction(instruction)?,
            ordered_qargs,
        })
    }

    pub(crate) fn stable_sort_key(&self) -> String {
        format!("{:?}:{:?}", self.instruction, self.ordered_qargs)
    }
}

/// Result of planning one requested exact-device root.
#[derive(Debug, Clone)]
pub(crate) enum NativePlanAvailability {
    /// The root has a selected exact-qargs native lowering plan.
    Feasible(NativePlanSummary),
    /// The root was requested, but no native lowering plan exists.
    Unsupported(DeviceLoweringFailure),
}

/// Immutable exact-device planning results used by routing cost preparation.
///
/// Every requested root is retained. A missing map entry therefore means the
/// caller failed to prepare that state, rather than that planning proved it
/// unsupported.
#[derive(Debug, Clone)]
pub(crate) struct NativePlanCatalog {
    availability: HashMap<DeviceGateState, NativePlanAvailability>,
}

impl NativePlanCatalog {
    pub(crate) fn build(
        device: &Device,
        roots: impl IntoIterator<Item = DeviceGateState>,
    ) -> Result<Self, CompilerError> {
        let library = RuleLibrary::builtin_rules()
            .map_err(|error| CompilerError::InvariantViolation(error.to_string()))?;
        let mut roots = roots.into_iter().collect::<Vec<_>>();
        roots.sort_by_key(DeviceGateState::stable_sort_key);
        roots.dedup();
        let planner = DevicePlanner::build(device, library, roots.iter().cloned())
            .map_err(DevicePlannerError::into_compiler_error)?;
        let mut availability = HashMap::with_capacity(roots.len());
        for root in roots {
            let planned = if let Some(summary) = planner
                .summary_for(&root)
                .map_err(DevicePlannerError::into_compiler_error)?
            {
                NativePlanAvailability::Feasible(summary)
            } else {
                NativePlanAvailability::Unsupported(planner.failure_for(&root))
            };
            availability.insert(root, planned);
        }
        Ok(Self { availability })
    }

    pub(crate) fn availability(&self, state: &DeviceGateState) -> Option<&NativePlanAvailability> {
        self.availability.get(state)
    }

    pub(crate) fn summary(&self, state: &DeviceGateState) -> Option<&NativePlanSummary> {
        match self.availability(state) {
            Some(NativePlanAvailability::Feasible(summary)) => Some(summary),
            Some(NativePlanAvailability::Unsupported(_)) | None => None,
        }
    }

    pub(crate) fn iter(&self) -> impl Iterator<Item = (&DeviceGateState, &NativePlanSummary)> {
        self.availability
            .iter()
            .filter_map(|(state, availability)| {
                let NativePlanAvailability::Feasible(summary) = availability else {
                    return None;
                };
                Some((state, summary))
            })
    }

    pub(crate) fn iter_availability(
        &self,
    ) -> impl Iterator<Item = (&DeviceGateState, &NativePlanAvailability)> {
        self.availability.iter()
    }
}
