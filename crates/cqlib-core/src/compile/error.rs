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

use crate::circuit::{CircuitError, Instruction};
use crate::device::{DeviceValidationError, PhysicalQubit};
use std::fmt;
use thiserror::Error;

/// One exact device-lowering dependency for which no plan exists.
#[derive(Debug, Clone)]
pub struct DeviceLoweringDependency {
    /// Instruction required by the candidate.
    pub instruction: Instruction,
    /// Exact ordered physical arguments required by the candidate.
    pub qargs: Vec<PhysicalQubit>,
}

/// One unsuccessful candidate considered while lowering a device instruction.
#[derive(Debug, Clone)]
pub struct DeviceLoweringCandidateFailure {
    /// Stable direction-template or knowledge-rule name.
    pub template: String,
    /// Unique candidate dependencies for which the planner found no native plan.
    pub unsatisfied_dependencies: Vec<DeviceLoweringDependency>,
}

/// Structured diagnostics for an operation with no device instruction plan.
#[derive(Debug, Clone)]
pub struct DeviceLoweringFailure {
    /// Source instruction that could not be lowered.
    pub instruction: Instruction,
    /// Ordered physical arguments on which lowering was attempted.
    pub qargs: Vec<PhysicalQubit>,
    /// Candidate templates and their unresolved exact-qargs dependencies.
    pub attempted_candidates: Vec<DeviceLoweringCandidateFailure>,
}

impl fmt::Display for DeviceLoweringFailure {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "no device instruction lowering plan for {} on ordered physical qargs {:?}",
            self.instruction, self.qargs
        )
    }
}

impl std::error::Error for DeviceLoweringFailure {}

/// Errors raised by compiler infrastructure and compiler state validation.
#[derive(Debug, Error)]
pub enum CompilerError {
    /// Conversion or validation of the circuit control-flow graph failed.
    #[error(transparent)]
    Circuit(#[from] CircuitError),
    /// The input compiler state or circuit does not satisfy a pass precondition.
    #[error("invalid compiler input: {0}")]
    InvalidInput(String),
    /// A compiler transform could not complete its declared operation.
    #[error("compiler transform '{name}' failed: {reason}")]
    TransformFailed {
        /// Stable transform or synthesis primitive name.
        name: &'static str,
        /// Human-readable diagnostic describing why the transform failed.
        reason: String,
    },
    /// A compiler pass produced a state that violates its declared contract.
    #[error("compiler invariant violation: {0}")]
    InvariantViolation(String),
    /// No exact-qargs native lowering plan exists for an operation.
    #[error("device instruction lowering failed: {0}")]
    DeviceLoweringFailed(#[source] DeviceLoweringFailure),
    /// The final circuit violates the configured device execution contract.
    #[error("device validation failed: {0}")]
    DeviceValidationFailed(#[from] DeviceValidationError),
}
