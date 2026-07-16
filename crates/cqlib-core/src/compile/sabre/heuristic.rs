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

//! SABRE trial and SWAP-selection configuration.
//!
//! The router evaluates candidate SWAPs with a weighted distance score:
//!
//! ```text
//! score = basic_weight * sum(front_layer_distance)
//!       + sum_i lookahead_weights[i] * sum(lookahead_layer_i_distance) / device_width
//!       + decay_penalty(swap)
//! ```
//!
//! Lower scores are preferred. The front layer contains unary and two-qubit
//! routing requirements that are blocked by their current physical placement.
//! Lookahead layers bias the local decision toward requirements that become
//! relevant soon after the current front layer is routed. Keeping the front as
//! a sum preserves its weight relative to lookahead as the front grows;
//! device-width normalization keeps lookahead comparable across targets.
//!
//! Decay is optional. When enabled, physical qubits recently used in heuristic
//! SWAPs receive a slightly larger additive penalty, discouraging repeated movement
//! around the same area of the device and improving parallelism. The decay
//! table is reset after [`SabreHeuristicConfig::decay_reset`] heuristic SWAPs.
//!
//! [`SabreTrialObjective`] controls how independent routing trials are compared
//! after they produce complete routed circuits. It does not change the local
//! SWAP score; it changes only final trial selection and layout refinement
//! tie-breaking.

use crate::compile::CompilerError;

/// Objective used to select the best result among independent SABRE trials.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SabreTrialObjective {
    /// Minimize inserted SWAP count only.
    SwapCount,
    /// Minimize routed two-qubit depth only.
    ///
    /// Use this for depth-sensitive targets where a few extra SWAPs may be
    /// acceptable if they shorten the two-qubit critical path.
    Depth,
    /// Select final native quality within a bounded abstract-SWAP regret.
    ///
    /// This objective filters complete trials by [`SabreConfig::swap_regret_ratio`],
    /// then compares native two-qubit count/depth, robust error, duration, and
    /// total native operations. Local greedy SWAP selection remains topology
    /// constrained; broader budget exploration belongs to bounded search.
    NativeQualityWithinSwapBudget,
    /// Minimize routed two-qubit depth first, then SWAP count.
    ///
    /// Use this when depth is the primary objective but SWAP count should still
    /// break ties deterministically.
    DepthThenSwap,
}

/// Bounded VF2 prepass used to seed SABRE layout candidates.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SabreVf2PrepassConfig {
    /// Maximum number of complete perfect mappings scored by VF2.
    pub candidate_limit: usize,
    /// Maximum number of partial mapping extensions attempted by VF2.
    pub call_limit: usize,
}

/// Configuration shared by SABRE layout refinement and routing.
#[derive(Debug, Clone, PartialEq)]
pub struct SabreConfig {
    /// Number of starting-layout trials considered during layout refinement.
    pub layout_trials: usize,
    /// Maximum component-assignment states explored before layout reports
    /// budget exhaustion. Exhaustion is distinct from proven infeasibility.
    pub layout_assignment_budget: usize,
    /// Optional bounded VF2 prepass used to add a topology-perfect candidate.
    /// `None` disables the prepass.
    pub vf2_prepass: Option<SabreVf2PrepassConfig>,
    /// Number of forward/backward refinement iterations per layout trial.
    pub refinement_iterations: usize,
    /// Number of routing trials used to score each refined layout candidate.
    pub layout_scoring_trials: usize,
    /// Number of random routing trials used to select a final routed circuit.
    pub routing_trials: usize,
    /// Objective used to choose among equally valid routing trials.
    pub trial_objective: SabreTrialObjective,
    /// Maximum relative abstract-SWAP regret accepted by the native-quality
    /// trial objective. `0.05` permits `ceil(best_swap_count * 0.05)` extra
    /// SWAPs. Other objectives ignore this field.
    pub swap_regret_ratio: f64,
    /// Optional deterministic seed.  Equal seeds produce equal cqlib results.
    pub seed: Option<u64>,
    /// Swap-selection heuristic configuration.
    pub heuristic: SabreHeuristicConfig,
}

/// Swap-selection heuristic used by SABRE.
///
/// The score combines the current front-layer sum, device-width-normalized
/// lookahead and an optional additive decay penalty. Lower scores are preferred.
#[derive(Debug, Clone, PartialEq)]
pub struct SabreHeuristicConfig {
    /// Weight of the current front-layer total distance.
    pub basic_weight: f64,
    /// Weights of each device-width-normalized lookahead-layer total.
    pub lookahead_weights: Vec<f64>,
    /// Amount added to a physical qubit's decay value after using it in a
    /// heuristic SWAP. `None` disables the additive decay penalty.
    pub decay_increment: Option<f64>,
    /// Number of heuristic SWAP attempts before decay values reset.
    pub decay_reset: usize,
    /// Number of heuristic SWAPs allowed without routing a front-layer node
    /// before SABRE falls back to a shortest-path escape.
    pub attempt_limit: usize,
    /// Floating-point tolerance for treating candidate SWAP scores as tied.
    pub best_epsilon: f64,
}

impl Default for SabreHeuristicConfig {
    fn default() -> Self {
        Self {
            basic_weight: 1.0,
            lookahead_weights: vec![0.5],
            decay_increment: Some(0.001),
            decay_reset: 5,
            attempt_limit: 1000,
            best_epsilon: 1e-10,
        }
    }
}

impl Default for SabreConfig {
    fn default() -> Self {
        Self {
            layout_trials: 10,
            layout_assignment_budget: 1_000_000,
            vf2_prepass: Some(SabreVf2PrepassConfig {
                candidate_limit: 10,
                call_limit: 1_000_000,
            }),
            refinement_iterations: 1,
            layout_scoring_trials: 1,
            routing_trials: 5,
            trial_objective: SabreTrialObjective::NativeQualityWithinSwapBudget,
            swap_regret_ratio: 0.05,
            seed: None,
            heuristic: SabreHeuristicConfig::default(),
        }
    }
}

impl SabreConfig {
    /// Returns a compact deterministic SABRE configuration for reproducible tests and examples.
    ///
    /// This keeps all trial counts small, fixes the random seed, and uses a
    /// bounded swap-attempt limit so small fixtures run quickly while still
    /// exercising the SABRE routing path.
    pub fn deterministic_seeded(seed: u64) -> Self {
        Self {
            layout_trials: 2,
            layout_assignment_budget: 100_000,
            vf2_prepass: Some(SabreVf2PrepassConfig {
                candidate_limit: 10,
                call_limit: 100_000,
            }),
            refinement_iterations: 1,
            layout_scoring_trials: 1,
            routing_trials: 1,
            trial_objective: SabreTrialObjective::NativeQualityWithinSwapBudget,
            swap_regret_ratio: 0.05,
            seed: Some(seed),
            heuristic: SabreHeuristicConfig {
                lookahead_weights: vec![0.5],
                attempt_limit: 20,
                ..SabreHeuristicConfig::default()
            },
        }
    }
}

impl SabreHeuristicConfig {
    pub(crate) fn validate(&self) -> Result<(), CompilerError> {
        validate_weight(self.basic_weight, "sabre basic_weight")?;
        for (index, weight) in self.lookahead_weights.iter().copied().enumerate() {
            validate_weight(weight, &format!("sabre lookahead_weights[{index}]"))?;
        }
        if let Some(increment) = self.decay_increment {
            validate_weight(increment, "sabre decay_increment")?;
            if self.decay_reset == 0 {
                return Err(CompilerError::InvalidInput(
                    "sabre decay_reset must be greater than zero when decay is enabled".to_string(),
                ));
            }
        }
        if !(self.best_epsilon.is_finite() && self.best_epsilon >= 0.0) {
            return Err(CompilerError::InvalidInput(
                "sabre best_epsilon must be finite and non-negative".to_string(),
            ));
        }
        Ok(())
    }
}

fn validate_weight(value: f64, name: &str) -> Result<(), CompilerError> {
    if value.is_finite() && value >= 0.0 {
        Ok(())
    } else {
        Err(CompilerError::InvalidInput(format!(
            "{name} must be finite and non-negative"
        )))
    }
}

#[cfg(test)]
#[path = "heuristic_test.rs"]
mod heuristic_test;
