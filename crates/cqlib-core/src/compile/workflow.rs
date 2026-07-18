// This code is part of Cqlib.
//
// (C) Copyright China Telecom Quantum Group 2025-2026
//
// This code is licensed under the Apache License, Version 2.0.
// You may obtain a copy of this license in the LICENSE.txt file in
// the root directory of this source tree or at
// http://www.apache.org/licenses/LICENSE-2.0.
//
// Any modifications or derivative works of this code must retain this
// copyright notice, and modified files need to carry a notice indicating
// that they have been altered from the originals.

//! Workflow-level orchestration for the new compiler optimization pipeline.
//!
//! The workflow is a staged composition layer, not an optimization algorithm.
//! It resolves target constraints, runs completed compiler transforms in a
//! deterministic order, and records only the postconditions it can actually
//! verify with the compiler capabilities currently implemented.
//!
//! The normal workflow follows the stable pass order:
//! canonicalize input, expand circuit-backed definitions, apply production
//! knowledge rewrite, decompose unitary and multi-controlled gates,
//! canonicalize again, optimize the decomposed circuit, optionally lower to a
//! routing-compatible basis, optionally route on a device, optionally translate
//! to the resolved target basis, and canonicalize the output representation. A
//! device workflow then lowers every gate to exact ordered native capabilities,
//! closes a bounded native-optimization loop, and validates the completed
//! physical circuit before returning it.
//!
//! The enhanced workflow uses the same required correctness stages but raises
//! rewrite budgets, uses stronger SABRE trial settings, performs a
//! post-routing cleanup pass, and adds a target-aware cleanup pass after
//! target-basis translation. This keeps `Normal` suitable for predictable
//! production compilation while giving `Enhanced` more chances to recover
//! simplifications exposed by decomposition, routing, and lowering.
//!
//! Stages are deliberately ordered around compiler invariants. Early
//! canonicalization gives later passes a stable representation, definition and
//! high-level gate decomposition remove operations that routing cannot accept,
//! routing runs before final target-basis cleanup because it may insert SWAPs,
//! and output canonicalization removes representation noise before exact device
//! lowering. Native optimization is re-legalized and costed on exact physical
//! qargs; device validation remains terminal, with no transform after it.

use crate::circuit::{Circuit, Instruction};
use crate::compile::CompilerError;
use crate::compile::physical_target::PhysicalLayoutGraph;
use crate::compile::resource::ResourceLimits;
use crate::compile::sabre::{SabreConfig, SabreHeuristicConfig, SabreTrialObjective};
use crate::compile::transform::decompose::unitary::{
    DeviceSynthesisPlacement, DeviceTwoQubitSynthesisContext, TwoQubitSynthesisTarget,
};
use crate::compile::transform::decompose::{
    DecomposeDefinitions, DecomposeMcGates, DecomposeUnitaries, McGateDecomposeConfig,
    UnitaryDecomposeConfig,
};
use crate::compile::transform::native_optimization::NativeOptimizer;
use crate::compile::transform::{
    Canonicalizer, CircuitAnalysis, DeviceLowerer, KnowledgeRewriter, LayoutObjective,
    LowerToRoutingBasis, ResynthesizeTwoQubitBlocks, RewriteConfig, TargetBasisLowerer,
    TransformResult, Transformer, TwoQubitBlockResynthesisConfig, route_sabre, route_with_layout,
};

use super::{
    CompileConfig, CompileMode, CompileResult, CompileTarget, DeviceCompilationMetadata,
    DeviceCompileTarget,
};

/// Per-step execution record produced by a workflow run.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WorkflowStepReport {
    /// Coarse workflow stage, following the staged-pass-manager model.
    pub stage: &'static str,
    /// Workflow-local step name.
    pub name: &'static str,
    /// Whether this step changed the circuit representation.
    pub changed: bool,
    /// Whether the step was intentionally skipped.
    pub skipped: bool,
    /// Optional skip or configuration note.
    pub reason: Option<String>,
}

struct WorkflowState {
    current: Circuit,
    analysis: CircuitAnalysis,
    changed: bool,
    steps: Vec<WorkflowStepReport>,
    target_basis: Option<Vec<Instruction>>,
    two_qubit_target: TwoQubitSynthesisTarget,
    device_metadata: Option<DeviceCompilationMetadata>,
}

impl WorkflowState {
    fn apply_transform(
        &mut self,
        stage: &'static str,
        name: &'static str,
        transform: impl FnOnce(&Circuit, &CircuitAnalysis) -> Result<TransformResult, CompilerError>,
    ) -> Result<(), CompilerError> {
        let TransformResult { circuit, changed } = transform(&self.current, &self.analysis)?;
        if changed {
            self.analysis = CircuitAnalysis::analyze(&circuit);
        }
        self.current = circuit;
        self.changed |= changed;
        self.steps.push(WorkflowStepReport {
            stage,
            name,
            changed,
            skipped: false,
            reason: None,
        });
        Ok(())
    }

    fn record_skipped(
        &mut self,
        stage: &'static str,
        name: &'static str,
        reason: impl Into<String>,
    ) {
        self.steps.push(WorkflowStepReport {
            stage,
            name,
            changed: false,
            skipped: true,
            reason: Some(reason.into()),
        });
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum RewritePhase {
    PreDecomposition,
    PostDecomposition,
    PostRouting,
    TargetCleanup,
}

/// Compiler optimization workflow built from completed compiler transforms.
pub struct CompilerWorkflow {
    config: CompileConfig,
}

impl CompilerWorkflow {
    /// Creates a compiler workflow from a complete configuration.
    pub const fn new(config: CompileConfig) -> Self {
        Self { config }
    }

    /// Returns the workflow configuration.
    pub const fn config(&self) -> &CompileConfig {
        &self.config
    }

    /// Runs the workflow over `circuit` and returns the rebuilt circuit plus
    /// execution metadata.
    pub fn run(&self, circuit: &Circuit) -> Result<CompileResult, CompilerError> {
        let resolved_target = self.resolve_target_basis()?;
        let two_qubit_target =
            TwoQubitSynthesisTarget::from_instructions(resolved_target.as_deref())?;
        let mut state = WorkflowState {
            current: circuit.clone(),
            analysis: CircuitAnalysis::analyze(circuit),
            changed: false,
            steps: Vec::new(),
            target_basis: resolved_target,
            two_qubit_target,
            device_metadata: None,
        };

        self.record_pre_init(&mut state);
        self.validate_resources(circuit, &mut state)?;
        self.lower_init(&mut state)?;
        self.lower_decompose(&mut state)?;
        self.lower_optimize(&mut state)?;
        self.lower_routing_basis(&mut state)?;
        self.lower_physical(&mut state)?;
        self.lower_target(&mut state)?;
        self.lower_output(&mut state)?;
        self.lower_device_instructions(&mut state)?;
        self.canonicalize_native_input(&mut state)?;
        self.optimize_native_instructions(&mut state)?;
        self.validate_device(&mut state)?;

        Ok(CompileResult {
            circuit: state.current,
            changed: state.changed,
            mode: self.config.mode,
            steps: state.steps,
            device_metadata: state.device_metadata,
        })
    }

    /// Establishes a stable high-level IR before gate-specific lowering.
    ///
    /// Definition expansion precedes the first rewrite pass so knowledge rules
    /// see the operations contained by user-defined gates.
    fn lower_init(&self, state: &mut WorkflowState) -> Result<(), CompilerError> {
        state.apply_transform("init", "canonicalize.input", |circuit, analysis| {
            Canonicalizer::production().transform(circuit, Some(analysis))
        })?;
        self.apply_definition_decomposition(state)?;
        let rewrite_config =
            self.rewrite_config_for_state(RewritePhase::PreDecomposition, state)?;
        state.apply_transform(
            "optimization",
            "optimize.pre_decomposition",
            |circuit, analysis| {
                KnowledgeRewriter::new(rewrite_config).transform(circuit, Some(analysis))
            },
        )
    }

    /// Lowers opaque unitary and multi-controlled operations.
    ///
    /// Routing and target-basis translation only operate on concrete operation
    /// families, so this stage runs before physical and target lowering.
    fn lower_decompose(&self, state: &mut WorkflowState) -> Result<(), CompilerError> {
        self.apply_unitary_decomposition(state)?;
        self.apply_mc_gate_decomposition(state)?;
        state.apply_transform(
            "optimization",
            "canonicalize.after_decomposition",
            |circuit, analysis| Canonicalizer::production().transform(circuit, Some(analysis)),
        )?;
        self.apply_two_qubit_resynthesis(
            state,
            "optimization",
            "resynthesize.two_qubit_blocks",
            DeviceSynthesisPlacement::PreLayoutEnvelope,
        )
    }

    fn lower_optimize(&self, state: &mut WorkflowState) -> Result<(), CompilerError> {
        let rewrite_config =
            self.rewrite_config_for_state(RewritePhase::PostDecomposition, state)?;
        state.apply_transform(
            "optimization",
            "optimize.post_decomposition",
            |circuit, analysis| {
                KnowledgeRewriter::new(rewrite_config).transform(circuit, Some(analysis))
            },
        )
    }

    /// Applies optional routing-basis lowering before physical routing.
    ///
    /// This stage is intentionally separate from final target-basis
    /// translation. It only guarantees that standard gates unsupported by
    /// SABRE's arity model, such as `CCX`, are lowered to operations with at
    /// most two qubits before layout and routing run.
    fn lower_routing_basis(&self, state: &mut WorkflowState) -> Result<(), CompilerError> {
        self.apply_routing_basis_decomposition(state)
    }

    /// Applies optional physical lowering from logical to physical qubits.
    fn lower_physical(&self, state: &mut WorkflowState) -> Result<(), CompilerError> {
        self.apply_layout_and_routing(state)?;
        if self.config.mode == CompileMode::Enhanced {
            self.apply_post_routing_resynthesis(state)?;
            self.apply_post_routing_cleanup(state)?;
        }
        Ok(())
    }

    /// Applies optional target-basis translation and target-aware cleanup.
    ///
    /// This stage runs after routing because routing may insert SWAPs and expose
    /// new target-aware rewrite opportunities.
    fn lower_target(&self, state: &mut WorkflowState) -> Result<(), CompilerError> {
        self.apply_target_translation(state)?;
        if self.config.mode == CompileMode::Enhanced {
            let cleanup_config =
                self.rewrite_config_for_state(RewritePhase::TargetCleanup, state)?;
            state.apply_transform(
                "optimization",
                "optimize.target_cleanup",
                |circuit, analysis| {
                    KnowledgeRewriter::new(cleanup_config).transform(circuit, Some(analysis))
                },
            )?;
        }
        Ok(())
    }

    fn lower_output(&self, state: &mut WorkflowState) -> Result<(), CompilerError> {
        state.apply_transform("output", "canonicalize.output", |circuit, analysis| {
            Canonicalizer::production().transform(circuit, Some(analysis))
        })
    }

    fn lower_device_instructions(&self, state: &mut WorkflowState) -> Result<(), CompilerError> {
        let Some(target) = self.device_target() else {
            state.record_skipped(
                "translation",
                "lower.device_instructions",
                "no target device configured",
            );
            return Ok(());
        };
        let lowerer = DeviceLowerer::new(&target.device);
        state.apply_transform(
            "translation",
            "lower.device_instructions",
            |circuit, analysis| lowerer.transform(circuit, Some(analysis)),
        )
    }

    fn canonicalize_native_input(&self, state: &mut WorkflowState) -> Result<(), CompilerError> {
        if self.device_target().is_none() {
            state.record_skipped(
                "optimization",
                "canonicalize.native_input",
                "no target device configured",
            );
            return Ok(());
        }
        state.apply_transform(
            "optimization",
            "canonicalize.native_input",
            |circuit, analysis| Canonicalizer::production().transform(circuit, Some(analysis)),
        )
    }

    fn optimize_native_instructions(&self, state: &mut WorkflowState) -> Result<(), CompilerError> {
        let Some(target) = self.device_target() else {
            state.record_skipped(
                "optimization",
                "optimize.native_fixed_point",
                "no target device configured",
            );
            return Ok(());
        };
        let (max_rounds, max_stale_rounds) = match self.config.mode {
            CompileMode::Normal => (2, 1),
            CompileMode::Enhanced => (8, 3),
        };
        let optimizer = NativeOptimizer::new(
            &target.device,
            self.two_qubit_resynthesis_config_for_state(state),
            max_rounds,
            max_stale_rounds,
        );
        let result = optimizer.run(&state.current)?;
        if result.changed {
            state.analysis = CircuitAnalysis::analyze(&result.circuit);
        }
        state.current = result.circuit;
        state.changed |= result.changed;
        state.steps.push(WorkflowStepReport {
            stage: "optimization",
            name: "optimize.native_fixed_point",
            changed: result.changed,
            skipped: false,
            reason: Some(format!(
                "rounds={}; restored_best={}; native_2q_ops={}->{}; native_2q_depth={}->{}; native_depth={}->{}; native_ops={}->{}; predicted_log_error={:?}->{:?}; unavailable_error_count={}->{}; imputed_error_count={}->{}",
                result.rounds,
                result.restored_best,
                result.before.native_two_qubit_ops,
                result.after.native_two_qubit_ops,
                result.before.native_two_qubit_depth,
                result.after.native_two_qubit_depth,
                result.before.total_native_depth,
                result.after.total_native_depth,
                result.before.native_total_ops,
                result.after.native_total_ops,
                result.before.predicted_log_error,
                result.after.predicted_log_error,
                result.before.unavailable_error_count,
                result.after.unavailable_error_count,
                result.before.imputed_error_count,
                result.after.imputed_error_count,
            )),
        });
        Ok(())
    }

    fn validate_device(&self, state: &mut WorkflowState) -> Result<(), CompilerError> {
        let Some(target) = self.device_target() else {
            state.record_skipped(
                "validation",
                "validate.device",
                "no target device configured",
            );
            return Ok(());
        };
        target.device.validate_circuit(&state.current)?;
        state.steps.push(WorkflowStepReport {
            stage: "validation",
            name: "validate.device",
            changed: false,
            skipped: false,
            reason: None,
        });
        Ok(())
    }

    /// Builds the rewrite configuration for a workflow phase.
    ///
    /// Rewrite phases use the production optimizer. Enhanced mode only
    /// increases bounded search budgets rather than changing correctness
    /// requirements.
    fn rewrite_config(&self, phase: RewritePhase) -> Result<RewriteConfig, CompilerError> {
        let mut config = match phase {
            RewritePhase::PreDecomposition
            | RewritePhase::PostDecomposition
            | RewritePhase::PostRouting
            | RewritePhase::TargetCleanup => RewriteConfig::production(),
        };

        if self.config.mode == CompileMode::Enhanced {
            config = config
                .with_max_rounds(16)
                .with_max_window_ops(32)
                .with_max_pattern_len(12);
        }

        Ok(config)
    }

    fn rewrite_config_for_state(
        &self,
        phase: RewritePhase,
        state: &WorkflowState,
    ) -> Result<RewriteConfig, CompilerError> {
        let config = self.rewrite_config(phase)?;
        if !matches!(phase, RewritePhase::TargetCleanup) {
            return Ok(config);
        }

        let Some(target_basis) = state.target_basis.as_deref() else {
            return Ok(config);
        };

        config.with_target_instructions(target_basis.to_vec())
    }

    fn apply_definition_decomposition(
        &self,
        state: &mut WorkflowState,
    ) -> Result<(), CompilerError> {
        state.apply_transform("init", "decompose.definitions", |circuit, analysis| {
            DecomposeDefinitions.transform(circuit, Some(analysis))
        })
    }

    fn apply_unitary_decomposition(&self, state: &mut WorkflowState) -> Result<(), CompilerError> {
        let config = self.unitary_decompose_config_for_state(state);
        let decomposer = if let Some(target) = self
            .device_target()
            .filter(|_| state.analysis.has_unitary_gates)
        {
            let context = DeviceTwoQubitSynthesisContext::build(
                &target.device,
                &state.current,
                DeviceSynthesisPlacement::PreLayoutEnvelope,
            )?;
            DecomposeUnitaries::new_device_aware(config, context)
        } else {
            DecomposeUnitaries::new(config)
        };
        state.apply_transform("translation", "decompose.unitary", |circuit, analysis| {
            decomposer.transform(circuit, Some(analysis))
        })
    }

    fn unitary_decompose_config_for_state(&self, state: &WorkflowState) -> UnitaryDecomposeConfig {
        UnitaryDecomposeConfig {
            two_qubit_target: state.two_qubit_target.clone(),
            ..Default::default()
        }
    }

    fn apply_mc_gate_decomposition(&self, state: &mut WorkflowState) -> Result<(), CompilerError> {
        let config = self.mc_gate_decompose_config();
        state.apply_transform("translation", "decompose.mc_gates", |circuit, analysis| {
            DecomposeMcGates::new(config).transform(circuit, Some(analysis))
        })
    }

    fn apply_two_qubit_resynthesis(
        &self,
        state: &mut WorkflowState,
        stage: &'static str,
        name: &'static str,
        placement: DeviceSynthesisPlacement,
    ) -> Result<(), CompilerError> {
        let config = self.two_qubit_resynthesis_config_for_state(state);
        let resynthesizer = if let Some(target) = self
            .device_target()
            .filter(|_| ResynthesizeTwoQubitBlocks::is_applicable(&state.current))
        {
            let context =
                DeviceTwoQubitSynthesisContext::build(&target.device, &state.current, placement)?;
            ResynthesizeTwoQubitBlocks::new_device_aware(config, context)
        } else {
            ResynthesizeTwoQubitBlocks::new(config)
        };
        state.apply_transform(stage, name, |circuit, analysis| {
            resynthesizer.transform(circuit, Some(analysis))
        })
    }

    fn apply_post_routing_resynthesis(
        &self,
        state: &mut WorkflowState,
    ) -> Result<(), CompilerError> {
        if self.device_target().is_none() {
            state.record_skipped(
                "optimization",
                "resynthesize.two_qubit_blocks.post_routing",
                "routing was skipped",
            );
            return Ok(());
        }
        self.apply_two_qubit_resynthesis(
            state,
            "optimization",
            "resynthesize.two_qubit_blocks.post_routing",
            DeviceSynthesisPlacement::ExactPhysical,
        )
    }

    fn two_qubit_resynthesis_config_for_state(
        &self,
        state: &WorkflowState,
    ) -> TwoQubitBlockResynthesisConfig {
        match self.config.mode {
            CompileMode::Normal => {
                TwoQubitBlockResynthesisConfig::normal(state.two_qubit_target.clone())
            }
            CompileMode::Enhanced => {
                TwoQubitBlockResynthesisConfig::enhanced(state.two_qubit_target.clone())
            }
        }
    }

    fn apply_routing_basis_decomposition(
        &self,
        state: &mut WorkflowState,
    ) -> Result<(), CompilerError> {
        if self.device_target().is_none() {
            state.record_skipped(
                "translation",
                "decompose.routing_basis",
                "no target device configured",
            );
            return Ok(());
        }

        state.apply_transform(
            "translation",
            "decompose.routing_basis",
            |circuit, analysis| LowerToRoutingBasis::new(None).transform(circuit, Some(analysis)),
        )
    }

    fn mc_gate_decompose_config(&self) -> McGateDecomposeConfig {
        McGateDecomposeConfig {
            resource_policy: self.config.resource_policy,
            resource_limits: self.resource_limits(),
        }
    }

    fn resource_limits(&self) -> ResourceLimits {
        ResourceLimits {
            max_total_qubits: self
                .device_target()
                .map(|target| target.device.num_usable_qubits()),
        }
    }

    /// Performs capacity-style resource preflight before lowering starts.
    ///
    /// Detailed ancillary leasing is still enforced by the decomposition
    /// resource manager when a specific synthesis candidate is selected.
    fn validate_resources(
        &self,
        circuit: &Circuit,
        state: &mut WorkflowState,
    ) -> Result<(), CompilerError> {
        let resource_limits = self.resource_limits();
        if let Some(max_total_qubits) = resource_limits.max_total_qubits
            && circuit.qubits().len() > max_total_qubits
        {
            return Err(CompilerError::InvalidInput(format!(
                "source circuit uses {} logical qubits but target capacity is {max_total_qubits}",
                circuit.qubits().len()
            )));
        }
        state.steps.push(WorkflowStepReport {
            stage: "pre_init",
            name: "validate.resources",
            changed: false,
            skipped: false,
            reason: resource_limits
                .max_total_qubits
                .map(|capacity| format!("target capacity permits {capacity} total logical qubits")),
        });
        Ok(())
    }

    /// Runs layout selection and SABRE routing, or records a skipped routing step.
    ///
    /// A caller-supplied initial layout bypasses layout search but still uses
    /// the same SABRE router and trial settings. Without a supplied layout, the
    /// workflow derives a layout objective from the configured target device.
    fn apply_layout_and_routing(&self, state: &mut WorkflowState) -> Result<(), CompilerError> {
        let Some(target) = self.device_target() else {
            state.record_skipped("routing", "route.sabre", "no target device configured");
            return Ok(());
        };

        let device = &target.device;
        let config = sabre_config_for_mode(self.config.mode, target.seed);
        let (route_changed, swap_count, trials_evaluated, supplied_layout) =
            if let Some(initial_layout) = target.initial_layout.as_ref() {
                let routed = route_with_layout(&state.current, device, initial_layout, &config)?;
                let route_changed = routed.changed(&state.current);
                let swap_count = routed.swap_count();
                let trials_evaluated = routed.diagnostics().trials_evaluated;
                state.device_metadata = Some(DeviceCompilationMetadata {
                    initial_layout: routed.initial_layout().clone(),
                    final_layout: routed.final_layout().clone(),
                });
                state.current = routed.into_circuit();
                (route_changed, swap_count, trials_evaluated, true)
            } else {
                let physical = PhysicalLayoutGraph::from_device(device)?;
                let objective = match self.config.mode {
                    CompileMode::Normal => LayoutObjective::auto_from_physical(&physical),
                    CompileMode::Enhanced => {
                        if physical.has_fidelity_data() {
                            LayoutObjective::fidelity_required(&physical)?
                        } else {
                            LayoutObjective::topology_only()
                        }
                    }
                };
                let routed = route_sabre(&state.current, device, &objective, &config)?;
                let route_changed = routed.changed(&state.current);
                let swap_count = routed.swap_count();
                let trials_evaluated = routed.diagnostics().trials_evaluated;
                state.device_metadata = Some(DeviceCompilationMetadata {
                    initial_layout: routed.initial_layout().clone(),
                    final_layout: routed.final_layout().clone(),
                });
                state.current = routed.into_routed().into_circuit();
                (route_changed, swap_count, trials_evaluated, false)
            };
        state.changed |= route_changed;

        let reason = if supplied_layout {
            format!(
                "inserted {} swap operations using {} routing trials from supplied initial layout",
                swap_count, trials_evaluated
            )
        } else {
            format!(
                "inserted {} swap operations using {} routing trials",
                swap_count, trials_evaluated
            )
        };

        state.steps.push(WorkflowStepReport {
            stage: "routing",
            name: "route.sabre",
            changed: route_changed,
            skipped: false,
            reason: Some(reason),
        });
        Ok(())
    }

    fn apply_post_routing_cleanup(&self, state: &mut WorkflowState) -> Result<(), CompilerError> {
        if self.device_target().is_none() {
            state.record_skipped(
                "optimization",
                "optimize.post_routing",
                "routing was skipped",
            );
            return Ok(());
        }

        let rewrite_config = self.rewrite_config_for_state(RewritePhase::PostRouting, state)?;
        state.apply_transform(
            "optimization",
            "optimize.post_routing",
            |circuit, analysis| {
                KnowledgeRewriter::new(rewrite_config).transform(circuit, Some(analysis))
            },
        )
    }

    fn apply_target_translation(&self, state: &mut WorkflowState) -> Result<(), CompilerError> {
        let Some(target_basis) = state.target_basis.as_deref() else {
            state.record_skipped(
                "translation",
                "translate.target_basis",
                "no target basis configured",
            );
            return Ok(());
        };
        let target_basis = target_basis.to_vec();
        let lowerer = TargetBasisLowerer::new(target_basis)?;

        state.apply_transform(
            "translation",
            "translate.target_basis",
            |circuit, analysis| lowerer.transform(circuit, Some(analysis)),
        )
    }

    /// Resolves an explicit basis target. Device capabilities are local and
    /// ordered, so they are handled by the exact device-lowering stage.
    fn resolve_target_basis(&self) -> Result<Option<Vec<Instruction>>, CompilerError> {
        if let CompileTarget::Basis(target_basis) = &self.config.target {
            validate_workflow_target_basis_config(target_basis)?;
            return Ok(Some(target_basis.to_vec()));
        }
        Ok(None)
    }

    fn record_pre_init(&self, state: &mut WorkflowState) {
        let reason = match (&self.config.target, &state.target_basis) {
            (CompileTarget::Basis(_), Some(basis)) => Some(format!(
                "resolved explicit target basis with {} instructions",
                basis.len()
            )),
            (CompileTarget::Device(_), _) => {
                Some("resolved device target with ordered native capabilities".to_string())
            }
            (CompileTarget::Logical, _) => Some("no target constraints configured".to_string()),
            (CompileTarget::Basis(_), None) => None,
        };

        state.steps.push(WorkflowStepReport {
            stage: "pre_init",
            name: "resolve.target",
            changed: false,
            skipped: false,
            reason,
        });
    }

    fn device_target(&self) -> Option<&DeviceCompileTarget> {
        match &self.config.target {
            CompileTarget::Device(target) => Some(target),
            CompileTarget::Logical | CompileTarget::Basis(_) => None,
        }
    }
}

fn sabre_config_for_mode(mode: CompileMode, seed: Option<u32>) -> SabreConfig {
    let mut config = SabreConfig {
        seed: seed.map(u64::from),
        ..SabreConfig::default()
    };

    if mode == CompileMode::Enhanced {
        config.layout_trials = 24;
        config.layout_assignment_budget = 5_000_000;
        if let Some(vf2) = &mut config.vf2_prepass {
            vf2.call_limit = 5_000_000;
        }
        config.refinement_iterations = 2;
        config.layout_scoring_trials = 3;
        config.routing_trials = 12;
        config.trial_objective = SabreTrialObjective::NativeQualityWithinSwapBudget;
        config.heuristic = SabreHeuristicConfig {
            lookahead_weights: vec![0.5, 0.25],
            ..SabreHeuristicConfig::default()
        };
    }

    config
}

fn validate_workflow_target_basis_config(
    target_basis: &[Instruction],
) -> Result<(), CompilerError> {
    if target_basis.is_empty() {
        return Err(CompilerError::InvalidInput(
            "workflow target basis must not be empty".to_string(),
        ));
    }

    // Rewrite lowering can represent `McGate` as a target instruction, but the
    // current workflow decomposes all multi-controlled gates before target-basis
    // translation. Native multi-controlled target support therefore needs an
    // explicit workflow policy before it can be accepted here.
    for instruction in target_basis {
        if !matches!(instruction, Instruction::Standard(_)) {
            return Err(CompilerError::InvalidInput(format!(
                "unsupported workflow target instruction {instruction:?}"
            )));
        }
    }
    Ok(())
}

#[cfg(test)]
#[path = "./workflow_test.rs"]
mod workflow_test;
