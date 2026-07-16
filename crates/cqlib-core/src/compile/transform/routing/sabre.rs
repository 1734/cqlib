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

//! SABRE routing transform.
//!
//! This module adapts the compiler SABRE core into transform-level routing
//! entry points. The algorithm implementation remains in
//! [`crate::compile::sabre`]; this layer owns only the public result types
//! and the high-level `route_sabre` / `route_with_layout` APIs.
//!
//! # Two entry points
//!
//! - [`route_with_layout`] — route with a caller-supplied initial layout.
//!   Returns [`RoutedCircuit`], a pure routing result without layout metadata.
//! - [`route_sabre`] — run SABRE layout refinement then routing in one call.
//!   Returns [`SabreRouteResult`], which wraps a [`RoutedCircuit`] and adds
//!   the layout score.

use crate::circuit::Circuit;
use crate::compile::CompilerError;
use crate::compile::sabre::{
    SabreConfig, SabreRoutingDiagnostics, SabreRoutingResult, sabre_route, sabre_route_prepared,
};
use crate::compile::transform::layout::{
    LayoutDiagnostics, LayoutObjective, LayoutScore, prepare_sabre_circuit,
    prepare_sabre_device_target, sabre_layout_prepared,
};
use crate::device::{Device, Layout, LogicalQubit, PhysicalQubit};

/// A physical circuit produced by routing, plus routing metadata.
///
/// This is the result of routing from a caller-supplied layout. It does not
/// carry layout-selection metadata; callers that also need the layout score
/// should use [`route_sabre`] which returns [`SabreRouteResult`].
#[derive(Debug, Clone, PartialEq)]
pub struct RoutedCircuit {
    circuit: Circuit,
    initial_layout: Layout,
    final_layout: Layout,
    swap_count: usize,
    diagnostics: SabreRoutingDiagnostics,
}

impl RoutedCircuit {
    /// The routed physical circuit.
    pub fn circuit(&self) -> &Circuit {
        &self.circuit
    }

    /// Consumes `self` and returns the owned physical circuit.
    pub fn into_circuit(self) -> Circuit {
        self.circuit
    }

    /// The initial logical-to-physical layout used for routing.
    pub fn initial_layout(&self) -> &Layout {
        &self.initial_layout
    }

    /// The final logical-to-physical layout after all routed operations.
    pub fn final_layout(&self) -> &Layout {
        &self.final_layout
    }

    /// Number of inserted SWAP operations.
    pub fn swap_count(&self) -> usize {
        self.swap_count
    }

    /// Routing diagnostics (trials evaluated, fallback count, etc.).
    pub fn diagnostics(&self) -> &SabreRoutingDiagnostics {
        &self.diagnostics
    }

    /// Whether routing observably changed the original circuit.
    ///
    /// A route is considered changed when any of the following holds:
    /// - SWAP operations were inserted,
    /// - the physical qubit set differs from the input qubit set,
    /// - the global phase changed, or
    /// - a non-identity initial layout was selected.
    pub fn changed(&self, original: &Circuit) -> bool {
        if self.swap_count > 0
            || original.qubits() != self.circuit.qubits()
            || original.global_phase() != self.circuit.global_phase()
        {
            return true;
        }

        if !original.qubits().into_iter().all(|qubit| {
            self.initial_layout
                .get_physical(LogicalQubit::from_qubit(qubit))
                == Some(PhysicalQubit::from_qubit(qubit))
        }) {
            return true;
        }

        format!("{:?}", original.operations()) != format!("{:?}", self.circuit.operations())
    }
}

/// Full SABRE pipeline result: layout selection + routing.
///
/// Returned by [`route_sabre`]. Wraps a [`RoutedCircuit`] and adds the layout
/// score so callers can inspect layout quality.
#[derive(Debug, Clone, PartialEq)]
pub struct SabreRouteResult {
    routed: RoutedCircuit,
    layout_score: Option<LayoutScore>,
    layout_diagnostics: LayoutDiagnostics,
}

impl SabreRouteResult {
    /// The routed physical circuit and routing metadata.
    pub fn routed(&self) -> &RoutedCircuit {
        &self.routed
    }

    /// Consumes `self` and returns the owned [`RoutedCircuit`].
    pub fn into_routed(self) -> RoutedCircuit {
        self.routed
    }

    /// Score of the selected initial layout, when available.
    pub fn layout_score(&self) -> Option<&LayoutScore> {
        self.layout_score.as_ref()
    }

    /// Diagnostics produced while selecting the initial layout.
    pub fn layout_diagnostics(&self) -> &LayoutDiagnostics {
        &self.layout_diagnostics
    }

    // ── Transparent accessors for common routed fields ──

    /// The routed physical circuit.
    pub fn circuit(&self) -> &Circuit {
        self.routed.circuit()
    }

    /// Number of inserted SWAP operations.
    pub fn swap_count(&self) -> usize {
        self.routed.swap_count()
    }

    /// The initial layout used for routing.
    pub fn initial_layout(&self) -> &Layout {
        self.routed.initial_layout()
    }

    /// The final layout after routing.
    pub fn final_layout(&self) -> &Layout {
        self.routed.final_layout()
    }

    /// Routing diagnostics.
    pub fn diagnostics(&self) -> &SabreRoutingDiagnostics {
        self.routed.diagnostics()
    }

    /// Whether routing changed the original circuit.
    pub fn changed(&self, original: &Circuit) -> bool {
        self.routed.changed(original)
    }
}

struct SabreRoutingResultWithScore {
    routing: SabreRoutingResult,
    layout_score: Option<LayoutScore>,
    layout_diagnostics: LayoutDiagnostics,
}

fn sabre_layout_and_route(
    circuit: &Circuit,
    device: &Device,
    objective: &LayoutObjective,
    config: &SabreConfig,
) -> Result<SabreRoutingResultWithScore, CompilerError> {
    let prepared = prepare_sabre_circuit(circuit)?;
    let prepared_target = prepare_sabre_device_target(&prepared, device)?;
    let layout_result = sabre_layout_prepared(&prepared, &prepared_target, objective, config)?;
    let routed = sabre_route_prepared(
        circuit,
        prepared.routing_dag(),
        prepared_target.routing_target(),
        prepared_target.routing_metadata(),
        &layout_result.layout,
        config,
    )?;
    Ok(SabreRoutingResultWithScore {
        routing: routed,
        layout_score: layout_result.score,
        layout_diagnostics: layout_result.diagnostics,
    })
}

/// Routes `circuit` from a caller-supplied initial layout.
///
/// This is the low-level entry point for callers that already have a layout
/// (e.g. from VF2, greedy, or a previous SABRE run). It does not perform
/// layout refinement or scoring.
///
/// Use [`route_sabre`] if you need automatic layout selection.
pub fn route_with_layout(
    circuit: &Circuit,
    device: &Device,
    initial_layout: &Layout,
    config: &SabreConfig,
) -> Result<RoutedCircuit, CompilerError> {
    let result = sabre_route(circuit, device, initial_layout, config)?;
    Ok(RoutedCircuit {
        circuit: result.circuit,
        initial_layout: result.initial_layout,
        final_layout: result.final_layout,
        swap_count: result.swap_count,
        diagnostics: result.diagnostics,
    })
}

/// Selects a SABRE initial layout and routes `circuit` for `device`.
///
/// This function first runs SABRE layout refinement, then routes the original
/// forward circuit from the selected initial layout. The returned circuit is
/// rebuilt over usable physical qubit identifiers and includes inserted
/// [`StandardGate::SWAP`](crate::circuit::StandardGate::SWAP) operations when
/// the selected layout alone cannot satisfy the physical topology.
///
/// If you already have an initial layout, use [`route_with_layout`] to skip
/// the layout-selection step.
///
/// Equal deterministic seeds in [`SabreConfig`] produce equal cqlib routing
/// results for the same circuit and device.
///
/// # Limitations
///
/// This transform checks exact-qargs native-plan feasibility while routing, but
/// it does not emit those native plans or perform target-basis lowering. It also
/// does not select a compiler workflow. Callers should run required
/// decomposition, device lowering, and basis translation passes explicitly.
///
/// # Errors
///
/// Returns [`CompilerError::InvalidInput`] for invalid SABRE configuration,
/// insufficient usable physical qubits, unreachable routing requirements, or
/// unsupported circuit operations such as undecomposed gates that touch more
/// than two qubits. Bounded layout-search and native-feasibility failures are
/// returned as [`CompilerError::SabreRoutingFailed`].
pub fn route_sabre(
    circuit: &Circuit,
    device: &Device,
    objective: &LayoutObjective,
    config: &SabreConfig,
) -> Result<SabreRouteResult, CompilerError> {
    let result = sabre_layout_and_route(circuit, device, objective, config)?;

    Ok(SabreRouteResult {
        routed: RoutedCircuit {
            circuit: result.routing.circuit,
            initial_layout: result.routing.initial_layout,
            final_layout: result.routing.final_layout,
            swap_count: result.routing.swap_count,
            diagnostics: result.routing.diagnostics,
        },
        layout_score: result.layout_score,
        layout_diagnostics: result.layout_diagnostics,
    })
}

#[cfg(test)]
#[path = "./sabre_test.rs"]
mod routing_test;
