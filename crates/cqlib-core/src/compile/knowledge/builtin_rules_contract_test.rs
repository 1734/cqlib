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

//! Contract tests for builtin `.rule` libraries (knowledge module only).
//!
//! Covers parse/validate health, layered [`Rule::verify`], and gate-formula
//! equivalence for selected decompositions. Does not exercise `transform` passes.

use crate::circuit::{Circuit, Qubit};
use crate::compile::knowledge::rule::Rule;
use crate::compile::knowledge::rule_dsl::load::load_rules_from_str;
use crate::compile::knowledge::rule_equivalence::VerifyResult;
use crate::compile::knowledge::{RuleKind, RuleLibrary};
use crate::util::test_utils::assert_circuits_equivalent_up_to_global_phase;
use std::f64::consts::PI;

const RULE_FILES: &[(&str, &str, RuleKind)] = &[
    (
        "cancel.rule",
        include_str!("rules/cancel.rule"),
        RuleKind::Cancel,
    ),
    (
        "merge.rule",
        include_str!("rules/merge.rule"),
        RuleKind::Merge,
    ),
    (
        "commutation.rule",
        include_str!("rules/commutation.rule"),
        RuleKind::Commute,
    ),
    (
        "normalize.rule",
        include_str!("rules/normalize.rule"),
        RuleKind::Canonicalize,
    ),
    (
        "identity.rule",
        include_str!("rules/identity.rule"),
        RuleKind::Simplify,
    ),
    (
        "specialize.rule",
        include_str!("rules/specialize.rule"),
        RuleKind::Simplify,
    ),
    (
        "decompose_ccx.rule",
        include_str!("rules/decompose_ccx.rule"),
        RuleKind::Decompose,
    ),
    (
        "decompose_controlled_pauli.rule",
        include_str!("rules/decompose_controlled_pauli.rule"),
        RuleKind::Decompose,
    ),
    (
        "decompose_controlled_rotation.rule",
        include_str!("rules/decompose_controlled_rotation.rule"),
        RuleKind::Decompose,
    ),
    (
        "decompose_mc_gate.rule",
        include_str!("rules/decompose_mc_gate.rule"),
        RuleKind::Decompose,
    ),
    (
        "decompose_fsim.rule",
        include_str!("rules/decompose_fsim.rule"),
        RuleKind::Decompose,
    ),
    (
        "decompose_ising.rule",
        include_str!("rules/decompose_ising.rule"),
        RuleKind::Decompose,
    ),
    (
        "decompose_phase.rule",
        include_str!("rules/decompose_phase.rule"),
        RuleKind::Decompose,
    ),
    (
        "decompose_qcis.rule",
        include_str!("rules/decompose_qcis.rule"),
        RuleKind::Decompose,
    ),
    (
        "decompose_single_clifford.rule",
        include_str!("rules/decompose_single_clifford.rule"),
        RuleKind::Decompose,
    ),
    (
        "decompose_single_rotation.rule",
        include_str!("rules/decompose_single_rotation.rule"),
        RuleKind::Decompose,
    ),
    (
        "decompose_swap.rule",
        include_str!("rules/decompose_swap.rule"),
        RuleKind::Decompose,
    ),
];

fn classify_rule_verification(rule: &Rule) -> VerifyResult {
    rule.verify().expect("verify setup should succeed")
}

fn assert_rule_verification_passes(rule: &Rule) {
    match classify_rule_verification(rule) {
        VerifyResult::Equivalent | VerifyResult::SampledEqual { .. } => {}
        VerifyResult::NotEquivalent => {
            panic!(
                "rule `{}` failed layered equivalence verification",
                rule.name
            );
        }
        VerifyResult::Inconclusive { reason } => {
            panic!("rule `{}` inconclusive: {reason}", rule.name);
        }
    }
}

fn circuits_equivalent_up_to_global_phase(
    actual: &Circuit,
    expected: &Circuit,
    epsilon: f64,
) -> bool {
    std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        assert_circuits_equivalent_up_to_global_phase(actual, expected, epsilon);
    }))
    .is_ok()
}

#[test]
fn all_rule_files_parse_and_validate() {
    for (file_name, source, _kind) in RULE_FILES {
        let rules = load_rules_from_str(source)
            .unwrap_or_else(|err| panic!("failed to parse {file_name}: {err}"));
        assert!(
            !rules.is_empty(),
            "{file_name} should contain at least one rule"
        );
        for rule in rules {
            rule.validate()
                .unwrap_or_else(|err| panic!("invalid rule `{}` in {file_name}: {err}", rule.name));
        }
    }
}

#[test]
fn rule_file_rule_counts_are_stable() {
    let counts: Vec<(&str, usize)> = RULE_FILES
        .iter()
        .map(|(file_name, source, _)| {
            let count = load_rules_from_str(source).unwrap().len();
            (*file_name, count)
        })
        .collect();
    assert!(counts.iter().all(|(_, count)| *count > 0));
    let total = counts.iter().map(|(_, count)| count).sum::<usize>();
    assert!(
        total > 200,
        "expected a large builtin rule set, got {total} rules: {counts:?}"
    );
}

#[test]
fn selected_semantic_rules_verify_via_matrix() {
    let library = RuleLibrary::builtin_rules().unwrap();
    for name in [
        "cancel_h",
        "cancel_cx",
        "identity_hxh_to_z",
        "decompose_cx_to_cz",
        "decompose_swap_to_ising",
        "decompose_cz_to_rzz",
        "decompose_cx_to_rzz",
    ] {
        let rule = library.get_by_name(name).expect("selected semantic rule");
        assert_rule_verification_passes(rule);
    }
}

#[test]
fn cancel_xy2_inverse_pair_rules_pass_layered_verify() {
    let library = RuleLibrary::builtin_rules().unwrap();
    for name in ["cancel_xy2p_xy2m", "cancel_xy2m_xy2p"] {
        let rule = library.get_by_name(name).expect("XY2 cancel rule");
        assert_rule_verification_passes(rule);
    }
}

#[test]
fn parametric_builtin_rules_pass_layered_verify() {
    let library = RuleLibrary::builtin_rules().unwrap();
    for name in [
        "decompose_crz_to_rzz",
        "decompose_crx_to_rzz",
        "decompose_cry_to_rzz",
        "merge_rx",
        "merge_rzz",
        "decompose_rzz_to_cx",
    ] {
        let rule = library.get_by_name(name).expect("parametric builtin rule");
        let result = classify_rule_verification(rule);
        assert!(
            matches!(
                result,
                VerifyResult::SampledEqual { .. } | VerifyResult::Equivalent
            ),
            "rule `{name}` should pass layered verify, got {result:?}"
        );
    }
}

#[test]
fn decompose_crz_to_rzz_formula_matches_gate_definition() {
    let q0 = Qubit::new(0);
    let q1 = Qubit::new(1);
    let theta = 0.42;

    let mut crz = Circuit::new(2);
    crz.crz(q0, q1, theta).unwrap();

    let mut target_on_q1 = Circuit::new(2);
    target_on_q1.rz(q1, theta / 2.0).unwrap();
    target_on_q1.rzz(q0, q1, -theta / 2.0).unwrap();
    assert_circuits_equivalent_up_to_global_phase(&crz, &target_on_q1, 1e-9);

    let mut wrong_control_rz = Circuit::new(2);
    wrong_control_rz.rz(q0, theta / 2.0).unwrap();
    wrong_control_rz.rzz(q0, q1, -theta / 2.0).unwrap();
    assert!(
        !circuits_equivalent_up_to_global_phase(&crz, &wrong_control_rz, 1e-9),
        "RZ on control qubit should not match CRZ decomposition"
    );
}

#[test]
fn decompose_crx_and_cry_to_rzz_formulas_match_gate_definition() {
    let q0 = Qubit::new(0);
    let q1 = Qubit::new(1);
    let theta = 0.37;

    let mut crx = Circuit::new(2);
    crx.crx(q0, q1, theta).unwrap();
    let mut crx_decomposed = Circuit::new(2);
    crx_decomposed.h(q1).unwrap();
    crx_decomposed.rz(q1, theta / 2.0).unwrap();
    crx_decomposed.rzz(q0, q1, -theta / 2.0).unwrap();
    crx_decomposed.h(q1).unwrap();
    assert_circuits_equivalent_up_to_global_phase(&crx, &crx_decomposed, 1e-9);

    let mut cry = Circuit::new(2);
    cry.cry(q0, q1, theta).unwrap();
    let mut cry_decomposed = Circuit::new(2);
    cry_decomposed.rx(q1, PI / 2.0).unwrap();
    cry_decomposed.rz(q1, theta / 2.0).unwrap();
    cry_decomposed.rzz(q0, q1, -theta / 2.0).unwrap();
    cry_decomposed.rx(q1, -PI / 2.0).unwrap();
    assert_circuits_equivalent_up_to_global_phase(&cry, &cry_decomposed, 1e-9);
}

#[test]
fn decompose_cz_to_rzz_formula_matches_gate_definition() {
    let q0 = Qubit::new(0);
    let q1 = Qubit::new(1);

    let mut cz = Circuit::new(2);
    cz.cz(q0, q1).unwrap();

    let mut decomposed = Circuit::new(2);
    decomposed.rz(q0, PI / 2.0).unwrap();
    decomposed.rz(q1, PI / 2.0).unwrap();
    decomposed.rzz(q0, q1, -PI / 2.0).unwrap();
    assert_circuits_equivalent_up_to_global_phase(&cz, &decomposed, 1e-9);
}

#[test]
fn decompose_cx_to_rzz_formula_matches_gate_definition() {
    let q0 = Qubit::new(0);
    let q1 = Qubit::new(1);

    let mut cx = Circuit::new(2);
    cx.cx(q0, q1).unwrap();

    let mut decomposed = Circuit::new(2);
    decomposed.h(q1).unwrap();
    decomposed.rz(q0, PI / 2.0).unwrap();
    decomposed.rz(q1, PI / 2.0).unwrap();
    decomposed.rzz(q0, q1, -PI / 2.0).unwrap();
    decomposed.h(q1).unwrap();
    assert_circuits_equivalent_up_to_global_phase(&cx, &decomposed, 1e-9);
}

#[test]
fn decompose_cy_to_rzz_formula_matches_gate_definition() {
    let q0 = Qubit::new(0);
    let q1 = Qubit::new(1);

    let mut cy = Circuit::new(2);
    cy.cy(q0, q1).unwrap();

    let mut decomposed = Circuit::new(2);
    decomposed.sdg(q1).unwrap();
    decomposed.h(q1).unwrap();
    decomposed.rz(q0, PI / 2.0).unwrap();
    decomposed.rz(q1, PI / 2.0).unwrap();
    decomposed.rzz(q0, q1, -PI / 2.0).unwrap();
    decomposed.h(q1).unwrap();
    decomposed.s(q1).unwrap();
    assert_circuits_equivalent_up_to_global_phase(&cy, &decomposed, 1e-9);
}

#[test]
fn new_rzz_native_lowering_rules_pass_layered_verify() {
    let library = RuleLibrary::builtin_rules().unwrap();
    let name = "decompose_cy_to_rzz";
    let rule = library.get_by_name(name).expect(name);
    assert_rule_verification_passes(rule);
}

#[test]
fn ion_trap_direct_ising_rules_pass_layered_verify() {
    let library = RuleLibrary::builtin_rules().unwrap();
    for name in [
        "decompose_cz_to_rx_ry_rzz",
        "decompose_cx_to_rx_ry_rzz",
        "decompose_cy_to_rx_ry_rzz",
        "decompose_rxx_to_rx_ry_rzz",
        "decompose_ryy_to_rx_ry_rzz",
        "decompose_rzx_to_rx_ry_rzz",
        "decompose_crz_to_rx_ry_rzz",
        "decompose_crx_to_rx_ry_rzz",
        "decompose_cry_to_rx_ry_rzz",
        "decompose_fsim_to_rx_ry_rzz",
        "decompose_swap_to_rx_ry_rzz",
    ] {
        let rule = library.get_by_name(name).expect(name);
        assert_rule_verification_passes(rule);
    }
}

#[test]
fn ion_trap_direct_ccx_rules_pass_layered_verify() {
    let library = RuleLibrary::builtin_rules().unwrap();
    {
        let name = "decompose_ccx_to_rx_ry_rzz";
        let rule = library.get_by_name(name).expect(name);
        assert_rule_verification_passes(rule);
    }
}

#[test]
fn ion_trap_direct_mc_gate_rules_pass_layered_verify() {
    let library = RuleLibrary::builtin_rules().unwrap();
    for name in [
        "decompose_mch0_to_rx_ry",
        "decompose_mcrxx0_to_rx_ry_rzz",
        "decompose_mcrxy0_to_rx_ry",
        "decompose_mcryy0_to_rx_ry_rzz",
        "decompose_mcrz0_to_rx_ry",
        "decompose_mcrzx0_to_rx_ry_rzz",
        "decompose_mcs0_to_rx_ry",
        "decompose_mcsdg0_to_rx_ry",
        "decompose_mcswap0_to_rx_ry_rzz",
        "decompose_mct0_to_rx_ry",
        "decompose_mctdg0_to_rx_ry",
        "decompose_mcu0_to_rx_ry",
        "decompose_mcx0_to_rx",
        "decompose_mcxy0_to_rx_ry",
        "decompose_mcx2p0_to_rx",
        "decompose_mcx2m0_to_rx",
        "decompose_mcxy2p0_to_rx_ry",
        "decompose_mcxy2m0_to_rx_ry",
        "decompose_mcy0_to_ry",
        "decompose_mcy2p0_to_ry",
        "decompose_mcy2m0_to_ry",
        "decompose_mcz0_to_rx_ry",
        "decompose_mcphase0_to_rx_ry",
        "decompose_mccx0_to_rx_ry_rzz",
        "decompose_mcccx0_to_rx_ry_rzz",
        "decompose_mccy0_to_rx_ry_rzz",
        "decompose_mccz0_to_rx_ry_rzz",
        "decompose_mcfsim0_to_rx_ry_rzz",
        "decompose_mcx1_to_rx_ry_rzz",
        "decompose_mcx2_to_rx_ry_rzz",
        "decompose_mcy1_to_rx_ry_rzz",
        "decompose_mcz1_to_rx_ry_rzz",
        "decompose_mcrx1_to_rx_ry_rzz",
        "decompose_mcry1_to_rx_ry_rzz",
        "decompose_mcrz1_to_rx_ry_rzz",
        "decompose_mcy2_to_rx_ry_rzz",
        "decompose_mcz2_to_rx_ry_rzz",
        "decompose_mcs1_to_rx_ry_rzz",
        "decompose_mcsdg1_to_rx_ry_rzz",
        "decompose_mct1_to_rx_ry_rzz",
        "decompose_mctdg1_to_rx_ry_rzz",
        "decompose_mcphase1_to_rx_ry_rzz",
    ] {
        let rule = library.get_by_name(name).expect(name);
        assert_rule_verification_passes(rule);
    }
}

#[test]
fn ion_trap_direct_single_rotation_rules_pass_layered_verify() {
    let library = RuleLibrary::builtin_rules().unwrap();
    for name in [
        "decompose_h_to_rx_ry",
        "decompose_s_to_rx_ry",
        "decompose_sdg_to_rx_ry",
        "decompose_t_to_rx_ry",
        "decompose_tdg_to_rx_ry",
        "decompose_z_to_rx_ry",
        "decompose_phase_to_rx_ry",
        "decompose_rxy_to_rx_ry",
        "decompose_xy_to_rx_ry",
        "decompose_xy2p_to_rx_ry",
        "decompose_xy2m_to_rx_ry",
        "decompose_u_to_rx_ry",
    ] {
        let rule = library.get_by_name(name).expect(name);
        assert_rule_verification_passes(rule);
    }
}

#[test]
fn new_ising_swapped_merge_rules_pass_layered_verify() {
    let library = RuleLibrary::builtin_rules().unwrap();
    for name in ["merge_rxx_swapped", "merge_ryy_swapped"] {
        let rule = library.get_by_name(name).expect(name);
        assert_rule_verification_passes(rule);
    }
}

#[test]
fn merge_rxx_swapped_matches_direct_sum_unitary() {
    let q0 = Qubit::new(0);
    let q1 = Qubit::new(1);

    let mut chained = Circuit::new(2);
    chained.rxx(q0, q1, 0.3).unwrap();
    chained.rxx(q1, q0, 0.4).unwrap();

    let mut merged = Circuit::new(2);
    merged.rxx(q0, q1, 0.7).unwrap();

    assert_circuits_equivalent_up_to_global_phase(&chained, &merged, 1e-9);
}

#[test]
fn merge_ryy_swapped_matches_direct_sum_unitary() {
    let q0 = Qubit::new(0);
    let q1 = Qubit::new(1);

    let mut chained = Circuit::new(2);
    chained.ryy(q0, q1, 0.3).unwrap();
    chained.ryy(q1, q0, 0.4).unwrap();

    let mut merged = Circuit::new(2);
    merged.ryy(q0, q1, 0.7).unwrap();

    assert_circuits_equivalent_up_to_global_phase(&chained, &merged, 1e-9);
}

#[test]
fn merge_rzz_swapped_rule_passes_layered_verify() {
    let library = RuleLibrary::builtin_rules().unwrap();
    let rule = library
        .get_by_name("merge_rzz_swapped")
        .expect("merge_rzz_swapped");
    assert_rule_verification_passes(rule);
}

#[test]
fn merge_rzz_swapped_matches_direct_sum_unitary() {
    let q0 = Qubit::new(0);
    let q1 = Qubit::new(1);

    let mut chained = Circuit::new(2);
    chained.rzz(q0, q1, 0.3).unwrap();
    chained.rzz(q1, q0, 0.4).unwrap();

    let mut merged = Circuit::new(2);
    merged.rzz(q0, q1, 0.7).unwrap();

    assert_circuits_equivalent_up_to_global_phase(&chained, &merged, 1e-9);
}

#[test]
fn ion_trap_rzz_intermediate_rules_are_present() {
    let library = RuleLibrary::builtin_rules().unwrap();
    for name in [
        "merge_rzz",
        "merge_rzz_swapped",
        "merge_rxx_swapped",
        "merge_ryy_swapped",
        "cancel_rzz_inverse",
        "comm_rzz_rzz",
        "decompose_cry_to_rzz",
        "decompose_crx_to_rzz",
        "decompose_cz_to_rzz",
        "decompose_cx_to_rzz",
        "decompose_cy_to_rzz",
        "decompose_rzz_to_cx",
        "decompose_rzz_to_rxx",
        "decompose_rzz_to_rzx",
        "specialize_rzz_pi_to_cz",
        "decompose_swap_to_ising",
        "decompose_cz_to_rx_ry_rzz",
        "decompose_cx_to_rx_ry_rzz",
        "decompose_cy_to_rx_ry_rzz",
        "decompose_rxx_to_rx_ry_rzz",
        "decompose_ryy_to_rx_ry_rzz",
        "decompose_rzx_to_rx_ry_rzz",
        "decompose_crz_to_rx_ry_rzz",
        "decompose_crx_to_rx_ry_rzz",
        "decompose_cry_to_rx_ry_rzz",
        "decompose_fsim_to_rx_ry_rzz",
        "decompose_swap_to_rx_ry_rzz",
        "decompose_ccx_to_rx_ry_rzz",
    ] {
        assert!(
            library.get_by_name(name).is_some(),
            "expected ion-trap intermediate rule `{name}`"
        );
    }
}

#[test]
fn documented_missing_rules_for_rzz_native_targets() {
    let library = RuleLibrary::builtin_rules().unwrap();
    {
        let missing = "decompose_ms_to_rzz";
        assert!(
            library.get_by_name(missing).is_none(),
            "rule `{missing}` is not implemented yet (documented gap)"
        );
    }
}
