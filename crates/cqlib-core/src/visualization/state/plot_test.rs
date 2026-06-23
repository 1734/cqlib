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

//! Tests for quantum-state visualization.

use super::*;
use crate::qis::{DensityMatrix, Statevector};
use crate::visualization::test_utils::assert_svg_visual_match;
use num_complex::Complex64;

fn zero_state() -> Statevector {
    Statevector::from_state(1, vec![Complex64::new(1.0, 0.0), Complex64::new(0.0, 0.0)]).unwrap()
}

fn plus_state() -> Statevector {
    Statevector::from_state(
        1,
        vec![
            Complex64::new(1.0 / 2.0_f64.sqrt(), 0.0),
            Complex64::new(1.0 / 2.0_f64.sqrt(), 0.0),
        ],
    )
    .unwrap()
}

fn assert_state_visual_match(svg: &str, filename: &str) {
    assert_svg_visual_match(&["state", "figure"], filename, |output_path| {
        render_state_plot_to_file(svg, &output_path.to_string_lossy())
    });
}

#[test]
fn state_city_matches_reference_image() {
    let svg = plot_state_city(&zero_state(), &StatePlotOptions::default()).unwrap();
    assert_state_visual_match(&svg, "state_city_zero.png");
}

#[test]
fn bloch_multivector_matches_reference_image() {
    let svg = plot_bloch_multivector(&plus_state(), &StatePlotOptions::default()).unwrap();
    assert_state_visual_match(&svg, "bloch_multivector_plus.png");
}

#[test]
fn paulivec_matches_reference_image() {
    let svg = plot_state_paulivec(&plus_state(), &StatePlotOptions::default()).unwrap();
    assert_state_visual_match(&svg, "paulivec_plus.png");
}

#[test]
fn density_matrix_uses_same_state_plot_api() {
    let statevector = plus_state();
    let density_matrix =
        DensityMatrix::from_state(statevector.num_qubits, statevector.data().to_vec()).unwrap();

    let sv_svg = plot_state_paulivec(&statevector, &StatePlotOptions::default()).unwrap();
    let dm_svg = plot_state_paulivec(&density_matrix, &StatePlotOptions::default()).unwrap();
    assert_eq!(sv_svg, dm_svg);
}
