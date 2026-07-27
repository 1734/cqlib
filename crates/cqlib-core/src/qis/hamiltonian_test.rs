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

use super::Hamiltonian;
use crate::qis::{PauliString, Phase, QisError};
use ndarray::{Array2, arr2};
use num_complex::Complex64;

#[test]
fn empty_hamiltonian_to_matrix_is_zero() {
    assert_eq!(
        Hamiltonian::new(2).to_matrix().unwrap(),
        Array2::from_elem((4, 4), Complex64::new(0.0, 0.0))
    );
}

#[test]
fn hamiltonian_to_matrix_sums_coefficients_and_pauli_phases() {
    let mut x: PauliString = "X".parse().unwrap();
    x.phase = Phase::Minus;
    let z: PauliString = "Z".parse().unwrap();
    let hamiltonian = Hamiltonian::from_list(vec![
        (x, Complex64::new(2.0, 0.0)),
        (z, Complex64::new(0.0, 1.0)),
    ])
    .unwrap();

    assert_eq!(
        hamiltonian.to_matrix().unwrap(),
        arr2(&[
            [Complex64::new(0.0, 1.0), Complex64::new(-2.0, 0.0)],
            [Complex64::new(-2.0, 0.0), Complex64::new(0.0, -1.0)],
        ])
    );
}

#[test]
fn zero_qubit_hamiltonian_to_matrix_is_scalar() {
    let mut pauli = PauliString::new(0);
    pauli.phase = Phase::I;
    let hamiltonian = Hamiltonian::from_list(vec![(pauli, Complex64::new(2.0, 0.0))]).unwrap();

    assert_eq!(
        hamiltonian.to_matrix().unwrap(),
        arr2(&[[Complex64::new(0.0, 2.0)]])
    );
}

#[test]
fn hamiltonian_to_matrix_rejects_mismatched_terms() {
    let mut hamiltonian = Hamiltonian::new(1);
    hamiltonian
        .terms
        .push((PauliString::new(2), Complex64::new(1.0, 0.0)));

    assert!(matches!(
        hamiltonian.to_matrix(),
        Err(QisError::QubitMismatch {
            expected: 1,
            actual: 2
        })
    ));
}
