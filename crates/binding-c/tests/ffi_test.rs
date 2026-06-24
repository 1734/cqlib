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

use binding_c::circuit::{
    circuit_assign_params, circuit_cx, circuit_cz, circuit_free, circuit_h, circuit_measure,
    circuit_new, circuit_num_operations, circuit_num_parameters, circuit_num_qubits, circuit_reset,
    circuit_rx, circuit_rx_param, circuit_ry, circuit_ry_param, circuit_rz, circuit_rz_param,
    circuit_validate, circuit_x, circuit_y, circuit_z, param_evaluate, param_free, param_parse,
};
use std::ffi::CString;

#[test]
fn circuit_lifecycle_and_basic_gates() {
    let circuit = circuit_new(2);
    assert!(!circuit.is_null());
    assert_eq!(circuit_num_qubits(circuit), 2);
    assert_eq!(circuit_num_operations(circuit), 0);

    assert_eq!(circuit_h(circuit, 0), 0);
    assert_eq!(circuit_x(circuit, 1), 0);
    assert_eq!(circuit_y(circuit, 0), 0);
    assert_eq!(circuit_z(circuit, 1), 0);
    assert_eq!(circuit_rx(circuit, 0, 0.25), 0);
    assert_eq!(circuit_ry(circuit, 1, 0.5), 0);
    assert_eq!(circuit_rz(circuit, 0, 0.75), 0);
    assert_eq!(circuit_cx(circuit, 0, 1), 0);
    assert_eq!(circuit_cz(circuit, 1, 0), 0);
    assert_eq!(circuit_measure(circuit, 0), 0);
    assert_eq!(circuit_reset(circuit, 1), 0);

    assert_eq!(circuit_num_operations(circuit), 11);
    assert_eq!(circuit_validate(circuit), 0);
    circuit_free(circuit);
}

#[test]
fn circuit_error_codes_are_stable() {
    assert_eq!(circuit_num_qubits(std::ptr::null()), 0);
    assert_eq!(circuit_num_operations(std::ptr::null()), 0);
    assert_eq!(circuit_num_parameters(std::ptr::null()), 0);
    assert_eq!(circuit_validate(std::ptr::null()), -1);
    assert_eq!(circuit_h(std::ptr::null_mut(), 0), -1);
    circuit_free(std::ptr::null_mut());

    let circuit = circuit_new(1);
    assert_eq!(circuit_h(circuit, 1), -2);
    assert_eq!(circuit_cx(circuit, 0, 1), -2);
    assert_eq!(circuit_rx(circuit, 0, f64::NAN), -3);
    circuit_free(circuit);
}

#[test]
fn symbolic_parameters_can_be_evaluated_and_assigned() {
    let theta = CString::new("theta").unwrap();
    let phi = CString::new("phi").unwrap();
    let theta_ptr = param_parse(theta.as_ptr());
    let phi_ptr = param_parse(phi.as_ptr());
    assert!(!theta_ptr.is_null());
    assert!(!phi_ptr.is_null());

    let bindings = CString::new("theta:0.5,phi:1.25").unwrap();
    assert!((param_evaluate(theta_ptr, bindings.as_ptr()) - 0.5).abs() < 1e-12);
    assert!((param_evaluate(phi_ptr, bindings.as_ptr()) - 1.25).abs() < 1e-12);

    let circuit = circuit_new(2);
    assert_eq!(circuit_rx_param(circuit, 0, theta_ptr), 0);
    assert_eq!(circuit_ry_param(circuit, 1, phi_ptr), 0);
    assert_eq!(circuit_rz_param(circuit, 0, theta_ptr), 0);
    assert_eq!(circuit_cx(circuit, 0, 1), 0);
    assert_eq!(circuit_num_operations(circuit), 4);
    assert_eq!(circuit_num_parameters(circuit), 2);

    let assigned = circuit_assign_params(circuit, bindings.as_ptr());
    assert!(!assigned.is_null());
    assert_eq!(circuit_num_operations(assigned), 4);
    assert_eq!(circuit_num_parameters(assigned), 0);
    assert_eq!(circuit_validate(assigned), 0);

    circuit_free(assigned);
    circuit_free(circuit);
    param_free(theta_ptr);
    param_free(phi_ptr);
}

#[test]
fn invalid_parameters_return_null_or_error() {
    assert!(param_parse(std::ptr::null()).is_null());
    assert_eq!(param_evaluate(std::ptr::null(), std::ptr::null()), 0.0);

    let circuit = circuit_new(1);
    assert_eq!(circuit_rx_param(circuit, 0, std::ptr::null()), -1);
    assert!(circuit_assign_params(std::ptr::null(), std::ptr::null()).is_null());
    circuit_free(circuit);
    param_free(std::ptr::null_mut());
}
