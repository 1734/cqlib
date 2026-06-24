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

//! Minimal C ABI for circuit construction and symbolic parameters.

#![allow(clippy::not_unsafe_ptr_arg_deref)]

use cqlib_core::circuit::{Circuit, Parameter, ParameterValue, Qubit};
use std::collections::HashMap;
use std::ffi::CStr;
use std::os::raw::c_char;

pub struct CircuitWrapper {
    pub inner: Circuit,
}

pub struct ParameterWrapper {
    pub inner: Parameter,
}

fn parse_bindings(bindings: *const c_char) -> Option<HashMap<String, f64>> {
    if bindings.is_null() {
        return None;
    }

    let c_str = unsafe { CStr::from_ptr(bindings) };
    let text = c_str.to_str().ok()?;
    let mut map = HashMap::new();
    for pair in text.split(',') {
        let pair = pair.trim();
        if pair.is_empty() {
            continue;
        }
        let (name, value) = pair.split_once(':')?;
        let value = value.trim().parse::<f64>().ok()?;
        if !value.is_finite() {
            return None;
        }
        map.insert(name.trim().to_string(), value);
    }
    Some(map)
}

fn binding_refs(bindings: &HashMap<String, f64>) -> HashMap<&str, f64> {
    bindings
        .iter()
        .map(|(name, value)| (name.as_str(), *value))
        .collect()
}

fn check_qubit(circuit: &Circuit, qubit: u32) -> Result<Qubit, i32> {
    if qubit as usize >= circuit.num_qubits() {
        return Err(-2);
    }
    Ok(Qubit::new(qubit))
}

fn apply_single(
    ptr: *mut CircuitWrapper,
    qubit: u32,
    apply: impl FnOnce(&mut Circuit, Qubit) -> Result<(), cqlib_core::circuit::CircuitError>,
) -> i32 {
    if ptr.is_null() {
        return -1;
    }
    let wrapper = unsafe { &mut *ptr };
    let qubit = match check_qubit(&wrapper.inner, qubit) {
        Ok(qubit) => qubit,
        Err(code) => return code,
    };
    apply(&mut wrapper.inner, qubit).map_or(-3, |_| 0)
}

fn apply_two(
    ptr: *mut CircuitWrapper,
    first: u32,
    second: u32,
    apply: impl FnOnce(&mut Circuit, Qubit, Qubit) -> Result<(), cqlib_core::circuit::CircuitError>,
) -> i32 {
    if ptr.is_null() {
        return -1;
    }
    let wrapper = unsafe { &mut *ptr };
    let first = match check_qubit(&wrapper.inner, first) {
        Ok(qubit) => qubit,
        Err(code) => return code,
    };
    let second = match check_qubit(&wrapper.inner, second) {
        Ok(qubit) => qubit,
        Err(code) => return code,
    };
    apply(&mut wrapper.inner, first, second).map_or(-3, |_| 0)
}

fn apply_param(
    ptr: *mut CircuitWrapper,
    qubit: u32,
    param_ptr: *const ParameterWrapper,
    apply: impl FnOnce(
        &mut Circuit,
        Qubit,
        ParameterValue,
    ) -> Result<(), cqlib_core::circuit::CircuitError>,
) -> i32 {
    if ptr.is_null() || param_ptr.is_null() {
        return -1;
    }
    let wrapper = unsafe { &mut *ptr };
    let qubit = match check_qubit(&wrapper.inner, qubit) {
        Ok(qubit) => qubit,
        Err(code) => return code,
    };
    let param = unsafe { &(*param_ptr).inner };
    apply(
        &mut wrapper.inner,
        qubit,
        ParameterValue::Param(param.clone()),
    )
    .map_or(-3, |_| 0)
}

/// Create a new quantum circuit with `num_qubits` logical qubits.
#[unsafe(no_mangle)]
pub extern "C" fn circuit_new(num_qubits: usize) -> *mut CircuitWrapper {
    Box::into_raw(Box::new(CircuitWrapper {
        inner: Circuit::new(num_qubits),
    }))
}

/// Free a quantum circuit. Passing NULL is allowed.
#[unsafe(no_mangle)]
pub extern "C" fn circuit_free(ptr: *mut CircuitWrapper) {
    if !ptr.is_null() {
        unsafe {
            let _ = Box::from_raw(ptr);
        }
    }
}

/// Return the number of qubits in the circuit, or 0 for NULL.
#[unsafe(no_mangle)]
pub extern "C" fn circuit_num_qubits(ptr: *const CircuitWrapper) -> usize {
    if ptr.is_null() {
        return 0;
    }
    unsafe { (*ptr).inner.num_qubits() }
}

/// Return the number of operations in the circuit, or 0 for NULL.
#[unsafe(no_mangle)]
pub extern "C" fn circuit_num_operations(ptr: *const CircuitWrapper) -> usize {
    if ptr.is_null() {
        return 0;
    }
    unsafe { (*ptr).inner.operations().len() }
}

/// Return the number of interned symbolic parameters in the circuit, or 0 for NULL.
#[unsafe(no_mangle)]
pub extern "C" fn circuit_num_parameters(ptr: *const CircuitWrapper) -> usize {
    if ptr.is_null() {
        return 0;
    }
    unsafe { (*ptr).inner.parameters().len() }
}

/// Validate circuit consistency.
#[unsafe(no_mangle)]
pub extern "C" fn circuit_validate(ptr: *const CircuitWrapper) -> i32 {
    if ptr.is_null() {
        return -1;
    }
    unsafe { (*ptr).inner.validate().map_or(-3, |_| 0) }
}

#[unsafe(no_mangle)]
pub extern "C" fn circuit_h(ptr: *mut CircuitWrapper, qubit: u32) -> i32 {
    apply_single(ptr, qubit, Circuit::h)
}

#[unsafe(no_mangle)]
pub extern "C" fn circuit_x(ptr: *mut CircuitWrapper, qubit: u32) -> i32 {
    apply_single(ptr, qubit, Circuit::x)
}

#[unsafe(no_mangle)]
pub extern "C" fn circuit_y(ptr: *mut CircuitWrapper, qubit: u32) -> i32 {
    apply_single(ptr, qubit, Circuit::y)
}

#[unsafe(no_mangle)]
pub extern "C" fn circuit_z(ptr: *mut CircuitWrapper, qubit: u32) -> i32 {
    apply_single(ptr, qubit, Circuit::z)
}

#[unsafe(no_mangle)]
pub extern "C" fn circuit_rx(ptr: *mut CircuitWrapper, qubit: u32, theta: f64) -> i32 {
    if !theta.is_finite() {
        return -3;
    }
    apply_single(ptr, qubit, |circuit, qubit| circuit.rx(qubit, theta))
}

#[unsafe(no_mangle)]
pub extern "C" fn circuit_ry(ptr: *mut CircuitWrapper, qubit: u32, theta: f64) -> i32 {
    if !theta.is_finite() {
        return -3;
    }
    apply_single(ptr, qubit, |circuit, qubit| circuit.ry(qubit, theta))
}

#[unsafe(no_mangle)]
pub extern "C" fn circuit_rz(ptr: *mut CircuitWrapper, qubit: u32, theta: f64) -> i32 {
    if !theta.is_finite() {
        return -3;
    }
    apply_single(ptr, qubit, |circuit, qubit| circuit.rz(qubit, theta))
}

#[unsafe(no_mangle)]
pub extern "C" fn circuit_cx(ptr: *mut CircuitWrapper, control: u32, target: u32) -> i32 {
    apply_two(ptr, control, target, Circuit::cx)
}

#[unsafe(no_mangle)]
pub extern "C" fn circuit_cz(ptr: *mut CircuitWrapper, control: u32, target: u32) -> i32 {
    apply_two(ptr, control, target, Circuit::cz)
}

#[unsafe(no_mangle)]
pub extern "C" fn circuit_measure(ptr: *mut CircuitWrapper, qubit: u32) -> i32 {
    apply_single(ptr, qubit, |circuit, qubit| {
        circuit.measure(qubit).map(|_| ())
    })
}

#[unsafe(no_mangle)]
pub extern "C" fn circuit_reset(ptr: *mut CircuitWrapper, qubit: u32) -> i32 {
    apply_single(ptr, qubit, Circuit::reset)
}

/// Parse a symbolic parameter expression.
#[unsafe(no_mangle)]
pub extern "C" fn param_parse(expr: *const c_char) -> *mut ParameterWrapper {
    if expr.is_null() {
        return std::ptr::null_mut();
    }

    let c_str = unsafe { CStr::from_ptr(expr) };
    let expr = match c_str.to_str() {
        Ok(expr) => expr,
        Err(_) => return std::ptr::null_mut(),
    };

    match Parameter::try_from(expr) {
        Ok(param) => Box::into_raw(Box::new(ParameterWrapper { inner: param })),
        Err(_) => std::ptr::null_mut(),
    }
}

/// Free a parameter. Passing NULL is allowed.
#[unsafe(no_mangle)]
pub extern "C" fn param_free(ptr: *mut ParameterWrapper) {
    if !ptr.is_null() {
        unsafe {
            let _ = Box::from_raw(ptr);
        }
    }
}

/// Evaluate a parameter expression with bindings formatted as "name:value,name2:value2".
#[unsafe(no_mangle)]
pub extern "C" fn param_evaluate(ptr: *const ParameterWrapper, bindings: *const c_char) -> f64 {
    if ptr.is_null() {
        return 0.0;
    }
    let bindings = parse_bindings(bindings);
    let refs = bindings.as_ref().map(binding_refs);
    unsafe { (*ptr).inner.evaluate(&refs).unwrap_or(0.0) }
}

#[unsafe(no_mangle)]
pub extern "C" fn circuit_rx_param(
    ptr: *mut CircuitWrapper,
    qubit: u32,
    param_ptr: *const ParameterWrapper,
) -> i32 {
    apply_param(ptr, qubit, param_ptr, Circuit::rx)
}

#[unsafe(no_mangle)]
pub extern "C" fn circuit_ry_param(
    ptr: *mut CircuitWrapper,
    qubit: u32,
    param_ptr: *const ParameterWrapper,
) -> i32 {
    apply_param(ptr, qubit, param_ptr, Circuit::ry)
}

#[unsafe(no_mangle)]
pub extern "C" fn circuit_rz_param(
    ptr: *mut CircuitWrapper,
    qubit: u32,
    param_ptr: *const ParameterWrapper,
) -> i32 {
    apply_param(ptr, qubit, param_ptr, Circuit::rz)
}

/// Return a new circuit with matching symbolic parameters assigned.
#[unsafe(no_mangle)]
pub extern "C" fn circuit_assign_params(
    circuit: *const CircuitWrapper,
    bindings: *const c_char,
) -> *mut CircuitWrapper {
    if circuit.is_null() {
        return std::ptr::null_mut();
    }

    let bindings = parse_bindings(bindings);
    let refs = bindings.as_ref().map(binding_refs);
    match unsafe { (*circuit).inner.assign_parameters(&refs) } {
        Ok(inner) => Box::into_raw(Box::new(CircuitWrapper { inner })),
        Err(_) => std::ptr::null_mut(),
    }
}
