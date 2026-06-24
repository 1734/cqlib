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

#include <assert.h>
#include <math.h>
#include <stdio.h>

#include "cqlib_c.h"

static void test_basic_circuit(void) {
    CircuitWrapper* circuit = circuit_new(2);
    assert(circuit != NULL);
    assert(circuit_num_qubits(circuit) == 2);
    assert(circuit_num_operations(circuit) == 0);

    assert(circuit_h(circuit, 0) == 0);
    assert(circuit_x(circuit, 1) == 0);
    assert(circuit_y(circuit, 0) == 0);
    assert(circuit_z(circuit, 1) == 0);
    assert(circuit_rx(circuit, 0, 0.25) == 0);
    assert(circuit_ry(circuit, 1, 0.5) == 0);
    assert(circuit_rz(circuit, 0, 0.75) == 0);
    assert(circuit_cx(circuit, 0, 1) == 0);
    assert(circuit_cz(circuit, 1, 0) == 0);
    assert(circuit_measure(circuit, 0) == 0);
    assert(circuit_reset(circuit, 1) == 0);

    assert(circuit_num_operations(circuit) == 11);
    assert(circuit_validate(circuit) == 0);
    circuit_free(circuit);
}

static void test_errors(void) {
    assert(circuit_num_qubits(NULL) == 0);
    assert(circuit_num_operations(NULL) == 0);
    assert(circuit_num_parameters(NULL) == 0);
    assert(circuit_validate(NULL) == -1);
    assert(circuit_h(NULL, 0) == -1);
    circuit_free(NULL);

    CircuitWrapper* circuit = circuit_new(1);
    assert(circuit_h(circuit, 1) == -2);
    assert(circuit_cx(circuit, 0, 1) == -2);
    circuit_free(circuit);
}

static void test_symbolic_parameters(void) {
    ParameterWrapper* theta = param_parse("theta");
    ParameterWrapper* phi = param_parse("phi");
    assert(theta != NULL);
    assert(phi != NULL);

    const char* bindings = "theta:0.5,phi:1.25";
    assert(fabs(param_evaluate(theta, bindings) - 0.5) < 1e-12);
    assert(fabs(param_evaluate(phi, bindings) - 1.25) < 1e-12);

    CircuitWrapper* circuit = circuit_new(2);
    assert(circuit_rx_param(circuit, 0, theta) == 0);
    assert(circuit_ry_param(circuit, 1, phi) == 0);
    assert(circuit_rz_param(circuit, 0, theta) == 0);
    assert(circuit_cx(circuit, 0, 1) == 0);
    assert(circuit_num_operations(circuit) == 4);
    assert(circuit_num_parameters(circuit) == 2);

    CircuitWrapper* assigned = circuit_assign_params(circuit, bindings);
    assert(assigned != NULL);
    assert(circuit_num_operations(assigned) == 4);
    assert(circuit_num_parameters(assigned) == 0);
    assert(circuit_validate(assigned) == 0);

    circuit_free(assigned);
    circuit_free(circuit);
    param_free(theta);
    param_free(phi);
}

int main(void) {
    test_basic_circuit();
    test_errors();
    test_symbolic_parameters();
    printf("binding-c circuit tests passed\n");
    return 0;
}
