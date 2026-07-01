# This code is part of Cqlib.
#
# (C) Copyright China Telecom Quantum Group 2026
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Indexing tests for the Python circuit API."""

import pytest

from cqlib import Circuit
from cqlib.circuit import CircuitError


def test_operation_access_by_positive_index():
    circuit = Circuit(2)
    circuit.h(0)
    circuit.cx(0, 1)

    assert circuit[0].instruction.instruction.name == "H"
    assert circuit[1].instruction.instruction.name == "CX"
    assert [qubit.index for qubit in circuit[1].qubits] == [0, 1]


def test_operations_property_returns_ordered_operations():
    circuit = Circuit(1)
    circuit.x(0)
    circuit.y(0)
    circuit.z(0)

    operations = circuit.operations
    assert len(operations) == 3
    assert [op.instruction.instruction.name for op in operations] == ["X", "Y", "Z"]


def test_operation_method_and_getitem_report_out_of_range():
    circuit = Circuit(1)
    circuit.h(0)

    with pytest.raises(CircuitError):
        circuit.operation(1)

    with pytest.raises(IndexError):
        circuit[1]

    empty = Circuit(1)
    with pytest.raises(IndexError):
        empty[0]
