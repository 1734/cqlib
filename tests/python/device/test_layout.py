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

"""Tests for Layout APIs."""

import pytest

from cqlib import Qubit
from cqlib.device import Layout


class TestLayout:
    """Tests layout mapping and swap behaviors."""

    def test_layout_mapping_and_swap(self):
        """Layout should expose maps and update mapping after swap."""
        layout = Layout(
            logical=[0, 1], physical=[10, 11, 12], init_map={Qubit(0): Qubit(11)}
        )
        assert layout.num_logical == 2
        assert layout.num_physical == 3
        assert layout.num_vacant_physical == 1

        assert set(layout.logical_qubits) == {Qubit(0), Qubit(1)}
        assert set(layout.physical_qubits) == {Qubit(10), Qubit(11), Qubit(12)}
        assert set(layout.l2p_map.keys()).issuperset({Qubit(0), Qubit(1)})
        assert set(layout.p2l_map.keys()).issubset({Qubit(10), Qubit(11), Qubit(12)})

        assert layout.get_physical(0) == Qubit(11)
        v_on_11 = layout.get_logical(11)
        v_on_12 = layout.get_logical(12)

        layout.swap_physical(11, 12)
        assert layout.get_logical(11) == v_on_12
        assert layout.get_logical(12) == v_on_11

    def test_layout_swap_rejects_unknown_physical(self):
        """swap_physical should reject physical qubits outside layout."""
        layout = Layout(logical=[0], physical=[10], init_map=None)
        with pytest.raises(ValueError):
            layout.swap_physical(10, 99)
