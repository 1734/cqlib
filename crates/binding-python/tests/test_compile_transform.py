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

import copy
import sys
import threading
import time

import pytest

from cqlib.circuit import Circuit, Instruction, StandardGate, UnitaryGate
from cqlib.compile import CompilerConfigError
from cqlib.compile.commutation import CommutationConfig
from cqlib.compile.knowledge import RuleKind
from cqlib.compile.transform import (
    CanonicalizeConfig,
    CanonicalizeResult,
    Canonicalizer,
    KnowledgeRewriteResult,
    KnowledgeRewriteStats,
    KnowledgeRewriter,
    LowerToRoutingBasis,
    RewriteConfig,
    RewriteMode,
    TransformResult,
    canonicalize_circuit,
    lower_to_routing_basis,
    rewrite_circuit,
)
from cqlib.compile.transform.decompose import (
    TwoQubitUnitaryDecomposeBasis,
    UnitaryDecomposeConfig,
    expand_definitions,
)
from cqlib.compile.transform.result import (
    TransformResult as ResultModuleTransformResult,
)
from cqlib.compile.transform.resynthesis import (
    ResynthesizeTwoQubitBlocks,
    TwoQubitBlockResynthesisConfig,
    resynthesize_two_qubit_blocks,
)


def test_transform_module_and_public_types_are_registered() -> None:
    assert "cqlib._native.compile.transform" in sys.modules
    assert CanonicalizeConfig.__module__ == "cqlib.compile.transform"
    assert Canonicalizer.__module__ == "cqlib.compile.transform"
    assert CanonicalizeResult.__module__ == "cqlib.compile.transform"
    assert RewriteMode.__module__ == "cqlib.compile.transform"
    assert RewriteConfig.__module__ == "cqlib.compile.transform"
    assert KnowledgeRewriter.__module__ == "cqlib.compile.transform"
    assert KnowledgeRewriteStats.__module__ == "cqlib.compile.transform"
    assert KnowledgeRewriteResult.__module__ == "cqlib.compile.transform"
    assert LowerToRoutingBasis.__module__ == "cqlib.compile.transform"
    assert TransformResult is ResultModuleTransformResult


def test_canonicalize_config_exposes_immutable_options() -> None:
    config = CanonicalizeConfig(
        round_limit=3,
        recurse_control_flow=False,
        fold_gphase=False,
        canonicalize_instruction_form=False,
        drop_noops=False,
        canonicalize_barriers=False,
    )

    assert config.round_limit == 3
    assert config.recurse_control_flow is False
    assert config.fold_gphase is False
    assert config.canonicalize_instruction_form is False
    assert config.drop_noops is False
    assert config.canonicalize_barriers is False
    assert copy.copy(config) == config
    assert copy.deepcopy(config) == config
    assert repr(config).startswith("CanonicalizeConfig(round_limit=3,")

    with pytest.raises(AttributeError):
        config.round_limit = 4


def test_production_canonicalization_does_not_mutate_input() -> None:
    circuit = Circuit(1)
    circuit.i(0)

    result = canonicalize_circuit(circuit)

    assert len(circuit.operations) == 1
    assert len(result.circuit.operations) == 0
    assert result.changed is True
    assert result.rounds >= 1


def test_configured_canonicalizer_can_preserve_noops() -> None:
    circuit = Circuit(1)
    circuit.i(0)
    config = CanonicalizeConfig(drop_noops=False)
    canonicalizer = Canonicalizer(config)

    result = canonicalizer.run(circuit)

    assert canonicalizer.config == config
    assert len(result.circuit.operations) == 1
    assert result.changed is False


def test_canonicalization_is_idempotent() -> None:
    circuit = Circuit(1)
    circuit.i(0)
    circuit.h(0)

    first = canonicalize_circuit(circuit)
    second = canonicalize_circuit(first.circuit)

    assert first.changed is True
    assert second.changed is False
    assert len(second.circuit.operations) == 1


def test_transform_results_have_value_equality() -> None:
    circuit = Circuit(1)
    circuit.i(0)

    first = canonicalize_circuit(circuit)
    second = canonicalize_circuit(circuit)

    assert first == second
    assert first.__eq__(object()) is NotImplemented
    assert first in [second]

    transform_first = expand_definitions(circuit)
    transform_second = expand_definitions(circuit)
    assert transform_first == transform_second
    assert transform_first.__eq__(object()) is NotImplemented


@pytest.mark.parametrize(
    "run",
    [canonicalize_circuit, lambda circuit: Canonicalizer().run(circuit)],
)
def test_canonicalization_releases_gil(run) -> None:
    circuit = Circuit(1)
    for _ in range(20_000):
        circuit.h(0)

    started = threading.Event()
    finished = threading.Event()
    progressed = threading.Event()

    def worker() -> None:
        started.wait()
        time.sleep(0.01)
        if not finished.is_set():
            progressed.set()

    thread = threading.Thread(target=worker)
    thread.start()
    started.set()
    run(circuit)
    finished.set()
    thread.join()

    assert progressed.is_set()


def test_zero_round_limit_is_rejected_when_run() -> None:
    canonicalizer = Canonicalizer(CanonicalizeConfig(round_limit=0))

    with pytest.raises(
        CompilerConfigError, match="round_limit must be greater than zero"
    ):
        canonicalizer.run(Circuit(1))


def test_rewrite_modes_and_config_expose_immutable_options() -> None:
    optimize = RewriteMode.optimize()
    lowering = RewriteMode.lowering()
    kinds = [RuleKind.cancel(), RuleKind.merge()]
    h = Instruction.from_standard_gate(StandardGate.H)
    config = RewriteConfig(
        max_rounds=3,
        max_window_ops=7,
        max_pattern_len=4,
        recurse_control_flow=False,
        skip_labeled_ops=False,
        enabled_kinds=kinds,
        mode=lowering,
        target_instructions=[h, h],
    )

    assert optimize.name == "optimize"
    assert lowering.name == "lowering"
    assert lowering == RewriteMode.lowering()
    assert hash(lowering) == hash(RewriteMode.lowering())
    assert config.max_rounds == 3
    assert config.max_window_ops == 7
    assert config.max_pattern_len == 4
    assert config.recurse_control_flow is False
    assert config.skip_labeled_ops is False
    assert config.enabled_kinds == kinds
    assert config.mode == lowering
    assert [instruction.name for instruction in config.target_instructions] == ["H"]
    assert copy.copy(config) == config
    assert copy.deepcopy(config) == config
    assert repr(config).startswith("RewriteConfig(max_rounds=3,")

    with pytest.raises(AttributeError):
        config.max_rounds = 4


def test_lowering_mode_selects_lowering_rule_defaults() -> None:
    config = RewriteConfig(mode=RewriteMode.lowering())

    assert config == RewriteConfig.lowering()
    assert RuleKind.decompose() in config.enabled_kinds
    assert RuleKind.hardware_native() in config.enabled_kinds


def test_production_rewrite_does_not_mutate_input_and_reports_stats() -> None:
    circuit = Circuit(1)
    circuit.h(0)
    circuit.h(0)

    result = rewrite_circuit(circuit)

    assert len(circuit.operations) == 2
    assert len(result.circuit.operations) == 0
    assert result.changed is True
    assert result.stats.rules_applied == 1
    assert result.stats.changed_sequences == 1
    assert result.stats.rounds_executed >= 1
    assert result.stats.reached_fixpoint is True
    assert copy.copy(result.stats) == result.stats


def test_rewriter_lowers_to_explicit_target_basis() -> None:
    circuit = Circuit(2)
    circuit.cx(0, 1)
    config = RewriteConfig(
        mode=RewriteMode.lowering(),
        target_instructions=[
            Instruction.from_standard_gate(StandardGate.H),
            Instruction.from_standard_gate(StandardGate.CZ),
        ],
    )
    rewriter = KnowledgeRewriter(config)

    result = rewriter.run(circuit)

    assert rewriter.config == config
    assert [
        operation.instruction.instruction.name
        for operation in result.circuit.operations
    ] == ["H", "CZ", "H"]
    assert result.stats.rules_applied >= 1


def test_rewrite_rejects_invalid_configuration_and_unsatisfied_basis() -> None:
    with pytest.raises(CompilerConfigError, match="must not be empty"):
        RewriteConfig(target_instructions=[])
    with pytest.raises(
        CompilerConfigError, match="unsupported rewrite target instruction"
    ):
        RewriteConfig(target_instructions=[Instruction.delay()])

    zero_round_rewriter = KnowledgeRewriter(RewriteConfig(max_rounds=0))
    with pytest.raises(
        CompilerConfigError, match="max_rounds must be greater than zero"
    ):
        zero_round_rewriter.run(Circuit(1))

    circuit = Circuit(1)
    circuit.h(0)
    config = RewriteConfig(
        mode=RewriteMode.lowering(),
        target_instructions=[Instruction.from_standard_gate(StandardGate.CZ)],
    )
    with pytest.raises(
        CompilerConfigError, match="target instruction basis not satisfied"
    ):
        rewrite_circuit(circuit, config)


def test_lower_to_routing_basis_lowers_toffoli_to_two_qubit_ops() -> None:
    circuit = Circuit(3)
    circuit.ccx(0, 1, 2)

    result = lower_to_routing_basis(circuit)

    assert result.changed is True
    names = [
        operation.instruction.instruction.name
        for operation in result.circuit.operations
    ]
    assert "CCX" not in names
    assert "CX" in names
    assert len(circuit.operations) == 1


def test_lower_to_routing_basis_prefers_cz_only_basis() -> None:
    circuit = Circuit(3)
    circuit.ccx(0, 1, 2)
    basis = [
        Instruction.from_standard_gate(StandardGate.H),
        Instruction.from_standard_gate(StandardGate.T),
        Instruction.from_standard_gate(StandardGate.TDG),
        Instruction.from_standard_gate(StandardGate.CZ),
        Instruction.from_standard_gate(StandardGate.GPhase),
    ]
    transform = LowerToRoutingBasis(preferred_basis=basis)

    result = transform.run(circuit)
    names = [
        operation.instruction.instruction.name
        for operation in result.circuit.operations
    ]

    assert transform.preferred_basis == basis
    assert "CCX" not in names
    assert "CX" not in names
    assert "CZ" in names


def test_lower_to_routing_basis_reports_route_sabre_contract() -> None:
    circuit = Circuit(3)
    circuit.unitary(UnitaryGate("three_q", 3), [0, 1, 2])

    with pytest.raises(
        CompilerConfigError,
        match="routing-basis lowering did not satisfy route.sabre input contract",
    ):
        lower_to_routing_basis(circuit)


def test_two_qubit_block_resynthesis_config_exposes_all_options() -> None:
    commutation = CommutationConfig(
        enable_rule_oracle=False,
        enable_matrix_fallback=True,
        max_matrix_qubits=2,
    )
    config = TwoQubitBlockResynthesisConfig(
        two_qubit_basis=TwoQubitUnitaryDecomposeBasis.cx(),
        enhanced=True,
        max_block_ops=20,
        max_crossed_ops=5,
        max_scan_span=40,
        skip_labeled_ops=False,
        recurse_control_flow=False,
        commutation=commutation,
    )

    assert config.two_qubit_basis == TwoQubitUnitaryDecomposeBasis.cx()
    assert config.max_block_ops == 20
    assert config.max_crossed_ops == 5
    assert config.max_scan_span == 40
    assert config.skip_labeled_ops is False
    assert config.recurse_control_flow is False
    assert config.commutation == commutation
    assert copy.copy(config) == config
    assert copy.deepcopy(config) == config
    assert repr(config).startswith("TwoQubitBlockResynthesisConfig(")


def test_unitary_decompose_config_preserves_legacy_python_basis() -> None:
    config = UnitaryDecomposeConfig(
        two_qubit_basis=TwoQubitUnitaryDecomposeBasis.rzz(),
        recurse_control_flow=False,
    )

    assert config.two_qubit_basis == TwoQubitUnitaryDecomposeBasis.rzz()
    assert config.recurse_control_flow is False
    assert copy.copy(config) == config
    assert copy.deepcopy(config) == config
    assert repr(config).startswith("UnitaryDecomposeConfig(")


def test_two_qubit_block_resynthesis_python_api_preserves_input() -> None:
    circuit = Circuit(2)
    circuit.cx(0, 1)
    circuit.cx(0, 1)
    config = TwoQubitBlockResynthesisConfig(
        two_qubit_basis=TwoQubitUnitaryDecomposeBasis.cx()
    )

    result = resynthesize_two_qubit_blocks(circuit, config)
    transformer = ResynthesizeTwoQubitBlocks(config)
    transformer_result = transformer.run(circuit)

    assert len(circuit.operations) == 2
    assert result.changed is True
    assert len(result.circuit.operations) == 0
    assert transformer.config == config
    assert transformer_result == result
