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
import threading
import time

import pytest

import cqlib.compile as compile_module
from cqlib.circuit import Circuit, Instruction, StandardGate
from cqlib.compile import (
    CompileConfig,
    CompileMode,
    CompileResult,
    CompilerConfigError,
    CompilerError,
    CompilerInternalError,
    CompilerTransformError,
    CompilerWorkflow,
    WorkflowStepReport,
    compile,
)
from cqlib.compile.resource import ResourcePolicy
from cqlib.device import Device, Layout


def instruction_names(instructions: list[Instruction] | None) -> list[str] | None:
    if instructions is None:
        return None
    return [instruction.name for instruction in instructions]


def test_workflow_types_are_public_compile_types() -> None:
    assert CompileConfig.__module__ == "cqlib.compile"
    assert CompilerWorkflow.__module__ == "cqlib.compile"
    assert "CompileConfig" in compile_module.__all__
    assert "CompilerWorkflow" in compile_module.__all__
    assert repr(CompileMode.normal()) == "CompileMode.Normal"
    assert repr(CompileMode.enhanced()) == "CompileMode.Enhanced"


def test_compiler_errors_are_public_compile_exceptions() -> None:
    assert CompilerError.__module__ == "cqlib.compile"
    assert CompilerConfigError.__module__ == "cqlib.compile"
    assert CompilerTransformError.__module__ == "cqlib.compile"
    assert CompilerInternalError.__module__ == "cqlib.compile"
    assert issubclass(CompilerConfigError, CompilerError)
    assert issubclass(CompilerTransformError, CompilerError)
    assert issubclass(CompilerInternalError, CompilerError)
    assert not issubclass(CompilerConfigError, ValueError)
    assert {
        "CompilerError",
        "CompilerConfigError",
        "CompilerTransformError",
        "CompilerInternalError",
    } <= set(compile_module.__all__)


def test_compile_config_exposes_immutable_defaults_and_copy_protocol() -> None:
    config = CompileConfig()

    assert config.mode == CompileMode.normal()
    assert config.target_basis is None
    assert config.device is None
    assert config.initial_layout is None
    assert config.resource_policy == ResourcePolicy()
    assert config.seed is None
    assert copy.copy(config) is not config
    assert copy.deepcopy(config) is not config
    assert repr(config).startswith("CompileConfig(mode=CompileMode.Normal,")

    with pytest.raises(AttributeError):
        config.seed = 3


def test_compile_config_takes_target_basis_and_device_snapshots() -> None:
    basis = ["H"]
    device = Device.line("line-2", 2)
    layout = Layout.from_pairs([(0, 0)], physical_count=2)
    policy = ResourcePolicy(max_pre_layout_clean_ancillas=2)
    config = CompileConfig(
        target_basis=basis,
        device=device,
        initial_layout=layout,
        resource_policy=policy,
    )

    basis.append("CZ")
    device.native_gates = [Instruction.from_standard_gate(StandardGate.X)]
    layout.bind(1, 1)

    assert instruction_names(config.target_basis) == ["H"]
    assert config.device is not None
    assert config.device.native_gates == []
    assert config.initial_layout is not None
    assert config.initial_layout.num_logical == 1
    assert config.resource_policy == policy

    returned_device = config.device
    assert returned_device is not None
    returned_device.native_gates = [Instruction.from_standard_gate(StandardGate.Z)]
    assert config.device is not None
    assert config.device.native_gates == []


def test_compiler_workflow_owns_config_snapshot_and_is_reusable() -> None:
    config = CompileConfig(mode=CompileMode.enhanced(), seed=7)
    workflow = CompilerWorkflow(config)
    circuit = Circuit(1)
    circuit.h(0)
    circuit.h(0)

    first = workflow.run(circuit)
    second = workflow.run(circuit)

    assert isinstance(first, CompileResult)
    assert first.mode == CompileMode.enhanced()
    assert first.changed is True
    assert len(first.circuit.operations) == 0
    assert len(second.circuit.operations) == 0
    assert len(circuit.operations) == 2
    assert workflow.config.seed == 7
    assert workflow.config is not workflow.config
    assert all(isinstance(step, WorkflowStepReport) for step in first.steps)
    assert any(step.name == "optimize.target_cleanup" for step in first.steps)


def test_compile_and_explicit_workflow_have_equivalent_results() -> None:
    circuit = Circuit(2)
    circuit.cx(0, 1)
    basis = ["H", "CZ"]

    direct = compile(circuit, target_basis=basis)
    explicit = CompilerWorkflow(CompileConfig(target_basis=basis)).run(circuit)

    direct_names = [
        str(operation.instruction) for operation in direct.circuit.operations
    ]
    explicit_names = [
        str(operation.instruction) for operation in explicit.circuit.operations
    ]
    assert direct_names == explicit_names == ["H", "CZ", "H"]
    assert [step.name for step in direct.steps] == [
        step.name for step in explicit.steps
    ]
    assert direct == explicit
    assert direct.__eq__(object()) is NotImplemented


@pytest.mark.parametrize(
    "run",
    [compile, lambda circuit: CompilerWorkflow().run(circuit)],
)
def test_compiler_entry_points_release_gil(run) -> None:
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


def test_workflow_validates_cross_field_configuration_when_run() -> None:
    config = CompileConfig(
        initial_layout=Layout.from_pairs([(0, 0)], physical_count=1),
    )

    with pytest.raises(
        CompilerConfigError, match="initial layout requires a target device"
    ):
        CompilerWorkflow(config).run(Circuit(1))


def test_compile_config_rejects_unknown_target_gate_name() -> None:
    with pytest.raises(CompilerConfigError, match="unknown standard gate"):
        CompileConfig(target_basis=["not-a-gate"])


def test_workflow_rejects_non_standard_target_instruction_when_run() -> None:
    config = CompileConfig(target_basis=(Instruction.delay(),))

    with pytest.raises(
        CompilerConfigError, match="unsupported workflow target instruction"
    ):
        CompilerWorkflow(config).run(Circuit(1))
