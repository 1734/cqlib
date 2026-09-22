# Noise model

cqlib.device provides a complete modeling interface ranging from single-qubit noise and two-qubit noise to the overall noise container, used for channel modeling in quantum noise simulation.

---

## Single-qubit noise

SingleQubitNoise provides several static factory methods to create different types of noise channels:

| Factory method | Channel | Meaning |
|---|---|---|
| bit_flip(p) | Bit flip | Executes an X gate with probability p |
| phase_flip(p) | Phase flip | Executes a Z gate with probability p |
| pauli(px, py, pz) | General Pauli | Independently specifies the X/Y/Z error probabilities |
| depolarizing(p) | Depolarizing | Applies an X/Y/Z error randomly with probability p |
| amplitude_damping(gamma) | Amplitude damping | T1 energy relaxation model |
| phase_damping(lambda_) | Phase damping | T2 pure decoherence model |

```python
from cqlib.device import SingleQubitNoise

sq = SingleQubitNoise.depolarizing(0.01)
pauli = SingleQubitNoise.pauli(px=0.001, py=0.0005, pz=0.002)
bitflip = SingleQubitNoise.bit_flip(0.01)
amp = SingleQubitNoise.amplitude_damping(gamma=0.05)
phase = SingleQubitNoise.phase_damping(0.1)

# Kraus 算符导出
kraus = sq.to_kraus()
print("Kraus 算符数:", len(kraus))          # 4
print("Kraus 矩阵形状:", kraus[0].shape)    # (2, 2)

# 有效性校验
print("去极化噪声有效:", sq.is_valid())       # True
print("振幅阻尼有效:", amp.is_valid())        # True
```

**Note**:
- SingleQubitNoise has no .kind property. To distinguish noise types, the noise type must be tracked through the construction method
- pauli(px, py, pz) requires px + py + pz <= 1, otherwise is_valid() returns False

---

## Two-qubit noise

```python
from cqlib.device import SingleQubitNoise, TwoQubitNoise
from cqlib.qis import Pauli

# 去极化噪声（15 个非单位 Pauli 算符等概率）
tq = TwoQubitNoise.depolarizing(0.01)
print("去极化 Kraus 形状:", tq.to_kraus()[0].shape)  # (4, 4)

# 独立噪声（每个比特各自独立施加单比特噪声）
bind = TwoQubitNoise.independent(
    SingleQubitNoise.phase_flip(0.02),
    SingleQubitNoise.bit_flip(0.03),
)
print("独立噪声 Kraus 形状:", bind.to_kraus()[0].shape)

# 关联 Pauli 噪声
corr = TwoQubitNoise.correlated_pauli(Pauli.x(), Pauli.x(), p=0.01)
print("关联 Pauli Kraus 形状:", corr.to_kraus()[0].shape)
```

**Pauli constructor notes** (all are static methods that return a Pauli object):

| Correct usage | Incorrect usage |
|---|---|
| Pauli.x() | Pauli.X |
| Pauli.y() | Pauli.Y |
| Pauli.z() | Pauli.Z |
| Pauli.i() | Pauli.I |

---

## Readout error

ReadoutError describes the discrimination error in the measurement process:

```python
from cqlib.device import ReadoutError

ro = ReadoutError(p_0_given_1=0.02, p_1_given_0=0.01)
print("P(测到 0 | 制备为 1):", ro.p_0_given_1)  # 0.02
print("P(测到 1 | 制备为 0):", ro.p_1_given_0)  # 0.01
print("读出误差是否有效:", ro.is_valid())          # True
```

---

## NoiseModel: the noise container

NoiseModel aggregates all noise sources, and supports addition, removal and query by qubit/gate type:

```python
from cqlib.circuit import StandardGate
from cqlib.device import NoiseModel, OperationKey, ReadoutError, SingleQubitNoise, TwoQubitNoise

nm = NoiseModel()

# 添加噪声（所有 add_* 方法返回 None，参数校验不通过时抛出 ValueError）
nm.add_readout_error(0, ReadoutError(0.02, 0.01))
nm.add_single_qubit_error(StandardGate.X, 0, SingleQubitNoise.bit_flip(0.005))
nm.add_two_qubit_error(StandardGate.CX, 0, 1, TwoQubitNoise.depolarizing(0.02))

# 读出误差查询
ro = nm.get_readout_error(0)
print("读出误差:", ro)

# 单比特噪声查询（通过 OperationKey）
skey = OperationKey.new_single(StandardGate.X, 0)
errs = nm.get_single_qubit_errors(skey)
print("X 门噪声通道数:", len(errs))

# 双比特噪声查询
tkey = OperationKey.new_double(StandardGate.CX, 0, 1)
errs2 = nm.get_two_qubit_errors(tkey)
print("CX 门噪声通道数:", len(errs2))
```

**Notes**:
- The gate parameter of add_single_qubit_error(gate, qubit, noise) takes a StandardGate enum value (such as StandardGate.X) rather than an Instruction object
- add_two_qubit_error(gate, q0, q1, noise) also uses StandardGate enum values
- OperationKey.new_single(gate, q0) and 
new_double(gate, q0, q1) use StandardGate
- get_single_qubit_errors(key) returns list[SingleQubitNoise] | None, and returns None when there is no match

---

## Parameter validation

NoiseModel.add_*() and the noise constructors raise ValueError when a parameter is invalid:

```python
from cqlib.circuit import StandardGate
from cqlib.device import NoiseModel, SingleQubitNoise, TwoQubitNoise

nm = NoiseModel()

try:
    # 概率超出 [0, 1] 范围
    nm.add_single_qubit_error(StandardGate.X, 0, SingleQubitNoise.bit_flip(1.5))
except ValueError as e:
    print("无效概率被拦截:", e)

try:
    # 双比特门作用在同一比特上
    nm.add_two_qubit_error(StandardGate.CX, 0, 0, TwoQubitNoise.depolarizing(0.01))
except ValueError as e:
    print("无效配置被拦截:", e)
```

---

## Combining noise with a device

In actual use, Device provides gate error rate query interfaces, while NoiseModel provides detailed channel models. The two can be used together:

```python
from cqlib.circuit import Instruction, StandardGate
from cqlib.device import Device, NoiseModel, OperationKey, SingleQubitNoise, Topology

topo = Topology([0, 1, 2], [(0, 1, "CX"), (1, 2, "CZ")])
device = Device("noisy", [0, 1, 2], topo)
device.default_single_qubit_error = 0.001

noise = NoiseModel()
noise.add_single_qubit_error(StandardGate.H, 0, SingleQubitNoise.depolarizing(0.002))

# Device 查询（注意需要 Instruction 对象）
h_inst = Instruction.from_standard_gate(StandardGate.H)
print("设备报告的 H 门误差:", device.single_qubit_error(0, h_inst))

# NoiseModel 查询噪声通道
skey = OperationKey.new_single(StandardGate.H, 0)
channels = noise.get_single_qubit_errors(skey)
print("噪声通道数:", len(channels))
```

---

## Next steps

- [Execution result and status](5_result.md): become familiar with the complete lifecycle and error handling of Outcome, Status and ExecutionResult
- [Quantum Information](../3_qis/0_overview.md): master the basics such as Statevector, DensityMatrix and Pauli
- [Compilation](../4_compiler/0_overview.md): learn about the layout, routing and optimization of the compilation pipeline
