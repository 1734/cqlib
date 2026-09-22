# Device modeling

The Device module is responsible for aggregating the full hardware characteristics of a backend. It combines the abstract physical topology (Topology) with concrete calibration parameters, providing data support for noise-aware compilation and high-fidelity simulation.

Cqlib adopts a **"global default plus local override"** calibration strategy: when a parameter is queried, the local value is returned first, and the query falls back to the global default when no local value is configured.

---

## Core objects

| Object | Use |
|---|---|
| InstructionProp | The physical behavior of a specific gate instruction (error rate, execution duration); an Instruction object (created through Instruction.from_standard_gate()) must be passed to the constructor |
| QubitProp | Single-qubit properties (T1/T2, readout error, frequency, native gate list) |
| EdgeProp | Coupling edge properties (native two-qubit instruction set) |
| Device | Top-level entity that integrates topology and properties, providing global defaults and query interfaces |

---

## Building a device baseline

```python
from cqlib.circuit import StandardGate
from cqlib.device import Device, Topology

topo = Topology([0, 1, 2], [(0, 1, "CX"), (1, 2, "CZ")])
device = Device("demo_backend", [0, 1, 2], topo)
device.default_t1 = 50.0
device.default_t2 = 35.0
device.default_readout_error = 0.05
device.default_single_qubit_error = 0.001
device.default_two_qubit_error = 0.01

print("设备名:", device.name)
print("寄存器比特数:", len(device.qubits))
```

**Note**: global defaults are used only for subsequent fallback queries. If no local value is set for a qubit, calling get_t1(q) returns default_t1.

### Factory methods

Device provides factory methods for quickly constructing common topology structures, saving the step of creating a Topology manually:

```python
from cqlib.device import Device

d1 = Device.line("line_dev", num_qubits=5)                    # 单向线型
d2 = Device.bidirectional_line("bi_line", num_qubits=5)      # 双向线型
d3 = Device.ring("ring_dev", num_qubits=4)                    # 双向环形
d4 = Device.star("star_dev", num_qubits=5, center=0)         # 双向星形
d5 = Device.grid("grid_dev", rows=3, cols=4)                  # 双向网格（行主序）
d6 = Device.from_edges("custom", num_qubits=4, edges=[(0, 1), (1, 2)])  # 自定义有向边

print("线型:", d1.num_usable_qubits)
print("双向线型:", d2.num_usable_qubits)
print("环形:", d3.num_usable_qubits)
print("星形:", d4.num_usable_qubits)
print("网格:", d5.num_usable_qubits)
print("自定义:", d6.num_usable_qubits)
```

---

## Injecting local calibration data

```python
from cqlib.circuit import Instruction, StandardGate
from cqlib.device import Device, EdgeProp, InstructionProp, QubitProp, Topology

topo = Topology([0, 1, 2], [(0, 1, "CX"), (1, 2, "CZ")])
device = Device("dev", [0, 1, 2], topo)

# ---- 单比特标定 ----
q0 = QubitProp(readout_error=0.02)
q0.t1 = 80.0              # T1 弛豫时间（微秒）
q0.t2 = 70.0              # T2 退相干时间（微秒）
q0.frequency = 5.1        # 频率（GHz）

# 设置测量判别误差
q0.prob_meas0_prep1 = 0.02  # P(测到 0 | 制备为 1)
q0.prob_meas1_prep0 = 0.01  # P(测到 1 | 制备为 0)

# Add a native single-qubit gate (note: use add_native_instruction(); appending to native_instructions has no effect)
x_prop = InstructionProp(
    Instruction.from_standard_gate(StandardGate.X),
    error_rate=0.001,
)
x_prop.length = 20.0  # 门时长（纳秒）
q0.add_native_instruction(x_prop)

device.add_qubit_properties(0, q0)

# ---- 耦合边标定 ----
cx_prop = InstructionProp(
    Instruction.from_standard_gate(StandardGate.CX),
    error_rate=0.015,
)
cx_prop.length = 200.0

edge = EdgeProp()
edge.add_native_instruction(cx_prop)
device.add_edge_properties(0, 1, edge)

print("比特 0 局部属性已注入")
```

**Note**:
- The first parameter of the InstructionProp constructor must be an Instruction object, created with Instruction.from_standard_gate(StandardGate.X). Passing StandardGate.X directly is not allowed
- QubitProp.native_instructions is a read-only property that returns a new list on every read: it cannot be assigned directly (q0.native_instructions = [...] raises AttributeError), and calling .append() on it has no effect (only a temporary copy is modified); entries must be added through add_native_instruction()
- EdgeProp.native_instructions is also a read-only property, and entries must be added through the add_native_instruction() method

---

## Parameter queries and the fallback mechanism

```python
from cqlib.circuit import Instruction, StandardGate
from cqlib.device import Device, Topology

topo = Topology([0, 1, 2], [(0, 1, "CX"), (1, 2, "CZ")])
device = Device("dev", [0, 1, 2], topo)
device.default_t1 = 50.0
device.default_single_qubit_error = 0.001

# 未设置局部值时回退至默认
print("比特 1 的 T1（回退默认）:", device.get_t1(1))

# 为比特 0 设置局部标定后，局部值优先
q0_prop = QubitProp(readout_error=0.02)
q0_prop.t1 = 80.0
device.add_qubit_properties(0, q0_prop)
print("比特 0 的 T1（局部值）:", device.get_t1(0))

# 查询单比特门误差率（注意：第三个参数需传入 Instruction 对象）
x_inst = Instruction.from_standard_gate(StandardGate.X)
print("比特 0 的 X 门误差（回退默认）:", device.single_qubit_error(0, x_inst))

# 查询读出误差（全局默认）
device.default_readout_error = 0.05
print("比特 0 的读出误差（回退默认）:", device.get_readout_error(0))
```

**Fallback chain**:
- get_t1(q): returns QubitProp(q).t1 first, and returns device.default_t1 if it is not set
- get_readout_error(q): returns QubitProp(q).readout_error first, and returns device.default_readout_error if it is not set
- single_qubit_error(q, inst): looks up in order → ① native gate error ② single-qubit default error ③ device global default
- If a qubit is unavailable (not registered or marked as invalid), all the queries above return None

---

## Invalid qubit management

```python
from cqlib.device import Device, Topology

topo = Topology([0, 1, 2], [(0, 1, "CX"), (1, 2, "CZ")])
device = Device("dev", [0, 1, 2], topo)

# 标记比特 2 为无效（离线/故障）
device.invalid_qubits = [2]   # 注意：使用列表，不是集合

print("可用比特数:", device.num_usable_qubits)  # 2
print("可用比特列表:", device.usable_qubits)     # [Qubit(0), Qubit(1)]
print("比特 2 是否可用:", device.is_usable_qubit(2))  # False
```

---

## Robustness validation

```python
from cqlib.device import Device, QubitProp, Topology

topo = Topology([0, 1, 2], [(0, 1, "CX"), (1, 2, "CZ")])
device = Device("dev", [0, 1, 2], topo)

try:
    device.add_qubit_properties(99, QubitProp(0.01))
except ValueError as e:
    print("添加不存在的比特属性:", e)

try:
    device.add_edge_properties(0, 9, EdgeProp())
except ValueError as e:
    print("添加不存在的边属性:", e)
```

---

## Next steps

- [Layout mapping](3_layout.md): learn the bidirectional logical-physical qubit mapping and SWAP routing operations of Layout
- [Noise model](4_noise.md): understand the use of noise channels such as NoiseModel, SingleQubitNoise, TwoQubitNoise and ReadoutError
- [Execution result and status](5_result.md): become familiar with the complete lifecycle and error handling of Outcome, Status and ExecutionResult
