# 设备模块

`cqlib.device` 是 Cqlib 中用于描述量子后端硬件能力、噪声特性与执行结果的数据模块。

在量子计算工作流里，如果说 `Circuit` 表达的是“算法怎么做”，那么 `device` 表达的是“硬件允许怎么做”。它为以下关键工程环节提供统一的底层支撑：

- 物理约束感知：定义硬件拓扑（哪些比特支持双比特门耦合）。
- 高保真度建模：精细化管理比特相干时间（T1/T2）、读出误差及门保真度。
- 动态布局追踪：在编译路由阶段实时维护逻辑比特与物理比特的映射。
- 噪声数字孪生：构建可用于量子噪声仿真的信道模型。
- 任务全生命周期管理：追踪任务从提交、入队、运行到结果回传的完整闭环。

---

## 核心能力

- **拓扑模型**：使用 `Topology` 描述物理比特与耦合关系。
- **设备参数**：使用 `Device`、`QubitProp`、`EdgeProp`、`InstructionProp` 描述标定数据。
- **拓扑映射**：使用 `Layout` 管理逻辑比特到物理比特的映射关系。
- **噪声模型**：使用 `NoiseModel` 及噪声通道对象描述读出误差与门误差。
- **执行结果与状态**：使用 `Outcome`、`Status`、`ExecutionResult` 统一管理任务状态与统计结果。

---

## 快速示例：从拓扑到结果

```python
from cqlib.circuit import Instruction, StandardGate
from cqlib.device import (
    Device,
    EdgeProp,
    ExecutionResult,
    InstructionProp,
    Layout,
    NoiseModel,
    OperationKey,
    QubitProp,
    ReadoutError,
    SingleQubitNoise,
    Topology,
    TwoQubitNoise,
)

# 1) 定义硬件拓扑（支持 (u, v) 或 (u, v, gate_name)）
topo = Topology([0, 1, 2], [(0, 1, "CX"), (1, 2, "CZ")])

# 2) 创建设备并设置默认参数 + 原生门
# 注意：Device 构造器现在需要显式传入 qubits 集合
device = Device("demo_backend", [0, 1, 2], topo)
device.default_t1 = 50.0
device.default_t2 = 35.0
device.default_readout_error = 0.05
device.default_single_qubit_error = 0.001
device.default_two_qubit_error = 0.01
device.native_gates = [
    Instruction.from_standard_gate(StandardGate.X),
    Instruction.from_standard_gate(StandardGate.CX),
]

# 3) 覆盖局部标定参数（逐比特、逐边）
q0_prop = QubitProp(readout_error=0.02)
q0_prop.t1 = 80.0
q0_prop.t2 = 70.0
device.add_qubit_properties(0, q0_prop)

cx_prop = InstructionProp(
    Instruction.from_standard_gate(StandardGate.CX),
    error_rate=0.02
)
cx_prop.length = 220.0
edge_prop = EdgeProp()
edge_prop.native_instructions = [cx_prop]
device.add_edge_properties(0, 1, edge_prop)

# 4) 布局映射（逻辑比特 -> 物理比特）
layout = Layout(logical=[0, 1], physical=[10, 11, 12], init_map={0: 11})
layout.swap_physical(11, 12)

# 5) 噪声模型（可选）
noise = NoiseModel()
noise.add_readout_error(0, ReadoutError(0.02, 0.01))
noise.add_single_qubit_error(StandardGate.X, 0, SingleQubitNoise.bit_flip(0.005))
noise.add_two_qubit_error(StandardGate.CX, 0, 1, TwoQubitNoise.depolarizing(0.02))

# 6) 任务结果管理（示意）
result = ExecutionResult("task-1", [0, 1], 100, 2, "demo_backend")
result.start()
result.finish({"00": 60, "11": 40})
result.calc_probabilities()

print(device.name)                     # demo_backend
print(device.get_t1(0), device.get_t1(2))  # 80.0, 50.0
print(layout.l2p_map)                  # 当前映射
print(noise.get_readout_error(0))      # ReadoutError(...)
print(result.status.kind)              # completed
print(result.probabilities)            # {'11': 0.4, '00': 0.6}

# 按操作键查询噪声（可选）
skey = OperationKey.new_single(StandardGate.X, 0)
print(noise.get_single_qubit_errors(skey))
```

---

## 下一步

- [拓扑建模](1_topology.md)
- [设备属性建模](2_device.md)
- [布局映射](3_layout.md)
- [噪声模型](4_noise.md)
- [执行结果与状态](5_result.md)
