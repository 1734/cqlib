# 设备属性

`Device` 模块负责汇总后端的全量硬件特征。它将抽象的物理拓扑（`Topology`）与具体的标定参数（如相干时间、门保真度等）相结合，为噪声感知编译和高保真度仿真提供数据支撑。

Cqlib 采用 “全局默认 + 局部覆盖” 的策略，允许开发者在定义大规模设备基准的同时，对性能特殊的比特或边进行精细化刻画。

---

## 核心对象

- `InstructionProp`：描述特定指令的物理表现，核心参数包括平均误差率与执行时长。
- `QubitProp`：描述单比特特性。包含读出误差、相干时间（T1/T2）、共振频率以及该比特支持的原生单比特指令集。
- `EdgeProp`：描述耦合边特性。主要用于定义该耦合路径支持的原生双比特指令及关联的标定参数。
- `Device`：顶层实体。整合拓扑结构与上述所有属性，对外提供统一的参数查询接口。

---

## 构建设备基准

在初始化设备时，建议您先设定全局默认值。这可以作为该芯片各参数的期望水平：

```python
from cqlib.circuit import Instruction, StandardGate
from cqlib.device import Device, Topology

# 定义基础拓扑
topo = Topology([0, 1, 2], [(0, 1, "CX"), (1, 2, "CZ")])

# 创建设备并注入“全局默认值”
# 注意：Device 构造器现在需要显式传入 qubits 集合
device = Device("demo_backend", [0, 1, 2], topo)
device.default_t1 = 50.0                 # 默认 T1 50μs
device.default_t2 = 35.0                 # 默认 T2 35μs
device.default_readout_error = 0.05      # 默认读出误差 5%
device.default_single_qubit_error = 0.001 # 默认单比特门误差 0.1%
device.default_two_qubit_error = 0.01    # 默认双比特门误差 1%
device.native_gates = [
    Instruction.from_standard_gate(StandardGate.X),
    Instruction.from_standard_gate(StandardGate.CX),
]
```

### 使用快速生成器

对于标准拓扑，您可以直接使用设备生成器：

```python
# 线型设备（单向）
device = Device.line("line_device", num_qubits=5)

# 双向线型设备
device = Device.bidirectional_line("bi_line", num_qubits=5)

# 环形设备
device = Device.ring("ring_device", num_qubits=4)

# 星形设备
device = Device.star("star_device", num_qubits=5, center=0)

# 网格设备
device = Device.grid("grid_device", rows=3, cols=4)

# 从边列表创建设备
device = Device.from_edges("custom", num_qubits=4, edges=[(0, 1), (1, 2), (2, 3)])
```

---

## 注入局部标定数据

在实际芯片中，每个比特的表现往往参差不齐。通过 `add_qubit_properties` 和 `add_edge_properties`，您可以覆盖特定的局部参数。

```python
from cqlib.device import QubitProp, EdgeProp, InstructionProp

# 刻画 Q0 的优越性能
q0_prop = QubitProp(readout_error=0.02)
q0_prop.t1 = 80.0
q0_prop.t2 = 70.0
q0_prop.frequency = 5.1  # 频率 5.1 GHz

x_prop = InstructionProp(
    Instruction.from_standard_gate(StandardGate.X),
    error_rate=0.001
)
x_prop.length = 20.0  # X 门脉冲长度 20ns
q0_prop.native_instructions = [x_prop]

device.add_qubit_properties(0, q0_prop)

# 刻画 (0, 1) 耦合边的双比特门表现
cx_prop = InstructionProp(
    Instruction.from_standard_gate(StandardGate.CX),
    error_rate=0.015
)
cx_prop.length = 200.0
edge_prop = EdgeProp()
edge_prop.native_instructions = [cx_prop]
device.add_edge_properties(0, 1, edge_prop)
```

---

## 参数查询与回退机制

`Device` 对象实现了智能查询逻辑。当您请求某个属性时，它会遵循以下优先级：

- 局部值：检查目标比特/边是否拥有独立的 `QubitProp`/ `EdgeProp`。
- 全局值：若局部未配置，则自动回退到初始化时设定的 `default` 值。

```python
# 参数查询示例
print(f"Q0 T1: {device.get_t1(0)}")  # 输出 80.0 (局部覆盖生效)
print(f"Q2 T1: {device.get_t1(2)}")  # 输出 50.0 (回退至全局默认值)

# 错误率查询
print(f"Q0 X 门误差: {device.single_qubit_error(0, Instruction.from_standard_gate(StandardGate.X))}")
print(f"(0,1) CX 误差: {device.two_qubit_error(0, 1, Instruction.from_standard_gate(StandardGate.CX))}")
print(f"(0,1) 最佳边误差: {device.edge_error(0, 1)}")
```

---

## 无效比特管理

对于暂时离线或故障的比特，可以将其标记为无效：

```python
# 标记比特 2 为无效（离线或故障）
device.invalid_qubits = {2}

# 查询可用比特
print(f"可用比特数: {device.num_usable_qubits}")
print(f"比特 2 是否可用: {device.is_usable_qubit(2)}")  # False
```

---

## 健壮性校验

为了保证硬件模型的一致性，`Device` 会实时校验输入的比特索引是否超出了 `Topology` 定义的范围：

```python
from cqlib.device import EdgeProp, QubitProp

# 尝试为不存在的比特 99 添加属性将触发异常
try:
    device.add_qubit_properties(99, QubitProp(0.01))
except ValueError as e:
    print(f"操作拦截: {e}") # 提示比特不在拓扑内
```

---

## 下一步

接下来您可以深入了解以下主题：

- [布局映射](3_layout.md)
- [噪声模型](4_noise.md)
- [执行结果与状态](5_result.md)
