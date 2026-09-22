# Layout mapping

Layout manages the bidirectional mapping from logical qubits to physical qubits, and is the core data structure of the compilation and routing stage.

---

## Core concepts

- **Logical qubit (LogicalQubit)**: a virtual qubit defined in an algorithm/circuit
- **Physical qubit (PhysicalQubit)**: the index of a real physical qubit on a hardware chip
- **Vacant physical qubit (Vacant PhysicalQubit)**: a physical position not occupied by any logical qubit, which can be used for subsequent binding

---

## Constructing a mapping

Layout supports several construction forms:

```python
from cqlib.device import Layout

# 方式一：自动顺序映射
# 逻辑比特 [0, 1] 自动映射到物理比特 [10, 11]
# 物理比特 12 保持空闲
layout = Layout(logical=[0, 1], physical=[10, 11, 12])
print("逻辑比特数:", layout.num_logical)       # 2
print("物理比特数:", layout.num_physical)      # 3
print("空闲物理比特数:", layout.num_vacant_physical)  # 1

# 方式二：通过 from_pairs 指定初始映射
# (逻辑, 物理) 对明确指定映射关系，其余物理比特保持空闲
layout2 = Layout.from_pairs([(0, 2), (1, 0)], physical_count=4)
print("from_pairs 空闲数:", layout2.num_vacant_physical)  # 2
```

**Note**:
- The init_map parameter of Layout.__init__ requires the dict[Qubit, Qubit] type; passing {0: 11} directly raises TypeError due to the type mismatch
- If an initial mapping needs to be specified, Layout.from_pairs() is recommended instead
- The lengths of the logical and physical lists should satisfy len(logical) <= len(physical), otherwise ValueError is raised

---

## Querying a mapping

```python
from cqlib.device import Layout

layout = Layout.from_pairs([(0, 11), (1, 10)], physical_count=13)

print("逻辑比特列表:", layout.logical_qubits)
print("物理比特列表:", layout.physical_qubits)
print("空闲物理比特:", layout.vacant_physical_qubits)

# 正向查询：逻辑 → 物理
print("逻辑 0 映射到物理:", layout.get_physical(0))   # Qubit(11)

# 反向查询：物理 → 逻辑
print("物理 11 映射到逻辑:", layout.get_logical(11))  # Qubit(0)

# 查询物理比特是否空闲
print("物理 10 是否空闲:", layout.is_physical_vacant(10))  # False（被逻辑 1 占用）
print("物理 12 是否空闲:", layout.is_physical_vacant(12))  # True

# 获取完整映射字典
print("逻辑→物理映射:", layout.l2p_map)
print("物理→逻辑映射:", layout.p2l_map)
```

**Notes**:
- get_physical(logical_id) returns None for an unbound logical qubit
- get_logical(physical_id) returns None for a vacant physical qubit
- p2l_map contains only the physical qubits already occupied by logical qubits; vacant qubits do not appear in it

---

## Updating a mapping

```python
from cqlib.device import Layout

# 初始映射：逻辑 0→物理 11，逻辑 1→物理 12
layout = Layout.from_pairs([(0, 11), (1, 12)], physical_count=13)
print("初始空闲:", layout.num_vacant_physical)  # 11

# bind：将新的逻辑比特绑定到空闲物理比特
layout.bind(2, 10)
print("绑定后空闲:", layout.num_vacant_physical)  # 10

# unbind：解绑逻辑比特，释放物理比特
released = layout.unbind(0)
print("解绑后释放:", released)                    # Qubit(11)
print("解绑后空闲:", layout.num_vacant_physical)  # 11

# swap_physical：交换两个物理比特上承载的逻辑比特（核心路由操作）
layout3 = Layout.from_pairs([(0, 11), (1, 12)], physical_count=13)
layout3.swap_physical(11, 12)
print("SWAP 后物理 11→逻辑:", layout3.get_logical(11))  # Qubit(1)
print("SWAP 后物理 12→逻辑:", layout3.get_logical(12))  # Qubit(0)
```

**Edge cases**:
- bind(logical, physical): if physical is already occupied, or logical is already bound, ValueError is raised
- unbind(logical): if logical is not bound, ValueError is raised
- swap_physical(a, b): if  or  is not in the layout, ValueError is raised. One of them is allowed to be a vacant qubit (equivalent to moving a logical qubit)

---

## Robustness guarantees

```python
from cqlib.device import Layout

layout = Layout.from_pairs([(0, 11), (1, 12)], physical_count=13)

try:
    layout.swap_physical(11, 99)  # 99 不在布局中
except ValueError as e:
    print("无效的物理比特:", e)

try:
    layout.bind(0, 11)  # 物理 11 已被占用
except ValueError as e:
    print("绑定已被占用的物理比特:", e)

try:
    layout.swap_physical(11, 12)  # 正常操作，不会抛出异常
    print("SWAP 操作成功")
    print("物理 11→逻辑:", layout.get_logical(11))
    print("物理 12→逻辑:", layout.get_logical(12))
except ValueError as e:
    print("SWAP 操作失败:", e)
```

---

## Next steps

- [Noise model](4_noise.md): understand the use of noise channels such as NoiseModel, SingleQubitNoise, TwoQubitNoise and ReadoutError
- [Execution result and status](5_result.md): become familiar with the complete lifecycle and error handling of Outcome, Status and ExecutionResult
