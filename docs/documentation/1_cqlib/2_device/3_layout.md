# 拓扑映射

`Layout` 类提供了一个高性能的、双向一致的映射管理工具，帮助开发者在编译与路由过程中精准追踪比特位置的变化。

---

## 核心概念

- **逻辑比特**：算法代码中定义的比特（`LogicalQubit`）。
- **物理比特**：硬件芯片上真实的比特索引（`PhysicalQubit`）。
- **空闲物理比特**：物理比特中未被逻辑比特占用的位置，可用于路由或绑定额外的逻辑比特。

---

## 构造映射

您可以手动定义初始的映射关系，未指定的逻辑比特将被自动分配到剩余的可用物理位置：

```python
from cqlib.device import Layout

# 场景：将 2 个逻辑比特映射到 3 比特芯片的特定位置
layout = Layout(
    logical=[0, 1],
    physical=[10, 11, 12],
    init_map={0: 11}   # 指定逻辑比特 0 对应物理比特 11
)

print(f"逻辑比特数: {layout.num_logical}")   # 2
print(f"物理比特数: {layout.num_physical}")  # 3
print(f"空闲物理比特数: {layout.num_vacant_physical}")  # 1 (物理 10 或 12 其中之一空闲)
```

如果 `physical` 数量多于 `logical`，多余的物理比特将保持空闲状态（`vacant`），可用于后续路由或绑定。

### 从配对列表构造

您也可以通过显式的 `(logical, physical)` 配对列表来构造布局：

```python
from cqlib.device import Layout

# 逻辑 0 -> 物理 2, 逻辑 1 -> 物理 0
# 物理比特总数为 4，因此物理 1 和 3 保持空闲
layout = Layout.from_pairs([(0, 2), (1, 0)], physical_count=4)
print(layout.num_vacant_physical)  # 2
```

---

## 查询映射
`Layout` 维护了严格的双向索引以便于查询：

```python
# 获取完整的比特列表视图
print(f"逻辑比特集: {sorted(layout.logical_qubits)}")
print(f"空闲物理比特集: {sorted(layout.vacant_physical_qubits)}")
print(f"物理比特集: {sorted(layout.physical_qubits)}")

# 正向查询：逻辑 -> 物理
p_idx = layout.get_physical(0)
print(f"逻辑比特 0 当前位于物理比特: {p_idx}")

# 反向查询：物理 -> 逻辑
l_idx = layout.get_logical(11)
print(f"物理比特 11 上当前承载的逻辑比特: {l_idx}")

# 检查物理比特是否空闲
print(layout.is_physical_vacant(10))  # True 或 False

# 获取全量映射快照 (Dict)
l2p = layout.l2p_map  # {logical: physical}
p2l = layout.p2l_map  # {physical: logical}
```

---

## 更新映射

### 绑定与解绑

您可以在运行时动态绑定新的逻辑比特到空闲物理比特，或解绑已有逻辑比特：

```python
# 绑定新的逻辑比特 2 到空闲物理比特 12
layout.bind(2, 12)
print(layout.num_vacant_physical)  # 0

# 解绑逻辑比特 0，释放其占用的物理比特
released_physical = layout.unbind(0)
print(f"释放的物理比特: {released_physical}")
print(layout.num_vacant_physical)  # 1
```

### 交换物理比特

在路由算法（如基于最短路径的 SWAP 插入）中，最核心的操作是模拟物理层面的 SWAP 门。通过 `swap_physical`，您可以同步更新双向映射表：

```python
# 模拟在物理比特 11 和 12 之间执行一次 SWAP
# 操作后，原本在 11 上的逻辑比特将移至 12，反之亦然
before_11 = layout.get_logical(11)
before_12 = layout.get_logical(12)

layout.swap_physical(11, 12)

after_11 = layout.get_logical(11)
after_12 = layout.get_logical(12)

print(before_11, before_12)
print(after_11, after_12)
```

---

## 健壮性保障

为了防止由于比特索引误操作导致的编译错误，Layout 会在更新时进行严格的成员检查：

```python
try:
    # 尝试交换一个不存在于布局中的物理比特
    layout.swap_physical(11, 99)
except ValueError as e:
    print(f"路由校验拦截: {e}")

try:
    # 尝试绑定已占用的物理比特
    layout.bind(3, 11)  # 假设 11 已被占用
except ValueError as e:
    print(f"绑定校验拦截: {e}")
```

---

## 下一步

接下来您可以深入了解以下主题：

- [噪声模型](4_noise.md)
- [执行结果与状态](5_result.md)
