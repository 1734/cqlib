# 拓扑模型

`Topology` 是对量子芯片物理比特及其耦合连通性的图论抽象。它是所有高级编译与路由算法的核心输入：只有准确掌握了芯片的硬件约束，编译器才能在有限的连通路径下，寻找最优的门映射方案，从而最大化计算保真度。

---

## 构造拓扑结构

Cqlib 支持通过显式边列表进行精细化建模，也支持针对标准布局的快速生成器。

### 1. 显式建模
您可以通过节点序列与边序列构建自定义的连接关系图：

```python
from cqlib.device import Topology

# 定义 3 个比特的连接图
# 元组结构: (源比特, 目标比特, 默认门类型)
topo = Topology(
    [0, 1, 2],
    [
        (0, 1),          # 默认门类型设为 "CX"
        (1, 2, "CZ"),    # 显式指定门类型为 "CZ"
    ],
)
```

### 2. 快速生成器
针对常见的芯片排列类型，您可以直接调用内置生成器：

```python
# 快速构建线性链式结构: 0->1->2->3（单向）
line_topo = Topology.line([0, 1, 2, 3])

# 双向线型: 0<->1<->2<->3
bidirectional_topo = Topology.bidirectional_line([0, 1, 2, 3])

# 环形: 0<->1<->2<->0
ring_topo = Topology.ring([0, 1, 2])

# 星形: 中心节点 0 与所有其他节点双向连接
star_topo = Topology.star([0, 1, 2, 3, 4], center=0)

# 网格: 2x3 双向网格
grid_topo = Topology.grid([0, 1, 2, 3, 4, 5], rows=2, cols=3)
```

---

## 拓扑查询与拓扑分析

`Topology` 类提供了丰富的高性能查询接口，便于在路由算法中进行实时寻径或连通性判断。

```python
# 基础规模查询
print(f"比特总数: {topo.num_qubits}")
print(f"耦合总数: {topo.num_couplings}")
print(f"节点列表: {topo.qubits}")

# 连通性分析
print(topo.supports_directed_coupling(0, 1))      # 检查 (0, 1) 间是否存在直接耦合
print(topo.supports_coupling_either_direction(0, 1))  # 检查任一方向是否存在耦合
print(topo.get_coupling_name(1, 2))               # 获取该耦合路径支持的物理门名（如 "CZ"）

# 节点邻序查询（有向图语义）
print(topo.contains_qubit(2))       # 检查特定物理索引是否存在
print(topo.successors(1))            # 获取比特 1 的后继节点（出边邻居）
print(topo.predecessors(1))          # 获取比特 1 的前驱节点（入边邻居）
print(topo.neighbors_undirected(1))  # 获取比特 1 的无向邻居（合并双向）
print(topo.out_degree(1))            # 获取比特 1 的出度
print(topo.in_degree(1))             # 获取比特 1 的入度

# 获取所有无向边（去重双向边）
print(topo.undirected_edges)        # [(Qubit(0), Qubit(1)), (Qubit(1), Qubit(2))]
```

---

## 动态修改拓扑
`Topology` 对象支持运行时动态调整，适用于描述多芯片互联或模拟硬件故障（如比特屏蔽）场景：

```python
# 动态增量更新
topo.add_qubits([3, 4])
topo.add_couplings([(2, 3, "CX"), (3, 4, "CZ")])

# 动态剔除（用于模拟硬件降级或比特故障）
topo.remove_couplings([(2, 3)])
topo.remove_qubits([4])
```

---

## 有向耦合语义

Cqlib 的 `Topology` 默认采用有向图模型。这意味着 (0, 1, "CX") 并不等同于 (1, 0, "CX")。

- 在物理层，这通常对应于 Control-Target 的方向限制：例如，硬件可能只支持 0 号对比特 1 号执行受控操作。
- 无向化建议：如果您使用的算法或仿真后端不考虑耦合方向，建议在构造时显式添加双向边：

```python
# 表示 0 和 1 互为控制/目标位
topo = Topology([0, 1], [(0, 1, "CX"), (1, 0, "CX")])
```

---

## 下一步
接下来您可以深入了解以下主题：

- [设备属性建模](2_device.md)
- [布局映射](3_layout.md)
- [噪声模型](4_noise.md)
- [执行结果与状态](5_result.md)
