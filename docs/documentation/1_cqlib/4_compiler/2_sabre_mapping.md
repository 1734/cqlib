# SABRE 路由映射

SABRE 通过启发式 SWAP 将逻辑两比特门路由到设备允许的物理耦合上。

---

## 两个入口

| API | 作用 |
|-----|------|
| `route_sabre(circuit, device, objective, config)` | 自动 `sabre_layout` + 路由 |
| `route_with_layout(circuit, device, initial_layout, config)` | 仅路由，跳过布局搜索 |

```python
from cqlib import Circuit
from cqlib.compile.transform.layout import LayoutObjective
from cqlib.compile.transform.routing import route_sabre, route_with_layout
from cqlib.compile.sabre import SabreConfig
from cqlib.device import Device, Layout

circuit = Circuit(3)
circuit.cx(0, 2)

device = Device.line("line-3", 3)
objective = LayoutObjective.topology_only()
config = SabreConfig.deterministic_seeded(42)

result = route_sabre(circuit, device, objective, config)
print("swap_count:", result.swap_count)
print("ops:", len(result.circuit.operations))
```

---

## 与 compile 工作流集成

```python
from cqlib.compile import CompileMode, compile

result = compile(
    circuit,
    mode=CompileMode.enhanced(),
    device=device,
    seed=42,
)

for step in result.steps:
    if step.name == "route.sabre":
        print(step.changed, step.reason)
```

`initial_layout` 已提供时，工作流跳过自动布局，仍用相同 SABRE 路由器。

---

## 示例：仅路由（跳过布局搜索）

```python
from cqlib.device import Layout

initial = Layout.from_pairs([(0, 0), (1, 2), (2, 1)], physical_count=3)
routed = route_with_layout(circuit, device, initial, config)
print("swaps:", routed.swap_count)
```

---

## 示例：compile 传入 initial_layout

```python
from cqlib.compile import compile

result = compile(
    circuit,
    device=device,
    initial_layout=initial,
    seed=42,
)
```

---

## SabreConfig

| 字段 | 含义 |
|------|------|
| `layout_trials` | 随机初始布局试次数 |
| `refinement_iterations` | 每个候选的前向+后向精修轮数 |
| `layout_scoring_trials` | 评分每个精修布局的路由试次数 |
| `routing_trials` | 最终选路的并行试次数 |
| `trial_objective` | `swap_then_depth` / `depth_then_swap` 等 |
| `seed` | 确定性种子 |
| `heuristic` | `SabreHeuristicConfig`：lookahead、decay 等 |

```python
from cqlib.compile.sabre import SabreConfig, SabreTrialObjective

config = SabreConfig(
    seed=42,
    layout_trials=24,
    refinement_iterations=2,
    routing_trials=12,
    trial_objective=SabreTrialObjective.swap_then_depth(),
)
```

快捷构造：`SabreConfig.deterministic_seeded(42)`。

---

## 示例：不同 trial_objective 对比

```python
from cqlib.compile.sabre import SabreTrialObjective

for obj in (
    SabreTrialObjective.swap_count(),
    SabreTrialObjective.depth(),
    SabreTrialObjective.swap_then_depth(),
):
    cfg = SabreConfig(routing_trials=4, seed=123, trial_objective=obj)
    routed = route_sabre(circuit, device, objective, cfg)
    print(obj, "swaps:", routed.swap_count)
```

---

## 示例：固定 seed 的可复现记录

```python
compile_record = {
    "seed": 123,
    "layout_trials": 24,
    "routing_trials": 12,
    "trial_objective": "swap_then_depth",
}
```

路由含随机性，正式实验必须固定并记录 `seed`。

---

## 输出语义

- `result.circuit`：物理比特编号上的线路，含插入的 `SWAP`；
- `swap_count`：应与线路中 SWAP 门数量一致；
- 路由保证无向物理邻接。

---

## 下一步

- [模板匹配与知识规则优化](3_template_optimization.md)：了解路由后知识规则如何进一步清理与改写线路。
- [初始布局（Layout）](1_layout.md)：复习初始布局算法与 `LayoutObjective` 的评分方式。
