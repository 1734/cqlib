# 后端与设备配置

完成认证后，需要选择一个量子后端。天衍平台上的后端由 `TianyanBackend` 表示，它包含后端名称、显示名称、状态、计费类型、可用比特数，以及设备配置下载接口。

## 1. 列举后端

```python
import os
from cqlib_tianyan import TianyanPlatform

platform = TianyanPlatform.login(os.environ["TIANYAN_API_KEY"])
backends = platform.list_backends()

for backend in backends:
    print(backend.name, backend.display_name, backend.status, backend.num_qubits)
```

`list_backends()` 会返回 `TianyanBackend` 列表。

## 2. 后端对象字段

| 字段 | 类型 | 说明 |
|---|---|---|
| `name` | `str` | 后端设备标识，用于提交任务，例如 `tianyan-287` |
| `display_name` | `str` | 用户友好的显示名称 |
| `status` | `DeviceStatus` | 设备运行状态 |
| `toll` | `DeviceToll` | 计费类型 |
| `num_qubits` | `int | None` | 平台返回的物理比特数，可能为空 |

示例：

```python
for backend in platform.list_backends():
    print(f"name={backend.name}")
    print(f"display={backend.display_name}")
    print(f"status={backend.status}")
    print(f"toll={backend.toll}")
    print(f"qubits={backend.num_qubits}")
```

## 3. 后端状态

`DeviceStatus` 可以与字符串比较：

| 值 | 含义 |
|---|---|
| `running` | 设备在线，可接受任务 |
| `calibration` | 设备校准中，任务可能排队 |
| `under_maintenance` | 维护中，暂不可用 |
| `offline` | 离线 |
| `unknown` | 平台返回了未识别状态 |

推荐提交前先检查：

```python
backend = platform.get_backend("tianyan-287")

if not backend.is_available():
    raise RuntimeError(f"后端不可用: {backend.status}")
```

也可以直接比较：

```python
if backend.status == "running":
    print("后端可提交")
```

## 4. 计费类型

`DeviceToll` 表示后端计费类型：

| 值 | 含义 |
|---|---|
| `free` | 免费任务 |
| `paid` | 消耗额度或计费 |
| `unknown` | 平台返回了未识别计费类型 |

```python
if backend.toll == "paid":
    print("该后端可能消耗额度，请确认后再提交")
```

## 5. 获取指定后端

```python
backend = platform.get_backend("tianyan-287")
```

如果不存在该后端，会抛出异常。

## 6. 获取设备配置

`device_config()` 会下载设备拓扑、校准和误差信息，并返回 `cqlib.device.Device` 对象：

```python
device = backend.device_config()

print(device.name)
print(device.num_usable_qubits)
print(device.topology)
```

该结果会在后端对象内部缓存；同一个 `TianyanBackend` 上重复调用时，不需要每次重新下载。

## 7. 设备配置的用途

设备配置可用于：

- 查看可用物理比特。
- 查看耦合拓扑。
- 辅助编译器做布局映射和门路由。
- 获取读出保真度，用于读取误差矫正。
- 分析单比特门、双比特门和测量误差。

典型流程：

```python
backend = platform.get_backend("tianyan-287")
device = backend.device_config()

# 后续可把 device 交给编译或分析模块使用
```

## 8. 后端选择建议

| 场景 | 建议 |
|---|---|
| 只是测试 API 链路 | 选择 `running` 且免费或低成本后端 |
| 需要指定物理比特 | 先查看设备拓扑和可用比特 |
| 需要较好结果质量 | 关注读出误差、双比特门误差和校准时间 |
| 批量提交 | 先用少量线路试跑，再扩大批量 |
| 需要误差矫正 | 确认设备配置中有可用校准数据 |

## 下一步

- [任务提交与结果获取](3_task_result.md)：使用选定后端提交 QCIS 线路，查询任务状态并获取结果。
- [QCIS 与 IR 联动](4_qcis_ir_workflow.md)：学习如何从 Cqlib `Circuit` 导出 QCIS，再提交到天衍平台。
