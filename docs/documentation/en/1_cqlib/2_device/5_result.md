# Execution result and status

The execution result module of cqlib.device provides measurement result encapsulation, task status tracking and probability calculation.

---

## Outcome: measurement result

Outcome uses little-endian bit order: the rightmost character of the string corresponds to Qubit 0, and the leftmost corresponds to Qubit N-1.

```python
from cqlib.device import Outcome

# 从比特字符串创建（小端序：最右 = Qubit 0）
o = Outcome("101")
print("比特 0 是否为 1:", o.is_one(0))   # True
print("比特 1 是否为 1:", o.is_one(1))   # False
print("完整比特串 (3):", o.to_bitstring(3))  # "101"

# 从比特字符串构造（与直接构造等价）
o2 = Outcome.from_bitstring("101")
print("o == o2:", o == o2)  # True

# 从索引列表构造（指定哪些位置的比特为 1）
o3 = Outcome.from_indices(width=3, indices=[0, 2])
print("索引构造:", o3.to_bitstring(3))  # "101"
```

**Notes**:
- The string may contain only '0' and '1'; other characters raise ValueError
- 
to_bitstring(num_qubits) if 
num_qubits is greater than the actual width, the high-order bits are padded with zeros

---

## Status: task status

Status represents the execution stage of a quantum task, and supports the following five states:

| State | Constructor | Terminal |
|---|---|---|
| Queued | Status.queued() | No |
| Running | Status.running() | No |
| Completed | Status.completed() | Yes |
| Failed | Status.failed(msg, code) | Yes |
| Cancelled | Status.cancelled() | Yes |

```python
from cqlib.device import Status

q = Status.queued()
c = Status.completed()
f = Status.failed("backend down", 500)
x = Status.cancelled()

print("queued:", q.kind, "终态?", q.is_terminal())
print("completed:", c.kind, "成功?", c.is_success())
print("failed:", f.kind, f.error_msg, f.error_code)
print("cancelled:", x.kind, "终态?", x.is_terminal())
```

**Notes**:
- Status.kind is a property (not a method) and returns a string, such as "completed" or "failed"
- is_terminal() returns True for completed, failed and cancelled
- is_success() returns True only for completed
- error_msg and error_code have values only when kind == "failed"; otherwise they return None

---

## ExecutionResult: the complete execution result

```python
from cqlib.device import ExecutionResult

result = ExecutionResult("q-task-001", [0, 1], 1000, 2, "Tianyan-176-2")
print("创建时状态:", result.status.kind)  # queued

# 标记为运行中
result.start()
print("启动后状态:", result.status.kind)  # running

# 完成并填入测量计数
result.finish({"00": 600, "11": 400})
result.calc_probabilities()  # 计算概率分布

print("任务 ID:", result.task_id)
print("测量次数:", result.shots)
print("计数结果:", result.counts)
print("概率分布:", result.probabilities)
print("后端名称:", result.backend)
```

### Constructing directly from counts

If measurement results are already available, from_counts can be used to complete creation and filling in one step:

```python
from cqlib.device import ExecutionResult

result = ExecutionResult.from_counts(
    task_id="q-task-002", qubits=[0, 1],
    shots=1024, num_qubits=2,
    counts={"00": 512, "11": 512},
    backend="simulator",
)
print("状态:", result.status.kind)        # completed（已自动完成）
print("概率分布:", result.probabilities)  # {"00": 0.5, "11": 0.5}
```

---

## Exception flow handling

```python
from cqlib.device import ExecutionResult

# 失败场景
f = ExecutionResult("task-fail", [0], 10, 1, None)
f.fail("timeout", 408)
print("失败状态:", f.status.kind)              # failed
print("错误消息:", f.status.error_msg)          # timeout
print("错误码:", f.status.error_code)           # 408

# 取消场景
c = ExecutionResult("task-cancel", [0], 10, 1, None)
c.cancel()
print("取消状态:", c.status.kind)               # cancelled
```

---

## Input validation

```python
from cqlib.device import ExecutionResult

r = ExecutionResult("bad", [0], 10, 1, None)
try:
    r.finish({"2": 1})  # "2" 不是有效的二进制字符串
except ValueError as e:
    print("无效计数:", e)
```

---

## Next steps

- [Quantum Information](../3_qis/0_overview.md): master the basics such as Statevector, DensityMatrix and Pauli
- [Compilation and optimization](../4_compiler/0_overview.md): learn about the layout, routing and optimization of the compilation pipeline
- [Visualization](../5_visualization/0_overview.md): learn about circuit drawing and result visualization
