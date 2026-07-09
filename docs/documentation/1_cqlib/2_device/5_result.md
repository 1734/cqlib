# 执行结果与状态

`cqlib.device` 的执行结果模块为开发者提供了一套工业级的任务追踪与结果抽象方案，确保从模拟器到真机后端的任务行为高度一致。

---

## Outcome：量子比特串的语义封装

`Outcome` 支持对量子测量观测值的结构化封装。它支持对比特位的高效查询与哈希校验，是构建统计直方图的基础单元:

```python
from cqlib.device import Outcome

# 从二进制字符串初始化观测结果
o = Outcome("101")

# 语义化查询：检查第 0 位比特是否为 '1' (LSB，即最右边)
print(f"Index 0 is One: {o.is_one(0)}")  # True

# 标准化输出
print(f"3-bit String: {o.to_bitstring(3)}") # "101"

# 支持等值比较与哈希，方便作为字典的 Key
o2 = Outcome.from_bitstring("101")
print(f"Match: {o == o2}")  # True

# 从位索引构造（设置第 0 位和第 2 位为 1）
o3 = Outcome.from_indices(width=3, indices=[0, 2])
print(o3.to_bitstring(3))  # "101"
```

---

## Status：任务生命周期的状态机

量子任务通常是异步执行的。`Status` 对象定义了一套标准的状态机模型，用于描述任务从提交到终止的全过程：

- `queued()`：已入队，等待资源分配。
- `running()`：后端正在执行。
- `completed()`：任务成功完成，结果就绪。
- `failed(error_msg, error_code)`：执行异常，记录错误信息与错误码。
- `cancelled()`：用户或系统主动取消执行。

```python
from cqlib.device import Status

queued = Status.queued()
running = Status.running()
completed = Status.completed()
failed = Status.failed("backend down", 500)
cancelled = Status.cancelled()

print(queued.kind, queued.is_terminal())      # queued False
print(completed.kind, completed.is_success()) # completed True
print(failed.kind, failed.error_msg, failed.error_code)
print(cancelled.kind, cancelled.is_terminal())
```

---

## ExecutionResult：全生命周期容器

`ExecutionResult` 是任务执行的“黑盒记录仪”。它完整记录了任务的元数据（ID、比特集、采样数）、多维时间戳（创建、开始、完成）以及最终的测量统计：

```python
from cqlib.device import ExecutionResult

# 1. 任务初始化：进入 'queued' 状态
result = ExecutionResult(
    task_id="q-task-001",
    qubits=[0, 1],
    shots=1000,
    num_qubits=2,
    backend="Tianyan-176-2",
)

# 2. 模拟任务开始执行
result.start()
print(f"Started at: {result.started_at}")

# 3. 任务回传并结束：更新状态为 'completed' 并存入计数字典
result.finish({"00": 600, "11": 400})
print(f"Status: {result.status.kind}")  # completed

# 4. 统计分析：计算状态概率分布
result.calc_probabilities()
print(f"Probabilities: {result.probabilities}") # {'11': 0.4, '00': 0.6}
```

### 从计数直接构造

如果您已经有完整的计数数据，可以使用便捷构造器直接创建已完成的结果：

```python
from cqlib.device import ExecutionResult

# 直接从计数构造已完成的结果
result = ExecutionResult.from_counts(
    task_id="q-task-002",
    qubits=[0, 1],
    shots=1024,
    num_qubits=2,
    counts={"00": 512, "11": 512},
    backend="simulator",
)
print(result.status.kind)       # completed
print(result.probabilities)     # {'00': 0.5, '11': 0.5}
```

---

## 异常流程处理

Cqlib 提供了明确的方法来中止任务或记录失败原因，并对输入数据进行严格校验：

```python
from cqlib.device import ExecutionResult

failed = ExecutionResult("task-fail", [0], 10, 1, None)
failed.fail("timeout", 408)
print(failed.status.kind)       # failed
print(failed.status.error_msg)  # timeout

cancelled = ExecutionResult("task-cancel", [0], 10, 1, None)
cancelled.cancel()
print(cancelled.status.kind)    # cancelled
```

---

## 输入校验

`finish(...)` 方法会严格检查计数字典的有效性。如果 Key 包含非二进制字符（如 "2" 或 "A"），系统将拒绝处理并抛出异常：

```python
from cqlib.device import ExecutionResult

r = ExecutionResult("task-invalid", [0], 10, 1, None)
try:
    r.finish({"2": 1})
except ValueError as e:
    print("invalid counts:", e)
```

---

## 下一步

接下来您可以深入了解以下主题：

- [量子信息](../3_information/0_overview.md)
- [错误缓解](../4_mitigation/0_overview.md)
- [编译优化](../5_transpilation/0_overview.md)
