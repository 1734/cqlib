# 错误缓解总览

当前源码包的 Rust 核心层包含 `error_mitigation` 模块，主要实现：

- 零噪声外推 ZNE；
- 虚拟蒸馏 Virtual Distillation；
- 统一的 ErrorMitigation facade。

从 Python 绑定目录看，当前 Python 包尚未像 `circuit`、`qis`、`device` 那样公开 `cqlib.error_mitigation` 模块。因此本章以“Rust 核心能力说明 + 后续 Python 绑定建议”为主，避免在官方 Python 教程中误导用户调用尚未暴露的接口。

错误缓解的一般流程：

```text
原始线路
  ↓
构造缓解所需线路族，例如折叠线路或 copy-swap 线路
  ↓
在模拟器或真实后端执行
  ↓
收集 noisy expectation
  ↓
外推或比值估计
  ↓
得到 mitigated expectation
```

<!-- expanded_by_chatgpt_20260617 -->
## 误差缓解的典型数据流

```python
noise_factors = [1.0, 3.0, 5.0]
expectations = [-0.72, -0.65, -0.58]

# 目标：根据多个噪声强度下的期望值，估计零噪声处的期望值。
```

误差缓解不是量子纠错。它通常增加执行次数，通过后处理降低期望值偏差，但不能保证恢复完整无噪声分布。
