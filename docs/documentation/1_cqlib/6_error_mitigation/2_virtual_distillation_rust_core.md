# 虚拟蒸馏 Virtual Distillation（Rust 核心）

虚拟蒸馏通过多份态拷贝和 copy-swap 线路估计：

```text
Tr(O rho^M) / Tr(rho^M)
```

Rust 核心中的 `VirtualDistillation` 支持：

- 设置 copies 数；
- 构造 copy-swap 线路；
- 分别运行 numerator 和 denominator；
- 根据比值计算缓解后的期望值。

概念流程：

```text
原始 Circuit
  ↓
复制 M 份制备线路
  ↓
添加 pairwise SWAP
  ↓
估计 numerator: Tr(O rho^M)
估计 denominator: Tr(rho^M)
  ↓
计算比值得到缓解期望值
```

未来 Python 教程可在绑定完成后加入端到端示例：

```python
vd = VirtualDistillation(circuit, copies=2)
copy_swap = vd.build_copy_swap_circuit()
```

当前阶段建议将该内容放在“核心能力说明”或“开发者参考”章节。

<!-- expanded_by_chatgpt_20260617 -->
## ：方法理解

```python
# Virtual Distillation 的直观形式：
# rho_mitigated ∝ rho^M
# M 越大，主导本征态成分越突出，但资源开销也越高。
```

## ：Rust Core 侧形态示意

```rust
use cqlib_core::error_mitigation::virtual_distillation::VirtualDistillation;

let vd = VirtualDistillation::new(2);
// let value = vd.estimate(&measurements)?;
```

该能力更适合作为高级教程或开发者文档，普通入门用户只需理解其适用条件和额外资源开销。
