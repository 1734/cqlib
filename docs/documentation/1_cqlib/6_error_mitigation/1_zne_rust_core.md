# 零噪声外推 ZNE（Rust 核心）

ZNE 的基本思想是主动构造不同噪声强度下的等价线路，得到多个 noisy expectation，再外推到零噪声极限。

Rust 核心中的关键对象包括：

- `ZNEMitigation`；
- `ExtrapolateMethod::Polynomial`；
- `ExtrapolateMethod::Exponential`；
- `fold_circuits`；
- `run_em_sequence_with_shots`；
- `extrapolate`。

概念流程：

```text
Circuit U
  ↓ fold level = 0,1,2
U, U(U†U), U(U†U)(U†U)
  ↓
noise factors = 1,3,5
  ↓
后端估计 <H>
  ↓
多项式/指数外推到 noise_factor = 0
```

未来 Python 绑定建议提供类似接口：

```python
zne = ZNEMitigation(circuit, fold_levels=[0, 1, 2])
folded = zne.fold_circuits(gate_set=None)
values = [estimator(c) for c in folded]
mitigated = zne.extrapolate(values, method="polynomial", degree=1)
```

在 Python 绑定正式暴露前，官方 Python 教程不应直接使用上述代码作为可运行示例。

<!-- expanded_by_chatgpt_20260617 -->
## ：Python 概念验证

```python
import numpy as np

noise_factors = np.array([1.0, 3.0, 5.0])
values = np.array([-0.72, -0.65, -0.58])

coef = np.polyfit(noise_factors, values, deg=1)
zero_noise_value = np.polyval(coef, 0.0)

print(zero_noise_value)
```

## ：Rust Core 侧调用形态示意

```rust
use cqlib_core::error_mitigation::zne::ZNEMitigation;

let zne = ZNEMitigation::default();
// let result = zne.mitigate(&circuit, &observable, &executor)?;
```

正式文档中应以当前 Rust API 为准补全执行器、观测量和返回结果类型。
