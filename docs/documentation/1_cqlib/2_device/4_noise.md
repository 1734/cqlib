# 噪声模型

`cqlib.device` 提供了从单比特噪声、双比特噪声到整体噪声容器 `NoiseModel` 的完整建模接口，可用于仿真、误差注入和后处理分析。

通过这些接口，您可以构建出与真实芯片特性高度契合的 “噪声数字孪生” ，为误差缓解算法提供完美的实验场。

---

## 单比特噪声

`SingleQubitNoise` 类提供了多种经典的量子信道模型，用于描述比特在演化过程中的信息丢失：

- `bit_flip(p)`：比特翻转噪声。
- `phase_flip(p)`：相位翻转噪声。
- `pauli(px, py, pz)`：一般泡利噪声，独立指定 X、Y、Z 错误概率。
- `depolarizing(p)`：去极化噪声，最常用的综合噪声模型。
- `amplitude_damping(gamma)`：能量弛豫（T1 过程）建模。
- `phase_damping(lambda_)`：纯退相干（T2 过程）建模。

```python
from cqlib.device import SingleQubitNoise

# 创建 1% 强度的去极化噪声
sq = SingleQubitNoise.depolarizing(0.01)

# 创建一般泡利噪声
pauli_noise = SingleQubitNoise.pauli(px=0.001, py=0.0005, pz=0.002)

# 将噪声转换为 Kraus 算符表示（用于密度矩阵仿真）
kraus_ops = sq.to_kraus()
print(f"Kraus 算符数量: {len(kraus_ops)}") # 去极化噪声对应 4 个算符
print(f"算符维度: {kraus_ops[0].shape}")    # (2, 2)

# 验证噪声参数
assert sq.is_valid()  # True
```

---

## 双比特噪声

双比特门由于执行时间长、物理交互复杂，通常是电路噪声的主要来源：

- `depolarizing(p)`：标准双比特去极化噪声，15 个非恒等泡利算符均匀混合。
- `independent(q0_noise, q1_noise)`：两个比特独立受单比特噪声影响。
- `correlated_pauli(op_q0, op_q1, p)`：两个比特间存在物理耦合引发的协同泡利错误。

```python
from cqlib.device import SingleQubitNoise, TwoQubitNoise
from cqlib.qis import Pauli

# 标准双比特去极化噪声
tq = TwoQubitNoise.depolarizing(0.01)

# 构造相互独立的异构噪声：Q0 发生相位翻转，Q1 发生比特翻转
tq = TwoQubitNoise.independent(
    SingleQubitNoise.phase_flip(0.02),
    SingleQubitNoise.bit_flip(0.03),
)

# 相关泡利噪声：两个比特同时发生 XX 错误
tq = TwoQubitNoise.correlated_pauli(Pauli.X, Pauli.X, p=0.01)

# 转换后的 Kraus 算符维度为 (4, 4)
print(f"双比特噪声矩阵维度: {tq.to_kraus()[0].shape}")
```

---

## 读出误差

读出误差描述了测量过程中的比特翻转。它通常由热激发或测量串扰引起，表现为测量结果与真实量子态的不一致：

```python
from cqlib.device import ReadoutError

# 参数含义: (P(0|1): 1 测成 0 的概率, P(1|0): 0 测成 1 的概率)
ro = ReadoutError(p_0_given_1=0.02, p_1_given_0=0.01)

if ro.is_valid():
    print(f"Qubit 1->0 翻转概率: {ro.p_0_given_1}")
    print(f"Qubit 0->1 翻转概率: {ro.p_1_given_0}")
```

---

## NoiseModel：构建整体噪声视图
`NoiseModel` 是噪声定义的集合容器。您可以将上述定义的各种噪声 “挂载” 到特定的量子比特或量子门操作上：

```python
from cqlib.circuit import StandardGate
from cqlib.device import (
    NoiseModel,
    OperationKey,
    ReadoutError,
    SingleQubitNoise,
    TwoQubitNoise,
)

nm = NoiseModel()

# 1) 添加读出误差
nm.add_readout_error(0, ReadoutError(0.02, 0.01))

# 2) 添加单比特门误差
nm.add_single_qubit_error(
    StandardGate.X,
    0,
    SingleQubitNoise.bit_flip(0.005),
)

# 3) 添加双比特门误差
nm.add_two_qubit_error(
    StandardGate.CX,
    0,
    1,
    TwoQubitNoise.depolarizing(0.02),
)

# 4) 查询
print(nm.get_readout_error(0))

skey = OperationKey.new_single(StandardGate.X, 0)
print([n.kind for n in (nm.get_single_qubit_errors(skey) or [])])

tkey = OperationKey.new_double(StandardGate.CX, 0, 1)
print([n.kind for n in (nm.get_two_qubit_errors(tkey) or [])])
```

---

## 安全性校验

Cqlib 会在构造和添加过程中实时监控参数合法性。`NoiseModel` 的 `add_*` 方法现在会返回 `Result` 类型进行参数校验：

```python
from cqlib.circuit import StandardGate
from cqlib.device import NoiseModel, SingleQubitNoise

nm = NoiseModel()

# 尝试设置超过 100% 的错误概率
try:
    nm.add_single_qubit_error(
        StandardGate.X,
        0,
        SingleQubitNoise.bit_flip(1.5),
    )
except ValueError as e:
    print("invalid noise parameter:", e)

# 尝试使用无效的双比特配置
try:
    nm.add_two_qubit_error(
        StandardGate.CX,
        0, 0,  # 错误：同一个比特
        TwoQubitNoise.depolarizing(0.01),
    )
except ValueError as e:
    print("invalid qubit configuration:", e)
```

---

## 下一步

接下来您可以深入了解以下主题：

- [执行结果与状态](5_result.md)
